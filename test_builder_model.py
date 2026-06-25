"""
Tests for builder_model.py - the fully configurable model (W_O / activations / bias /
positional encoding / pooling / output heads) and the classification & regression tasks.

Run:  python -m pytest test_builder_model.py -v   (or: python test_builder_model.py)
"""
from __future__ import annotations

import torch

from builder_model import (
    BuilderConfig, AttnCfg, FFNCfg, BuilderModel, Pooling, count_params,
    MajorityTask, DensityTask, quick_train, quick_eval,
)
from tasks import ReverseTask
from tiny_gpt import evaluate


def _lm_cfg(task, **kw):
    base = dict(vocab_size=task.vocab_size, block_size=task.block_size, d_model=32, n_heads=4,
                layers=(("attn", AttnCfg()), ("ffn", FFNCfg(hidden=(64,)))), head="lm")
    base.update(kw)
    return BuilderConfig(**base)


# ---------------------------------------------------------------------------
# configurable W_O
# ---------------------------------------------------------------------------
def test_wo_linear_vs_mlp():
    task = ReverseTask(length=4)
    plain = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(wo_hidden=())),)))
    mlp = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(wo_hidden=(64,))),)))
    # plain W_O is a single linear (no activation); MLP W_O has more modules + more params
    import torch.nn as nn
    plain_wo = plain.attention_layers()[0].attn.wo
    assert sum(isinstance(m, nn.Linear) for m in plain_wo) == 1
    mlp_wo = mlp.attention_layers()[0].attn.wo
    assert sum(isinstance(m, nn.Linear) for m in mlp_wo) == 2
    assert count_params(mlp) > count_params(plain)
    # both still produce next-token logits of the right shape
    x = torch.zeros(2, task.block_size, dtype=torch.long)
    assert plain(x)[0].shape == (2, task.block_size, task.vocab_size)
    assert mlp(x)[0].shape == (2, task.block_size, task.vocab_size)


# ---------------------------------------------------------------------------
# FFN activations + bias toggle + positional encoding
# ---------------------------------------------------------------------------
def test_ffn_activations_build_and_run():
    task = ReverseTask(length=4)
    x = torch.zeros(2, task.block_size, dtype=torch.long)
    for act in ("gelu", "relu", "silu", "tanh"):
        m = BuilderModel(_lm_cfg(task, layers=(("ffn", FFNCfg(hidden=(48,), activation=act)),)))
        assert m(x)[0].shape == (2, task.block_size, task.vocab_size)


def test_bias_toggle_changes_params():
    task = ReverseTask(length=4)
    with_bias = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(bias=True)),
                                                   ("ffn", FFNCfg(hidden=(64,), bias=True)))))
    no_bias = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(bias=False)),
                                                 ("ffn", FFNCfg(hidden=(64,), bias=False)))))
    assert count_params(no_bias) < count_params(with_bias)


def test_positional_encoding():
    task = ReverseTask(length=4)
    learned = BuilderModel(_lm_cfg(task, pos_encoding="learned"))
    sinus = BuilderModel(_lm_cfg(task, pos_encoding="sinusoidal"))
    assert hasattr(learned, "pos_emb") and not hasattr(sinus, "pos_emb")
    x = torch.zeros(2, task.block_size, dtype=torch.long)
    assert learned(x)[0].shape == sinus(x)[0].shape           # both run, same output shape
    assert count_params(sinus) < count_params(learned)        # sinusoidal table is not learned


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------
def test_pooling_methods():
    B, T, d = 3, 5, 8
    x = torch.randn(B, T, d)
    assert torch.allclose(Pooling("mean", d)(x), x.mean(1))
    assert torch.equal(Pooling("last", d)(x), x[:, -1, :])
    assert torch.equal(Pooling("cls", d)(x), x[:, 0, :])
    pa = Pooling("attn", d)
    out = pa(x)
    assert out.shape == (B, d)
    assert torch.allclose(pa.last_weights.sum(1), torch.ones(B), atol=1e-5)   # weights are a distribution


# ---------------------------------------------------------------------------
# output head shapes
# ---------------------------------------------------------------------------
def test_head_shapes():
    task = MajorityTask(length=9)
    x = torch.zeros(4, task.block_size, dtype=torch.long)
    lm = BuilderModel(BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size, head="lm"))
    assert lm(x)[0].shape == (4, task.block_size, task.vocab_size)
    cl = BuilderModel(BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size,
                                    head="classify", n_classes=3, pooling="mean"))
    assert cl(x)[0].shape == (4, 3)
    rg = BuilderModel(BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size,
                                    head="regression", out_dim=1, pooling="last"))
    assert rg(x)[0].shape == (4, 1)


# ---------------------------------------------------------------------------
# causal mask
# ---------------------------------------------------------------------------
def test_causal_mask():
    task = ReverseTask(length=5)
    m = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(causal=True)),)))
    x = torch.randint(0, task.vocab_size, (1, task.block_size))
    m.eval(); m(x)
    W = m.attention_layers()[0].attn.last_attn[0]             # [n_heads, T, T]
    T = task.block_size
    for h in range(W.shape[0]):
        for i in range(T):
            assert abs(float(W[h, i].sum()) - 1.0) < 1e-4
            for j in range(i + 1, T):
                assert float(W[h, i, j]) < 1e-6                # no looking ahead
    # non-causal attends to future too
    m2 = BuilderModel(_lm_cfg(task, layers=(("attn", AttnCfg(causal=False)),)))
    m2.eval(); m2(x)
    W2 = m2.attention_layers()[0].attn.last_attn[0]
    assert float(W2[0, 0, T - 1]) > 1e-6


# ---------------------------------------------------------------------------
# learning: lm / classification / regression
# ---------------------------------------------------------------------------
def test_lm_head_learns_reverse():
    task = ReverseTask(length=4)
    m = BuilderModel(_lm_cfg(task))
    before = evaluate(m, task, "cpu", generator=torch.Generator().manual_seed(1))[1]
    quick_train(m, task, steps=400, lr=3e-3)
    after = evaluate(m, task, "cpu", generator=torch.Generator().manual_seed(1))[1]
    assert after > before and after > 0.5


def test_classification_learns():
    task = MajorityTask(length=13)
    cfg = BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size, d_model=32, n_heads=4,
                        layers=(("attn", AttnCfg()), ("ffn", FFNCfg(hidden=(64,)))),
                        head="classify", pooling="mean", n_classes=2)
    m = BuilderModel(cfg)
    quick_train(m, task, steps=500, lr=3e-3)
    assert quick_eval(m, task)["acc"] > 0.75                  # majority is learnable


def test_regression_learns():
    task = DensityTask(length=12)
    cfg = BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size, d_model=32, n_heads=4,
                        layers=(("attn", AttnCfg()), ("ffn", FFNCfg(hidden=(64,)))),
                        head="regression", pooling="attn", out_dim=1)
    m = BuilderModel(cfg)
    quick_train(m, task, steps=500, lr=3e-3)
    assert quick_eval(m, task)["r2"] > 0.5                    # density (fraction of 1s) is learnable


# ---------------------------------------------------------------------------
# replay/figure interface compatibility
# ---------------------------------------------------------------------------
def test_trace_interface():
    task = ReverseTask(length=4)
    m = BuilderModel(_lm_cfg(task))
    x = torch.randint(0, task.vocab_size, (1, task.block_size))
    stages = m.forward_trace(x)
    assert stages[0][0] == "embed"
    assert len(stages) == 1 + len(m.cfg.layers)
    assert len(m.attention_layers()) == 1 and len(m.dense_layers()) == 1
    assert m.attention_layers()[0].attn.last_attn is not None
    assert m.dense_layers()[0].last_hidden is not None


def test_init_schemes():
    """Each weight-init scheme builds and produces a distinct weight distribution."""
    task = ReverseTask(length=3)
    stds = {}
    for scheme in ("normal", "xavier", "kaiming", "orthogonal", "zeros"):
        torch.manual_seed(0)
        m = BuilderModel(_lm_cfg(task, init=scheme, init_scale=0.02))
        stds[scheme] = float(m.layers[0].attn.qkv.weight.std())
    assert abs(stds["normal"] - 0.02) < 0.01           # normal honors init_scale
    assert stds["zeros"] == 0.0                          # zeros = all zero
    assert stds["kaiming"] > stds["normal"]             # self-scaling schemes are larger
    assert stds["xavier"] > stds["normal"]


def test_arithmetic_decodes_each_op():
    """Arithmetic generates correct +/−/× examples (non-negative subtract) and defaults to add."""
    from tasks import ArithmeticTask
    t = ArithmeticTask(n_digits=2, ops=["add", "subtract", "multiply"])
    _, _, seq = t.make_batch(800, generator=torch.Generator().manual_seed(0))
    n = t.n_digits
    ops = {"+": lambda a, b: a + b, "−": lambda a, b: a - b, "×": lambda a, b: a * b}
    seen = set()
    for s in seq:
        toks = [t.id_to_str[int(i)] for i in s]
        a = int("".join(toks[:n])); op = toks[n]; b = int("".join(toks[n + 1:2 * n + 1]))
        res = sum(int(d) * 10 ** i for i, d in enumerate(toks[2 * n + 2:]))   # LSB first
        assert ops[op](a, b) == res, (a, op, b, res)
        seen.add(op)
    assert seen == {"+", "−", "×"}                  # all three operations appear
    assert ArithmeticTask().ops == ["add"]          # default is add only
    assert ArithmeticTask(ops=["add"]).gen_len < ArithmeticTask(ops=["multiply"]).gen_len  # multiply is wider


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); print(f"  ok   {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL {fn.__name__}"); traceback.print_exc(); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
