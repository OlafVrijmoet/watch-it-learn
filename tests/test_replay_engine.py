"""
Tests for replay_engine.py - the training-history / replay backbone.

Focus areas:
  * checkpoint cadence (log-spaced; full-fidelity)
  * training actually learns
  * EXACT, faithful reconstruction (stored checkpoint == deterministic replay == truth)
  * snap-to-nearest (no weight interpolation)
  * activation trace: shapes, causal mask, JSON-serializable
  * sampling methods
  * device handling

Run with:  PYTHONPATH=. python tests/test_replay_engine.py (from the repo root)
"""
from __future__ import annotations

import json
import math

import torch

from replay_engine import (
    RunConfig, TrainingRun, log_spaced_steps, trace_forward, sample_token,
    generate_sampled, build_task, per_category_eval,
)


def test_per_category_eval_lm():
    """per_category_eval returns one entry per Arithmetic operation, with valid accuracy + counts."""
    cfg = RunConfig(task_name="Arithmetic", task_kwargs={"n_digits": 1, "ops": ["add", "subtract"]},
                    layer_specs=(("attn",), ("ffn", (64,))), d_model=32, n_heads=4,
                    steps=60, n_checkpoints=6, seed=0)
    run = TrainingRun.train(cfg)
    cats = per_category_eval(run.reconstruct(run.cfg.steps), run.task, batch=256)
    assert set(cats) == {"add", "subtract"}                         # one bucket per selected op
    assert all(0.0 <= a <= 1.0 and n > 0 for a, n in cats.values())  # valid accuracy + non-empty counts


def test_per_category_eval_classify():
    """per_category_eval splits a classification task (Majority) by its true class."""
    cfg = RunConfig(task_name="Majority", task_kwargs={}, layer_specs=(("attn",), ("ffn", (64,))),
                    d_model=32, n_heads=4, head="classify", pooling="mean", steps=60, n_checkpoints=6, seed=0)
    run = TrainingRun.train(cfg)
    cats = per_category_eval(run.reconstruct(run.cfg.steps), run.task, batch=256)
    assert cats and set(cats) <= {"majority of 0s", "majority of 1s"}
    assert all(0.0 <= a <= 1.0 and n > 0 for a, n in cats.values())


# small configs to keep tests fast
def _struct_cfg(**kw):
    base = dict(task_name="Reverse", task_kwargs={"length": 4},
                layer_specs=(("attn",), ("ffn", (48,))),
                d_model=32, n_heads=4, steps=60, batch=64, lr=3e-3,
                n_checkpoints=12, seed=0, device="cpu")
    base.update(kw)
    return RunConfig(**base)


def _states_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# 1. checkpoint cadence
# ---------------------------------------------------------------------------
def test_log_spaced_steps_basic():
    steps = log_spaced_steps(1000, 40)
    assert steps[0] == 0 and steps[-1] == 1000
    assert steps == sorted(set(steps))                  # sorted + unique
    assert len(steps) <= 42                              # ~n, not more
    # dense early, sparse late: first gaps smaller than last gaps
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert gaps[0] <= gaps[-1]


def test_log_spaced_full_fidelity():
    steps = log_spaced_steps(50, 999)                   # n >= total+1 -> every step
    assert steps == list(range(51))


def test_log_spaced_edge():
    assert log_spaced_steps(0, 10) == [0]


# ---------------------------------------------------------------------------
# 2. training learns
# ---------------------------------------------------------------------------
def test_training_learns_reverse():
    cfg = _struct_cfg(steps=300, n_checkpoints=20)
    run = TrainingRun.train(cfg)
    first, last = run.checkpoints[0].acc, run.checkpoints[-1].acc
    assert last > first                                  # it improved
    assert last > 0.6                                    # short Reverse is easy -> mostly solved
    # train loss trends down
    assert run.metrics["train_loss"][-1] < run.metrics["train_loss"][0]


# ---------------------------------------------------------------------------
# 3. checkpoint structure
# ---------------------------------------------------------------------------
def test_checkpoint_structure():
    cfg = _struct_cfg()
    run = TrainingRun.train(cfg)
    assert run.checkpoint_steps == log_spaced_steps(cfg.steps, cfg.n_checkpoints)
    # all weights live on CPU (device-agnostic storage)
    for ck in run.checkpoints:
        for v in ck.state_dict.values():
            assert v.device.type == "cpu"
    # dense metrics: one per training step
    assert len(run.metrics["step"]) == cfg.steps
    assert run.metrics["step"] == list(range(cfg.steps))
    assert run.nbytes() > 0


# ---------------------------------------------------------------------------
# 4 + 5. EXACT, faithful reconstruction  (the crucial guarantee)
# ---------------------------------------------------------------------------
def test_reconstruct_loads_exact_checkpoint():
    run = TrainingRun.train(_struct_cfg())
    ck = run.checkpoints[len(run.checkpoints) // 2]
    model = run.reconstruct(ck.step)
    assert _states_equal(_cpu(model), ck.state_dict)    # loaded weights == stored weights


def test_deterministic_replay_matches_stored_checkpoint():
    """Re-running training from scratch to a checkpoint step reproduces that checkpoint
    BIT-FOR-BIT -> the stored state is the true state, and any step is exactly reconstructable."""
    run = TrainingRun.train(_struct_cfg())
    for idx in (1, len(run.checkpoints) // 2, len(run.checkpoints) - 1):
        ck = run.checkpoints[idx]
        replayed = run.reconstruct_exact(ck.step)
        assert _states_equal(_cpu(replayed), ck.state_dict), f"mismatch at step {ck.step}"


def test_two_runs_identical():
    a = TrainingRun.train(_struct_cfg(seed=7))
    b = TrainingRun.train(_struct_cfg(seed=7))
    assert _states_equal(a.checkpoints[-1].state_dict, b.checkpoints[-1].state_dict)
    # different seed -> different trajectory
    c = TrainingRun.train(_struct_cfg(seed=8))
    assert not _states_equal(a.checkpoints[-1].state_dict, c.checkpoints[-1].state_dict)


def test_snap_to_nearest_no_interpolation():
    run = TrainingRun.train(_struct_cfg())
    cs = run.checkpoint_steps
    mid = (cs[1] + cs[2]) // 2                            # a step with no checkpoint
    model = run.reconstruct(mid)
    nearest = run.nearest_checkpoint(mid)
    assert _states_equal(_cpu(model), nearest.state_dict)   # exactly a real checkpoint, not blended


# ---------------------------------------------------------------------------
# 6. activation trace
# ---------------------------------------------------------------------------
def test_trace_shapes_and_causal_mask():
    run = TrainingRun.train(_struct_cfg())
    task = run.task
    _, _, probe = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    model = run.reconstruct(run.checkpoint_steps[-1])
    tr = trace_forward(model, task, probe[0])

    T = tr["T"]
    assert len(tr["token_strs"]) == T
    # one stage per (embed + each sublayer)
    assert len(tr["stages"]) == 1 + len(cfg_layers(run))
    assert len(tr["stages"][0]["residual"]) == T
    assert len(tr["stages"][0]["residual"][0]) == tr["d_model"]

    # attention: one per attn layer; grid is [n_heads, T, T]; causal + rows sum to 1
    assert len(tr["attention"]) == 1
    W = torch.tensor(tr["attention"][0]["weights"])
    assert W.shape == (run.cfg.n_heads, T, T)
    for h in range(W.shape[0]):
        for i in range(T):
            assert abs(float(W[h, i].sum()) - 1.0) < 1e-4          # rows are a distribution
            for j in range(i + 1, T):
                assert float(W[h, i, j]) < 1e-6                    # causal: no looking ahead
    # Q/K/V captured with head shape [n_heads, T, d_k]
    q = torch.tensor(tr["attention"][0]["q"])
    assert q.shape == (run.cfg.n_heads, T, run.cfg.d_model // run.cfg.n_heads)

    # dense hidden + logits/probs present and finite; probs are distributions
    assert len(tr["ffn"]) == 1
    probs = torch.tensor(tr["probs"])
    assert probs.shape == (T, tr["vocab_size"])
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones(T), atol=1e-4)


def cfg_layers(run):
    return run.cfg.layer_specs


def test_trace_json_serializable():
    run = TrainingRun.train(_struct_cfg())
    task = run.task
    _, _, probe = task.make_batch(1, generator=torch.Generator().manual_seed(1))
    fr = run.frame(run.checkpoint_steps[-1], probe[0])
    s = json.dumps(fr)                                   # must not raise
    assert len(s) > 100
    assert set(fr) >= {"requested_step", "snapped_step", "metrics", "checkpoint_steps", "trace"}


def test_ffn_only_stack_traces():
    """A stack with no attention still trains and traces (empty attention list)."""
    cfg = _struct_cfg(layer_specs=(("ffn", (32,)),), steps=30, n_checkpoints=6)
    run = TrainingRun.train(cfg)
    task = run.task
    _, _, probe = task.make_batch(1, generator=torch.Generator().manual_seed(0))
    tr = trace_forward(run.reconstruct(cfg.steps), task, probe[0])
    assert tr["attention"] == []
    assert len(tr["ffn"]) == 1


# ---------------------------------------------------------------------------
# 7. sampling
# ---------------------------------------------------------------------------
def test_sampling_methods():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.05, 3.0, 0.0, 0.0, 0.0])
    assert sample_token(logits, "greedy") == 1                       # argmax
    # top-k=2 can only ever return one of the two largest (idx 1, 4)
    g = torch.Generator().manual_seed(0)
    picks = {sample_token(logits, "top_k", top_k=2, temperature=1.0, generator=g) for _ in range(50)}
    assert picks <= {1, 4}
    # top-p small -> essentially the top token
    g = torch.Generator().manual_seed(0)
    picks = {sample_token(logits, "top_p", top_p=0.5, temperature=1.0, generator=g) for _ in range(50)}
    assert picks <= {1, 4}
    # seeded sampling is reproducible
    a = sample_token(logits, "temperature", temperature=1.5, generator=torch.Generator().manual_seed(2))
    b = sample_token(logits, "temperature", temperature=1.5, generator=torch.Generator().manual_seed(2))
    assert a == b


def test_generate_sampled_greedy_matches_argmax():
    run = TrainingRun.train(_struct_cfg(steps=200))
    task = run.task
    model = run.reconstruct(run.checkpoint_steps[-1])
    _, _, seq = task.make_batch(3, generator=torch.Generator().manual_seed(9))
    from lm_utils import generate as greedy_generate
    out_greedy = greedy_generate(model, task, seq, "cpu")
    out_sampled = generate_sampled(model, task, seq, "greedy", device="cpu")
    assert torch.equal(out_greedy, out_sampled)


# ---------------------------------------------------------------------------
# 8. full fidelity + device
# ---------------------------------------------------------------------------
def test_full_fidelity_every_step_exact():
    cfg = _struct_cfg(steps=20, n_checkpoints=10_000)    # -> every step
    run = TrainingRun.train(cfg)
    assert run.checkpoint_steps == list(range(cfg.steps + 1))
    # every step is an exact checkpoint: snap-to-nearest == deterministic replay
    for s in (0, 7, cfg.steps):
        assert _states_equal(_cpu(run.reconstruct(s)), _cpu(run.reconstruct_exact(s)))


def test_device_cpu_runs():
    run = TrainingRun.train(_struct_cfg(device="cpu"))
    assert run.checkpoints[-1].state_dict  # ran and stored


def test_device_gpu_if_available():
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        return
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    run = TrainingRun.train(_struct_cfg(device=dev, steps=30, n_checkpoints=6))
    for v in run.checkpoints[-1].state_dict.values():
        assert v.device.type == "cpu"                    # stored on CPU regardless of train device


def _cpu(model):
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# 9. pooled heads through the full replay pipeline (BuilderModel-backed)
# ---------------------------------------------------------------------------
def test_classify_run_learns_and_reconstructs_exactly():
    cfg = RunConfig(task_name="Majority", task_kwargs={"length": 13}, head="classify", pooling="mean",
                    layer_specs=(("attn",), ("ffn", (48,))), d_model=32, n_heads=4,
                    steps=150, batch=64, lr=3e-3, n_checkpoints=12, seed=0, device="cpu")
    run = TrainingRun.train(cfg)
    assert run.checkpoints[-1].acc > run.checkpoints[0].acc          # learned
    assert run.checkpoints[-1].acc > 0.6
    # faithful reconstruction holds for the BuilderModel path too
    ck = run.checkpoints[len(run.checkpoints) // 2]
    assert _states_equal(_cpu(run.reconstruct_exact(ck.step)), ck.state_dict)
    # trace exposes the pooled head output
    task = run.task
    x, _ = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    tr = trace_forward(run.reconstruct(cfg.steps), task, x[0])
    assert tr["head"] == "classify"
    assert "output" in tr and len(tr["probs"]) == task.n_classes
    assert len(tr["attention"]) == 1 and len(tr["ffn"]) == 1       # introspection still works


def test_regression_run_with_attention_pooling():
    cfg = RunConfig(task_name="Density", task_kwargs={"length": 12}, head="regression", pooling="attn",
                    layer_specs=(("attn",), ("ffn", (48,))), d_model=32, n_heads=4,
                    steps=150, batch=64, lr=3e-3, n_checkpoints=12, seed=0, device="cpu")
    run = TrainingRun.train(cfg)
    assert run.checkpoints[-1].acc > run.checkpoints[0].acc          # R^2 improved
    assert run.checkpoints[-1].acc > 0.0
    task = run.task
    x, _ = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    tr = trace_forward(run.reconstruct(cfg.steps), task, x[0])
    assert tr["head"] == "regression" and len(tr["output"]) == task.out_dim
    assert "pool_weights" in tr                                      # attention pooling weights captured


# ---------------------------------------------------------------------------
# run as a script (no pytest needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
