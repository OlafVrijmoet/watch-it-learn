"""
replay_engine.py - train a configurable tiny transformer while recording its
*training history*, so the whole run can be scrubbed and every state faithfully
reconstructed and inspected.

This is the technical backbone for the model-builder UI. It implements the locked
storage design:

  * DENSE scalar metrics, every step           -> the smooth loss/accuracy curves.
  * SPARSE weight checkpoints, LOG-SPACED       -> the only heavy thing; a `state_dict`
    (on CPU) at each checkpoint step. A "# checkpoints" knob can crank density up to
    EVERY step (full fidelity).
  * EXACT reconstruction. A checkpoint is the real `state_dict` (bit-for-bit), so
    loading it reproduces the model exactly; the replay forward pass runs in eval mode
    (dropout off, no grad) and is deterministic, so the activations you see are the true
    ones for that step. We also keep seed + config + initial weights, so ANY step is
    reconstructable exactly via deterministic replay (re-run from scratch).
  * Activations are NOT stored - they are recomputed on demand (`trace_forward`) for a
    chosen probe input. Storage is checkpoints x params; per-view compute is O(1).
  * Device toggle (cpu / mps / cuda). Checkpoints are stored on CPU (device-agnostic);
    deterministic replay runs on CPU for reproducibility.

Everything reuses the model / tasks / optimizers already in the repo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import torch
import torch.nn.functional as F

from tiny_gpt import get_device, generate, evaluate
from tasks import TASKS, Task
from training_utils import make_optimizer, _lr_at
from builder_model import (BuilderModel, BuilderConfig, AttnCfg, FFNCfg, MajorityTask, DensityTask,
                           AttnBlock, FFNBlock, count_params)

# next-token (LM) tasks + the pooled-head (classification / regression) tasks
ALL_TASKS = {**TASKS, "Majority": MajorityTask, "Density": DensityTask}


# ---------------------------------------------------------------------------
# Checkpoint cadence
# ---------------------------------------------------------------------------
def log_spaced_steps(total_steps: int, n_checkpoints: int) -> list[int]:
    """Step indices to checkpoint: dense early, sparse later, always incl. 0 and total.

    `n_checkpoints` is the target density knob. If it is >= total_steps+1 you get EVERY
    step (full fidelity). Otherwise the points are spaced geometrically (training moves
    fast early, slowly later), de-duplicated, with 0 and `total_steps` always present.
    """
    total = int(total_steps)
    if total <= 0:
        return [0]
    if n_checkpoints >= total + 1:
        return list(range(total + 1))
    n_mid = max(1, int(n_checkpoints) - 1)
    pts = np.geomspace(1, total, num=n_mid)
    steps = sorted({0, total, *(int(round(p)) for p in pts)})
    return [s for s in steps if 0 <= s <= total]


# ---------------------------------------------------------------------------
# Run configuration  (architecture + task + training + reproducibility)
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    # task
    task_name: str = "Sort"
    task_kwargs: dict = field(default_factory=dict)
    # architecture: layer_specs are ("attn", {...}) or ("dense", {...}) -> BuilderModel
    layer_specs: tuple = (("attn",), ("dense", (64,)))
    d_model: int = 32
    n_heads: int = 4
    dropout: float = 0.0
    # training
    steps: int = 200
    batch: int = 128
    lr: float = 3e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    lr_schedule: str = "constant"
    # output head (lm = next-token; classify/regression use a pooled head + BuilderModel)
    head: str = "lm"                # "lm" | "classify" | "regression"
    pooling: str = "last"           # for pooled heads: "last"|"mean"|"cls"|"attn"
    n_classes: int = 2
    out_dim: int = 1
    pos_encoding: str = "learned"   # "learned" | "sinusoidal"
    init: str = "normal"            # weight init: normal|xavier|kaiming|orthogonal|zeros
    init_scale: float = 0.02        # std for "normal" init
    # reproducibility / replay
    seed: int = 0
    n_checkpoints: int = 80          # fidelity knob (>= steps+1 -> every step)
    device: str = "cpu"             # "cpu" | "mps" | "cuda"

    def to_json(self) -> str:
        d = asdict(self)
        d["layer_specs"] = [list(s) for s in self.layer_specs]
        return json.dumps(d)


def build_task(cfg: RunConfig) -> Task:
    if cfg.task_name not in ALL_TASKS:
        raise ValueError(f"unknown task {cfg.task_name!r}; have {list(ALL_TASKS)}")
    return ALL_TASKS[cfg.task_name](**cfg.task_kwargs)


def task_kind(task) -> str:
    """'lm' (next-token), 'classify', or 'regression' - the tasks declare it via .kind."""
    return getattr(task, "kind", "lm")


def _spec_to_layer(spec, cfg):
    """Map a layer_spec entry to a BuilderModel layer config. Entries may be:
       ("attn",) / ("attn", {causal,bias,attn_dropout,resid_dropout,wo_hidden,wo_activation})
       ("dense", hidden) / ("dense", {hidden,activation,bias,dropout})  (hidden = int or tuple)."""
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 else None
    if kind == "attn":
        d = opts if isinstance(opts, dict) else {}
        return ("attn", AttnCfg(
            causal=d.get("causal", True), bias=d.get("bias", True),
            attn_dropout=d.get("attn_dropout", cfg.dropout), resid_dropout=d.get("resid_dropout", cfg.dropout),
            wo_hidden=tuple(d.get("wo_hidden", ())), wo_activation=d.get("wo_activation", "gelu"),
            n_heads=d.get("n_heads")))
    d = opts if isinstance(opts, dict) else {"hidden": opts}
    hraw = d.get("hidden", 4 * cfg.d_model)
    hidden = tuple(hraw) if isinstance(hraw, (list, tuple)) else (int(hraw),)
    return ("ffn", FFNCfg(hidden=hidden, activation=d.get("activation", "gelu"),
                          bias=d.get("bias", True), dropout=d.get("dropout", cfg.dropout)))


def build_model(cfg: RunConfig, task, device="cpu", seed: int | None = None):
    """Construct the model (always the fully-configurable `BuilderModel`, so per-section
    settings — activation, bias, causal, W_O hidden, positional encoding, head/pooling —
    all take effect). Pass `seed` to make the random init reproducible."""
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
    if seed is not None:
        torch.manual_seed(seed)
    bc = BuilderConfig(
        vocab_size=task.vocab_size, block_size=task.block_size, d_model=cfg.d_model,
        n_heads=cfg.n_heads, pos_encoding=cfg.pos_encoding, dropout=cfg.dropout,
        layers=tuple(_spec_to_layer(s, cfg) for s in cfg.layer_specs),
        head=cfg.head, pooling=cfg.pooling,
        n_classes=getattr(task, "n_classes", cfg.n_classes),
        out_dim=getattr(task, "out_dim", cfg.out_dim),
        init=cfg.init, init_scale=cfg.init_scale)
    return BuilderModel(bc).to(device)


def _cpu_state(model) -> dict:
    """A detached, CPU copy of the model's state_dict (the exact weights)."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _token_accuracy(logits, y) -> float:
    """Per-token accuracy on the scored (non -100) positions of one batch (cheap, dense)."""
    mask = y != -100
    if mask.sum() == 0:
        return float("nan")
    correct = (logits.argmax(-1) == y) & mask
    return float(correct.sum().item() / mask.sum().item())


def _batch_score(out, target, kind) -> float:
    """A dense per-step quality score for pooled heads: accuracy (classify) or R^2 (regression)."""
    if kind == "classify":
        return float((out.argmax(-1) == target).float().mean().item())
    mse = float(F.mse_loss(out, target.float()).item())
    var = float(target.float().var().item()) + 1e-9
    return 1.0 - mse / var


# ---------------------------------------------------------------------------
# Held-out train/test split: a deterministic hash of each input row puts every
# example permanently in train (~80%) or test (~20%), so checkpoint accuracy is
# measured on examples the model was NEVER trained on (generalization, not memorization).
# ---------------------------------------------------------------------------
def _split_mask(x, split, ncols=None, frac=0.2):
    cols = x.shape[1] if ncols is None else max(1, min(int(ncols), x.shape[1]))
    flat = x[:, :cols].long()                                # hash the INPUT identity (prompt for LM)
    keys = torch.zeros(x.shape[0], dtype=torch.long)
    # polynomial rolling hash of the prompt. NB: for long prompts (>~8 tokens) this overflows int64, and
    # that wraparound is INTENTIONAL — torch wraps two's-complement deterministically, so the split stays
    # stable + correct in proportion (it only slightly degrades hash uniformity for the longest tasks).
    for c in range(flat.shape[1]):
        keys = keys * 131 + flat[:, c] + 1
    is_test = (keys.abs() % 100) < int(frac * 100)
    return is_test if split == "test" else ~is_test


def is_heldout(task, x) -> bool:
    """True if input `x` (prompt tokens for LM / full input for pooled) is in the held-out test split."""
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return bool(_split_mask(x, "test", getattr(task, "prompt_len", None))[0].item())


def split_batch(task, batch_size, device, generator, split):
    """task.make_batch restricted to the 'train' or 'test' partition (rejection-sampled on the
    hashed input identity). split='all' = unrestricted. Returns the same tuple shape as make_batch."""
    if split == "all":
        return task.make_batch(batch_size, device, generator=generator)
    ncols = getattr(task, "prompt_len", None)
    acc = None
    for _ in range(500):
        out = task.make_batch(max(2 * batch_size, 64), "cpu", generator=generator)
        sel = tuple(t[_split_mask(out[0], split, ncols)] for t in out)
        acc = sel if acc is None else tuple(torch.cat([a, b], 0) for a, b in zip(acc, sel))
        if acc[0].shape[0] >= batch_size:
            break
    return tuple(t[:batch_size].to(device) for t in acc)


# ---------------------------------------------------------------------------
# gradient introspection — computed ON THE FLY (never stored): one backward pass
# on a reconstructed model. Storing per-layer grads would be model-size × steps.
# ---------------------------------------------------------------------------
def _grad_norms(model) -> dict:
    """Per-block L2 gradient norms aligned with the flow bands; call AFTER loss.backward()."""
    dm = model.cfg.d_model

    def gnorm(ps):
        s = 0.0
        for p in ps:
            if p.grad is not None:
                s += float(p.grad.detach().pow(2).sum().item())
        return s ** 0.5

    res = {"attn": [], "ffn": []}
    emb = [model.tok_emb.weight]
    if getattr(model.cfg, "pos_encoding", "learned") == "learned":
        emb.append(model.pos_emb.weight)
    res["embed"] = gnorm(emb)
    def nrm(t):
        return float(t.pow(2).sum().item()) ** 0.5
    for layer in model.layers:
        if isinstance(layer, AttnBlock):
            g = layer.attn.qkv.weight.grad                     # [3*dm, dm]: Q | K | V stacked
            nh, dk = layer.attn.n_heads, layer.attn.d_k        # head h of Q = rows [h*dk:(h+1)*dk]
            wo_lins = [m for m in layer.attn.wo.modules() if isinstance(m, torch.nn.Linear)]
            wo_g = wo_lins[0].weight.grad if wo_lins else None  # [out, dm]; head h = input cols [h*dk:..]
            heads = []
            for h in range(nh):
                heads.append({
                    "q": nrm(g[h * dk:(h + 1) * dk]),
                    "k": nrm(g[dm + h * dk:dm + (h + 1) * dk]),
                    "v": nrm(g[2 * dm + h * dk:2 * dm + (h + 1) * dk]),
                    "o": nrm(wo_g[:, h * dk:(h + 1) * dk]) if wo_g is not None else 0.0})
            res["attn"].append({
                "q": nrm(g[0:dm]), "k": nrm(g[dm:2 * dm]), "v": nrm(g[2 * dm:3 * dm]),
                "o": gnorm(layer.attn.wo.parameters()),
                "all": gnorm(layer.attn.parameters()),
                "heads": heads})
        elif isinstance(layer, FFNBlock):
            lins = list(layer.hidden_layers)                  # up-projections (one per hidden layer)
            res["ffn"].append({
                "all": gnorm(layer.parameters()),
                "out": gnorm(layer.out_proj.parameters()),    # down-projection
                "layers": [gnorm(lin.parameters()) for lin in lins],
                "neurons": [[nrm(row) for row in lin.weight.grad] for lin in lins]})  # per-unit ∇
    res["head"] = gnorm(model.head.parameters())
    res["total"] = gnorm(model.parameters())
    return res


def _avg_grad_dicts(ds: list) -> dict:
    """Average a list of _grad_norms dicts element-wise."""
    n = len(ds)
    out = {k: sum(d[k] for d in ds) / n for k in ("embed", "head", "total")}
    out["attn"] = []
    for i in range(len(ds[0]["attn"])):
        band = {k: sum(d["attn"][i][k] for d in ds) / n for k in ("q", "k", "v", "o", "all")}
        nh = len(ds[0]["attn"][i]["heads"])
        band["heads"] = [{k: sum(d["attn"][i]["heads"][h][k] for d in ds) / n for k in ("q", "k", "v", "o")}
                         for h in range(nh)]
        out["attn"].append(band)
    out["ffn"] = []
    for i in range(len(ds[0]["ffn"])):
        fb = {k: sum(d["ffn"][i][k] for d in ds) / n for k in ("all", "out")}
        nl = len(ds[0]["ffn"][i]["layers"])
        fb["layers"] = [sum(d["ffn"][i]["layers"][l] for d in ds) / n for l in range(nl)]
        fb["neurons"] = [[sum(d["ffn"][i]["neurons"][l][u] for d in ds) / n
                          for u in range(len(ds[0]["ffn"][i]["neurons"][l]))] for l in range(nl)]
        out["ffn"].append(fb)
    return out


def layer_gradients(model, batches) -> dict:
    """One backward pass per (x, target) batch (eval mode = no dropout noise); returns the
    per-block gradient norms averaged over the batches. Computed on the fly, nothing stored."""
    was_training = model.training
    model.eval()
    accs = []
    for x, target in batches:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, target)
        loss.backward()
        accs.append(_grad_norms(model))
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return _avg_grad_dicts(accs)


def sample_train_batch(cfg: "RunConfig", task, seed: int):
    """A fresh train-split (x, target) batch with its own seed (for averaged / scrubbable batches)."""
    gen = torch.Generator().manual_seed(seed)
    out = split_batch(task, cfg.batch, "cpu", gen, "train")
    return out[0], out[1]


def exact_train_batch(cfg: "RunConfig", task, step: int):
    """Replay the deterministic training generator to return the exact (x, target) used at `step`."""
    gen = torch.Generator().manual_seed(cfg.seed + 12345)
    for _ in range(max(0, step)):
        split_batch(task, cfg.batch, "cpu", gen, "train")     # advance exactly as training did
    out = split_batch(task, cfg.batch, "cpu", gen, "train")
    return out[0], out[1]


def gradient_scale(run, task) -> float:
    """Largest per-block gradient magnitude over the WHOLE run (sampled checkpoints, one fresh train
    batch each) — used to PIN the gradient-bar axis so magnitudes are comparable across training steps."""
    steps = run.checkpoint_steps
    if len(steps) > 14:                                        # sample to keep it cheap
        steps = sorted({steps[round(k * (len(steps) - 1) / 13)] for k in range(14)})
    vals = [1e-9]
    for i, stp in enumerate(steps):
        g = layer_gradients(run.reconstruct(stp), [sample_train_batch(run.cfg, task, 90000 + i)])
        vals += [g["embed"], g["head"]]
        for a in g["attn"]:
            vals += [a["q"], a["k"], a["v"], a["o"]]
            vals += [h[p] for h in a["heads"] for p in ("q", "k", "v", "o")]
        for f in g["ffn"]:
            vals += [f["all"], f["out"], *f["layers"]]
    return max(vals)


# ---------------------------------------------------------------------------
# Checkpoint + Run
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    step: int
    state_dict: dict          # exact weights, on CPU
    eval_loss: float
    acc: float                # exact-match accuracy on a fixed eval batch
    lr: float


class TrainingRun:
    """A completed training run: dense metrics + sparse checkpoints + everything needed
    to reconstruct any state exactly."""

    def __init__(self, cfg: RunConfig, task, metrics: dict, checkpoints: list[Checkpoint],
                 init_state: dict, n_params: int):
        self.cfg = cfg
        self.task = task
        self.metrics = metrics                 # dense: step/train_loss/train_acc/lr/grad_norm
        self.checkpoints = checkpoints         # sparse
        self.init_state = init_state           # weights before any training (step 0)
        self.n_params = n_params

    # --- training ---------------------------------------------------------
    @classmethod
    def train(cls, cfg: RunConfig, on_step=None, on_checkpoint=None) -> "TrainingRun":
        model, metrics, checkpoints, init_state, _ = _run_training(
            cfg, stop_at=cfg.steps, capture=True, on_step=on_step, on_checkpoint=on_checkpoint)
        return cls(cfg, build_task(cfg), metrics, checkpoints, init_state, count_params(model))

    # --- checkpoint access ------------------------------------------------
    @property
    def checkpoint_steps(self) -> list[int]:
        return [c.step for c in self.checkpoints]

    def nearest_checkpoint(self, step: int) -> Checkpoint:
        return min(self.checkpoints, key=lambda c: abs(c.step - step))

    def reconstruct(self, step: int, device="cpu") -> BuilderModel:
        """Load the NEAREST checkpoint's exact weights (snap-to-nearest; never interpolated)."""
        ck = self.nearest_checkpoint(step)
        model = build_model(self.cfg, self.task, device)
        model.load_state_dict(ck.state_dict)
        model.eval()
        return model

    def reconstruct_exact(self, step: int) -> BuilderModel:
        """Reconstruct the model EXACTLY at `step` by deterministic replay (re-run on CPU)."""
        model, *_ = _run_training(replace(self.cfg, device="cpu"), stop_at=step, capture=False)
        model.eval()
        return model

    # --- inspection -------------------------------------------------------
    def frame(self, step: int, probe_seq, exact=False) -> dict:
        """Everything the viz needs to draw one scrub position: snapped step, the metric
        curves, and the recomputed activation trace for `probe_seq`."""
        model = self.reconstruct_exact(step) if exact else self.reconstruct(step)
        snapped = step if exact else self.nearest_checkpoint(step).step
        return {
            "requested_step": int(step),
            "snapped_step": int(snapped),
            "metrics": {k: list(v) for k, v in self.metrics.items()},
            "checkpoint_steps": self.checkpoint_steps,
            "trace": trace_forward(model, self.task, probe_seq),
        }

    # --- storage accounting ----------------------------------------------
    def nbytes(self) -> int:
        """Bytes held by the checkpoints (the only heavy thing) - for the efficiency story."""
        return sum(v.element_size() * v.nelement()
                   for c in self.checkpoints for v in c.state_dict.values())


def _run_training(cfg: RunConfig, *, stop_at: int, capture: bool, on_step=None, on_checkpoint=None):
    """The single, deterministic training loop used BOTH for the recorded run and for
    exact replay. Determinism: the model init uses the global RNG (seeded once); training
    batches use a dedicated CPU generator; the checkpoint/eval code uses its own generators
    and runs in eval mode, so it never perturbs the training RNG stream. Re-running with the
    same seed therefore reproduces the trajectory bit-for-bit on CPU.
    """
    task = build_task(cfg)
    kind = task_kind(task)
    device = get_device(cfg.device)
    model = build_model(cfg, task, device, seed=cfg.seed)
    init_state = _cpu_state(model)
    opt = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)
    batch_gen = torch.Generator().manual_seed(cfg.seed + 12345)   # separate from the global RNG

    ckpt_steps = set(log_spaced_steps(cfg.steps, cfg.n_checkpoints)) if capture else set()
    metrics = {"step": [], "train_loss": [], "train_acc": [], "lr": [], "grad_norm": []}
    checkpoints: list[Checkpoint] = []

    for s in range(stop_at + 1):
        if capture and s in ckpt_steps:
            ck = _capture_checkpoint(model, task, device, s, cfg)
            checkpoints.append(ck)
            if on_checkpoint:
                on_checkpoint(ck, checkpoints)
        if s == stop_at:
            break

        lr_now = _lr_at(s, cfg.steps, cfg.lr, cfg.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = lr_now
        if kind == "lm":
            x, y, _ = split_batch(task, cfg.batch, device, batch_gen, "train")
            out, loss = model(x, y)
            train_acc = _token_accuracy(out.detach(), y)
        else:
            x, target = split_batch(task, cfg.batch, device, batch_gen, "train")
            out, loss = model(x, target)
            train_acc = _batch_score(out.detach(), target, kind)
        opt.zero_grad()
        loss.backward()
        gnorm = _grad_norm(model)
        opt.step()

        if capture:
            metrics["step"].append(s)
            metrics["train_loss"].append(float(loss.item()))
            metrics["train_acc"].append(train_acc)
            metrics["lr"].append(lr_now)
            metrics["grad_norm"].append(gnorm)
            if on_step:
                on_step(s, metrics)

    return model, metrics, checkpoints, init_state, device


def continue_training(run: "TrainingRun", extra_steps: int, on_step=None, on_checkpoint=None):
    """Warm-start from the run's FINAL checkpoint and train `extra_steps` more, APPENDING the new
    checkpoints + dense metrics to `run` in place. Additive: the run object grows, reconstruct
    (snap-to-nearest) keeps working, and the config signature is untouched so the trained view stays.
    Caveat: optimizer momentum restarts here, and reconstruct_exact won't cover the appended span."""
    cfg, task = run.cfg, run.task
    device = get_device(cfg.device)
    kind = task_kind(task)
    model = run.reconstruct(run.checkpoint_steps[-1], device)
    model.train()
    opt = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)
    base = int(run.checkpoint_steps[-1])
    new_total = base + int(extra_steps)
    new_cfg = replace(cfg, steps=new_total)                     # for correct lr scheduling/capture
    gen = torch.Generator().manual_seed(cfg.seed + 12345 + base)   # reproducible continuation stream
    want = {s for s in log_spaced_steps(new_total, cfg.n_checkpoints) if s > base} | {new_total}
    for s in range(base, new_total + 1):
        if s in want:
            ck = _capture_checkpoint(model, task, device, s, new_cfg)
            run.checkpoints.append(ck)
            if on_checkpoint:
                on_checkpoint(ck, run.checkpoints)
        if s == new_total:
            break
        lr_now = _lr_at(s, new_total, cfg.lr, cfg.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = lr_now
        if kind == "lm":
            x, y, _ = split_batch(task, cfg.batch, device, gen, "train")
            out, loss = model(x, y)
            acc = _token_accuracy(out.detach(), y)
        else:
            x, target = split_batch(task, cfg.batch, device, gen, "train")
            out, loss = model(x, target)
            acc = _batch_score(out.detach(), target, kind)
        opt.zero_grad()
        loss.backward()
        gn = _grad_norm(model)
        opt.step()
        run.metrics["step"].append(s)
        run.metrics["train_loss"].append(float(loss.item()))
        run.metrics["train_acc"].append(acc)
        run.metrics["lr"].append(lr_now)
        run.metrics["grad_norm"].append(gn)
        if on_step:
            on_step(s, run.metrics)
    run.checkpoints.sort(key=lambda c: c.step)
    return run


def _capture_checkpoint(model, task, device, step, cfg) -> Checkpoint:
    """Exact weights (CPU) + eval metrics at this step. Eval uses its own seeded generator
    and eval mode, so it does not consume the training RNG (keeps replay deterministic)."""
    was = model.training
    model.eval()
    if task_kind(task) == "lm":
        eval_loss, score = _eval_lm(model, task, device, split="test")
    else:
        eval_loss, score = _eval_pooled(model, task, device, split="test")
    state = _cpu_state(model)
    model.train(was)
    return Checkpoint(step=step, state_dict=state, eval_loss=eval_loss, acc=score,
                      lr=_lr_at(min(step, max(0, cfg.steps - 1)), cfg.steps, cfg.lr, cfg.lr_schedule))


@torch.no_grad()
def _eval_lm(model, task: Task, device, split="test", batch=512):
    """(loss, exact-match accuracy) for an LM task on the given split (default = held-out test)."""
    x, y, seq = split_batch(task, batch, device, torch.Generator().manual_seed(1), split)
    model.eval()
    _, loss = model(x, y)
    out = generate(model, task, seq, device)
    target = seq[:, task.prompt_len:]
    exact = float((out == target).all(dim=1).float().mean().item())
    return float(loss.item()), exact


@torch.no_grad()
def _eval_pooled(model, task: Task, device, split="test", batch=512):
    """(loss, score) for a pooled head on the given split: accuracy (classify) / R^2 (regression)."""
    x, target = split_batch(task, batch, device, torch.Generator().manual_seed(1), split)
    out, loss = model(x, target)
    return float(loss.item()), _batch_score(out, target, task_kind(task))


@torch.no_grad()
def per_category_eval(model, task: Task, device="cpu", batch=512, split="test"):
    """For a task that defines `category_of(x_row, y_row)`, return {category: (accuracy, count)} on the
    given split (default = held-out). Accuracy is exact-match of the generated output (LM) or argmax ==
    label (classify); the counts double as the category distribution. Tasks without `category_of` skip this."""
    model.eval()
    x, y, *rest = split_batch(task, batch, device, torch.Generator().manual_seed(1), split)
    if task_kind(task) == "lm":
        preds = generate(model, task, rest[0], device)
        correct = (preds == rest[0][:, task.prompt_len:]).all(dim=1)
    else:
        logits, _ = model(x, y)
        correct = (logits.argmax(dim=-1) == y)
    agg = {}
    for i in range(x.shape[0]):
        cat = getattr(task, "category_of")(x[i], y[i])   # opt-in (not on the base Task protocol)
        if cat is None:
            continue
        a = agg.setdefault(cat, [0, 0])
        a[0] += int(correct[i].item()); a[1] += 1
    return {c: (n / t, t) for c, (n, t) in sorted(agg.items()) if t}


def train_eval_curve(run):
    """Per-checkpoint TRAIN-split (losses, accs), computed on the fly: reconstruct each checkpoint
    and evaluate on the train split with the SAME method as the stored held-out eval. The held-out
    curve already lives on the checkpoints; this is its train-set counterpart for the gap plot."""
    task = run.task
    ev = _eval_lm if task_kind(task) == "lm" else _eval_pooled
    losses, accs = [], []
    for stp in run.checkpoint_steps:
        loss, score = ev(run.reconstruct(stp), task, "cpu", split="train")
        losses.append(loss)
        accs.append(score)
    return losses, accs


# ---------------------------------------------------------------------------
# Activation trace (recomputed on demand; JSON-serializable for the D3 viz)
# ---------------------------------------------------------------------------
@torch.no_grad()
def trace_forward(model: BuilderModel, task: Task, seq) -> dict:
    """Run one eval-mode forward on `seq` and return all the values the viz draws:
    the residual stream at each stage, each attention layer's grid + Q/K/V, each dense
    block's hidden activations, and the next-token logits/probs. Everything is plain
    Python lists/numbers so it serializes straight to JSON.
    """
    if not torch.is_tensor(seq):
        seq = torch.tensor(seq, dtype=torch.long)
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    device = next(model.parameters()).device
    seq = seq.to(device)
    model.eval()

    # capture Q/K/V via a hook on each attention block's packed qkv projection
    attn_layers = model.attention_layers()
    qkv_cache: dict[int, torch.Tensor] = {}
    handles = []
    for i, al in enumerate(attn_layers):
        handles.append(al.attn.qkv.register_forward_hook(
            lambda m, inp, out, i=i: qkv_cache.__setitem__(i, out.detach())))

    stages = model.forward_trace(seq)          # sets last_attn / last_hidden; returns residuals
    for h in handles:
        h.remove()

    T = seq.shape[1]
    tokens = seq[0].tolist()
    out = {
        "tokens": tokens,
        "token_strs": [task.id_to_str.get(int(t), "?") for t in tokens],
        "vocab_strs": [task.id_to_str.get(i, "?") for i in range(task.vocab_size)],
        "T": int(T),
        "d_model": int(model.cfg.d_model),
        "vocab_size": int(task.vocab_size),
        "stages": [{"label": lbl, "residual": resid[0].cpu().tolist()} for lbl, resid in stages],
        "attention": [],
        "dense": [],
    }

    for i, al in enumerate(attn_layers):
        weights = al.attn.last_attn[0].cpu()            # [n_heads, T, T]
        entry = {"layer": i, "weights": weights.tolist()}
        if i in qkv_cache:
            lnh = al.attn.n_heads                       # THIS layer's head count (may differ per layer)
            ldk = getattr(al.attn, "d_k", model.cfg.d_model // lnh)
            q, k, v = qkv_cache[i][0].split(model.cfg.d_model, dim=-1)   # each [T, C]
            reshape = lambda t: t.view(T, lnh, ldk).permute(1, 0, 2).cpu().tolist()  # [lnh,T,ldk]
            entry.update(q=reshape(q), k=reshape(k), v=reshape(v))
        out["attention"].append(entry)

    for i, dl in enumerate(model.dense_layers()):
        hiddens = [hh[0].cpu().tolist() for hh in (getattr(dl, "last_hiddens", None) or [dl.last_hidden])]
        out["dense"].append({"layer": i, "hidden": hiddens[-1], "hiddens": hiddens})

    head_kind = getattr(getattr(model, "cfg", None), "head", "lm")
    if head_kind == "lm":
        final = stages[-1][1].to(device)
        logits = model.head(model.ln_f(final))[0]        # [T, vocab]
        out["head"] = "lm"
        out["logits"] = logits.cpu().tolist()
        out["probs"] = torch.softmax(logits, dim=-1).cpu().tolist()
    else:                                                 # pooled head: one output per sequence
        o, _ = model(seq)                                # [1, n_classes] or [1, out_dim]
        out["head"] = head_kind
        out["output"] = o[0].cpu().tolist()
        if head_kind == "classify":
            out["probs"] = torch.softmax(o[0], dim=-1).cpu().tolist()
        pool = getattr(model, "pool", None)
        if pool is not None and getattr(pool, "last_weights", None) is not None:
            out["pool_weights"] = pool.last_weights[0].cpu().tolist()   # attention-pooling weights
    return out


# ---------------------------------------------------------------------------
# Sampling (the Output head's generation knobs)
# ---------------------------------------------------------------------------
def sample_token(logits, method="greedy", temperature=1.0, top_k=0, top_p=0.0, generator=None):
    """Turn a 1-D logits vector into one chosen token id (int).

    method: 'greedy' (argmax) | 'temperature' | 'top_k' | 'top_p' (nucleus).
    'temperature' alone also applies to the top_k/top_p variants.
    """
    logits = logits.detach().float().clone()
    if method == "greedy":
        return int(logits.argmax().item())

    if temperature and temperature > 0:
        logits = logits / temperature

    if method == "top_k" and top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    if method == "top_p" and top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cum > top_p
        cutoff[0] = False                       # always keep the top token
        remove = sorted_idx[cutoff]
        logits[remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate_sampled(model, task, seq, method="greedy", temperature=1.0, top_k=0, top_p=0.0,
                     seed=0, device="cpu"):
    """Autoregressively generate the output region with the chosen sampling method.

    Works for a batch: each row is sampled independently. With method='greedy' this is
    exactly the argmax decoding in tiny_gpt.generate.
    """
    p = task.prompt_len
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    if not torch.is_tensor(seq):
        seq = torch.tensor(seq, dtype=torch.long)
    ctx = seq[:, :p].to(device)
    B = ctx.shape[0]
    for _ in range(task.gen_len):
        logits, _ = model(ctx)
        last = logits[:, -1, :]                                      # [B, vocab]
        nxt = torch.tensor(
            [sample_token(last[b], method, temperature, top_k, top_p, gen) for b in range(B)],
            device=ctx.device, dtype=torch.long).unsqueeze(1)        # [B, 1]
        ctx = torch.cat([ctx, nxt], dim=1)
    return ctx[:, p:]


# ---------------------------------------------------------------------------
# tiny end-to-end demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = RunConfig(task_name="Reverse", task_kwargs={"length": 5},
                    layer_specs=(("attn",), ("dense", (64,))),
                    d_model=32, n_heads=4, steps=150, n_checkpoints=20, seed=0, device="cpu")
    run = TrainingRun.train(cfg)
    print(f"params={run.n_params}  checkpoints={len(run.checkpoints)} at {run.checkpoint_steps}")
    print(f"first acc={run.checkpoints[0].acc:.2f}  last acc={run.checkpoints[-1].acc:.2f}")
    print(f"checkpoint storage = {run.nbytes()/1e6:.2f} MB")
    task = run.task
    _, _, probe = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    fr = run.frame(run.checkpoint_steps[-1], probe[0])
    print("trace keys:", list(fr["trace"]))
    print("attention layers in trace:", len(fr["trace"]["attention"]))
    print("JSON-serializable:", bool(json.dumps(fr)))
