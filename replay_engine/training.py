"""replay_engine.training — the deterministic training loop, checkpoints, TrainingRun, and evals.

The heart of the package: TrainingRun records dense metrics + sparse log-spaced checkpoints and can
reconstruct any step exactly. _run_training is the single deterministic loop used for both the recorded
run and exact replay; continue_training appends more steps. Evals run on the held-out split.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from lm_utils import get_device, generate
from tasks import Task
from training_utils import make_optimizer, _lr_at
from builder_model import BuilderModel, count_params

from .config import (RunConfig, build_task, build_model, task_kind, log_spaced_steps,
                     EVAL_SEED, TRAIN_STREAM_OFFSET)
from .splits import split_batch
from .gradients import _grad_norm
from .trace import trace_forward


def _cpu_state(model) -> dict:
    """A detached, CPU copy of the model's state_dict (the exact weights)."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


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


def _train_one_step(model, opt, task, kind, gen, cfg, s, total, device):
    """One optimizer step on a fresh train batch; returns (loss, accuracy, lr, grad_norm). The single
    source of truth for the step, shared by _run_training (record) and continue_training (append)."""
    lr_now = _lr_at(s, total, cfg.lr, cfg.lr_schedule)
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
    gnorm = _grad_norm(model)
    opt.step()
    return float(loss.item()), acc, lr_now, gnorm


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
    batch_gen = torch.Generator().manual_seed(cfg.seed + TRAIN_STREAM_OFFSET)   # separate from the global RNG

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

        loss_v, train_acc, lr_now, gnorm = _train_one_step(
            model, opt, task, kind, batch_gen, cfg, s, cfg.steps, device)
        if capture:
            metrics["step"].append(s)
            metrics["train_loss"].append(loss_v)
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
    gen = torch.Generator().manual_seed(cfg.seed + TRAIN_STREAM_OFFSET + base)   # reproducible continuation stream
    want = {s for s in log_spaced_steps(new_total, cfg.n_checkpoints) if s > base} | {new_total}
    for s in range(base, new_total + 1):
        if s in want:
            ck = _capture_checkpoint(model, task, device, s, new_cfg)
            run.checkpoints.append(ck)
            if on_checkpoint:
                on_checkpoint(ck, run.checkpoints)
        if s == new_total:
            break
        loss_v, acc, lr_now, gn = _train_one_step(model, opt, task, kind, gen, cfg, s, new_total, device)
        run.metrics["step"].append(s)
        run.metrics["train_loss"].append(loss_v)
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
    x, y, seq = split_batch(task, batch, device, torch.Generator().manual_seed(EVAL_SEED), split)
    model.eval()
    _, loss = model(x, y)
    out = generate(model, task, seq, device)
    target = seq[:, task.prompt_len:]
    exact = float((out == target).all(dim=1).float().mean().item())
    return float(loss.item()), exact


@torch.no_grad()
def _eval_pooled(model, task: Task, device, split="test", batch=512):
    """(loss, score) for a pooled head on the given split: accuracy (classify) / R^2 (regression)."""
    x, target = split_batch(task, batch, device, torch.Generator().manual_seed(EVAL_SEED), split)
    out, loss = model(x, target)
    return float(loss.item()), _batch_score(out, target, task_kind(task))


@torch.no_grad()
def per_category_eval(model, task: Task, device="cpu", batch=512, split="test"):
    """For a task that defines `category_of(x_row, y_row)`, return {category: (accuracy, count)} on the
    given split (default = held-out). Accuracy is exact-match of the generated output (LM) or argmax ==
    label (classify); the counts double as the category distribution. Tasks without `category_of` skip this."""
    model.eval()
    x, y, *rest = split_batch(task, batch, device, torch.Generator().manual_seed(EVAL_SEED), split)
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
