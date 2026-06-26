"""replay_engine.config — run configuration, task/model construction, checkpoint cadence.

The leaf of the package: RunConfig (the full architecture/task/training/repro spec), the functions that
build a task + model from it, and the log-spaced checkpoint schedule. Everything else builds on this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from tasks import TASKS, Task
from builder_model import BuilderModel, BuilderConfig, AttnCfg, FFNCfg, MajorityTask, DensityTask

# next-token (LM) tasks + the pooled-head (classification / regression) tasks
ALL_TASKS = {**TASKS, "Majority": MajorityTask, "Density": DensityTask}

# Seed offsets — each carves an independent-but-deterministic RNG stream off cfg.seed, so they never
# collide with the model-init RNG or with each other (named so the intent + the spacing are legible).
TRAIN_STREAM_OFFSET = 12345      # training-batch generator (continuation adds the base step on top)
EVAL_SEED = 1                    # the fixed held-out / train eval batch
GRAD_SCALE_SEED_BASE = 90000     # per-checkpoint batches when pinning the gradient-bar axis


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


@dataclass
class RunConfig:
    # task
    task_name: str = "Sort"
    task_kwargs: dict = field(default_factory=dict)
    # architecture: layer_specs are ("attn", {...}) or ("ffn", {...}) -> BuilderModel
    layer_specs: tuple = (("attn",), ("ffn", (64,)))
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
       ("ffn", hidden) / ("ffn", {hidden,activation,bias,dropout})  (hidden = int or tuple).
    The legacy "dense" tag is still accepted (any non-"attn" entry builds an FFN block)."""
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
