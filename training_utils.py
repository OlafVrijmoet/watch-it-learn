"""
training_utils.py - small, dependency-light helpers for building and training the tiny models:
an optimizer factory, a learning-rate schedule, and two hyperparameter heuristics for the UI.
"""
from __future__ import annotations

import math

import torch


def make_optimizer(name, params, lr, weight_decay=0.0):
    """Build an optimizer by name: 'adamw' | 'adam' | 'sgd' | 'rmsprop'."""
    name = name.lower().split()[0]                    # 'sgd (momentum)' -> 'sgd'
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def _lr_at(step, total, base_lr, schedule):
    """Learning rate at a given step for the schedule: constant | cosine | warmup + cosine."""
    if schedule == "constant":
        return base_lr
    warmup = int(0.1 * total) if "warmup" in schedule else 0
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup                          # linear ramp up
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))       # cosine decay to ~0


def divisor_heads(d_model, options=(1, 2, 4, 8, 16)):
    """Head counts that evenly divide d_model (so d_k is an integer)."""
    return [h for h in options if d_model % h == 0]


def suggest_lr_transformer(d_model, n_layers, batch=128, optimizer="adamw"):
    """Heuristic peak LR for a transformer of this size.

    Anchored at the ~3e-4 AdamW standard for width 256; scaled ~1/width (wider -> lower),
    lower for deeper stacks (1/sqrt(depth)), higher for bigger batches (sqrt). SGD needs a
    much larger step, so it's bumped ~30x. Tiny models land higher, which is expected.
    """
    lr = 3e-4 * (256 / max(1, d_model)) / math.sqrt(max(1, n_layers)) * math.sqrt(batch / 128)
    if optimizer.lower().startswith("sgd"):
        return float(min(0.5, max(1e-2, lr * 30)))
    return float(min(1e-2, max(1e-4, lr)))
