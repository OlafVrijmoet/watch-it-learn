"""replay_engine.splits — the deterministic held-out train/test split + batch sampling.

A hash of each input row puts every example permanently in train (~80%) or test (~20%), so checkpoint
accuracy is measured on examples the model was NEVER trained on (generalization, not memorization).
"""
from __future__ import annotations

import torch

from .config import RunConfig, TRAIN_STREAM_OFFSET


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


def sample_train_batch(cfg: RunConfig, task, seed: int):
    """A fresh train-split (x, target) batch with its own seed (for averaged / scrubbable batches)."""
    gen = torch.Generator().manual_seed(seed)
    out = split_batch(task, cfg.batch, "cpu", gen, "train")
    return out[0], out[1]


def exact_train_batch(cfg: RunConfig, task, step: int):
    """Replay the deterministic training generator to return the exact (x, target) used at `step`."""
    gen = torch.Generator().manual_seed(cfg.seed + TRAIN_STREAM_OFFSET)
    for _ in range(max(0, step)):
        split_batch(task, cfg.batch, "cpu", gen, "train")     # advance exactly as training did
    out = split_batch(task, cfg.batch, "cpu", gen, "train")
    return out[0], out[1]
