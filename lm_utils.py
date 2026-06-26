"""lm_utils.py — task-generic language-model helpers the engine reuses.

`generate` / `evaluate` operate on any `nn.Module` exposing the `(logits, loss) = model(idx, targets)`
interface (i.e. `BuilderModel`) and any task exposing `prompt_len` / `gen_len` / `make_batch` (the Task
protocol). `get_device` picks cpu / mps / cuda.

(This file was once `tiny_gpt.py`, which began as a "teach a tiny GPT to sort" notebook. The hand-written
`TinyGPT` model was superseded by `BuilderModel` and removed; the Sort task moved to `tasks.py`. What's
left here is the model-agnostic LM utility code.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:                       # type hints only (tasks no longer imports us, so no cycle)
    from tasks import Task


@torch.no_grad()
def generate(model: nn.Module, task: Task, seq: torch.Tensor, device) -> torch.Tensor:
    """Greedily decode the output region, given the prompt (input + separator).

    Real autoregressive generation: feed the prompt, take the argmax, append it,
    feed it back, repeat `gen_len` times. Works for ANY task that defines
    `prompt_len` and `gen_len`.
    """
    p = task.prompt_len
    was_training = model.training
    model.eval()
    ctx = seq[:, :p].to(device)                 # the prompt (input + separator)
    for _ in range(task.gen_len):
        logits, _ = model(ctx)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ctx = torch.cat([ctx, nxt], dim=1)
    model.train(was_training)
    return ctx[:, p:]                           # the generated output block


@torch.no_grad()
def evaluate(model: nn.Module, task: Task, device, batch_size=512, generator=None):
    """Return (loss, exact_match_accuracy, per_token_accuracy) on a fresh batch."""
    was_training = model.training
    model.eval()
    x, y, seq = task.make_batch(batch_size, device, generator)
    _, loss = model(x, y)
    preds = generate(model, task, seq, device)
    truth = seq[:, task.prompt_len :]
    exact = (preds == truth).all(dim=1).float().mean().item()
    per_token = (preds == truth).float().mean().item()
    model.train(was_training)
    return loss.item(), exact, per_token


def get_device(prefer: str = "cpu") -> torch.device:
    """Pick a device. These models are tiny, so CPU is the safe, fast default.

    MPS (an M1 GPU) shines on much larger models; for ~50k params the kernel
    launch overhead can actually make MPS *slower*. Pass prefer="mps" to try it.
    """
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
