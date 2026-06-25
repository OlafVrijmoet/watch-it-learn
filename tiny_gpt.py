"""
tiny_gpt.py — the Sort task + the language-model helpers (generate / evaluate / decode).

This file once held a hand-written `TinyGPT` model (the project began as a "teach a tiny GPT to
sort" notebook). That model was superseded by the fully-configurable `BuilderModel` and removed;
what remains is the Sort task and the task-generic LM utilities the engine still uses:

  - `SortTask`  : sort short sequences of DISTINCT digits, framed as next-token prediction.
  - `generate`  : greedy autoregressive decode of the output region, for any task with prompt_len/gen_len.
  - `evaluate`  : loss + exact-match + per-token accuracy on a fresh batch.
  - `count_params` / `get_device` / `decode` helpers.

These operate on any `nn.Module` exposing the `(logits, loss) = model(idx, targets)` interface
(i.e. `BuilderModel`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# The task: sort short sequences of DISTINCT digits, as next-token prediction.
#
#   layout:   d d d d d d  |  s s s s s s
#             \--input---/  ^  \--sorted--/
#                          sep
#
# We use *distinct* digits so each sorted output token maps to exactly ONE input
# position. That makes the attention pattern crisp and readable: the model has to
# learn "to emit the k-th smallest, look at the input position holding it".
# ---------------------------------------------------------------------------
SEP = 10                                  # separator token id (digits use ids 0..9)
ID_TO_STR = {i: str(i) for i in range(10)}
ID_TO_STR[SEP] = "|"


def decode(ids) -> str:
    """Turn a sequence of token ids into a readable string."""
    return " ".join(ID_TO_STR[int(i)] for i in ids)


@dataclass
class SortTask:
    n_digits: int = 6           # how many digits to sort
    vocab_size: int = 11        # digits 0..9  +  SEP

    # --- shared task interface (see tasks.py for more tasks with this shape) ---
    name = "Sort"
    description = "Sort the input digits into ascending order."
    params = {"n_digits": (4, 9, 6)}   # {kwarg: (min, max, default)} for the playground UI
    id_to_str = ID_TO_STR

    @property
    def block_size(self) -> int:
        return self.n_digits * 2 + 1     # input + separator + output

    @property
    def prompt_len(self) -> int:
        return self.n_digits + 1         # input digits + the separator (fed before generating)

    @property
    def gen_len(self) -> int:
        return self.n_digits             # how many tokens to generate

    def decode(self, ids) -> str:
        return " ".join(self.id_to_str.get(int(i), "?") for i in ids)

    def make_batch(self, batch_size: int, device="cpu", generator=None):
        """Return (x, y, seq).

        seq : [B, 2n+1]  the full sequence  (input | sorted)
        x   : [B, 2n]    model input        (seq without the last token)
        y   : [B, 2n]    next-token targets (seq shifted left); the input region
                         is set to -100 so loss is only counted on the sorted output.
        """
        n = self.n_digits
        B = batch_size

        # sample n distinct digits per row: argsort random keys, take the first n.
        keys = torch.rand(B, 10, generator=generator)
        unsorted = keys.argsort(dim=1)[:, :n]        # [B, n] distinct ids in 0..9
        sorted_, _ = unsorted.sort(dim=1)
        sep = torch.full((B, 1), SEP, dtype=torch.long)
        seq = torch.cat([unsorted, sep, sorted_], dim=1)   # [B, 2n+1]

        x = seq[:, :-1].contiguous()      # everything but the last token
        y = seq[:, 1:].clone()            # next token at each position
        y[:, :n] = -100                   # only learn on the sorted-output region
        return x.to(device), y.to(device), seq.to(device)


@torch.no_grad()
def generate(model: nn.Module, task, seq: torch.Tensor, device) -> torch.Tensor:
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
def evaluate(model: nn.Module, task, device, batch_size=512, generator=None):
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
    """Pick a device. This model is tiny, so CPU is the safe, fast default.

    MPS (an M1 GPU) shines on much larger models; for ~50k params the kernel
    launch overhead can actually make MPS *slower*. Pass prefer="mps" to try it.
    """
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
