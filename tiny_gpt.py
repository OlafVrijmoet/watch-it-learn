"""
tiny_gpt.py - a tiny decoder-only Transformer (GPT-style), written for learning.

This maps term-for-term onto the "How Transformers Work" notes:

    token IDs
      -> embedding table        (tok_emb: [vocab_size x d_model])
      -> + positional embedding  (so the model knows token order)
      -> attention layers        (Q/K/V from learned W_Q/W_K/W_V, context-aware blend)
      -> stacked blocks          (attention + feed-forward, repeated n_layers times)
      -> next-token logits        (head: [d_model -> vocab_size])

Everything is kept small and explicit. In particular, attention is written out
by hand (instead of using nn.MultiheadAttention) so that:
  1. the Q/K/V math is visible, and
  2. we can capture the attention grid on every forward pass and *plot* it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config - these are the "knobs" from your notes.
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int        # number of distinct tokens   (rows of the embedding table)
    block_size: int        # context window n            (how many tokens fed in at once)
    d_model: int = 64      # width of each token vector as it flows through the model
    n_heads: int = 4       # number of attention heads   (d_k = d_model // n_heads)
    n_layers: int = 2      # how many transformer blocks are stacked
    dropout: float = 0.0

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Attention - the heart of the model, written out so Q/K/V are explicit.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head *causal* self-attention.

    For every token we compute (per head):

        Q = x @ W_Q        K = x @ W_K        V = x @ W_V      (the vectors)
        scores  = Q @ K^T / sqrt(d_k)              -> how much each token attends to each
        weights = softmax(scores)                  -> the n x n attention grid (rows sum to 1)
        out     = weights @ V                      -> a context-aware blend of the values

    "Causal" means a token may only attend to itself and earlier tokens, never
    the future - that is what makes it a *next-token predictor*.

    W_Q / W_K / W_V are shared across all positions (the learned matrices). Here
    they are packed into one Linear (`qkv`) for speed, then split apart.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_k
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # produces Q, K, V together
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # mixes heads back together
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Lower-triangular mask: position i may attend to positions <= i.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

        # The attention grid from the most recent forward pass, kept for plotting.
        # Shape: [batch, n_heads, T, T].
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)                       # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)           # each [B, T, C]

        # split the channels into heads -> [B, n_heads, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)        # [B, nh, T, T]
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                            # the attention grid
        self.last_attn = weights.detach()                              # save it for viz
        weights = self.attn_drop(weights)

        out = weights @ v                                              # [B, nh, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, C)          # recombine heads
        out = self.resid_drop(self.proj(out))
        return out


class Block(nn.Module):
    """One transformer block: attention + feed-forward, each with a residual.

    The residual ("x + sublayer(x)") is what lets many blocks stack without the
    signal washing out - it keeps d_model constant from input to output.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sub-layer  (+ residual)
        x = x + self.mlp(self.ln2(x))    # feed-forward sub-layer (+ residual)
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # the embedding table
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)   # tells the model the order
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # -> next-token logits
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: [B, T] token ids. targets: [B, T] next-token ids (-100 = ignore)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]   # [B, T, d_model]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention grids [B, n_heads, T, T] from the most recent forward, per layer."""
        return [blk.attn.last_attn for blk in self.blocks]


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
def generate(model: TinyGPT, task, seq: torch.Tensor, device) -> torch.Tensor:
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


# kept for the notebook, which imports this name
generate_sorted = generate


@torch.no_grad()
def evaluate(model: TinyGPT, task, device, batch_size=512, generator=None):
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

    MPS (your M1 GPU) shines on much larger models; for ~50k params the kernel
    launch overhead can actually make MPS *slower*. Pass prefer="mps" to try it.
    """
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
