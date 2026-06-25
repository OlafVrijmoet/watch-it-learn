"""
builder_model.py - the fully configurable model behind the builder UI.

This realizes the "Layer settings reference" spec: every block type and its knobs.

  * Attention      - # heads, causal, bias, dropout, and a CONFIGURABLE W_O
                     (output projection): plain linear by default, or an MLP
                     (hidden layers + activation) for the non-standard experiment.
  * Feed-forward   - hidden sizes, activation (gelu/relu/silu/tanh), bias, dropout.
  * Positional     - learned-absolute or sinusoidal.
  * Output head    - LM (next-token, weight-tying optional) / classification / regression,
                     with pooling (last / mean / CLS / attention) for the pooled heads.

It is a SUPERSET of the LM stack used elsewhere: with the default config it is an
ordinary pre-LN decoder, exposing the same `forward(idx, targets)`, `forward_trace`,
`attention_layers`, `dense_layers` interface (so the replay engine / figures work).
Classification & regression add a pooled head and their own tiny tasks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


def _mlp(in_dim, hidden, out_dim, activation="gelu", bias=True, dropout=0.0):
    """in_dim -> hidden... -> out_dim. Empty `hidden` => a single linear (no activation)."""
    dims = [in_dim, *hidden, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            layers.append(ACTS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass
class AttnCfg:
    causal: bool = True
    bias: bool = True
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    wo_hidden: tuple = ()          # () = plain linear W_O; (h,) = non-standard MLP W_O
    wo_activation: str = "gelu"
    n_heads: int | None = None     # per-layer head count; None = use the model-wide default


@dataclass
class FFNCfg:
    hidden: tuple = (128,)
    activation: str = "gelu"
    bias: bool = True
    dropout: float = 0.0


@dataclass
class BuilderConfig:
    vocab_size: int = 11
    block_size: int = 16
    d_model: int = 32
    n_heads: int = 4
    pos_encoding: str = "learned"          # "learned" | "sinusoidal"
    dropout: float = 0.0                    # embedding dropout
    # layer stack: list of ("attn", AttnCfg) | ("ffn", FFNCfg)
    layers: tuple = (("attn", AttnCfg()), ("ffn", FFNCfg()))
    # output head
    head: str = "lm"                        # "lm" | "classify" | "regression"
    pooling: str = "last"                   # for pooled heads: "last"|"mean"|"cls"|"attn"
    n_classes: int = 2
    out_dim: int = 1
    weight_tied: bool = False               # LM head tied to the embedding table
    final_ln: bool = True
    init: str = "normal"                    # weight init: normal|xavier|kaiming|orthogonal|zeros
    init_scale: float = 0.02                # std for "normal" (ignored by the self-scaling schemes)

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Attention (with a configurable W_O)
# ---------------------------------------------------------------------------
class ConfigurableAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, cfg: AttnCfg):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_k, self.causal = n_heads, d_model // n_heads, cfg.causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=cfg.bias)
        self.wo = _mlp(d_model, cfg.wo_hidden, d_model, cfg.wo_activation, cfg.bias)  # W_O (linear or MLP)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))
        self.last_attn = None
        self.last_qkv = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        self.last_qkv = qkv.detach()
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.causal:
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        self.last_attn = w.detach()
        w = self.attn_drop(w)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.wo(out))


class AttnBlock(nn.Module):
    """Pre-LN attention sub-layer with a residual."""
    def __init__(self, d_model, n_heads, block_size, cfg: AttnCfg):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = ConfigurableAttention(d_model, n_heads, block_size, cfg)

    def forward(self, x):
        return x + self.attn(self.ln(x))


class FFNBlock(nn.Module):
    """Pre-LN position-wise feed-forward with configurable hidden/activation/bias."""
    def __init__(self, d_model, cfg: FFNCfg):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        hidden = list(cfg.hidden) or [d_model]
        dims = [d_model, *hidden]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=cfg.bias) for i in range(len(dims) - 1))
        self.act = ACTS[cfg.activation]()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.out_proj = nn.Linear(dims[-1], d_model, bias=cfg.bias)
        self.out_drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.last_hidden = None                  # final hidden activation (back-compat)
        self.last_hiddens = None                 # activation after EACH hidden layer (for the viz)

    def forward(self, x):
        h = self.ln(x)
        hs = []
        for lin in self.hidden_layers:
            h = self.drop(self.act(lin(h)))
            hs.append(h.detach())
        self.last_hiddens = hs
        self.last_hidden = hs[-1] if hs else None
        return x + self.out_drop(self.out_proj(h))


# ---------------------------------------------------------------------------
# Pooling (turn [B,T,d] into [B,d]) for the pooled heads
# ---------------------------------------------------------------------------
class Pooling(nn.Module):
    def __init__(self, kind, d_model):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)   # learned pooling query q*
        self.last_weights = None

    def forward(self, x):                       # x: [B, T, d]
        if self.kind == "last":
            return x[:, -1, :]
        if self.kind == "mean":
            return x.mean(dim=1)
        if self.kind == "cls":
            return x[:, 0, :]                    # first token acts as [CLS]
        if self.kind == "attn":
            scores = (x @ self.query) / math.sqrt(x.shape[-1])      # [B, T]
            w = F.softmax(scores, dim=1)
            self.last_weights = w.detach()
            return (w.unsqueeze(-1) * x).sum(dim=1)
        raise ValueError(f"unknown pooling {self.kind!r}")


# ---------------------------------------------------------------------------
# The configurable model
# ---------------------------------------------------------------------------
class BuilderModel(nn.Module):
    def __init__(self, cfg: BuilderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        else:
            self.register_buffer("pos_sin", _sinusoidal(cfg.block_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

        mods = []
        for kind, lc in cfg.layers:
            if kind == "attn":
                nh = lc.n_heads or cfg.n_heads
                if cfg.d_model % nh != 0:
                    raise ValueError(f"d_model {cfg.d_model} not divisible by attention heads {nh}")
                mods.append(AttnBlock(cfg.d_model, nh, cfg.block_size, lc))
            elif kind == "ffn":
                mods.append(FFNBlock(cfg.d_model, lc))
            else:
                raise ValueError(f"unknown layer {kind!r}")
        self.layers = nn.ModuleList(mods)
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.final_ln else nn.Identity()

        if cfg.head == "lm":
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.weight_tied:
                self.head.weight = self.tok_emb.weight
        elif cfg.head == "classify":
            self.pool = Pooling(cfg.pooling, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        elif cfg.head == "regression":
            self.pool = Pooling(cfg.pooling, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.out_dim)
        else:
            raise ValueError(f"unknown head {cfg.head!r}")
        self.apply(self._init)

    def _init(self, m):
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            return
        w, scheme = m.weight, self.cfg.init
        if scheme == "xavier":
            nn.init.xavier_normal_(w)
        elif scheme == "kaiming":
            nn.init.kaiming_normal_(w, nonlinearity="relu")
        elif scheme == "orthogonal":
            nn.init.orthogonal_(w)
        elif scheme == "zeros":
            nn.init.zeros_(w)
        else:                                            # "normal" (default GPT-style)
            nn.init.normal_(w, 0.0, self.cfg.init_scale)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    # --- embedding ---
    def _embed(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx)
        if self.cfg.pos_encoding == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]
        else:
            x = x + self.pos_sin[:T][None, :, :].to(x.dtype)
        return self.drop(x)

    def forward(self, idx, targets=None):
        x = self._embed(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        if self.cfg.head == "lm":
            out = self.head(x)                                  # [B, T, vocab]
            loss = None
            if targets is not None:
                loss = F.cross_entropy(out.reshape(-1, out.size(-1)), targets.reshape(-1),
                                       ignore_index=-100)
            return out, loss
        pooled = self.pool(x)                                    # [B, d]
        out = self.head(pooled)                                  # [B, classes] or [B, out_dim]
        loss = None
        if targets is not None:
            loss = (F.cross_entropy(out, targets) if self.cfg.head == "classify"
                    else F.mse_loss(out, targets.float()))
        return out, loss

    # --- introspection (matches the replay/figure interface) ---
    def attention_layers(self):
        return [l for l in self.layers if isinstance(l, AttnBlock)]

    def dense_layers(self):
        return [l for l in self.layers if isinstance(l, FFNBlock)]

    @torch.no_grad()
    def forward_trace(self, idx):
        x = self._embed(idx)
        stages = [("embed", x.detach())]
        na = nf = 0
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, AttnBlock):
                na += 1; lbl = f"attn {na}"
            else:
                nf += 1; lbl = f"ffn {nf}"
            stages.append((lbl, x.detach()))
        return stages


def _sinusoidal(T, d):
    pos = torch.arange(T).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def count_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# Tiny tasks for the pooled heads (classification / regression on whole sequences)
# ---------------------------------------------------------------------------
class MajorityTask:
    """Binary classification: is symbol '1' the majority of the sequence?"""
    name = "Majority"
    kind = "classify"

    def __init__(self, length=13, n_symbols=2):
        self.length = length
        self.n_symbols = n_symbols
        self.vocab_size = n_symbols
        self.block_size = length
        self.n_classes = 2
        self.id_to_str = {i: str(i) for i in range(n_symbols)}

    def make_batch(self, batch_size, device="cpu", generator=None):
        x = torch.randint(0, self.n_symbols, (batch_size, self.length), generator=generator)
        label = (x.sum(dim=1) * 2 > self.length).long()         # majority of 1s
        return x.to(device), label.to(device)


class DensityTask:
    """Regression: predict the fraction of '1's in the sequence (a number in [0, 1])."""
    name = "Density"
    kind = "regression"

    def __init__(self, length=12, n_symbols=2):
        self.length = length
        self.n_symbols = n_symbols
        self.vocab_size = n_symbols
        self.block_size = length
        self.out_dim = 1
        self.id_to_str = {i: str(i) for i in range(n_symbols)}

    def make_batch(self, batch_size, device="cpu", generator=None):
        x = torch.randint(0, self.n_symbols, (batch_size, self.length), generator=generator)
        target = x.float().mean(dim=1, keepdim=True)            # [B, 1]
        return x.to(device), target.to(device)


# ---------------------------------------------------------------------------
# A tiny generic train/eval (handles lm / classify / regression) - for demos & tests
# ---------------------------------------------------------------------------
def quick_train(model: BuilderModel, task, steps=300, lr=3e-3, batch=128, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 1)
    losses = []
    for _ in range(steps):
        if getattr(task, "kind", "lm") == "lm":
            x, y, _ = task.make_batch(batch, device, generator=gen)
            _, loss = model(x, y)
        else:
            x, t = task.make_batch(batch, device, generator=gen)
            _, loss = model(x, t)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def quick_eval(model: BuilderModel, task, batch=512, device="cpu", seed=1):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    if task.kind == "classify":
        x, label = task.make_batch(batch, device, generator=gen)
        out, _ = model(x)
        return {"acc": float((out.argmax(-1) == label).float().mean().item())}
    if task.kind == "regression":
        x, target = task.make_batch(batch, device, generator=gen)
        out, _ = model(x)
        mse = float(F.mse_loss(out, target).item())
        var = float(target.var().item()) + 1e-9
        return {"mse": mse, "r2": 1.0 - mse / var}          # R^2: 1.0 = perfect
    raise ValueError("use tiny_gpt.evaluate for lm tasks")


if __name__ == "__main__":
    # classification demo
    task = MajorityTask(length=13)
    cfg = BuilderConfig(vocab_size=task.vocab_size, block_size=task.block_size, d_model=32, n_heads=4,
                        layers=(("attn", AttnCfg()), ("ffn", FFNCfg(hidden=(64,)))),
                        head="classify", pooling="mean", n_classes=2)
    m = BuilderModel(cfg)
    quick_train(m, task, steps=400)
    print("Majority (mean-pool, classify):", quick_eval(m, task))

    # regression demo
    rtask = DensityTask(length=12)
    rcfg = BuilderConfig(vocab_size=rtask.vocab_size, block_size=rtask.block_size, d_model=32, n_heads=4,
                         head="regression", pooling="attn")
    rm = BuilderModel(rcfg)
    quick_train(rm, rtask, steps=400)
    print("Density (attn-pool, regression):", quick_eval(rm, rtask))
