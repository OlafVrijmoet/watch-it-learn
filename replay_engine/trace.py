"""replay_engine.trace — the activation trace, recomputed on demand (JSON-serializable for the viz)."""
from __future__ import annotations

import torch

from builder_model import BuilderModel
from tasks import Task


@torch.no_grad()
def trace_forward(model: BuilderModel, task: Task, seq) -> dict:
    """Run one eval-mode forward on `seq` and return all the values the viz draws:
    the residual stream at each stage, each attention layer's grid + Q/K/V, each FFN
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
        "ffn": [],
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

    for i, dl in enumerate(model.ffn_layers()):
        hiddens = [hh[0].cpu().tolist() for hh in (getattr(dl, "last_hiddens", None) or [dl.last_hidden])]
        out["ffn"].append({"layer": i, "hidden": hiddens[-1], "hiddens": hiddens})

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
