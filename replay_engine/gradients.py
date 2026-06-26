"""replay_engine.gradients — gradient introspection, computed ON THE FLY (never stored).

One backward pass on a reconstructed model gives the per-block gradient norms aligned with the flow
bands. Storing per-layer grads would be model-size x steps, so they're always recomputed on demand.
"""
from __future__ import annotations

import torch

from builder_model import AttnBlock, FFNBlock
from .config import GRAD_SCALE_SEED_BASE
from .splits import sample_train_batch


def param_grad_norm(params) -> float:
    """L2 norm of the gradients across a group of parameters (skips params whose grad is None)."""
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().pow(2).sum().item())
    return s ** 0.5


def tensor_norm(t) -> float:
    """L2 norm of a single tensor (e.g. a slice of a gradient)."""
    return float(t.pow(2).sum().item()) ** 0.5


def _grad_norm(model) -> float:
    return param_grad_norm(model.parameters())


def _grad_norms(model) -> dict:
    """Per-block L2 gradient norms aligned with the flow bands; call AFTER loss.backward()."""
    dm = model.cfg.d_model
    gnorm, nrm = param_grad_norm, tensor_norm    # reuse the module-level helpers (no duplicated bodies)

    res = {"attn": [], "ffn": []}
    emb = [model.tok_emb.weight]
    if getattr(model.cfg, "pos_encoding", "learned") == "learned":
        emb.append(model.pos_emb.weight)
    res["embed"] = gnorm(emb)
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


def gradient_scale(run, task) -> float:
    """Largest per-block gradient magnitude over the WHOLE run (sampled checkpoints, one fresh train
    batch each) — used to PIN the gradient-bar axis so magnitudes are comparable across training steps."""
    steps = run.checkpoint_steps
    if len(steps) > 14:                                        # sample to keep it cheap
        steps = sorted({steps[round(k * (len(steps) - 1) / 13)] for k in range(14)})
    vals = [1e-9]
    for i, stp in enumerate(steps):
        g = layer_gradients(run.reconstruct(stp), [sample_train_batch(run.cfg, task, GRAD_SCALE_SEED_BASE + i)])
        vals += [g["embed"], g["head"]]
        for a in g["attn"]:
            vals += [a["q"], a["k"], a["v"], a["o"]]
            vals += [h[p] for h in a["heads"] for p in ("q", "k", "v", "o")]
        for f in g["ffn"]:
            vals += [f["all"], f["out"], *f["layers"]]
    return max(vals)
