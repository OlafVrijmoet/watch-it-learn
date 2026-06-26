"""
Compare page — overlay or stack the training curves of every trained version (they share the task).

Reads `st.session_state.experiment` (filled on the Builder page); colors mean *versions*, not metrics.
Held-out curves come straight off each run's checkpoints; the train-set counterpart is computed on the
fly (and cached) via `train_eval_curve`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from replay_engine import train_eval_curve, layer_gradients, sample_train_batch, exact_train_batch
from app_charts import _show

VERSION_COLORS = ["#2563eb", "#ef4444", "#7c3aed", "#059669", "#d97706", "#db2777"]


@st.cache_data(show_spinner=False)
def _train_curve(run_sig, _run):                 # cache by the run's signature; _run not hashed
    return train_eval_curve(_run)


@st.cache_data(show_spinner=False)
def _final_grads(run_sig, source, _run):
    """Per-block ‖∇‖ at the final step on the chosen source; aggregates summed across blocks/heads so
    versions with different architectures stay comparable (Q/K/V/O kept separate for the per-component view)."""
    model = _run.reconstruct(_run.cfg.steps)
    if source == "exact training batch":
        batches = [exact_train_batch(_run.cfg, _run.task, _run.cfg.steps - 1)]
    else:                                            # "averaged (3 batches)"
        batches = [sample_train_batch(_run.cfg, _run.task, s) for s in range(3)]
    g = layer_gradients(model, batches)
    return {"embed": g["embed"], "head": g["head"],
            "Q": sum(a["q"] for a in g["attn"]), "K": sum(a["k"] for a in g["attn"]),
            "V": sum(a["v"] for a in g["attn"]), "O": sum(a["o"] for a in g["attn"]),
            "attention": sum(a["all"] for a in g["attn"]), "FFN": sum(f["all"] for f in g["ffn"])}


def _held(run):
    return run.checkpoint_steps, [c.eval_loss for c in run.checkpoints], [c.acc for c in run.checkpoints]


st.title("📊 Compare runs")

exp = st.session_state.get("experiment")
versions = exp["versions"] if exp else []
trained = [v for v in versions if v.get("run") is not None]
if not trained:
    st.info("No trained versions yet — go to the **Builder** page, add versions, and **▶ Train all untrained**.")
    st.stop()

task_name = st.session_state.get("task", "?")
st.caption(f"shared task: **{task_name}**  ·  colors mean **versions**, not metrics")
if any(v.get("stale") for v in trained):
    st.warning("⚠ Some versions are **stale** — their config or the shared task changed since they were "
               "trained, so their curves below are from the *old* settings. Re-train them on the Builder "
               "(**↻ Retrain all**) for an apples-to-apples comparison.")

_color = {v["id"]: VERSION_COLORS[i % len(VERSION_COLORS)] for i, v in enumerate(versions)}
_name = {v["id"]: v["name"] for v in trained}

shown_ids = st.multiselect("versions", [v["id"] for v in trained], default=[v["id"] for v in trained],
                           format_func=lambda i: _name[i], key="cmp_versions")
c1, c2 = st.columns(2)
view = c1.radio("view", ["Overlay", "Small-multiples"], horizontal=True, key="cmp_view")
curves = c2.radio("curves", ["Held-out", "Train", "Both"], horizontal=True, key="cmp_curves")
shown = [v for v in trained if v["id"] in shown_ids]
if not shown:
    st.info("Select at least one version to compare.")
    st.stop()


def _plot(ax, vs, which, title):                  # which = "loss" | "acc"
    for v in vs:
        run = v["run"]
        x, hl, ha = _held(run)
        col = _color[v["id"]]
        if curves in ("Held-out", "Both"):
            ax.plot(x, hl if which == "loss" else ha, color=col, label=v["name"])
        if curves in ("Train", "Both"):
            tl, ta = _train_curve(v.get("run_sig") or v["name"], run)
            ax.plot(x, tl if which == "loss" else ta, color=col, linestyle="--", alpha=0.65,
                    label=f"{v['name']} (train)" if curves == "Both" else v["name"])
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("step", fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)


def _chart(vs, which, title, h=3.0):
    fig, ax = plt.subplots(figsize=(5, h))
    _plot(ax, vs, which, title)
    fig.tight_layout()
    _show(st, fig)


if view == "Overlay":
    g1, g2 = st.columns(2)
    with g1:
        _chart(shown, "loss", "loss · held-out" if curves == "Held-out" else "loss")
    with g2:
        _chart(shown, "acc", "accuracy · held-out" if curves == "Held-out" else "accuracy")
else:                                             # small-multiples: one row per version
    for v in shown:
        st.markdown(f"**{v['name']}**")
        g1, g2 = st.columns(2)
        with g1:
            _chart([v], "loss", "loss", h=2.4)
        with g2:
            _chart([v], "acc", "accuracy", h=2.4)

# ---- gradient flow — compare ----
st.divider()
st.markdown("🔥 **gradient flow — compare**")
st.caption("how the learning signal ‖∇‖ decays over training, and which blocks carry it · "
           "Q/K often start near zero and grow")
gc1, gc2, gc3 = st.columns([2, 1, 1])
grad_src = gc1.selectbox("gradient from", ["averaged (3 batches)", "exact training batch"], key="cmp_gradsrc")
per_comp = gc2.checkbox("per Q/K/V/O", key="cmp_gradhead",
                        help="split attention into Q/K/V/O (summed across heads & blocks)")
log_g = gc3.checkbox("log scale", value=True, key="cmp_gradlog")

gg1, gg2 = st.columns(2)
with gg1:                                         # total ‖∇‖ over training (per version)
    fig, ax = plt.subplots(figsize=(5, 3))
    for v in shown:
        ax.plot(v["run"].metrics["step"], v["run"].metrics["grad_norm"],
                color=_color[v["id"]], label=v["name"])
    ax.set_title("total ‖∇‖ over training", fontsize=10, loc="left")
    ax.set_xlabel("step", fontsize=8); ax.grid(alpha=0.25)
    if log_g:
        ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); _show(st, fig)

with gg2:                                         # ‖∇‖ per block, final step (grouped by version)
    cats = ["embed", "Q", "K", "V", "O", "FFN", "head"] if per_comp else ["embed", "attention", "FFN", "head"]
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(cats)); bh = 0.8 / max(1, len(shown))
    for j, v in enumerate(shown):
        g = _final_grads(v.get("run_sig") or v["name"], grad_src, v["run"])
        vals = [max(g[c], 1e-9) if log_g else g[c] for c in cats]
        ax.barh(y + (j - (len(shown) - 1) / 2) * bh, vals, height=bh,
                color=_color[v["id"]], label=v["name"])
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=8); ax.invert_yaxis()
    ax.set_title("‖∇‖ per block · final step", fontsize=10, loc="left")
    ax.set_xlabel("‖∇‖", fontsize=8); ax.grid(alpha=0.25, axis="x")
    if log_g:
        ax.set_xscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); _show(st, fig)

st.divider()
rows = []
for v in shown:
    run = v["run"]
    rows.append({"version": v["name"], "status": "⚠ stale" if v.get("stale") else "✓ trained",
                 "final loss": round(run.checkpoints[-1].eval_loss, 4),
                 "final acc": round(run.checkpoints[-1].acc, 3),
                 "params": f"{run.n_params / 1000:.1f}k",
                 "memory": f"{run.nbytes() / 1e6:.1f} MB",
                 "steps": run.cfg.steps})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("memory = checkpoint weights held in RAM (`TrainingRun.nbytes`) — lower **# checkpoints** to shrink.")
