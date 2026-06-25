"""
builder_app.py — the model BUILDER: the expanded model is the canvas.

You see the whole model rendered from the start (even untrained), build it by adding
sections (➕ between sections) with per-section settings, then TRAIN and scrub the
training history — the same view fills in with the real values at each step.

Constraint: a Streamlit st.iframe (srcdoc) is one-way (no npm custom component),
so the ➕/settings are Streamlit widgets in a section spine beside the render, and the SVG
render reflects the current architecture. This is one of two pages — run the whole app with
`streamlit run app.py` (Builder + Compare).
"""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from replay_engine import (RunConfig, TrainingRun, build_model, build_task, trace_forward,
                           generate_sampled, ALL_TASKS, is_heldout,
                           layer_gradients, sample_train_batch, exact_train_batch, train_eval_curve,
                           continue_training, gradient_scale)
from flow_svg import flow_svg_component, model_svg, svg_document, replay_html
from flow_component import flow_html, component_height
from training_utils import divisor_heads, suggest_lr_transformer

def _show(target, fig):
    """Render a matplotlib figure into `target` (st or a column) then close it. st.pyplot does NOT close
    it, so without this every scrub/toggle leaks a figure into matplotlib's global registry."""
    target.pyplot(fig)
    plt.close(fig)


def _metric_chart(xs, ys, cur, color, title, ys2=None, color2="#9ca3af"):
    """A small held-out metric curve with a dashed marker at the current training step. If `ys2`
    (the train-set series) is given, it's overlaid (dashed) with a legend so you see the gap."""
    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.plot(xs, ys, color=color, lw=1.6, label="held-out")
    if ys2 is not None:
        ax.plot(xs, ys2, color=color2, lw=1.4, ls="--", label="train")
        ax.legend(fontsize=6, loc="best", frameon=False)
    ax.axvline(cur, color="#374151", lw=1, ls="--")
    yc = ys[min(range(len(xs)), key=lambda i: abs(xs[i] - cur))]
    ax.scatter([cur], [yc], color=color, s=20, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig


def _grad_bars(grads, arch, log=False, per_head=False, xmax=None):
    """Horizontal bars of gradient magnitude per block, in flow order (embed → blocks → head).
    `per_head` expands each attention's Q/K/V/O into one group per head; `log` uses a log x-axis;
    `xmax` pins the axis to a fixed value (run-wide max) so steps are comparable."""
    labels, vals, colors = ["embed"], [grads["embed"]], ["#64748b"]
    ai = fi = 0
    for b in arch:
        if b["type"] == "attn" and ai < len(grads["attn"]):
            g = grads["attn"][ai]; ai += 1
            if per_head and g.get("heads"):
                for h, hg in enumerate(g["heads"]):
                    for part in ("q", "k", "v", "o"):
                        labels.append(f"A{ai}H{h}·{part.upper()}"); vals.append(hg[part]); colors.append("#2563eb")
            else:
                for part in ("q", "k", "v", "o"):
                    labels.append(f"A{ai}·{part.upper()}"); vals.append(g[part]); colors.append("#2563eb")
        elif b["type"] == "ffn" and fi < len(grads["ffn"]):
            f = grads["ffn"][fi]; fi += 1
            if per_head:                                      # break FFN into hidden layers + down-proj
                for l, ln in enumerate(f["layers"]):
                    labels.append(f"F{fi}·h{l}"); vals.append(ln); colors.append("#16a34a")
                labels.append(f"F{fi}·out"); vals.append(f["out"]); colors.append("#0d9488")
            else:
                labels.append(f"FFN{fi}"); vals.append(f["all"]); colors.append("#16a34a")
    labels.append("head"); vals.append(grads["head"]); colors.append("#64748b")
    fig, ax = plt.subplots(figsize=(3.6, max(1.8, 0.22 * len(labels))))
    top = xmax if xmax else max(vals + [1e-12])               # fixed (run-wide) or per-step max
    if log:
        floor = top / 1e4                                      # show ~0 bars as a tiny stub
        ax.barh(range(len(labels)), [max(v, floor) for v in vals], color=colors)
        ax.set_xscale("log"); ax.set_xlim(floor, top)
    else:
        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_xlim(0, top)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)
    ax.set_title("‖gradient‖ per block" + (" (log)" if log else ""), fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig


st.title("◆ Tiny Model Builder")
st.caption("Build the model by adding sections — it renders live from the start; then train and scrub the history.")

# ---------------------------------------------------------------------------
# experiment state — multiple model "versions" sharing one task, compared side by side.
# Each version owns its architecture, config, and trained run. The active version's config lives
# in the (global) widget keys; switching versions loads that version's saved config into them and
# saving happens each run — so each version keeps its own settings without per-widget namespacing.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "d_model": 32, "n_heads": 4, "ffn_mult": 4, "def_dropout": 0.0, "def_act": "gelu",
    "def_bias": True, "init_scheme": "normal", "init_scale": 0.02, "seed": 0,
    "steps": 400, "batch": 128, "optimizer": "AdamW",
    "lr": round(suggest_lr_transformer(32, 1, 128, "AdamW"), 5), "lr_sched": "constant",
    "nck": 60, "device": "cpu", "pos_encoding": "learned", "pool_sel": "mean",
    "samp": "greedy", "temp": 1.0, "topk": 3, "topp": 0.9,
}
# The TRAINING PROTOCOL is SHARED by every version (so "Train all" trains them the same way — you set
# steps/lr/etc. once, not per model). Each version owns only its architecture / init / output settings.
TRAIN_KEYS = ["steps", "batch", "optimizer", "lr", "lr_sched", "nck", "device"]
CONFIG_KEYS = [k for k in DEFAULTS if k not in TRAIN_KEYS]    # per-version (saved/loaded per version)

if "experiment" not in st.session_state:
    st.session_state.experiment = {
        "versions": [{"id": 0, "name": "v1", "note": "", "config": {},
                      "arch": [{"type": "attn", "_id": 0}, {"type": "ffn", "_id": 1}],
                      "run": None, "run_sig": None}],
        "active": 0, "next_vid": 1, "next_block_id": 2,
        "shared": {},                                         # shared widget state: task + params + training
    }
exp = st.session_state.experiment
# Restore SHARED widget state (the task + its params + the training protocol) when it was dropped — Streamlit
# discards unrendered widgets' session_state when you visit another page (Compare) or on a mid-sidebar rerun.
# Only when missing, so live edits are preserved (saved back at the end of each run). Per-version config is
# restored separately from each version's saved config (load-on-switch below).
for _k, _v in (exp.get("shared") or {}).items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
ver = next(v for v in exp["versions"] if v["id"] == exp["active"])   # the active version
arch = ver["arch"]
for _b in arch:                                              # stable ids -> stable section widget keys
    if "_id" not in _b:
        _b["_id"] = exp["next_block_id"]
        exp["next_block_id"] += 1


def version_summary(v):                                      # steps/lr are shared now, so not shown per version
    c = v.get("config") or {}
    return (f"{c.get('n_heads', DEFAULTS['n_heads'])}h · d{c.get('d_model', DEFAULTS['d_model'])} · "
            f"FFN×{c.get('ffn_mult', DEFAULTS['ffn_mult'])} · {len(v['arch'])} blk")


def create_version(name, source_id=None):
    e = st.session_state.experiment
    nid = e["next_vid"]; e["next_vid"] += 1
    if source_id is not None:                                # Duplicate / Add-from-copy: clone config + arch
        src = next(v for v in e["versions"] if v["id"] == source_id)
        new_arch = deepcopy(src["arch"])
        for b in new_arch:
            b["_id"] = e["next_block_id"]; e["next_block_id"] += 1
        config = dict(src.get("config") or DEFAULTS)
    else:                                                    # Add blank
        new_arch = [{"type": "attn", "_id": e["next_block_id"]},
                    {"type": "ffn", "_id": e["next_block_id"] + 1}]
        e["next_block_id"] += 2
        config = dict(DEFAULTS)
    e["versions"].append({"id": nid, "name": name, "note": "", "config": config,
                          "arch": new_arch, "run": None, "run_sig": None})
    e["active"] = nid                                        # the load-on-switch (below) applies its config


def delete_active_version():
    e = st.session_state.experiment
    if len(e["versions"]) <= 1:
        return
    e["versions"] = [v for v in e["versions"] if v["id"] != e["active"]]
    e["active"] = e["versions"][0]["id"]


# --- load the active version's config into the widget keys: fully on a version switch, and otherwise
# restore any field whose widget state was dropped (e.g. after visiting the Compare page and back) — so a
# freshly trained run doesn't read as "stale" just because navigating away reset the Builder's widgets. ---
_switching = st.session_state.get("_loaded_vid", ver["id"]) != ver["id"]
for base in CONFIG_KEYS:
    if base in (ver.get("config") or {}) and (_switching or base not in st.session_state):
        st.session_state[base] = ver["config"][base]
st.session_state["_loaded_vid"] = ver["id"]

# ---------------------------------------------------------------------------
# model-wide defaults (sidebar)
# ---------------------------------------------------------------------------
# 3 tabs, ALWAYS rendered (so every config widget stays mounted every run → the trained run's
# signature never changes out from under it). Build/Train hold the config; Run (filled later, once
# cfg/trained are known) holds the post-training views.
with st.sidebar:
    focus = st.toggle("🖥 full-width model (fold the editor)", key="focus",
                      help="hide the section editor so the model canvas gets the full width")
    tab_build, tab_train, run_tab = st.tabs(["🛠 Build", "🎯 Train", "🔎 Run"])
    with tab_build:
        st.caption("dataset/task is **shared** by every version · model settings below are per-version")
        task_name = st.selectbox("dataset / task  ◆ shared", list(ALL_TASKS), index=1, key="task")
        task_cls = ALL_TASKS[task_name]
        task_kwargs = {}
        for kw, spec in getattr(task_cls, "params", {}).items():
            if isinstance(spec, dict) and spec.get("type") == "multiselect":   # categorical setting
                sel = st.multiselect(kw, spec["options"], default=spec["default"], key=f"tk_{kw}")
                task_kwargs[kw] = sorted(sel) or [spec["options"][0]]           # stable order, never empty
            else:
                lo, hi, default = spec
                task_kwargs[kw] = st.slider(kw, int(lo), int(hi), int(default), key=f"tk_{kw}")
        head_kind = getattr(task_cls, "kind", "lm")          # pooling/sampling live in the Output card

        # ---- Versions panel (select / add / duplicate / rename / delete) ----
        st.divider()
        st.markdown(f"**Versions** · {len(exp['versions'])} in experiment")
        for v in exp["versions"]:
            is_active = v["id"] == exp["active"]
            badge = "⚠ stale" if v.get("stale") else ("✓" if v["run"] is not None else "○")
            label = f"{'🔵' if is_active else '⚪'} {v['name']} · {version_summary(v)}  {badge}"
            if st.button(label, key=f"selv{v['id']}", width="stretch",
                         type=("primary" if is_active else "secondary")):
                if not is_active:
                    exp["active"] = v["id"]; st.rerun()
        if ver.get("note"):
            st.caption(f"↳ {ver['note']}")
        _a = st.columns(4)
        with _a[0].popover("➕ Add", width="stretch"):
            _nm = st.text_input("name", value=f"v{len(exp['versions']) + 1}", key="addname")
            _from = st.radio("start from", ["Default (blank)", f"Copy of {ver['name']}"], key="addfrom")
            if st.button("Create", key="addgo", type="primary"):
                create_version(_nm or f"v{exp['next_vid']}",
                               source_id=ver["id"] if _from.startswith("Copy") else None)
                st.rerun()
        if _a[1].button("⧉ Duplicate", key="dupv", width="stretch",
                        help="clone this version (untrained)"):
            create_version(f"{ver['name']} copy", source_id=ver["id"]); st.rerun()
        with _a[2].popover("✎", width="stretch"):
            _rn = st.text_input("name", value=ver["name"], key="rnname")
            _rnote = st.text_input("note (optional)", value=ver.get("note", ""), key="rnnote")
            if st.button("Save", key="rngo", type="primary"):
                ver["name"], ver["note"] = _rn or ver["name"], _rnote; st.rerun()
        if _a[3].button("🗑", width="stretch", disabled=len(exp["versions"]) <= 1,
                        help="delete this version"):
            delete_active_version(); st.rerun()
        st.divider()
        st.markdown(f"**Model — {ver['name']}**  ·  this version's architecture & init (training is **shared** — Train tab)")

        d_model = st.select_slider("d_model", options=[16, 24, 32, 48, 64], value=32, key="d_model")
        _hopts = divisor_heads(d_model)
        n_heads = st.select_slider("default # heads", options=_hopts,
                                   value=(4 if 4 in _hopts else _hopts[-1]), key="n_heads")
        ffn_mult = st.select_slider("FFN width (× d_model)", options=[1, 2, 4], value=4, key="ffn_mult")
        st.caption("defaults — inherited by every section unless it overrides:")
        dropout = st.slider("default dropout", 0.0, 0.5, 0.0, 0.05, key="def_dropout")
        default_activation = st.selectbox("default activation", ["gelu", "relu", "silu", "tanh"], key="def_act")
        default_bias = st.checkbox("default bias", True, key="def_bias")
        st.divider()
        init_scheme = st.selectbox("weight init", ["normal", "xavier", "kaiming", "orthogonal", "zeros"],
                                   key="init_scheme", help="how weights start — watch its effect in Run › gradients")
        init_scale = (st.slider("init std", 0.005, 0.5, 0.02, 0.005, key="init_scale")
                      if init_scheme == "normal" else 0.02)
        seed = int(st.number_input("random seed", 0, 9999, 0, key="seed"))
    with tab_train:
        st.caption("⚙ training protocol — **shared by every version** (Train all / Retrain all use these)")
        steps = st.slider("steps", 50, 1500, 400, 50, key="steps")
        batch = st.select_slider("batch size", options=[32, 64, 128, 256], value=128, key="batch")
        optimizer = st.selectbox("optimizer", ["AdamW", "Adam", "SGD (momentum)", "RMSprop"], key="optimizer")
        lr = st.number_input("learning rate",
                             value=float(f"{suggest_lr_transformer(d_model, 1, batch, optimizer):.5f}"),
                             format="%.5f", step=1e-4, key="lr")
        lr_schedule = st.selectbox("LR schedule", ["constant", "cosine", "warmup + cosine"], key="lr_sched")
        n_checkpoints = st.slider("# checkpoints (fidelity)", 10, 200, 60, 10, key="nck")
        device = st.radio("device", ["cpu", "gpu"], horizontal=True, key="device")
        dev = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available()
               else "cpu") if device == "gpu" else "cpu"
        st.divider()
        _has_run = ver["run"] is not None
        train_clicked = st.button("↻ Restart training" if _has_run else "▶ Start training",
                                  type="primary", width="stretch", key="train")
        if _has_run:
            extra = int(st.number_input("continue: extra steps", 50, 2000, 200, 50, key="extra_steps"))
            continue_clicked = st.button(f"➕ Continue training (+{extra})", width="stretch", key="continue")
        else:
            extra, continue_clicked = 0, False

    # ---- Train-all / Retrain-all / Continue-all (sidebar bottom, always visible below the tabs) ----
    st.divider()
    _untrained_n = sum(1 for v in exp["versions"] if v["run"] is None)
    _trained_n = sum(1 for v in exp["versions"] if v["run"] is not None)
    trainall_clicked = st.button(f"▶ Train all untrained  ({_untrained_n})", key="trainall",
                                 type="primary", width="stretch", disabled=_untrained_n == 0)
    retrainall_clicked = st.button(f"↻ Retrain all  ({len(exp['versions'])})", key="retrainall",
                                   width="stretch", help="force-retrain every version (e.g. after a task change)")
    continueall_extra = int(st.number_input("continue: +steps for all", 50, 2000, 200, 50, key="continueall_extra"))
    continueall_clicked = st.button(f"➕ Continue all  (+{continueall_extra} × {_trained_n})", key="continueall",
                                    width="stretch", disabled=_trained_n == 0,
                                    help="warm-start every trained version and append more steps")
    exec_mode = st.radio("execution", ["Sequential", "Parallel (background)"],
                         horizontal=True, key="execmode")
    if exec_mode.startswith("Parallel"):
        st.caption("⚠ parallel is experimental — runs sequentially for now")
    st.radio("live charts", ["Overlay", "Stacked"], horizontal=True, key="trainall_view",
             help="Overlay = every version in one loss/accuracy chart; Stacked = a chart per version")
    st.caption("**Train all untrained** fills versions with no run · **Retrain all** re-trains every version "
               "— all use the **shared** Train-tab settings (same steps/lr for every model)")

# ---------------------------------------------------------------------------
# config derivation for the active version
# ---------------------------------------------------------------------------
def _active_config():
    """The active version's config as a dict: the live widget values (with DEFAULTS as fallback)."""
    return {base: st.session_state.get(base, DEFAULTS[base]) for base in CONFIG_KEYS}


def version_layer_specs(arch_, c):
    """layer_specs for ANY version (the single source of truth): per-section settings persist by their
    unique _id; defaults come from `c`. Reads live widget values, so a setting change shows immediately
    (regardless of widget vs render order)."""
    ss = st.session_state
    dm, nh_def, ffn = c["d_model"], c["n_heads"], c["ffn_mult"]
    act_def, bias_def, drop = c["def_act"], c["def_bias"], c["def_dropout"]
    specs = []
    for b in arch_:
        i = b["_id"]
        ovr = ss.get(f"ov{i}", b.get("override", False))     # does this section override the defaults?
        if b["type"] == "attn":
            nh = ss.get(f"nh{i}", b.get("n_heads", nh_def))
            if dm % nh != 0:
                nh = nh_def
            wo_on = ss.get(f"wo{i}", bool(b.get("wo_hidden")))
            wo_hidden = (int(ss.get(f"wou{i}", b.get("wo_units", dm))),) if wo_on else ()
            specs.append(("attn", {"n_heads": nh, "causal": ss.get(f"ca{i}", b.get("causal", True)),
                                   "bias": ss.get(f"ba{i}", b.get("bias", bias_def)) if ovr else bias_def,
                                   "attn_dropout": ss.get(f"ad{i}", b.get("attn_dropout", drop)) if ovr else drop,
                                   "wo_hidden": wo_hidden,
                                   "wo_activation": ss.get(f"woa{i}", b.get("wo_act", "gelu"))}))
        else:
            units = int(ss.get(f"hu{i}", b.get("hidden", ffn * dm)))
            nl = int(ss.get(f"nl{i}", b.get("n_layers", 1)))
            specs.append(("dense", {"hidden": tuple([units] * nl),
                                    "activation": ss.get(f"ac{i}", b.get("activation", act_def)) if ovr else act_def,
                                    "bias": ss.get(f"fb{i}", b.get("bias", bias_def)) if ovr else bias_def,
                                    "dropout": ss.get(f"fd{i}", b.get("dropout", drop)) if ovr else drop}))
    return tuple(specs)


def _build_cfg(c, arch_):
    """A RunConfig from a per-version config dict `c` (architecture / init / output) + its arch, plus the
    SHARED task and training protocol (steps/batch/lr/optimizer/schedule/checkpoints/device) from the Train tab."""
    pool = c["pool_sel"] if head_kind in ("classify", "regression") else "last"
    return RunConfig(task_name=task_name, task_kwargs=task_kwargs,
                     layer_specs=version_layer_specs(arch_, c),
                     d_model=c["d_model"], n_heads=c["n_heads"], dropout=c["def_dropout"],
                     steps=steps, batch=batch, lr=lr, optimizer=optimizer,
                     lr_schedule=lr_schedule, head=head_kind, pooling=pool,
                     pos_encoding=c["pos_encoding"], init=c["init_scheme"], init_scale=c["init_scale"],
                     seed=c["seed"], n_checkpoints=n_checkpoints, device=dev)


def make_cfg():                                              # the active version's RunConfig (live widgets)
    return _build_cfg(_active_config(), arch)


def version_cfg(v):                                         # any version's RunConfig (its saved config + arch)
    return _build_cfg({**DEFAULTS, **(v.get("config") or {})}, v["arch"])


cfg = make_cfg()
task = build_task(cfg)
sig = cfg.to_json()                                          # detect when a trained run is stale


def version_sig(v):
    return version_cfg(v).to_json()


# Mark each version's run STALE when its current config/task no longer matches what it trained on
# (active version compares against the live sig; others against their saved config). The Versions
# panel + the Compare page read v["stale"]. Changing the shared task makes every trained version stale.
for _v in exp["versions"]:
    _cur = sig if _v["id"] == exp["active"] else version_sig(_v)
    _v["stale"] = _v["run"] is not None and _v["run_sig"] != _cur


def _active_version():                                       # read fresh from session_state (safe in callbacks)
    e = st.session_state.experiment
    return e, next(v for v in e["versions"] if v["id"] == e["active"])


def insert_block(idx, kind):                                 # on_click callback; Streamlit auto-reruns
    e, v = _active_version()
    v["arch"].insert(idx, {"type": kind, "_id": e["next_block_id"]})
    e["next_block_id"] += 1


def remove_block(idx):
    e, v = _active_version()
    if len(v["arch"]) > 1:
        v["arch"].pop(idx)


# ---------------------------------------------------------------------------
# derive cfg-dependent state, then fill the sidebar "Run" tab
# ---------------------------------------------------------------------------
run = ver["run"]
trained = run is not None and ver["run_sig"] == sig

full_flow = head_kind == "lm" and any(b["type"] == "attn" for b in arch) and any(b["type"] == "ffn" for b in arch)
n_in = task.prompt_len if head_kind == "lm" else task.block_size
opt_strs = [task.id_to_str.get(v, str(v)) for v in range(task.vocab_size)]
# Sanitize: drop any stored probe field that isn't valid for the current task's vocab (e.g. a leftover '+'
# from a previous task) so neither opt_strs.index() nor the selectbox ever sees an out-of-vocab token.
for p in range(n_in):
    if st.session_state.get(f"pf{p}") not in opt_strs:
        st.session_state.pop(f"pf{p}", None)
# Seed defaults from an example on a task change or for any field still missing (incl. ones just dropped).
if st.session_state.get("probe_task") != task_name or any(f"pf{p}" not in st.session_state for p in range(n_in)):
    if head_kind == "lm":
        _, _, _seq = task.make_batch(1, generator=torch.Generator().manual_seed(3))
        dft = _seq[0][:n_in].tolist()
    else:
        _x, _ = task.make_batch(1, generator=torch.Generator().manual_seed(3))
        dft = _x[0].tolist()
    for p in range(n_in):
        st.session_state.setdefault(f"pf{p}", task.id_to_str.get(dft[p], str(dft[p])))
    st.session_state["probe_task"] = task_name
steps_all = run.checkpoint_steps if trained else None

# folded defaults (used before training / when a control isn't shown)
chosen = [opt_strs.index(st.session_state[f"pf{p}"]) if st.session_state.get(f"pf{p}") in opt_strs else 0
          for p in range(n_in)]
k_ro = int(st.session_state.get("rpos", 1))
step = (steps_all[-1] if trained else None)                  # Run tab (below) sets the real snapped step
model = grads = None
show_grads = False

with run_tab:                                                 # always-visible while the canvas scrolls
    if run is None:
        st.info("No training has taken place yet — set it up in **Build** / **Train**, then ▶ Start training.")
    elif not trained:
        st.warning("Settings changed since training — **↻ Restart training** (Train tab) to inspect the new model.")
    else:
        st.markdown("**Probe input** — edit any field; the model runs *your* input")
        per_row = min(n_in, 3)                                # 3/row → wide enough to read the value
        for r0 in range(0, n_in, per_row):
            cols = st.columns(per_row)
            for j, p in enumerate(range(r0, min(r0 + per_row, n_in))):
                chosen[p] = opt_strs.index(cols[j].selectbox(f"#{p}", opt_strs, key=f"pf{p}",
                                                             label_visibility="collapsed"))
        st.caption("reads as:  `" + " ".join(opt_strs[c] for c in chosen) + "`")   # full input, always legible
        st.caption("🟡 **held-out** — never trained on this exact input"
                   if is_heldout(task, torch.tensor([chosen])) else "🟢 this input was in the training set")
        if full_flow and task.gen_len > 1:
            k_ro = st.select_slider("visualize output token #", options=list(range(1, task.gen_len + 1)),
                                    value=1, key="rpos")
        st.markdown("**Training stage** — the input re-runs at *this* stage")
        _smax = int(steps_all[-1])                            # LINEAR slider over the step range
        if int(st.session_state.get("stepslider", 0)) > _smax:   # clamp a stale value after retrain
            st.session_state["stepslider"] = _smax
        step_raw = st.slider("training step (model version)", 0, _smax, _smax, key="stepslider")
        ckp = run.nearest_checkpoint(step_raw)               # snap to the nearest stored checkpoint
        step = ckp.step
        show_train = st.checkbox("also plot the training set (vs held-out)", key="showtrain")
        tr_l = tr_a = None
        if show_train:
            if st.session_state.get("traincurve_sig") != sig:
                with st.spinner("evaluating train-set curves…"):
                    st.session_state.traincurve = train_eval_curve(run)
                    st.session_state.traincurve_sig = sig
            tr_l, tr_a = st.session_state.traincurve
        xs = [c.step for c in run.checkpoints]
        ci = run.checkpoint_steps.index(ckp.step)
        _show(st, _metric_chart(xs, [c.eval_loss for c in run.checkpoints], step, "#dc2626",
                                f"loss · held {ckp.eval_loss:.3f}" + (f" · train {tr_l[ci]:.3f}" if tr_l else ""),
                                ys2=tr_l, color2="#f59e0b"))
        _show(st, _metric_chart(xs, [c.acc for c in run.checkpoints], step, "#16a34a",
                                f"acc · held {ckp.acc:.2f}" + (f" · train {tr_a[ci]:.2f}" if tr_a else ""),
                                ys2=tr_a, color2="#f59e0b"))
        model = run.reconstruct(step)                         # the model AT the chosen stage
        in_ids_now = torch.tensor([chosen])
        show_grads = st.toggle("🔥 show gradients (bars + ∇ overlay in the canvas)", key="showgrad")
        if show_grads:
            src = st.radio("gradient from", ["averaged (3 batches)", "scrub a batch",
                           "exact training batch", "your probe input"], key="gradsrc")
            gmodel = model
            if src == "scrub a batch":
                bidx = st.slider("batch #", 0, 49, 0, key="gradbatch")
                batches = [sample_train_batch(cfg, task, 70000 + bidx)]
            elif src == "exact training batch":
                batches = [exact_train_batch(cfg, task, step)]
                gmodel = run.reconstruct_exact(step)
            elif src == "your probe input":
                final = run.reconstruct(steps_all[-1])
                if head_kind == "lm":
                    o = generate_sampled(final, task, in_ids_now, method="greedy", device="cpu")[0]
                    seq = torch.cat([in_ids_now[0], o])
                    batches = [(seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0))]
                else:
                    with torch.no_grad():
                        po, _ = final(in_ids_now)
                    batches = [(in_ids_now, po.argmax(-1) if head_kind == "classify" else po)]
            else:
                batches = [sample_train_batch(cfg, task, 80000 + i) for i in range(3)]
            grads = layer_gradients(gmodel, batches)

in_ids = torch.tensor([chosen])
readout_pos = (task.prompt_len - 1 + (k_ro - 1)) if (full_flow and task.gen_len > 1) \
    else ((task.prompt_len - 1) if full_flow else None)


def make_probe(m):                                            # LM: prompt + the model's own generation
    if head_kind == "lm":
        o = generate_sampled(m, task, in_ids, method="greedy", device="cpu")[0]
        return torch.cat([in_ids[0], o])
    return in_ids[0]


rsig = f"{sig}|rp{readout_pos}|in{chosen}"                    # cache key includes readout + the input

# ---- live training: prominent at the TOP, and we skip the untrained re-render while it runs ----
if train_clicked:
    st.subheader("⏳ Training — loss & held-out accuracy (live)")
    g1, g2 = st.columns(2)
    g1.caption("loss (held-out)")
    g2.caption("accuracy (held-out)")
    lph, aph = g1.empty(), g2.empty()
    bar = st.progress(0.0, text="starting…")
    hist = {"step": [], "loss": [], "acc": []}
    _every = max(1, cfg.steps // 100)                         # throttle bar updates to ~100

    def on_step(s, metrics):                                  # bar tracks the ACTUAL step fraction
        if s % _every == 0 or s == cfg.steps - 1:
            bar.progress(min(1.0, (s + 1) / cfg.steps), text=f"step {s + 1}/{cfg.steps}")

    def on_ck(ck, all_):                                      # checkpoints (log-spaced) drive the charts
        hist["step"].append(ck.step)
        hist["loss"].append(round(ck.eval_loss, 4))
        hist["acc"].append(round(ck.acc, 4))
        lph.line_chart(pd.DataFrame({"loss": hist["loss"]}, index=hist["step"]))
        aph.line_chart(pd.DataFrame({"accuracy": hist["acc"]}, index=hist["step"]))

    ver["run"] = TrainingRun.train(cfg, on_step=on_step, on_checkpoint=on_ck)
    ver["run_sig"] = sig
    st.rerun()

# ---- train a batch of versions sequentially, with LIVE loss + accuracy for every run ----
#   "Train all untrained" -> only versions with no run · "Retrain all" -> every version (force)
# --- batch train / retrain / continue, with LIVE loss + accuracy for every version ---
#   Train all untrained -> versions with no run · Retrain all -> every version (force)
#   Continue all -> every trained version, warm-started + `continueall_extra` more steps appended
if trainall_clicked:
    _batch, _mode = [v for v in exp["versions"] if v["run"] is None], "train"
elif retrainall_clicked:
    _batch, _mode = list(exp["versions"]), "train"
elif continueall_clicked:
    _batch, _mode = [v for v in exp["versions"] if v["run"] is not None], "continue"
else:
    _batch, _mode = None, None

if _batch:
    todo = _batch
    view = st.session_state.get("trainall_view", "Overlay")
    names = {v["id"]: v["name"] for v in todo}

    def _curve(cks):                                            # held-out curve from a list of checkpoints
        return {"step": [c.step for c in cks], "loss": [round(c.eval_loss, 4) for c in cks],
                "acc": [round(c.acc, 4) for c in cks]}

    # continue seeds each curve with the run's existing history; train/retrain start empty
    curves = {v["id"]: (_curve(v["run"].checkpoints) if _mode == "continue"
                        else {"step": [], "loss": [], "acc": []}) for v in todo}
    st.subheader(f"⏳ {'Continuing' if _mode == 'continue' else 'Training'} {len(todo)} version(s) "
                 f"— live ({view.lower()})")
    bar = st.progress(0.0, text="starting…")
    loss_ph = acc_ph = None
    phs = {}
    if view == "Overlay":
        gl, ga = st.columns(2)
        gl.caption("loss (held-out)"); ga.caption("accuracy (held-out)")
        loss_ph, acc_ph = gl.empty(), ga.empty()
    else:                                                        # one row of charts per version
        for v in todo:
            st.caption(f"**{v['name']}** — loss · accuracy (held-out)")
            c1, c2 = st.columns(2)
            phs[v["id"]] = (c1.empty(), c2.empty())

    def redraw(cur_id):                                         # plot each version at its OWN steps (no gaps)
        if view == "Overlay":                                   # long format -> one colored line per version
            lr = [{"step": s, "loss": l, "version": names[i]}
                  for i in curves for s, l in zip(curves[i]["step"], curves[i]["loss"])]
            ar = [{"step": s, "accuracy": a, "version": names[i]}
                  for i in curves for s, a in zip(curves[i]["step"], curves[i]["acc"])]
            loss_ph.line_chart(pd.DataFrame(lr), x="step", y="loss", color="version")
            acc_ph.line_chart(pd.DataFrame(ar), x="step", y="accuracy", color="version")
        else:                                                   # just the version that updated (its own steps)
            lp, ap = phs[cur_id]
            lp.line_chart(pd.DataFrame({"step": curves[cur_id]["step"], "loss": curves[cur_id]["loss"]}),
                          x="step", y="loss")
            ap.line_chart(pd.DataFrame({"step": curves[cur_id]["step"], "accuracy": curves[cur_id]["acc"]}),
                          x="step", y="accuracy")

    if _mode == "continue":                                     # show each run's existing curve before appending
        if view == "Overlay":
            redraw(next(iter(curves)))
        else:
            for cur_id in curves:
                redraw(cur_id)

    for k, v in enumerate(todo):
        vid = v["id"]
        if _mode == "continue":
            total, base = continueall_extra, v["run"].checkpoint_steps[-1]
        else:
            vcfg = cfg if vid == exp["active"] else version_cfg(v)   # active uses the live cfg
            vsig = sig if vid == exp["active"] else version_sig(v)
            total, base = vcfg.steps, 0

        def on_step(s, metrics, _k=k, _name=names[vid], _total=total, _base=base):
            d = s - _base                                       # continue passes absolute steps; show relative
            bar.progress(min(1.0, (_k + (d + 1) / _total) / len(todo)),
                         text=f"{_name} ({_k + 1}/{len(todo)}) · step {d + 1}/{_total}")

        def on_ck(ck, all_, _vid=vid):                          # rebuild from all_ (covers seeded + new)
            curves[_vid] = _curve(all_)
            redraw(_vid)

        if _mode == "continue":
            continue_training(v["run"], continueall_extra, on_step=on_step, on_checkpoint=on_ck)
        else:
            v["run"] = TrainingRun.train(vcfg, on_step=on_step, on_checkpoint=on_ck)
            v["run_sig"] = vsig
    bar.empty()
    st.rerun()

# ---- continue: warm-start from the current run's final weights and train `extra` more steps ----
if continue_clicked and run is not None:
    _base = run.checkpoint_steps[-1]
    st.subheader(f"⏳ Continuing training (+{extra} steps, from step {_base})…")
    cg1, cg2 = st.columns(2)
    cg1.caption("loss (held-out)")
    cg2.caption("accuracy (held-out)")
    clph, caph = cg1.empty(), cg2.empty()

    def _draw_cont(cks):                                      # live charts CONTINUE from the existing curve
        xs = [c.step for c in cks]
        clph.line_chart(pd.DataFrame({"loss": [c.eval_loss for c in cks]}, index=xs))
        caph.line_chart(pd.DataFrame({"accuracy": [c.acc for c in cks]}, index=xs))

    _draw_cont(run.checkpoints)                               # seed with the run so far
    cbar = st.progress(0.0, text="continuing…")
    _cev = max(1, extra // 100)

    def on_cstep(s, metrics):
        d = s - _base
        if d % _cev == 0 or d == extra - 1:
            cbar.progress(min(1.0, (d + 1) / extra), text=f"step {s + 1}/{_base + extra}")

    continue_training(run, extra, on_step=on_cstep,          # mutates `run` (== ver["run"]) in place
                      on_checkpoint=lambda ck, all_: _draw_cont(all_))
    st.rerun()

# ---- model render (left) + section editor (right); the Run-tab controls above drive both ----
if focus:
    col_model, col_edit = st.container(), None
else:
    col_model, col_edit = st.columns([2.3, 1])

with col_model:
    if trained:
        if full_flow:
            gkey = "off"
            if show_grads:
                gkey = st.session_state.get("gradsrc", "averaged (3 batches)")
                if gkey == "scrub a batch":
                    gkey += f"-{st.session_state.get('gradbatch', 0)}"
            fkey = f"{sig}|s{step}|rp{readout_pos}|in{chosen}|grad{gkey}|t{steps_all[-1]}"
            if st.session_state.get("frame_key") != fkey:    # re-render on step / input / readout / grad change
                with st.spinner("rendering…"):
                    trc = trace_forward(model, task, make_probe(model))
                    st.session_state.frame_doc = flow_svg_component(trc, readout_pos, grads)
                    st.session_state.frame_key = fkey
            fdoc, fhh = st.session_state.frame_doc
            st.iframe(fdoc, height=fhh)
            if show_grads:
                st.caption("🔥 ∇ badges on each section header show that block's gradient magnitude "
                           "at this stage (redder = larger learning signal).")
            with st.expander("▶ Play the whole training as an animation"):
                psig = f"{sig}|rp{readout_pos}|in{chosen}|t{steps_all[-1]}"
                if st.session_state.get("replay_sig") != psig:
                    replay_steps = steps_all if len(steps_all) <= 18 else \
                        sorted({steps_all[round(k * (len(steps_all) - 1) / 17)] for k in range(18)})
                    frames, W, H = [], 0, 0
                    prog = st.progress(0.0, text="rendering frames…")
                    for nn, stp in enumerate(replay_steps):
                        ck = run.nearest_checkpoint(stp)
                        m = run.reconstruct(stp)
                        trc = trace_forward(m, task, make_probe(m))
                        inner, w, h = model_svg(trc, readout_pos)
                        frames.append((stp, ck.eval_loss, ck.acc, svg_document(inner, w, h)))
                        W, H = max(W, w), max(H, h)
                        prog.progress((nn + 1) / len(replay_steps))
                    prog.empty()
                    st.session_state.replay_html = replay_html(frames, W, H)
                    st.session_state.replay_sig = psig
                rhtml, rheight = st.session_state.replay_html
                st.iframe(rhtml, height=rheight)
        else:
            trc = trace_forward(model, task, make_probe(model))
            st.caption(f"{head_kind} head — heatmap view")
            st.iframe(flow_html(trc), height=component_height(trc))

        if grads is not None:
            st.divider()
            st.markdown("**Gradient flow** at this stage — one backward pass, computed live (never stored)")
            o1, o2, o3 = st.columns(3)
            log_g = o1.checkbox("log scale", value=True, key="gradlog")
            per_head = o2.checkbox("per-head Q/K/V/O", value=False, key="gradhead")
            fixed = o3.checkbox("🔒 fixed scale", value=False, key="gradfixed",
                                help="pin the axis to the run-wide max so you can compare across steps")
            xmax = None
            if fixed:
                gkey2 = f"{sig}|t{steps_all[-1]}"             # recompute only per run (incl. continue)
                if st.session_state.get("gscale_sig") != gkey2:
                    with st.spinner("measuring run-wide gradient scale…"):
                        st.session_state.gscale = gradient_scale(run, task)
                        st.session_state.gscale_sig = gkey2
                xmax = st.session_state.gscale
            gg1, gg2 = st.columns(2)
            _show(gg1, _grad_bars(grads, arch, log=log_g, per_head=per_head, xmax=xmax))
            _show(gg2, _metric_chart(run.metrics["step"], run.metrics["grad_norm"], step, "#7c3aed",
                                     f"total ‖∇‖ over training (now {grads['total']:.3f})"))
            gg2.caption("The total gradient shrinks as it converges; Q/K often start near zero "
                        "(flat attention) and grow as the model learns what to attend to.")
    else:
        if run is not None:
            st.info("Architecture/settings changed since training — showing the **untrained** model. Re-train to refill.")
        else:
            st.markdown("**Untrained model** (random weights) — build it, then ▶ Train.")
        if st.session_state.get("flow_sig") != rsig:         # re-render only when config/readout/input changes
            with st.spinner("rendering model…"):
                mu = build_model(cfg, task, device="cpu", seed=cfg.seed)
                tr = trace_forward(mu, task, make_probe(mu))
                if full_flow:
                    st.session_state.flow_doc = flow_svg_component(tr, readout_pos)
                else:
                    st.session_state.flow_doc = (flow_html(tr), component_height(tr))
                st.session_state.flow_full = full_flow
                st.session_state.flow_sig = rsig
        if not st.session_state.flow_full:
            st.info("Add at least one **attention** and one **feed-forward** section to see the full flow.")
        doc, hh = st.session_state.flow_doc
        st.iframe(doc, height=hh)

# ---- section spine editor (right column, or folded into an expander in full-width mode) ----
editor_box = col_edit if col_edit is not None else st.expander("✏️ Edit sections & settings", expanded=True)
with editor_box:
    st.subheader("Sections")
    st.caption("settings cards — adjust per section; ➕ inserts a section")

    with st.container(border=True):                           # Embedding card
        st.markdown("**◆ Embedding**")
        st.selectbox("positional encoding", ["learned", "sinusoidal"], key="pos_encoding")
        st.caption(f"token + positional · vocab {task.vocab_size} · block {task.block_size}")

    for i, b in enumerate(arch):
        bid = b["_id"]
        st.button("➕ insert attention above", key=f"ins{bid}", width="stretch",
                  on_click=insert_block, args=(i, "attn"))
        with st.container(border=True):                       # per-section settings card
            if b["type"] == "attn":
                st.markdown("**▸ Attention**")
                opts = divisor_heads(d_model)
                cur = b.get("n_heads", n_heads)
                cur = cur if cur in opts else opts[-1]
                b["n_heads"] = st.selectbox("# heads", opts, index=opts.index(cur), key=f"nh{bid}")
                st.caption(f"d_head = {d_model // b['n_heads']}  (= d_model ÷ heads, fixed)")
                b["causal"] = st.checkbox("causal mask", b.get("causal", True), key=f"ca{bid}")
                b["override"] = st.toggle("override model defaults", b.get("override", False), key=f"ov{bid}")
                if b["override"]:
                    b["bias"] = st.checkbox("bias (Q/K/V/O)", b.get("bias", default_bias), key=f"ba{bid}")
                    b["attn_dropout"] = st.slider("attention dropout", 0.0, 0.5,
                                                  float(b.get("attn_dropout", dropout)), 0.05, key=f"ad{bid}")
                else:
                    st.caption(f"bias **{'on' if default_bias else 'off'}** · dropout **{dropout}**  ·  _model defaults_")
                wo = st.checkbox("W_O: add hidden layer (non-standard)", bool(b.get("wo_hidden")), key=f"wo{bid}")
                if wo:
                    b["wo_units"] = st.number_input("W_O hidden units", 4, 512,
                                                    int(b.get("wo_units", d_model)), step=4, key=f"wou{bid}")
                    b["wo_act"] = st.selectbox("W_O activation", ["gelu", "relu", "silu", "tanh"],
                                               index=["gelu", "relu", "silu", "tanh"].index(b.get("wo_act", "gelu")),
                                               key=f"woa{bid}")
                    b["wo_hidden"] = (int(b["wo_units"]),)
                else:
                    b["wo_hidden"] = ()
                    st.caption("W_O = standard linear (n_heads·d_head → d_model)")
            else:
                st.markdown("**▸ Feed-forward**")
                b["n_layers"] = st.number_input("# hidden layers", 1, 4, int(b.get("n_layers", 1)), key=f"nl{bid}")
                b["hidden"] = st.number_input("units / hidden layer", 8, 1024,
                                              int(b.get("hidden", ffn_mult * d_model)), step=8, key=f"hu{bid}")
                b["override"] = st.toggle("override model defaults", b.get("override", False), key=f"ov{bid}")
                if b["override"]:
                    b["activation"] = st.selectbox("activation", ["gelu", "relu", "silu", "tanh"],
                                                   index=["gelu", "relu", "silu", "tanh"].index(b.get("activation", default_activation)),
                                                   key=f"ac{bid}")
                    b["bias"] = st.checkbox("bias", b.get("bias", default_bias), key=f"fb{bid}")
                    b["dropout"] = st.slider("dropout", 0.0, 0.5, float(b.get("dropout", dropout)), 0.05, key=f"fd{bid}")
                else:
                    st.caption(f"activation **{default_activation}** · bias **{'on' if default_bias else 'off'}** · "
                               f"dropout **{dropout}**  ·  _model defaults_")
            st.button("✕ remove", key=f"rm{bid}", on_click=remove_block, args=(i,))

    add = st.columns(2)
    add[0].button("➕ Attention", key="add_attn", width="stretch",
                  on_click=insert_block, args=(len(arch), "attn"))
    add[1].button("➕ Feed-fwd", key="add_ffn", width="stretch",
                  on_click=insert_block, args=(len(arch), "ffn"))

    with st.container(border=True):                           # Output settings card
        st.markdown("**◆ Output**")
        if head_kind == "lm":
            st.caption("head: **unembed / LM** (weight-tied + final LayerNorm) → softmax")
            st.caption("pooling: **last token** (reads the predicting position)")
            samp = st.selectbox("sampling", ["greedy", "temperature", "top_k", "top_p"], key="samp")
            if samp == "temperature":
                st.slider("temperature", 0.1, 2.0, 1.0, 0.1, key="temp")
            elif samp == "top_k":
                st.slider("top-k", 1, task.vocab_size, 3, key="topk")
            elif samp == "top_p":
                st.slider("top-p", 0.1, 1.0, 0.9, 0.05, key="topp")
        else:
            st.selectbox("pooling", ["last", "mean", "cls", "attn"], index=1, key="pool_sel")
            if head_kind == "classify":
                st.caption(f"head: **classification** · {getattr(task, 'n_classes', '?')} classes (dense → linear)")
            else:
                st.caption(f"head: **regression** · {getattr(task, 'out_dim', 1)} output(s)")
            st.caption("sampling: argmax (the predicted class / value)")
        st.caption("↳ head type is set by the dataset/task")

# ---------------------------------------------------------------------------
# generation (only meaningful once trained, LM)
# ---------------------------------------------------------------------------
if trained and head_kind == "lm":
    st.divider()
    st.subheader("Try the trained model — uses the **probe input** + **Output-card sampling** above")
    sampling = st.session_state.get("samp", "greedy")        # set in the Output settings card
    temp = st.session_state.get("temp", 1.0) if sampling == "temperature" else 1.0
    top_k = st.session_state.get("topk", 0) if sampling == "top_k" else 0
    top_p = st.session_state.get("topp", 0.0) if sampling == "top_p" else 0.0
    out = generate_sampled(model, task, in_ids, method=sampling, temperature=temp,
                           top_k=top_k, top_p=top_p, device="cpu")[0]
    st.write(f"**prompt** `{task.decode(in_ids[0])}`  →  **model** `{task.decode(out)}`  ·  sampling: **{sampling}**")
    if task_name == "Add":                                    # answer is least-significant-digit first
        nd = task.n_digits
        p = in_ids[0].tolist()
        a = int("".join(map(str, p[:nd])))
        b = int("".join(map(str, p[nd + 1:2 * nd + 1])))
        val = sum(int(d) * 10 ** k for k, d in enumerate(out.tolist()))
        ok = "✅" if val == a + b else f"❌ (should be {a + b})"
        st.caption(f"↳ reads as **{a} + {b} = {val}** {ok}  ·  the answer is written "
                   f"least-significant digit first (reversed), so read it right-to-left.")


# --- persist the active version's config (read straight from the global widget keys) so switching
# versions keeps each one's settings; the load-on-switch near the top restores it on the next switch ---
ver["config"] = {base: st.session_state.get(base, DEFAULTS[base]) for base in CONFIG_KEYS}
# Snapshot the SHARED widget state (task + its tk_* params + the training protocol) so it survives a page visit.
_shared_keys = ["task"] + [k for k in st.session_state if isinstance(k, str) and k.startswith("tk_")] + TRAIN_KEYS
exp["shared"] = {k: st.session_state[k] for k in _shared_keys if k in st.session_state}
