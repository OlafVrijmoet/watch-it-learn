"""app_state — the per-version config/spec builders for the Builder page.

Extracted from builder_app.py to keep it a thin view layer. These take explicit config dicts (no
builder_app module globals) and read the live `st.session_state` only for per-section overrides, so
the build logic lives in one place. builder_app's `make_cfg`/`version_cfg` call `_build_cfg` with the
active (or a saved) config + the shared training protocol.
"""
import streamlit as st

from replay_engine import RunConfig


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
            specs.append(("ffn", {"hidden": tuple([units] * nl),
                                    "activation": ss.get(f"ac{i}", b.get("activation", act_def)) if ovr else act_def,
                                    "bias": ss.get(f"fb{i}", b.get("bias", bias_def)) if ovr else bias_def,
                                    "dropout": ss.get(f"fd{i}", b.get("dropout", drop)) if ovr else drop}))
    return tuple(specs)


def _build_cfg(c, arch_, shared):
    """A RunConfig from a per-version config dict `c` (architecture / init / output) + its arch + the SHARED
    task/training-protocol dict (steps/batch/lr/optimizer/schedule/checkpoints/device + task/head). Reads only
    its arguments (plus live st.session_state for per-section overrides) — no builder_app module globals."""
    pool = c["pool_sel"] if shared["head"] in ("classify", "regression") else "last"
    return RunConfig(task_name=shared["task_name"], task_kwargs=shared["task_kwargs"],
                     layer_specs=version_layer_specs(arch_, c),
                     d_model=c["d_model"], n_heads=c["n_heads"], dropout=c["def_dropout"],
                     steps=shared["steps"], batch=shared["batch"], lr=shared["lr"], optimizer=shared["optimizer"],
                     lr_schedule=shared["lr_schedule"], head=shared["head"], pooling=pool,
                     pos_encoding=c["pos_encoding"], init=c["init_scheme"], init_scale=c["init_scale"],
                     seed=c["seed"], n_checkpoints=shared["n_checkpoints"], device=shared["dev"])
