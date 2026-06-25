"""
Smoke/integration test for builder_app.py using Streamlit's AppTest.

Drives the real app headlessly: configure a tiny run, click Train, scrub the
training timeline, and assert the whole pipeline (train -> reconstruct -> render
the flow / attention / logit-lens figures) runs without errors.

Run:  python -m pytest test_builder_app.py -v   (or: python test_builder_app.py)
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _by_key(collection, key):
    for el in collection:
        if getattr(el, "key", None) == key:
            return el
    raise KeyError(f"no element with key {key!r}")


def _fresh_app():
    at = AppTest.from_file("builder_app.py", default_timeout=300)
    at.run()
    return at


def _run(at):
    """The active version's TrainingRun (or None) — the run now lives inside the experiment."""
    exp = at.session_state["experiment"]
    return next(v for v in exp["versions"] if v["id"] == exp["active"])["run"]


def test_renders_before_training():
    at = _fresh_app()
    assert not at.exception                       # the build/preview renders cleanly
    assert not at.error
    assert _run(at) is None                       # nothing trained yet


def test_probe_tolerates_stale_token():
    """A probe token left over from another task (not in the current vocab) must not crash the Run
    computation — it gets sanitized rather than blowing up opt_strs.index()."""
    at = _fresh_app()                             # untrained: the 'chosen' computation runs before the Run tab
    at.session_state["pf0"] = "ZZZ_not_a_token"   # simulate a leftover token from a different task
    at.run()
    assert not at.exception, at.exception


def test_experiment_state_foundation():
    """Phase 1: state lives in an `experiment` with one active version that owns arch + run."""
    at = _fresh_app()
    exp = at.session_state["experiment"]
    assert len(exp["versions"]) == 1 and exp["active"] == exp["versions"][0]["id"]
    v = exp["versions"][0]
    assert v["run"] is None and isinstance(v["arch"], list) and len(v["arch"]) >= 2


def test_versions_duplicate_and_switch_keep_independent_configs():
    """Phase 2: duplicating copies the config; editing one version doesn't touch the other,
    and switching back restores each version's own settings."""
    at = _fresh_app()
    v1_id = at.session_state["experiment"]["active"]
    _by_key(at.select_slider, "d_model").set_value(64); at.run()      # v1: d_model 64
    _by_key(at.button, "dupv").click(); at.run()                      # duplicate -> v2 (active), config copied
    exp = at.session_state["experiment"]
    assert len(exp["versions"]) == 2 and exp["active"] != v1_id
    v2_id = exp["active"]
    assert _by_key(at.select_slider, "d_model").value == 64           # the clone copied v1's config
    _by_key(at.select_slider, "d_model").set_value(16); at.run()      # edit only v2
    _by_key(at.button, f"selv{v1_id}").click(); at.run()             # switch back to v1
    assert _by_key(at.select_slider, "d_model").value == 64           # v1's config preserved
    cfgs = {v["id"]: v["config"]["d_model"] for v in at.session_state["experiment"]["versions"]}
    assert cfgs[v1_id] == 64 and cfgs[v2_id] == 16                    # independent


def test_train_all_untrained_versions():
    """Phase 3: 'Train all untrained' trains every version without a run, sequentially."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(50); _by_key(at.slider, "nck").set_value(8); at.run()
    _by_key(at.button, "dupv").click(); at.run()                      # 2 versions, both untrained
    assert all(v["run"] is None for v in at.session_state["experiment"]["versions"])
    _by_key(at.button, "trainall").click(); at.run()
    assert not at.exception, at.exception
    vs = at.session_state["experiment"]["versions"]
    assert len(vs) == 2 and all(v["run"] is not None for v in vs)     # both trained
    assert all(v["run_sig"] is not None for v in vs)


def test_train_all_live_charts_both_views():
    """Phase 3: the live Train-all loss/accuracy charts render in both Overlay and Stacked layouts."""
    for layout in ("Overlay", "Stacked"):
        at = _fresh_app()
        _by_key(at.slider, "steps").set_value(50); _by_key(at.slider, "nck").set_value(6)
        _by_key(at.radio, "trainall_view").set_value(layout); at.run()
        _by_key(at.button, "dupv").click(); at.run()                  # 2 untrained versions
        _by_key(at.button, "trainall").click(); at.run()
        assert not at.exception, (layout, at.exception)
        assert all(v["run"] is not None for v in at.session_state["experiment"]["versions"])


def test_version_goes_stale_on_edit():
    """A freshly trained version isn't stale; editing its config marks it stale (run kept, not deleted)."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(50); _by_key(at.slider, "nck").set_value(8); at.run()
    _by_key(at.button, "train").click(); at.run()
    v = at.session_state["experiment"]["versions"][0]
    assert v["run"] is not None and not v.get("stale")                # just trained -> fresh
    _by_key(at.select_slider, "d_model").set_value(64); at.run()      # change the architecture
    v = at.session_state["experiment"]["versions"][0]
    assert v["run"] is not None and v.get("stale")                   # run kept, now flagged stale


def test_trained_survives_widget_state_drop():
    """Returning to the Builder after visiting another page (widgets reset, but the experiment persists in
    session_state) must NOT falsely flag a freshly trained run as stale — the active version's config is
    restored from the experiment (this is the Compare→Builder 'session lost' bug)."""
    at = _fresh_app()
    opts = list(_by_key(at.selectbox, "task").options)
    _by_key(at.selectbox, "task").set_value(opts[0]); at.run()        # a non-default SHARED task (default = index 1)
    _by_key(at.select_slider, "d_model").set_value(48); at.run()      # a non-default per-version config
    _by_key(at.slider, "steps").set_value(100); _by_key(at.slider, "nck").set_value(10); at.run()
    _by_key(at.button, "train").click(); at.run()
    exp = at.session_state["experiment"]
    assert exp["versions"][0]["run"] is not None and not exp["versions"][0].get("stale")
    # simulate the page-nav return: a fresh Builder (no widget state) but the SAME experiment object
    at2 = AppTest.from_file("builder_app.py", default_timeout=300)
    at2.session_state["experiment"] = exp
    at2.run()
    assert not at2.exception, at2.exception
    assert not at2.session_state["experiment"]["versions"][0].get("stale"), \
        "run went stale after returning to a fresh Builder (task/config not restored)"


def test_retrain_all_replaces_every_run():
    """'Retrain all' force-retrains every version (new run objects), even already-trained ones."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(50); _by_key(at.slider, "nck").set_value(8); at.run()
    _by_key(at.button, "dupv").click(); at.run()                      # 2 versions
    _by_key(at.button, "trainall").click(); at.run()                  # both trained
    before = [v["run"] for v in at.session_state["experiment"]["versions"]]
    _by_key(at.button, "retrainall").click(); at.run()
    assert not at.exception, at.exception
    after = [v["run"] for v in at.session_state["experiment"]["versions"]]
    assert all(r is not None for r in after)
    assert all(a is not b for a, b in zip(before, after))            # every run was re-created


def test_train_all_uses_shared_steps():
    """Training protocol is shared: every version trains with the single Train-tab steps, not a per-version one."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(100); _by_key(at.slider, "nck").set_value(10); at.run()
    _by_key(at.button, "dupv").click(); at.run()                     # 2 versions
    _by_key(at.button, "trainall").click(); at.run()
    vs = at.session_state["experiment"]["versions"]
    assert all(v["run"].cfg.steps == 100 for v in vs)              # all trained with the shared 100
    _by_key(at.slider, "steps").set_value(150); at.run()            # change the shared steps
    _by_key(at.button, "retrainall").click(); at.run()
    vs = at.session_state["experiment"]["versions"]
    assert all(v["run"].cfg.steps == 150 for v in vs)              # Retrain-all applies it to every version


def test_train_then_scrub():
    at = _fresh_app()
    # tiny, fast run
    _by_key(at.slider, "steps").set_value(50)
    _by_key(at.slider, "nck").set_value(10)
    at.run()
    _by_key(at.button, "train").click()
    at.run()

    # no exception means train -> reconstruct -> render (flow/attention/logit-lens) all ran
    assert not at.exception, at.exception
    assert not at.error, at.error
    assert _run(at) is not None
    run = _run(at)
    assert len(run.checkpoints) >= 2
    assert run.checkpoint_steps[0] == 0 and run.checkpoint_steps[-1] == 50
    # the training-replay component was rendered (scrubbing is now client-side inside it)
    assert "replay_html" in at.session_state


def test_exact_reconstruction_through_app():
    """The model the app reconstructs at a checkpoint == deterministic replay (faithful)."""
    import torch
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(50)
    _by_key(at.slider, "nck").set_value(10)
    at.run()
    _by_key(at.button, "train").click()
    at.run()
    run = _run(at)
    s = run.checkpoint_steps[len(run.checkpoint_steps) // 2]
    a = run.reconstruct(s).state_dict()
    b = run.reconstruct_exact(s).state_dict()
    assert all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def test_classify_task_in_app():
    """Selecting a classification task drives the pooled-head path end to end (train + render)."""
    at = _fresh_app()
    _by_key(at.selectbox, "task").set_value("Majority")
    at.run()
    _by_key(at.slider, "steps").set_value(50)
    _by_key(at.slider, "nck").set_value(10)
    at.run()
    _by_key(at.button, "train").click()
    at.run()
    assert not at.exception, at.exception
    assert not at.error, at.error
    assert _run(at) is not None
    assert _run(at).cfg.head == "classify"


def test_generation_responds_to_input():
    """The output must change when the editable probe input (a per-field selectbox) changes."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(150)
    _by_key(at.slider, "nck").set_value(8)
    at.run()
    _by_key(at.button, "train").click()
    at.run()
    assert _run(at) is not None

    def out_text():
        return next((m.value for m in at.markdown if "**model**" in m.value), "")

    opts = list(_by_key(at.selectbox, "pf0").options)  # first probe-input field's tokens
    _by_key(at.selectbox, "pf0").set_value(opts[1]); at.run(); a = out_text()
    _by_key(at.selectbox, "pf0").set_value(opts[2]); at.run(); b = out_text()
    assert a and b, (a, b)
    assert a != b, f"output did not change with input: {a!r} vs {b!r}"


def test_train_set_overlay():
    """The 'also plot the training set' overlay computes + renders without error."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(120)
    _by_key(at.slider, "nck").set_value(6)
    at.run()
    _by_key(at.button, "train").click()
    at.run()
    _by_key(at.checkbox, "showtrain").set_value(True)
    at.run()
    assert not at.exception, at.exception
    assert "traincurve" in at.session_state


def test_sidebar_run_tab():
    """Run tab is empty before training and holds the views (stage slider, train overlay) after."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(120)
    _by_key(at.slider, "nck").set_value(6)
    at.run()
    assert any(getattr(w, "key", None) == "train" for w in at.button)          # Train tab button
    assert not any(getattr(w, "key", None) == "stepslider" for w in at.slider)  # no run yet
    _by_key(at.button, "train").click()
    at.run()
    assert not at.exception, at.exception
    assert any(getattr(w, "key", None) == "stepslider" for w in at.slider)      # Run-tab stage slider
    assert any(getattr(w, "key", None) == "showtrain" for w in at.checkbox)
    assert any(getattr(w, "key", None) == "showgrad" for w in at.toggle)
    # continue training appends checkpoints to the same run
    n0 = len(_run(at).checkpoints)
    _by_key(at.button, "continue").click()
    at.run()
    assert not at.exception, at.exception
    assert len(_run(at).checkpoints) > n0


def test_gradient_panel_all_sources():
    """The gradient panel renders without error for every gradient source."""
    at = _fresh_app()
    _by_key(at.slider, "steps").set_value(120)
    _by_key(at.slider, "nck").set_value(8)
    at.run()
    _by_key(at.button, "train").click()
    at.run()
    _by_key(at.toggle, "showgrad").set_value(True)
    at.run()
    assert not at.exception, at.exception
    for src in ("scrub a batch", "exact training batch", "your probe input", "averaged (3 batches)"):
        _by_key(at.radio, "gradsrc").set_value(src)
        at.run()
        assert not at.exception, (src, at.exception)
    _by_key(at.checkbox, "gradhead").set_value(True)         # per-head Q/K/V/O bars + overlay
    _by_key(at.checkbox, "gradlog").set_value(False)         # linear scale
    at.run()
    assert not at.exception, at.exception
    _by_key(at.checkbox, "gradfixed").set_value(True)        # run-wide fixed scale
    at.run()
    assert not at.exception, at.exception
    assert "gscale" in at.session_state                       # the run-wide scale was computed + cached


def test_compare_page_renders():
    """Phase 4: Compare page shows a friendly note with no runs, and charts + table with a trained run."""
    from replay_engine import RunConfig, TrainingRun
    at = AppTest.from_file("compare_page.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception                      # empty experiment -> info, no crash

    cfg = RunConfig(task_name="Reverse", task_kwargs={"length": 3},
                    layer_specs=(("attn",), ("dense", (48,))), d_model=24, n_heads=3,
                    steps=40, n_checkpoints=6, seed=0)
    run = TrainingRun.train(cfg)
    at2 = AppTest.from_file("compare_page.py", default_timeout=300)
    at2.session_state["experiment"] = {
        "versions": [{"id": 0, "name": "v1", "note": "", "config": {}, "arch": [],
                      "run": run, "run_sig": "x"}],
        "active": 0, "next_vid": 1, "next_block_id": 2}
    at2.session_state["task"] = "Reverse"
    at2.run()
    assert not at2.exception, at2.exception                    # charts + status table render
    _by_key(at2.radio, "cmp_curves").set_value("Both"); at2.run()       # exercises train_eval_curve
    assert not at2.exception, at2.exception
    _by_key(at2.radio, "cmp_view").set_value("Small-multiples"); at2.run()
    assert not at2.exception, at2.exception
    # gradient-flow compare options: log scale, per-Q/K/V/O split, and the gradient source
    _by_key(at2.checkbox, "cmp_gradlog").set_value(False); at2.run()
    assert not at2.exception, at2.exception
    _by_key(at2.checkbox, "cmp_gradhead").set_value(True); at2.run()        # split attention into Q/K/V/O
    assert not at2.exception, at2.exception
    _by_key(at2.selectbox, "cmp_gradsrc").set_value("exact training batch"); at2.run()
    assert not at2.exception, at2.exception


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); print(f"  ok   {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL {fn.__name__}"); traceback.print_exc(); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
