"""
Tests for flow_component.py - the interactive D3 view.

We can't run the browser JS headlessly, but we verify the Python contract: `flow_html`
produces a self-contained document that embeds VALID JSON matching the trace, pulls in
D3, and renders for every head type. (The replay engine produces the trace; these tests
use it end to end.)

Run:  PYTHONPATH=. python tests/test_flow_component.py (from the repo root)
"""
from __future__ import annotations

import json
import re

import torch

from replay_engine import RunConfig, TrainingRun, trace_forward
from flow_component import flow_html, component_height


def _trace(cfg):
    run = TrainingRun.train(cfg)
    task = run.task
    if getattr(task, "kind", "lm") == "lm":
        _, _, seq = task.make_batch(1, generator=torch.Generator().manual_seed(3))
        probe = seq[0]
    else:
        x, _ = task.make_batch(1, generator=torch.Generator().manual_seed(3))
        probe = x[0]
    return trace_forward(run.reconstruct(cfg.steps), task, probe), task


def _embedded_json(html):
    m = re.search(r"const D = (\{.*?\});\nconst svg", html, re.S)
    assert m, "could not find embedded DATA"
    return json.loads(m.group(1))


def _lm_cfg(**kw):
    base = dict(task_name="Reverse", task_kwargs={"length": 4},
                layer_specs=(("attn",), ("ffn", (48,))), d_model=32, n_heads=4,
                steps=40, n_checkpoints=8, seed=0, device="cpu")
    base.update(kw)
    return RunConfig(**base)


def test_flow_html_self_contained_and_valid_json():
    tr, _ = _trace(_lm_cfg())
    html = flow_html(tr)
    assert html.lstrip().startswith("<!doctype html>") and "</html>" in html
    assert "d3@7" in html and "<svg" in html
    # embedded JSON parses and matches the trace
    data = _embedded_json(html)
    assert data["T"] == tr["T"]
    assert data["token_strs"] == tr["token_strs"]
    assert len(data["stages"]) == len(tr["stages"])
    assert len(data["attention"]) == len(tr["attention"])


def test_flow_html_handles_all_head_types():
    # lm
    tr_lm, _ = _trace(_lm_cfg())
    html_lm = flow_html(tr_lm)
    assert tr_lm["head"] == "lm"
    assert "next-token" in html_lm
    # classify
    tr_cl, _ = _trace(_lm_cfg(task_name="Majority", task_kwargs={"length": 13},
                              head="classify", pooling="mean"))
    html = flow_html(tr_cl)
    assert _embedded_json(html)["head"] == "classify"
    assert "class probabilities" in html
    # regression
    tr_rg, _ = _trace(_lm_cfg(task_name="Density", task_kwargs={"length": 12},
                              head="regression", pooling="attn"))
    html = flow_html(tr_rg)
    assert _embedded_json(html)["head"] == "regression"
    assert "predicted value" in html


def test_component_height_positive():
    tr, _ = _trace(_lm_cfg())
    assert component_height(tr) > 100
    assert isinstance(component_height(tr), int)


def test_vocab_strs_in_trace():
    tr, task = _trace(_lm_cfg())
    assert tr["vocab_strs"] == [task.id_to_str.get(i, "?") for i in range(task.vocab_size)]


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
