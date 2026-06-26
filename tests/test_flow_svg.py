"""
Tests for flow_svg.py - the live Figma-matching SVG renderer (model_svg + flow_svg_component).

It's plain SVG, so cairosvg can rasterize it to a real PNG for a genuine self-check (no browser).
These verify the whole-model render builds for different head counts, the Streamlit component HTML
is well-formed with hover hooks, the SVG actually rasterizes, and the gradient overlay draws.

Run:  PYTHONPATH=. python tests/test_flow_svg.py (from the repo root)
"""
from __future__ import annotations

import os
import tempfile

import torch

from replay_engine import RunConfig, TrainingRun, trace_forward, layer_gradients, sample_train_batch
from flow_svg import (model_svg, flow_svg_component, svg_document, render_png,
                      _band_sequence, _grad_normalizers)


def _run(d_model=24, n_heads=3, length=3):
    cfg = RunConfig(task_name="Reverse", task_kwargs={"length": length},
                    layer_specs=(("attn",), ("ffn", (48,))), d_model=d_model, n_heads=n_heads,
                    steps=40, n_checkpoints=8, seed=0)
    return TrainingRun.train(cfg)


def _trace(d_model=24, n_heads=3, length=3):
    run = _run(d_model, n_heads, length)
    task = run.task
    _, _, seq = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    return trace_forward(run.reconstruct(run.cfg.steps), task, seq[0])


def test_model_svg_builds_all_sections():
    inner, w, h = model_svg(_trace())
    assert w > 400 and h > 400 and len(inner) > 1000
    for label in ("EMBEDDINGS", "ATTENTION", "FEED-FORWARD", "OUTPUT"):
        assert label in inner, label


def test_component_html_well_formed():
    doc, hh = flow_svg_component(_trace())
    assert doc.lstrip().startswith("<!doctype") and "</html>" in doc
    assert "<svg" in doc and "data-tip" in doc                  # hover wiring present
    assert isinstance(hh, int) and hh > 400


def test_adaptive_width_for_more_heads():
    _, w3, _ = model_svg(_trace(24, 3))
    _, w4, _ = model_svg(_trace(32, 4))
    assert w4 >= w3                                             # 4 heads at least as wide as 3


def test_model_svg_rasterizes_to_png():
    inner, w, h = model_svg(_trace())
    p = os.path.join(tempfile.gettempdir(), "model_svg_test.png")
    render_png(svg_document(inner, w, h), p, scale=1.0)
    assert os.path.getsize(p) > 1000                           # produced a real image


def test_gradient_overlay_renders():
    run = _run()
    task = run.task
    _, _, seq = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    m = run.reconstruct(run.cfg.steps)
    grads = layer_gradients(m, [sample_train_batch(run.cfg, task, 1)])
    tr = trace_forward(m, task, seq[0])
    assert "∇" in model_svg(tr, grads=grads)[0]            # ∇ badges drawn with grads
    assert "∇" not in model_svg(tr)[0]                     # back-compat: none without grads


def test_band_sequence_from_labels():
    """_band_sequence reads the stage labels into ordered (type, layer_index, stage_index) sublayers."""
    stages = [{"label": "embed"}, {"label": "attn 1"}, {"label": "ffn 1"}, {"label": "attn 2"}]
    assert _band_sequence(stages) == [("attn", 0, 1), ("ffn", 0, 2), ("attn", 1, 3)]


def test_grad_normalizers_empty():
    """_grad_normalizers returns zeros when there are no gradients (the no-overlay path)."""
    assert _grad_normalizers(None) == (0.0, 0.0, 0.0)


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
