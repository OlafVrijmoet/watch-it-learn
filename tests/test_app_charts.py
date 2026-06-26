"""Unit tests for app_charts — the pure matplotlib chart helpers carved out of builder_app.py.

These run WITHOUT the slow Streamlit AppTest: that's the point of extracting them.

Run:  PYTHONPATH=. python tests/test_app_charts.py (from the repo root)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app_charts import _metric_chart, _grad_bars


def test_metric_chart_returns_figure():
    fig = _metric_chart([0, 1, 2], [1.0, 0.5, 0.2], cur=1, color="#dc2626", title="loss")
    assert fig is not None and len(fig.axes) == 1
    plt.close(fig)


def test_metric_chart_overlays_train_series():
    fig = _metric_chart([0, 1], [1.0, 0.4], cur=0, color="#16a34a", title="acc", ys2=[1.1, 0.6])
    assert fig.axes[0].get_legend() is not None      # the train overlay adds a legend
    plt.close(fig)


def test_grad_bars_per_block_and_per_head():
    grads = {"embed": 0.1, "head": 0.2,
             "attn": [{"q": 0.3, "k": 0.2, "v": 0.1, "o": 0.05,
                       "heads": [{"q": 0.3, "k": 0.2, "v": 0.1, "o": 0.05}]}],
             "ffn": [{"all": 0.4, "out": 0.1, "layers": [0.3]}]}
    arch = [{"type": "attn"}, {"type": "ffn"}]
    for fig in (_grad_bars(grads, arch), _grad_bars(grads, arch, per_head=True, log=True)):
        assert fig is not None and len(fig.axes) == 1
        plt.close(fig)


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
