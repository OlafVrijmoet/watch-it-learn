"""app_charts — pure matplotlib chart helpers for the Builder page.

Extracted from builder_app.py so they're unit-testable on their own (no Streamlit AppTest needed).
Each builder returns a Figure; `_show` renders it into a target then closes it.
"""
import matplotlib.pyplot as plt


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
