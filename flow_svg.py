"""
flow_svg.py - the live, Figma-matching SVG renderer for the transformer flow.

Generates the locked visual language from the Figma "Model Builder — ALL EXPANDED" frame as
precise SVG (full coordinate control):
  * embeddings / residual vectors as TALL colored arrays (one column per token), red=+ / blue=−
  * Q / K / V as colored collectors (blue / teal / orange)
  * the attention score grid as a dark navy panel, brighter cell = more weight
  * the FFN as node rows; the gradient overlay as heat-tinted ∇ badges + per-neuron halos

`model_svg(trace, readout_pos, grads)` renders the whole stack; `flow_svg_component` wraps it in a
zoomable/scrollable Streamlit iframe; `replay_html` plays the per-checkpoint frames. Being plain SVG it
rasterizes to PNG via cairosvg (`render_png`) for browser-free self-verification.
"""
from __future__ import annotations

import html

# locked colors
Q_COL, K_COL, V_COL = "#2563eb", "#14b8a6", "#f97316"   # query / key / value
INK, MUTED = "#1f2430", "#5a6072"


def _esc(s) -> str:
    return html.escape(str(s))


def div_color(v: float, mx: float) -> str:
    """Diverging red(+)/blue(−) like the Figma vectors; white near zero."""
    if mx <= 0:
        return "#f5f6f8"
    t = max(-1.0, min(1.0, v / mx))
    if t >= 0:                                   # white -> red
        r, g, b = 255, int(255 - 165 * t), int(255 - 175 * t)
    else:                                        # white -> blue
        t = -t
        r, g, b = int(255 - 218 * t), int(255 - 158 * t), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def grid_color(w: float) -> str:
    """Attention weight on the dark navy grid: navy (0) -> bright amber (1)."""
    t = max(0.0, min(1.0, w))
    bg = (30, 27, 75)                            # navy #1e1b4b
    fg = (253, 224, 71)                          # amber #fde047
    r, g, b = (int(bg[i] + (fg[i] - bg[i]) * t) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _vector(x, y, vals, cw, ch, mx, outline="#cbd2dc", tip=""):
    """A tall colored vector (column of cells), top-left at (x,y). Returns SVG string."""
    out = [f'<g>']
    for i, v in enumerate(vals):
        title = f"{tip} [{i}] = {v:.3f}" if tip else f"{v:.3f}"
        out.append(
            f'<rect class="cell" x="{x:.1f}" y="{y + i*ch:.1f}" width="{cw}" height="{ch}" '
            f'fill="{div_color(v, mx)}" data-tip="{_esc(title)}"/>')
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw}" height="{ch*len(vals):.1f}" '
               f'fill="none" stroke="{outline}" stroke-width="0.8"/>')
    out.append('</g>')
    return "".join(out)


def _scurve(x1, y1, x2, y2, color, w=1.4, op=0.85):
    """Vertical S-curve: leaves downward, arrives aligned (no hard turns)."""
    dy = (y2 - y1) * 0.5
    return (f'<path d="M{x1:.1f} {y1:.1f} C {x1:.1f} {y1+dy:.1f}, {x2:.1f} {y2-dy:.1f}, '
            f'{x2:.1f} {y2:.1f}" fill="none" stroke="{color}" stroke-width="{w}" '
            f'stroke-opacity="{op}"/>')


def _text(x, y, s, size=11, col=MUTED, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="Inter,system-ui,sans-serif">{_esc(s)}</text>')


def _dotted_section(x0, x1, y, label):
    return (f'<line x1="{x0:.0f}" y1="{y:.0f}" x2="{x1:.0f}" y2="{y:.0f}" stroke="#cbd2dc" '
            f'stroke-width="1" stroke-dasharray="3 4"/>' + _text(x0, y - 5, label, 10, MUTED, "start", "700"))


def _heat(t):
    """Light gray (t=0) → strong red (t=1) hex, for gradient-magnitude tinting."""
    t = max(0.0, min(1.0, t))
    a, b = (0xe5, 0xe7, 0xeb), (0xdc, 0x26, 0x26)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _grad_badge(xr, y, val, vmax):
    """A heat-tinted '∇ val' pill right-aligned at xr on a section header — a block's gradient size."""
    t = val / vmax if vmax > 0 else 0.0
    w = 60
    txt = f"∇ {val:.3f}" if val < 100 else f"∇ {val:.0f}"
    return (f'<rect x="{xr-w:.0f}" y="{y-9:.0f}" width="{w}" height="13" rx="3" fill="{_heat(t)}" '
            f'stroke="#dc2626" stroke-width="0.5"/>'
            + _text(xr - w / 2, y + 1, txt, 8.5, "#fff" if t > 0.5 else "#7f1d1d", "middle", "700"))


def _node_row(cx, y, vals, mx, *, row_w=70, cap=14, r=None, gvals=None, gmax=1.0):
    """A horizontal row of nodes centered at cx, colored by value. Shows all values when
    `cap` is None (or n<=cap); dot radius adapts to the count. If `gvals` (per-unit gradients)
    is given, each node gets a red ∇ halo sized by its gradient. Returns (svg, [(x,y)...])."""
    n = len(vals)
    idx = list(range(n)) if (cap is None or n <= cap) else [round(k * (n - 1) / (cap - 1)) for k in range(cap)]
    m = len(idx)
    rad = r if r is not None else max(1.0, min(2.6, row_w / max(2, m) * 0.5))
    xs = [cx - row_w / 2 + (row_w * k / (m - 1) if m > 1 else row_w / 2) for k in range(m)]
    halos = ""
    if gvals is not None and gmax > 0:
        halos = "".join(
            f'<circle cx="{xs[k]:.1f}" cy="{y:.1f}" r="{rad + 1 + 5 * (gvals[idx[k]] / gmax):.1f}" '
            f'fill="#dc2626" opacity="{0.10 + 0.55 * (gvals[idx[k]] / gmax):.2f}"/>'
            for k in range(m) if idx[k] < len(gvals))
    svg = halos + "".join(
        f'<circle class="cell" cx="{xs[k]:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{div_color(vals[idx[k]], mx)}" '
        f'stroke="#cbd2dc" stroke-width="0.3" data-tip="{vals[idx[k]]:.3f}'
        + (f"  ∇ {gvals[idx[k]]:.3f}" if (gvals is not None and idx[k] < len(gvals)) else "")
        + f'"/>' for k in range(m))
    return svg, [(xs[k], y) for k in range(m)]


def model_svg(trace: dict, readout_pos=None, grads=None) -> tuple[str, int, int]:
    """The whole model as a vertical stack: residual-stream rows (tall vectors) with an
    ATTENTION band or FEED-FORWARD band between each, for ANY number/order of sublayers
    (derived from the trace's stage labels). Ends in pooling → unembed → next-token."""
    toks = trace["token_strs"]
    T = len(toks)
    stages = trace["stages"]
    dm = len(stages[0]["residual"][0])
    amx = lambda M: max((abs(v) for r in M for v in r), default=1.0)

    gvmax = ghmax = gfnmax = 0.0                             # gradient overlay normalizers
    if grads:
        gvmax = max([grads["embed"], grads["head"]] + [a["all"] for a in grads["attn"]]
                    + [f["all"] for f in grads["ffn"]] + [1e-9])
        ghmax = max([1e-9] + [hh[p] for a in grads["attn"] for hh in a.get("heads", [])
                              for p in ("q", "k", "v")])
        gfnmax = max([1e-9] + [u for f in grads["ffn"] for ln in f.get("neurons", []) for u in ln])

    subs = []                                                # (type, layer_index, stage_index)
    ai = di = 0
    for k in range(1, len(stages)):
        if "attn" in stages[k]["label"].lower():
            subs.append(("attn", ai, k)); ai += 1
        else:
            subs.append(("ffn", di, k)); di += 1
    nh_max = max((len(a["weights"]) for a in trace["attention"]), default=1)
    qk_mx = max((abs(v) for a in trace["attention"] for M in (a["q"], a["k"], a["v"])
                 for hh in M for r in hh for v in r), default=1.0)
    head_cols = [Q_COL, V_COL, K_COL, "#a855f7", "#e11d48", "#0891b2"]

    gcell = 24
    grid = T * gcell
    kv_cw, kv_ch = 11, 7
    gut = kv_cw + 30                                          # wide gutters so Q/V curves route around
    Hw = gut + grid + gut
    head_gap = 66
    heads_w_max = nh_max * Hw + (nh_max - 1) * head_gap       # widest band sets the frame width
    PADL, PADR = 124, 56
    W = max(900, PADL + PADR + heads_w_max + 40)
    mL, mR = PADL, W - PADR
    colx = [mL + 12 + (mR - mL - 24) * (i / (T - 1) if T > 1 else 0.5) for i in range(T)]
    emb_cw, emb_ch = 14, 4
    emb_h = emb_ch * dm

    s = ['<g>']
    s.append(_text(PADL, 26, "◆ Model — residual stream flows top → bottom (build by adding sections)", 14, INK, weight="700"))

    def res_row(y, vals, mx, prefix, labeltok=False):
        for i in range(T):
            s.append(_vector(colx[i], y, vals[i], emb_cw, emb_ch, mx, tip=f"{prefix} '{toks[i]}'"))
            if labeltok:
                s.append(_text(colx[i] + emb_cw / 2, y - 4, toks[i], 9, INK, "middle", "600"))

    y = 70
    s.append(_dotted_section(20, mR, y - 14, "EMBEDDINGS"))
    if grads:
        s.append(_grad_badge(mR, y - 14, grads["embed"], gvmax))
    res_row(y, stages[0]["residual"], amx(stages[0]["residual"]), "emb", labeltok=True)
    yb = y + emb_h

    for (typ, idx, k) in subs:
        out_vals = stages[k]["residual"]
        out_mx = amx(out_vals)
        if typ == "attn" and trace["attention"]:
            a = trace["attention"][idx]
            Wts, Ka, Qa, Va = a["weights"], a["k"], a["q"], a["v"]
            nh_b = len(Wts)                                   # this layer's head count
            dk_b = len(Qa[0][0])
            kv_h = kv_ch * dk_b
            heads_w_b = nh_b * Hw + (nh_b - 1) * head_gap
            hstart_b = (mL + mR - heads_w_b) / 2
            hx = [hstart_b + h * (Hw + head_gap) for h in range(nh_b)]
            gxs = [x + gut for x in hx]
            y_hub = yb + 56                                   # collector hubs (Q/K/V)
            y_ktop = y_hub + 40                               # K row on top of grid
            y_grid = y_ktop + kv_h + 22                       # score grid
            y_junc = y_grid + grid + 44                       # junction dots (weights + values)
            y_out = y_junc + 66                               # new embeddings (attention output)
            s.append(_dotted_section(20, mR, yb + 16, f"ATTENTION · {nh_b} heads × d_head {dk_b} · weights × V → new embeddings"))
            if grads and idx < len(grads["attn"]):
                s.append(_grad_badge(mR, yb + 16, grads["attn"][idx]["all"], gvmax))
            for h in range(nh_b):
                Qcx, Kcx, Vcx = gxs[h], gxs[h] + grid / 2, gxs[h] + grid
                # embeddings (above) gather into the 3 colored collectors (Q blue / K teal / V orange)
                for i in range(T):
                    sx = colx[i] + emb_cw / 2
                    s.append(_scurve(sx, yb, Qcx, y_hub, Q_COL, w=0.5, op=0.13))
                    s.append(_scurve(sx, yb, Kcx, y_hub, K_COL, w=0.5, op=0.13))
                    s.append(_scurve(sx, yb, Vcx, y_hub, V_COL, w=0.5, op=0.13))
                for cxh, col, nm in ((Qcx, Q_COL, "Q"), (Kcx, K_COL, "K"), (Vcx, V_COL, "V")):
                    r = 3.0                                    # per-head collector dot, sized by ∇ if shown
                    if grads and idx < len(grads["attn"]) and grads["attn"][idx].get("heads"):
                        hv = grads["attn"][idx]["heads"][h][nm.lower()]
                        r = 3.0 + 7.0 * (hv / ghmax if ghmax > 0 else 0.0)
                    s.append(f'<circle cx="{cxh:.1f}" cy="{y_hub:.1f}" r="{r:.1f}" fill="{col}"/>')
                # K → top of grid (columns)
                for j in range(T):
                    kx = gxs[h] + j * gcell + (gcell - kv_cw) / 2
                    s.append(_scurve(Kcx, y_hub, kx + kv_cw / 2, y_ktop, K_COL, w=0.8, op=0.45))
                    s.append(_vector(kx, y_ktop, Ka[h][j], kv_cw, kv_ch, qk_mx, outline=K_COL, tip=f"K{h} '{toks[j]}'"))
                # Q → left of grid (rows): loop out left then into Q
                for i in range(T):
                    qx = hx[h] + (gut - kv_cw) / 2
                    qy = y_grid + i * gcell + (gcell - kv_h) / 2
                    s.append(_scurve(Qcx, y_hub, qx + kv_cw / 2, qy, Q_COL, w=0.8, op=0.45))
                    s.append(_vector(qx, qy, Qa[h][i], kv_cw, kv_ch, qk_mx, outline=Q_COL, tip=f"Q{h} '{toks[i]}'"))
                # V → right of grid: loop out right then into V
                vx = gxs[h] + grid + (gut - kv_cw) / 2
                for i in range(T):
                    vy = y_grid + i * gcell + (gcell - kv_h) / 2
                    s.append(_scurve(Vcx, y_hub, vx + kv_cw / 2, vy, V_COL, w=0.8, op=0.45))
                    s.append(_vector(vx, vy, Va[h][i], kv_cw, kv_ch, qk_mx, outline=V_COL, tip=f"V{h} '{toks[i]}'"))
                # score grid
                s.append(f'<rect x="{gxs[h]:.1f}" y="{y_grid:.1f}" width="{grid}" height="{grid}" fill="#1e1b4b" rx="2"/>')
                for i in range(T):
                    for j in range(i + 1):
                        s.append(f'<rect class="cell" x="{gxs[h]+j*gcell:.1f}" y="{y_grid+i*gcell:.1f}" '
                                 f'width="{gcell-1}" height="{gcell-1}" fill="{grid_color(Wts[h][i][j])}" '
                                 f'data-tip="h{h}: \'{_esc(toks[i])}\'&#8594;\'{_esc(toks[j])}\' {Wts[h][i][j]:.3f}"/>')
                s.append(_text(gxs[h] + grid / 2, y_grid - kv_h - 12, f"head {h}", 9, INK, "middle", "700"))
                # weights (grid) and values (V) each flow DOWN via a junction → new embeddings
                gj_x, vj_x = gxs[h] + grid / 2, vx + kv_cw / 2
                s.append(_scurve(gj_x, y_grid + grid, gj_x, y_junc, "#1e1b4b", w=1.2, op=0.7))
                s.append(_scurve(vx + kv_cw / 2, y_grid + grid, vj_x, y_junc, V_COL, w=1.2, op=0.7))
                s.append(f'<circle cx="{gj_x:.1f}" cy="{y_junc:.1f}" r="3.4" fill="#1e1b4b"/>')
                s.append(f'<circle cx="{vj_x:.1f}" cy="{y_junc:.1f}" r="3.4" fill="{V_COL}"/>')
                for i in range(T):                            # junction → every new embedding
                    nx = colx[i] + emb_cw / 2
                    s.append(_scurve(gj_x, y_junc, nx, y_out, "#5b5ba8", w=0.7, op=0.32))
                    s.append(_scurve(vj_x, y_junc, nx, y_out, V_COL, w=0.7, op=0.30))
            s.append(_text(gxs[0], y_junc + 4, "weights × V → new embeddings", 9, MUTED, "start"))
            res_row(y_out, out_vals, out_mx, "after attn")
            yb = y_out + emb_h
        else:                                                 # FFN band (one row per hidden layer)
            hiddens = trace["ffn"][idx].get("hiddens") or [trace["ffn"][idx]["hidden"]]
            L = len(hiddens)
            hmx = max((amx(h) for h in hiddens), default=1.0)
            units = len(hiddens[0][0]) if hiddens and hiddens[0] else 0
            lay_gap = 42
            y_lay = [yb + 60 + li * lay_gap for li in range(L)]
            y_out = y_lay[-1] + 60
            s.append(_dotted_section(20, mR, yb + 16,
                     f"FEED-FORWARD · {L} hidden layer{'s' if L > 1 else ''} × {units} units (all shown)"))
            if grads and idx < len(grads["ffn"]):
                s.append(_grad_badge(mR, yb + 16, grads["ffn"][idx]["all"], gvmax))

            def _samp(nodes, cap=16):                          # sample positions for faint edges
                k = len(nodes)
                return nodes if k <= cap else [nodes[round(t * (k - 1) / (cap - 1))] for t in range(cap)]

            fgrad = grads["ffn"][idx]["neurons"] if (grads and idx < len(grads["ffn"])) else None
            for i in range(T):
                cxc = colx[i] + emb_cw / 2
                prev = [(cxc, yb)]                             # input = the residual point above
                for li in range(L):
                    gv = fgrad[li] if (fgrad and li < len(fgrad)) else None
                    hsvg, hpos = _node_row(cxc, y_lay[li], hiddens[li][i], hmx,
                                           row_w=112, cap=64, gvals=gv, gmax=gfnmax)
                    for (ax, ay) in _samp(prev):
                        for (bx, by) in _samp(hpos):
                            s.append(_scurve(ax, ay, bx, by, "#94a3b8", w=0.35, op=0.12))
                    s.append(hsvg)
                    prev = hpos
                for (ax, ay) in _samp(prev):                   # last hidden → residual out
                    s.append(_scurve(ax, ay, cxc, y_out, "#94a3b8", w=0.35, op=0.14))
            res_row(y_out, out_vals, out_mx, "refined")
            yb = y_out + emb_h

    # output tail: prediction at the chosen position → unembed → next-token bars
    final = stages[-1]["residual"]
    rp = T - 1 if readout_pos is None else max(0, min(T - 1, int(readout_pos)))
    ap = trace.get("probs", None)
    probs = ap[rp] if (ap and isinstance(ap[0], list)) else (ap or [1.0])
    vstr = trace.get("vocab_strs", [str(i) for i in range(len(probs))])
    s.append(_dotted_section(20, mR, yb + 14,
             f"OUTPUT · prediction after position {rp} (token '{trace['token_strs'][rp]}') → next token"))
    if grads:
        s.append(_grad_badge(mR, yb + 14, grads["head"], gvmax))
    y_pool = yb + 44
    px = (mL + mR) / 2
    s.append(_scurve(colx[rp] + emb_cw / 2, yb, px, y_pool, "#16a34a", w=1.3, op=0.8))
    s.append(_vector(px - emb_cw / 2, y_pool, final[rp], emb_cw, emb_ch, amx(final), tip=f"position {rp}"))
    y_bars = y_pool + emb_h + 24
    V = len(probs)
    bw = min(34, (mR - mL) / max(1, V))
    bx0 = (mL + mR - bw * V) / 2
    top = probs.index(max(probs)) if probs else 0
    BH = 50
    for kk in range(V):
        hbar = max(1.0, probs[kk] * BH)
        s.append(f'<rect x="{bx0+kk*bw:.1f}" y="{y_bars+BH-hbar:.1f}" width="{bw-3:.1f}" height="{hbar:.1f}" '
                 f'fill="{"#16a34a" if kk==top else "#cbd2dc"}" data-tip="{_esc(vstr[kk])}: {probs[kk]:.3f}"/>')
        s.append(_text(bx0 + kk * bw + (bw - 3) / 2, y_bars + BH + 11, vstr[kk], 9, MUTED, "middle"))
    s.append(_text(px, y_bars - 6, f"next token: '{vstr[top]}'", 10, INK, "middle", "600"))

    s.append('</g>')
    return "".join(s), int(W), int(y_bars + BH + 40)


def svg_document(inner: str, width: int, height: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="#ffffff"/>'
            f'{inner}</svg>')


def replay_html(frames, width, height):
    """Client-side playback of training-checkpoint renders. `frames` = list of
    (step:int, loss:float, acc:float, svg:str) (all same size). Returns (html, height).
    A ▶/⏸ button auto-advances through the steps; a slider scrubs; speed selector; hover tips."""
    n = len(frames)
    divs = []
    for k, (stp, loss, acc, svg) in enumerate(frames):
        divs.append(f'<div class="frame" data-step="{stp}" data-loss="{loss:.3f}" data-acc="{acc:.2f}" '
                    f'style="display:{"block" if k == 0 else "none"}">{svg}</div>')
    losses = [round(f[1], 4) for f in frames]
    accs = [round(f[2], 4) for f in frames]
    chart = (
        '<div style="display:flex;gap:8px;border-bottom:1px solid #eee;padding:2px 4px">'
        '<svg id="lchart" width="186" height="72"><line x1="8" y1="62" x2="178" y2="62" stroke="#e5e7eb"/>'
        '<polyline id="llg" fill="none" stroke="#fecaca" stroke-width="1.4"/>'
        '<polyline id="ll" fill="none" stroke="#dc2626" stroke-width="1.8"/>'
        '<circle id="dl" r="2.8" fill="#dc2626"/>'
        '<text x="8" y="12" font-size="9" fill="#dc2626">loss</text>'
        '<text id="cll" x="34" y="12" font-size="9" fill="#555"></text></svg>'
        '<svg id="achart" width="186" height="72"><line x1="8" y1="62" x2="178" y2="62" stroke="#e5e7eb"/>'
        '<polyline id="lag" fill="none" stroke="#bbf7d0" stroke-width="1.4"/>'
        '<polyline id="la" fill="none" stroke="#16a34a" stroke-width="1.8"/>'
        '<circle id="da" r="2.8" fill="#16a34a"/>'
        '<text x="8" y="12" font-size="9" fill="#16a34a">held-out acc</text>'
        '<text id="cla" x="86" y="12" font-size="9" fill="#555"></text></svg></div>')
    view = min(int(height), 800)
    html = (
        '<!doctype html><html><head><meta charset="utf-8"><style>' + _ZOOM_CSS +
        '</style></head><body>'
        f'<div id="bar"><button id="play">▶ Play training</button>'
        f'<button id="rep">🔁 Repeat: off</button>'
        f'<input id="slider" type="range" min="0" max="{max(0, n - 1)}" value="0"/>'
        f'<span id="lab"></span><select id="spd">'
        f'<option value="700">1×</option><option value="350">2×</option><option value="140">5×</option></select>'
        + _ZOOM_BAR + '</div>' + chart +
        f'<div id="tt"></div><div id="scroll" style="height:{view}px"><div id="wrap">{"".join(divs)}</div></div>'
        '<script>'
        'const F=[...document.querySelectorAll(".frame")],play=document.getElementById("play"),'
        'rep=document.getElementById("rep"),slider=document.getElementById("slider"),'
        'lab=document.getElementById("lab"),spd=document.getElementById("spd");'
        f'const SVGW={int(width)},SVGH={int(height)},VIEW={view};' + _ZOOM_JS +
        f'const LO={losses},AC={accs},NP={n};'
        'const ll=document.getElementById("ll"),la=document.getElementById("la"),'
        'llg=document.getElementById("llg"),lag=document.getElementById("lag"),'
        'dl=document.getElementById("dl"),da=document.getElementById("da"),'
        'cll=document.getElementById("cll"),cla=document.getElementById("cla");'
        'const CW=186,CH=72,pad=10,topm=16,maxL=Math.max(...LO,1e-9);'
        'function CX(k){return pad+(CW-2*pad)*(NP>1?k/(NP-1):0);}'
        'function CYL(v){return CH-pad-(CH-topm-pad)*(v/maxL);}'
        'function CYA(v){return CH-pad-(CH-topm-pad)*v;}'
        'function full(Yf,a){let p="";for(let k=0;k<NP;k++)p+=CX(k).toFixed(1)+","+Yf(a[k]).toFixed(1)+" ";return p;}'
        'llg.setAttribute("points",full(CYL,LO));lag.setAttribute("points",full(CYA,AC));'
        'function drawChart(u){let lp="",ap="";for(let k=0;k<=u;k++){'
        'lp+=CX(k).toFixed(1)+","+CYL(LO[k]).toFixed(1)+" ";ap+=CX(k).toFixed(1)+","+CYA(AC[k]).toFixed(1)+" ";}'
        'll.setAttribute("points",lp);la.setAttribute("points",ap);'
        'dl.setAttribute("cx",CX(u));dl.setAttribute("cy",CYL(LO[u]));'
        'da.setAttribute("cx",CX(u));da.setAttribute("cy",CYA(AC[u]));'
        'cll.textContent=LO[u].toFixed(3);cla.textContent=AC[u].toFixed(2);}'
        'let i=0,timer=null,repeat=false;'
        'function show(k){F[i].style.display="none";i=k;F[i].style.display="block";slider.value=i;'
        'const f=F[i];lab.textContent="step "+f.dataset.step;drawChart(i);bindtips(F[i]);}'
        'function stop(){clearInterval(timer);timer=null;play.textContent="▶ Play training";}'
        'function nxt(){if(i>=F.length-1){if(repeat){show(0);}else{stop();}}else{show(i+1);}}'
        'play.onclick=()=>{if(timer){stop();}else{if(i>=F.length-1)show(0);'
        'play.textContent="⏸ Pause";timer=setInterval(nxt,+spd.value);}};'
        'rep.onclick=()=>{repeat=!repeat;rep.textContent="🔁 Repeat: "+(repeat?"on":"off");};'
        'slider.oninput=()=>{if(timer)stop();show(+slider.value);};'
        'spd.onchange=()=>{if(timer){clearInterval(timer);timer=setInterval(nxt,+spd.value);}};'
        'show(0);'
        '</script></body></html>')
    return html, view + 120


def render_png(svg: str, path: str, scale: float = 1.0):
    """Render an SVG string to PNG (for self-verification). Requires cairosvg."""
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=path, scale=scale)


_ZOOM_CSS = (
    'body{margin:0;font-family:Inter,system-ui,sans-serif;background:#fff}'
    '#bar{position:sticky;top:0;background:#f8fafc;border-bottom:1px solid #e5e7eb;padding:5px 10px;'
    'display:flex;gap:8px;align-items:center;z-index:9;font-size:12px}'
    '#bar button{cursor:pointer;border:1px solid #cbd2dc;background:#fff;border-radius:5px;padding:2px 9px}'
    '#slider{flex:1}#scroll{overflow:auto}#wrap{transform-origin:top left;margin:0 auto}'
    '#tt{position:fixed;pointer-events:none;background:#111827;color:#fff;padding:3px 7px;border-radius:4px;'
    'font-size:11px;opacity:0;z-index:20;white-space:nowrap}'
    '.cell:hover{stroke:#111827 !important;stroke-width:1.2 !important}')

_ZOOM_JS = (
    'const wrap=document.getElementById("wrap"),zlab=document.getElementById("zlab"),'
    'tt=document.getElementById("tt");let z=1;'
    'function setZoom(nz){z=Math.max(0.05,Math.min(3,nz));wrap.style.transform="scale("+z+")";'
    'wrap.style.width=(SVGW*z)+"px";wrap.style.height=(SVGH*z)+"px";'   # layout box = scaled → centers
    'zlab.textContent=Math.round(z*100)+"%";}'
    'function fitZ(){var aw=document.getElementById("scroll").clientWidth-18;'
    'return Math.min(VIEW/SVGH, aw/SVGW);}'                 # fit BOTH height and width
    'document.getElementById("zin").onclick=()=>setZoom(z*1.25);'
    'document.getElementById("zout").onclick=()=>setZoom(z/1.25);'
    'document.getElementById("zfit").onclick=()=>setZoom(fitZ());'
    'setZoom(fitZ());'                                      # default to the centered Fit overview
    'window.addEventListener("resize",()=>setZoom(z));'    # keep layout box correct on resize
    'function bindtips(root){root.querySelectorAll("[data-tip]").forEach(el=>{'
    'el.onmousemove=e=>{tt.style.opacity=1;tt.innerHTML=el.getAttribute("data-tip");'
    'tt.style.left=(e.clientX+12)+"px";tt.style.top=(e.clientY+12)+"px";};'
    'el.onmouseleave=()=>{tt.style.opacity=0;};});}')

_ZOOM_BAR = ('<b>zoom</b><button id="zout">−</button><button id="zfit">Fit (overview)</button>'
             '<button id="zin">+</button><span id="zlab">100%</span>')


def flow_svg_component(trace: dict, readout_pos=None, grads=None):
    """(html, height) for st.components.v1.html — the full multi-block SVG flow in a scrollable,
    zoomable viewport (− / Fit / +) with hover tooltips. Detail is preserved; Fit zooms out to an
    overview of the whole model. `readout_pos` chooses which position's next-token prediction to show;
    `grads` (per-block norms) overlays heat-tinted ∇ badges on each section header."""
    inner, w, h = model_svg(trace, readout_pos, grads)
    svg = svg_document(inner, w, h)
    view = min(h, 820)
    doc = (
        '<!doctype html><html><head><meta charset="utf-8"><style>' + _ZOOM_CSS +
        '</style></head><body>'
        '<div id="bar">' + _ZOOM_BAR + '<span style="color:#5a6072">· scroll to pan · hover for values</span></div>'
        '<div id="tt"></div>'
        f'<div id="scroll" style="height:{view}px"><div id="wrap">{svg}</div></div>'
        f'<script>const SVGW={w},SVGH={h},VIEW={view};' + _ZOOM_JS + 'bindtips(document);'
        '</script></body></html>')
    return doc, view + 46
