"""
flow_component.py - the interactive D3 view of one forward pass.

`flow_html(trace)` returns a self-contained HTML document (D3 from a CDN, no npm) that
renders the JSON produced by `replay_engine.trace_forward`:

    tokens  ->  residual-stream heatmaps (one per stage)  ->  attention grids
            ->  readout (next-token bars / class probs / regression value)

Hovering any cell shows a tooltip. It is dropped into Streamlit with
`st.components.v1.html(flow_html(trace), height=...)`. This is the first cut of the
Figma flow design; it consumes exactly what the replay engine already serializes, so
scrubbing the training timeline re-renders it at any step.
"""
from __future__ import annotations

import json

_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
 body{font-family:Inter,system-ui,-apple-system,sans-serif;margin:0;background:#fff;color:#1f2430}
 .lbl{font-size:10px;fill:#5a6072}
 .title{font-size:13px;font-weight:600;fill:#1f2430}
 .sub{font-size:11px;fill:#5a6072}
 #tt{position:fixed;pointer-events:none;background:#111827;color:#fff;padding:3px 7px;
     border-radius:4px;font-size:11px;opacity:0;transition:opacity .05s;z-index:10;white-space:nowrap}
</style></head>
<body><div id="tt"></div><svg id="svg"></svg>
<script>
const D = __DATA__;
const svg = d3.select("#svg"), tt = d3.select("#tt");
const ML = 130, MT = 28, MR = 24;
const T = D.T, dm = D.d_model;
const tok = D.token_strs, vstr = D.vocab_strs || [];
let y = MT;

function show(html, e){ tt.style("opacity",1).html(html)
  .style("left",(e.clientX+12)+"px").style("top",(e.clientY+12)+"px"); }
function hide(){ tt.style("opacity",0); }
function rowLabel(text){ svg.append("text").attr("x",ML-10).attr("y",y+11)
  .attr("text-anchor","end").attr("class","lbl").text(text); }

svg.append("text").attr("x",ML).attr("y",16).attr("class","title")
   .text("forward pass · " + (D.head||"lm") + " head   ·   tokens: " + tok.join(" "));
y = MT + 8;

// ---- colour-scale legends ----
{
  const g = svg.append("g").attr("transform",`translate(${ML},${y})`);
  const mk = (ox, label, interp, lo, hi) => {
    g.append("text").attr("x",ox).attr("y",9).attr("class","lbl").text(label);
    const x0 = ox + label.length*6 + 8, lw = 70, n = 24;
    for(let i=0;i<n;i++) g.append("rect").attr("x",x0+i*(lw/n)).attr("y",2)
      .attr("width",lw/n+0.6).attr("height",8).attr("fill",interp(i/(n-1)));
    g.append("text").attr("x",x0-3).attr("y",9).attr("text-anchor","end").attr("class","lbl").text(lo);
    g.append("text").attr("x",x0+lw+3).attr("y",9).attr("class","lbl").text(hi);
  };
  mk(0, "residual / FFN", t => d3.interpolateRdBu(1-t), "−", "+");
  mk(250, "attention", d3.interpolateViridis, "0", "1");
  y += 26;
}

// ---- residual-stream heatmaps, one per stage (rows = tokens, cols = features) ----
const CW = Math.max(4, Math.min(8, Math.floor(360/dm))), CH = 10;
function diverge(v, mx){ return d3.interpolateRdBu(0.5 - 0.5*(v/(mx||1))); }
D.stages.forEach(st => {
  const R = st.residual;                                  // [T][dm]
  let mx = 1e-9; R.forEach(r=>r.forEach(v=>{ if(Math.abs(v)>mx) mx=Math.abs(v); }));
  rowLabel(st.label);
  const g = svg.append("g").attr("transform",`translate(${ML},${y})`);
  for(let r=0;r<T;r++) for(let c=0;c<dm;c++){
    g.append("rect").attr("x",c*CW).attr("y",r*CH).attr("width",CW-0.4).attr("height",CH-0.4)
     .attr("fill",diverge(R[r][c],mx))
     .on("mousemove",e=>show(`${st.label} · '${tok[r]}' · feat ${c} = ${R[r][c].toFixed(3)}`,e))
     .on("mouseleave",hide);
  }
  y += T*CH + 14;
});

// ---- attention grids: each head shown separately, per attention layer ----
const AC = 13, HGAP = 22;
let attnW = 0;
D.attention.forEach(a => {
  const W = a.weights;                                    // [nh][T][T]
  const nh = W.length;
  rowLabel(`attention ${a.layer+1}`);
  svg.append("text").attr("x",ML).attr("y",y-2).attr("class","sub")
     .text(`${nh} head${nh>1?"s":""} · each grid: row = query, col = key (causal)`);
  const g = svg.append("g").attr("transform",`translate(${ML},${y+8})`);
  const gridW = T*AC + HGAP;
  attnW = Math.max(attnW, nh*gridW);
  for(let h=0; h<nh; h++){
    const ox = h*gridW;
    g.append("text").attr("x",ox).attr("y",-3).attr("class","lbl").text("head " + h);
    for(let i=0;i<T;i++) for(let j=0;j<T;j++){
      g.append("rect").attr("x",ox+j*AC).attr("y",i*AC).attr("width",AC-1).attr("height",AC-1)
       .attr("fill", d3.interpolateViridis(W[h][i][j]))
       .on("mousemove",e=>show(`head ${h} · '${tok[i]}' → '${tok[j]}' : ${W[h][i][j].toFixed(3)}`,e))
       .on("mouseleave",hide);
    }
  }
  y += T*AC + 30;
});

// ---- Q · K · V per head (the queries / keys / values feeding each attention layer) ----
D.attention.forEach(a => {
  if(!a.q) return;
  const nh = a.q.length, dk = a.q[0][0].length;
  const QC = 6, mini = dk*QC, sub = mini + 12, pw = 3*sub + 18;
  rowLabel(`attn ${a.layer+1} Q·K·V`);
  const g = svg.append("g").attr("transform",`translate(${ML},${y+12})`);
  attnW = Math.max(attnW, nh*pw);
  for(let h=0; h<nh; h++){
    const base = h*pw;
    g.append("text").attr("x",base).attr("y",-2).attr("class","lbl").text("head " + h);
    [["Q",a.q[h]],["K",a.k[h]],["V",a.v[h]]].forEach((it, qi) => {
      const name = it[0], M = it[1];                       // [T][dk]
      const ox = base + qi*sub;
      g.append("text").attr("x",ox).attr("y",10).attr("class","lbl").text(name);
      let mx = 1e-9; M.forEach(r=>r.forEach(v=>{ if(Math.abs(v)>mx) mx = Math.abs(v); }));
      for(let r=0;r<T;r++) for(let c=0;c<dk;c++){
        g.append("rect").attr("x",ox+c*QC).attr("y",14+r*QC).attr("width",QC-0.4).attr("height",QC-0.4)
         .attr("fill",diverge(M[r][c],mx))
         .on("mousemove",e=>show(`head ${h} ${name} · '${tok[r]}' · dim ${c} = ${M[r][c].toFixed(3)}`,e))
         .on("mouseleave",hide);
      }
    });
  }
  y += T*QC + 30;
});

// ---- feed-forward hidden activations (the FFN's internal "capacity" layer) ----
let denseW = 0;
if(D.dense && D.dense.length){
  let maxH = 1; D.dense.forEach(d=>{ if(d.hidden[0]) maxH = Math.max(maxH, d.hidden[0].length); });
  const FW = Math.max(3, Math.min(7, Math.floor(380/maxH))), FH = 10;
  D.dense.forEach(d => {
    const H = d.hidden; const hid = H[0] ? H[0].length : 0;
    let mx = 1e-9; H.forEach(r=>r.forEach(v=>{ if(Math.abs(v)>mx) mx = Math.abs(v); }));
    rowLabel(`ffn ${d.layer+1} hidden`);
    svg.append("text").attr("x",ML).attr("y",y-2).attr("class","sub").text(`${hid} units`);
    const g = svg.append("g").attr("transform",`translate(${ML},${y+4})`);
    denseW = Math.max(denseW, hid*FW);
    for(let r=0;r<T;r++) for(let c=0;c<hid;c++){
      g.append("rect").attr("x",c*FW).attr("y",r*FH).attr("width",FW-0.4).attr("height",FH-0.4)
       .attr("fill",diverge(H[r][c],mx))
       .on("mousemove",e=>show(`ffn ${d.layer+1} · '${tok[r]}' · unit ${c} = ${H[r][c].toFixed(3)}`,e))
       .on("mouseleave",hide);
    }
    y += T*FH + 18;
  });
}

// ---- readout ----
rowLabel("output");
if(D.head === "regression"){
  svg.append("text").attr("x",ML).attr("y",y+12).attr("class","sub")
     .text("predicted value: " + (D.output ? D.output.map(v=>v.toFixed(3)).join(", ") : "—"));
  y += 26;
} else {
  const probs = D.head === "classify" ? D.probs : (D.probs ? D.probs[T-1] : []);
  const labels = D.head === "classify"
     ? probs.map((_,i)=>"class "+i)
     : (vstr.length ? vstr : probs.map((_,i)=>""+i));
  const BW = Math.max(16, Math.min(40, Math.floor(420/Math.max(1,probs.length)))), BH = 70;
  const top = probs.indexOf(Math.max(...probs));
  const g = svg.append("g").attr("transform",`translate(${ML},${y})`);
  svg.append("text").attr("x",ML).attr("y",y-2).attr("class","sub")
     .text(D.head==="classify" ? "class probabilities" : "next-token probability (last position)");
  probs.forEach((p,i)=>{
    const h = Math.max(1, p*BH);
    g.append("rect").attr("x",i*BW).attr("y",BH-h).attr("width",BW-3).attr("height",h)
     .attr("fill", i===top ? "#16a34a" : "#cbd2dc")
     .on("mousemove",e=>show(`${labels[i]} : ${p.toFixed(3)}`,e)).on("mouseleave",hide);
    g.append("text").attr("x",i*BW+(BW-3)/2).attr("y",BH+11).attr("text-anchor","middle")
     .attr("class","lbl").text(labels[i]);
  });
  y += BH + 24;
}

// size the svg to fit
const wide = ML + Math.max(dm*CW, attnW, denseW, 460) + MR;
svg.attr("width", wide).attr("height", y + 10);
</script></body></html>"""


def flow_html(trace: dict) -> str:
    """Self-contained HTML rendering of one `trace_forward(...)` result."""
    # escape "<" so a vocab token containing "</script>" can't break out of the <script> block
    return _TEMPLATE.replace("__DATA__", json.dumps(trace).replace("<", "\\u003c"))


def component_height(trace: dict) -> int:
    """A reasonable iframe height for `st.components.v1.html` given this trace."""
    T = trace.get("T", 8)
    stages = len(trace.get("stages", []))
    attn = len(trace.get("attention", []))
    dense = len(trace.get("dense", []))
    return int(60 + stages * (T * 10 + 14) + attn * (T * 16 + 24 + T * 6 + 30)
               + dense * (T * 10 + 22) + 150)
