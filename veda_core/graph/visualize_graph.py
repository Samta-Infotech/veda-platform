#!/usr/bin/env python3
"""
graph/visualize_graph.py — Phase 5: interactive visualization of the unified graph.

Produces a SELF-CONTAINED, offline artifacts/unified_graph.html:
  • zoom + pan
  • search (highlight by name)
  • filter by node type (checkboxes)
  • click a node → highlight its edges + neighbours
  • each node type a distinct colour

Dependency policy (matches CLAUDE.md "no new deps" + on-prem "no external"):
  • If `pyvis` is installed, use it (richest UX).
  • Otherwise fall back to a zero-dependency, fully self-contained HTML (layout precomputed
    in Python; rendered with inline vanilla-JS canvas — NO CDN, NO external assets).

Usage:
    python3 graph/visualize_graph.py
    python3 graph/visualize_graph.py --include-synonyms   # also draw the 2.8k synonym nodes
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    import config as _cfg
    _GRAPH_FILE = os.path.join(_ROOT, getattr(_cfg, "UNIFIED_GRAPH_FILE", "data/veda_unified_graph.json"))
except Exception:
    _GRAPH_FILE = os.path.join(_ROOT, "data", "veda_unified_graph.json")
_OUT_HTML = os.path.join(_ROOT, "artifacts", "unified_graph.html")

# Per-type colour (also the legend).
_COLORS = {
    "TABLE":     "#2563eb",
    "COLUMN":    "#0891b2",
    "METRIC":    "#16a34a",
    "DIMENSION": "#9333ea",
    "CONCEPT":   "#ea580c",
    "SYNONYM":   "#94a3b8",
    "VALUE":     "#d97706",
    "ENTITY":    "#dc2626",
}


def _layout(nodes: List[dict]) -> Dict[str, Dict[str, float]]:
    """Deterministic clustered layout: each node type gets a region on a circle; nodes
    within a type fan out on a spiral. No deps, no live simulation → fast + stable."""
    by_type: Dict[str, List[dict]] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)
    types = sorted(by_type)
    pos: Dict[str, Dict[str, float]] = {}
    R = 1100.0
    for ti, ty in enumerate(types):
        ang = 2 * math.pi * ti / max(1, len(types))
        cx, cy = R * math.cos(ang), R * math.sin(ang)
        group = sorted(by_type[ty], key=lambda n: n["id"])
        for i, n in enumerate(group):
            r = 12 * math.sqrt(i + 1)
            a = i * 2.399963          # golden-angle spiral
            pos[n["id"]] = {"x": cx + r * math.cos(a), "y": cy + r * math.sin(a)}
    return pos


def _build_self_contained_html(graph: Dict[str, Any], include_synonyms: bool) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not include_synonyms:
        nodes = [n for n in nodes if n["type"] != "SYNONYM"]
        keep = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in keep and e["target"] in keep]

    pos = _layout(nodes)
    payload = {
        "nodes": [{"id": n["id"], "t": n["type"], "n": n["name"],
                   "x": round(pos[n["id"]]["x"], 1), "y": round(pos[n["id"]]["y"], 1)}
                  for n in nodes],
        "edges": [{"s": e["source"], "g": e["target"], "t": e["type"]} for e in edges],
        "colors": _COLORS,
    }
    data_json = json.dumps(payload)
    stats = graph.get("stats", {})
    legend = "".join(
        f'<label><input type="checkbox" class="tf" value="{t}" checked> '
        f'<span class="sw" style="background:{c}"></span>{t}</label>'
        for t, c in _COLORS.items() if any(n["type"] == t for n in nodes)
    )

    # NOTE: the JS is inline + self-contained (no CDN). {{ }} escape literal braces.
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>VEDA Unified Knowledge Graph</title>
<style>
 html,body{{margin:0;height:100%;font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0}}
 #bar{{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:#1e293b;z-index:10;
   display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #334155}}
 #bar h1{{font-size:14px;margin:0;font-weight:600}}
 #bar .meta{{color:#94a3b8;font-size:12px}}
 #search{{padding:5px 8px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#e2e8f0}}
 .legend{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px}}
 .legend label{{display:flex;gap:4px;align-items:center;cursor:pointer}}
 .sw{{width:11px;height:11px;border-radius:2px;display:inline-block}}
 #cv{{position:fixed;top:0;left:0}}
 #tip{{position:fixed;pointer-events:none;background:#020617;border:1px solid #475569;
   padding:6px 9px;border-radius:6px;font-size:12px;display:none;max-width:340px;z-index:20}}
</style></head><body>
<div id="bar">
  <h1>VEDA Unified Knowledge Graph</h1>
  <span class="meta">{stats.get('nodes','?')} nodes · {stats.get('edges','?')} edges
   · scroll=zoom · drag=pan · click=highlight</span>
  <input id="search" placeholder="search name…">
  <div class="legend">{legend}</div>
</div>
<canvas id="cv"></canvas><div id="tip"></div>
<script>
const DATA = {data_json};
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
let view={{x:0,y:0,s:0.45}},nodes=DATA.nodes,edges=DATA.edges,colors=DATA.colors;
const byId={{}};nodes.forEach(n=>byId[n.id]=n);
let active=new Set(Object.keys(colors)),sel=null,q="";
function resize(){{cv.width=innerWidth;cv.height=innerHeight;draw();}}
addEventListener('resize',resize);
function vis(n){{return active.has(n.t);}}
function neigh(id){{const s=new Set([id]);edges.forEach(e=>{{if(e.s===id)s.add(e.g);if(e.g===id)s.add(e.s);}});return s;}}
function draw(){{
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);
 ctx.save();ctx.translate(cv.width/2+view.x,cv.height/2+view.y);ctx.scale(view.s,view.s);
 const hl=sel?neigh(sel):null;
 ctx.lineWidth=0.6/view.s;
 edges.forEach(e=>{{const a=byId[e.s],b=byId[e.g];if(!a||!b||!vis(a)||!vis(b))return;
   const on=hl&&(e.s===sel||e.g===sel);ctx.strokeStyle=on?'#fbbf24':'rgba(148,163,184,0.13)';
   ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}});
 nodes.forEach(n=>{{if(!vis(n))return;
   const dim=(hl&&!hl.has(n.id))||(q&&!n.n.toLowerCase().includes(q));
   ctx.globalAlpha=dim?0.12:1;ctx.fillStyle=colors[n.t]||'#999';
   const r=(n.t==='TABLE'?6:n.t==='CONCEPT'?6:3.5);
   ctx.beginPath();ctx.arc(n.x,n.y,r,0,7);ctx.fill();
   if((view.s>0.9||n.t==='TABLE'||n.t==='CONCEPT')&&!dim){{ctx.globalAlpha=1;ctx.fillStyle='#e2e8f0';
     ctx.font=(9/view.s>3?9:9)+'px sans-serif';ctx.fillText(n.n.split('.').pop(),n.x+r+1,n.y+3);}}
 }});ctx.globalAlpha=1;ctx.restore();
}}
function pick(mx,my){{const x=(mx-cv.width/2-view.x)/view.s,y=(my-cv.height/2-view.y)/view.s;
 let best=null,bd=12/view.s;nodes.forEach(n=>{{if(!vis(n))return;const d=Math.hypot(n.x-x,n.y-y);
 if(d<bd){{bd=d;best=n;}}}});return best;}}
let drag=null;
cv.addEventListener('mousedown',e=>drag={{x:e.clientX,y:e.clientY,vx:view.x,vy:view.y,moved:false}});
addEventListener('mouseup',e=>{{if(drag&&!drag.moved){{const n=pick(e.clientX,e.clientY);sel=n?n.id:null;draw();}}drag=null;}});
addEventListener('mousemove',e=>{{
 if(drag){{view.x=drag.vx+(e.clientX-drag.x);view.y=drag.vy+(e.clientY-drag.y);
   if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;draw();return;}}
 const n=pick(e.clientX,e.clientY);
 if(n){{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
   tip.innerHTML='<b>'+n.t+'</b><br>'+n.n;}}else tip.style.display='none';}});
cv.addEventListener('wheel',e=>{{e.preventDefault();const f=e.deltaY<0?1.1:0.9;
 view.s=Math.max(0.05,Math.min(6,view.s*f));draw();}},{{passive:false}});
document.getElementById('search').addEventListener('input',e=>{{q=e.target.value.toLowerCase();draw();}});
document.querySelectorAll('.tf').forEach(c=>c.addEventListener('change',()=>{{
 active=new Set([...document.querySelectorAll('.tf:checked')].map(x=>x.value));draw();}}));
resize();
</script></body></html>"""


def visualize(include_synonyms: bool = False) -> str:
    try:
        with open(_GRAPH_FILE) as f:
            graph = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot read unified graph ({e}); run unified_graph_builder.py first")

    os.makedirs(os.path.dirname(_OUT_HTML), exist_ok=True)

    # Prefer pyvis if available; else self-contained fallback.
    used = "self-contained (no deps)"
    try:
        from pyvis.network import Network  # noqa: F401
        used = "pyvis"
        html = _build_pyvis_html(graph, include_synonyms)
    except Exception:
        html = _build_self_contained_html(graph, include_synonyms)

    with open(_OUT_HTML, "w") as f:
        f.write(html)
    return f"{_OUT_HTML}  [{used}]"


def _build_pyvis_html(graph: Dict[str, Any], include_synonyms: bool) -> str:
    from pyvis.network import Network
    net = Network(height="100vh", width="100%", bgcolor="#0f172a", font_color="#e2e8f0",
                  notebook=False, directed=True)
    nodes = graph.get("nodes", [])
    if not include_synonyms:
        nodes = [n for n in nodes if n["type"] != "SYNONYM"]
    keep = {n["id"] for n in nodes}
    for n in nodes:
        net.add_node(n["id"], label=n["name"].split(".")[-1], title=f'{n["type"]}: {n["name"]}',
                     color=_COLORS.get(n["type"], "#999"))
    for e in graph.get("edges", []):
        if e["source"] in keep and e["target"] in keep:
            net.add_edge(e["source"], e["target"], title=e["type"])
    net.toggle_physics(True)
    return net.generate_html()


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize the VEDA unified graph.")
    ap.add_argument("--include-synonyms", action="store_true",
                    help="also draw SYNONYM nodes (~2.8k; off by default for clarity)")
    args = ap.parse_args()
    out = visualize(include_synonyms=args.include_synonyms)
    print(f"wrote → {out}")
    print("open it in a browser (fully offline, no external assets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
