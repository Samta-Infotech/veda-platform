#!/usr/bin/env python3
"""
graph/api.py — Phase 6: read-only HTTP API over the unified knowledge graph.

Endpoints:
    GET /graph/node/{id}              → the node + its metadata
    GET /graph/neighbors/{id}         → [{neighbor, edge_type, node}]
    GET /graph/path?a=...&b=...       → shortest path (list of nodes)
    GET /graph/search?q=...           → nodes whose name matches q (+ resolved columns)

Dependency policy: uses the Python **stdlib** `http.server` (zero new deps) by default, so it
runs anywhere. If FastAPI/uvicorn are installed you can mount `build_fastapi_app()` instead.

The API is strictly READ-ONLY (GET only) — consistent with VEDA's read-only posture.

Usage:
    python3 graph/api.py                 # serve on 127.0.0.1:8077
    python3 graph/api.py --port 9000
    curl 'http://127.0.0.1:8077/graph/search?q=severity'
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from graph.query_graph import get_graph


# ── core handlers (transport-agnostic: return (status, dict)) ────────────────
def _err(msg, code=404):
    return code, {"error": msg}


def handle_node(node_id: str):
    g = get_graph()
    if not g:
        return _err("graph not built", 503)
    n = g.node(node_id)
    return (200, n) if n else _err(f"node not found: {node_id}")


def handle_neighbors(node_id: str):
    g = get_graph()
    if not g:
        return _err("graph not built", 503)
    if node_id not in g.nodes:
        return _err(f"node not found: {node_id}")
    out = [{"neighbor": nb, "edge_type": ty, "node": g.node(nb)}
           for nb, ty in g.get_neighbors(node_id)]
    return 200, {"id": node_id, "count": len(out), "neighbors": out}


def handle_path(a: str, b: str):
    g = get_graph()
    if not g:
        return _err("graph not built", 503)
    path = g.shortest_path(a, b)
    return 200, {"a": a, "b": b, "hops": max(0, len(path) - 1),
                 "path": [g.node(p) for p in path]}


def handle_search(q: str, limit: int = 30):
    g = get_graph()
    if not g:
        return _err("graph not built", 503)
    ql = (q or "").strip().lower()
    if not ql:
        return _err("empty query", 400)
    matches = [n for n in g.nodes.values() if ql in n["name"].lower()][:limit]
    resolved = [g.node(c) for c in g.resolve_term(q)]
    return 200, {"q": q, "match_count": len(matches),
                 "matches": matches, "resolved_columns": resolved,
                 "synonyms": g.get_synonyms(q)}


def route(path: str, qs: dict):
    parts = [p for p in path.split("/") if p]
    # root / health → a friendly index so the bare URL is not a 404
    if not parts or parts == ["graph"]:
        return 200, {
            "service": "VEDA Unified Graph API (read-only)",
            "endpoints": [
                "/graph/node/{id}        e.g. /graph/node/table:incident",
                "/graph/neighbors/{id}   e.g. /graph/neighbors/table:incident",
                "/graph/path?a=&b=       e.g. /graph/path?a=syn:priority&b=table:incident",
                "/graph/search?q=        e.g. /graph/search?q=severity",
            ],
        }
    if len(parts) >= 2 and parts[0] == "graph":
        sub = parts[1]
        if sub == "node" and len(parts) >= 3:
            return handle_node(unquote("/".join(parts[2:])))
        if sub == "neighbors" and len(parts) >= 3:
            return handle_neighbors(unquote("/".join(parts[2:])))
        if sub == "path":
            a, b = (qs.get("a", [""])[0], qs.get("b", [""])[0])
            if not a or not b:
                return _err("path requires ?a=&b=", 400)
            return handle_path(a, b)
        if sub == "search":
            return handle_search(qs.get("q", [""])[0])
    return _err("unknown endpoint", 404)


def _ui_html() -> bytes:
    """The visual graph page. Serves the prebuilt HTML; builds it on first hit if absent."""
    out = os.path.join(_ROOT, "artifacts", "unified_graph.html")
    if not os.path.exists(out):
        try:
            from graph.visualize_graph import visualize
            visualize(include_synonyms=False)
        except Exception:
            return b"<h1>graph UI unavailable</h1><p>run: python3 graph/visualize_graph.py</p>"
    try:
        with open(out, "rb") as f:
            return f.read()
    except OSError:
        return b"<h1>graph UI not found</h1>"


# ── stdlib server ────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        # Root and /ui serve the VISUAL graph (HTML); everything else is JSON data.
        if u.path in ("/", "/ui"):
            html = _ui_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        status, body = route(u.path, parse_qs(u.query))
        payload = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):   # quiet
        pass


def build_fastapi_app():
    """Optional FastAPI app (only if fastapi installed). Same routes."""
    from fastapi import FastAPI                         # noqa
    from fastapi.responses import JSONResponse          # noqa
    app = FastAPI(title="VEDA Unified Graph API")

    def _wrap(res):
        st, body = res
        return JSONResponse(status_code=st, content=json.loads(json.dumps(body, default=str)))

    @app.get("/graph/node/{node_id:path}")
    def node(node_id: str):  return _wrap(handle_node(node_id))

    @app.get("/graph/neighbors/{node_id:path}")
    def neighbors(node_id: str):  return _wrap(handle_neighbors(node_id))

    @app.get("/graph/path")
    def path(a: str, b: str):  return _wrap(handle_path(a, b))

    @app.get("/graph/search")
    def search(q: str):  return _wrap(handle_search(q))

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the VEDA unified graph (read-only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()

    if get_graph() is None:
        print("unified graph not built — run: python3 ingestion/unified_graph_builder.py")
        return 1

    # Auto-pick a free port if the requested one is busy (avoids 'Address already in use').
    srv = None
    for port in range(args.port, args.port + 20):
        try:
            srv = ThreadingHTTPServer((args.host, port), _Handler)
            args.port = port
            break
        except OSError:
            continue
    if srv is None:
        print(f"no free port in {args.port}..{args.port + 19}")
        return 1
    print(f"VEDA graph → open the VISUAL UI:  http://{args.host}:{args.port}/")
    print(f"  JSON API: /graph/search?q=  /graph/node/{{id}}  /graph/neighbors/{{id}}  /graph/path?a=&b=")
    print("  Leave this terminal open; Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
