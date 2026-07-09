#!/usr/bin/env python3
"""
ollama_proxy.py — Simple round-robin load balancer for multiple Ollama hosts.

Distributes VEDA's L3 semantic-layer LLM calls (Stage 3/4) across N independently
hosted Ollama instances, so SEMANTIC_PARALLEL_QWEN_ENABLED can actually use all N
machines instead of hammering a single OLLAMA_URL (which is all VEDA's own code
supports today — one fixed backend URL, no built-in multi-host distribution).

Stdlib only (http.server + urllib) — no new pip dependency, matching the rest of
this codebase's preference for stdlib on the thin/infra layers.

USAGE
-----
    OLLAMA_BACKENDS="http://laptop1.local:11434,http://laptop2.local:11434,http://laptop3.local:11434" \\
        python3 ollama_proxy.py --port 11434

Then point VEDA at THIS proxy instead of a single Ollama host:
    OLLAMA_URL=http://<machine-running-this-proxy>:11434

Pair with (config.py):
    SEMANTIC_PARALLEL_QWEN_ENABLED = True
    SEMANTIC_MAX_PARALLEL_REQUESTS = 6   # ~2 concurrent per backend, 3 backends

BEHAVIOR
--------
- Round-robins each incoming request across the configured backends.
- If a backend fails (connection refused / timeout / 5xx), it is marked "down"
  for a cooldown window and the SAME incoming request is transparently retried
  against the next backend — so VEDA's own retry/circuit-breaker logic sees a
  clean pass or a clean fail, never a hang or a spurious failure just because
  one of the three machines happened to be the one round-robin picked.
- Only if ALL backends fail does the proxy return a 502, which VEDA's existing
  `_call_ollama` retry/backoff and circuit breaker handle exactly as they would
  handle a single unreachable Ollama today — no VEDA code changes needed.
- Logs which backend served every request, so you can visually confirm load is
  actually spreading across all three machines (tail the log while ingesting).

CAVEAT
------
This proxies non-streaming requests (reads the full response, then forwards it
in one shot). VEDA's Ollama backend calls (`slm/_call_slm.py::OllamaBackend`)
already read the response as a single JSON blob, not a stream, so this matches
today's usage. If that ever changes to streaming (`"stream": true`), this proxy
would need to forward chunks incrementally instead of buffering — not needed
for VEDA's current call pattern.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ollama_proxy")

# How long a backend stays "deprioritized" after a failure before we try it
# again. Mirrors the spirit of VEDA's own circuit-breaker cooldown.
COOLDOWN_S = 30

# Ollama calls (Stage 3/4 batches) can legitimately take a while — generous
# per-attempt timeout so we don't cut off a slow-but-working backend early.
UPSTREAM_TIMEOUT_S = 280


class BackendPool:
    """Round-robin backend selection with simple failure cooldown."""

    def __init__(self, backends: list[str]):
        if not backends:
            raise ValueError(
                "No backends configured — set OLLAMA_BACKENDS to a comma-separated "
                "list, e.g. http://laptop1:11434,http://laptop2:11434,http://laptop3:11434"
            )
        self.backends = backends
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(backends)))
        self._down_until = {i: 0.0 for i in range(len(backends))}

    def mark_down(self, idx: int) -> None:
        with self._lock:
            self._down_until[idx] = time.time() + COOLDOWN_S
        log.warning(f"Backend #{idx} ({self.backends[idx]}) marked DOWN for {COOLDOWN_S}s")

    def next_order(self) -> list[int]:
        """Indices to try, in order: round-robin start, healthy ones first,
        then any currently-cooling-down ones last (so total failure is still
        possible to recover from rather than permanently skipping a backend
        that has actually come back up)."""
        with self._lock:
            start = next(self._cycle)
        order = [(start + i) % len(self.backends) for i in range(len(self.backends))]
        now = time.time()
        healthy = [i for i in order if self._down_until[i] <= now]
        cooling = [i for i in order if self._down_until[i] > now]
        return healthy + cooling


class ProxyHandler(BaseHTTPRequestHandler):
    pool: BackendPool = None  # set once in main()

    def log_message(self, fmt, *args):
        pass  # silence the default per-request access log; we log our own summary

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None

        last_err = None
        for idx in self.pool.next_order():
            base = self.pool.backends[idx]
            url = base.rstrip("/") + self.path
            try:
                req = urllib.request.Request(url, data=body, method=method)
                if "Content-Type" in self.headers:
                    req.add_header("Content-Type", self.headers["Content-Type"])
                with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
                    payload = resp.read()
                    self.send_response(resp.status)
                    self.send_header(
                        "Content-Type", resp.headers.get("Content-Type", "application/json")
                    )
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                log.info(f"{method} {self.path} -> backend #{idx} ({base}) OK")
                return
            except Exception as e:
                last_err = e
                log.warning(f"{method} {self.path} -> backend #{idx} ({base}) FAILED: {e}")
                self.pool.mark_down(idx)
                continue  # transparently try the next backend for this SAME request

        # Every backend failed this request — surface a clean 502 so VEDA's own
        # retry/backoff/circuit-breaker sees an ordinary failure, not a hang.
        self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": f"all backends failed: {last_err}"}).encode())

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=11434, help="Port this proxy listens on")
    ap.add_argument(
        "--backends",
        default=os.environ.get("OLLAMA_BACKENDS", ""),
        help="Comma-separated Ollama base URLs, e.g. "
             "http://laptop1:11434,http://laptop2:11434,http://laptop3:11434 "
             "(or set the OLLAMA_BACKENDS env var)",
    )
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    ProxyHandler.pool = BackendPool(backends)

    log.info(f"Ollama round-robin proxy listening on 0.0.0.0:{args.port}")
    log.info(f"Backends ({len(backends)}): {backends}")
    log.info("Point VEDA at this proxy via OLLAMA_URL, e.g.:")
    log.info(f"  OLLAMA_URL=http://<this-machine>:{args.port}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
