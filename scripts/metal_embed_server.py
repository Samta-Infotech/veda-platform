#!/usr/bin/env python3
"""
metal_embed_server.py — HOST-side BGE-M3 + reranker server on Apple Metal (MPS).

Docker-on-macOS has no GPU passthrough, so the in-container BGE-M3 encoder and the
cross-encoder reranker run on CPU (~28s/retrieval — the dominant query-latency cost).
This mirrors what VEDA already does for the SLM (host Metal Ollama): it runs the SAME
two models on the host GPU (device=mps) and serves them over HTTP, so the inference
container offloads encode/rerank to Metal instead of grinding on CPU.

Point the container at it with:
    METAL_EMBED_URL=http://host.docker.internal:11435
(m3_encoder / reranker proxy to this when the env var is set; otherwise they stay
in-process on CPU exactly as before.)

Endpoints (all POST, JSON):
    /encode_dense   {texts:[...]}      -> {vecs:[[float×1024], ...]}         (L2-normalized)
    /encode_sparse  {texts:[...]}      -> {sparse:[{token:weight}, ...]}
    /encode_query   {text:"..."}       -> {dense:[float×1024], sparse:{...}}
    /rerank         {pairs:[[q,d],...]}-> {scores:[float, ...]}
    /healthz        -> {status, device}

HOST PREREQUISITES (run this OUTSIDE Docker, in a python env with the ML stack):
    pip install "FlagEmbedding" "sentence-transformers" "torch"      # torch with MPS
    # models must be cached locally (offline), same weights the container uses:
    export HF_HOME=~/models/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python3 scripts/metal_embed_server.py --port 11435

Env: BGE_MODEL (default BAAI/bge-m3), RERANKER_MODEL (default BAAI/bge-reranker-v2-m3),
     EMBED_DEVICE (default mps; falls back to cpu if MPS unavailable).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("metal_embed")

BGE_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def _resolve_device(pref: str) -> str:
    try:
        import torch
        if pref == "mps" and torch.backends.mps.is_available():
            return "mps"
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Models:
    """Lazily loaded, thread-safe singletons for the two models."""
    def __init__(self, device: str):
        self.device = device
        self._m3 = None
        self._rr = None
        self._lock = threading.Lock()

    def m3(self):
        if self._m3 is None:
            with self._lock:
                if self._m3 is None:
                    from FlagEmbedding import BGEM3FlagModel
                    log.info("loading BGE-M3 (%s) on %s ...", BGE_MODEL, self.device)
                    try:
                        self._m3 = BGEM3FlagModel(BGE_MODEL, use_fp16=False, device=self.device)
                    except TypeError:
                        # older/newer FlagEmbedding signatures differ on the device kwarg
                        self._m3 = BGEM3FlagModel(BGE_MODEL, use_fp16=False)
                    log.info("BGE-M3 ready")
        return self._m3

    def rr(self):
        if self._rr is None:
            with self._lock:
                if self._rr is None:
                    from sentence_transformers import CrossEncoder
                    log.info("loading reranker (%s) on %s ...", RERANKER_MODEL, self.device)
                    self._rr = CrossEncoder(RERANKER_MODEL, device=self.device)
                    log.info("reranker ready")
        return self._rr


def _l2(vecs):
    import numpy as np
    a = np.asarray(vecs, dtype="float32")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (a / n).tolist()


def _clean_sparse(lw: dict) -> dict:
    # drop the empty/degenerate token that BGE-M3 sometimes emits, keep floats
    return {str(k): float(v) for k, v in lw.items() if str(k).strip() and float(v) > 0}


class Handler(BaseHTTPRequestHandler):
    models: Models = None

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "device": self.models.device})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = self._read()
            if self.path == "/encode_dense":
                out = self.models.m3().encode(list(req["texts"]), return_dense=True,
                                              return_sparse=False, return_colbert_vecs=False)
                self._send(200, {"vecs": _l2(out["dense_vecs"])})
            elif self.path == "/encode_sparse":
                out = self.models.m3().encode(list(req["texts"]), return_dense=False,
                                              return_sparse=True, return_colbert_vecs=False)
                self._send(200, {"sparse": [_clean_sparse(lw) for lw in out["lexical_weights"]]})
            elif self.path == "/encode_query":
                out = self.models.m3().encode([req["text"]], return_dense=True,
                                              return_sparse=True, return_colbert_vecs=False)
                self._send(200, {"dense": _l2(out["dense_vecs"])[0],
                                 "sparse": _clean_sparse(out["lexical_weights"][0])})
            elif self.path == "/rerank":
                pairs = [list(p) for p in req["pairs"]]
                scores = self.models.rr().predict(pairs, batch_size=int(req.get("batch_size", 64)))
                self._send(200, {"scores": [float(s) for s in scores]})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            log.warning("%s failed: %s", self.path, e)
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("EMBED_HOST", "0.0.0.0"),
                    help="bind address (default 0.0.0.0 — reachable from other hosts on the LAN)")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "mps"))
    args = ap.parse_args()

    device = _resolve_device(args.device)
    Handler.models = Models(device)
    log.info("Metal embed server on %s:%d  device=%s", args.host, args.port, device)
    log.info("BGE_MODEL=%s  RERANKER_MODEL=%s", BGE_MODEL, RERANKER_MODEL)
    log.info("Point the container at:  METAL_EMBED_URL=http://host.docker.internal:%d", args.port)
    # warm both so the first real query is hot
    try:
        Handler.models.m3().encode(["warm up"], return_dense=True, return_sparse=True,
                                   return_colbert_vecs=False)
        Handler.models.rr().predict([["warm", "up"]])
        log.info("models warm")
    except Exception as e:
        log.warning("warm-up deferred: %s", e)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True   # threads don't block a clean shutdown/kickstart restart
    server.serve_forever()


if __name__ == "__main__":
    main()
