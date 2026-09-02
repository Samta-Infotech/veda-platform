import traceback
"""Tests for routing→RAG query-embedding reuse (DUP-1 / P0, flag-gated).

Verifies the RAG encoder reuses the routing embed-once vector on an exact query match, and re-encodes
otherwise / when the flag is off. Run: `python tests/test_embed_reuse.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

import config as _config  # noqa: E402
import query.rag_layer as RL  # noqa: E402
from query.source_coordinator import _ROUTING_QV  # noqa: E402
from ingestion import m3_encoder  # noqa: E402


def _spy_encoder():
    calls = {"n": 0}
    orig = m3_encoder.encode_dense

    def fake(texts):
        calls["n"] += 1
        return ["FRESH"]                 # encode_dense([q]) → [vec]; _encode_rag_query takes [0]
    m3_encoder.encode_dense = fake
    return calls, (lambda: setattr(m3_encoder, "encode_dense", orig))


def test_reuse_on_exact_match_when_flag_on():
    _config.RETRIEVAL_EMBED_REUSE_ENABLED = True
    _ROUTING_QV.set(("what is the leave policy", ["ROUTED_VEC"]))
    calls, restore = _spy_encoder()
    try:
        out = RL._encode_rag_query("what is the leave policy")
        assert out == ["ROUTED_VEC"]        # reused
        assert calls["n"] == 0              # NO fresh encode
    finally:
        restore()


def test_reencode_on_query_mismatch():
    _config.RETRIEVAL_EMBED_REUSE_ENABLED = True
    _ROUTING_QV.set(("a different query", ["ROUTED_VEC"]))
    calls, restore = _spy_encoder()
    try:
        out = RL._encode_rag_query("what is the leave policy")
        assert out == "FRESH"               # fell through to fresh encode
        assert calls["n"] == 1
    finally:
        restore()


def test_no_reuse_when_flag_off():
    _config.RETRIEVAL_EMBED_REUSE_ENABLED = False
    _ROUTING_QV.set(("what is the leave policy", ["ROUTED_VEC"]))
    calls, restore = _spy_encoder()
    try:
        out = RL._encode_rag_query("what is the leave policy")
        assert out == "FRESH"               # byte-identical: always fresh encode
        assert calls["n"] == 1
    finally:
        restore()


def test_no_reuse_when_cache_empty():
    _config.RETRIEVAL_EMBED_REUSE_ENABLED = True
    _ROUTING_QV.set((None, None))
    calls, restore = _spy_encoder()
    try:
        out = RL._encode_rag_query("q")
        assert out == "FRESH" and calls["n"] == 1
    finally:
        restore()


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
