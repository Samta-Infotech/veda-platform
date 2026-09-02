"""Tests for cross-encoder RAG chunk re-ranking (query/reranker.rerank_chunks).

The reranker is injected — no model/endpoint. Run: `python tests/test_rerank_chunks.py`.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))

from query import reranker as R  # noqa: E402


class _Chunk:
    def __init__(self, text, similarity=0.0):
        self.text = text
        self.similarity = similarity


class _FakeReranker:
    """Returns a fixed score per (query, text) pair by looking up a text→score map."""
    def __init__(self, scoremap):
        self.scoremap = scoremap
    def predict(self, pairs, **_):
        return [self.scoremap.get(t, 0.0) for _, t in pairs]


def _with_reranker(rr, fn):
    orig = R._RERANKER_INSTANCE
    R._RERANKER_INSTANCE = rr
    try:
        return fn()
    finally:
        R._RERANKER_INSTANCE = orig


def test_rerank_reorders_by_crossencoder():
    # dense order puts a weak chunk first; the reranker should promote the strong one
    chunks = [_Chunk("football court upkeep", 0.62), _Chunk("maternity leave policy", 0.60)]
    rr = _FakeReranker({"football court upkeep": 0.01, "maternity leave policy": 0.99})
    out = _with_reranker(rr, lambda: R.rerank_chunks("what is the leave policy", chunks, 2))
    assert out[0].text == "maternity leave policy"      # promoted despite lower dense sim
    assert getattr(out[0], "rerank_score", 0) == 0.99


def test_rerank_truncates_to_top_n():
    chunks = [_Chunk("a"), _Chunk("b"), _Chunk("c")]
    rr = _FakeReranker({"a": 0.1, "b": 0.9, "c": 0.5})
    out = _with_reranker(rr, lambda: R.rerank_chunks("q", chunks, 2))
    assert [c.text for c in out] == ["b", "c"]          # top-2 by rerank score


def test_rerank_empty_input():
    assert R.rerank_chunks("q", [], 5) == []


def test_rerank_none_reranker_keeps_order():
    chunks = [_Chunk("a"), _Chunk("b"), _Chunk("c")]
    out = _with_reranker(None, lambda: R.rerank_chunks("q", chunks, 2))
    assert [c.text for c in out] == ["a", "b"]          # graceful: dense order, top-n


def test_rerank_score_mismatch_falls_back():
    chunks = [_Chunk("a"), _Chunk("b")]

    class _Bad:
        def predict(self, pairs, **_):
            return [0.5]                                # wrong length
    out = _with_reranker(_Bad(), lambda: R.rerank_chunks("q", chunks, 1))
    assert [c.text for c in out] == ["a"]              # fallback: input order, top-n


def test_rerank_predict_raises_falls_back():
    chunks = [_Chunk("a"), _Chunk("b")]

    class _Boom:
        def predict(self, pairs, **_):
            raise RuntimeError("endpoint down")
    out = _with_reranker(_Boom(), lambda: R.rerank_chunks("q", chunks, 2))
    assert [c.text for c in out] == ["a", "b"]          # never regress on failure


if __name__ == "__main__":
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print("PASS", name)
        except Exception:
            failed += 1; print("FAIL", name); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
