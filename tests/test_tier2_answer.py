"""Regression tests for veda_hybrid.py's Tier-2 response shape.

1. The SLM-assisted SQL fallback (envelope/shared-planner/IR paths) executed
   correct rows but never computed a natural-language answer, so the chat UI
   always fell back to the generic "Here's what I found." for every
   Tier-2-answered query — the same class of bug as the NoSQL fix (see
   test_result_explainer.py's test_queryresult_carries_answer_field).

2. Structural parity with Tier-1 (veda/pipeline.py's _done()): Tier-1 ALWAYS
   returns "table" and a real "explain", regardless of INSIGHT_ENGINE_ENABLED.
   Tier-2 previously never set "table" at all, and only set "explain" inside
   the (default-off) Insight Engine branch — so with the flag at its default,
   every Tier-2 answer had a visibly different response shape than Tier-1
   (missing table, empty/placeholder explainability on the frontend).

Pure-python, no DB, no network."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def test_tier2_finish_attaches_answer(monkeypatch):
    import slm
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: "Found 2 matching payments.")
    result = _tier2_finish(
        "show me the payments", sm=None,
        cols=["id", "amount"], rows=[(1, 100), (2, 200)],
        sql="SELECT id, amount FROM payments", source="tier2",
    )
    assert result["status"] == "answered" and result["ok"] is True
    assert result["source"] == "tier2"
    assert result["answer"] == "Found 2 matching payments."


def test_tier2_finish_has_table_and_explain_even_with_insight_engine_off(monkeypatch):
    """The core parity fix: table/explain must be present regardless of the
    Insight Engine flag — INSIGHT_ENGINE_ENABLED defaults to False, so this is
    the shape most real Tier-2 traffic actually gets today."""
    import config
    import slm
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(config, "INSIGHT_ENGINE_ENABLED", False)
    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: "Two payments were found.")
    result = _tier2_finish(
        "show me the payments", sm=None,
        cols=["id", "amount"], rows=[(1, 100), (2, 200)],
        sql="SELECT id, amount FROM payments", source="tier2",
    )
    assert result["table"] == "payments"
    assert result["explain"] is not None
    assert result["explain"]["version"] == "1.0"
    # Insight-Engine-only fields must NOT appear when the flag is off — same
    # additive-only contract as the deterministic path.
    assert "insights" not in result
    assert "confidence" not in result


def test_tier2_finish_table_derived_from_sql_entities():
    from veda_hybrid import _tier2_finish

    result = _tier2_finish(
        "q", sm=None, cols=["a"], rows=[(1,)],
        sql='SELECT t.a FROM ledger t JOIN accounts a ON t.acc_id = a.id', source="tier2",
    )
    # first referenced table, AST-derived — never invented, never guessed
    assert result["table"] in ("ledger", "accounts")


def test_tier2_finish_response_keys_match_tier1_shape(monkeypatch):
    """Same key SET as veda/pipeline.py's _done() success path (status, ok,
    cols, rows, sql, table, answer, explain), modulo Tier-2's own "source"
    addition and Tier-1's "trace" (an internal debug object never sent over
    SSE per apps/chat/services.py's own docstring — not part of the
    user-visible contract)."""
    import slm
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: "ok")
    result = _tier2_finish(
        "q", sm=None, cols=["a"], rows=[(1,)],
        sql="SELECT a FROM t", source="tier2",
    )
    tier1_keys = {"status", "ok", "cols", "rows", "sql", "table", "answer", "explain"}
    assert tier1_keys.issubset(result.keys())


def test_tier2_finish_never_raises_on_slm_failure(monkeypatch):
    import slm
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("SLM down")))
    result = _tier2_finish(
        "show me the payments", sm=None,
        cols=["id", "amount"], rows=[(1, 100), (2, 200)],
        sql="SELECT id, amount FROM payments", source="tier2",
    )
    # rows/sql/status are always correct even if summarization degrades;
    # the deterministic fallback still produces SOME answer, never a KeyError
    assert result["status"] == "answered"
    assert result["cols"] == ["id", "amount"]
    assert "answer" in result and result["answer"]


def test_tier2_finish_accepts_dict_rows():
    import slm
    from veda_hybrid import _tier2_finish

    slm.call_slm = lambda *a, **k: "One record found."
    result = _tier2_finish(
        "q", sm=None, cols=["id"], rows=[{"id": 1}], sql="SELECT id FROM t", source="envelope",
    )
    assert result["answer"] == "One record found."


# ---------------------------------------------------------------------------
# INSIGHT_ENGINE_ENABLED=True — Tier-2 gets full parity with the deterministic
# path (insights/follow-ups/confidence/real explainability), not just the
# plain answer text.
# ---------------------------------------------------------------------------

def test_tier2_finish_insight_engine_enabled(monkeypatch):
    import json
    import slm
    import config
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(config, "INSIGHT_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "INSIGHT_FOLLOW_UPS_ENABLED", True, raising=False)
    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: json.dumps({
        "summary": "Two payments were found.",
        "insights": ["Both payments are for the same amount."],
        "follow_up_questions": ["Show only totals above 150"],
        "visualization": {"type": "bar", "x_axis": "payer_name", "y_axis": "total",
                          "reason": "category vs numeric"},
    }))
    # A genuinely GROUPED query (aggregation + GROUP BY) — a raw, ungrouped
    # listing correctly gets no chart under the shape-aware validation
    # (DETAIL_TABLE has no canonical chart type; see result_analyzer.py).
    rows = [{"payer_name": "Alice", "total": 200}, {"payer_name": "Bob", "total": 100}]
    result = _tier2_finish(
        "total paid per payer", sm=None,
        cols=["payer_name", "total"], rows=rows,
        sql="SELECT payer_name, SUM(amount) AS total FROM payments GROUP BY payer_name",
        source="tier2",
    )
    assert result["answer"] == "Two payments were found."
    assert result["insights"] == ["Both payments are for the same amount."]
    assert result["follow_up_questions"] == ["Show only totals above 150"]
    assert result["visualization"]["type"] == "bar"
    # Confidence is no longer a top-level result key — it now lives inside `explain`
    # (build_explain's "confidence"), and the api tier reads it from there
    # (apps/chat/services.py). The Tier-2 result carrying its own confidence was
    # only meaningful when the removed `insights` SSE event surfaced it.
    assert "explain" in result and result["explain"]["version"] == "1.0"


def test_tier2_finish_insight_engine_falls_back_on_failure(monkeypatch):
    import config
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(config, "INSIGHT_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "INSIGHT_FOLLOW_UPS_ENABLED", True, raising=False)

    def _broken_analyze_result(*a, **k):
        raise RuntimeError("boom")

    import veda.result_analyzer as ra_mod
    monkeypatch.setattr(ra_mod, "analyze_result", _broken_analyze_result)

    result = _tier2_finish(
        "q", sm=None, cols=["id"], rows=[{"id": 1}], sql="SELECT id FROM t", source="tier2",
    )
    # Insight Engine blew up -> gracefully falls back to the plain NL answer,
    # never propagates the exception or drops the (correct) rows/status.
    assert result["status"] == "answered"
    assert "answer" in result


def test_tier2_finish_answer_never_missing_even_when_both_paths_fail_outside_their_own_try(monkeypatch):
    """Regression (2026-07): unlike run_insight_engine/run_nl_answer's OWN
    internal SLM-call fallback (already covered above), an exception from
    OUTSIDE both of those calls entirely — e.g. analyze_result() AND
    run_nl_answer() both raising before reaching their internal try/except —
    used to leave "answer" completely ABSENT from the Tier-2 result. That's
    worse than a raw-text fallback: chatbot/nodes.py's format_reply_node then
    shows its own generic "Here's what I found.", silently masking that
    Tier-2 produced NO grounded summary at all. _tier2_finish must now set a
    deterministic fallback UPFRONT (mirroring veda/pipeline.py's L7b), so
    "answer" is guaranteed present regardless of where the failure occurs."""
    import config
    from veda_hybrid import _tier2_finish

    monkeypatch.setattr(config, "INSIGHT_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "INSIGHT_FOLLOW_UPS_ENABLED", True, raising=False)

    import veda.result_analyzer as ra_mod
    monkeypatch.setattr(ra_mod, "analyze_result",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("analysis boom")))

    # _tier2_finish imports run_nl_answer from the query.nl_answer shim (not
    # directly from result_explainer) — patch where it's actually looked up.
    import query.nl_answer as nl_answer_mod
    monkeypatch.setattr(nl_answer_mod, "run_nl_answer",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nl_answer boom")))

    result = _tier2_finish(
        "how many payments", sm=None, cols=["id", "amount"], rows=[(1, 100), (2, 200)],
        sql="SELECT id, amount FROM payments", source="tier2",
    )
    assert result["status"] == "answered"
    assert "answer" in result and result["answer"]
    assert "Returned" in result["answer"] or "row" in result["answer"].lower()
