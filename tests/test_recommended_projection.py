"""veda/routing.py::recommended_projection() — composes THREE already-computed
signals into a small, deduplicated, business-facing SELECT list, without
re-ranking columns from scratch:

  1. entity identity  — veda/generation.py::_resolve_display_column(), the SAME
     governed label-column resolver joins/grouped-breakdown/aggregate already
     use (falls back to the concept registry's default_display_columns only
     when that resolver finds nothing at all for the table)
  2. per-query retrieval relevance — RetrievalResult.final_score (user intent)
  3. importance_class == "HIGH"    — ingestion metadata (business importance)

Plus two safety overrides, both added before the cap: `must_include` (columns
the CALLER already knows are structurally required, e.g. an active ORDER BY
column), and a column the query literally names (or a known alias).

Pure-python, no DB, no LLM, no network — `results` are plain fake objects
carrying only the two attributes the function actually reads (col_id,
column_name, final_score); `semantic.registry` is not mocked, so the registry
FALLBACK naturally no-ops in this test process (no active scope) — proven
separately not to matter via the "fallback when everything is empty" test,
and via pipeline-level coverage in tests/test_projection_wiring.py. Fixture
SM dicts here deliberately omit "table_name" so _resolve_display_column also
no-ops (returns None) in the tests that aren't specifically about it —
test_uses_resolve_display_column_for_entity_identity below is the one that
gives it real metadata to prove it actually fires.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

TABLE = "accounts_paymenttransaction"
ALLOWED = ["id", "payment_number", "amount", "status", "created_at", "updated_at",
           "created_by_id", "updated_by_id", "deleted_at", "payment_signature",
           "third_party_name", "third_party_email"]
SM = {"columns": {
    f"{TABLE}.id": {"importance_class": "LOW"},
    f"{TABLE}.payment_number": {"importance_class": "HIGH"},
    f"{TABLE}.amount": {"importance_class": "HIGH"},
    f"{TABLE}.status": {"importance_class": "HIGH"},
    f"{TABLE}.created_at": {"importance_class": "MEDIUM"},
    f"{TABLE}.updated_at": {"importance_class": "LOW"},
    f"{TABLE}.created_by_id": {"importance_class": "LOW"},
    f"{TABLE}.updated_by_id": {"importance_class": "LOW"},
    f"{TABLE}.deleted_at": {"importance_class": "LOW"},
    f"{TABLE}.payment_signature": {"importance_class": "LOW", "aliases": ["signature"]},
    f"{TABLE}.third_party_name": {"importance_class": "LOW"},
    f"{TABLE}.third_party_email": {"importance_class": "LOW"},
}}


class _FakeResult:
    def __init__(self, col_id, score):
        self.col_id = col_id
        self.column_name = col_id.split(".", 1)[1]
        self.final_score = score


def _results(*pairs):
    return [_FakeResult(f"{TABLE}.{col}", score) for col, score in pairs]


def test_composes_retrieval_relevance_and_high_importance():
    from veda.routing import recommended_projection
    results = _results(("amount", 0.9), ("status", 0.85), ("created_at", 0.6))
    out = recommended_projection(TABLE, ALLOWED, results, SM, "show recent payments")

    # Retrieval-relevant columns present, and payment_number (HIGH importance,
    # not itself retrieval-relevant here) still surfaces via component 3.
    assert "amount" in out and "status" in out and "created_at" in out
    assert "payment_number" in out
    # Audit/internal columns never picked by any of the three signals must
    # stay excluded — this is the whole point of the fix.
    for excluded in ("created_by_id", "updated_by_id", "deleted_at"):
        assert excluded not in out


def test_explicit_mention_overrides_low_importance_and_survives_cap():
    """Safety requirement: a column the user explicitly named (here, via its
    business alias 'signature') must always appear, even though it's LOW
    importance and wasn't retrieval-relevant in this fake result set."""
    from veda.routing import recommended_projection
    results = _results(("amount", 0.9), ("status", 0.85))
    out = recommended_projection(TABLE, ALLOWED, results, SM, "show payment signature")
    assert "payment_signature" in out
    assert out[0] == "payment_signature"   # explicit mentions are added first


def test_explicit_mention_by_bare_column_name():
    from veda.routing import recommended_projection
    out = recommended_projection(TABLE, ALLOWED, [], SM, "what is the third party email")
    assert "third_party_email" in out


def test_falls_back_to_allowed_columns_when_every_signal_is_empty():
    """No registry, no retrieval results, no query text, no importance metadata
    -> must return allowed_columns UNCHANGED, never an empty/degraded SELECT."""
    from veda.routing import recommended_projection
    out = recommended_projection(TABLE, ALLOWED, [], {}, "")
    assert out == ALLOWED


def test_never_recommends_a_column_outside_allowed_columns():
    """Safety: even if metadata/retrieval somehow names a column NOT in
    allowed_columns (stale registry, mismatched table), it must never leak
    into the recommendation — allowed_columns is validation's own contract
    and this function must never expand what could be selected."""
    from veda.routing import recommended_projection
    results = _results(("amount", 0.9))
    sm = {"columns": {f"{TABLE}.amount": {"importance_class": "HIGH"},
                      f"{TABLE}.not_allowed": {"importance_class": "HIGH"}}}
    narrow_allowed = ["amount", "status"]
    out = recommended_projection(TABLE, narrow_allowed, results, sm, "amounts")
    assert "not_allowed" not in out
    assert set(out) <= set(narrow_allowed)


def test_respects_max_cols_cap():
    from veda.routing import recommended_projection
    import config
    wide_allowed = [f"col{i}" for i in range(30)]
    sm = {"columns": {f"{TABLE}.{c}": {"importance_class": "HIGH"} for c in wide_allowed}}
    out = recommended_projection(TABLE, wide_allowed, [], sm, "")
    assert len(out) <= config.RECOMMENDED_PROJECTION_MAX_COLS


def test_deduplicates_across_signals():
    """A column that's BOTH retrieval-relevant AND HIGH importance (e.g.
    'amount') must appear exactly once, not twice."""
    from veda.routing import recommended_projection
    results = _results(("amount", 0.9))
    out = recommended_projection(TABLE, ALLOWED, results, SM, "amount")
    assert out.count("amount") == 1


def test_must_include_survives_cap():
    """The caller-supplied structural requirement (e.g. an active ORDER BY
    column) must appear even when it's neither retrieval-relevant nor HIGH
    importance, and even after RECOMMENDED_PROJECTION_MAX_COLS trims a wide
    HIGH-importance table down."""
    from veda.routing import recommended_projection
    import config
    wide_allowed = [f"col{i}" for i in range(30)] + ["sort_col"]
    sm = {"columns": {f"{TABLE}.{c}": {"importance_class": "HIGH"} for c in wide_allowed[:30]}}
    sm["columns"][f"{TABLE}.sort_col"] = {"importance_class": "LOW"}
    out = recommended_projection(TABLE, wide_allowed, [], sm, "", must_include=["sort_col"])
    assert "sort_col" in out
    assert out[0] == "sort_col"
    assert len(out) <= config.RECOMMENDED_PROJECTION_MAX_COLS


def test_must_include_ignores_columns_outside_allowed():
    """Same safety guarantee as every other signal: must_include cannot smuggle
    in a column that isn't in allowed_columns."""
    from veda.routing import recommended_projection
    out = recommended_projection(TABLE, ["amount", "status"], [], SM, "",
                                 must_include=["not_a_real_column"])
    assert "not_a_real_column" not in out


def test_uses_resolve_display_column_for_entity_identity():
    """Item 6 (consolidation): entity identity now comes from veda/generation.py's
    _resolve_display_column() — the SAME resolver joins/grouped-breakdown/
    aggregate already depend on — not a second, independent registry read.
    Proven by giving it metadata shaped the way _resolve_display_column
    actually expects (table_name/col_name keys), which the other fixtures in
    this file deliberately omit."""
    from veda.routing import recommended_projection
    sm = {"columns": {
        f"{TABLE}.id": {"table_name": TABLE, "col_name": "id", "importance_class": "LOW"},
        f"{TABLE}.reference_no": {"table_name": TABLE, "col_name": "reference_no",
                                  "importance_class": "MEDIUM"},
        f"{TABLE}.internal_flag": {"table_name": TABLE, "col_name": "internal_flag",
                                   "importance_class": "LOW"},
    }}
    allowed = ["id", "reference_no", "internal_flag"]
    out = recommended_projection(TABLE, allowed, [], sm, "")
    # _resolve_display_column's heuristic picks the *_no column as the label
    # column (no name/title/display_name/label column present here) — proving
    # its actual resolution logic ran, not a no-op.
    assert "reference_no" in out
    assert "internal_flag" not in out


def test_high_importance_is_filler_not_flood(monkeypatch):
    """2026-07-19: the static HIGH-importance signal only TOPS UP a thin
    query-driven projection (to RECOMMENDED_PROJECTION_IMPORTANCE_FLOOR) — it no
    longer floods every SELECT with every HIGH column of the table."""
    import config
    from veda.routing import recommended_projection
    # A wide table where MANY columns are HIGH but only two are query-relevant.
    table = "payments"
    allowed = ["amount", "status", "sig", "attempt", "sched", "due", "pdate", "ref", "name"]
    sm = {"columns": {f"{table}.{c}": {"importance_class": "HIGH"} for c in allowed}}
    class R:  # minimal RetrievalResult shape
        def __init__(s, c, sc): s.col_id, s.column_name, s.final_score = f"{table}.{c}", c, sc
    results = [R("amount", .9), R("name", .8)]
    monkeypatch.setattr(config, "RECOMMENDED_PROJECTION_IMPORTANCE_FLOOR", 4, raising=False)
    out = recommended_projection(table, allowed, results, sm, "top payments by amount",
                                 must_include=["amount"])
    assert "amount" in out and "name" in out       # query-driven picks intact
    assert len(out) <= 4                            # topped up to the floor, NOT all 9 HIGH cols


def test_high_importance_still_tops_up_thin_projections(monkeypatch):
    """When the query-driven signals produce almost nothing, HIGH columns still
    fill in (the original audit-column fix's behaviour is preserved as fallback)."""
    import config
    from veda.routing import recommended_projection
    table = "payments"
    allowed = ["amount", "status", "audit_id"]
    sm = {"columns": {f"{table}.amount": {"importance_class": "HIGH"},
                      f"{table}.status": {"importance_class": "HIGH"},
                      f"{table}.audit_id": {"importance_class": "LOW"}}}
    monkeypatch.setattr(config, "RECOMMENDED_PROJECTION_IMPORTANCE_FLOOR", 6, raising=False)
    out = recommended_projection(table, allowed, [], sm, "payments overview")
    assert "amount" in out and "status" in out      # HIGH filler fired (picks were empty)
    assert "audit_id" not in out                     # LOW never enters via this signal
