"""Model-free tests for the shared analytical-semantics validator
(veda/semantic_validation.py). No SLM, no embeddings, no DB — pure AST + metadata.

Proves the generic invariants (operator preserved, group-by present, user-facing
dimension not an unnecessary identifier, explicit-id respected, joins grounded) and
— mandatorily — that the SAME logic works across DIFFERENTLY NAMED schemas driven
only by semantic_type metadata (no `_id`/`_name` assumption, no hardcoded names).

Run from repo root: ``pytest tests/test_semantic_validation.py``
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


# --- two schemas with DIFFERENT naming conventions, same business semantics -----
# Schema A: conventional Django-ish names. Schema B: deliberately anti-convention —
# the identifier does NOT end in _id, the display does NOT end in _name. Metadata
# (semantic_type) is the only signal; the validator must behave identically.
def _col(table, name, stype, arole):
    # the real semantic-model per-column shape (col_name/table_name/semantic_type/
    # analytics_role) that _resolve_display_column reads.
    return {"col_name": name, "table_name": table, "semantic_type": stype,
            "analytics_role": arole}


SCHEMA_A = {
    "columns": {
        "sales.entity_id":   _col("sales", "entity_id", "IDENTIFIER", "IDENTIFIER"),
        "sales.entity_name": _col("sales", "entity_name", "CATEGORY", "DIMENSION"),
        "sales.amount":      _col("sales", "amount", "MONETARY", "MEASURE"),
    },
    "tables": {"sales": {}},
}
# Deliberately anti-convention: the identifier does NOT end in _id, the display does
# NOT end in _name. Only semantic_type/analytics_role identify them → the validator
# and display resolver must behave identically to SCHEMA_A (no suffix assumption).
SCHEMA_B = {
    "columns": {
        "txn.object_key":     _col("txn", "object_key", "IDENTIFIER", "IDENTIFIER"),
        "txn.display_label":  _col("txn", "display_label", "CATEGORY", "DIMENSION"),
        "txn.metric_value":   _col("txn", "metric_value", "METRIC", "MEASURE"),
    },
    "tables": {"txn": {}},
}


def _validate(query, sql, sm, graph=None):
    from veda.semantic_validation import validate_analytical_semantics
    return validate_analytical_semantics(query, sql, sm, graph=graph)


def _codes(findings):
    return {f["code"] for f in findings}


# ---------------------------------------------------------------------------
# operator preservation
# ---------------------------------------------------------------------------
def test_operator_mismatch_avg_vs_sum():
    f = _validate("average amount per entity",
                  "SELECT entity_name, SUM(amount) FROM sales GROUP BY entity_name", SCHEMA_A)
    assert "operator_mismatch" in _codes(f)


def test_operator_preserved_no_finding():
    f = _validate("average amount per entity",
                  "SELECT entity_name, AVG(amount) FROM sales GROUP BY entity_name", SCHEMA_A)
    assert "operator_mismatch" not in _codes(f) and "operator_dropped" not in _codes(f)


def test_operator_dropped_when_grouped_agg_has_no_aggregate():
    f = _validate("average amount per entity",
                  "SELECT entity_name, amount FROM sales GROUP BY entity_name, amount", SCHEMA_A)
    assert "operator_dropped" in _codes(f)


def test_sum_request_satisfied_by_sum():
    f = _validate("total amount per entity",
                  "SELECT entity_name, SUM(amount) FROM sales GROUP BY entity_name", SCHEMA_A)
    assert "operator_mismatch" not in _codes(f)


def test_min_max_operator_preserved():
    assert "operator_mismatch" not in _codes(_validate(
        "minimum amount per entity",
        "SELECT entity_name, MIN(amount) FROM sales GROUP BY entity_name", SCHEMA_A))
    assert "operator_mismatch" in _codes(_validate(
        "maximum amount per entity",
        "SELECT entity_name, MIN(amount) FROM sales GROUP BY entity_name", SCHEMA_A))


# ---------------------------------------------------------------------------
# group-by presence
# ---------------------------------------------------------------------------
def test_missing_group_by_flagged():
    f = _validate("average amount per entity",
                  "SELECT AVG(amount) FROM sales", SCHEMA_A)
    assert "missing_group_by" in _codes(f)


# ---------------------------------------------------------------------------
# identifier-as-dimension (metadata-driven, both schemas)
# ---------------------------------------------------------------------------
def test_identifier_dimension_flagged_schema_a():
    f = _validate("average amount per entity",
                  "SELECT entity_id, AVG(amount) FROM sales GROUP BY entity_id", SCHEMA_A)
    assert "identifier_dimension" in _codes(f)


def test_identifier_dimension_flagged_schema_b_no_suffix_convention():
    # object_key (IDENTIFIER) has no _id suffix; display_label has no _name suffix.
    # Purely semantic_type-driven → still flagged, proving NO suffix assumption.
    f = _validate("average metric per object",
                  "SELECT object_key, AVG(metric_value) FROM txn GROUP BY object_key", SCHEMA_B)
    assert "identifier_dimension" in _codes(f)


def test_display_dimension_not_flagged_both_schemas():
    assert "identifier_dimension" not in _codes(_validate(
        "average amount per entity",
        "SELECT entity_name, AVG(amount) FROM sales GROUP BY entity_name", SCHEMA_A))
    assert "identifier_dimension" not in _codes(_validate(
        "average metric per object",
        "SELECT display_label, AVG(metric_value) FROM txn GROUP BY display_label", SCHEMA_B))


def test_explicit_identifier_request_suppresses_warning():
    # user explicitly asked for the id → grouping by it is correct, no warning
    f = _validate("average amount per entity id",
                  "SELECT entity_id, AVG(amount) FROM sales GROUP BY entity_id", SCHEMA_A)
    assert "identifier_dimension" not in _codes(f)


def test_user_requested_identifier_helper():
    from veda.semantic_validation import user_requested_identifier
    assert user_requested_identifier("list the project codes")
    assert user_requested_identifier("show entity id and amount")
    assert not user_requested_identifier("average amount per entity")


# ---------------------------------------------------------------------------
# join grounding
# ---------------------------------------------------------------------------
def test_ungrounded_join_flagged():
    sm = {"columns": {"a.x": {"semantic_type": "METRIC"}, "b.y": {"semantic_type": "CATEGORY"}},
          "tables": {"a": {}, "b": {}}}
    graph = {"edges": []}  # no relationship between a and b
    f = _validate("y and x", "SELECT b.y, a.x FROM a JOIN b ON a.k = b.k", sm, graph=graph)
    assert "ungrounded_join" in _codes(f)


def test_grounded_join_not_flagged():
    sm = {"columns": {"a.x": {"semantic_type": "METRIC"}, "b.y": {"semantic_type": "CATEGORY"}},
          "tables": {"a": {}, "b": {}}}
    graph = {"edges": [{"source_table": "a", "target_table": "b"}]}
    f = _validate("y and x", "SELECT b.y, a.x FROM a JOIN b ON a.k = b.k", sm, graph=graph)
    assert "ungrounded_join" not in _codes(f)


def test_no_graph_skips_join_check():
    sm = {"columns": {"a.x": {"semantic_type": "METRIC"}}, "tables": {"a": {}, "b": {}}}
    f = _validate("y and x", "SELECT b.y, a.x FROM a JOIN b ON a.k = b.k", sm, graph=None)
    assert "ungrounded_join" not in _codes(f)


# ---------------------------------------------------------------------------
# non-analytical queries produce no analytical findings
# ---------------------------------------------------------------------------
def test_plain_detail_query_clean():
    f = _validate("show all sales", "SELECT entity_name, amount FROM sales LIMIT 100", SCHEMA_A)
    assert f == []


def test_has_errors_helper():
    from veda.semantic_validation import has_errors
    assert has_errors([{"severity": "error"}, {"severity": "warning"}])
    assert not has_errors([{"severity": "warning"}])
    assert not has_errors([])
