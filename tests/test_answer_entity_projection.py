"""Follow-up fix from the implementation audit: pipeline.py's answer_entity
"projection" mode (`find_answer_entity(...)["mode"] == "projection"` — e.g.
"incidents and their handler") used to build `SELECT a.*, t."<disp>" AS ...`,
i.e. every anchor-table column, completely bypassing recommended_projection().
It now projects the anchor's own recommended_projection() instead, same as
every other deterministic branch.

Drives run_query() itself (not a smaller unit), same dependency-monkeypatch
pattern as tests/test_projection_wiring.py — no DB, no LLM, no embedding
model. query.answer_entity.find_answer_entity() and veda.graph_guard.
verify_joins_against_graph() are stubbed since real FK-graph/entity-discovery
data isn't the point of this test; the SELECT clause construction is.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

TABLE = "worklists_incident"
TARGET = "users_user"
ALL_COLS = [f"{TABLE}.{c}" for c in (
    "id", "title", "status", "priority", "assigned_to_id", "created_at",
    "updated_at", "created_by_id", "deleted_at", "internal_notes",
)] + [f"{TARGET}.{c}" for c in ("id", "name", "email")]
SM = {"columns": {
    f"{TABLE}.id": {"importance_class": "LOW"},
    f"{TABLE}.title": {"importance_class": "HIGH"},
    f"{TABLE}.status": {"importance_class": "HIGH"},
    f"{TABLE}.priority": {"importance_class": "MEDIUM"},
    f"{TABLE}.assigned_to_id": {"importance_class": "LOW"},
    f"{TABLE}.created_at": {"importance_class": "MEDIUM"},
    f"{TABLE}.updated_at": {"importance_class": "LOW"},
    f"{TABLE}.created_by_id": {"importance_class": "LOW"},
    f"{TABLE}.deleted_at": {"importance_class": "LOW"},
    f"{TABLE}.internal_notes": {"importance_class": "LOW"},
    f"{TARGET}.name": {"importance_class": "HIGH"},
}}


class _FakeResult:
    def __init__(self, col_id, score):
        self.col_id = col_id
        self.column_name = col_id.split(".", 1)[1]
        self.final_score = score


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def retrieve(self, **kwargs):
        return list(self._rows)


def test_answer_entity_projection_mode_no_longer_uses_select_star(monkeypatch):
    import config
    import query.answer_entity as answer_entity
    import veda.graph_guard as graph_guard
    import veda.pipeline as pipeline

    results = [_FakeResult(f"{TABLE}.title", 0.9), _FakeResult(f"{TABLE}.status", 0.85)]

    monkeypatch.setattr(pipeline, "verified_cache_lookup", lambda q: (None, 0.0))
    monkeypatch.setattr(pipeline, "get_engine", lambda sm: _FakeEngine(results))
    monkeypatch.setattr(pipeline, "select_primary_table", lambda *a, **k: TABLE)
    monkeypatch.setattr(pipeline, "vet_primary", lambda *a, **k: TABLE)
    monkeypatch.setattr(pipeline, "execute_sql",
                        lambda sql, params=None: (["title", "status", "name"],
                                                  [("Bug", "open", "Alice")], None))
    monkeypatch.setattr(answer_entity, "find_answer_entity", lambda query, primary, graph, sm: {
        "reason": "test-forced", "fk_col": "assigned_to_id", "target_table": TARGET,
        "display_col": "name", "target_pk": "id", "mode": "projection",
        "rel_label": "handler",
    })
    # Real FK-graph edge verification isn't this test's concern (no real graph
    # data supplied) — bypassed so validation reaches the actual assertion.
    monkeypatch.setattr(graph_guard, "verify_joins_against_graph", lambda sql, graph=None: (True, None))
    monkeypatch.setattr(config, "GRAPH_EXPAND_ENABLED", False)
    monkeypatch.setattr(config, "PRIMARY_RERANK_ENABLED", False)
    monkeypatch.setattr(config, "FAST_PATH_ENABLED", False)
    monkeypatch.setattr(config, "SUPERLATIVE_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "GROUPED_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "RATIO_PLAN_ENABLED", False)
    monkeypatch.setattr(config, "ANSWER_ENTITY_DISCOVERY_ENABLED", True)

    res = pipeline.run_query("incidents and their handler", sm=SM, all_cols=ALL_COLS,
                             return_result=True)

    assert res["status"] == "answered"
    sql = res["sql"]
    assert "a.*" not in sql
    assert '"title"' in sql and '"status"' in sql and '"name"' in sql
    for excluded in ("created_by_id", "deleted_at", "internal_notes"):
        assert excluded not in sql
