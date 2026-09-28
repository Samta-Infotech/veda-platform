"""The continuity lane (veda/understanding/continuity.py): a chat follow-up is the PREVIOUS
turn's query with one slot changed, compiled by the frame compiler and checked by the
firewall structurally — never re-entered as a first turn.

Deterministic fixtures modelled on the real homzhub schema (source 2): properties
(assets_asset) → asset type (assets_assettype, via asset_type_id, the declared FK);
lease listings (assets_leaselisting) → lease unit → property, where the rent lives.
No SLM (the ambiguous path uses a stub), no DB, no Redis.

Run: pytest tests/test_continuity.py
"""
import copy
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veda_core"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import pytest                                                   # noqa: E402

from veda.understanding import continuity as C                  # noqa: E402
from veda.understanding import frame_grounding as G             # noqa: E402
from veda.understanding.frame_path import FrameLane             # noqa: E402
from veda.understanding.vocabulary import ScopeVocab            # noqa: E402
from veda import firewall                                       # noqa: E402

SM = {
    "tables": {"assets_asset": {}, "assets_assettype": {}, "assets_leaseunit": {},
               "assets_leaselisting": {}, "vendors": {}},
    "columns": {
        "assets_asset.id": {"semantic_type": "IDENTIFIER"},
        "assets_asset.project_name": {"semantic_type": "CATEGORY"},
        "assets_asset.city_name": {"semantic_type": "CATEGORY",
                                   "sample_values": ["Mumbai", "Pune", "Kochi", "Bangalore"]},
        "assets_asset.furnishing": {"semantic_type": "CATEGORY", "sample_values": ["FULL", "SEMI", "NONE"]},
        "assets_asset.is_gated": {"semantic_type": "FLAG", "sample_values": ["true", "false"]},
        # a real column whose name shares the ENTITY's word: "property type" must not land here
        "assets_asset.corner_property": {"semantic_type": "CATEGORY", "sample_values": ["true", "false"]},
        "assets_asset.carpet_area": {"semantic_type": "METRIC"},
        "assets_asset.total_floors": {"semantic_type": "METRIC"},
        "assets_asset.asset_type_id": {"semantic_type": "IDENTIFIER"},
        "assets_asset.floor_number": {"semantic_type": "METRIC"},
        "assets_asset.created_at": {"semantic_type": "TEMPORAL"},
        "assets_assettype.id": {"semantic_type": "IDENTIFIER"},
        "assets_assettype.name": {"semantic_type": "CATEGORY",
                                  "sample_values": ["Apartment / Condo", "Villa", "Office Space"]},
        "assets_leaseunit.id": {"semantic_type": "IDENTIFIER"},
        "assets_leaseunit.name": {"semantic_type": "CATEGORY"},
        "assets_leaseunit.asset_id": {"semantic_type": "IDENTIFIER"},
        "assets_leaselisting.id": {"semantic_type": "IDENTIFIER"},
        "assets_leaselisting.expected_monthly_rent": {"semantic_type": "MONETARY"},
        "assets_leaselisting.lease_unit_id": {"semantic_type": "IDENTIFIER"},
        "assets_leaselisting.status": {"semantic_type": "CATEGORY"},
        "vendors.id": {"semantic_type": "IDENTIFIER"},
        "vendors.name": {"semantic_type": "CATEGORY"},
        "vendors.city": {"semantic_type": "CATEGORY"},
    },
}

EDGES = [
    # the declared FK the lane must use …
    {"source_table": "assets_asset", "source_column": "asset_type_id", "target_table": "assets_assettype",
     "target_column": "id", "discovery": "declared_fk", "polymorphic": False, "cardinality": "N:1"},
    # … and the data-inferred noise the real graph carries, which it must NOT join over
    {"source_table": "assets_asset", "source_column": "floor_number", "target_table": "assets_assettype",
     "target_column": "id", "discovery": "data_inferred", "polymorphic": False, "cardinality": "N:1"},
    {"source_table": "assets_leaseunit", "source_column": "asset_id", "target_table": "assets_asset",
     "target_column": "id", "discovery": "declared_fk", "polymorphic": False, "cardinality": "1:1"},
    {"source_table": "assets_leaselisting", "source_column": "lease_unit_id", "target_table": "assets_leaseunit",
     "target_column": "id", "discovery": "declared_fk", "polymorphic": False, "cardinality": "1:1"},
]


def _vocab():
    v = ScopeVocab()
    v.cards = {
        "assets_asset": {"table": "assets_asset", "business_name": "property", "plural": "properties",
                         "aliases": ["asset", "assets", "unit", "units", "property", "real estate"],
                         "key_dimensions": ["project_name", "city_name"],
                         "key_measures": ["carpet_area", "total_floors"], "display_column": "project_name",
                         "parent_entities": ["assets_assettype"]},
        "assets_assettype": {"table": "assets_assettype", "business_name": "asset type",
                             "plural": "asset types", "aliases": ["asset category", "type of asset"],
                             "display_column": "name", "parent_entities": []},
        "assets_leaseunit": {"table": "assets_leaseunit", "business_name": "lease unit",
                             "plural": "lease units", "aliases": [], "display_column": "name",
                             "parent_entities": ["assets_asset"]},
        "assets_leaselisting": {"table": "assets_leaselisting", "business_name": "lease listing",
                                "plural": "lease listings", "aliases": ["rental listing"],
                                "key_measures": ["expected_monthly_rent"],
                                "parent_entities": ["assets_leaseunit"]},
        "vendors": {"table": "vendors", "business_name": "vendor", "plural": "vendors",
                    "aliases": [], "display_column": "name", "parent_entities": []},
    }
    v.measure_glossary = {
        "assets_asset.carpet_area": {"type": "METRIC", "phrases": ["carpet area", "area"]},
        "assets_asset.total_floors": {"type": "METRIC", "phrases": ["total floors", "floors"]},
        "assets_leaselisting.expected_monthly_rent": {"type": "MONETARY",
                                                      "phrases": ["expected monthly rent", "rent", "monthly rental rate"]},
    }
    v.value_glossary = {}
    v.source_of = {"assets_asset": "2", "assets_assettype": "2", "assets_leaseunit": "2",
                   "assets_leaselisting": "2", "vendors": "4"}
    return v


# the sampled value store (query.resolution.typed_value_lookup), as (bare table, column, raw)
SAMPLED = {"kochi": [("assets_asset", "city_name", "Kochi")],
           "acme plumbing": [("vendors", "name", "Acme Plumbing")]}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(C, "_edges", lambda vocab, table: EDGES)
    # the AST firewall's join-integrity guard reads the relationship graph (veda.runtime)
    import veda.runtime as R
    _g = {"edges": EDGES}
    monkeypatch.setattr(R, "get_graph", lambda *a, **k: _g)
    monkeypatch.setattr(C, "_sampled_owners", lambda value: SAMPLED.get(str(value).lower(), []))
    monkeypatch.setattr(C, "_embedding_owners", lambda value, columns: [])
    monkeypatch.setattr(C, "_live_domain", lambda vocab, sm, t, c: [])
    _cd = G.column_domain
    monkeypatch.setattr(G, "column_domain", lambda vocab, sm, t, c, live=True: _cd(vocab, sm, t, c, live=False))
    # the lexical qualifier gate must never judge a continuity IR
    import veda.validation as V
    monkeypatch.setattr(V, "qualifier_completeness",
                        lambda *a, **k: pytest.fail("lexical qualifier gate ran on a continuity IR"))


# prior turn: "how many properties are there in each city" → count grouped by city_name
PRIOR = {"entity_table": "assets_asset", "source_id": 2, "group_by": ["city_name"],
         "aggregation": "count", "route": "frame"}


def _ctx(message, delta, **over):
    return {**copy.deepcopy(PRIOR), **over, "user_message": message, "delta": delta}


def _run(message, delta, slm=None, **over):
    return C.run_continuity(message, SM, _ctx(message, delta, **over), vocab=_vocab(), slm=slm)


def _firewall(res, message):
    fr = res.frame_result
    lane = FrameLane(fr)
    return firewall.check(fr.ir, fr.sql, SM, query=message, allowed_tables=set(lane.tables),
                          allowed_columns=lane.columns, head=fr.ir.head, run_rbac=True)


def _norm(sql):
    return re.sub(r"\s+", " ", sql).strip()


ROWS = []          # (case, op, grounded_to, sql, checks_run) — printed by the last test


def _record(case, res, message):
    v = _firewall(res, message)
    assert v.ok, v.reason
    assert "qualifier:lexical_skipped(continuity_ir_complete)" in v.checks_run
    assert "alignment:text_guards_skipped(continuity_ir_complete)" in v.checks_run
    assert {"value_grounding", "rbac", "ast_parameterize"} <= set(v.checks_run)
    ROWS.append((case, res.frame_result.ir.head, res.trace.get("grounded_to"), _norm(res.frame_result.sql),
                 v.checks_run))
    return v


# ── the five follow-ups ──────────────────────────────────────────────────────────────
def test_only_mumbai_adds_the_filter_to_the_prior_ir():
    res = _run("only Mumbai", {"op": "add_filter", "slot": "filters", "concept": "city_name",
                               "value": "Mumbai", "confidence": 0.95, "applied": False})
    assert res.kind == "sql", res
    ir = res.frame_result.ir
    assert ir.head == "continuity.add_filter" and ir.ir_partial is False
    assert ir.anchor == "assets_asset" and ir.group_keys == ["city_name"]
    assert ir.measure.aggregation == "count"
    assert [(f.column, f.op, f.value) for f in ir.filters] == [("city_name", "=", "Mumbai")]
    sql = _norm(res.frame_result.sql)
    assert "WHERE LOWER(CAST(t0.\"city_name\" AS TEXT)) = 'mumbai'" in sql
    assert "GROUP BY t0.\"city_name\"" in sql
    assert res.trace["grounded_to"] == "assets_asset.city_name" and res.trace["method"] == "domain"
    _record("only Mumbai", res, "only Mumbai")


def test_value_without_a_column_grounds_by_its_owner():
    res = _run("only the ones in Kochi", {"op": "add_filter", "value": "Kochi", "applied": False})
    assert res.kind == "sql"
    assert [(f.column, f.value) for f in res.frame_result.ir.filters] == [("city_name", "Kochi")]


def test_break_down_by_property_type_groups_by_the_parent_display_column():
    res = _run("break that down by property type",
               {"op": "change_group", "slot": "group_keys", "concept": "property type",
                "confidence": 0.95, "applied": False})
    assert res.kind == "sql", res
    sql = _norm(res.frame_result.sql)
    # joined over the DECLARED FK asset_type_id, never the data-inferred floor_number edge
    assert 'LEFT JOIN "assets_assettype" t1 ON t1."id" = t0."asset_type_id"' in sql
    assert "floor_number" not in sql and "corner_property" not in sql
    assert 'GROUP BY t1."name"' in sql and "COUNT(*)" in sql
    assert res.frame_result.ir.group_keys == ["name"]
    assert res.trace["grounded_to"] == "assets_assettype.name"
    assert res.trace["method"] == "parent_display"
    _record("break that down by property type", res, "break that down by property type")


def test_top_3_orders_the_count_desc_with_limit_3():
    res = _run("show me just the top 3", {"op": "change_order", "slot": "limit", "value": 3,
                                          "confidence": 0.95, "applied": False})
    assert res.kind == "sql", res
    sql = _norm(res.frame_result.sql)
    assert sql.endswith('ORDER BY "count" DESC LIMIT 3')
    ir = res.frame_result.ir
    assert ir.limit == 3 and ir.order["direction"] == "desc"
    _record("top 3", res, "show me just the top 3")


def test_go_back_compiles_the_prior_of_prior_as_sent():
    # turn 2 was "only Mumbai" — its context carries the filter
    t2 = {"filters": [{"column": "city_name", "operator": "equals", "value": "mumbai"}]}
    back = _run("go back", {"op": "drill_up", "confidence": 1.0, "applied": True},
                **{**PRIOR})                                    # the chat popped it: turn 1's state
    assert back.kind == "sql"
    assert back.frame_result.ir.filters == []
    assert "WHERE" not in back.frame_result.sql
    assert back.frame_result.ir.head == "continuity.drill_up"
    # same IR as the prior itself (the hash ignores the head)
    first = _run("x", {"op": "change_group", "concept": "city_name", "applied": True})
    assert back.trace["ir_hash"] == first.trace["ir_hash"]
    # an UNAPPLIED drill_up (the chat had no level to pop) drops one level here
    one = _run("go back", {"op": "drill_up", "applied": False}, **t2)
    assert one.kind == "sql" and one.frame_result.ir.filters == []
    _record("go back", back, "go back")


def test_average_rent_of_those_clarifies_naming_the_listing_measure():
    res = _run("what is the average rent of those",
               {"op": "change_measure", "slot": "measure", "value": "avg", "confidence": 0.8,
                "applied": False})
    assert res.kind == "clarify", res
    assert "expected monthly rent" in res.message and "lease listings" in res.message
    _no_identifiers(res.message)
    assert res.candidates and all("_" not in c for c in res.candidates)
    ROWS.append(("average rent of those", "continuity.change_measure", "clarify", res.message, []))


def test_average_of_an_anchor_measure_compiles():
    res = _run("what is the average carpet area of those",
               {"op": "change_measure", "value": "avg", "applied": False})
    assert res.kind == "sql", res
    sql = _norm(res.frame_result.sql)
    assert 'AVG(t0."carpet_area")' in sql and 'GROUP BY t0."city_name"' in sql
    _record("average carpet area of those", res, "what is the average carpet area of those")


# ── pre-applied vs lane-applied ──────────────────────────────────────────────────────
def test_pre_applied_delta_is_not_applied_twice():
    ctx = {"filters": [{"column": "city_name", "operator": "equals", "value": "Pune"}]}
    res = _run("what about Pune", {"op": "replace", "concept": "city_name", "value": "Pune",
                                   "applied": True}, **ctx)
    assert res.kind == "sql"
    assert [(f.column, f.value) for f in res.frame_result.ir.filters] == [("city_name", "Pune")]


def test_lane_applied_add_filter_is_idempotent_and_replaces_same_column():
    ctx = {"filters": [{"column": "city_name", "operator": "equals", "value": "Mumbai"}]}
    same = _run("only Mumbai", {"op": "add_filter", "concept": "city_name", "value": "Mumbai",
                                "applied": False}, **ctx)
    assert len(same.frame_result.ir.filters) == 1
    swap = _run("only Pune", {"op": "add_filter", "concept": "city_name", "value": "Pune",
                              "applied": False}, **ctx)
    assert [(f.column, f.value) for f in swap.frame_result.ir.filters] == [("city_name", "Pune")]
    assert swap.trace.get("replaced") == ["Mumbai"]


def test_unapplied_replace_with_a_value_is_a_filter_and_keeps_the_grouping():
    """Live 2026-09-27: after "how many properties are there in each city", the classifier
    sent "only Mumbai" as replace city_name=Mumbai, unapplied (no city filter existed). It is
    the same edit as add_filter — and the per-city grouping must survive."""
    res = _run("only Mumbai", {"op": "replace", "concept": "city_name", "value": "Mumbai",
                               "applied": False})
    assert res.kind == "sql", res
    ir = res.frame_result.ir
    assert [(f.column, str(f.value).lower()) for f in ir.filters] == [("city_name", "mumbai")]
    assert ir.group_keys == ["city_name"]
    assert res.trace["op"] == "replace→add_filter"


def test_unapplied_replace_without_a_value_goes_to_the_one_call():
    calls = []

    def slm(user, **kw):
        calls.append(kw)
        return '{"op": "change_group", "slot": "group_keys", "concept": "furnishing", "value": null}'
    res = _run("by furnishing instead", {"op": "replace", "concept": "group_by", "applied": False},
               slm=slm)
    assert res.kind == "sql", res
    assert len(calls) == 1 and res.frame_result.ir.group_keys == ["furnishing"]


def test_agent_planned_prior_is_edited_by_the_lane_not_declined():
    """Live 2026-09-27: every follow-up of an agent-planned first turn was declined to a
    4–5 call agent re-plan. The context carries the same state; the lane edits it."""
    res = _run("only Mumbai", {"op": "add_filter", "concept": "city_name", "value": "Mumbai",
                               "applied": False},
               agent_plan={"tables": ["assets_asset"], "group_by": ["assets_asset.city_name"]})
    assert res.kind == "sql", res
    assert res.frame_result.ir.group_keys == ["city_name"]


def test_unapplied_switch_frame_declines():
    res = _run("the earlier one", {"op": "switch_frame", "applied": False})
    assert res.kind == "decline" and res.reason == "not_applied:switch_frame"


# ── ambiguous: one constrained SLM call ──────────────────────────────────────────────
def test_ambiguous_uses_one_constrained_slm_call():
    calls = []

    def slm(user, **kw):
        calls.append(kw)
        return '{"op": "change_group", "slot": "group_keys", "concept": "furnishing", "value": null}'
    res = _run("and how are they furnished", {"op": "ambiguous", "confidence": 0.0, "applied": False},
               slm=slm)
    assert res.kind == "sql", res
    assert len(calls) == 1
    assert calls[0]["purpose"] == "continuity_delta" and calls[0]["temperature"] == 0.0
    assert calls[0]["json_schema"]["properties"]["op"]["enum"][:5] == list(C.LANE_OPS)
    assert res.frame_result.ir.group_keys == ["furnishing"]
    assert res.trace["op"] == "ambiguous→change_group"
    _record("ambiguous → furnishing (stub SLM)", res, "and how are they furnished")


def test_ambiguous_slm_none_declines():
    res = _run("hmm", {"op": "ambiguous", "applied": False},
               slm=lambda user, **kw: '{"op": "none", "concept": null, "value": null}')
    assert res.kind == "decline" and res.reason == "ambiguous_unresolved"


# ── clarify copy, decline ────────────────────────────────────────────────────────────
def _no_identifiers(msg):
    assert "_" not in msg, msg
    for ident in ("assets_asset", "assettype", "leaselisting", "city_name", "asset_type_id"):
        assert ident not in msg, msg


def test_unknown_value_clarifies_in_business_words():
    res = _run("only Mumbay", {"op": "add_filter", "concept": "city_name", "value": "Mumbay",
                               "applied": False})
    assert res.kind == "clarify"
    _no_identifiers(res.message)
    assert "Mumbai" in res.candidates and len(res.candidates) <= 5
    assert "city name" in res.message and "properties" in res.message


def test_unknown_dimension_clarifies_listing_dimensions():
    res = _run("break it down by colour", {"op": "change_group", "concept": "colour", "applied": False})
    assert res.kind == "clarify"
    _no_identifiers(res.message)
    assert 0 < len(res.candidates) <= 5 and "asset type" in res.candidates


def test_decline_when_the_message_names_another_entity():
    res = _run("which vendors handled them", {"op": "ambiguous", "applied": False},
               slm=lambda *a, **k: pytest.fail("no SLM call for an out-of-scope entity"))
    assert res.kind == "decline" and res.reason == "names_other_entity:vendors"


def test_decline_when_the_value_belongs_to_another_entity():
    res = _run("only Acme Plumbing", {"op": "add_filter", "value": "Acme Plumbing", "applied": False})
    assert res.kind == "decline" and res.reason == "value_outside_entity"


def test_decline_when_the_prior_cannot_be_rebuilt():
    res = _run("only Pune", {"op": "add_filter", "value": "Pune", "applied": False},
               filters=[{"column": "no_such_column", "operator": "equals", "value": "x"}])
    assert res.kind == "decline" and res.reason.startswith("prior_unreconstructable")


def test_inactive_for_new_topic_and_first_turns():
    assert not C.is_active({})
    assert not C.is_active({"user_message": "x"})
    assert not C.is_active({"entity_table": "assets_asset", "delta": {"op": "new_topic"}})
    assert C.is_active({"entity_table": "assets_asset", "delta": {"op": "add_filter"}})


# ── firewall: continuity heads are structural-only ───────────────────────────────────
def test_firewall_still_refuses_a_continuity_sql_that_drops_its_filter():
    res = _run("only Mumbai", {"op": "add_filter", "concept": "city_name", "value": "Mumbai",
                               "applied": False})
    fr = res.frame_result
    bad = 'SELECT t0."city_name", COUNT(*) AS "count" FROM "assets_asset" t0 GROUP BY t0."city_name"'
    v = firewall.check(fr.ir, bad, SM, query="only Mumbai", allowed_tables={"assets_asset"},
                       allowed_columns=["city_name"], head=fr.ir.head)
    assert v.verdict == firewall.QUALIFIER_DROPPED and v.slot == "filter:city_name"


def test_frame_heads_keep_their_label():
    from veda.ir import QueryIR
    assert firewall._compiled_head(QueryIR(anchor="t", head="frame.name")) == "frame"
    assert firewall._compiled_head(QueryIR(anchor="t", head="continuity.add_filter")) == "continuity"
    assert firewall._compiled_head(QueryIR(anchor="t", head="branch.x")) is None


# ── explain compact ──────────────────────────────────────────────────────────────────
def test_explain_compact_carries_continuity_op():
    from veda.explain import ExplainTrace
    t = ExplainTrace("only Mumbai")
    t.set("continuity", op="add_filter", declined=None)
    assert t.compact()["continuity_op"] == "add_filter"


# ── the wire: inference validator ────────────────────────────────────────────────────
def test_inference_validator_accepts_and_shape_checks_delta():
    from inference.routes.hybrid import _validated_conversation_context as val

    def v(delta):
        return (val({"conversation_context": {"user_message": "x", "entity_table": "t",
                                              "delta": delta}}) or {}).get("delta")
    good = {"op": "add_filter", "slot": "filters", "concept": "city", "value": "Mumbai",
            "confidence": 0.95, "applied": False}
    assert v(good) == good
    assert v({"op": "change_order", "value": 3})["value"] == 3
    assert v({"op": "drill_up", "applied": True}) == {"op": "drill_up", "applied": True}
    for bad in ({"op": "drop_table"}, {"op": "add_filter", "extra": 1},
                {"op": "add_filter", "value": {"x": 1}}, {"op": "add_filter", "confidence": 3},
                {"op": "add_filter", "applied": "yes"}, {"op": "add_filter", "concept": "x" * 300},
                "add_filter", ["add_filter"]):
        assert v(bad) is None, bad


# ── the chat side: wire_delta ────────────────────────────────────────────────────────
def test_wire_delta_marks_what_the_chat_already_applied():
    from chatbot.memory.context import wire_delta
    rule = {"op": "add_filter", "slot": "filters", "concept": "city", "value": "Mumbai",
            "confidence": 0.95}
    d = wire_delta(rule, "drill_down")
    assert d == {"op": "add_filter", "slot": "filters", "concept": "city", "value": "Mumbai",
                 "confidence": 0.95, "applied": False}
    assert wire_delta({"op": "drill_up", "confidence": 1.0}, "drill_up", chat_applied=True)["applied"]
    assert wire_delta(None, "remove", "city", "", chat_applied=True)["op"] == "remove_filter"
    r = wire_delta(None, "replace", "limit", "10", chat_applied=True)
    assert r["op"] == "replace" and r["applied"] is True
    # a shape delta the chat applied wins over the rule op it came from
    s = wire_delta({"op": "change_group", "concept": "city", "confidence": 0.95}, "replace",
                   "group_by", "city", chat_applied=True)
    assert s["op"] == "replace" and s["applied"] is True
    assert wire_delta({"op": "ambiguous", "confidence": 0.0}, "refine")["op"] == "ambiguous"
    assert wire_delta(None, "new_topic")["op"] == "new_topic"
    assert wire_delta(None, "refine", comparison=True)["op"] == "compare"


def test_zz_print_unit_table():
    for case, op, g, sql, checks in ROWS:
        print(f"| {case} | {op} | {g} | {sql} | {', '.join(checks)} |")


# ── the chat side, graph-level: the delta reaches the engine on every follow-up ────────
from test_chat_graph_contract import session, _FakeInference   # noqa: E402,F401
import test_chat_graph_contract as _gc                               # noqa: E402


def test_chat_sends_the_delta_through_the_real_graph(session, monkeypatch):
    turn, _slm_calls = session
    monkeypatch.setattr(_gc, "REPLIES", _gc.REPLIES + [_gc.REPLIES[1]])
    turn("how many vendors are there")
    turn("only in Mumbai")
    turn("group by city")
    turn("go back")
    calls = _FakeInference.calls
    assert calls[0]["flags"] is None                       # a first turn carries nothing
    d2, d3, d4 = (_gc._ctx(c)["delta"] for c in calls[1:4])
    # "only in Mumbai": the rule layer's add_filter, grounded in turn 1's own values —
    # NOT applied by the chat (the frame has no filter yet), so the engine applies it
    assert d2["op"] == "add_filter" and d2["value"] == "Mumbai" and d2["applied"] is False
    assert not _gc._ctx(calls[1]).get("filters")
    # "group by city": a known dimension → change_group, lane-applied
    assert d3["op"] == "change_group" and d3["concept"] == "city" and d3["applied"] is False
    # "go back": the chat popped the drill level itself → pre-applied, compiled as sent
    assert d4["op"] == "drill_up" and d4["applied"] is True
    # and every delta survives the inference boundary's validator unchanged
    for c in calls[1:4]:
        wire = _gc._validated_conversation_context(c["flags"])
        assert wire["delta"] == _gc._ctx(c)["delta"]


# ── the pipeline hook: stage 0, before the frame path ────────────────────────────────
def test_pipeline_returns_the_lane_clarify_before_any_other_head(monkeypatch):
    from veda import pipeline
    from veda_core.context import set_conversation_context
    seen = {}

    def fake(query, sm, conv, **k):
        seen["conv"] = conv
        return C.ContinuityResult("clarify", "unresolved:group_by", message="Break the properties down by what?",
                                  trace={"op": "change_group", "declined": None})
    monkeypatch.setattr(C, "run_continuity", fake)
    monkeypatch.setattr("veda.understanding.frame_path.run_frame_path",
                        lambda *a, **k: pytest.fail("the frame path ran on a continuity turn"))
    tok = set_conversation_context(_ctx("break it down", {"op": "change_group", "concept": "colour",
                                                          "applied": False}))
    try:
        out = pipeline.run_query("break it down", SM, [], return_result=True)
    finally:
        set_conversation_context(None)
    assert seen["conv"]["entity_table"] == "assets_asset"
    assert out["status"] == "clarify" and out["msg"] == "Break the properties down by what?"
    assert out["trace"]["sections"]["continuity"]["op"] == "change_group"


# ── live 2026-09-27 fixes ────────────────────────────────────────────────────────────
def test_ambiguous_with_a_stored_value_is_a_filter_without_a_model_call():
    """"only Mumbai" arrived as ambiguous; the one call read it as remove_filter. A value the
    entity holds is a filter — no model call."""
    calls = []

    def slm(user, **kw):
        calls.append(kw)
        return '{"op": "remove_filter", "slot": "filters", "concept": null, "value": null}'
    res = _run("only Mumbai", {"op": "ambiguous", "applied": False}, slm=slm)
    assert res.kind == "sql", res
    assert calls == []
    ir = res.frame_result.ir
    assert [(f.column, str(f.value).lower()) for f in ir.filters] == [("city_name", "mumbai")]
    assert ir.group_keys == ["city_name"]


def test_prior_grouped_by_a_parent_display_column_is_rebuilt_when_qualified():
    """"top 3" after "break that down by property type": the prior grouping is the asset
    type's `name`, ambiguous as a bare name; remembered qualified it rebuilds."""
    res = _run("top 3", {"op": "change_order", "applied": False},
               group_by=["assets_assettype.name"])
    assert res.kind == "sql", res
    sql = res.frame_result.sql
    assert 'JOIN "assets_assettype"' in sql and 'LIMIT 3' in sql and 'GROUP BY t1."name"' in sql.replace('"t1"', 't1')


def test_explain_group_operation_carries_its_table_and_chat_qualifies_a_parent_column():
    from veda.business_explain import build_explain
    import chatbot.memory.frame as MF
    sql = ('SELECT "t1"."name", COUNT(*) AS "count" FROM "assets_asset" AS "t0" '
           'LEFT JOIN "assets_assettype" AS "t1" ON "t1"."id" = "t0"."asset_type_id" '
           'GROUP BY "t1"."name" ORDER BY "count" DESC')
    ex = build_explain(sql=sql, table="assets_asset", sm=SM, params=[])
    ops = [o for o in (ex.get("operations") or []) if o.get("type") == "group"]
    assert ops and ops[0].get("table") == "assets_assettype"
    fr = MF.harvest_frame({"explain": ex, "table": "assets_asset", "sql": sql,
                           "rows": [["Villa", 3]], "status": "answered"}) or {}
    assert fr.get("group_by") == ["assets_assettype.name"]
