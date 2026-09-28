"""Unit tests for the planner agent (veda/agent/): validation contract, budgets, the six
worked traces, and the fast lane staying first.

No SLM, no DB: the model is a stub that replays recorded actions and the tool box is a
stub that returns recorded results. Run:
    python -m pytest tests/test_planner_agent.py -q
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from veda.agent.tools import ToolBox  # noqa: E402
from veda.agent import plan as P  # noqa: E402
from veda.agent import planner as PL  # noqa: E402
from veda.agent.examples import TRACES  # noqa: E402

# ── the illustrative shop schema the worked traces use ───────────────────────────────
SHOP = {
    "tables": {t: {"table_type": "MASTER"} for t in ("orders", "order_items", "customers", "stores", "tickets", "staff")},
    "columns": {k: {"semantic_type": v} for k, v in {
        "orders.id": "IDENTIFIER", "orders.order_no": "IDENTIFIER", "orders.placed_at": "TEMPORAL",
        "orders.status": "CATEGORY", "orders.total": "MONETARY", "orders.channel": "CATEGORY",
        "orders.customer_id": "IDENTIFIER", "orders.store_id": "IDENTIFIER",
        "order_items.id": "IDENTIFIER", "order_items.order_id": "IDENTIFIER", "order_items.sku": "IDENTIFIER",
        "order_items.quantity": "METRIC",
        "customers.id": "IDENTIFIER", "customers.full_name": "FREE_TEXT", "customers.city": "CATEGORY",
        "customers.joined_on": "TEMPORAL", "customers.segment": "CATEGORY",
        "stores.id": "IDENTIFIER", "stores.name": "FREE_TEXT", "stores.region": "CATEGORY",
        "tickets.id": "IDENTIFIER", "tickets.category": "CATEGORY", "tickets.state": "CATEGORY",
        "tickets.opened_at": "TEMPORAL", "tickets.title": "FREE_TEXT", "tickets.priority": "CATEGORY",
        "tickets.assigned_to_id": "IDENTIFIER", "tickets.created_by_id": "IDENTIFIER",
        "staff.id": "IDENTIFIER", "staff.full_name": "FREE_TEXT", "staff.team": "CATEGORY",
    }.items()},
}


class StubBox(ToolBox):
    """Returns recorded results by (tool, args); an unrecorded probe gets generic counts."""

    def __init__(self, trace):
        super().__init__(SHOP, None, trace["q"], ctx=None)
        self.rec = {}
        self.rec[("find_entities", json.dumps({"phrase": trace["q"], "k": 5}, sort_keys=True))] = \
            {"phrase": trace["q"], "entities": copy.deepcopy(trace["entities"])}
        for st in list(trace.get("auto") or []) + list(trace["steps"]):
            self.rec[(st["tool"], json.dumps(st["args"], sort_keys=True))] = copy.deepcopy(st["result"])

    def tables(self):
        return sorted(SHOP["tables"])

    def call(self, tool, args=None, *, auto=False):
        args = dict(args or {})
        cid = f"c{len(self.log) + 1}"
        r = self.rec.get((tool, json.dumps(args, sort_keys=True)))
        if r is None and tool == "probe":
            fl = [dict(f, rows=50) for f in args.get("filters") or []]
            r = {"rows_total": 900, **({"rows_after_filters": 50, "filters": fl} if fl else {})}
            if args.get("order"):
                r["distinct_order_col"] = 800
        if r is None and tool == "retrieve":
            r = {"columns": [], "tables": []}
        if r is None:
            r = {"error": f"unrecorded {tool} {args}"}
        for o in (r.get("routes") or []) if isinstance(r, dict) else []:
            self.routes[o["id"]] = P.route_edges(o)
        self.log.append({"id": cid, "tool": tool, "args": args, "result": r, "ms": 0.0, "auto": auto})
        return cid, r


def replay_slm(trace):
    acts = [{"thought": st["thought"], "action": {"tool": st["tool"], "args": st["args"]}}
            for st in trace["steps"]]
    from veda.agent.examples import full_final
    acts.append({"thought": "plan ready", "action": {"final": full_final(trace["final"])}})
    it = iter(acts)

    def slm(system, user, schema, timeout):
        assert "RULES" in system and "PLAN" in system
        return json.dumps(next(it)), {"prompt_tokens": 1, "completion_tokens": 1}
    return slm


# ── (5) the six worked traces replay to their expected plans ─────────────────────────
@pytest.mark.parametrize("i", range(len(TRACES)))
def test_worked_traces_replay_to_expected_plans(i):
    tr = TRACES[i]
    box = StubBox(tr)
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=replay_slm(tr), wall_s=60)
    if tr["final"].get("kind") == "rag":
        assert r.kind == "rag"
        return
    assert r.kind == "sql", (r.reason, r.validation)
    expected = P.from_model(tr["final"], box.log)
    got = r.plan.to_dict()
    exp = expected.to_dict()
    for k in ("tables", "joins", "projection", "filters", "group_by", "aggregates", "order", "limit", "time"):
        assert got[k] == exp[k], k
    assert r.validation[-1]["errors"] == []
    assert r.compiled.ir.head == "frame.agent" and not r.compiled.ir.ir_partial
    assert "SELECT" in r.compiled.sql
    # the replay never needed a second attempt
    assert len(r.steps) == len(tr["steps"]) + 1


def test_worked_trace_sql_shapes():
    """Spot-check the compiled SQL goes through the existing builders."""
    tr = TRACES[4]                       # two group keys over the assigned-to edge
    box = StubBox(tr)
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=replay_slm(tr), wall_s=60)
    sql = r.compiled.sql
    assert 'JOIN "staff" t1 ON t1."id" = t0."assigned_to_id"' in sql
    assert 'GROUP BY t0."category", t1."full_name"' in sql
    tr = TRACES[1]                       # single-table scalar aggregate → build_aggregate_sql
    box = StubBox(tr)
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=replay_slm(tr), wall_s=60)
    assert r.compiled.sql.startswith('SELECT AVG("total") AS avg_result FROM "orders" t0 WHERE')


# ── (1)–(3) the validation contract ──────────────────────────────────────────────────
def _log_of(trace):
    box = StubBox(trace)
    box.call("find_entities", {"phrase": trace["q"], "k": 5})
    for st in list(trace.get("auto") or []) + list(trace["steps"]):
        box.call(st["tool"], st["args"])
    return box.log


def test_table_not_in_tool_log_is_rejected():
    log = _log_of(TRACES[0])
    plan = P.from_model({"tables": ["invoices"], "select": ["invoices.id"]}, log)
    errs = P.validate(plan, log, SHOP)
    assert any("table invoices was not returned" in e for e in errs)
    # arguments alone never license an identifier: a describe that ERRORED doesn't count
    log2 = log + [{"id": "c99", "tool": "describe", "args": {"table": "invoices"},
                   "result": {"error": "unknown table"}}]
    assert any("invoices" in e for e in P.validate(plan, log2, SHOP))


def test_join_not_returned_by_join_path_is_rejected():
    log = _log_of(TRACES[2])
    ok = P.from_model(TRACES[2]["final"], log)
    assert P.validate(ok, log, SHOP) == []
    # a route id brings every edge of its route and the table it passes through
    assert ok.tables == ["order_items", "customers", "orders"] and len(ok.joins) == 2
    bad = copy.deepcopy(ok)
    bad.joins[0] = dict(bad.joins[0], via="r7")
    errs = P.validate(bad, log, SHOP)
    assert any("was not returned by join_path" in e for e in errs)
    # the right route id with different keys is also rejected
    bad.joins[0] = dict(ok.joins[0], on=[("id", "id")])
    assert any("keys" in e for e in P.validate(bad, log, SHOP))
    # tables and no join at all
    bad.joins = []
    assert any("not joined" in e for e in P.validate(bad, log, SHOP))


def test_filter_value_not_from_values_or_probe_is_rejected():
    log = _log_of(TRACES[0])
    plan = P.from_model({"tables": ["orders"], "filters": [
        {"col": "orders.status", "op": "=", "value": "REFUNDED"}]}, log)
    errs = P.validate(plan, log, SHOP)
    assert any("REFUNDED" in e for e in errs)
    # a value that values()/describe returned is licensed
    plan = P.from_model({"tables": ["orders"], "filters": [
        {"col": "orders.status", "op": "=", "value": "cancelled"}]}, log)
    assert P.validate(plan, log, SHOP) == []
    # a probe that matched rows licenses a value; one that matched 0 does not
    for rows, ok in ((12, True), (0, False)):
        log2 = log + [{"id": "c50", "tool": "probe", "args": {}, "result": {
            "rows_total": 900, "rows_after_filters": rows,
            "filters": [{"col": "orders.channel", "op": "=", "value": "PHONE", "rows": rows}]}}]
        plan = P.from_model({"tables": ["orders"], "filters": [
            {"col": "orders.channel", "op": "=", "value": "PHONE"}]}, log2)
        assert (P.validate(plan, log2, SHOP) == []) is ok
    # a numeric comparison must have been probed
    plan = P.from_model({"tables": ["orders"], "filters": [
        {"col": "orders.total", "op": ">", "value": 10}]}, log)
    assert any("never probed" in e for e in P.validate(plan, log, SHOP))


def test_fanned_out_sum_is_rejected():
    log = _log_of(TRACES[3])
    plan = P.from_model({"tables": ["stores", "orders"], "joins": ["r1"], "group_by": ["stores.region"],
                         "aggregates": [{"fn": "sum", "col": "orders.total"}]}, log)
    assert P.validate(plan, log, SHOP) == []          # summing the N side: fine
    plan = P.from_model({"tables": ["orders", "stores"], "joins": ["r1"],
                         "aggregates": [{"fn": "count", "col": "*"}]}, log)
    assert P.validate(plan, log, SHOP) == []          # counting the N side through N:1: fine
    plan = P.from_model({"tables": ["stores", "orders"], "joins": ["r1"],
                         "aggregates": [{"fn": "count", "col": "stores.id"}]}, log)
    assert any("multiplied" in e for e in P.validate(plan, log, SHOP))


# ── (4) budget exhaustion yields a clarify, never SQL ───────────────────────────────
def test_budget_exhaustion_is_a_clarify_never_sql():
    tr = TRACES[0]
    box = StubBox(tr)
    n = {"i": 0}

    def looping(system, user, schema, timeout):
        n["i"] += 1
        return json.dumps({"thought": "look again",
                           "action": {"tool": "find_entities", "args": {"phrase": f"orders {n['i']}"}}}), {}
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=looping, wall_s=60, max_steps=6, max_tool_calls=4)
    assert r.kind == "clarify" and r.compiled is None and r.message
    assert r.reason.startswith("budget:")
    assert r.budget["steps"] <= 6 and r.budget["tool_calls"] <= 5

    def bad_final(system, user, schema, timeout):
        return json.dumps({"thought": "x", "action": {"final": {"tables": ["nope"]}}}), {}
    box = StubBox(tr)
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=bad_final, wall_s=60)
    assert r.kind == "clarify" and r.compiled is None and len(r.validation) == 3


def test_probe_anomaly_buys_one_revision_then_clarifies():
    tr = TRACES[0]
    box = StubBox(tr)
    box.rec[("probe", json.dumps({"tables": ["orders"], "filters": [
        {"col": "orders.status", "op": "=", "value": "CANCELLED"}], "order": "orders.placed_at"}, sort_keys=True))] = {
        "rows_total": 900, "rows_after_filters": 0,
        "filters": [{"col": "orders.status", "op": "=", "value": "CANCELLED", "rows": 0}], "distinct_order_col": 870}
    fin = {"thought": "done", "action": {"final": tr["final"]}}
    seq = iter([{"thought": "p", "action": {"tool": "probe", "args": tr["steps"][0]["args"]}}, fin, fin])
    r = PL.run_planner(tr["q"], SHOP, None, toolbox=box, slm=lambda *a: (json.dumps(next(seq)), {}), wall_s=60)
    assert r.kind == "clarify" and r.reason == "probe" and "0 of 900" in r.message


def test_step_schema_enums_are_the_seen_identifiers():
    log = _log_of(TRACES[2])
    box = ToolBox(SHOP, None, "q", ctx=None)
    box.log = log
    sch = PL.step_schema(box)
    fin = [a for a in sch["properties"]["action"]["anyOf"] if "final" in a["properties"]][0]
    plan = fin["properties"]["final"]["anyOf"][0]
    tables = plan["properties"]["tables"]["items"]["enum"]
    assert set(tables) == {"order_items", "orders", "customers"}
    joins = plan["properties"]["joins"]["items"]["enum"]
    assert joins == ["r1"]
    # nothing joined yet → the plan cannot name a join at all
    box.log = _log_of(TRACES[0])
    plan0 = [a for a in PL.step_schema(box)["properties"]["action"]["anyOf"]
             if "final" in a["properties"]][0]["properties"]["final"]["anyOf"][0]
    assert plan0["properties"]["joins"]["maxItems"] == 0


# ── (6) the fast lane is not bypassed for a recognised shape ─────────────────────────
def test_fast_lane_not_bypassed_for_recognised_shape(monkeypatch):
    from test_frame_path import SM as FSM, _vocab
    from veda.understanding import frame_path as FP
    from veda.understanding import frame_extractor as FX
    from veda.understanding.frame import Frame
    import config as _cfg
    import veda.understanding.vocabulary as VOC
    import veda.agent.planner as APL
    monkeypatch.setattr(_cfg, "AGENT_PLANNER_ENABLED", True, raising=False)
    monkeypatch.setattr(_cfg, "FRAME_PROBES_ENABLED", False, raising=False)
    monkeypatch.setattr(VOC, "scope_vocab", lambda sm, *a, **k: _vocab())
    monkeypatch.setattr(FX, "extract_frame", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-extracted")))
    called = []
    monkeypatch.setattr(APL, "run_planner", lambda *a, **k: called.append(a) or (_ for _ in ()).throw(AssertionError("agent called")))
    fr = Frame(entity="sale listing", aggregation="count")
    fr.provenance["entity_table"] = "listing"
    tok = FP.inject_frame(fr)
    try:
        res = FP.run_frame_path("How many sale listings are there", FSM)
    finally:
        FP.reset_injected(tok)
    assert res.kind == "sql" and res.reason == "compiled" and not called
    assert "agent" not in res.trace


def test_declined_compile_goes_to_the_agent(monkeypatch):
    """The other half of (6): where the compiler declines, the agent is asked."""
    from test_frame_path import SM as FSM, _vocab
    from veda.understanding import frame_path as FP
    from veda.understanding import frame_compiler as FC
    from veda.understanding.frame import Frame
    import config as _cfg
    import veda.understanding.vocabulary as VOC
    import veda.agent.planner as APL
    monkeypatch.setattr(_cfg, "AGENT_PLANNER_ENABLED", True, raising=False)
    monkeypatch.setattr(_cfg, "FRAME_PROBES_ENABLED", False, raising=False)
    monkeypatch.setattr(VOC, "scope_vocab", lambda sm, *a, **k: _vocab())
    monkeypatch.setattr(FC, "compile_frame", lambda *a, **k: FC.Declined("order", "not expressible"))
    called = []

    def fake(q, sm, vocab, **k):
        called.append(q)
        return APL.AgentResult(kind="fail", reason="stub")
    monkeypatch.setattr(APL, "run_planner", fake)
    fr = Frame(entity="sale listing", aggregation="count")
    fr.provenance["entity_table"] = "listing"
    tok = FP.inject_frame(fr)
    try:
        res = FP.run_frame_path("How many sale listings are there", FSM)
    finally:
        FP.reset_injected(tok)
    assert called == ["How many sale listings are there"]
    assert res.kind == "clarify" and res.reason == "declined:order"      # agent failed → frame clarify
    assert res.trace["agent"]["trigger"] == "declined:order"


# ── chat follow-ups: the previous turn's plan is the draft, its log is attached ──────
def test_follow_up_edits_the_draft_with_the_prior_log():
    tr = TRACES[2]                                    # orders + customers, joined by e1
    first = PL.run_planner(tr["q"], SHOP, None, toolbox=StubBox(tr), slm=replay_slm(tr), wall_s=60)
    sec = first.trace()
    assert sec["draft"]["joins"] == ["r1"] and sec["draft"]["tables"] == ["order_items", "customers", "orders"]
    # exactly what chat memory keeps (chatbot/memory/frame._harvest_agent): tool, args, result
    prior = [{"tool": c["tool"], "args": c["args"], "result": c["result"]} for c in sec["tool_calls"]]
    # "of those, how many per city" — one step: edit the draft, reuse route r1 from the log
    edit = {"thought": "group the same join by city", "action": {"final": {
        "tables": ["order_items", "customers"], "joins": ["r1"], "group_by": ["customers.city"],
        "aggregates": [{"fn": "count", "col": "*"}]}}}
    box = StubBox({"q": "of those, how many per city", "entities": [], "steps": []})
    seen_user = {}

    def slm(system, user, schema, timeout):
        seen_user["u"] = user
        return json.dumps(edit), {}
    r = PL.run_planner("of those, how many per city", SHOP, None, toolbox=box, slm=slm, wall_s=60,
                       draft=sec["draft"], prior_log=prior, draft_question=tr["q"])
    assert r.kind == "sql", r.validation
    assert "DRAFT" in seen_user["u"] and len(r.steps) == 1
    assert 'GROUP BY t2."city"' in r.compiled.sql
    # every identifier came from the PRIOR log (ids p1…), none from a new tool call
    assert all(k.startswith("p") for k in r.plan.evidence["tool_call_ids"])
    assert r.trace()["tool_calls"][0]["id"] == "p1"


def test_conversation_context_carries_the_agent_plan():
    from chatbot.memory.context import ConversationContext
    from chatbot.memory import frame as MF
    sec = {"kind": "sql", "draft": {"tables": ["orders"]}, "question": "latest orders",
           "tool_calls": [{"tool": "describe", "args": {"table": "orders"}, "result": {"table": "orders"}}]}
    er = {"status": "answered", "analytics": {"row_count": 1}, "rows": [[1]], "cols": ["id"],
          "ir": {"anchor": "orders"}, "trace": {"sections": {"agent": sec}}}
    entry = MF.harvest_entry(er, "latest orders", 1)
    assert entry["agent"]["plan"] == {"tables": ["orders"]}
    ctx = ConversationContext.from_frame({"entity": "orders", "stack": [entry]}, "only cancelled ones")
    p = ctx.to_payload()
    assert p["agent_plan"] == {"tables": ["orders"]} and p["agent_question"] == "latest orders"
    assert p["agent_log"][0]["tool"] == "describe"
    # a new topic carries nothing
    assert "agent_plan" not in ConversationContext.from_frame(
        {"entity": "orders", "stack": [entry]}, "x", carry_state=False).to_payload()
    # a non-agent answer harvests no agent entry
    er2 = dict(er, trace={"sections": {}})
    assert "agent" not in MF.harvest_entry(er2, "q", 1)


def test_unlicensed_filters_and_limits():
    log = _log_of(TRACES[0])
    # a date filter the question never states ("latest" is an order, not a filter)
    plan = P.from_model({"tables": ["orders"], "filters": [
        {"col": "orders.placed_at", "op": ">", "value": "2023-12-31"}]}, log)
    log2 = log + [{"id": "c9", "tool": "probe", "args": {}, "result": {"rows_total": 900, "rows_after_filters": 10,
                   "filters": [{"col": "orders.placed_at", "op": ">", "value": "2023-12-31", "rows": 10}]}}]
    errs = P.validate(plan, log2, SHOP, question="latest cancelled orders")
    assert any("does not ask for it" in e for e in errs)
    assert not any("does not ask" in e for e in P.validate(plan, log2, SHOP, question="orders placed after 2023"))
    # a status value the question never names
    plan = P.from_model({"tables": ["orders"], "filters": [
        {"col": "orders.status", "op": "=", "value": "SHIPPED"}]}, log)
    assert any("does not ask" in e for e in P.validate(plan, log, SHOP, question="latest orders"))
    assert P.validate(plan, log, SHOP, question="latest shipped orders") == []
    # a row count the question does not write is dropped; a singular superlative keeps 1
    plan = P.from_model({"tables": ["orders"], "limit": 5}, log)
    P.license_limit(plan, "latest orders")
    assert plan.limit is None and plan.notes
    plan = P.from_model({"tables": ["orders"], "limit": 1}, log)
    P.license_limit(plan, "which order has the highest total?")
    assert plan.limit == 1
    plan = P.from_model({"tables": ["orders"], "limit": 10}, log)
    P.license_limit(plan, "top 10 orders by total")
    assert plan.limit == 10


def test_qualifier_gate_runs_before_acceptance():
    """A plan that drops a stated qualifier is sent back (the lexical gate is asked before
    the plan is accepted); the same plan with the filter passes."""
    log = _log_of(TRACES[1])
    q = "average order value for web orders over 500"
    good = P.from_model(TRACES[1]["final"], log)
    comp = P.compile_plan(good, SHOP)
    assert PL._post_compile_checks(good, comp, q, None, SHOP, None, log) == []
    bad = P.from_model({"tables": ["orders"], "filters": [{"col": "orders.channel", "op": "=", "value": "ONLINE"}],
                        "aggregates": [{"fn": "avg", "col": "orders.total"}]}, log)
    errs = PL._post_compile_checks(bad, P.compile_plan(bad, SHOP), q, None, SHOP, None, log)
    assert errs and "500" in errs[0] or "over" in errs[0]
    # a per-X breakdown the plan does not group by
    flat = P.from_model({"tables": ["orders"], "aggregates": [{"fn": "sum", "col": "orders.total"}]}, log)
    errs = PL._post_compile_checks(flat, P.compile_plan(flat, SHOP), "total order value per channel", None,
                                   SHOP, None, log)
    assert any("group_by" in e for e in errs)


def test_join_routes_rank_chains_before_link_tables_before_shared_parents():
    from veda.agent.tools import _routes
    E = lambda s, sc, t: {"source_table": s, "source_column": sc, "target_table": t, "target_column": "id",
                          "discovery": "declared_fk", "cardinality": "N:1"}
    edges = [E("listing", "unit_id", "unit"), E("unit", "asset_id", "asset"),          # chain
             E("doc", "asset_id", "asset"), E("doc", "listing_id", "listing"),         # link table
             E("asset", "country_id", "country"), E("listing", "country_id", "country"),  # shared parent
             {**E("listing", "created_by_id", "user"), "relationship_type": "audit"},
             {**E("asset", "created_by_id", "user"), "relationship_type": "audit"}]
    routes = _routes("asset", "listing", edges)
    kinds = [why.split(" via")[0] for _s, _p, why in routes]
    assert kinds[:3] == ["chain", "link table", "shared parent"]
    # an audit edge is never a stepping stone (asset → user → listing is not a relationship)
    assert not any("user" in why for _s, _p, why in routes)


def test_grouped_order_by_a_measured_column_means_its_aggregate():
    log = _log_of(TRACES[3])
    plan = P.from_model({"tables": ["orders", "stores"], "joins": ["r1"], "group_by": ["stores.name"],
                         "aggregates": [{"fn": "sum", "col": "orders.total"}],
                         "order": [{"by": "orders.total", "dir": "desc"}]}, log)
    assert plan.order == [{"expr": "sum_total", "dir": "desc"}]
    bad = P.from_model({"tables": ["orders", "stores"], "joins": ["r1"], "group_by": ["stores.name"],
                        "aggregates": [{"fn": "sum", "col": "orders.total"}],
                        "order": [{"by": "orders.placed_at", "dir": "desc"}]}, log)
    assert any("neither a group_by column" in e for e in P.validate(bad, log, SHOP))


def test_every_stated_number_is_used():
    log = _log_of(TRACES[1])
    q = "average order value for web orders over 500"
    dropped = P.from_model({"tables": ["orders"], "filters": [{"col": "orders.channel", "op": "=", "value": "ONLINE"}],
                            "aggregates": [{"fn": "avg", "col": "orders.total"}]}, log)
    errs = PL._post_compile_checks(dropped, P.compile_plan(dropped, SHOP), q, None, SHOP, None, log)
    assert any("number 500" in e for e in errs)
    assert P.stated_numbers("V-3 in 2025, above 5 lakh, top 10") == {2025.0, 500000.0, 10.0}
    # a changed number is named back to the model with the question's own number
    wrong = P.from_model({"tables": ["orders"], "filters": [{"col": "orders.total", "op": ">", "value": 450}]}, log)
    log2 = log + [{"id": "c9", "tool": "probe", "args": {}, "result": {"rows_total": 900, "rows_after_filters": 10,
                   "filters": [{"col": "orders.total", "op": ">", "value": 450, "rows": 10}]}}]
    errs = P.validate(wrong, log2, SHOP, question=q)
    assert any("it states 500" in e for e in errs)


def test_federated_plan_compiles_catalog_qualified(monkeypatch):
    """A plan whose route is a cross_source_fk edge compiles through the same builders with
    every table written catalog-qualified for the federated executor."""
    from veda.agent import federated as F
    import query.federated_route as FR
    monkeypatch.setattr(FR, "_catalog_table",
                        lambda sid, t, kind: (f'src_{sid}.public."{t}"' if kind == "postgres" else f'src_{sid}."{t}"'))
    sm = {"tables": {"assets_amenity": {"_source_id": "2"}, "amenities_catalog": {"_source_id": "5"}},
          "columns": {"assets_amenity.id": {"semantic_type": "IDENTIFIER"},
                      "assets_amenity.name": {"semantic_type": "FREE_TEXT"},
                      "amenities_catalog.amenity_name": {"semantic_type": "FREE_TEXT"},
                      "amenities_catalog.monthly_fee": {"semantic_type": "MONETARY"}}}
    log = [{"id": "c1", "tool": "join_path", "args": {}, "result": {"routes": [
        {"id": "r1", "path": ["assets_amenity.name=amenities_catalog.amenity_name"], "basis": "cross_source_fk"}]}},
           {"id": "c2", "tool": "describe", "args": {}, "result": {"table": "amenities_catalog",
                                                                   "measures": [{"column": "monthly_fee"}]}}]
    src = {"assets_amenity": "2", "amenities_catalog": "5"}
    plan = P.from_model({"tables": ["assets_amenity", "amenities_catalog"], "joins": ["r1"],
                         "select": ["assets_amenity.name", "amenities_catalog.monthly_fee"]}, log,
                        source_of=src.get)
    assert plan.kind == "federated" and P.validate(plan, log, sm) == []
    sql, ir = F.compile_federated(plan, sm, {"2": "postgres", "5": "parquet"}, src.get)
    assert 'FROM src_2.public."assets_amenity" t0' in sql
    assert 'JOIN src_5."amenities_catalog" t1 ON t1."amenity_name" = t0."name"' in sql
    assert ir.head == "federated.agent"


def test_a_table_joined_twice_is_declined_not_misbound():
    """staff as assignee AND as creator: columns are named by table, so which occurrence a
    column means is not expressible — the compiler declines instead of binding to the first."""
    log = _log_of(TRACES[4])
    plan = P.from_model({"tables": ["tickets", "staff"], "joins": ["r1", "r2"],
                         "group_by": ["staff.full_name"], "aggregates": [{"fn": "count", "col": "*"}]}, log)
    from veda.understanding.frame_compiler import Declined
    out = P.compile_plan(plan, SHOP)
    assert isinstance(out, Declined) and "joined twice" in out.reason


def test_follow_up_route_ids_never_collide_with_prior_ones():
    from veda.agent.tools import ToolBox
    box = ToolBox(SHOP, None, "q", ctx=None)
    box.routes = {"r3": [], "r7": []}                   # a trimmed prior log kept r3 and r7
    r = box._route([{"source_table": "orders", "source_column": "store_id", "target_table": "stores",
                     "target_column": "id"}], "declared", "chain", "orders", "stores")
    assert r["id"] == "r8"


def test_unlicensed_min_becomes_a_ranking_and_unlicensed_sum_is_sent_back():
    log = _log_of(TRACES[0])
    plan = P.from_model({"tables": ["orders"], "aggregates": [{"fn": "min", "col": "orders.total"}]}, log)
    assert PL._license_aggregates(plan, "the cheapest orders") == []
    assert not plan.aggregates and plan.order == [{"expr": "orders.total", "dir": "asc"}]
    plan = P.from_model({"tables": ["orders"], "aggregates": [{"fn": "min", "col": "orders.total"}]}, log)
    assert PL._license_aggregates(plan, "the minimum order total") == [] and plan.aggregates
    plan = P.from_model({"tables": ["orders"], "aggregates": [{"fn": "sum", "col": "orders.total"}]}, log)
    assert PL._license_aggregates(plan, "orders over 500")


def test_rejection_names_the_route_that_reaches_a_missing_table():
    log = _log_of(TRACES[2])
    plan = P.from_model({"tables": ["order_items"], "select": ["customers.city"]}, log)
    errs = P.validate(plan, log, SHOP)
    assert "Add route r1" in PL._join_hint(errs, plan, log)
    # an unlicensed MAX turned into a ranking keeps an order that pointed at it
    log0 = _log_of(TRACES[0])
    plan = P.from_model({"tables": ["orders"], "aggregates": [{"fn": "max", "col": "orders.placed_at"}],
                         "order": [{"by": "agg1", "dir": "desc"}]}, log0)
    PL._license_aggregates(plan, "latest orders")
    assert plan.order == [{"expr": "orders.placed_at", "dir": "desc"}] and P.validate(plan, log0, SHOP) == []


# ══ pass 2: the embedding does the finding and the checking ══════════════════════════
def test_seed_retrieval_identifiers_count_as_tool_provenance():
    log = [{"id": "c1", "tool": "retrieve", "args": {"text": "q"}, "result": {
        "columns": [{"table": "orders", "column": "total", "kind": "MONETARY", "score": 0.9}],
        "tables": [{"table": "orders", "business_name": "order"}]}}]
    plan = P.from_model({"tables": ["orders"], "select": ["orders.total"]}, log)
    assert P.validate(plan, log, SHOP) == []
    s = P.seen_identifiers(log)
    assert ("orders", "total") in s.columns and s.kinds[("orders", "total")] == "MONETARY"


def test_unmapped_noun_phrase_forces_a_find_entities_call():
    from veda.understanding.vocabulary import ScopeVocab
    v = ScopeVocab()
    v.cards = {"tickets": {"business_name": "ticket", "plural": "tickets", "aliases": []},
               "staff": {"business_name": "staff member", "plural": "staff", "aliases": []}}
    tr = TRACES[4]
    box = StubBox(tr)
    box.rec[("find_entities", json.dumps({"phrase": "assigned", "k": 4}, sort_keys=True))] = {
        "phrase": "assigned", "entities": [{"table": "tickets", "business_name": "ticket", "score": 0.9}]}
    # a plan that counts tickets per category and never looks at "assigned"
    fin = {"thought": "done", "action": {"final": {"tables": ["tickets"], "joins": [], "select": [], "filters": [],
                                                   "group_by": ["tickets.category"],
                                                   "aggregates": [{"fn": "count", "col": "*"}], "order": [], "limit": None}}}
    seq = iter([fin, fin, fin])
    import config as _cfg
    old = getattr(_cfg, "AGENT_JUDGE_MODE", "enforce")
    _cfg.AGENT_JUDGE_MODE = "off"
    try:
        r = PL.run_planner(tr["q"], SHOP, v, toolbox=box,
                           slm=lambda *a: (json.dumps(next(seq)), {}), wall_s=60)
    finally:
        _cfg.AGENT_JUDGE_MODE = old
    forced = [c for c in r.tool_calls if c["tool"] == "find_entities" and c["args"].get("phrase") != r.question]
    assert forced and any("assigned" in c["args"]["phrase"] for c in forced)
    assert any(any(str(e).startswith("unmapped") for e in v_["errors"]) for v_ in r.validation)


def test_below_tau_plan_yields_one_revision_then_a_clarify(monkeypatch):
    import veda.agent.judge as J
    from veda.understanding.vocabulary import ScopeVocab
    v = ScopeVocab()
    v.cards = {"orders": {"business_name": "order", "plural": "orders"}}
    calls = []

    def fake_judge(q, plan, vocab, **k):
        calls.append(1)
        return J.Verdict(False, 0.2, None, "orders ordered by total", [], None, "reads as orders by total")
    monkeypatch.setattr(J, "judge", fake_judge)
    import config as _cfg
    monkeypatch.setattr(_cfg, "AGENT_JUDGE_MODE", "enforce", raising=False)
    monkeypatch.setattr(J, "unmapped_phrases", lambda *a, **k: [])
    tr = TRACES[0]
    fin = {"thought": "done", "action": {"final": P.__dict__ and __import__("veda.agent.examples", fromlist=["x"]).full_final(tr["final"])}}
    seq = iter([{"thought": "p", "action": {"tool": "probe", "args": tr["steps"][0]["args"]}}, fin, fin])
    r = PL.run_planner(tr["q"], SHOP, v, toolbox=StubBox(tr), slm=lambda *a: (json.dumps(next(seq)), {}), wall_s=60)
    assert r.kind == "clarify" and r.reason == "judge" and len(calls) == 2 and r.compiled is None


def test_link_card_from_a_two_fk_table():
    from ingestion.link_text import link_tables, fk_phrases
    sm = {"tables": {t: {} for t in ("worklists_ticket", "users_user", "worklists_ticketuser")},
          "columns": {"worklists_ticketuser.id": {"semantic_type": "IDENTIFIER"},
                      "worklists_ticketuser.ticket_id": {"semantic_type": "IDENTIFIER"},
                      "worklists_ticketuser.user_id": {"semantic_type": "IDENTIFIER"},
                      "worklists_ticketuser.assigned_by_id": {"semantic_type": "IDENTIFIER"},
                      "worklists_ticketuser.is_current": {"semantic_type": "FLAG"},
                      "worklists_ticketuser.created_at": {"semantic_type": "TEMPORAL"},
                      "worklists_ticket.id": {"semantic_type": "IDENTIFIER"},
                      "worklists_ticket.title": {"semantic_type": "FREE_TEXT"},
                      "users_user.id": {"semantic_type": "IDENTIFIER"}}}
    E = lambda c, t, rt="business_core": {"source_table": "worklists_ticketuser", "source_column": c,
                                           "target_table": t, "target_column": "id", "discovery": "declared_fk",
                                           "cardinality": "N:1", "relationship_type": rt}
    g = {"edges": [E("ticket_id", "worklists_ticket"), E("user_id", "users_user"),
                   E("assigned_by_id", "users_user", "audit")]}
    cards = {"worklists_ticket": {"business_name": "ticket", "drafted_by": "slm"},
             "users_user": {"business_name": "user", "drafted_by": "slm"}}
    L = link_tables(sm, g, cards)
    lk = L["worklists_ticketuser"]
    assert (lk["a"], lk["b"]) == ("worklists_ticket", "users_user") and lk["verbs"] == ["assigned"]
    assert lk["one_row_is"] == "links a ticket to the user it is assigned to"
    assert {"assigned", "assignee", "ticket assignment"} <= set(lk["aliases"])
    ph = fk_phrases(sm, g, cards, L)
    assert ph["worklists_ticketuser.user_id"] == "the user this ticket is assigned to"
    # an entity with its own name column is not a link table
    sm["columns"]["worklists_ticketuser.name"] = {"semantic_type": "FREE_TEXT"}
    assert "worklists_ticketuser" not in link_tables(sm, g, cards)


_IN_CONTAINER = os.path.exists("/app/veda_core")


@pytest.mark.skipif(not _IN_CONTAINER, reason="needs the engine DB + BGE-M3 (inference container)")
def test_route_ranking_puts_the_meant_relationship_first():
    """12a / 16a / 19a: the route that carries the question's relationship ranks 1."""
    from veda_core.context import RequestContext, set_context
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,), cache_back=False))
    import veda_hybrid as VH
    from veda.understanding.vocabulary import scope_vocab
    sm, _cols = VH._load_semantic_model()
    vocab = scope_vocab(sm)
    cases = [("show recent ticket activities with their ticket categories", "worklists_ticketactivity",
              "worklists_ticket", "worklists_ticketactivity.ticket_id"),
             ("assigned users handling tickets per ticket category", "worklists_ticket", "users_user",
              "worklists_ticketuser"),
             ("which service plans have the most subscribers", "services_serviceplanpricing", "users_user",
              "services_userserviceplan")]
    for q, a, b, must in cases:
        tb = ToolBox(sm, vocab, q)
        r = tb.t_join_path(a, b)
        assert r["routes"], q
        assert any(must in hop for hop in r["routes"][0]["path"]), (q, r["routes"][0])


@pytest.mark.skipif(not _IN_CONTAINER, reason="needs the engine DB + BGE-M3 (inference container)")
def test_judge_rejects_the_six_wrong_v3_plans_and_accepts_13a():
    from veda_core.context import RequestContext, set_context
    set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,), cache_back=False))
    import veda_hybrid as VH
    from veda.understanding.vocabulary import scope_vocab
    from veda.agent.judge import judge
    from ingestion.link_text import for_source
    sm, _cols = VH._load_semantic_model()
    vocab = scope_vocab(sm)
    phrases = for_source("2")[1]
    fx = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "agent_v3_replay_plans.json")))
    got = {}
    for x in fx:
        pd = x["plan"]
        plan = P.Plan(**{k: pd[k] for k in P.Plan.__dataclass_fields__ if k in pd})
        got[x["id"]] = (x["label"], judge(x["question"], plan, vocab, phrases=phrases).ok)
    wrong_passed = [k for k, (lab, ok) in got.items() if lab == "wrong" and ok]
    assert got["13a"][1], got
    # 20c is the RIGHT meaning on the WRONG source (homzhub's amenity categories where the
    # catalog's were asked): a meaning judge cannot see a source choice — recorded, not hidden
    assert wrong_passed == ["20c"], got


# ── §10.7: compile_plan emits no LIMIT unless the plan named one explicitly ────────────
def test_compile_plan_list_no_limit_when_none_asked():
    plan = P.Plan(tables=["orders"], projection=[{"table": "orders", "column": "id"}])
    comp = P.compile_plan(plan, SHOP)
    from veda.understanding.frame_compiler import Declined
    assert not isinstance(comp, Declined), comp
    assert "LIMIT" not in comp.sql
    assert comp.ir.limit is None


def test_compile_plan_list_keeps_explicit_limit():
    plan = P.Plan(tables=["orders"], projection=[{"table": "orders", "column": "id"}], limit=5)
    comp = P.compile_plan(plan, SHOP)
    from veda.understanding.frame_compiler import Declined
    assert not isinstance(comp, Declined), comp
    assert "LIMIT 5" in comp.sql
    assert comp.ir.limit == 5


def test_compile_plan_grouped_no_limit_when_none_asked():
    plan = P.Plan(tables=["orders"], group_by=[{"table": "orders", "column": "status"}])
    comp = P.compile_plan(plan, SHOP)
    from veda.understanding.frame_compiler import Declined
    assert not isinstance(comp, Declined), comp
    assert "LIMIT" not in comp.sql
    assert comp.ir.limit is None


def test_compile_plan_grouped_keeps_explicit_limit():
    plan = P.Plan(tables=["orders"], group_by=[{"table": "orders", "column": "status"}], limit=3)
    comp = P.compile_plan(plan, SHOP)
    from veda.understanding.frame_compiler import Declined
    assert not isinstance(comp, Declined), comp
    assert "LIMIT 3" in comp.sql
    assert comp.ir.limit == 3
