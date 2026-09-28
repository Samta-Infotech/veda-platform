"""veda.agent.examples — the six worked traces in the planner's system block.

Written over an ILLUSTRATIVE shop schema (orders / customers / stores / tickets / staff)
that exists in no source, so the model learns the MOVES — which tool answers which
question, and how the plan is assembled from the results — and cannot copy identifiers
(it may only use identifiers its own tool calls return; Plan.validate enforces that).

Each trace is DATA: the question, the automatic first calls (find_entities on the question,
describe of a confident top entity), every step (thought, action, full tool result) and the
expected final plan. `render()` compresses them into the prompt; the tests replay them
through the real loop with a stubbed SLM and a stubbed tool box, so the prompt can never
show a trace the planner itself would reject.

  1 single table · value filter · order        4 join · grouped by parent · time window
  2 single table · value + numeric filter · avg 5 grouped by two keys (pick the FK route)
  3 two-hop route · projection across tables   6 document question → rag
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

TRACES: List[Dict[str, Any]] = [
    {
        "q": "latest 5 cancelled orders",
        "entities": [{"table": "orders", "business_name": "order", "score": 1.0},
                     {"table": "customers", "business_name": "customer", "score": 0.4}],
        "auto": [{"tool": "describe", "args": {"table": "orders"},
                  "result": {"table": "orders", "business_name": "order", "one_row_is": "one customer order",
                             "business_date": "placed_at",
                             "lifecycle": {"column": "status", "values": ["PLACED", "SHIPPED", "CANCELLED"]},
                             "measures": [{"column": "total", "phrases": ["order value"]}],
                             "dimensions": ["channel"], "display": "order_no",
                             "parents": [{"table": "customers", "via": "customer_id", "name": "customer"}],
                             "children": [], "n_columns": 9}}],
        "steps": [
            {"thought": "cancelled is a status value; latest = business date desc; check counts",
             "tool": "probe", "args": {"tables": ["orders"],
                                       "filters": [{"col": "orders.status", "op": "=", "value": "CANCELLED"}],
                                       "order": "orders.placed_at"},
             "result": {"rows_total": 900, "rows_after_filters": 41,
                        "filters": [{"col": "orders.status", "op": "=", "value": "CANCELLED", "rows": 41}],
                        "distinct_order_col": 870}},
        ],
        "final": {"tables": ["orders"], "select": ["orders.order_no", "orders.placed_at", "orders.total"],
                  "filters": [{"col": "orders.status", "op": "=", "value": "CANCELLED"}],
                  "order": [{"by": "orders.placed_at", "dir": "desc"}], "limit": 5},
    },
    {
        "q": "average order value for web orders over 500",
        "entities": [{"table": "orders", "business_name": "order", "score": 1.0}],
        "auto": [{"tool": "describe", "args": {"table": "orders"},
                  "result": {"table": "orders", "business_name": "order", "business_date": "placed_at",
                             "lifecycle": None, "measures": [{"column": "total", "phrases": ["order value"]}],
                             "dimensions": ["channel"], "display": "order_no", "parents": [], "children": [],
                             "n_columns": 9}}],
        "steps": [
            {"thought": "web is probably a channel value",
             "tool": "values", "args": {"table": "orders", "column": "channel", "phrase": "web"},
             "result": {"table": "orders", "column": "channel", "kind": "CATEGORY",
                        "values": [{"value": "ONLINE", "phrases": ["web", "website"]}, {"value": "STORE"}],
                        "match": {"value": "ONLINE", "method": "glossary"}}},
            {"thought": "check both filters keep rows",
             "tool": "probe", "args": {"tables": ["orders"],
                                       "filters": [{"col": "orders.channel", "op": "=", "value": "ONLINE"},
                                                   {"col": "orders.total", "op": ">", "value": 500}]},
             "result": {"rows_total": 900, "rows_after_filters": 97,
                        "filters": [{"col": "orders.channel", "op": "=", "value": "ONLINE", "rows": 402},
                                    {"col": "orders.total", "op": ">", "value": 500, "rows": 212}]}},
        ],
        "final": {"tables": ["orders"],
                  "filters": [{"col": "orders.channel", "op": "=", "value": "ONLINE"},
                              {"col": "orders.total", "op": ">", "value": 500}],
                  "aggregates": [{"fn": "avg", "col": "orders.total"}]},
    },
    {
        "q": "list order items with the customer's name and city",
        "entities": [{"table": "order_items", "business_name": "order item", "score": 1.0},
                     {"table": "customers", "business_name": "customer", "score": 0.9}],
        "auto": [{"tool": "describe", "args": {"table": "order_items"},
                  "result": {"table": "order_items", "business_name": "order item", "business_date": None,
                             "lifecycle": None, "measures": [{"column": "quantity", "phrases": ["units"]}],
                             "dimensions": [], "display": "sku", "key": "id",
                             "parents": [{"table": "orders", "via": "order_id", "name": "order"}],
                             "children": [], "n_columns": 6}}],
        "steps": [
            {"thought": "customers are two hops away: get the route",
             "tool": "join_path", "args": {"a": "order_items", "b": "customers"},
             "result": {"a": "order_items", "b": "customers", "routes": [
                 {"id": "r1", "path": ["order_items.order_id=orders.id", "orders.customer_id=customers.id"],
                  "why": "chain via orders", "basis": "declared", "via": ["orders"]}]}},
            {"thought": "which customer columns hold name and city",
             "tool": "describe", "args": {"table": "customers"},
             "result": {"table": "customers", "business_name": "customer", "one_row_is": "one customer",
                        "business_date": "joined_on", "lifecycle": None, "measures": [],
                        "dimensions": ["city", "segment"], "display": "full_name", "key": "id", "parents": [],
                        "children": [{"table": "orders", "via": "customer_id", "name": "order"}], "n_columns": 7}},
        ],
        "final": {"tables": ["order_items", "customers"], "joins": ["r1"],
                  "select": ["order_items.sku", "customers.full_name", "customers.city"]},
    },
    {
        "q": "total sales per store in March 2024",
        "entities": [{"table": "stores", "business_name": "store", "score": 1.0},
                     {"table": "orders", "business_name": "order", "score": 0.7}],
        "auto": [{"tool": "describe", "args": {"table": "stores"},
                  "result": {"table": "stores", "business_name": "store", "business_date": None,
                             "lifecycle": None, "measures": [], "dimensions": ["region"], "display": "name",
                             "parents": [], "children": [{"table": "orders", "via": "store_id", "name": "order"}],
                             "n_columns": 5}}],
        "steps": [
            {"thought": "sales are measured on orders, not stores",
             "tool": "describe", "args": {"table": "orders"},
             "result": {"table": "orders", "business_name": "order", "business_date": "placed_at",
                        "lifecycle": None, "measures": [{"column": "total", "phrases": ["sales", "revenue"]}],
                        "dimensions": ["channel"], "display": "order_no",
                        "parents": [{"table": "stores", "via": "store_id", "name": "store"}],
                        "children": [], "n_columns": 9}},
            {"thought": "orders is the grain; join to stores",
             "tool": "join_path", "args": {"a": "orders", "b": "stores"},
             "result": {"a": "orders", "b": "stores", "routes": [
                 {"id": "r1", "path": ["orders.store_id=stores.id"], "why": "chain", "basis": "declared"}]}},
        ],
        "final": {"tables": ["orders", "stores"], "joins": ["r1"], "group_by": ["stores.name"],
                  "aggregates": [{"fn": "sum", "col": "orders.total"}],
                  "order": [{"by": "agg1", "dir": "desc"}],
                  "time": {"col": "orders.placed_at", "from": "2024-03-01", "to": "2024-03-31"}},
    },
    {
        "q": "number of tickets per category and assigned staff member",
        "entities": [{"table": "tickets", "business_name": "ticket", "score": 1.0},
                     {"table": "staff", "business_name": "staff member", "score": 0.8}],
        "auto": [{"tool": "describe", "args": {"table": "tickets"},
                  "result": {"table": "tickets", "business_name": "ticket", "business_date": "opened_at",
                             "lifecycle": {"column": "state", "values": ["OPEN", "CLOSED"]}, "measures": [],
                             "dimensions": ["category", "priority"], "display": "title",
                             "parents": [{"table": "staff", "via": "assigned_to_id", "name": "staff member"},
                                         {"table": "staff", "via": "created_by_id", "name": "staff member"}],
                             "children": [], "n_columns": 11}}],
        "steps": [
            {"thought": "two group keys: category on tickets, the assigned staff member",
             "tool": "join_path", "args": {"a": "tickets", "b": "staff"},
             "result": {"a": "tickets", "b": "staff", "routes": [
                 {"id": "r1", "path": ["tickets.assigned_to_id=staff.id"], "why": "direct via assigned_to_id",
                  "basis": "declared"},
                 {"id": "r2", "path": ["tickets.created_by_id=staff.id"], "why": "direct via created_by_id",
                  "basis": "declared"}]}},
            {"thought": "assigned means r1; need the staff name column",
             "tool": "describe", "args": {"table": "staff"},
             "result": {"table": "staff", "business_name": "staff member", "business_date": None,
                        "lifecycle": None, "measures": [], "dimensions": ["team"], "display": "full_name",
                        "parents": [], "children": [{"table": "tickets", "via": "assigned_to_id", "name": "ticket"}],
                        "n_columns": 6}},
        ],
        "final": {"tables": ["tickets", "staff"], "joins": ["r1"],
                  "group_by": ["tickets.category", "staff.full_name"],
                  "aggregates": [{"fn": "count", "col": "*"}], "order": [{"by": "agg1", "dir": "desc"}]},
    },
    {
        "q": "how many days of annual leave do new employees get",
        "entities": [{"table": "staff", "business_name": "staff member", "score": 0.3}],
        "auto": [],
        "steps": [
            {"thought": "leave rules are policy text, not rows; check the documents",
             "tool": "doc_sections", "args": {"query": "annual leave for new employees"},
             "result": {"sections": [{"title": "Employee Handbook", "section": "Leave policy", "page": 14,
                                      "snippet": "New employees accrue 1.5 days of annual leave per month"}]}},
        ],
        "final": {"kind": "rag"},
    },
]


_REQUIRED = ("tables", "joins", "select", "filters", "group_by", "aggregates", "order", "limit")


def full_final(fin: Dict[str, Any]) -> Dict[str, Any]:
    """A trace's final plan with every required key, in schema order (a rag final is
    just {"kind": "rag"})."""
    if fin.get("kind") == "rag":
        return {"kind": "rag"}
    out = {k: fin.get(k, None if k == "limit" else []) for k in _REQUIRED}
    for k in ("distinct", "time"):
        if k in fin:
            out[k] = fin[k]
    return out


_EX_DROP = {"business_name", "reason", "hops", "a", "b", "name", "method", "cardinality", "kind", "basis"}


def _slim(o):
    if isinstance(o, dict):
        return {k: _slim(v) for k, v in o.items() if k not in _EX_DROP}
    if isinstance(o, list):
        return [_slim(x) for x in o]
    return o


def _j(o) -> str:
    from veda.agent.tools import compact
    return json.dumps(_slim(compact(o)), separators=(",", ":"), ensure_ascii=False)


def _call(tool: str, args: Dict[str, Any]) -> str:
    return f"{tool}({json.dumps(args, separators=(',', ':'), ensure_ascii=False)[1:-1]})"


def render() -> str:
    """The traces as the system block shows them: auto calls, then 'you:' actions (the
    thought + the call), each result, and the final plan as the exact JSON to emit."""
    out = []
    for i, tr in enumerate(TRACES, 1):
        ents = ", ".join(f"{e['table']} {e['score']}" for e in tr["entities"])
        lines = [f"Q{i}: {tr['q']}", f"[c1] find_entities → {ents}"]
        n = 2
        for pre in tr.get("auto") or []:
            lines.append(f"[c{n}] {_call(pre['tool'], pre['args'])} → {_j(pre['result'])}")
            n += 1
        for st in tr["steps"]:
            lines.append(f"you: ({st['thought']}) {_call(st['tool'], st['args'])}")
            lines.append(f"[c{n}] → {_j(st['result'])}")
            n += 1
        lines.append("you: final " + json.dumps(full_final(tr["final"]), separators=(",", ":"), ensure_ascii=False))
        out.append("\n".join(lines))
    return "\n".join(out)
