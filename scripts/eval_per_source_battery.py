"""Per-source behavioural battery — the M0 gate for "every registered source works the
same way" (docs/backlog/ARCH_REVIEW_2026-09_RECONCILED.md; SESSION_HANDOFF_2026-09.md §11).

Runs SAME-shaped questions pinned to each ready source and asserts, per question, an
EXPECTED SHAPE — not just "answered":

  route   "sql" | "rag"        — which head must answer (a document source answers on RAG)
  agg     True                 — the executed SQL must contain an aggregate function
  group   True                 — the executed SQL must contain GROUP BY
  filter  True                 — the executed SQL must contain WHERE
  order   True                 — the executed SQL must contain ORDER BY / LIMIT (ranking)

plus the two invariants that distinguish "isolated" from "contaminated":
  • every table named in the SQL a source executes is one of THAT source's own tables
    (engine store `column_embeddings_v2`, always source-scoped);
  • a grouping phrase in the question with no GROUP BY in an answered SQL is a silently
    dropped breakdown (the "how many X per Y → scalar COUNT(*)" class the verified-query
    cache replayed on 2026-09-15 before its shape guard).

`known_gap` marks a question the engine is NOT expected to answer yet (documented M2/M3
territory — e.g. value filters on lite-model sources). A known-gap question that returns
a typed clarify/refuse is reported as WARN, not counted as a failure; one that returns a
WRONG-SHAPED answer still fails. The gate is honest about what it doesn't cover.

`expect="refuse"` inverts the expectation for a question whose ONLY honest outcome is a
typed refusal/clarify — e.g. a filter value that does not exist in the data ("tickets with
high priority" on a source whose priorities are LOW/MEDIUM only). An *answer* there is a
silently unfiltered row list and FAILS (found live 2026-09-16: the shared planner answered
it with the qualifier dropped before `_tier2_validate` was applied to that branch).

`xfail="..."` carries a documented, currently-failing case without blocking the gate: a
FAIL becomes XFAIL (reported, counted separately, exit 0); an unexpected pass is reported
as XPASS so the marker gets removed when the underlying fix lands. Used for the verified-
query cache similarity replay ("which vendor has the highest rating" replays the cached
"top 3 vendors by rating" at cosine 0.86 → LIMIT 3) — per docs/backlog/query-engine-open-
items.md the fix is an IR-shape-aware cache key (M2/M6), NOT another demotion heuristic.

Usage (inside the inference container; real engine, real stack):
    cd /app/veda_core && python /app/scripts/eval_per_source_battery.py [--sources 2,3,4,5]

Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")


def Q(q, route="sql", agg=False, group=False, filter=False, order=False, known_gap=None,
      expect="answer", xfail=None):
    return {"q": q, "route": route, "agg": agg, "group": group, "filter": filter,
            "order": order, "known_gap": known_gap, "expect": expect, "xfail": xfail}


# One battery per source, phrased with that source's OWN nouns/columns, ≥15 questions each
# across shapes: count / filtered / grouped / ranked / temporal / existence / list.
BATTERY = {
    2: [  # homzhub — relational Postgres, full semantic model
        Q("how many properties are there", agg=True),
        Q("how many users are there", agg=True),
        Q("how many payment transactions are there", agg=True),
        Q("total paid amount across all payment transactions", agg=True,
          known_gap="ungrouped SUM over a whole table has no deterministic branch (M2)"),
        Q("list all users"),
        Q("show all vendors"),
        Q("list properties in Mumbai", filter=True,
          known_gap="value grounding picks a state column for a city literal (M2 grounding)"),
        Q("how many properties are in Pune", agg=True, filter=True,
          known_gap="filtered count: no deterministic branch consumes the filter (M2)"),
        Q("how many properties per city", agg=True, group=True,
          known_gap="grouped COUNT per dimension is not on the fast path for this table (M2)"),
        Q("average carpet area per project", agg=True, group=True,
          known_gap="ambiguous measure (carpet_area vs carpet_area_sqft) → grounded clarify by design"),
        Q("total paid amount per currency", agg=True, group=True,
          known_gap="grouped SUM over an IDENTIFIER dimension (M2 grounding)"),
        Q("list top 5 properties by monthly rent", order=True,
          known_gap="rank a grain (properties) by a measure on a joined child "
                    "(assets_leaselisting.expected_monthly_rent): the arbiter mis-anchors on a "
                    "lookup table and Tier-2's join answer drops the grain → typed refusal (M2 grounding)"),
        Q("which project has the highest carpet area", agg=True,
          known_gap="superlative planner is trace/route-only today (SUPERLATIVE_JOIN_ROUTING off)"),
        Q("users created last month", filter=True),
        Q("latest 10 payment transactions", order=True),
        Q("properties with more than 3 floors", filter=True,
          known_gap="numeric comparison filter needs grounded predicate (M2)"),
        Q("how many users have a last login", agg=True, filter=True,
          known_gap="existence_count semantics on a nullable column (M2)"),
        # worklists_ticket.priority holds only LOW (223) / MEDIUM (8) in this copy — there is
        # no HIGH, so the only honest outcome is a typed refusal; an answer = unfiltered list.
        Q("show tickets with high priority", expect="refuse"),
    ],
    3: [  # docs_contracts — document/filesystem source, RAG head only
        Q("is there any penalty for paying invoices late", route="rag"),
        Q("what does the maintenance policy say", route="rag"),
        Q("how many properties are there", route="rag"),
        Q("what is the late fee percentage", route="rag"),
        Q("which assets does the maintenance policy govern", route="rag"),
        Q("what does the master services agreement cover", route="rag"),
        Q("summarize the employee handbook leave policy", route="rag"),
        Q("who is responsible for basketball court upkeep", route="rag"),
        Q("what are the payment terms in the MSA", route="rag"),
        Q("is professional course reimbursement available to employees", route="rag"),
        Q("what does the site notes document say", route="rag"),
        Q("are society charges covered by the late fee clause", route="rag"),
        Q("what is the notice period for termination", route="rag"),
        Q("list the documents available", route="rag"),
        Q("total rent value across all leases", route="rag"),
    ],
    4: [  # invoices_csv — tabular (csv), lite model: maintenance(ticket_id, asset_id, category, amount, status), vendors(vendor_id, ticket_id, city, rating)
        Q("how many maintenance records are there", agg=True),
        Q("how many vendors are there", agg=True),
        Q("list vendors"),
        Q("list all maintenance records"),
        Q("how many maintenance records per vendor", agg=True, group=True),
        Q("how many maintenance records per category", agg=True, group=True),
        Q("how many maintenance records per status", agg=True, group=True),
        Q("total maintenance amount per category", agg=True, group=True),
        Q("average vendor rating per city", agg=True, group=True),
        Q("total maintenance amount", agg=True,
          known_gap="ungrouped SUM over a whole table has no deterministic branch (M2)"),
        Q("vendors in Kochi", filter=True,
          known_gap="value filter on a lite-model source (M2 grounding)"),
        Q("maintenance records with status open", filter=True,
          known_gap="value filter on a lite-model source (M2 grounding)"),
        Q("top 3 vendors by rating", order=True),
        # Runs right after "top 3 vendors by rating" ON PURPOSE: the verified-query cache
        # matches by cosine (≥0.85) on question text and replays that row's LIMIT 3 here
        # (sim 0.86, live 2026-09-16). This is the scalar-vs-ranked replay case the cache
        # key must stop colliding on once it is IR-shape-aware (M2/M6) — kept as xfail, no
        # further demotion heuristic. Expected shape once fixed: ORDER BY rating DESC
        # LIMIT 1 or a MAX — hence agg=True (a LIMIT-1 answer is also accepted below).
        Q("which vendor has the highest rating", agg=True,
          xfail="verified-cache similarity replay of 'top 3 vendors by rating' (LIMIT 3)"),
        Q("maintenance records with amount above 500", filter=True,
          known_gap="numeric comparison filter needs grounded predicate (M2)"),
    ],
    5: [  # catalog_parquet — tabular (parquet), lite model: amenities_catalog(amenity_id, amenity_name, category, monthly_fee)
        Q("how many amenities are there", agg=True),
        Q("list all amenities"),
        Q("average monthly fee per category", agg=True, group=True),
        Q("total monthly fee per category", agg=True, group=True),
        Q("how many amenities per category", agg=True, group=True),
        Q("highest monthly fee per category", agg=True, group=True),
        Q("lowest monthly fee per category", agg=True, group=True),
        Q("top 3 amenities by monthly fee", order=True),
        Q("amenities in the Sports category", filter=True,
          known_gap="value filter on a lite-model source (M2 grounding)"),
        Q("amenities with a monthly fee above 100", filter=True,
          known_gap="numeric comparison filter needs grounded predicate (M2)"),
        Q("which amenity has the highest monthly fee", agg=True,
          ),   # a superlative on a row answers as ORDER BY … LIMIT 1 (or a grouped MAX) — both fine
        Q("total monthly fee", agg=True,
          known_gap="ungrouped SUM over a whole table has no deterministic branch (M2)"),
        Q("how many amenity categories are there", agg=True,
          known_gap="COUNT DISTINCT of a dimension has no deterministic branch (M2)"),
        Q("list amenity names"),
        Q("average monthly fee", agg=True,
          known_gap="ungrouped AVG over a whole table has no deterministic branch (M2)"),
    ],
}

_TABLE_RE = re.compile(r'(?:FROM|JOIN)\s+(?:[\w.]+\.)?"?([A-Za-z_][A-Za-z0-9_]*)"?', re.I)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
_GROUPING_Q_RE = re.compile(r"\b(per|by each|for each|grouped by|broken down by)\b", re.I)


def _own_tables(source_id: int) -> set[str]:
    from config import BIENCODER_COL_TABLE
    from ingestion.db_abstraction import get_internal_connection, release_internal_connection
    conn = get_internal_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT table_name FROM {BIENCODER_COL_TABLE} WHERE source_id=%s",
                        [str(source_id)])
            return {r[0] for r in cur.fetchall()}
    finally:
        release_internal_connection(conn)


def _sql_tables(sql: str) -> set[str]:
    return {m.group(1) for m in _TABLE_RE.finditer(sql or "")}


def _check_shape(spec, sql: str) -> list[str]:
    """Shape assertions on an answered SQL — a list of violated expectations."""
    bad = []
    if spec["agg"] and not _AGG_RE.search(sql):
        # a bare superlative ("which X has the highest Y") is also correctly answered as
        # ORDER BY … LIMIT 1 — a row, not an aggregate; any other LIMIT is the wrong N.
        _superl = re.search(r"\b(highest|lowest|largest|smallest|most|least|max|min)\b", spec["q"].lower())
        _top1 = re.search(r"\bORDER BY\b.*\bLIMIT\s+1\b", sql, re.I | re.S)
        if not (_superl and _top1):
            bad.append("expected an aggregate function (or ORDER BY … LIMIT 1 for a superlative)")
    if spec["group"] and not re.search(r"\bGROUP BY\b", sql, re.I):
        bad.append("expected GROUP BY")
    if spec["filter"] and not re.search(r"\bWHERE\b", sql, re.I):
        bad.append("expected WHERE")
    if spec["order"] and not re.search(r"\b(ORDER BY|LIMIT)\b", sql, re.I):
        bad.append("expected ORDER BY / LIMIT")
    # silently dropped breakdown, independent of the spec (the cache-replay class)
    if _GROUPING_Q_RE.search(spec["q"]) and not re.search(r"\bGROUP BY\b", sql, re.I):
        bad.append("grouping asked, no GROUP BY in executed SQL")
    # grouping DIMENSION: "per vendor" must group by a vendor column (found live: the
    # per-vendor join grouped by `category` — right join, wrong dimension, and the
    # GROUP-BY-present check alone called it OK)
    mdim = re.search(r"\b(?:per|by each|for each|grouped by|broken down by)\s+([a-z]+)", spec["q"].lower())
    mgb = re.search(r"\bGROUP BY\s+(.+?)(?:\s+ORDER BY|\s+LIMIT|$)", sql, re.I | re.S)
    if mdim and mgb:
        stem = mdim.group(1).rstrip("s")[:5]
        if stem not in mgb.group(1).lower():
            bad.append(f"asked per '{mdim.group(1)}', SQL groups by {mgb.group(1).strip()[:60]!r}")
    # "top N / latest N / last N" must compile to LIMIT N exactly (found live: "top 3 vendors
    # by rating" answered with ORDER BY rating DESC LIMIT 1 — a silently wrong N).
    mtop = re.search(r"\b(?:top|latest|last|bottom|first)\s+(\d+)\b", spec["q"].lower())
    if mtop:
        mlim = re.search(r"\bLIMIT\s+(\d+)\b", sql, re.I)
        if not mlim or mlim.group(1) != mtop.group(1):
            bad.append(f"asked for top {mtop.group(1)}, SQL has LIMIT {mlim.group(1) if mlim else 'none'}")
    # comparator shape: "more than / above / over N" must compile to > or >=, "less than /
    # below / under N" to < or <= — an equality here is a silently WRONG filter (found live:
    # "properties with more than 3 floors" → WHERE "total_floors" = %s, and the WHERE-present
    # check alone called it OK).
    q = spec["q"].lower()
    if re.search(r"\b(more than|above|over|greater than|at least|higher than)\b", q) and \
            not re.search(r"(?<![<!])>=?", sql):
        bad.append("'more than'-style question compiled without a > comparator")
    if re.search(r"\b(less than|below|under|fewer than|at most|lower than)\b", q) and \
            not re.search(r"(?<!-)<=?(?!>)", sql):
        bad.append("'less than'-style question compiled without a < comparator")
    return bad


def run(source_ids):
    from veda_core.context import RequestContext, set_context
    from veda_hybrid import run_hybrid_query

    failures = warns = xfails = xpass = 0
    for sid in source_ids:
        own = _own_tables(sid)
        for spec in BATTERY.get(sid, []):
            q = spec["q"]
            set_context(RequestContext(source_id=sid, tenant="default"))
            t0 = time.time()
            row = {"source": sid, "q": q}
            try:
                r = run_hybrid_query(q, verbose=False)
                it = r.items[0] if r.items else None
                res = it.result if it is not None else None
                route = getattr(it, "route", None)
                row["route"] = route
                if isinstance(res, dict):
                    status = res.get("status")
                    sql = res.get("sql") or ""
                    row["status"], row["sql"] = status, sql[:160]
                    foreign = _sql_tables(sql) - own if sql else set()
                    if foreign:
                        row["FAIL"] = f"SQL names tables this source does not own: {sorted(foreign)}"
                    elif spec["route"] == "rag":
                        row["FAIL"] = "expected the RAG head, got the SQL head"
                    elif status == "answered":
                        if spec["expect"] == "refuse":
                            row["FAIL"] = ("expected a typed refusal (the asked-for value does "
                                           "not exist in this data), got an answer")
                        else:
                            bad = _check_shape(spec, sql)
                            if bad:
                                row["FAIL"] = "; ".join(bad)
                    elif status in ("clarify", "refuse", "not_materialized", "qualifier_dropped",
                                    "no_table", "exec_error", "ungrounded", "ir_mismatch",
                                    "tier2_rejected", "access_denied"):
                        if spec["expect"] == "refuse":
                            row["note"] = f"typed {status} as expected (refuse-over-guess)"
                        elif spec["known_gap"]:
                            row["WARN"] = f"typed {status} — known gap: {spec['known_gap']}"
                        else:
                            row["FAIL"] = f"expected an answer, got typed {status}"
                    else:
                        row["FAIL"] = f"unexpected status {status!r}"
                else:  # RAG / NoSQL result object
                    answered = bool(getattr(res, "answer", None)) and not getattr(res, "error", None)
                    row["status"] = "answered" if answered else "no-answer"
                    row["citations"] = len(getattr(res, "citations", []) or [])
                    if spec["route"] != "rag":
                        row["FAIL"] = "expected the SQL head, got a non-SQL result"
                    elif not answered:
                        row["FAIL"] = "RAG returned no answer"
            except Exception as e:
                row.update({"status": "CRASH", "FAIL": f"{type(e).__name__}: {str(e)[:120]}"})
            row["secs"] = round(time.time() - t0, 1)
            if spec["xfail"]:
                if "FAIL" in row:
                    row["XFAIL"] = f"{row.pop('FAIL')} — expected failure: {spec['xfail']}"
                    xfails += 1
                else:
                    row["XPASS"] = f"unexpected pass — remove the xfail marker: {spec['xfail']}"
                    xpass += 1
            if "FAIL" in row:
                failures += 1
            elif "WARN" in row:
                warns += 1
            print(json.dumps(row))
    return failures, warns, xfails, xpass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="2,3,4,5")
    args = ap.parse_args()
    sids = [int(s) for s in args.sources.split(",") if s.strip()]
    os.chdir("/app/veda_core")
    total = sum(len(BATTERY.get(s, [])) for s in sids)
    failures, warns, xfails, xpass = run(sids)
    print(json.dumps({"summary": "FAIL" if failures else "OK", "questions": total,
                      "failures": failures, "known_gap_warns": warns,
                      "xfail": xfails, "xpass": xpass}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
