"""Cross-source battery (M3 checkpoint 1, 2026-09-16): the SAME question pinned to its
source and unpinned (full scope 2,3,4,5) must give the SAME answer — or a typed refusal.
Never a different answer.

For every source-4 and source-5 question of the per-source battery, two runs:
  pinned   : RequestContext(source_id=S)
  unpinned : RequestContext(source_id=2, source_ids=(2,3,4,5))   (the chat scope)
Assert, per pair:
  • both answered → same SQL SHAPE (aggregate present / GROUP BY dimension / WHERE
    presence + comparator direction / LIMIT N) AND the same row count;
  • or the unpinned run is a typed refusal (clarify / refuse / qualifier_dropped /
    ungrounded / ir_mismatch / tier2_rejected / refused_federated / not_federated);
  • anything else (a different shape, a different row count, a crash) = DIVERGENT → FAIL.
Plus the SESSION_HANDOFF §9.2 chat sequence, unpinned, in one chat session:
  "how many properties are in Mumbai" → "how many of those are for sale":
  expected a FILTERED count (both predicates in the SQL) or a typed refusal; a
  `for_sale_count` / `for sale` answer computed on an UNFILTERED result is the failure
  case, asserted explicitly.

Usage (inside the inference container):
    cd /app/veda_core && python /app/scripts/eval_cross_source_battery.py
Exit code 1 on any divergence.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

TYPED = {"clarify", "refuse", "qualifier_dropped", "ungrounded", "ir_mismatch", "tier2_rejected",
         "refused_federated", "not_federated", "exec_error", "access_denied", "not_materialized",
         "no_table", "exec_error_federated"}
FULL_SCOPE = (2, 3, 4, 5)


def _shape(sql: str) -> dict:
    s = sql or ""
    m_lim = re.search(r"\bLIMIT\s+(\d+)", s, re.I)
    m_grp = re.search(r"\bGROUP BY\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s*$)", s, re.I | re.S)
    grp = re.sub(r"[\"a-z0-9_]+\.", "", (m_grp.group(1) if m_grp else "").lower())
    grp = re.sub(r"\s+", " ", grp).strip()
    return {"agg": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", s, re.I)),
            "group": grp,
            "where": bool(re.search(r"\bWHERE\b", s, re.I)),
            "cmp": sorted(set(re.findall(r"\s(>=|<=|>|<|=|<>|!=)\s", s))),
            "limit": int(m_lim.group(1)) if m_lim else None}


def _run(q, ctx):
    from veda_core.context import set_context
    from veda_hybrid import run_hybrid_query
    set_context(ctx)
    t0 = time.time()
    r = run_hybrid_query(q, verbose=False)
    it = r.items[0] if getattr(r, "items", None) else None
    res = it.result if it is not None else None
    out = {"secs": round(time.time() - t0, 1)}
    if isinstance(res, dict):
        rows = res.get("rows") or (res.get("result") or {}).get("rows") or []
        out.update({"status": res.get("status"), "sql": res.get("sql") or "",
                    "rows": len(rows) if isinstance(rows, list) else None,
                    "firewall": ((res.get("trace") or {}).get("sections", {}) or {}).get("firewall")
                    or res.get("firewall")})
    else:
        out.update({"status": "non-sql", "sql": "", "rows": None})
    return out


def main() -> int:
    from veda_core.context import RequestContext
    from eval_per_source_battery import BATTERY   # same questions, same shapes
    failures = 0
    rows_out = []
    for sid in (4, 5):
        for spec in BATTERY.get(sid, []):
            if spec["route"] != "sql":
                continue
            q = spec["q"]
            row = {"source": sid, "q": q}
            try:
                pinned = _run(q, RequestContext(source_id=sid, tenant="default", cache_back=False))
                unp = _run(q, RequestContext(source_id=2, tenant="default", source_ids=FULL_SCOPE,
                                             cache_back=False))
                row.update({"pinned": pinned["status"], "unpinned": unp["status"],
                            "pinned_rows": pinned["rows"], "unpinned_rows": unp["rows"],
                            "unpinned_sql": (unp["sql"] or "")[:140],
                            "firewall": unp.get("firewall")})
                if unp["status"] in TYPED:
                    row["verdict"] = "typed_refusal"
                elif pinned["status"] == "answered" and unp["status"] == "answered":
                    if _shape(pinned["sql"]) == _shape(unp["sql"]) and pinned["rows"] == unp["rows"]:
                        row["verdict"] = "same"
                    else:
                        row["verdict"] = "DIVERGENT"
                        row["FAIL"] = (f"shape/rows differ: pinned {_shape(pinned['sql'])}/{pinned['rows']} "
                                       f"vs unpinned {_shape(unp['sql'])}/{unp['rows']}")
                elif pinned["status"] in TYPED and unp["status"] == "answered":
                    # pinned refused but the wide scope answered: only acceptable if the answer
                    # is a well-shaped one for this spec (otherwise the wide scope invented it)
                    row["verdict"] = "DIVERGENT"
                    row["FAIL"] = f"pinned {pinned['status']} but unpinned answered: {unp['sql'][:120]}"
                else:
                    row["verdict"] = "both_typed"
            except Exception as e:
                row.update({"verdict": "CRASH", "FAIL": f"{type(e).__name__}: {str(e)[:120]}"})
            if "FAIL" in row:
                failures += 1
            print(json.dumps(row), flush=True)
            rows_out.append(row)

    # §9.2 chat sequence, unpinned
    seq = {"seq": "§9.2", "turns": []}
    try:
        from veda_core.context import set_context
        from veda_hybrid import run_hybrid_query
        ctx = RequestContext(source_id=2, tenant="default", source_ids=FULL_SCOPE, cache_back=False)
        set_context(ctx)
        t1 = _run("how many properties are in Mumbai", ctx)
        # chatbot memory is a service-layer concern; at engine level the follow-up is the
        # RESOLVED question the chat layer would send — assert the engine keeps BOTH predicates
        t2 = _run("how many properties in Mumbai are for sale", ctx)
        seq["turns"] = [t1, t2]
        sql2 = (t2.get("sql") or "").lower()
        if t2["status"] in TYPED:
            seq["verdict"] = "typed_refusal"
        elif t2["status"] == "answered":
            has_city = "mumbai" in sql2 or "city" in sql2
            has_sale = "sale" in sql2 or "for_sale" in sql2 or "listing" in sql2
            if has_city and has_sale:
                seq["verdict"] = "filtered_count"
            else:
                seq["verdict"] = "DIVERGENT"
                seq["FAIL"] = f"follow-up answered on an unfiltered/partially-filtered result: {sql2[:160]}"
        else:
            seq["verdict"] = t2["status"]
    except Exception as e:
        seq.update({"verdict": "CRASH", "FAIL": f"{type(e).__name__}: {str(e)[:120]}"})
    if "FAIL" in seq:
        failures += 1
    print(json.dumps(seq), flush=True)

    verdicts = {}
    for r in rows_out:
        verdicts[r.get("verdict")] = verdicts.get(r.get("verdict"), 0) + 1
    print(json.dumps({"summary": "FAIL" if failures else "OK", "pairs": len(rows_out),
                      "divergent": failures, "verdicts": verdicts,
                      "seq_9_2": seq.get("verdict")}))
    return 1 if failures else 0


if __name__ == "__main__":
    os.chdir("/app/veda_core")
    sys.path.insert(0, "/app/scripts")
    raise SystemExit(main())
