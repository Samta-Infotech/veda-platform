#!/usr/bin/env python3
"""
nl_query_suite.py — fire a fixed battery of natural-language questions at the VEDA
query API (POST /api/v1/query) and record, per query, whether the pipeline returned a
usable answer, plus route / status / row-count / latency, then write a markdown report
and a raw JSONL.

WHAT "CORRECT vs WRONG" MEANS HERE
----------------------------------
There are no golden answers for these ad-hoc questions, so the suite measures
ANSWERABILITY, not verified semantic correctness:

  PASS (answered)  — pipeline returned status ok/answered with non-empty rows OR a
                     grounded NL answer.
  REFUSED          — pipeline explicitly declined (refuse / clarify / no_table /
                     ungrounded / federated_refused). A *safe* outcome, not a crash.
  EMPTY            — ok but zero rows and no answer (ran, found nothing).
  ERROR            — exec error / HTTP error / exception.
  TIMEOUT          — no response within --timeout seconds.

In addition, cheap STRUCTURAL checks are applied where the wording implies one
(e.g. "top 5" -> expect <= 5 rows). These are advisory flags appended to the note
column; they do NOT by themselves flip PASS/FAIL.

Usage:
    python3 evaluation/nl_query_suite.py \
        --url http://localhost:8080/api/v1/query --source-id 2 --timeout 120
Outputs (next to this file):
    nl_query_suite_results.jsonl   (one raw record per query, written incrementally)
    nl_query_suite_report.md       (human-readable report, rewritten after each query)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- queries
# id, text.  Grouped loosely by complexity so the report reads sensibly.
QUERIES: list[tuple[str, str]] = [
    ("q01", 'What is the total outstanding amount for all maintenance items that are not yet completed or settled?'),
    ("q02", 'Which category contributes the highest value among all completed payments?'),
    ("q03", 'If all open repair requests are closed today, what would be the total repair expenditure recorded?'),
    ("q04", 'Which asset currently has multiple maintenance activities associated with it, and what are they?'),
    ("q05", 'What percentage of all maintenance records are related to repair work?'),
    ("q06", 'What is the average amount of all repair-related requests?'),
    ("q07", 'How much money has already been collected through completed payments?'),
    ("q08", 'If unpaid and open records are combined into a single pending bucket, how many pending items exist and what is their total value?'),
    ("q09", 'Among all assets with pending work or dues, which asset has the highest financial impact?'),
    ("q10", 'If all paid entries are excluded, what categories remain and how much does each contribute?'),
    ("q11", 'What is the ratio of completed payment value to pending value?'),
    ("q12", 'Which category appears most frequently in the records and what percentage of the dataset does it represent?'),
    ("q13", 'If a 10% surcharge is applied to all unresolved items, what would be the revised outstanding amount?'),
    ("q14", 'What is the difference between the highest and lowest transaction amounts recorded?'),
    ("q15", 'If all rent-related transactions are grouped together, what is the total rent value and how many entries contribute to it?'),
    ("q16", 'Which assets are currently covered under the maintenance arrangement?'),
    ("q17", 'What types of facilities are included in the maintenance coverage?'),
    ("q18", 'In which location are these facilities situated?'),
    ("q19", 'What is the monthly rent amount associated with these assets?'),
    ("q20", 'How much is charged for repair services when they are requested?'),
    ("q21", 'What is the annual insurance cost for the covered assets?'),
    ("q22", 'How frequently are society charges collected, and what is the amount?'),
    ("q23", 'What is the annual tax amount payable for these assets?'),
    ("q24", 'What is the recurring monthly loan payment amount?'),
    ("q25", 'Which charges need to be paid every month?'),
    ("q26", 'How much notice is required if either party wants to end the agreement?'),
    ("q27", 'Is there any penalty for paying invoices late?'),
    ("q28", 'Which types of invoices are subject to late payment charges?'),
    ("q29", 'What is the monthly rate charged on overdue invoices?'),
    ("q30", 'Which assets are included under this service arrangement?'),
    ("q31", 'Which locations are associated with the covered assets?'),
    ("q32", 'What amenities are included within the scope of services?'),
    ("q33", 'Is intercom maintenance included in the covered services?'),
    ("q34", 'Which financial categories are managed under this agreement?'),
    ("q35", 'Does the agreement include repair-related expenses?'),
    ("q36", "Which city has the highest-rated vendor, and what is the vendor's rating?"),
    ("q37", 'What is the average vendor rating across all service locations?'),
    ("q38", 'Which locations have vendors performing above the overall average rating?'),
    ("q39", 'If vendors with ratings below 4.0 must undergo a quality review, which vendors would be flagged?'),
    ("q40", 'What percentage of vendors have ratings of at least 4.0?'),
    ("q41", 'Which location has the greatest positive deviation from the average vendor rating?'),
    ("q42", 'If only vendors rated 4.2 or higher qualify for premium contracts, which locations remain eligible?'),
    ("q43", 'What is the rating spread between the best-performing and lowest-performing vendors?'),
    ("q44", 'Which city appears multiple times in the vendor network, and how do its vendors compare internally?'),
    ("q45", 'If vendors scoring below 4.0 are removed from the network, what would be the new average vendor rating?'),
    ("q46", 'How many vendors fall within one standard service band of 4.0 to 4.5 inclusive?'),
    ("q47", 'If the organization wants to expand operations in cities already served by high-performing vendors (rating above 4.3), which cities should be prioritized?'),
    ("q48", 'What proportion of the vendor network is concentrated in locations with ratings above the network average?'),
    ("q49", 'If a service quality score is calculated as (rating × 20), which location achieves the highest score?'),
    ("q50", 'Assuming each vendor can support one active maintenance ticket at a time, what is the maximum number of simultaneous tickets the current vendor network can handle?'),
]

# --------------------------------------------------------------------------- goldens
# GOLDEN_ANCHORS: qid → set of acceptable tables. Contract: when a query is ANSWERED
# (PASS or EMPTY), its SQL must reference AT LEAST ONE of these tables. A wrong-table
# answer is the dangerous failure mode — plausible-looking data from the wrong entity —
# and the plain answerability verdict cannot see it. Refusals/clarifies pass vacuously
# (refuse-over-guess is judged separately). Only assert where the correct table set is
# unambiguous from the question + schema; grow over time. Sets are deliberately
# generous (e.g. payments may legitimately resolve to the payment-transaction table or
# the general ledger) so a golden failure is a REAL failure.
GOLDEN_ANCHORS: dict[str, set[str]] = {
    # 2026-07-12 question set v2 — assert only where the correct table set is
    # unambiguous from the question; grows as answers are reviewed.
    "q02": {"accounts_paymenttransaction"},
    "q07": {"accounts_paymenttransaction", "accounts_generalledger"},
    "q11": {"accounts_paymenttransaction", "accounts_generalledger"},
    "q14": {"accounts_paymenttransaction", "accounts_generalledger"},
}

_TABLE_REF_RE = re.compile(r'\b(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', re.I)


def sql_tables(sql: str) -> set[str]:
    """Table names referenced by FROM/JOIN in generated SQL (quoted or bare)."""
    return {m.group(1).lower() for m in _TABLE_REF_RE.finditer(sql or "")}

# --------------------------------------------------------------------------- helpers
_LIMIT_RE = re.compile(r"\b(?:top|first|last)\s+(\d+)\b", re.I)


def expected_row_cap(text: str):
    """If the query names 'top/first/last N', return N (advisory structural check)."""
    m = _LIMIT_RE.search(text)
    return int(m.group(1)) if m else None


def post_query(url: str, query: str, source_id: int, tenant: str, timeout: float) -> dict:
    body = json.dumps({"query": query, "source_id": source_id, "tenant": tenant}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            dt = round((time.time() - t0) * 1000)
            try:
                return {"ok_http": True, "http": resp.status, "latency_ms": dt,
                        "json": json.loads(raw)}
            except Exception:
                return {"ok_http": True, "http": resp.status, "latency_ms": dt,
                        "json": None, "text": raw[:2000]}
    except urllib.error.HTTPError as e:
        dt = round((time.time() - t0) * 1000)
        detail = e.read().decode("utf-8", "replace")[:2000]
        return {"ok_http": False, "http": e.code, "latency_ms": dt, "error": detail}
    except Exception as e:  # timeout / URLError / socket
        dt = round((time.time() - t0) * 1000)
        kind = "TIMEOUT" if isinstance(e, (TimeoutError,)) or "timed out" in str(e).lower() else "ERROR"
        return {"ok_http": False, "http": None, "latency_ms": dt,
                "error": f"{kind}: {type(e).__name__}: {e}"}


def _first_item(payload: dict):
    """Walk the API response to the first MultiResult item, tolerant of shape drift."""
    if not isinstance(payload, dict):
        return None, None
    top_status = payload.get("status")
    result = payload.get("result", payload)
    items = None
    if isinstance(result, dict):
        items = result.get("items")
    if items and isinstance(items, list) and isinstance(items[0], dict):
        return top_status, items[0]
    return top_status, (result if isinstance(result, dict) else None)


_REFUSE = ("refuse", "refused", "clarify", "no_table", "ungrounded",
           "qualifier_dropped", "federated_refused", "not_federated",
           "tier2_rejected")   # the correctness gate declining an unsafe LLM answer
_ERROR = ("error", "exec_error", "tier2_exec_error")


def classify(resp: dict, cap: int | None, golden: set[str] | None = None) -> dict:
    """Return {verdict, status, route, rows, answer, sql, note}."""
    note = []
    if not resp.get("ok_http"):
        if resp.get("http") is None:
            v = "TIMEOUT" if str(resp.get("error", "")).startswith("TIMEOUT") else "ERROR"
        else:
            v = "ERROR"
        return {"verdict": v, "status": f"http_{resp.get('http')}", "route": "",
                "rows": 0, "answer": "", "sql": "", "note": str(resp.get("error", ""))[:200]}

    payload = resp.get("json")
    top_status, item = _first_item(payload)
    if item is None:
        return {"verdict": "ERROR", "status": str(top_status), "route": "",
                "rows": 0, "answer": "", "sql": "", "note": "unparseable payload"}

    # The item nests the executed result under item["result"] (rows/cols/sql/ok live
    # there); the outer item carries status/route/refuse_reason. Descend before reading.
    inner = item.get("result") if isinstance(item.get("result"), dict) else item
    status = str(item.get("status") or inner.get("status") or top_status or "").lower()
    inner_status = str(inner.get("status") or "").lower()
    route = str(item.get("route") or "")
    rows = inner.get("rows")
    rows = rows if isinstance(rows, list) else []
    n = len(rows)
    answer = str(item.get("answer") or inner.get("answer") or item.get("msg") or inner.get("msg") or "")
    sql = str(inner.get("sql") or item.get("sql") or "")
    err = str(inner.get("error") or item.get("error") or item.get("refuse_reason") or "")
    ok_flag = (inner.get("ok") is True) or (item.get("ok") is True)

    both = status + " " + inner_status
    if any(k in both for k in _ERROR):
        verdict = "ERROR"
        if err:
            note.append(err[:160])
    elif any(k in both for k in _REFUSE):
        verdict = "REFUSED"
        if answer or err:
            note.append((answer or err)[:120])
    elif ok_flag or status in ("ok", "answered") or inner_status == "answered":
        verdict = "PASS" if (n > 0 or answer) else "EMPTY"
    else:
        verdict = "EMPTY" if n == 0 else "PASS"

    # advisory structural check
    if cap is not None and n > cap:
        note.append(f"⚠ expected ≤{cap} rows, got {n}")

    # golden-anchor check: an ANSWER (even an empty one) built on none of the
    # acceptable tables is a wrong-table answer — the failure answerability can't see.
    if golden and verdict in ("PASS", "EMPTY") and sql:
        touched = sql_tables(sql)
        if touched and not (touched & golden):
            verdict = "GOLDEN-FAIL"
            note.append(f"✗ answered from {sorted(touched)[:3]}, expected one of {sorted(golden)}")

    return {"verdict": verdict, "status": status or str(top_status), "route": route,
            "rows": n, "answer": answer[:200], "sql": sql[:400], "note": "; ".join(note)}


def write_report(md_path: str, meta: dict, records: list[dict]) -> None:
    done = [r for r in records if r.get("verdict")]
    counts: dict[str, int] = {}
    for r in done:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lat = [r["latency_ms"] for r in done if isinstance(r.get("latency_ms"), (int, float))]
    npass = counts.get("PASS", 0)
    total = len(done)

    L = []
    L.append("# NL Query Suite — Report\n")
    L.append(f"- **Run:** {meta['ts']}")
    L.append(f"- **Endpoint:** `{meta['url']}`  |  **source_id:** {meta['source_id']}  |  **tenant:** {meta['tenant']}  |  **per-query timeout:** {meta['timeout']}s")
    L.append(f"- **Progress:** {total}/{meta['n_total']} queries executed")
    L.append("")
    L.append("> **What this measures:** *answerability* (did the pipeline return a grounded, non-empty, non-refused answer), **not** verified semantic correctness — there are no golden answers for these questions. `⚠` notes are advisory structural checks (e.g. \"top 5\" → ≤5 rows).")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"- **PASS (answered): {npass}/{total}**" + (f"  ({round(100*npass/total)}%)" if total else ""))
    for k in ("REFUSED", "EMPTY", "ERROR", "TIMEOUT"):
        if counts.get(k):
            L.append(f"- {k}: {counts[k]}")
    if lat:
        L.append(f"- Latency ms — min {min(lat)}, median {round(statistics.median(lat))}, max {max(lat)}")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| # | Verdict | Status | Route | Rows | ms | Query | Note |")
    L.append("|---|---------|--------|-------|------|----|-------|------|")
    emoji = {"PASS": "✅", "REFUSED": "🟠", "EMPTY": "⚪", "ERROR": "🔴", "TIMEOUT": "⏱️"}
    for r in records:
        if not r.get("verdict"):
            continue
        q = r["query"].replace("|", "\\|")
        q = (q[:70] + "…") if len(q) > 71 else q
        note = (r.get("note") or "").replace("|", "\\|").replace("\n", " ")
        L.append(f"| {r['id']} | {emoji.get(r['verdict'],'')} {r['verdict']} | {r['status']} "
                 f"| {r['route']} | {r['rows']} | {r['latency_ms']} | {q} | {note} |")
    L.append("")

    fails = [r for r in done if r["verdict"] in ("ERROR", "TIMEOUT", "EMPTY")]
    if fails:
        L.append("## Failures / empties — detail")
        L.append("")
        for r in fails:
            L.append(f"### {r['id']} — {r['verdict']}")
            L.append(f"> {r['query']}")
            L.append("")
            if r.get("sql"):
                L.append(f"- SQL: `{r['sql']}`")
            if r.get("note"):
                L.append(f"- Note: {r['note']}")
            L.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VEDA_QUERY_URL", "http://localhost:8080/api/v1/query"))
    ap.add_argument("--source-id", type=int, default=int(os.environ.get("VEDA_SUITE_SOURCE_ID", "2")))
    ap.add_argument("--tenant", default=os.environ.get("VEDA_SUITE_TENANT", "default"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("VEDA_SUITE_TIMEOUT", "120")))
    ap.add_argument("--limit", type=int, default=0, help="process at most N (pending) queries this run (0=all)")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing results; only run queries not already recorded (chunk-friendly)")
    ap.add_argument("--only", default="", help="comma-separated query ids to run (e.g. q17,q21) — A/B mode")
    ap.add_argument("--tag", default="", help="output file suffix, e.g. --tag ab_norepair → nl_query_suite_ab_norepair_*")
    ap.add_argument("--recheck", default="",
                    help="offline: re-apply GOLDEN_ANCHORS to an existing results jsonl (no API calls)")
    args = ap.parse_args()

    if args.recheck:
        recs = [json.loads(l) for l in open(args.recheck) if l.strip()]
        checked = fails = 0
        for r in recs:
            g = GOLDEN_ANCHORS.get(r.get("id"))
            # GOLDEN-FAIL included: a stored failure must re-fail (CI honesty)
            if not g or r.get("verdict") not in ("PASS", "EMPTY", "GOLDEN-FAIL"):
                continue
            checked += 1
            touched = sql_tables(r.get("sql") or "")
            if touched and not (touched & g):
                fails += 1
                print(f"GOLDEN-FAIL {r['id']}: answered from {sorted(touched)[:3]}, "
                      f"expected one of {sorted(g)} | {r.get('query', '')[:70]}")
        print(f"recheck {os.path.basename(args.recheck)}: {len(recs)} records, "
              f"{checked} golden-checked, {fails} GOLDEN-FAIL")
        return

    sfx = f"_{args.tag}" if args.tag else ""
    jsonl_path = os.path.join(HERE, f"nl_query_suite{sfx}_results.jsonl")
    md_path = os.path.join(HERE, f"nl_query_suite{sfx}_report.md")

    # Resume: load already-completed records; process only the pending ids. Lets a long
    # run be finished in bounded chunks (each background invocation has a lifetime cap).
    records: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    records.append(r)
                    done_ids.add(r.get("id"))
                except Exception:
                    pass
    else:
        open(jsonl_path, "w").close()  # fresh run → truncate

    if args.only:
        want = {q.strip() for q in args.only.split(",") if q.strip()}
        pending = [(qid, text) for (qid, text) in QUERIES if qid in want]
    else:
        pending = [(qid, text) for (qid, text) in QUERIES if qid not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    meta = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url": args.url, "source_id": args.source_id, "tenant": args.tenant,
            "timeout": args.timeout, "n_total": len(QUERIES)}
    print(f"resume={args.resume} done={len(done_ids)} pending_this_run={len(pending)}", flush=True)

    for i, (qid, text) in enumerate(pending, 1):
        cap = expected_row_cap(text)
        resp = post_query(args.url, text, args.source_id, args.tenant, args.timeout)
        cls = classify(resp, cap, GOLDEN_ANCHORS.get(qid))
        rec = {"id": qid, "query": text, "latency_ms": resp.get("latency_ms"), **cls,
               "raw": resp.get("json") if resp.get("ok_http") else resp.get("error")}
        records.append(rec)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        write_report(md_path, meta, records)  # keep report current for live monitoring
        print(f"[{i}/{len(pending)}] {qid} {cls['verdict']:8} {cls['status']:16} "
              f"route={cls['route']:12} rows={cls['rows']:<4} {resp.get('latency_ms')}ms",
              flush=True)

    done = [r for r in records if r.get("verdict")]
    npass = sum(1 for r in done if r["verdict"] == "PASS")
    print(f"\nDONE: {npass}/{len(done)} PASS. Report → {md_path}", flush=True)


if __name__ == "__main__":
    main()
