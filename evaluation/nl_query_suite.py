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
    ("q01", "Can you list the latest payments made for our properties, including whether they were debited or credited and the exact amounts?"),
    ("q02", "Show me the cheapest properties currently on the market for sale, along with their current market status."),
    ("q03", "Which properties have the most expensive financial records logged to date?"),
    ("q04", "What were the smallest payments processed across our real estate portfolio recently?"),
    ("q05", "What are the oldest properties we put up for sale, and what is their current status?"),
    ("q06", "Give me a quick breakdown of the most recent accounting entries for our assets."),
    ("q07", "Can I get a quick alphabetical list of our properties and the recent payments associated with them?"),
    ("q08", "Please show the financial logs ordered by property name in reverse alphabetical order."),
    ("q09", "Which of our properties on the market are priced above 10,000?"),
    ("q10", "Are there any recent payments processed that fall between 100 and 50,000?"),
    ("q11", "Show me the top 5 most recently dated accounting entries for our assets."),
    ("q12", "Could you pull the last 5 payment records based on their internal system processing ID?"),
    ("q13", "Which properties available for sale currently have an active status defined in the system?"),
    ("q14", "List our top most expensive properties that are currently on the market for sale."),
    ("q15", "What were the absolute smallest financial records we've ever logged?"),
    ("q16", "Can you sort our recent payments by their respective currency configurations?"),
    ("q17", "Group our properties for sale based on their currency configuration in descending order."),
    ("q18", "What are the oldest financial records we have on file by date?"),
    ("q19", "Which property payments were most recently modified or updated in our database?"),
    ("q20", "Show me the properties on the market that had their statuses updated most recently."),
    ("q21", "What is the average transaction amount for each property, categorized by whether it's an inflow or outflow?"),
    ("q22", "Which properties would generate the highest total revenue based on their market prices?"),
    ("q23", "Can you find the maximum financial value recorded for each property, broken down by entry type?"),
    ("q24", "What is the smallest single payment made towards each property by transaction type?"),
    ("q25", "How many active listings do we have for each property, grouped by their market status?"),
    ("q26", "What is the total aggregated sum of our accounting books, calculated for each property and entry type?"),
    ("q27", "On average, what is the expected sale price of our properties per current status category?"),
    ("q28", "If we sum up all payments received or paid out, which properties have the highest transaction volume?"),
    ("q29", "What is the average accounting entry amount we record across different properties?"),
    ("q30", "Can you flag any properties that have accumulated more than 1,000 in total payments?"),
    ("q31", "Which properties have a total combined valuation that is less than 10,000,000?"),
    ("q32", "How many individual accounting records does each property have registered in our system?"),
    ("q33", "What is the average payment amount when broken down by specific currency configurations?"),
    ("q34", "What is the highest sale price currently requested for a property, segmented by its assigned currency?"),
    ("q35", "Can we see the annual sum of financial records grouped by their transaction years?"),
    ("q36", "Identify the properties that have the most diverse set of transaction types (debits/credits/etc)."),
    ("q37", "Which properties on the market have the highest price variance, indicating shifting market expectations?"),
    ("q38", "What is the standard deviation in the amounts logged in our financial records per property?"),
    ("q39", "Are there any properties where every single payment is exactly equal to its maximum recorded payout?"),
    ("q40", "Find properties whose total market valuations are high, but their average expected price stays below 5,000,000."),
    ("q41", "If we rank our properties by their individual payment amounts, which ones take the top spots?"),
    ("q42", "Can we see a month-over-month running total of the expected revenue from our properties currently on the market?"),
    ("q43", "What percentage does each entry contribute to the total financial balance of its respective transaction type?"),
    ("q44", "How does the total payment amount for each property compare directly to its previously recorded date's total?"),
    ("q45", "Assign a dense ranking to our properties based strictly on their expected market sale prices."),
    ("q46", "Provide a chronological row-number mapping of the most recent financial accounting entries for each property."),
    ("q47", "What is the long-term yearly moving average of payments being processed for our assets?"),
    ("q48", "Divide our properties on the market into performance quartiles based on their expected pricing."),
    ("q49", "What is the statistical cumulative distribution of financial record amounts across our property portfolio?"),
    ("q50", "How far does each individual payment deviate from the overall average payment amount?"),
    ("q51", "Identify the very first expected sale price recorded historically for each of our properties."),
    ("q52", "What was the final recorded amount logged for each specific entry type in our accounting system?"),
    ("q53", "Can we forecast the immediate next payment amount sequentially based on our transaction history?"),
    ("q54", "What percentage does each listing's price represent compared to the maximum priced property in its category?"),
    ("q55", "What is the total rolling count of payments processed for each transaction type?"),
    ("q56", "Determine the exact percentile rank of each financial amount within our accounting records."),
    ("q57", "Calculate the short-term moving sum of the last 2 payments recorded for every property."),
    ("q58", "What is the historical running average of expected sale prices for properties added to our system?"),
    ("q59", "How much does each accounting record differ from the absolute minimum amount logged in its category?"),
    ("q60", "Rank all property payments within their specific assigned currency configuration."),
]

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
           "qualifier_dropped", "federated_refused", "not_federated")
_ERROR = ("error", "exec_error", "tier2_exec_error", "tier2_rejected")


def classify(resp: dict, cap: int | None) -> dict:
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
    args = ap.parse_args()

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
        cls = classify(resp, cap)
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
