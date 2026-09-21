"""Per-purpose SLM report over the explain-trace log — the Checkpoint A measurement gate.

Nothing about the SLM layer can be tuned until the per-purpose numbers exist. Every
call_slm() invocation records {purpose, model, duration_ms, ok} onto its query's trace
(slm/_call_slm.py -> veda/explain.py::slm_call), and finish() stamps the per-purpose
TOKEN totals from the ContextVar accumulator into the same trace's `llm_usage` section.
This script folds the last N trace records into one table:

    purpose                    calls  median   p90   ok%   tokens(p+c)  model(s)

so the questions that actually matter are answerable from one command:
  * which purpose is the turn's latency, and is it the one we think it is;
  * is every purpose running on the model it was configured to run on (a purpose
    silently falling back to the 7B *coder* model because its instruct model is not
    served reads here as the wrong model name, not as a mystery quality drop);
  * which purposes fire more than once per turn (the waste this pass is hunting);
  * is `llm_usage` actually being populated (it was not: _fold_usage() had no caller,
    so get_usage() always reported calls=0 and the section never got written).

Reads BOTH shapes of trace record: the compact record now carries `slm_calls` +
`llm_usage` directly, and older/verbose records carry the same under `full.sections`.
A record with neither is counted as "no SLM ledger" and reported — that count is the
Exit-A criterion "llm_usage present on 100% of new traces".

Usage (host or container — it only reads a file):
    python scripts/slm_purpose_report.py [-n 50] [--log logs/explain_trace.jsonl]
    python scripts/slm_purpose_report.py --since-marker   # only records after the marker
    python scripts/slm_purpose_report.py --mark           # drop a marker, then run queries

--mark writes the current line count to a sidecar file so a before/after comparison is
one flag rather than manual line arithmetic.

Exit code 0 always (this is a report, not a gate) unless --require-usage is passed, which
exits 1 when any record is missing its llm_usage section.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

_DEFAULT_LOG = "logs/explain_trace.jsonl"
_MARKER = ".slm_purpose_report.marker"


def _pctl(vals, q):
    """The q-th percentile (0-1) by nearest-rank — no numpy dependency, and correct
    for the small samples this report runs on (a 40-query battery gives 40 points)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _records(path, limit, offset=0):
    """The last `limit` trace records at/after line `offset`. Tolerates partial final
    lines (the engine appends under concurrency) by skipping unparseable ones."""
    if not os.path.exists(path):
        sys.exit(f"trace log not found: {path}\n"
                 f"(EXPLAIN_TRACE_PERSIST must be on, and at least one query must have run)")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (ValueError, TypeError):
                continue
    return out[-limit:] if limit else out


def _ledger(rec):
    """(slm_calls, llm_usage) from either record shape. The compact record carries both
    at the top level; a verbose record repeats them under full.sections."""
    calls = rec.get("slm_calls")
    usage = rec.get("llm_usage")
    if calls is None or usage is None:
        sections = ((rec.get("full") or {}).get("sections") or {})
        if calls is None:
            calls = (sections.get("slm") or {}).get("calls")
        if usage is None:
            usage = sections.get("llm_usage")
    return (calls or []), (usage or {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=50,
                    help="how many of the most recent trace records to fold (default 50; 0 = all)")
    ap.add_argument("--log", default=_DEFAULT_LOG, help=f"trace log path (default {_DEFAULT_LOG})")
    ap.add_argument("--mark", action="store_true",
                    help="record the current end-of-log position and exit (for before/after runs)")
    ap.add_argument("--since-marker", action="store_true",
                    help="only fold records written after the last --mark")
    ap.add_argument("--require-usage", action="store_true",
                    help="exit 1 if any folded record is missing its llm_usage section")
    ap.add_argument("--by-trace", action="store_true",
                    help="also print per-purpose CALLS PER TURN (the waste view)")
    args = ap.parse_args()

    if args.mark:
        n = sum(1 for _ in open(args.log)) if os.path.exists(args.log) else 0
        with open(_MARKER, "w") as f:
            f.write(str(n))
        print(f"marker set at line {n} of {args.log}")
        return 0

    offset = 0
    if args.since_marker:
        if not os.path.exists(_MARKER):
            sys.exit(f"no marker file ({_MARKER}) — run --mark first")
        offset = int(open(_MARKER).read().strip() or 0)

    recs = _records(args.log, args.num, offset)
    if not recs:
        print(f"no trace records in {args.log}"
              + (f" after line {offset}" if offset else ""))
        return 0

    # purpose -> durations / ok flags / models; purpose -> token totals
    durs = defaultdict(list)
    oks = defaultdict(list)
    models = defaultdict(set)
    toks = defaultdict(lambda: [0, 0])     # [prompt, completion]
    per_turn = defaultdict(list)           # purpose -> calls in each trace
    no_ledger = 0
    no_usage = 0

    for rec in recs:
        calls, usage = _ledger(rec)
        if not calls:
            no_ledger += 1
        if not usage:
            no_usage += 1
        counts = defaultdict(int)
        for c in calls:
            p = c.get("purpose") or "?"
            durs[p].append(float(c.get("duration_ms") or 0.0))
            oks[p].append(bool(c.get("ok")))
            if c.get("model"):
                models[p].add(str(c["model"]))
            counts[p] += 1
        for p, n in counts.items():
            per_turn[p].append(n)
        for p, u in (usage.get("per_purpose") or {}).items():
            toks[p][0] += int(u.get("prompt_tokens") or 0)
            toks[p][1] += int(u.get("completion_tokens") or 0)

    n_tr = len(recs)
    print(f"\nSLM per-purpose report — {n_tr} trace record(s) from {args.log}"
          + (f" (after line {offset})" if offset else ""))
    print(f"  records with an SLM ledger : {n_tr - no_ledger}/{n_tr}")
    print(f"  records with llm_usage     : {n_tr - no_usage}/{n_tr}"
          + ("   <-- llm_usage is NOT being stamped" if no_usage == n_tr else ""))

    if not durs:
        print("\n  no SLM calls recorded in these traces.")
        return 0

    hdr = f"\n  {'purpose':<26} {'calls':>5} {'/turn':>6} {'med ms':>8} {'p90 ms':>8} {'ok%':>5}  {'tokens p+c':>14}  model(s)"
    print(hdr)
    print("  " + "-" * (len(hdr) + 14))
    for p in sorted(durs, key=lambda k: -sum(durs[k])):
        d = durs[p]
        ok = oks[p]
        pt, ct = toks[p]
        avg_turn = (sum(per_turn[p]) / len(per_turn[p])) if per_turn[p] else 0.0
        mdl = ",".join(sorted(models[p])) or "-"
        print(f"  {p:<26} {len(d):>5} {avg_turn:>6.2f} {statistics.median(d):>8.0f} "
              f"{_pctl(d, 0.90):>8.0f} {100.0 * sum(ok) / len(ok):>5.0f}  "
              f"{pt:>6},{ct:<7}  {mdl}")

    tot_ms = sum(sum(v) for v in durs.values())
    tot_calls = sum(len(v) for v in durs.values())
    print(f"\n  total: {tot_calls} calls, {tot_ms:,.0f} ms of SLM time across {n_tr} turn(s)"
          f"  ({tot_calls / n_tr:.2f} calls/turn, {tot_ms / n_tr:,.0f} ms/turn)")

    if args.by_trace:
        print("\n  calls per turn (purposes firing more than once are the waste candidates):")
        for p in sorted(per_turn, key=lambda k: -(sum(per_turn[k]) / max(1, len(per_turn[k])))):
            v = per_turn[p]
            print(f"    {p:<26} mean {sum(v) / len(v):>4.2f}  max {max(v)}  "
                  f"(in {len(v)}/{n_tr} turns)")

    if args.require_usage and no_usage:
        print(f"\nFAIL: {no_usage}/{n_tr} records carried no llm_usage section")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
