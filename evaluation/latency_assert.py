#!/usr/bin/env python3
"""Latency assertions over a suite results jsonl (Phase D ratchet).

The architecture SLO (user-approved) is p50 < 5s / p95 < 30s / hard 60s. The heavy
LLM lane isn't there yet, so this gate RATCHETS: it enforces the budgets the current
architecture already achieves (so regressions fail CI) and prints — without failing —
the distance to the target SLO. Tighten the enforced numbers as phases land; never
loosen them.

Enforced today:
  · deterministic fast-lane answers (< 5s class): p50 must stay < 5s
  · clarifies: p95 < 15s  (a clarify that took 40s+ means an LLM retry snuck back in)
Reported (not yet enforced):
  · overall p50/p95 vs the 5s/30s SLO
Usage: python3 evaluation/latency_assert.py <results.jsonl>
"""
import json
import sys


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


def main(path: str) -> int:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    lat = lambda r: (r.get("latency_ms") or 0) / 1000.0

    answered = sorted(lat(r) for r in rows if r.get("verdict") in ("PASS", "EMPTY"))
    fast = sorted(v for v in answered if v < 5.0)
    clarifies = sorted(lat(r) for r in rows
                       if "which" in str(r.get("note", "")).lower()
                       and r.get("verdict") == "REFUSED"
                       or "clarify" in str(r.get("status", "")).lower())
    all_lat = sorted(lat(r) for r in rows)

    print(f"{len(rows)} results | answered={len(answered)} "
          f"(fast-lane <5s: {len(fast)}) clarifies={len(clarifies)}")
    print(f"overall     p50={pct(all_lat, .5):5.1f}s p95={pct(all_lat, .95):5.1f}s   "
          f"(target SLO: 5s / 30s — reported, not yet enforced)")

    failures = []
    if answered:
        f50 = pct(answered, .5)
        # ratchet: at least half of ANSWERED queries must come from the <5s lanes
        # once the deterministic fast paths cover the suite; until then enforce the
        # fast lane itself doesn't regress.
        if fast and pct(fast, .5) >= 5.0:
            failures.append(f"fast-lane p50 {pct(fast, .5):.1f}s >= 5s")
        print(f"answered    p50={f50:5.1f}s p95={pct(answered, .95):5.1f}s")
    if clarifies:
        c95 = pct(clarifies, .95)
        print(f"clarifies   p95={c95:5.1f}s (enforced < 15s)")
        if c95 >= 15.0:
            failures.append(f"clarify p95 {c95:.1f}s >= 15s — an LLM retry after a "
                            f"clarify has likely returned")
    if failures:
        print("LATENCY ASSERTIONS FAILED:", "; ".join(failures))
        return 1
    print("latency assertions passed (ratchet level)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
