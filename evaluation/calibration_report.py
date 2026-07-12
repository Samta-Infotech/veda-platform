#!/usr/bin/env python3
"""Calibration report (Phase D): decision confidences vs. actual outcomes.

Scans a suite results jsonl (whose `raw` field carries verbose explain traces) and
tabulates, per query: anchor confidence + top-2 margin + which mechanism decided
(source), against the verdict and golden check. This is the data the thresholds
(ANCHOR_CONFIDENCE_MARGIN, the 0.65 entity-accept, RERANK_NOISE_FLOOR) should be
fitted from — instead of hand-picked constants.

Usage: python3 evaluation/calibration_report.py <results.jsonl> [out.md]
"""
import json
import sys


def main(path, out=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    lines = ["| qid | verdict | anchor | conf | margin | source | lat_s |",
             "|-----|---------|--------|------|--------|--------|-------|"]
    stats = {"answered": [], "refused": [], "golden_fail": []}
    for r in rows:
        raw = r.get("raw")
        if not isinstance(raw, dict):
            continue
        items = (raw.get("result") or {}).get("items") or []
        it = items[0] if items else {}
        res = it.get("result") if isinstance(it.get("result"), dict) else {}
        secs = ((res.get("trace") or {}).get("sections") or {})
        anc = secs.get("anchor_selection") or {}
        conf, margin = anc.get("confidence"), anc.get("margin")
        lines.append(f"| {r['id']} | {r['verdict']} | {str(anc.get('anchor'))[:28]} "
                     f"| {conf} | {margin} | {anc.get('source')} "
                     f"| {(r.get('latency_ms') or 0)/1000:.1f} |")
        if conf is None:
            continue
        bucket = ("golden_fail" if r["verdict"] == "GOLDEN-FAIL"
                  else "answered" if r["verdict"] in ("PASS", "EMPTY")
                  else "refused")
        stats[bucket].append((conf, margin if margin is not None else 0.0))

    def _summ(name):
        vals = stats[name]
        if not vals:
            return f"- **{name}**: no traced samples"
        cs = sorted(c for c, _ in vals)
        ms = sorted(m for _, m in vals)
        return (f"- **{name}** (n={len(vals)}): confidence median "
                f"{cs[len(cs)//2]:.2f} (min {cs[0]:.2f}), margin median "
                f"{ms[len(ms)//2]:.3f} (min {ms[0]:.3f})")

    summary = ["# Calibration report", f"source: `{path}`", "",
               _summ("answered"), _summ("refused"), _summ("golden_fail"), "",
               "Reading: thresholds should separate the *answered* population from the",
               "*golden_fail* one. If their confidence ranges overlap, confidence alone",
               "cannot gate correctness at that stage — evidence for the typed-role",
               "signals rather than more threshold tuning.", "", ""]
    doc = "\n".join(summary + lines) + "\n"
    if out:
        open(out, "w").write(doc)
        print(f"wrote {out}")
    else:
        print(doc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
