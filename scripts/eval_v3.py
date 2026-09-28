"""questions_v3.txt — AUTOMATIC grader (no hand grading).

Grades the records written by scripts/eval_v3_inproc.py (`compound` and/or `alone` mode)
against the assertions in reports/raw/qv3_ground_truth.json. Each ground-truth part
carries ONE of:

  doc parts   "expect_text": [group, group, …]
                 every group is a list of alternatives (case-insensitive regexes); the
                 answer text must match at least one alternative of EVERY group.
  sql parts   "expect": {
                 "row_count": N | [min, max]      rows the answer returns (optional)
                 "contains":  [[alt, alt], …]     each group: some cell of the answer's rows
                                                  equals (normalised; numbers within 0.006)
                                                  one of the alts
                 "excludes":  [v, …]              no cell may equal any of these
                 "columns_any": [[alt, …], …]     each group: some answer column name
                                                  contains one of the alts (optional)
                 "col_all": {"like": s, "gt"|"ge"|"lt"|"le": x}   every value of every
                                                  column whose name contains s satisfies it
                 "scalar": {"value": x, "tol": t} a one-row answer with a numeric cell x ± t
                 "first_row": [alt, …]            the FIRST row has a cell equal to one alt
                                                  (a superlative answered as a sorted list)
               }
            or "expect_any": [expect, expect, …]  — the ground truth's documented
               alternative readings; the part is correct when ANY one holds.
               `grader_note` explains the assertion (why these values decide correctness).

Outcome per part: correct · wrong · clarify (clarify / refused / error — no answer) ·
timeout. A part the answer-less outcomes never reach "wrong".

Usage (host):
    python scripts/eval_v3.py --compound reports/raw/agent_v3/compound.jsonl \
                              [--alone reports/raw/agent_v3/alone.jsonl] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _flags  # noqa: E402


def _norm(v: Any) -> str:
    s = str(v if v is not None else "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    try:
        f = float(s.replace(",", ""))
        return repr(int(f)) if f.is_integer() else repr(round(f, 4))
    except ValueError:
        return s


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _has(cells: set, nums: list, alt) -> bool:
    if _norm(alt) in cells:
        return True
    a = _num(alt)
    return a is not None and any(abs(a - x) <= 0.006 for x in nums)


def _cells(rows) -> set:
    out = set()
    for r in rows or []:
        vals = r.values() if isinstance(r, dict) else (r if isinstance(r, (list, tuple)) else [r])
        for v in vals:
            out.add(_norm(v))
            if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}[T ]", v):
                out.add(_norm(v[:10]))
    return out


def grade_sql(item: Dict[str, Any], exp: Dict[str, Any]) -> (str, List[str]):
    if "any" in exp:
        whys = []
        for alt in exp["any"]:
            g, why = grade_sql(item, alt)
            if g == "correct":
                return g, []
            whys.append("; ".join(why))
        return "wrong", [" | ".join(whys)]
    rows = item.get("rows") or []
    n = item.get("n_rows") if item.get("n_rows") is not None else len(rows)
    why = []
    rc = exp.get("row_count")
    if rc is not None:
        lo, hi = (rc, rc) if isinstance(rc, int) else (rc[0], rc[1])
        if not (lo <= int(n or 0) <= hi):
            why.append(f"row_count {n} not in [{lo},{hi}]")
    cells = _cells(rows)
    nums = [x for x in (_num(c) for c in cells) if x is not None]
    for grp in exp.get("contains") or []:
        alts = grp if isinstance(grp, list) else [grp]
        if not any(_has(cells, nums, a) for a in alts):
            why.append(f"missing one of {alts}")
    for v in exp.get("excludes") or []:
        if _norm(v) in cells:
            why.append(f"contains excluded {v}")
    cols = [str(c).lower() for c in (item.get("cols") or [])]
    ca = exp.get("col_all")
    if ca:
        idx = [i for i, c in enumerate(cols) if ca["like"] in c]
        if not idx:
            why.append(f"no column like {ca['like']}")
        for r in rows:
            vals = list(r.values()) if isinstance(r, dict) else list(r)
            for i in idx:
                x = _num(vals[i]) if i < len(vals) else None
                if x is None:
                    continue
                if (("gt" in ca and not x > ca["gt"]) or ("ge" in ca and not x >= ca["ge"])
                        or ("lt" in ca and not x < ca["lt"]) or ("le" in ca and not x <= ca["le"])):
                    why.append(f"{cols[i]}={x} violates {ca}")
                    break
    for grp in exp.get("columns_any") or []:
        if not any(any(str(a).lower() in c for c in cols) for a in grp):
            why.append(f"no column like {grp}")
    fr = exp.get("first_row")
    if fr is not None:
        first = _cells(rows[:1])
        fnums = [x for x in (_num(c) for c in first) if x is not None]
        if not rows or not any(_has(first, fnums, a) for a in fr):
            why.append(f"first row lacks one of {fr}")
    sc = exp.get("scalar")
    if sc is not None:
        nums = []
        for r in rows[:1]:
            for v in (r.values() if isinstance(r, dict) else r):
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    pass
        if len(rows) != 1 or not any(abs(x - float(sc["value"])) <= float(sc.get("tol", 0.01)) for x in nums):
            why.append(f"scalar != {sc['value']}")
    return ("correct" if not why else "wrong"), why


def grade_doc(item: Dict[str, Any], groups: List[List[str]]) -> (str, List[str]):
    text = str(item.get("answer") or "")
    why = []
    for grp in groups:
        alts = grp if isinstance(grp, list) else [grp]
        if not any(re.search(a, text, re.I) for a in alts):
            why.append(f"text lacks {alts}")
    return ("correct" if not why else "wrong"), why


def grade_item(item: Optional[Dict[str, Any]], gp: Dict[str, Any]) -> Dict[str, Any]:
    if item is None:
        return {"grade": "clarify", "why": ["no part"], "outcome": None}
    oc = item.get("outcome") or ("answered" if item.get("status") == "ok" else item.get("status"))
    if oc == "timeout":
        return {"grade": "timeout", "why": [], "outcome": oc}
    if oc != "answered":
        # a handbook part's "the documents do not answer this" is WRONG: the ground truth
        # cites the passage that answers it (the hand grading counted it so too)
        if gp.get("type") == "doc" and oc == "refused":
            return {"grade": "wrong", "why": ["documents said to not answer: " + str(item.get("answer"))[:80]],
                    "outcome": oc}
        return {"grade": "clarify", "why": [str(item.get("refuse_reason") or item.get("pipeline_status") or oc)],
                "outcome": oc}
    if gp.get("type") == "doc" or "expect_text" in gp:
        g, why = grade_doc(item, gp.get("expect_text") or [])
    else:
        exp = {"any": gp["expect_any"]} if gp.get("expect_any") else (gp.get("expect") or {})
        g, why = grade_sql(item, exp)
    return {"grade": g, "why": why, "outcome": oc}


def _source_of_part(gp):
    s = str(gp.get("source") or "").lower()
    if "invoices" in s:
        return 4
    if "catalog" in s or "parquet" in s:
        return 5
    if "docs" in s or ".pdf" in s or gp.get("type") == "doc":
        return 3
    return 2


def _load(path):
    """Records only — a leading '_meta' line (scripts/eval_v3_inproc.py's flags stamp,
    if present) is dropped here and read separately by `_meta_of`."""
    return [r for r in (json.loads(l) for l in open(path) if l.strip()) if "_meta" not in r]


def _meta_of(path):
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if "_meta" in r:
                return r["_meta"]
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="reports/raw/qv3_ground_truth.json")
    ap.add_argument("--compound", default=None)
    ap.add_argument("--alone", default=None)
    ap.add_argument("--json", default=None)
    _flags.add_expect_arg(ap)
    a = ap.parse_args()

    run_flags = {}
    for label, path in (("compound", a.compound), ("alone", a.alone)):
        if not path:
            continue
        meta = _meta_of(path)
        if meta and meta.get("flags") is not None:
            run_flags[label] = meta["flags"]
            _flags.print_flags_header(meta["flags"], title=f"eval_v3: {label} input's recorded flags")
        else:
            print(f"NOTE: {path} has no recorded '_meta.flags' (collected before this wiring) "
                  "— cannot verify or --expect against collect-time flags.")
    if a.expect:
        if not run_flags:
            sys.exit("--expect given but no input file carries recorded flags to check it against")
        for label, flags in run_flags.items():
            _flags.enforce_expect(flags, a.expect)

    gt = {q["n"]: q for q in json.load(open(a.gt))["questions"]}
    out = {"compound": {}, "alone": {}}
    if a.compound:
        for r in _load(a.compound):
            n = r["n"]
            items = r.get("items") or []
            parts = gt[n]["parts"]
            aligned = r.get("split") and len(items) == len(parts)
            for i, gp in enumerate(parts):
                it = items[i] if aligned else None
                if not aligned and r.get("split"):
                    # a different number of parts than the ground truth: the FIRST part that
                    # ran on this ground-truth part's source (the hand grading's reading)
                    want = _source_of_part(gp)
                    it = next((x for x in items if str(x.get("source_id")) == str(want)), None)
                out["compound"][f"{n}{gp['part']}"] = grade_item(it, gp)
    if a.alone:
        for r in _load(a.alone):
            if r.get("skipped"):
                continue
            gp = next(p for p in gt[r["n"]]["parts"] if p["part"] == r["part"])
            it = (r.get("items") or [None])[0]
            out["alone"][f"{r['n']}{r['part']}"] = grade_item(it, gp)
    for mode, res in out.items():
        if not res:
            continue
        tot = {k: sum(1 for v in res.values() if v["grade"] == k) for k in ("correct", "wrong", "clarify", "timeout")}
        hz = {k: sum(1 for p, v in res.items() if p.endswith("a") and v["grade"] == k)
              for k in ("correct", "wrong", "clarify", "timeout")}
        print(f"{mode}: all {len(res)} → {tot} | homzhub(a) → {hz}")
        for p in sorted(res, key=lambda x: (int(re.match(r'\d+', x).group()), x[-1])):
            v = res[p]
            if v["grade"] != "correct":
                print(f"   {p:>4} {v['grade']:<8} {'; '.join(v['why'])[:160]}")
    if a.json:
        json.dump({"flags": run_flags, **out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
