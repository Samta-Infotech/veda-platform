"""Extractor / grounding evaluation for the meaning-first pass (Stages 0.1, 2, 3, 6).

  --mode legacy     Stage 0.1: the CURRENT understanding layer (veda/understanding/extractor +
                    grounding) on the 20 external questions, flags ON, pinned to source 2,
                    through the real pipeline — its trace (`understanding`: raw frame,
                    grounded anchor, anchor_method, confidence, fully_grounded, decision)
                    beside the router's primary (`schema_linking`) and the ground-truth table.
  --mode frame      Stages 2/3: the NEW frame path on the 20 — slot by slot (entity, order
                    column/kind/dir, limit, filters, group, clarify) against the ground
                    truth in scripts/eval_question_txt.py SPECS.
  --mode synthetic  Stage 2 exit: slot accuracy of the frame extractor on the synthetic
                    question set (veda_questions.jsonl), grading the GROUNDED predicted frame
                    against the grounded expected frame, per slot.

Run inside the inference container, pinned scope:
    cd /app/veda_core && python /app/scripts/eval_frame_extractor.py --mode frame --source 2
    QUERY_UNDERSTANDING_ENABLED=1 ANALYTICAL_SQL_V2=1 python /app/scripts/eval_frame_extractor.py --mode legacy
Writes JSON to --out (default /app/reports/raw/frame_eval_<mode>.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")
sys.path.insert(0, "/app/scripts")


def _ctx(sid):
    from veda_core.context import RequestContext, set_context
    set_context(RequestContext(source_id=int(sid), tenant="default", source_ids=(int(sid),),
                               cache_back=False))


def _load_qs(path="/app/question.txt"):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and l.strip().lower() != "question"]


def _sm():
    from veda_hybrid import _load_semantic_model
    sm, cols = _load_semantic_model()
    return sm, cols


# ── Stage 0.1 ──────────────────────────────────────────────────────────────────────────
def mode_legacy(a):
    import eval_question_txt as E
    from veda_hybrid import run_hybrid_query
    import veda.explain as X
    captured = {}
    _orig = X.ExplainTrace.finalize

    def _cap(self, *aa, **kw):
        try:
            captured["t"] = self.to_dict()
        except Exception:
            pass
        return _orig(self, *aa, **kw)
    X.ExplainTrace.finalize = _cap
    qs = _load_qs()
    rows = []
    for spec in E.SPECS:
        q = qs[spec["n"] - 1]
        captured.clear()
        _ctx(a.source)
        t0 = time.time()
        try:
            run_hybrid_query(q, verbose=False)
        except Exception as e:
            print(f"[{spec['n']:02d}] crash {type(e).__name__}: {e}")
        sec = ((captured.get("t") or {}).get("sections") or {})
        u = sec.get("understanding") or {}
        sl = sec.get("schema_linking") or {}
        row = {"n": spec["n"], "question": q, "gt_table": spec["table"],
               "router_primary": sl.get("router_primary") or sl.get("selected_table"),
               "status": u.get("status"), "decision": u.get("decision"),
               "raw": {k: u.get(k) for k in ("intent", "grain", "measure", "entities")},
               "grounded_anchor": u.get("anchor"), "anchor_method": u.get("anchor_method"),
               "confidence": u.get("confidence"), "fully_grounded": u.get("fully_grounded"),
               "unresolved": u.get("unresolved"), "elapsed_s": round(time.time() - t0, 1)}
        row["anchor_correct"] = row["grounded_anchor"] == spec["table"]
        row["anchor_by_name"] = row["anchor_method"] in ("exact_name", "glossary", "table_vocabulary", "name_tokens")
        rows.append(row)
        print(f"[{spec['n']:02d}] gt={spec['table']:<24} router={str(row['router_primary']):<28} "
              f"qu={str(row['grounded_anchor']):<28} method={row['anchor_method']} "
              f"decision={row['decision']} conf={row['confidence']}", flush=True)
    summ = {"n": len(rows),
            "extractor_anchor_correct": sum(r["anchor_correct"] for r in rows),
            "extractor_anchor_correct_by_name": sum(r["anchor_correct"] and r["anchor_by_name"] for r in rows),
            "router_correct": sum(r["router_primary"] == r["gt_table"] for r in rows),
            "decisions": dict(Counter(str(r["decision"]) for r in rows))}
    return {"summary": summ, "rows": rows}


# ── Stages 2/3 on the 20 ──────────────────────────────────────────────────────────────
def _grade_grounded(spec, res):
    import eval_question_txt as E
    out = {"entity": None, "order": None, "limit": None, "filters": None, "group": None,
           "verdict": None}
    if spec.get("expect") == "clarify":
        out["verdict"] = "ok_clarify" if res.kind == "clarify" else f"expected_clarify_got_{res.kind}"
        return out
    if res.kind != "sql":
        out["verdict"] = res.kind
        g = res.grounded
        cands = []
        if res.message:
            cands = [t for t in (spec["table"],) if t.split("_")[-1] in res.message.lower()]
        out["clarify_names_gt"] = bool(cands)
        return out
    g = res.grounded
    tabs = set(res.tables)
    out["entity"] = spec["table"] in tabs
    want = spec.get("order")
    if want:
        kind, d = want
        okc = E.ORDER_COLS.get(spec["table"], {}).get(kind, [])
        out["order"] = bool(g.order and g.order[1] in okc and (d == "any" or g.order[2] == d))
    else:
        out["order"] = True
    out["limit"] = (spec.get("limit") is None) or (g.limit == spec.get("limit"))
    import re
    sql = (res.sql or "").lower()
    out["filters"] = all(re.search(m, sql) for m in (spec.get("where") or []))
    out["group"] = (not spec.get("group")) or bool(g.group_by)
    out["verdict"] = "ok" if all(out[k] for k in ("entity", "order", "limit", "filters", "group")) else "slot_miss"
    return out


def mode_frame(a):
    import eval_question_txt as E
    from veda.understanding.frame_path import run_frame_path
    _ctx(a.source)
    sm, _ = _sm()
    qs = _load_qs()
    rows = []
    for spec in E.SPECS:
        q = qs[spec["n"] - 1]
        t0 = time.time()
        res = run_frame_path(q, sm, force=True)
        gr = _grade_grounded(spec, res)
        g = res.grounded
        row = {"n": spec["n"], "question": q, "gt_table": spec["table"], "kind": res.kind,
               "reason": res.reason, "anchor": res.anchor,
               "anchor_method": g.anchor_method if g else None,
               "frame": (res.trace or {}).get("frame"), "grounding": (res.trace or {}).get("grounding"),
               "probes": (res.trace or {}).get("probes"), "sql": res.sql, "message": res.message,
               "extract": (res.trace or {}).get("extract"), "grade": gr,
               "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"[{spec['n']:02d}] {res.kind:<8} {gr['verdict']:<22} anchor={res.anchor} "
              f"({row['anchor_method']}) {('SQL: ' + (res.sql or '')[:150]) if res.sql else (res.message or res.reason)[:150]}",
              flush=True)
    ents = [r for r in rows if r["grade"]["entity"] is not None]
    summ = {"n": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)),
            "verdicts": dict(Counter(r["grade"]["verdict"] for r in rows)),
            "entity_correct": sum(1 for r in rows if r["grade"]["entity"]),
            "entity_correct_or_named_clarify": sum(1 for r in rows if r["grade"]["entity"]
                                                   or r["grade"].get("clarify_names_gt")
                                                   or r["grade"]["verdict"] == "ok_clarify"),
            "slot_accuracy": {k: f"{sum(1 for r in ents if r['grade'][k])}/{len(ents)}"
                              for k in ("entity", "order", "limit", "filters", "group")},
            "parse_failures": sum(1 for r in rows if (r.get("extract") or {}).get("parse_failure")),
            "extract_ms_median": sorted([(r.get("extract") or {}).get("slm_ms") or 0 for r in rows])[len(rows) // 2]}
    return {"summary": summ, "rows": rows}


# ── Stage 2 exit: synthetic slot accuracy ──────────────────────────────────────────────
def mode_synthetic(a):
    from veda.understanding.vocabulary import scope_vocab
    from veda.understanding.frame_extractor import extract_frame
    from veda.understanding.frame import normalise
    from veda.understanding.frame_grounding import ground_frame, GroundedFrame
    _ctx(a.source)
    sm, _ = _sm()
    vocab = scope_vocab(sm)
    qs = [q for q in vocab.examples if q.get("origin") in ("template", "paraphrase")]
    by_kind = defaultdict(list)
    for q in qs:
        by_kind[q.get("kind")].append(q)
    rnd = random.Random(7)
    sample = []
    per = max(1, a.limit // max(1, len(by_kind)))
    for k, lst in sorted(by_kind.items()):
        rnd.shuffle(lst)
        sample += lst[:per]
    sample = sample[:a.limit]
    slots = ("anchor", "measure", "filters", "group_by", "order", "limit")
    hits, tot = Counter(), Counter()
    parse_fail = 0
    rows = []
    for q in sample:
        # the example itself must not be its own few-shot
        saved = vocab.examples
        vocab.examples = [e for e in saved if e.get("question") != q["question"]]
        if vocab.embeddings is not None and len(vocab.examples) != len(saved):
            keep = [i for i, e in enumerate(saved) if e.get("question") != q["question"]]
            emb_saved = vocab.embeddings
            vocab.embeddings = emb_saved[keep]
        else:
            emb_saved = None
        st = {}
        fr = extract_frame(q["question"], vocab, stats=st)
        vocab.examples = saved
        if emb_saved is not None:
            vocab.embeddings = emb_saved
        if fr is None:
            parse_fail += 1
            rows.append({"q": q["question"], "fail": st.get("errors")})
            continue
        gp = ground_frame(fr, vocab, sm, q["question"])
        ge = ground_frame(normalise(q["frame"]), vocab, sm, q["question"])

        def view(g):
            if not isinstance(g, GroundedFrame):
                return {s: ("<" + (type(g).__name__ if g is not None else "none") + ">") for s in slots}
            return {"anchor": g.anchor, "measure": g.measure,
                    "filters": sorted((f.column, f.op, str(f.value).lower()) for f in g.filters),
                    "group_by": sorted(g.group_by), "order": g.order, "limit": g.limit}
        vp, ve = view(gp), view(ge)
        row = {"q": q["question"], "kind": q.get("kind"), "origin": q.get("origin"), "table": q["table"]}
        for s in slots:
            tot[s] += 1
            ok = vp[s] == ve[s] if s != "anchor" else vp["anchor"] == q["table"]
            hits[s] += ok
            row[s] = ok
            if not ok:
                row[f"{s}_pred"], row[f"{s}_exp"] = vp[s], ve[s]
        rows.append(row)
    allslots = sum(hits.values()), sum(tot.values())
    summ = {"n": len(sample), "parse_failures": parse_fail,
            "slot_accuracy": {s: f"{hits[s]}/{tot[s]}" for s in slots},
            "overall": f"{allslots[0]}/{allslots[1]} = {allslots[0] / max(1, allslots[1]):.3f}"}
    return {"summary": summ, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("legacy", "frame", "synthetic"), required=True)
    ap.add_argument("--source", default="2")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fn = {"legacy": mode_legacy, "frame": mode_frame, "synthetic": mode_synthetic}[a.mode]
    res = fn(a)
    print(json.dumps(res["summary"], indent=1, default=str))
    out = a.out or f"/app/reports/raw/frame_eval_{a.mode}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("wrote", out)


if __name__ == "__main__":
    main()
