"""questions_v3.txt — IN-PROCESS measurement of the compound path (2026-09-26).

Runs the 20 three-part messages by calling `veda_hybrid.run_hybrid_query` directly inside the
inference container — no HTTP, no api tier — with every budget lifted, so the number measured
is the DESIGN's, not a timeout's. Then runs each of the 60 parts ALONE, pinned to its own
source, to separate "decomposition failed" from "the part itself fails".

Nothing in the engine is changed. The harness only:
  * sets the budget env vars before the engine imports config
    (FRAME_PATH_ENABLED=1, FRAME_INTENTS_TIMEOUT=600, COMPOUND_PART_BUDGET_S=600,
     COMPOUND_TOTAL_BUDGET_S=1800, SLM_TIMEOUT_SECS=600);
  * forces every SLM HTTP call's timeout to 600 s at the backend (per-site caps such as the
    summary's 15 s would otherwise still cut calls short) and records each call's wall time,
    purpose and tokens;
  * wraps (never replaces) the extractor, the grounding, the part runner and execute_sql to
    record what they saw and returned.

Usage (inside the inference container):
    cd /app/veda_core && python /app/scripts/eval_v3_inproc.py compound --out /app/reports/raw/v3inproc/compound.jsonl
    cd /app/veda_core && python /app/scripts/eval_v3_inproc.py alone    --out /app/reports/raw/v3inproc/alone.jsonl
    python scripts/eval_v3_inproc.py sheet --compound … --alone … > sheet.md   (host; grading aid)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _flags  # noqa: E402

_ENV = {"FRAME_PATH_ENABLED": "1", "FRAME_INTENTS_TIMEOUT": "600", "COMPOUND_PART_BUDGET_S": "600",
        "COMPOUND_TOTAL_BUDGET_S": "1800", "SLM_TIMEOUT_SECS": "600",
        # planner agent (2026-09-26): ON unless the caller exports AGENT_PLANNER_ENABLED=0;
        # its own wall budget lifted like every other budget (steps / tool calls stay)
        "AGENT_PLANNER_ENABLED": os.environ.get("AGENT_PLANNER_ENABLED", "1"),
        "AGENT_PART_BUDGET_S": os.environ.get("AGENT_PART_BUDGET_S", "600")}
PROFILES = {"2": {"name": "homzhub", "source_type": "relational"},
            "3": {"name": "docs_contracts", "source_type": "document"},
            "4": {"name": "invoices_csv", "source_type": "datalake"},
            "5": {"name": "catalog_parquet", "source_type": "datalake"}}
SCOPE = (2, 3, 4, 5)


def _questions(path):
    return [(int(n), q.strip()) for n, q in re.findall(r"^(\d+)\.\s+(.+)$", open(path).read(), re.M)]


def _inline(sql, params):
    out, i = [], 0
    for piece in re.split(r"(%s)", str(sql or "")):
        if piece == "%s" and i < len(params or []):
            v = params[i]
            i += 1
            out.append(str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'")
        else:
            out.append(piece)
    return "".join(out)


# ── instrumentation ─────────────────────────────────────────────────────────────────
class Rec:
    lock = threading.Lock()

    def __init__(self):
        self.reset()

    def reset(self):
        self.slm = []            # [{purpose?, ms, prompt_tokens, completion_tokens, part}]
        self.sql = []            # [{part, sql, params, inlined, probe}]
        self.extract = None      # {stats, raw, intents}
        self.groundings = None
        self.parts_ms = []
        self.part = None         # index of the part currently running (parts are sequential)
        self.agent = []          # [{part, trigger?, kind, reason, steps, tool_calls, wall_s, validation, …}]


R = Rec()


def _instrument():
    for k, v in _ENV.items():
        os.environ[k] = v
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/veda_core")
    import slm._call_slm as C
    be = C.get_backend()
    _orig_call = be.call

    def _call(user_message, *a, **kw):
        kw["timeout"] = 600
        t0 = time.time()
        content, usage = _orig_call(user_message, *a, **kw)
        with Rec.lock:
            R.slm.append({"ms": round((time.time() - t0) * 1000, 1), "part": R.part,
                          "prompt_tokens": (usage or {}).get("prompt_tokens"),
                          "completion_tokens": (usage or {}).get("completion_tokens"),
                          "schema": bool(kw.get("json_schema"))})
        return content, usage
    be.call = _call
    # the SLM deadline would still cap a call to the part's remaining budget — the
    # budgets above make that 600 s; nothing else to do.

    import veda.execution as VX
    import veda.pipeline as VP
    _orig_exec = VX.execute_sql

    def _exec(sql, params=None, *a, **kw):
        s = str(sql or "")
        probe = bool(kw.get("timeout_ms")) or "FILTER (WHERE" in s or s.startswith("SELECT DISTINCT CAST(")
        with Rec.lock:
            R.sql.append({"part": R.part, "sql": s, "params": list(params or []),
                          "inlined": _inline(s, list(params or [])), "probe": probe})
        return _orig_exec(sql, params, *a, **kw)
    VX.execute_sql = _exec
    if hasattr(VP, "execute_sql"):
        VP.execute_sql = _exec

    import veda.understanding.frame_extractor as FE
    _orig_ei, _orig_fcall = FE.extract_intents, FE._call

    def _fcall(user, system, schema, **kw):
        obj = _orig_fcall(user, system, schema, **kw)
        if schema and "intents" in (schema.get("properties") or {}):
            R.extract = dict(R.extract or {}, raw=obj, prompt_chars=len(user) + len(system))
        return obj
    FE._call = _fcall

    def _ei(query, vocab, **kw):
        st = kw.get("stats")
        if st is None:
            st = {}
            kw["stats"] = st
        t0 = time.time()
        its = _orig_ei(query, vocab, **kw)
        R.extract = dict(R.extract or {}, wall_ms=round((time.time() - t0) * 1000, 1),
                         stats={k: v for k, v in st.items() if k != "reconcile"},
                         reconcile=st.get("reconcile"),
                         intents=[f.to_dict() for f in (its.intents if its else [])],
                         relation=(its.relation if its else None))
        return its
    FE.extract_intents = _ei

    import veda.understanding.compound as CP
    _orig_gi = CP.ground_intents

    def _gi(its, vocab, **kw):
        gs = _orig_gi(its, vocab, **kw)
        R.groundings = [{"part": g.part, "kind": g.kind, "source_id": g.source_id,
                         "entity": g.entity, "entity_name": g.entity_name, "method": g.method,
                         "outcome": g.outcome, "message": g.message,
                         "evidence": {k: v for k, v in g.evidence.items()
                                      if k in ("doc", "name_hits", "coverage", "reason", "model_tables")}}
                        for g in gs]
        return gs
    CP.ground_intents = _gi

    import veda.agent.planner as APL
    from veda.agent.tools import compact as _compact
    _orig_planner = APL.run_planner

    def _planner(question, *a, **kw):
        t0 = time.time()
        r = _orig_planner(question, *a, **kw)
        with Rec.lock:
            R.agent.append({"part": R.part, "question": question, "kind": r.kind, "reason": r.reason,
                            "wall_s": round(time.time() - t0, 2), "budget": r.budget,
                            "validation": r.validation, "sql": getattr(r.compiled, "sql", None),
                            "plan": (r.plan.to_dict() if r.plan else None), "message": r.message,
                            "steps": [{k: v for k, v in st.items() if k != "raw"} for st in r.steps],
                            # full (compacted) results: the report re-validates every accepted
                            # plan against exactly this log, offline
                            "tool_calls": [{"id": c["id"], "tool": c["tool"], "args": c["args"],
                                            "auto": c.get("auto"), "ms": c["ms"],
                                            "result": _compact(c.get("result"))}
                                           for c in r.tool_calls]})
        return r
    APL.run_planner = _planner

    import veda_hybrid as VH
    _orig_rp = VH._run_part

    def _rp(fr, g, parent, deadline, verbose=False, on_event=None):
        R.part = len(R.parts_ms)
        t0 = time.time()
        try:
            return _orig_rp(fr, g, parent, deadline, verbose, on_event)
        finally:
            R.parts_ms.append(round((time.time() - t0) * 1000, 1))
            R.part = None
    VH._run_part = _rp
    return VH


def _item_view(it):
    res = it.result
    d = {"part": it.part or it.sub_query, "status": it.status, "outcome": it.outcome,
         "lane": it.lane, "source_id": it.source_id, "route": it.route,
         "refuse_reason": it.refuse_reason, "elapsed_ms": it.elapsed_ms}
    if isinstance(res, dict):
        fb = res.get("feedback") if isinstance(res.get("feedback"), dict) else {}
        secs = ((res.get("trace") or {}).get("sections") or {}) if isinstance(res.get("trace"), dict) else {}
        d.update(head=((res.get("ir") or {}).get("head") if isinstance(res.get("ir"), dict) else None),
                 firewall=(secs.get("firewall") or {}).get("verdict"),
                 frame_reason=(secs.get("frame_path") or {}).get("reason"),
                 agent_trigger=(secs.get("agent") or {}).get("trigger"))
        d.update(answer=res.get("answer") or fb.get("text") or res.get("msg"),
                 pipeline_status=res.get("status"), sql=res.get("sql"), cols=res.get("cols"),
                 rows=(res.get("rows") or [])[:25], n_rows=len(res.get("rows") or []),
                 table=res.get("table"), citations=res.get("citations"))
    elif res is not None:
        d.update(answer=getattr(res, "answer", None), citations=getattr(res, "citations", None),
                 no_answer=getattr(res, "no_answer", None), error=getattr(res, "error", None),
                 chunks=[(getattr(c, "doc_name", None), getattr(c, "page_num", None))
                         for c in (getattr(res, "chunks", None) or [])][:6],
                 cols=getattr(res, "cols", None), rows=(getattr(res, "rows", None) or [])[:25])
    return d


def _run_one(VH, message, ids):
    from veda_core.context import RequestContext, set_context, set_source_profiles
    set_context(RequestContext(source_id=ids[0], tenant="default", source_ids=tuple(ids), cache_back=False))
    set_source_profiles({str(i): PROFILES[str(i)] for i in ids})
    R.reset()
    t0 = time.time()
    err = None
    try:
        mr = VH.run_hybrid_query(message, verbose=False)
    except Exception as e:
        mr, err = None, f"{type(e).__name__}: {str(e)[:300]}"
    wall = round(time.time() - t0, 1)
    try:
        from veda.explain import current_trace  # noqa: F401  (trace already finalised)
    except Exception:
        pass
    rec = {"wall_s": wall, "error": err,
           "split": bool(getattr(mr, "compound", False)),
           "summary": getattr(mr, "summary", None) if mr else None,
           "items": [_item_view(it) for it in (getattr(mr, "items", None) or [])],
           "extract": R.extract, "groundings": R.groundings, "parts_ms": R.parts_ms, "agent": R.agent,
           "slm_calls": R.slm, "sql": R.sql}
    return rec


def cmd_compound(a):
    VH = _instrument()
    flags = _flags.effective_flags()
    _flags.print_flags_header(flags, title="compound: effective engine flags (budgets lifted by _ENV above)")
    _flags.enforce_expect(flags, a.expect)
    with open(a.out, "w") as out:
        out.write(json.dumps({"_meta": {"flags": flags}}) + "\n")
        out.flush()
        for n, q in _questions(a.questions):
            if a.only and n not in {int(x) for x in a.only.split(",")}:
                continue
            rec = {"n": n, "message": q, **_run_one(VH, q, SCOPE)}
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            ex = rec.get("extract") or {}
            print(f"[{n:02d}] split={rec['split']} intents={len(ex.get('intents') or [])} "
                  f"extract={ex.get('wall_ms')}ms wall={rec['wall_s']}s "
                  f"parts={[i.get('outcome') for i in rec['items']]}", flush=True)


def _part_texts(message):
    from veda.understanding.frame_extractor import segment
    return segment(message)


def _source_of_part(gt_part):
    s = str(gt_part.get("source") or "").lower()
    if "invoices" in s:
        return 4
    if "catalog" in s or "parquet" in s:
        return 5
    if "docs" in s or ".pdf" in s or gt_part.get("type") == "doc":
        return 3
    return 2


def cmd_alone(a):
    VH = _instrument()
    flags = _flags.effective_flags()
    _flags.print_flags_header(flags, title="alone: effective engine flags (budgets lifted by _ENV above)")
    _flags.enforce_expect(flags, a.expect)
    gt = {q["n"]: q for q in json.load(open(a.gt))["questions"]}
    with open(a.out, "w") as out:
        out.write(json.dumps({"_meta": {"flags": flags}}) + "\n")
        out.flush()
        for n, q in _questions(a.questions):
            if a.only and n not in {int(x) for x in a.only.split(",")}:
                continue
            segs = _part_texts(q)
            gparts = gt[n]["parts"]
            if len(segs) != len(gparts):
                print(f"[{n:02d}] segmenter gave {len(segs)} parts, ground truth has {len(gparts)} — "
                      f"skipping alone-run for this message", flush=True)
                out.write(json.dumps({"n": n, "message": q, "segments": segs, "skipped": True}) + "\n")
                continue
            for seg, gp in zip(segs, gparts):
                sid = _source_of_part(gp)
                rec = {"n": n, "part": gp["part"], "text": seg, "pinned": sid,
                       **_run_one(VH, seg, (sid,))}
                out.write(json.dumps(rec, default=str) + "\n")
                out.flush()
                print(f"[{n:02d}{gp['part']}] pinned={sid} wall={rec['wall_s']}s "
                      f"{[(i.get('outcome') or i.get('status')) for i in rec['items']]}", flush=True)


# ── grading aid (host): each part next to its ground truth ─────────────────────────
def cmd_sheet(a):
    gt = {q["n"]: q for q in json.load(open(a.gt))["questions"]}
    comp_lines = [json.loads(ln) for ln in open(a.compound) if ln.strip()]
    comp_meta = next((r["_meta"] for r in comp_lines if "_meta" in r), None)
    comp = {r["n"]: r for r in comp_lines if "n" in r}
    alone = {}
    alone_meta = None
    if a.alone:
        for l in open(a.alone):
            r = json.loads(l)
            if "_meta" in r:
                alone_meta = r["_meta"]
            elif not r.get("skipped"):
                alone[(r["n"], r["part"])] = r
    for label, meta in (("compound", comp_meta), ("alone", alone_meta)):
        if meta and meta.get("flags") is not None:
            _flags.print_flags_header(meta["flags"], title=f"sheet: {label} input's recorded flags")
        elif label == "compound" or a.alone:
            print(f"NOTE: {label} input has no recorded '_meta.flags' (collected before this wiring).")
    for n in sorted(gt):
        r = comp.get(n) or {}
        ex = r.get("extract") or {}
        print(f"\n## {n} — split={r.get('split')} intents={len(ex.get('intents') or [])} "
              f"extract={ex.get('wall_ms')}ms wall={r.get('wall_s')}s")
        for g in r.get("groundings") or []:
            print(f"  ground: {g['kind']}@{g['source_id']} {g.get('entity_name') or g.get('entity')} "
                  f"[{g['method']}/{g['outcome']}] :: {g['part'][:70]}")
        items = r.get("items") or []
        if r.get("split") and len(items) != len(gt[n]["parts"]):
            # the extractor returned a different number of parts: show them all, in order
            print(f"  NOTE: {len(items)} parts in the message vs {len(gt[n]['parts'])} in the ground truth")
            for i, it in enumerate(items):
                sq = [x for x in (r.get("sql") or []) if x.get("part") == i and not x.get("probe")]
                print(f"    PART {i + 1} [{it.get('outcome')}] {it.get('lane')}@{it.get('source_id')} "
                      f"{it.get('part')!r}: {str(it.get('answer'))[:300]}")
                if sq:
                    print(f"      SQL: {sq[-1]['inlined'][:400]}")
        for i, gp in enumerate(gt[n]["parts"]):
            print(f"\n### {n}{gp['part']} — EXPECTED: {str(gp.get('expected_answer'))[:300]}")
            if gp.get("result"):
                rs = gp["result"]
                print(f"    GT rows ({rs.get('row_count')}): {rs.get('columns')} {str(rs.get('rows'))[:300]}")
            it = items[i] if r.get("split") and len(items) == len(gt[n]["parts"]) else None
            if it:
                sq = [s for s in (r.get("sql") or []) if s.get("part") == i and not s.get("probe")]
                print(f"    IN-MESSAGE [{it.get('outcome')}] {it.get('lane')}@{it.get('source_id')}: "
                      f"{str(it.get('answer'))[:400]}")
                if sq:
                    print(f"      SQL: {sq[-1]['inlined'][:500]}")
                if it.get("rows"):
                    print(f"      rows({it.get('n_rows')}): {it.get('cols')} {str(it.get('rows'))[:300]}")
                if it.get("citations"):
                    print(f"      cites: {it.get('citations')}")
            elif not r.get("split"):
                print(f"    IN-MESSAGE: not split — single reply: {str((items[0] if items else {}).get('answer'))[:300]}")
            al = alone.get((n, gp["part"]))
            if al:
                its = al.get("items") or [{}]
                it0 = its[0]
                sq = [s for s in (al.get("sql") or []) if not s.get("probe")]
                print(f"    ALONE(pin {al['pinned']}) [{it0.get('outcome') or it0.get('status')}/"
                      f"{it0.get('pipeline_status')}] {str(it0.get('answer'))[:400]}")
                if sq:
                    print(f"      SQL: {sq[-1]['inlined'][:500]}")
                if it0.get("rows"):
                    print(f"      rows({it0.get('n_rows')}): {it0.get('cols')} {str(it0.get('rows'))[:300]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compound", "alone"):
        c = sub.add_parser(name)
        c.add_argument("--questions", default="/app/questions_v3.txt")
        c.add_argument("--gt", default="/app/reports/raw/qv3_ground_truth.json")
        c.add_argument("--out", required=True)
        c.add_argument("--only", default="")
        _flags.add_expect_arg(c)
    s = sub.add_parser("sheet")
    s.add_argument("--gt", default="reports/raw/qv3_ground_truth.json")
    s.add_argument("--compound", required=True)
    s.add_argument("--alone", default=None)
    a = ap.parse_args()
    {"compound": cmd_compound, "alone": cmd_alone, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
