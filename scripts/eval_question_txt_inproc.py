"""question.txt — IN-PROCESS variant of scripts/eval_question_txt.py (2026-09-24).

Same 20 questions, same ground truth, same grader — but the engine is called directly
(`veda_hybrid.run_hybrid_query`) instead of through the HTTP API, so the api tier's
worker / proxy time limits (INFERENCE_TIMEOUT_S, GUNICORN_TIMEOUT) cannot turn a slow
local-SLM answer into a 500. It measures the ENGINE; it does not exercise the chat /
api layer (conversation graph, SSE, persistence).

Two halves, because the source DB (`homzhub`, host postgres :5432) is reachable from the
host and the engine from the container:

  collect (inside the inference container):
      cd /app/veda_core && FRAME_PATH_ENABLED=1 python /app/scripts/eval_question_txt_inproc.py \\
          collect --source-ids 2 --out /app/reports/raw/qtxt_inproc_on_pinned.jsonl
  grade (on the host, .venv python):
      .venv/bin/python scripts/eval_question_txt_inproc.py grade \\
          --in reports/raw/qtxt_inproc_on_pinned.jsonl --label on_pinned

Each collected line is shaped like the API response the grader reads:
  {"data": {"summary", "metadata": {"explainability": <the item's explain payload>}}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _flags  # noqa: E402


def collect(a):
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/veda_core")
    from veda_core.context import RequestContext, set_context
    from veda_hybrid import run_hybrid_query
    flags = _flags.effective_flags()
    _flags.print_flags_header(flags, title="collect: effective engine flags")
    _flags.enforce_expect(flags, a.expect)
    # capture the statement that actually executed (+ its bound params): the explain payload
    # carries only the parameterised text, which the grader cannot re-run
    import veda.execution as _vx
    import veda.pipeline as _vp
    _executed = []
    _orig = _vx.execute_sql

    def _cap(sql, params=None, *aa, **kw):
        s = str(sql or "")
        if "FILTER (WHERE" not in s and not s.startswith("SELECT DISTINCT CAST("):
            _executed.append((s, list(params or [])))
        return _orig(sql, params, *aa, **kw)
    _vx.execute_sql = _cap
    if hasattr(_vp, "execute_sql"):
        _vp.execute_sql = _cap
    qs = [l.strip() for l in open(a.questions) if l.strip() and l.strip().lower() != "question"]
    ids = tuple(int(x) for x in a.source_ids.split(",") if x.strip())
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
    with open(a.out, "w") as out:
        out.write(json.dumps({"_meta": {"flags": flags}}) + "\n")
        out.flush()
        for n, q in enumerate(qs, start=1):
            if only and n not in only:
                continue
            set_context(RequestContext(source_id=ids[0], tenant="default", source_ids=ids,
                                       cache_back=False))
            t0 = time.time()
            _executed.clear()
            rec = {"n": n, "question": q}
            try:
                mr = run_hybrid_query(q, verbose=False)
                it = mr.items[0] if getattr(mr, "items", None) else None
                res = (it.result if it is not None else None) or {}
                summary = (getattr(mr, "summary", None) or res.get("answer") or res.get("msg")
                           or ((res.get("feedback") or {}).get("text") if isinstance(res.get("feedback"), dict) else None)
                           or "")
                rec["resp"] = {"data": {"summary": summary,
                                        "metadata": {"explainability": res.get("explain") or {}}}}
                rec["status"] = res.get("status") or (it.status if it else None)
                if _executed and rec["status"] in ("answered", "ok"):
                    rec["executed_sql"], rec["executed_params"] = _executed[-1]
                rec["route"] = getattr(it, "route", None)
                fp = ((res.get("trace") or {}).get("sections") or {}).get("frame_path") \
                    if isinstance(res.get("trace"), dict) else None
                if fp:
                    ex = fp.get("extract") or {}
                    pr = fp.get("probes") or {}
                    rv = fp.get("revision") or {}
                    gr = fp.get("grounding") or {}
                    rec["frame_path"] = {
                        "kind": fp.get("kind"), "reason": fp.get("reason"),
                        "producers": ex.get("producers"), "slm_ms": ex.get("slm_ms"),
                        "provenance": ((fp.get("frame") or {}).get("provenance") or {}),
                        "probes_ran": pr.get("ran"), "probe_problems": pr.get("problems"),
                        "revision_ran": bool(rv), "revision_result": rv.get("result"),
                        "anchor": gr.get("anchor"), "anchor_method": gr.get("anchor_method"),
                        "evidence": gr.get("evidence"), "clarify_slot": gr.get("clarify"),
                    }
                else:
                    rec["frame_path"] = None
            except Exception as e:
                rec["resp"] = {"_error": f"{type(e).__name__}: {str(e)[:300]}"}
            rec["resp"]["_elapsed_s"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            print(f"[{n:02d}] {rec.get('status')} {rec['resp']['_elapsed_s']}s", flush=True)


def grade(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_question_txt as E
    all_recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    meta = next((r["_meta"] for r in all_recs if "_meta" in r), None)
    recs = [r for r in all_recs if "n" in r]
    flags = (meta or {}).get("flags")
    if flags is not None:
        _flags.print_flags_header(flags, title=f"grade: flags recorded at collect time ({a.inp})")
        _flags.enforce_expect(flags, a.expect)
    else:
        print(f"NOTE: {a.inp} has no recorded '_meta.flags' (collected before this wiring) "
              "— cannot verify or --expect against collect-time flags.")
        if a.expect:
            sys.exit(f"--expect given but {a.inp} carries no flags to check it against")
    by_n = {r["n"]: r for r in recs}
    rows, counts = [], {}
    specs = E.SPECS
    if a.specs:
        # held-out specs: order is [[acceptable cols], dir]; same keys otherwise
        specs = json.load(open(a.specs))
        for sp in specs:
            if sp.get("order") and isinstance(sp["order"], list) and len(sp["order"]) == 2:
                sp["order"] = (sp["order"][0], sp["order"][1])
    for spec in specs:
        r = by_n.get(spec["n"])
        if r is None:
            continue
        resp = r["resp"]
        if r.get("executed_sql") and "%s" in r["executed_sql"]:
            # inline the bound values so the grader can re-run the statement that executed
            try:
                cur = E._conn().cursor()
                inlined = cur.mogrify(r["executed_sql"], r.get("executed_params") or []).decode()
                resp = json.loads(json.dumps(resp))
                resp["data"]["metadata"]["explainability"].setdefault("sql", {})["query"] = inlined
            except Exception as e:
                print(f"  (could not inline params for {spec['n']}: {e})")
        verdict, detail, sql, n_ret, gt_n = E.grade(spec, resp)
        counts[verdict] = counts.get(verdict, 0) + 1
        exp = ((r["resp"].get("data") or {}).get("metadata") or {}).get("explainability") or {}
        rows.append(dict(n=spec["n"], question=r["question"], verdict=verdict, detail=detail, sql=sql,
                         returned_rows=n_ret, ground_truth_rows=gt_n, status=r.get("status"),
                         frame_path=r.get("frame_path"), confidence=exp.get("confidence"),
                         summary=((r["resp"].get("data") or {}).get("summary") or "")[:300],
                         elapsed_s=r["resp"].get("_elapsed_s")))
        print(f"[{spec['n']:02d}] {verdict:<16} {detail[:110]}")
    correct, wrong = counts.get("correct", 0), counts.get("wrong_answer", 0)
    generic = counts.get("generic_refusal", 0)
    print("\n" + "=" * 78)
    print(f"label={a.label}  correct={correct}/{len(rows)}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "correct"))
    print(f"TARGETS  >=14 correct: {'OK' if correct >= 14 else 'MISS'} ({correct})   "
          f"0 wrong_answer: {'OK' if wrong == 0 else 'MISS'} ({wrong})   "
          f"0 generic_refusal: {'OK' if generic == 0 else 'MISS'} ({generic})")
    el = sorted(r["elapsed_s"] or 0 for r in rows)
    if el:
        print(f"latency  median={el[len(el) // 2]}s  max={el[-1]}s")
    if a.out:
        json.dump({"label": a.label, "flags": flags, "counts": counts, "correct": correct,
                   "total": len(rows), "results": rows}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--source-ids", default="2")
    c.add_argument("--questions", default="/app/question.txt")
    c.add_argument("--out", required=True)
    c.add_argument("--only", default="")
    _flags.add_expect_arg(c)
    g = sp.add_parser("grade")
    g.add_argument("--in", dest="inp", required=True)
    g.add_argument("--label", default="")
    g.add_argument("--out", default=None)
    g.add_argument("--specs", default=None, help="JSON list of specs (held-out set); default: eval_question_txt.SPECS")
    _flags.add_expect_arg(g)
    a = ap.parse_args()
    (collect if a.cmd == "collect" else grade)(a)


if __name__ == "__main__":
    main()
