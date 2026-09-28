"""Planner agent — tool smoke + single-question runs, IN-PROCESS (inference container).

    cd /app/veda_core && python /app/scripts/agent_smoke.py tools   [--source 2]
    cd /app/veda_core && python /app/scripts/agent_smoke.py plan  --source 2 "question" ["question" …]

`tools` calls every tool once on a real source and prints (call, latency, sample output)
— the smoke table of the report. `plan` runs the planner loop on each question pinned to
one source and prints the step log, the plan, the validation result and the compiled SQL
(the SQL is NOT executed). Nothing here changes the engine.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

PROFILES = {"2": {"name": "homzhub", "source_type": "relational"},
            "3": {"name": "docs_contracts", "source_type": "document"},
            "4": {"name": "invoices_csv", "source_type": "datalake"},
            "5": {"name": "catalog_parquet", "source_type": "datalake"}}


def _scope(sid: str):
    from veda_core.context import RequestContext, set_context, set_source_profiles
    set_context(RequestContext(source_id=int(sid), tenant="default", source_ids=(int(sid),), cache_back=False))
    set_source_profiles({sid: PROFILES[sid]})
    import veda_hybrid as VH
    sm, cols = VH._load_semantic_model()
    if PROFILES[sid]["source_type"] == "datalake":
        sm, cols = VH._augment_sm_for_datalake(sm, cols, sid)
    from veda.understanding.vocabulary import scope_vocab
    return sm, scope_vocab(sm)


def cmd_tools(sid: str):
    from veda.agent.tools import ToolBox, compact
    sm, vocab = _scope(sid)
    q = {"2": "latest ledger entries with the property name",
         "4": "vendors in Pune with their maintenance amounts",
         "5": "amenities with the highest monthly fee"}.get(sid, "list records")
    tb = ToolBox(sm, vocab, q)
    _, fe = tb.call("find_entities", {"phrase": q})
    ents = [e["table"] for e in fe.get("entities") or []]
    a = ents[0] if ents else None
    b = ents[1] if len(ents) > 1 else None
    d = tb.call("describe", {"table": a})[1] if a else {}
    par = (d.get("parents") or [{}])[0].get("table") if d else None
    date = d.get("business_date") if d else None
    life = (d.get("lifecycle") or {}).get("column") if d else None
    dim = life or ((d.get("dimensions") or [None])[0] if d else None)
    tb.call("columns", {"table": a, "kind": "MONETARY", "phrase": "amount"})
    if par or b:
        tb.call("join_path", {"a": a, "b": par or b})
    if dim:
        tb.call("values", {"table": a, "column": dim, "phrase": ""})
    tb.call("similar_questions", {"text": q, "k": 3})
    filt = [{"col": f"{a}.{dim}", "op": "=", "value": None}] if False else []
    tb.call("probe", {"tables": [a], "filters": filt, **({"order": f"{a}.{date}"} if date else {})})
    tb.call("doc_sections", {"query": "annual leave policy", "source_id": "3", "k": 3})
    print(f"| # | call | ms | sample output |\n|---|---|---|---|")
    for e in tb.log:
        out = json.dumps(compact(e["result"]), default=str, ensure_ascii=False)
        args = json.dumps(e["args"], ensure_ascii=False)
        print(f"| {e['id']} | `{e['tool']}({args[1:-1][:70]})` | {e['ms']:.0f} | `{out[:260]}` |")


def cmd_plan(sid: str, questions):
    from veda.agent.planner import run_planner
    sm, vocab = _scope(sid)
    for q in questions:
        t0 = time.time()
        r = run_planner(q, sm, vocab, source_scope=[sid])
        print("=" * 100)
        print(f"Q: {q}\n→ {r.kind} ({r.reason}) in {time.time() - t0:.1f}s  budget={r.budget}")
        for c in r.tool_calls:
            print(f"   [{c['id']}{' auto' if c.get('auto') else ''}] {c['tool']}({json.dumps(c['args'], ensure_ascii=False)[:120]}) "
                  f"{c['ms']:.0f}ms → {json.dumps(c['result'], default=str, ensure_ascii=False)[:220]}")
        for s in r.steps:
            print(f"   step {s['i']}: {s.get('ms')}ms p={s.get('prompt_tokens')} cached={s.get('cached_tokens')} c={s.get('completion_tokens')} "
                  f"{s.get('thought', '')!r} {json.dumps(s.get('action'), ensure_ascii=False)[:260]} {s.get('error', '')}")
        for v in r.validation:
            print(f"   validation@{v['step']}: {v['errors']}")
        for j in r.judge:
            print(f"   JUDGE@{j.get('step')}: ok={j.get('ok')} sim_q={j.get('sim_q')} ce={j.get('ce')} "
                  f"near={[(n.get('table'), n.get('sim')) for n in (j.get('nearest') or [])][:2]} "
                  f"text={j.get('plan_text')!r} dis={j.get('disagreement')}")
        if r.plan is not None:
            print("   PLAN:", json.dumps(r.plan.to_dict(), default=str))
        if r.compiled is not None:
            print(f"   SQL: {r.compiled.sql}")
            try:
                from veda.execution import execute_sql
                cols, rows, err = execute_sql(r.compiled.sql, None, timeout_ms=5000)
                print(f"   ROWS({len(rows or [])}): {cols} {str((rows or [])[:4])[:300]} {err or ''}")
            except Exception as e:
                print(f"   EXEC: {type(e).__name__}: {e}")
        if r.message:
            print(f"   MSG: {r.message}")


def main():
    args = sys.argv[1:]
    sid = "2"
    if "--source" in args:
        i = args.index("--source")
        sid = args[i + 1]
        del args[i:i + 2]
    if args[0] == "tools":
        cmd_tools(sid)
    elif args[0] == "file":
        # file mode: <source>|<question> lines, grouped by source
        rows = [l.split("|", 1) for l in open(args[1]) if l.strip() and not l.startswith("#")]
        for s_ in dict.fromkeys(r[0] for r in rows):
            cmd_plan(s_, [q.strip() for sid_, q in rows if sid_ == s_])
    else:
        cmd_plan(sid, args[1:])


if __name__ == "__main__":
    main()
