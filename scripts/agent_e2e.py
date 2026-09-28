"""Planner agent — END-TO-END through the real engine (inference container, in-process).

    cd /app/veda_core && python /app/scripts/agent_e2e.py --source 2 "question" ["question" …]
    cd /app/veda_core && python /app/scripts/agent_e2e.py --chat --source 2 "turn 1" "turn 2" …

Runs veda_hybrid.run_hybrid_query with FRAME_PATH_ENABLED=1 and AGENT_PLANNER_ENABLED=1
(set before the engine imports config), pinned to one source, and prints per question the
status, the answering head, the frame-path reason, the agent trigger/outcome, the firewall
verdict, the executed SQL, the row count and the answer. `--chat` threads the turns as a
conversation: each turn's answer is harvested into the chat frame (chatbot/memory) and the
next turn carries its ConversationContext — the agent follow-up path.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("FRAME_PATH_ENABLED", "1")
os.environ.setdefault("AGENT_PLANNER_ENABLED", "1")
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _flags  # noqa: E402

PROFILES = {"2": {"name": "homzhub", "source_type": "relational"},
            "3": {"name": "docs_contracts", "source_type": "document"},
            "4": {"name": "invoices_csv", "source_type": "datalake"},
            "5": {"name": "catalog_parquet", "source_type": "datalake"}}


def _res(mr):
    it = mr.items[0] if getattr(mr, "items", None) else None
    return (it.result if it is not None else None), it


def main():
    args = sys.argv[1:]
    sids = ["2"]
    chat = False
    expect = []
    if "--chat" in args:
        args.remove("--chat")
        chat = True
    if "--source" in args:
        i = args.index("--source")
        sids = args[i + 1].split(",")
        del args[i:i + 2]
    while "--expect" in args:
        i = args.index("--expect")
        expect.append(args[i + 1])
        del args[i:i + 2]
    from veda_core.context import RequestContext, set_context, set_source_profiles
    from veda_hybrid import run_hybrid_query
    flags = _flags.effective_flags()
    _flags.print_flags_header(flags, title="agent_e2e: effective engine flags")
    _flags.enforce_expect(flags, expect)
    conv = None
    frame = {}
    for n, q in enumerate(args, 1):
        set_context(RequestContext(source_id=int(sids[0]), tenant="default",
                                   source_ids=tuple(int(s) for s in sids), cache_back=False))
        set_source_profiles({s: PROFILES[s] for s in sids})
        t0 = time.time()
        mr = run_hybrid_query(q, verbose=False, conversation_context=conv)
        res, it = _res(mr)
        secs = ((res or {}).get("trace") or {}).get("sections") or {} if isinstance(res, dict) else {}
        ag = secs.get("agent") or {}
        print("=" * 100)
        print(f"[{n}] {q}  ({time.time() - t0:.1f}s)")
        if isinstance(res, dict):
            print(f"   status={res.get('status')} head={(res.get('ir') or {}).get('head')} "
                  f"frame={(secs.get('frame_path') or {}).get('reason')} "
                  f"firewall={(secs.get('firewall') or {}).get('verdict')}")
            print(f"   agent: trigger={ag.get('trigger')} kind={ag.get('kind')} reason={ag.get('reason')} "
                  f"budget={ag.get('budget')}")
            for v in ag.get("validation") or []:
                print(f"   agent validation@{v.get('step')}: {v.get('errors')}")
            print(f"   SQL: {res.get('sql')}")
            print(f"   rows={len(res.get('rows') or [])} cols={res.get('cols')} "
                  f"first={str((res.get('rows') or [])[:3])[:300]}")
            print(f"   ANSWER: {str(res.get('answer') or (res.get('feedback') or {}).get('text') or res.get('msg'))[:500]}")
        else:
            print(f"   route={getattr(it, 'route', None)} result={str(res)[:300]}")
        if chat and isinstance(res, dict):
            from chatbot.memory import frame as MF
            from chatbot.memory.context import ConversationContext
            e = MF.harvest_entry(res, q, n)
            if e:
                frame = {"entity": ((res.get("ir") or {}).get("anchor")), "stack": [e]}
            conv = ConversationContext.from_frame(frame, args[n] if n < len(args) else "").to_payload() \
                if frame else None
            if conv is not None and n < len(args):
                conv["user_message"] = args[n]
                print(f"   → next turn carries agent_plan={bool(conv.get('agent_plan'))}")


if __name__ == "__main__":
    main()
