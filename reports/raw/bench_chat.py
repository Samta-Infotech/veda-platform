"""Follow-up-turn benchmark — the chat path (chatbot.run.run_chat_turn), the same
entry point apps/chat/services.py uses per user turn.

Turn 1 opens the topic (pinned, per scripts/eval_sessions.py's documented reason:
this deployment cannot answer an UNPINNED multi-source opening turn from the chat
path). Turn 2 is the measured FOLLOW-UP: a refinement that adds a filter to the
prior answer and carries almost no routing signal on its own.

Run inside an api container with the tree under test at /app.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
import django  # noqa: E402
django.setup()

SCRIPTS = {
    # (script id, category label, pinned opening source, turns)
    "FU1": {"source": 4, "scope": [2, 3, 4, 5], "turns": [
        ("how many maintenance records are there", "opening"),
        ("how many of those are in the Repair category", "followup"),
    ], "expect": "follow-up answers a FILTERED count on source 4 (Repair only, < the turn-1 total), "
                 "staying on the source turn 1 answered from"},
    "FU2": {"source": 2, "scope": [2, 3, 4, 5], "turns": [
        ("how many properties are there", "opening"),
        ("how many of those are in Mumbai", "followup"),
    ], "expect": "follow-up answers a FILTERED count on source 2 (Mumbai only, < the turn-1 total), "
                 "staying on the source turn 1 answered from"},
}


def reset_verified_cache(baseline_ids):
    """Keep the verified-query cache at a fixed baseline between reps. The engine runs
    in the inference process (over HTTP), so the in-process patch the engine-level
    harness uses is not reachable from here; master has no cache_back kill switch at
    all, so the table is levelled instead. Same operation for both trees."""
    try:
        from django.db import connection
        with connection.cursor() as cur:
            if baseline_ids:
                cur.execute("DELETE FROM substrate_verifiedquerycache WHERE id <> ALL(%s)",
                            [list(baseline_ids)])
            else:
                cur.execute("DELETE FROM substrate_verifiedquerycache")
            return cur.rowcount
    except Exception as e:
        return f"FAILED: {e}"


def cache_ids():
    try:
        from django.db import connection
        with connection.cursor() as cur:
            cur.execute("SELECT id FROM substrate_verifiedquerycache")
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    from chatbot.run import run_chat_turn
    try:
        from apps.query.scope import source_profiles_for
        profiles = source_profiles_for([2, 3, 4, 5])
    except Exception as exc:
        print(f"WARNING: source profiles unavailable ({type(exc).__name__}: {exc})")
        profiles = {}
    print(f"profiles: {json.dumps(profiles, default=str)[:300]}", flush=True)

    baseline = cache_ids()
    print(f"verified-cache baseline ids: {baseline}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, "w", buffering=1)
    fh.write(json.dumps({"kind": "meta", "label": args.label, "profiles": profiles,
                         "cache_baseline": baseline,
                         "inference_url": os.environ.get("INFERENCE_URL"),
                         "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                        default=str) + "\n")

    def one_script(sid, spec, rep, timed=True):
        session = f"bench-{args.label}-{sid}-{uuid.uuid4().hex[:8]}"
        for turn_no, (msg, role) in enumerate(spec["turns"], 1):
            events = []

            def _on_event(phase, message, extra):
                events.append({"t_ms": round((time.perf_counter() - t0) * 1000, 1),
                               "phase": str(phase), "message": str(message)[:160],
                               "extra": json.loads(json.dumps(extra or {}, default=str))})

            pin = spec["source"] if turn_no == 1 else None
            t0 = time.perf_counter()
            row = {"kind_row": "timed" if timed else "warmup", "label": args.label,
                   "script": sid, "rep": rep, "turn": turn_no, "role": role,
                   "msg": msg, "session": session, "pinned": pin,
                   "expected_shape": spec["expect"]}
            try:
                res = run_chat_turn(msg, session, tenant="default",
                                    source_id=pin,
                                    source_ids=([pin] if pin else list(spec["scope"])),
                                    source_profiles=profiles, on_event=_on_event)
                row["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                er = res.get("engine_result") or {}
                row.update({
                    "status": res.get("status"),
                    "answer": str(res.get("answer_text") or "")[:400],
                    "sql": str(res.get("sql") or "")[:400],
                    "n_rows": len(res.get("rows") or []) if isinstance(res.get("rows"), list) else None,
                    "needs_clarification": res.get("needs_clarification"),
                    "supervisor_slm_calls": res.get("supervisor_slm_calls"),
                    "supervisor_slm_purposes": res.get("supervisor_slm_purposes"),
                    "engine_usage": er.get("usage"),
                    "engine_source_id": er.get("source_id"),
                    "engine_trace_id": er.get("trace_id") or (er.get("explain") or {}).get("trace_id"),
                    "engine_status": er.get("status"),
                    "engine_keys": sorted(er.keys())[:40],
                })
                row["route_source"] = next((e["extra"].get("route_source") for e in reversed(events)
                                            if e["extra"].get("route_source")), None)
                row["scope_seen"] = next((e["extra"].get("source_ids") for e in reversed(events)
                                          if e["extra"].get("route_source") and e["extra"].get("source_ids")), None)
            except Exception as e:
                row["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                row["status"] = "CRASH"
                row["error"] = f"{type(e).__name__}: {e}"
            row["events"] = events
            row["ttfb_ms"] = events[0]["t_ms"] if events else None
            fh.write(json.dumps(row, default=str) + "\n")
            print(f"[{args.label} rep{rep} {sid} t{turn_no}:{role:8s}] "
                  f"{str(row.get('status')):12s} wall={row.get('wall_ms')}ms "
                  f"sup_slm={row.get('supervisor_slm_calls')} "
                  f"usage={row.get('engine_usage')} route={row.get('route_source')} "
                  f"src={row.get('engine_source_id')} :: {str(row.get('answer'))[:90]}",
                  flush=True)

    for w in range(args.warmup):
        print(f"--- warmup {w+1} (not timed) ---", flush=True)
        reset_verified_cache(baseline)
        one_script("FU1", SCRIPTS["FU1"], 0, timed=False)

    for rep in range(1, args.reps + 1):
        for sid, spec in SCRIPTS.items():
            reset_verified_cache(baseline)
            one_script(sid, spec, rep, timed=True)

    reset_verified_cache(baseline)
    fh.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
