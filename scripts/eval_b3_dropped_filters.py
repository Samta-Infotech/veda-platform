#!/usr/bin/env python3
"""scripts/eval_b3_dropped_filters.py — B.3 A/B smoke test.

The 4 questions the 2026-09-27 integration pass found losing their filter under
front-door reuse (reports/VEDA_INTEGRATION_2026-09-27.md §B.1 / §10.3):
    "vendors in Kochi"                    source 4
    "amenities in the Sports category"    source 5
    "how many properties are gated"       source 2
    "users created last month"            source 2

Run inside the inference container, once per FRONT_DOOR_FRAME_REUSE value (an env
override on the ephemeral `docker compose run`, never touching .env or the live
container):

    docker compose run --rm -T -e FRONT_DOOR_FRAME_REUSE=1 --entrypoint sh inference \\
        -c "cd /app && python scripts/eval_b3_dropped_filters.py"
    docker compose run --rm -T -e FRONT_DOOR_FRAME_REUSE=0 --entrypoint sh inference \\
        -c "cd /app && python scripts/eval_b3_dropped_filters.py"

Prints the effective flags header, then one JSON line per question with: status, the
executed SQL, whether it has a WHERE, the RAW frame's filters slot
(trace.frame_path.frame.filters — set before grounding/compilation, so this is the
most direct signal of whether the extractor kept the filter), and frame_source
("front_door" when the compound front door's frame answered instead of a fresh
extraction by the SQL head — confirms reuse was actually exercised, not skipped).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _flags  # noqa: E402

CASES = [
    (4, "vendors in Kochi"),
    (5, "amenities in the Sports category"),
    (2, "how many properties are gated"),
    (2, "users created last month"),
]


def main():
    only = sys.argv[1:]
    cases = [(sid, q) for sid, q in CASES if not only or q in only]
    cfg = _flags._import_config()
    flags = _flags.effective_flags(cfg)
    flags["FRONT_DOOR_FRAME_REUSE"] = getattr(cfg, "FRONT_DOOR_FRAME_REUSE", "<missing>")
    _flags.print_flags_header(flags, title="B.3 dropped-filter smoke")

    from veda_core.context import RequestContext, set_context
    from veda_hybrid import run_hybrid_query

    for sid, q in cases:
        set_context(RequestContext(source_id=sid, tenant="default", cache_back=False))
        row = {"source": sid, "q": q, "reuse": flags["FRONT_DOOR_FRAME_REUSE"]}
        t0 = time.time()
        try:
            r = run_hybrid_query(q, verbose=False)
            it = r.items[0] if getattr(r, "items", None) else None
            res = it.result if it is not None else None
            if isinstance(res, dict):
                sql = res.get("sql") or ""
                row["status"] = res.get("status")
                row["sql"] = sql
                row["has_where"] = " WHERE " in sql.upper() if sql else False
                fp = ((res.get("trace") or {}).get("sections") or {}).get("frame_path") or {}
                frame = fp.get("frame") or {}
                row["frame_source"] = frame.get("source")
                row["frame_filters"] = frame.get("filters")
                row["frame_group_by"] = frame.get("group_by")
            else:
                row["status"] = getattr(res, "status", None) if res is not None else "no_result"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        row["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
