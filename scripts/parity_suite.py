"""Phase 7.1 — flow-parity: legacy substrate vs migrated substrate.

`veda_core` is the verbatim engine, so the migration's parity question is precisely:
does the SAME engine produce IDENTICAL MultiResults when it reads the LEGACY substrate
(on-disk `sm` + the engine's own FK store) versus the MIGRATED substrate (the
Django-assembled `sm` served via Redis + FK via storage_adapters)? Both read the same
underlying launchpad data, so any status/route/SQL difference is a migration bug.

For each query we run the front door twice (legacy, then migrated), resetting the
engine's cached `sm` between modes, and diff:
  - terminal status            (must match)
  - route / modality           (must match)
  - parameterized SQL text     (must match)
  - result rows                (ORDER-BY-insensitive multiset; for unordered LIMIT,
                                the plan §7.1 says assert row COUNT + SQL only)

Runs in the inference container (engine + models + redis + psycopg2). SQL generation is
deterministic (temperature=0, seed=0), so a matching anchor/sm yields identical SQL.
"""
import json
import os
import sys

# Representative set spanning ladder rungs AND terminal statuses: answered (direct count,
# aggregate) + a DETERMINISTIC ungrounded refusal ("mood is turquoise" → 0 rows) so parity
# covers the firewall/refuse path (§17). NOTE: pure-nonsense queries near real table names
# (e.g. "unicorns in the stable" → fuzzy-matches "state") are NON-DETERMINISTIC in the engine
# itself — the same query anchors to different wrong tables run-to-run at the adaptive-cutoff
# threshold — so they are unsuitable as a parity fixture (parity needs a deterministic
# reference, same principle as §7.1's unordered-LIMIT carve-out). Excluded deliberately.
QUERIES = [
    "how many users are there",
    "how many change requests are there",
    "count annotations",
    "list customers whose mood is turquoise",        # ungrounded value → deterministic 0-row refuse
]


def _reset_engine_sm():
    import veda_hybrid
    veda_hybrid._SM["sm"] = None
    veda_hybrid._SM["cols"] = None


def _clear_caches():
    """Clear BOTH verified caches so legacy (file) and migrated (Django) start identical —
    otherwise a stale entry in either makes a query resolve via cache in one mode only,
    producing a spurious 'rung (cached) != ...' diff. Cache behaviour is tested separately."""
    import os
    # file cache (legacy path)
    try:
        from veda.cache import VERIFIED_FILE
        if os.path.exists(VERIFIED_FILE):
            os.remove(VERIFIED_FILE)
    except Exception:
        pass
    # Django pgvector cache (migrated path)
    try:
        import psycopg2
        c = psycopg2.connect(host=os.environ.get("PGBOUNCER_HOST", "pgbouncer"),
                             port=int(os.environ.get("PGBOUNCER_PORT", "6432")),
                             dbname="veda", user=os.environ.get("POSTGRES_USER", "veda"),
                             password=os.environ.get("POSTGRES_PASSWORD", "change-me"))
        c.autocommit = True
        c.cursor().execute("DELETE FROM substrate_verifiedquerycache")
        c.close()
    except Exception:
        pass


def _extract(mr):
    """Normalize a MultiResult into a comparable dict: status/route/sql/rows PLUS the
    terminal refuse status and the escalation rung (§17 escalation + firewall parity)."""
    items = []
    for it in getattr(mr, "items", []):
        res = getattr(it, "result", None) or {}
        sql = res.get("sql") if isinstance(res, dict) else None
        rows = res.get("rows") if isinstance(res, dict) else None
        row_ms = sorted(json.dumps(r, default=str) for r in rows) if rows else None
        # Terminal engine status (answered/no_table/ungrounded/refuse/...) + which ladder
        # rung produced it — both must match across substrate modes.
        inner_status = res.get("status") if isinstance(res, dict) else None
        rung = None
        if isinstance(res, dict):
            rung = res.get("table") or (res.get("trace", {}) or {}).get("sections", {}).get(
                "schema_linking", {}).get("selected_table")
        items.append({
            "status": getattr(it, "status", None),
            "terminal_status": inner_status,
            "route": getattr(it, "route", None),
            "rung": rung,
            "refuse_reason": getattr(it, "refuse_reason", None),
            "sql": sql,
            "row_count": len(rows) if rows else 0,
            "rows_multiset": row_ms,
        })
    return items


def _run(query):
    from veda_hybrid import run_hybrid_query
    return _extract(run_hybrid_query(query))


def _is_unordered_limit(sql):
    if not sql:
        return False
    s = sql.upper()
    return "LIMIT" in s and "ORDER BY" not in s


def _diff(legacy, migrated):
    """Return list of mismatch strings ([] means parity)."""
    problems = []
    if len(legacy) != len(migrated):
        return [f"item count {len(legacy)} != {len(migrated)}"]
    for i, (a, b) in enumerate(zip(legacy, migrated)):
        if a["status"] != b["status"]:
            problems.append(f"item{i} status {a['status']} != {b['status']}")
        if a["terminal_status"] != b["terminal_status"]:
            problems.append(f"item{i} terminal_status {a['terminal_status']} != {b['terminal_status']}")
        if a["route"] != b["route"]:
            problems.append(f"item{i} route {a['route']} != {b['route']}")
        if a["rung"] != b["rung"]:
            problems.append(f"item{i} rung {a['rung']} != {b['rung']}")
        if (a["sql"] or "") != (b["sql"] or ""):
            problems.append(f"item{i} SQL differs:\n    L={a['sql']}\n    M={b['sql']}")
        # rows: unordered-LIMIT → count+SQL only (§7.1); else multiset must match
        if _is_unordered_limit(a["sql"]):
            if a["row_count"] != b["row_count"]:
                problems.append(f"item{i} unordered-LIMIT row_count {a['row_count']} != {b['row_count']}")
        else:
            if a["rows_multiset"] != b["rows_multiset"]:
                problems.append(f"item{i} row multiset differs")
    return problems


def main():
    from veda_core.context import RequestContext, set_context, _ctx

    baseline = {}
    results = {}
    all_ok = True

    for q in QUERIES:
        print(f"\n=== query: {q!r} ===")
        _clear_caches()   # both modes start with an empty verified cache (fair diff)

        # LEGACY: on-disk sm, engine FK store (no request context set).
        os.environ["VEDA_SM_REDIS"] = "0"
        _ctx.set(None)               # ensure FK shim uses engine store (context unset)
        _reset_engine_sm()
        legacy = _run(q)
        print(f"  legacy : status={legacy[0]['status']} route={legacy[0]['route']} rows={legacy[0]['row_count']}")

        # MIGRATED: Django-assembled sm via redis, FK via storage_adapters (context set).
        os.environ["VEDA_SM_REDIS"] = "1"
        set_context(RequestContext(source_id=1, tenant="default"))
        _reset_engine_sm()
        migrated = _run(q)
        print(f"  migrated: status={migrated[0]['status']} route={migrated[0]['route']} rows={migrated[0]['row_count']}")

        problems = _diff(legacy, migrated)
        baseline[q] = legacy
        results[q] = {"migrated": migrated, "parity": not problems, "problems": problems}
        if problems:
            all_ok = False
            print("  ✗ PARITY MISMATCH:")
            for p in problems:
                print(f"     - {p}")
        else:
            print("  ✓ parity: status/route/SQL/rows identical")

    # Commit the golden baseline (legacy outputs) for future-phase diffs.
    out = "/app/veda_core/data/parity_baseline.json"
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"\n[baseline] committed golden baseline → {out}")
    print(f"\n{'✓ ALL PARITY CHECKS PASSED' if all_ok else '✗ PARITY FAILURES — see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
