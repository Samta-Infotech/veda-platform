"""WP0 — build ``evaluation/golden_queries.jsonl`` from VerifiedQueryCache + parity fixtures.

Gold column/table labels are SOURCE-SPECIFIC (they name real tables/columns of the
ingested schema), so the golden set can only be produced against a live substrate. This
generator:

  1. Seeds from ``scripts/parity_suite.py:QUERIES`` (curated ladder/status coverage).
  2. Exports every ``VerifiedQueryCache`` row for the (source, tenant) and extracts the
     gold column/table sets by parsing ``verified_sql`` with sqlglot — the same AST idiom
     ``veda_core/veda/validation.py`` uses (there is no separate ``query/ast_validator.py``;
     that path referenced in the plan does not exist). ``columns_json`` is used as a
     fallback when a row's SQL fails to parse.
  3. Heuristically tags ``intent`` (single-table filter / aggregate / multi-table join /
     value-literal / temporal / document-rag) and ``expected_kind`` (sql|rag|hybrid).

Writes one JSON object per line:
    {"query", "gold_columns": ["table.col"...], "gold_tables": [...],
     "intent", "expected_kind"}

Runs in the inference/api container (needs Django + the substrate DB). Target >= 60
queries spanning every class; if the verified cache is thin, add hand-labelled lines to
the output file directly (the eval harness merges by reading whatever is present).

Usage:
    python scripts/build_golden_set.py --source-id 1 --tenant default \
        [--out evaluation/golden_queries.jsonl] [--min 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "veda_core"))
sys.path.insert(0, str(_REPO))

# Temporal SQL markers → "temporal"; aggregate markers → "aggregate".
_TEMPORAL_HINTS = ("date_trunc", "interval", "now()", "current_date", "extract(",
                   "::date", "age(", " day", " month", " year")
_AGG_HINTS = ("count(", "sum(", "avg(", "min(", "max(", "group by")


def _extract_gold(sql: str) -> tuple[list[str], list[str], str]:
    """Return (gold_columns, gold_tables, intent) parsed from a verified SQL string.

    Mirrors validation.py's use of sqlglot exp.Table / exp.Column. Columns are qualified
    to ``table.col`` when the AST carries the table (or when the query is single-table).
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return [], [], "unknown"
    if tree is None:
        return [], [], "unknown"

    # alias -> real table name, so alias-qualified columns resolve to their table.
    alias_to_table: dict[str, str] = {}
    tables: list[str] = []
    for t in tree.find_all(exp.Table):
        if not t.name:
            continue
        tname = t.name.lower()
        tables.append(tname)
        if t.alias:
            alias_to_table[t.alias.lower()] = tname
        alias_to_table[tname] = tname
    tables = list(dict.fromkeys(tables))

    only_table = tables[0] if len(tables) == 1 else None
    cols: list[str] = []
    for c in tree.find_all(exp.Column):
        if not c.name or c.name == "*":
            continue
        cn = c.name.lower()
        tbl = alias_to_table.get((c.table or "").lower()) if c.table else only_table
        if tbl:
            cols.append(f"{tbl}.{cn}")
    cols = list(dict.fromkeys(cols))

    s = sql.lower()
    if len(tables) >= 2:
        intent = "join"
    elif any(h in s for h in _TEMPORAL_HINTS):
        intent = "temporal"
    elif any(h in s for h in _AGG_HINTS):
        intent = "aggregate"
    elif " where " in s and ("'" in sql or '"' in s):
        intent = "value-literal"
    else:
        intent = "single-table-filter"
    return cols, tables, intent


def _seed_from_parity() -> list[dict]:
    """Curated queries from the parity suite (gold left empty — hand-label or let the
    verified cache supply matching entries). They still exercise the harness end-to-end."""
    try:
        sys.path.insert(0, str(_REPO / "scripts"))
        from parity_suite import QUERIES  # type: ignore
    except Exception:
        QUERIES = [
            "how many users are there",
            "how many change requests are there",
            "count annotations",
        ]
    out = []
    for q in QUERIES:
        out.append({
            "query": q, "gold_columns": [], "gold_tables": [],
            "intent": "aggregate" if q.lower().startswith(("how many", "count")) else "single-table-filter",
            "expected_kind": "sql", "_source": "parity_suite",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build golden_queries.jsonl (WP0)")
    ap.add_argument("--source-id", type=int, default=1)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--out", default=str(_REPO / "evaluation" / "golden_queries.jsonl"))
    ap.add_argument("--min", type=int, default=60)
    args = ap.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    import django
    django.setup()
    from apps.substrate.models import VerifiedQueryCache

    rows: list[dict] = _seed_from_parity()
    seen_queries = {r["query"].strip().lower() for r in rows}

    qs = VerifiedQueryCache.objects.filter(source_id=args.source_id, tenant=args.tenant)
    n_cache = 0
    for vqc in qs.iterator():
        q = (vqc.query_text or "").strip()
        if not q or q.lower() in seen_queries:
            continue
        cols, tables, intent = _extract_gold(vqc.verified_sql or "")
        if not cols and vqc.columns_json:
            # Fallback: columns_json entries are already "table.col"-ish.
            cols = [str(c) for c in vqc.columns_json if c]
            tables = list(dict.fromkeys(c.split(".")[-2] for c in cols if "." in c))
        if not cols:
            continue  # no usable gold — skip rather than emit an ungradeable line
        expected_kind = "sql"
        rows.append({
            "query": q, "gold_columns": cols, "gold_tables": tables,
            "intent": intent, "expected_kind": expected_kind, "_source": "verified_cache",
        })
        seen_queries.add(q.lower())
        n_cache += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    graded = sum(1 for r in rows if r["gold_columns"])
    print(f"[build_golden_set] wrote {len(rows)} queries ({graded} with gold, "
          f"{n_cache} from verified cache) -> {out_path}")
    if graded < args.min:
        print(f"[build_golden_set] WARNING: only {graded} graded queries (< --min {args.min}). "
              f"Add hand-labelled lines to {out_path.name} to cover all classes "
              f"(single-table filter, aggregate, join, value-literal, temporal, document/RAG).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
