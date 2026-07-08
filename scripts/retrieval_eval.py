"""WP0 — retrieval-quality evaluation harness (before/after number for every later WP).

Runs the ONE canonical column-selection path — ``query/retrieval_select.py:select_retrieval``
— over the golden query set and reports gold-column recall@5 / recall@15 / MRR, gold-table
recall@3, mean candidate-set size, and wall-clock per stage. Writes a JSON report to
``evaluation/results/retrieval_<git-sha>.json``.

Runs in the inference container (needs the engine, models, psycopg2, and the Django
substrate for the ambient (source, tenant) context). The golden set is produced by
``scripts/build_golden_set.py`` from ``VerifiedQueryCache`` — run that first.

Determinism: retrieval reads a fixed index and this script introduces no randomness, so
two runs against the same build + ingestion produce byte-identical reports (the WP0
acceptance gate). The output is sorted and the timestamp lives OUTSIDE the graded numbers.

Usage:
    python scripts/retrieval_eval.py --source-id 1 --tenant default \
        [--golden evaluation/golden_queries.jsonl] [--out PATH] [--label baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# select_retrieval and the engine import their siblings as top-level modules
# (``from config import ...``, ``from query... import ...``), so veda_core must be on
# the path exactly as it is inside the inference container.
sys.path.insert(0, str(_REPO / "veda_core"))
sys.path.insert(0, str(_REPO))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO)
        ).decode().strip()
    except Exception:
        return "nogit"


def _norm_col(ref: str) -> str:
    """Normalize any column reference to a comparable ``table.col`` (lowercased).

    Gold labels are ``schema.table.col`` (from parsed verified SQL); retrieved columns
    are ``table_name.col_name``. Collapsing both to the trailing two dotted components
    makes them comparable without needing the schema qualifier the engine doesn't carry.
    """
    parts = [p for p in str(ref).lower().split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return parts[0] if parts else ""


def _norm_table(ref: str) -> str:
    parts = [p for p in str(ref).lower().split(".") if p]
    return parts[-1] if parts else ""


def _retrieved_cols(selected) -> list[str]:
    out = []
    for r in selected.columns:
        t = (getattr(r, "table_name", "") or "").lower()
        c = (getattr(r, "col_name", "") or "").lower()
        if t and c:
            out.append(f"{t}.{c}")
    return out


def _recall_at_k(gold: set[str], ranked: list[str], k: int) -> float | None:
    if not gold:
        return None
    topk = set(ranked[:k])
    return len(gold & topk) / len(gold)


def _mrr(gold: set[str], ranked: list[str]) -> float | None:
    if not gold:
        return None
    for i, c in enumerate(ranked, start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def _setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        import django
        django.setup()
    except Exception as e:  # pragma: no cover - context is set best-effort
        print(f"[warn] django.setup() failed ({e}); continuing without substrate context",
              file=sys.stderr)


def _load_golden(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="VEDA retrieval-quality eval (WP0)")
    ap.add_argument("--source-id", type=int, default=int(os.environ.get("VEDA_EVAL_SOURCE_ID", "1")))
    ap.add_argument("--tenant", default=os.environ.get("VEDA_EVAL_TENANT", "default"))
    ap.add_argument("--golden", default=str(_REPO / "evaluation" / "golden_queries.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    _setup_django()
    from context import RequestContext, set_context  # veda_core/context.py
    set_context(RequestContext(source_id=args.source_id, tenant=args.tenant))

    from query.retrieval_select import select_retrieval

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[error] golden set not found: {golden_path}\n"
              f"        run scripts/build_golden_set.py first.", file=sys.stderr)
        return 2
    golden = _load_golden(golden_path)

    per_query = []
    graded = 0  # queries with gold_columns (recall is defined)
    for g in golden:
        query = g["query"]
        gold_cols = {_norm_col(c) for c in g.get("gold_columns", []) if c}
        gold_tables = {_norm_table(t) for t in g.get("gold_tables", []) if t}
        intent = g.get("intent", "sql")

        t0 = time.perf_counter()
        try:
            selected = select_retrieval(
                query=query, source_ids=[str(args.source_id)], intent=intent, verbose=False,
            )
            err = None
        except Exception as e:  # keep going — a single-query crash shouldn't void the run
            selected = None
            err = str(e)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if selected is None:
            per_query.append({"query": query, "error": err, "elapsed_ms": elapsed_ms})
            continue

        ranked = _retrieved_cols(selected)
        ranked_tables = [(_norm_table(t)) for t in selected.tables]
        has_gold = bool(gold_cols)
        if has_gold:
            graded += 1
        entry = {
            "query": query,
            "intent": intent,
            "expected_kind": g.get("expected_kind"),
            "source": selected.source,
            "short_circuit": selected.short_circuit,
            "n_candidates": len(ranked),
            "recall@5": _recall_at_k(gold_cols, ranked, 5),
            "recall@15": _recall_at_k(gold_cols, ranked, 15),
            "mrr": _mrr(gold_cols, ranked),
            "table_recall@3": (
                len(gold_tables & set(ranked_tables[:3])) / len(gold_tables)
                if gold_tables else None
            ),
            "elapsed_ms": elapsed_ms,
            "stage_stats": selected.stats,
            "graded": has_gold,
        }
        per_query.append(entry)

    graded_rows = [e for e in per_query if e.get("graded")]
    summary = {
        "n_queries": len(golden),
        "n_graded": graded,
        "recall@5": _mean([e["recall@5"] for e in graded_rows]),
        "recall@15": _mean([e["recall@15"] for e in graded_rows]),
        "mrr": _mean([e["mrr"] for e in graded_rows]),
        "table_recall@3": _mean([e["table_recall@3"] for e in graded_rows
                                 if e["table_recall@3"] is not None]),
        "mean_candidate_set_size": _mean([float(e["n_candidates"]) for e in per_query
                                          if "n_candidates" in e]),
        "mean_elapsed_ms": _mean([float(e["elapsed_ms"]) for e in per_query
                                  if "elapsed_ms" in e]),
    }
    # Per-class breakdown so WP acceptance can gate "every query class".
    by_class: dict[str, list[dict]] = {}
    for e in graded_rows:
        by_class.setdefault(e.get("intent", "sql"), []).append(e)
    per_class = {
        cls: {
            "n": len(rows),
            "recall@15": _mean([r["recall@15"] for r in rows]),
            "mrr": _mean([r["mrr"] for r in rows]),
        }
        for cls, rows in sorted(by_class.items())
    }

    sha = _git_sha()
    report = {
        "git_sha": sha,
        "label": args.label,
        "source_id": args.source_id,
        "tenant": args.tenant,
        "golden_set": str(golden_path.relative_to(_REPO)) if golden_path.is_relative_to(_REPO)
                      else str(golden_path),
        "summary": summary,
        "per_class": per_class,
        # graded numbers above are timestamp-free & deterministic; wall-clock stamp is
        # recorded separately so it never affects a diff of the graded report.
        "generated_at_epoch": int(time.time()),
        "per_query": sorted(per_query, key=lambda e: e["query"]),
    }

    out_path = Path(args.out) if args.out else (
        _REPO / "evaluation" / "results" / f"retrieval_{sha}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    s = summary
    print(f"[retrieval_eval] {s['n_graded']}/{s['n_queries']} graded  "
          f"recall@5={s['recall@5']:.4f} recall@15={s['recall@15']:.4f} "
          f"mrr={s['mrr']:.4f} table_recall@3={s['table_recall@3']:.4f}  "
          f"mean|cand|={s['mean_candidate_set_size']:.1f}  "
          f"mean_ms={s['mean_elapsed_ms']:.1f}")
    print(f"[retrieval_eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
