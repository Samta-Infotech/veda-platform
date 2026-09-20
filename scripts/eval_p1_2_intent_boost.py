"""P1-2 (2026-09-10) — before/after eval for retrieval-only intent boosting.

scripts/retrieval_eval.py measures query/retrieval_select.py (the bi-encoder/rerank
"v2" path), which never calls retrieval_engine_phase3.py's RRFMerger/IntentBooster —
so it cannot see this change at all (same gap P1-1 hit; see
docs/backlog/query-engine-open-items.md). This script targets the RIGHT path
directly: RetrievalEnginePhase3.retrieve(), the one veda/pipeline.py::run_query
actually calls.

Compares, over evaluation/golden_queries.jsonl:
  BASELINE — intent="SIMPLE" always (today's veda/pipeline.py:345 behaviour)
  AFTER    — the same grammar-derived intent veda/pipeline.py now computes
             (TEMPORAL / AGGREGATE / SIMPLE), with intent_boosting's deltas scaled
             by config.RETRIEVAL_INTENT_BOOST_SCALE

Same recall@5/recall@15/MRR/table_recall@3 definitions as scripts/retrieval_eval.py,
against the same golden set, so the two reports are comparable.

Usage (run inside the inference container — needs the real engine):
    python scripts/eval_p1_2_intent_boost.py --source-id 2 --tenant default
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "veda_core"))


def _norm_col(ref: str) -> str:
    parts = [p for p in str(ref).lower().split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else str(ref).lower()


def _retrieval_intent(query: str) -> str:
    """Mirrors veda/pipeline.py's _retrieval_intent computation (P1-2) — kept as a
    literal copy rather than an import so this script doesn't drag in the full
    pipeline module (heavy, many side-effecting imports) just to reuse 6 lines."""
    from veda.planning import aggregate_mode, grouped_mode
    from query.temporal_parser import run_temporal_parser

    tf = run_temporal_parser(query).temporal_filter
    if tf and (tf.start or tf.end):
        return "TEMPORAL"
    if aggregate_mode(query) or grouped_mode(query):
        return "AGGREGATE"
    return "SIMPLE"


def _score(results, gold_cols, gold_tables):
    ranked = [_norm_col(f"{r.table_name}.{r.column_name}") for r in results]
    gold = {_norm_col(c) for c in gold_cols}
    hit5 = len(set(ranked[:5]) & gold)
    hit15 = len(set(ranked[:15]) & gold)
    r5 = hit5 / len(gold) if gold else None
    r15 = hit15 / len(gold) if gold else None
    mrr = 0.0
    for i, c in enumerate(ranked):
        if c in gold:
            mrr = 1.0 / (i + 1)
            break
    gt = {t.lower() for t in (gold_tables or [])}
    top_tables = {r.table_name.lower() for r in results[:3]}
    tr3 = 1.0 if (gt & top_tables) else (0.0 if gt else None)
    return r5, r15, mrr, tr3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", default="2")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--golden", default=str(_REPO / "evaluation" / "golden_queries.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cap graded queries (0 = all)")
    ap.add_argument("--no-reranker", action="store_true",
                    help="disable the cross-encoder reranker (fast; also the path P1-1's "
                         "own acceptance criteria says a boost effect shows most clearly)")
    args = ap.parse_args()

    from context import RequestContext, set_context
    import veda.runtime as rt
    from config import RETRIEVAL_INTENT_BOOST_SCALE
    if args.no_reranker:
        import config as _cfg
        _cfg.RERANKER_ENABLED = False

    set_context(RequestContext(source_id=int(args.source_id), tenant=args.tenant, cache_back=False))
    eng = rt.get_engine()

    rows = []
    with open(args.golden) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("gold_columns") or r.get("gold_tables")]
    if args.limit:
        rows = rows[:args.limit]

    def _run(label, intent_fn):
        r5s, r15s, mrrs, tr3s = [], [], [], []
        n_graded = 0
        for row in rows:
            gold_cols = row.get("gold_columns") or []
            gold_tables = row.get("gold_tables") or []
            if not gold_cols and not gold_tables:
                continue
            n_graded += 1
            intent = intent_fn(row["query"])
            results = eng.retrieve(row["query"], intent=intent, top_k=15, use_cache=False)
            r5, r15, mrr, tr3 = _score(results, gold_cols, gold_tables)
            if r5 is not None: r5s.append(r5)
            if r15 is not None: r15s.append(r15)
            mrrs.append(mrr)
            if tr3 is not None: tr3s.append(tr3)
        def _avg(xs): return round(sum(xs) / len(xs), 4) if xs else None
        print(f"[{label}] {n_graded} graded  recall@5={_avg(r5s)}  recall@15={_avg(r15s)}  "
              f"mrr={_avg(mrrs)}  table_recall@3={_avg(tr3s)}")

    print(f"scale={RETRIEVAL_INTENT_BOOST_SCALE}")
    _run("BASELINE (intent=SIMPLE always)", lambda q: "SIMPLE")
    _run("AFTER (grammar-derived intent)", _retrieval_intent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
