"""WP6 — fusion-weight tuning harness (measurement, NOT training).

Random search (fixed seed) + local refinement over the six FUSION_WEIGHTS, scored by
recall@10 on the golden set (recall@15 as tiebreak). This exercises the 6-signal WEIGHTED
RRF engine (veda.runtime.get_engine().retrieve) — the path FUSION_WEIGHTS controls — NOT
the reranker path. Prints the best weight dict and per-query wins/losses.

The script NEVER writes config: the human runs it once after a fresh ingestion and pastes
the printed FUSION_WEIGHTS into config.py, then re-runs scripts/retrieval_eval.py to
confirm the tuned numbers.

Determinism: a fixed RNG seed makes the sampled weight sets reproducible, and retrieval is
deterministic, so two runs produce the identical best dict.

Usage:
    python scripts/tune_fusion_weights.py --source-id 1 --tenant default \
        [--samples 500] [--seed 0] [--golden evaluation/golden_queries.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "veda_core"))
sys.path.insert(0, str(_REPO))

_SIGNALS = ["dense", "sparse", "subgraph", "fk", "value", "table_prior"]
_LO, _HI = 0.25, 3.0


def _norm_col(ref: str) -> str:
    parts = [p for p in str(ref).lower().split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")


def _recall(gold: set, ranked: list, k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def _load_golden(path: Path) -> list:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            g = json.loads(line)
            if g.get("gold_columns"):
                rows.append(g)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune FUSION_WEIGHTS (WP6)")
    ap.add_argument("--source-id", type=int, default=1)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine-steps", type=int, default=100)
    ap.add_argument("--golden", default=str(_REPO / "evaluation" / "golden_queries.jsonl"))
    args = ap.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        import django
        django.setup()
    except Exception as e:
        print(f"[warn] django.setup() failed ({e})", file=sys.stderr)

    from context import RequestContext, set_context
    set_context(RequestContext(source_id=args.source_id, tenant=args.tenant))

    import config as _config
    from veda.runtime import get_engine

    golden_path = Path(args.golden)
    golden = _load_golden(golden_path)
    if not golden:
        print(f"[error] no graded golden queries in {golden_path} — run build_golden_set.py "
              f"and hand-label gold_columns first.", file=sys.stderr)
        return 2

    engine = get_engine()

    # Precompute gold sets; retrieval is re-run per weight set (cache OFF) because the
    # weights change the fused ranking.
    cases = [(g["query"], {_norm_col(c) for c in g["gold_columns"]}) for g in golden]

    def evaluate(weights: dict):
        _config.FUSION_WEIGHTS = weights   # merge() re-imports config.FUSION_WEIGHTS
        r10, r15, per_q = [], [], []
        for query, gold in cases:
            try:
                results = engine.retrieve(query, intent="SIMPLE", top_k=15, use_cache=False)
                ranked = [_norm_col(f"{r.table_name}.{r.column_name}") for r in results]
            except Exception:
                ranked = []
            a, b = _recall(gold, ranked, 10), _recall(gold, ranked, 15)
            r10.append(a); r15.append(b); per_q.append((query, a, b))
        n = max(len(r10), 1)
        return sum(r10) / n, sum(r15) / n, per_q

    rng = random.Random(args.seed)

    def sample():
        return {s: round(rng.uniform(_LO, _HI), 4) for s in _SIGNALS}

    identity = {s: 1.0 for s in _SIGNALS}
    base10, base15, base_per_q = evaluate(identity)
    print(f"[baseline identity] recall@10={base10:.4f} recall@15={base15:.4f}")

    best_w, best10, best15 = dict(identity), base10, base15
    # ── random search ──
    for i in range(args.samples):
        w = sample()
        r10, r15, _ = evaluate(w)
        if (r10, r15) > (best10, best15):
            best_w, best10, best15 = w, r10, r15
            print(f"  [rand {i+1}/{args.samples}] new best recall@10={r10:.4f} @15={r15:.4f}  {w}")

    # ── local refinement: coordinate perturbations around the best ──
    for step in range(args.refine_steps):
        s = _SIGNALS[step % len(_SIGNALS)]
        delta = rng.uniform(-0.4, 0.4)
        w = dict(best_w)
        w[s] = round(min(_HI, max(_LO, w[s] + delta)), 4)
        r10, r15, _ = evaluate(w)
        if (r10, r15) > (best10, best15):
            best_w, best10, best15 = w, r10, r15
            print(f"  [refine {step+1}] new best recall@10={r10:.4f} @15={r15:.4f}  {w}")

    _, _, best_per_q = evaluate(best_w)
    wins = sum(1 for (_, a, _), (_, ba, _) in zip(best_per_q, base_per_q) if a > ba)
    losses = sum(1 for (_, a, _), (_, ba, _) in zip(best_per_q, base_per_q) if a < ba)

    print("\n" + "=" * 60)
    print(f"BEST recall@10={best10:.4f} recall@15={best15:.4f}  "
          f"(vs identity @10={base10:.4f}); per-query wins={wins} losses={losses}")
    print("Paste into config.py (the script does NOT write it):")
    print("FUSION_WEIGHTS = " + json.dumps(best_w, indent=4))
    print("=" * 60)
    print("\nPer-query (query | best@10 | identity@10):")
    for (q, a, _), (_, ba, _) in zip(best_per_q, base_per_q):
        flag = "＋" if a > ba else ("－" if a < ba else " ")
        print(f"  {flag} {a:.3f} / {ba:.3f}  {q[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
