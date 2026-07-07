# VEDA retrieval evaluation (WP0)

Before/after measurement harness for the retrieval-quality upgrade. Every later work
package is graded against `retrieval_BASELINE.json`.

## Files
- `golden_queries.jsonl` — one query per line:
  `{"query", "gold_columns": ["table.col"...], "gold_tables": [...], "intent", "expected_kind": "sql|rag|hybrid"}`.
  The committed file is a **seed** (parity-suite queries, no gold labels). The real
  ≥60-query set is generated against a live substrate — see below.
- `results/` — `retrieval_<git-sha>.json` reports written by the evaluator.

## Workflow (run inside the inference/api container, with a current ingestion)

1. **Generate the golden set** from the verified-query cache:
   ```
   python scripts/build_golden_set.py --source-id 1 --tenant default --min 60
   ```
   This parses each `VerifiedQueryCache.verified_sql` with sqlglot to extract the gold
   column/table sets. If it reports fewer than 60 graded queries, hand-label additional
   lines directly in `golden_queries.jsonl` so every class is covered: single-table
   filter, aggregate, multi-table join, value-literal, temporal, document/RAG.

2. **Run the baseline** on the CURRENT build + ingestion and commit it:
   ```
   python scripts/retrieval_eval.py --source-id 1 --tenant default --label baseline
   cp evaluation/results/retrieval_<sha>.json evaluation/results/retrieval_BASELINE.json
   ```
   The report is deterministic across two runs (graded numbers carry no timestamp).

3. **Grade later WPs**: re-run `retrieval_eval.py` and compare `summary` +
   `per_class` against `retrieval_BASELINE.json`. Gate: recall@15 and MRR ≥ baseline on
   every query class.

## Metrics
`recall@5` / `recall@15` / `MRR` over gold columns, `table_recall@3` over gold tables,
`mean_candidate_set_size`, and per-query `elapsed_ms`. Queries with no `gold_columns`
run (exercising the path) but are excluded from graded aggregates.
