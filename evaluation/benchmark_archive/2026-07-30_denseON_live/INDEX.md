# Benchmark Archive — 2026-07-30 (dense-fix ON, SLM+Metal LIVE, all accuracy flags)
Frozen copies of every benchmark result from this run. **Do not overwrite** — these are the
"show at the end" reference. Config: `DENSE_ID_REMAP=1` + edge-cat + grain-subj + ER + content-bridges
+ understanding + Tier2, cache OFF, SLM `192.168.1.35:11500`, Metal `192.168.1.39:11435`.

| File | What it is | Headline numbers |
|---|---|---|
| `retrieval_perstage_denseON.json` | Per-stage retrieval, dense FIXED (182 q) | dense R@1 0.55/R@5 0.79 · fused R@1 0.70/R@5 0.90 · **+rerank R@1 0.79/R@5 0.91/MRR 0.90** |
| `retrieval_perstage_denseOFF.json` | Per-stage retrieval, dense DEAD (182 q) | dense R@1 **0.0** (dead) · fused R@1 0.62 · **proves dense was contributing nothing** |
| `semantic_layer.json` | Semantic layer eval (182 q, deterministic) | **entity 0.40** · measure 1.0 · ranking 1.0 · temporal 0.47 · dimension 0.37 |
| `homzhub_31gold_denseON.jsonl` | 31 REAL client Qs, dense-ON | ok 10/31 · refused 21 · error 0 · **GOLD-MATCH 0/31** · full-cov 6/31 |
| `homzhub_31gold_denseOFF.jsonl` | 31 REAL client Qs, dense-OFF | ok 9/31 · refused 20 · error 2 · GOLD-MATCH 0/31 · full-cov 3/31 |
| `component_simple.jsonl` | Simple suite Table/Viz/Summary (50 q) | Table 27 · Summary 46 · answered 42 · refused 7 |
| `component_aggregate.jsonl` | Aggregate suite (32 q) | Table 21 · Summary 16 · refused 4 |
| `component_grouped.jsonl` | Grouped suite (20 q) | Table 16 · Summary 20 · refused 2 |
| `component_ranking.jsonl` | Ranking suite (12 q) | Table 7 · Summary 11 · refused 5 (42%) |
| `component_filter.jsonl` | Filter suite (19 q) | Table 13 · Summary 19 · refused 3 |
| `component_temporal.jsonl` | Temporal suite (18 q) | Table 9 · Summary 18 · refused 8 (44%) |
| `benchmark_182queries.json` | The 182-query labeled benchmark (input) | 7 categories, expected tables/columns |

## Key headline numbers (this run)
- **Retrieval (fixed): R@1 0.79 · R@5 0.91 · MRR 0.90** — solid; NOT the bottleneck.
- **Semantic layer entity: 40%** — biggest upstream gap (registry coverage).
- **Component (151 category q): Table 62% structural** (~45% likely truly-correct); answered 79% / refused 19%.
- **31 real-client gold: 0/31 correct** (both dense OFF and ON) — answer-level accuracy ~0.
- **Dense OFF→ON delta on gold: full-coverage 3→6, error 2→0, but GOLD-MATCH 0→0** (retrieval fix did NOT reach answer-level).

## The one conclusion
Retrieval is fixed and solid (R@1 0.79). Answer-level correctness is still ~0 because the wall
is DOWNSTREAM: (1) join-planner picks spurious paths (society/usercoin/audit junctions),
(2) ratio/% queries not computed, (3) count/agg produce raw-dumps or wrong grain, (4) semantic
entity/measure coverage 40%. See `VEDA_GOLD_EVAL_FINDINGS.md`, `VEDA_COMPONENT_EVAL_FINDINGS.md`,
`VEDA_SEMANTIC_LAYER_FINDINGS.md`, `VEDA_PROGRESS_LOG.md`.
