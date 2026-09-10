# veda_core/query_engine/ — dead code

| File | State |
|------|-------|
| `__init__.py` | Empty package marker. |
| `intent_detector.py` | Rule-based 5-class intent classifier (`DIRECT` / `SYNONYM` / `MULTI_TABLE` / `TEMPORAL` / `AGGREGATE`). **Not wired into the deterministic head.** |

## Why it's dead

`veda/pipeline.py::run_query` hard-codes `intent = "SIMPLE"` (`pipeline.py:345`) and
the comment at `pipeline.py:340` states the exclusion is deliberate: the
classifier's keyword classes overlap the grammar planners in `veda/planning.py`,
and flipping intent to `MULTI_TABLE` / `AGGREGATE` here would re-open the
multi-table planning latency that `SUPERLATIVE_JOIN_ROUTING` gates off.

Query shape is derived entirely from `veda/planning.py`'s grammar classifiers
(`existence_mode` / `aggregate_mode` / `superlative_mode` / `grouped_mode` /
`ratio_mode`), and multi-table planning is reached via `TYPED_MULTITABLE_ROUTE`,
`_er_multi`, or existence — never via an intent class.

Only tests and older callers import `intent_detector`. `contracts/README.md` and
`contracts/HEADS.md` still describe it as the "L4 intent (fast-lane)" step — that
is stale. Safe to delete once the test imports are cleaned up.
