# L1 — TEMPORAL PARSE (contract)

> **Role:** extract and normalise temporal expressions from the raw NL query so
> later stages can apply a grounded date window.

Module: `query/temporal_parser.py :: run_temporal_parser`.

## Consumes

| Input | Shape |
|---|---|
| `query` | raw NL string. |

## Produces

`TemporalParserResult`:

| Field | Shape / meaning |
|---|---|
| `temporal_filter` | `{start, end}` as ISO-8601 strings, or `None` when no date range. |
| `cleaned_query` | query with temporal tokens stripped (fed to L2 retrieval). |
| `raw_expressions` | matched temporal fragments (debug). |

## Guarantees / invariants

- Deterministic, no model calls.
- Recognises: `Q1–Q4 YYYY`, `last N days/weeks/months/years`,
  `last/past/previous week/month/quarter/year`, `this/current …`, `yesterday`,
  `today`, `since/after <date>`, `before <date>`, `between … and …`,
  `from … to …`, `in/during/for YYYY`, `Month YYYY`, `recently/latest → last 30
  days`, `N days/hours ago`, `last N hours/minutes`.
- No temporal expression → `temporal_filter = None`; the pipeline runs without a
  date predicate.

## Failure semantics

Non-blocking: an unparseable date simply yields `None`. Never refuses the query.

## Downstream consumers

- `temporal_filter` → L2 (RAG doc-date restriction), L4 fast path, L5 grounded
  `BETWEEN/>=/<=` predicate on the anchor's canonical `TEMPORAL` column
  (`_temporal_predicate` in `pipeline.py`; literals parameterised at L6).
- Temporal-windowed results are **not** cached in the verified-query cache.
