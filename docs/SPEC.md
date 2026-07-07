# SPEC.md — VEDA POC Functional Specification

**System**: VEDA — Natural Language to SQL  
**Phase**: POC Phase 1  
**Scope**: L1–L4 pipeline (L5–L7 planned for next phase)

---

## What VEDA does

VEDA takes a natural language question about a database and returns a parameterised SQL query
ready to execute against that database. It requires no schema knowledge from the user —
they write English, VEDA writes SQL.

**In scope (this POC)**:
- SELECT queries with WHERE filters
- COUNT / aggregate queries (GROUP BY, ORDER BY, AVG, SUM)
- Multi-table queries with JOIN paths
- Temporal queries (date ranges, relative periods)
- Synonym resolution (business terms mapped to DB column names)

**Out of scope**:
- Data modification (INSERT, UPDATE, DELETE) — permanently excluded
- DDL (CREATE, DROP, ALTER) — permanently excluded
- Subqueries (supported in IR schema, not yet fully tested)
- Nested aggregations
- Window functions
- CTEs
- Cross-database queries

---

## Layer-by-layer input/output contracts

### L1 — Temporal Parser

| | Detail |
|--|--------|
| **Input** | Raw NL query string |
| **Output** | `TemporalResult(detected: bool, date_range: Optional[Tuple[str,str]], modified_query: str)` |
| **Transforms** | `"incidents last 30 days"` → `date_range=("2026-04-22", "2026-05-22")` |
| **No-op when** | Query contains no recognisable date expression |
| **Never** | Modifies the query in a way that changes non-temporal meaning |

### L2 — Semantic Layer

| | Detail |
|--|--------|
| **Input** | NL query string (post-L1) |
| **Output** | `SemanticLayerResult(top_k_columns: List[ColumnMeta], join_edges: List[JoinEdge], query_vector, stats)` |
| **Retrieves** | Top 20 column UUIDs by cosine similarity from pgvector |
| **Augments with** | FK bridge PKs, domain synonyms, single-table PK injection |
| **Never** | Returns raw column names in SQL form; returns UUID metadata only |

### L3 — SLM

| | Detail |
|--|--------|
| **Input** | NL query + top-10 column UUID metadata (JSON grounding context) |
| **Output** | `SLMResult(intent, ir_json, confidence, complexity, warnings, error)` |
| **Produces** | IR JSON v1 — UUID-only references, no raw column/table names |
| **Intent values** | `SELECT`, `COUNT`, `AGGREGATE` |
| **Complexity values** | `SIMPLE`, `MODERATE`, `COMPLEX` |
| **Confidence** | 0.0–1.0 self-reported by model |
| **Never** | Calls any external API; returns raw SQL; returns column names in IR |

### L4 — SQL Builder

| | Detail |
|--|--------|
| **Input** | `ir_json: dict`, `top_k_columns: List[ColumnMeta]` |
| **Output** | `SQLBuilderResult(sql, params, query_type, tables_used, warnings, error, duration_ms)` |
| **Produces** | Parameterised SQL with `%s` placeholders, all identifiers double-quoted |
| **Query types** | `SELECT`, `COUNT`, `AGGREGATE` |
| **Never** | Interpolates values into SQL strings; calls any model; raises on missing UUIDs |

---

## Supported query types

### DIRECT
Verbatim column name in query matches DB column name.  
Example: `"show me all workflow states"` → `workflow.state` column

### SYNONYM
Business term requires mapping to DB column name.  
Example: `"who raised the alert"` → `incident.raised_by_id` column  
Handled by: domain synonym map in `config.py` + MiniLM semantic similarity

### MULTI_TABLE
Query requires columns from 2+ tables joined via FK path.  
Example: `"show incidents with assigned user email"` → JOIN `incident` + `user`  
Handled by: FK bridge injection in L2 + IR JSON `joins` array in L3

### TEMPORAL
Query contains a time expression.  
Example: `"incidents created in the last 7 days"` → `WHERE created_datetime >= '2026-05-15'`  
Handled by: L1 temporal parser (dateparser) + filter injection in L2/L3

### AGGREGATE
Query requires COUNT, SUM, AVG, GROUP BY.  
Example: `"how many incidents per workflow state"` → `SELECT workflow_state, COUNT(*) GROUP BY workflow_state`  
Handled by: L3 intent=COUNT/AGGREGATE + aggregations/group_by in IR JSON → L4

---

## Success criteria

### Phase 1 (this POC) — target thresholds

| Metric | Target | POC Run 4 |
|--------|--------|-----------|
| Recall@20 (primary) | ≥ 0.85 | **0.929 ✓** |
| Hit Rate@20 | ≥ 0.70 | **0.767 ✓** |
| L3 IR success rate | ≥ 0.80 | **1.00 ✓** |
| L4 SQL success rate | ≥ 0.90 | **1.00 ✓** |
| L2 latency (p50) | ≤ 200ms | **99ms ✓** |

### Phase 2 targets (when L5–L7 are built)

| Metric | Target |
|--------|--------|
| End-to-end query latency (p50) | ≤ 60s |
| SQL execution success rate | ≥ 0.90 |
| Read-only enforcement | 100% (AST-verified) |
| Audit log coverage | 100% of executed queries |

---

## Security requirements

These are hard constraints, not configurable:

1. **No external API calls** from any pipeline component
2. **No client data (schema names, column values, query content) leaving the server**
3. **Parameterised SQL only** — bound values via `%s`, never string-interpolated
4. **Read-only database access** — pipeline must never execute write operations
5. **UUID-only IR JSON** — column/table names never appear in intermediate representations

---

## Data flow — what travels between layers

```
NL query (string)
  ──[L1]──► temporal context (Optional[DateRange])
                 +
            modified query (string)
  ──[L2]──► top_k_columns (List[ColumnMeta])   ← each has: uuid, table_name, col_name, semantic_type
             join_edges    (List[JoinEdge])      ← each has: from_uuid, to_uuid, from_table, to_table
  ──[L3]──► ir_json (dict)                      ← UUID-only, v1 schema (see ARCHITECTURE.md)
             intent (str)
             confidence (float)
  ──[L4]──► sql (str)                           ← parameterised, double-quoted identifiers
             params (list)                       ← bound values in order of %s placeholders
             query_type (str)
             warnings (List[str])
  ──[L5]──► validated_sql (str)                 [planned]
  ──[L6]──► result_rows (List[dict])            [planned]
  ──[L7]──► response (sql, rows, confidence)    [planned] + audit log entry
```

---

## Evaluation test suite

30 queries across 5 types and 3 difficulty levels (file: `evaluation/test_queries.py`):

| Type | Count | Difficulty spread |
|------|-------|------------------|
| DIRECT | 7 | Easy–Medium |
| SYNONYM | 8 | Medium–Hard |
| MULTI_TABLE | 6 | Medium–Hard |
| TEMPORAL | 4 | Medium |
| AGGREGATE | 5 | Medium–Hard |

Each query has ground-truth `expected_columns` as `(table_name, column_name)` pairs.
A query is a "hit" if all expected columns appear in the top-K retrieval result.

---

## Known gaps (as of POC Run 4)

| Gap | Impact | Workaround |
|-----|--------|-----------|
| MULTI_TABLE hit rate 33% | JOIN result columns sometimes missed | Extend FK bridge injection depth |
| L1 temporal detection 50% for TEMPORAL queries | "recently" not parsed | Add relative-term dictionary to L1 |
| L3 avg latency 52s | Too slow for interactive use | Qwen 4-bit quantisation / GPU inference |
| RELGT uses Xavier init, not RelBench weights | Structural embeddings are random | Load actual pre-trained weights |
| pgvector running as `in_memory_fallback` | No ivfflat index, slower at scale | Create `USING ivfflat` index |
| L5–L7 not built | Pipeline produces SQL but cannot execute it | Build L5–L7 (next phase) |
