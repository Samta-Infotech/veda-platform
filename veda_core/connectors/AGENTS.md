# veda_core/connectors/ — source connectors

`build_connector(source_config)` (`base.py`) dispatches on `type` / `engine` to a connector
that exposes a source's schema (and, for NoSQL, executes native queries at query time).
Which pipeline each source type runs: [../../docs/INGESTION.md](../../docs/INGESTION.md) §4.

| File | Role |
|------|------|
| `base.py` | `BaseConnector` abstract interface + `build_connector()` factory + `ConnectorStatus`, `RawSchema` / `RawTable` / `RawColumn` dataclasses + `DATA_TYPE_MAP`. Capabilities via `supports_*` props. |
| `relational.py` | **45 KB, the workhorse.** `RelationalConnector` (+ PostgreSQL / MySQL / SQLite / Oracle / SQL Server subclasses). Owns `get_real_schema()` — the live `INFORMATION_SCHEMA` introspection used by the relational L1. |
| `document.py` | `FilesystemDocumentConnector` — walks a local dir, yields `DocumentChunk`s for PDF / DOCX / TXT / MD / HTML. Optional deps: pdfplumber, python-docx, bs4. |
| `doc_parser.py` | Layout-aware parsing (Cross-source Phase 3): PDF via pymupdf4llm (markdown + heading hierarchy + tables), DOCX via python-docx. `ParsedDoc{metadata, sections[]}`, `chunk_sections`. **The "tabular lane" for derived doc tables was never built** (`DocTable.is_derived_table()` / `ParsedDoc.derived_tables()` have no caller). |
| `datalake.py` | `DatalakeConnector` — Delta / Parquet / CSV via in-process DuckDB → `RawSchema` (FK edges always empty). Lighter schema pipeline; mints **random** UUIDs. |
| `nosql.py` | `NoSQLConnector` — MongoDB / Elasticsearch / DynamoDB. Schema inference by sampling; `execute_query()` builds per-engine native query dicts at query time. |
| `tabular_files.py` | `TabularFileConnector` — CSV / Excel / Parquet presented through the **relational** interface so the FULL L1–L5 runs over a file. **Deterministic UUIDv5** ids (idempotent). `materialize_parquet()` is the Phase-5 execution surface. Contrast `datalake.py` (light pipeline + random ids). |

## Gotchas
- Two datalake connectors: `datalake.py` (light, random ids) vs `tabular_files.py` (full
  pipeline, deterministic ids). `dispatcher.dispatch` sends tabular engines
  (csv/csv_lake/parquet/xlsx/excel) through the **full** layered pipeline via
  `tabular_files.py`.
- `../schema/real_schema.py` is a one-liner re-export of `relational.get_real_schema`.
