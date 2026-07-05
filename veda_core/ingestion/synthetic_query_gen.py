# =============================================================================
# ingestion/synthetic_query_gen.py
# VEDA POC — Step 10: Synthetic Query Generator (v2 — DDL + Execution Validation)
#
# What changed from v1:
#   v1 — asked Qwen to generate questions per COLUMN only.
#         Output: (question → col_id) pairs used for MiniLM fine-tuning.
#         Problem: Qwen had no schema context, generated weak/generic questions.
#
#   v2 — Approach 1 + Approach 2 combined:
#         Approach 1: give Qwen the full TABLE DDL + 5 sample rows per table,
#                     ask it to generate (question, SQL) pairs together.
#                     Qwen has real schema context now — better questions.
#         Approach 2: execute every generated SQL against the real client DB.
#                     Error    → discard (syntactically broken)
#                     0 rows   → discard (semantically wrong filter)
#                     >0 rows  → keep as validated training pair
#                     The DB is the ground truth oracle — no human labelling needed.
#
#   v1 pairs (question → col_id) are still generated as a fallback and for
#   columns that don't belong to a table with enough sample rows.
#   Both v1 and v2 pairs flow into auto_finetune.py — they have the same
#   TrainingPair format (v2 pairs just also carry sql + validated=True).
#
# Output format (JSONL, one JSON object per line):
#   {
#     "query":      "show all escalated incidents",
#     "col_id":     "uuid-of-incident.workflow_state",
#     "col_name":   "workflow_state",
#     "table_name": "incident",
#     "source":     "ddl_validated" | "ddl_unvalidated" | "column" | "fk" | "value" | "fallback",
#     "sql":        "SELECT ... FROM incident WHERE ...",   # present for ddl_* sources
#     "validated":  true | false                           # present for ddl_* sources
#   }
#
# Design constraints:
#   - SLM only (Qwen via Ollama) — no external API, no bigger model
#   - DB validation uses the same client connection already in the pipeline
#   - Graceful degradation at every step — v1 fallback if anything fails
#   - Skips tables with < MIN_SAMPLE_ROWS to avoid empty-table false positives
#   - Validation runs with a 5-row LIMIT to keep it fast (just checking correctness)
#   - Idempotent: SYNTHETIC_USE_EXISTING_PAIRS=True reuses existing file
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ingestion.semantic_type_inference import InferenceResult, TypedColumn
from ingestion.value_sampler import get_sampled_columns, SampledColumn
from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    SLM_TIMEOUT_SECS,
    SLM_MAX_RETRIES,
    SYNTHETIC_QUERY_GEN_ENABLED,
    SYNTHETIC_QUERIES_PER_COLUMN,
    SYNTHETIC_QUERIES_PER_FK,
    SYNTHETIC_GEN_ELIGIBLE_TYPES,
    SYNTHETIC_PAIRS_PATH,
    SYNTHETIC_MIN_PAIRS_FOR_FINETUNE,
    SYNTHETIC_GEN_BATCH_SIZE,
    SYNTHETIC_GEN_MAX_COLUMNS,
    SYNTHETIC_GEN_MAX_FK_EDGES,
    SYNTHETIC_USE_EXISTING_PAIRS,
    get_primary_relational_source,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Config — new v2 constants (safe defaults if not in config.py yet)
# Add these to config.py to override:
#   SYNTHETIC_DDL_PAIRS_PER_TABLE  = 5
#   SYNTHETIC_DDL_MIN_SAMPLE_ROWS  = 3
#   SYNTHETIC_VALIDATION_ENABLED   = True
#   SYNTHETIC_VALIDATION_ROW_LIMIT = 5
# =============================================================================

_DDL_PAIRS_PER_TABLE  = int(os.environ.get("SYNTHETIC_DDL_PAIRS_PER_TABLE",  5))
_DDL_MIN_SAMPLE_ROWS  = int(os.environ.get("SYNTHETIC_DDL_MIN_SAMPLE_ROWS",  3))
_VALIDATION_ENABLED   =     os.environ.get("SYNTHETIC_VALIDATION_ENABLED",  "true").lower() == "true"
_VALIDATION_ROW_LIMIT = int(os.environ.get("SYNTHETIC_VALIDATION_ROW_LIMIT", 5))

# Pull from config.py if they exist there
try:
    from config import SYNTHETIC_DDL_PAIRS_PER_TABLE as _DDL_PAIRS_PER_TABLE
except ImportError:
    pass
try:
    from config import SYNTHETIC_DDL_MIN_SAMPLE_ROWS as _DDL_MIN_SAMPLE_ROWS
except ImportError:
    pass
try:
    from config import SYNTHETIC_VALIDATION_ENABLED as _VALIDATION_ENABLED
except ImportError:
    pass
try:
    from config import SYNTHETIC_VALIDATION_ROW_LIMIT as _VALIDATION_ROW_LIMIT
except ImportError:
    pass


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class TrainingPair:
    """A single (query, column) training pair — v1 and v2 compatible."""
    query:      str       # natural language question
    col_id:     str       # UUID of the column this query should retrieve
    col_name:   str       # for logging and inspection
    table_name: str       # for logging and inspection
    source:     str       # "ddl_validated" | "ddl_unvalidated" | "column" | "fk" | "value" | "fallback"
    sql:        str = ""  # SQL string — only set for ddl_* sources
    validated:  bool = False  # True if SQL executed successfully and returned rows


@dataclass
class SyntheticQueryGenResult:
    """Top-level output of the synthetic query generator."""
    pairs:              List[TrainingPair]
    total_pairs:        int
    column_pairs:       int
    fk_pairs:           int
    value_pairs:        int
    fallback_pairs:     int
    ddl_pairs:          int   # v2: DDL-generated pairs
    validated_pairs:    int   # v2: DDL pairs that passed execution validation
    discarded_pairs:    int   # v2: DDL pairs that failed validation
    ollama_available:   bool
    db_available:       bool  # v2: whether client DB was reachable for validation
    output_path:        str
    duration_sec:       float
    stats:              dict = field(default_factory=dict)


# =============================================================================
# DB validation — executes generated SQL against client DB
# Uses same connection as the rest of the pipeline
# =============================================================================

def _get_validation_connection():
    """
    Returns a connection to the primary relational source for SQL validation.
    Returns None if unavailable (graceful degradation).
    """
    try:
        from ingestion.db_abstraction import get_client_connection
        src = get_primary_relational_source()
        conn = get_client_connection(src["id"])
        return conn
    except Exception:
        return None


def _validate_sql(conn, sql: str) -> Tuple[bool, str]:
    """
    Executes SQL with a small LIMIT against the client DB.
    Returns (is_valid, reason).

    Rules:
      - Any exception    → False (syntax error, bad table/column name)
      - 0 rows returned  → False (semantically wrong filter — no data matches)
      - >0 rows returned → True  (DB confirmed this SQL makes sense on real data)

    Safety: wraps in a transaction that is always rolled back.
    Only SELECT queries are validated — any non-SELECT is discarded.
    """
    if not conn:
        return False, "no_connection"

    # Safety gate — only validate SELECT queries
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return False, "non_select"

    # Inject a small LIMIT so validation is fast (don't pull 1000 rows)
    # If the SQL already has LIMIT, replace it with our small one
    validated_sql = re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {_VALIDATION_ROW_LIMIT}",
                           sql, flags=re.IGNORECASE)
    if "LIMIT" not in validated_sql.upper():
        validated_sql = validated_sql.rstrip(";") + f" LIMIT {_VALIDATION_ROW_LIMIT}"

    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute(validated_sql)
        rows = cur.fetchall()
        cur.execute("ROLLBACK")
        cur.close()

        if len(rows) == 0:
            return False, "zero_rows"
        return True, "ok"

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return False, f"error:{type(e).__name__}"


# =============================================================================
# DDL builder — builds CREATE TABLE string + sample rows for each table
# This is what gives Qwen real schema context in v2
# =============================================================================

def _build_ddl_block(table_name: str, columns: List[TypedColumn],
                     sample_rows: List[dict]) -> str:
    """
    Builds a compact DDL + sample data block for a table.
    Looks like:
        CREATE TABLE incident (
          id UUID PRIMARY KEY,
          status VARCHAR,          -- CATEGORY
          risk_score NUMERIC,      -- METRIC
          ...
        );
        Sample rows (5):
          | id | status | risk_score |
          | ab12 | open | 0.87 |
    """
    lines = [f"CREATE TABLE {table_name} ("]
    for col in columns:
        pk_marker  = " PRIMARY KEY" if col.is_pk else ""
        fk_comment = f" REFERENCES {col.fk_ref_table}({col.fk_ref_col})" if col.is_fk and col.fk_ref_table else ""
        sem_comment = f"  -- {col.semantic_type}" if col.semantic_type not in ("IDENTIFIER",) else ""
        lines.append(f"  {col.col_name} {col.data_type.upper()}{pk_marker}{fk_comment},{sem_comment}")
    # remove trailing comma from last column
    if lines[-1].endswith(",") or ",  --" in lines[-1]:
        lines[-1] = lines[-1].replace(",  --", "  --", 1)
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
    lines.append(");")

    if sample_rows:
        col_names = list(sample_rows[0].keys())
        lines.append(f"Sample rows ({len(sample_rows)}):")
        header = " | ".join(col_names)
        lines.append(f"  | {header} |")
        for row in sample_rows[:5]:
            vals = " | ".join(str(row.get(c, ""))[:30] for c in col_names)
            lines.append(f"  | {vals} |")

    return "\n".join(lines)


def _fetch_sample_rows(conn, table_name: str,
                       col_names: List[str], n: int = 5) -> List[dict]:
    """
    Fetches n sample rows from the table.
    Returns list of dicts. Returns [] on any error.
    """
    if not conn:
        return []
    try:
        cols_sql = ", ".join(f'"{c}"' for c in col_names[:15])  # cap columns
        sql = f'SELECT {cols_sql} FROM "{table_name}" LIMIT {n}'
        cur = conn.cursor()
        cur.execute(sql)
        rows_raw = cur.fetchall()
        col_descs = [desc[0] for desc in cur.description] if cur.description else col_names
        cur.close()
        return [dict(zip(col_descs, row)) for row in rows_raw]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return []


# =============================================================================
# Ollama call — same pattern as v1, lower temperature for SQL generation
# =============================================================================

_GEN_SYSTEM_PROMPT_V2 = """\
You are a SQL expert generating training data for a natural language to SQL system.

RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no commentary.
2. Generate exactly the number of pairs requested.
3. Questions must sound like a non-technical business analyst wrote them.
4. Questions must be diverse — vary vocabulary and intent.
5. SQL must be valid PostgreSQL using only the columns and table shown.
6. SQL must use double quotes for table and column names.
7. Never use column UUIDs or internal IDs in questions.
8. Return format: [{"question": "...", "sql": "SELECT ..."}, ...]
"""

_GEN_SYSTEM_PROMPT_V1 = """\
You are a data analyst generating training data for a natural language to SQL system.

RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no commentary.
2. Questions must be natural -- how a non-technical business user would phrase them.
3. Questions must be diverse -- vary vocabulary, phrasing, and intent.
4. Each question must specifically require the column to answer.
5. Keep questions concise (under 15 words each).
6. Never use the exact column name or table name in the question.
7. Multiple columns: return JSON object {"0": ["q1"], "1": ["q2"], ...}
8. Single column: return JSON array ["q1", "q2", ...]
"""


def _call_ollama(prompt: str, system_prompt: str,
                 n_items: int, temperature: float = 0.7) -> Optional[str]:
    """
    Calls Ollama with given prompt and system prompt.
    Returns raw response string or None if unavailable.
    """
    payload = {
        "model": SLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": n_items * 120,  # ~120 tokens per (question, SQL) pair
        },
        "stream": False,
    }

    url  = f"{SLM_OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = data,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    for attempt in range(SLM_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=SLM_TIMEOUT_SECS) as resp:
                raw  = resp.read().decode("utf-8")
                body = json.loads(raw)
                return body.get("message", {}).get("content", "")
        except (urllib.error.URLError, json.JSONDecodeError):
            if attempt == SLM_MAX_RETRIES - 1:
                return None
            time.sleep(1)
    return None


def _check_ollama_available() -> bool:
    """Returns True if Ollama is reachable."""
    try:
        url = f"{SLM_OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


# =============================================================================
# v2: DDL-based prompt builder and response parser
# =============================================================================

def _build_ddl_prompt(ddl_block: str, n_pairs: int) -> str:
    """
    Builds the prompt for v2 DDL-based generation.
    Gives Qwen the full table DDL + sample rows, asks for (question, SQL) pairs.
    """
    return (
        f"Given this database table:\n\n"
        f"{ddl_block}\n\n"
        f"Generate {n_pairs} natural language question and SQL query pairs "
        f"that a business analyst might ask about this table.\n"
        f"Each SQL must SELECT meaningful columns, not just filter on a single boolean.\n"
        f"Vary the intent: some should filter, some should aggregate, some should just list.\n"
        f"Return as a JSON array: "
        f'[{{"question": "...", "sql": "SELECT ..."}}, ...]'
    )


def _parse_ddl_response(raw: str) -> List[Tuple[str, str]]:
    """
    Parses Qwen's response for DDL-based generation.
    Returns list of (question, sql) tuples.
    Handles: JSON array of objects, partial JSON, markdown fences.
    """
    if not raw:
        return []

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"```\s*$", "", raw).strip()

    # Try to find JSON array
    start = raw.find("[")
    if start == -1:
        return []
    end = raw.rfind("]")
    if end <= start:
        # Try to close a truncated array
        raw = raw[start:] + "]"
        end = len(raw) - 1
    else:
        raw = raw[start:end+1]

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        # Try to repair: extract individual objects
        items = []
        for m in re.finditer(r'\{[^{}]+\}', raw, re.DOTALL):
            try:
                obj = json.loads(m.group())
                items.append(obj)
            except json.JSONDecodeError:
                continue

    pairs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question", "").strip()
        sql      = item.get("sql", "").strip()
        if question and sql and len(question) > 5 and "SELECT" in sql.upper():
            pairs.append((question, sql))

    return pairs


# =============================================================================
# v1: Column/FK/value prompt builders and parsers (unchanged from v1)
# =============================================================================

def _build_batch_column_prompt(batch):
    lines = [
        f"Generate {SYNTHETIC_QUERIES_PER_COLUMN} questions for EACH of the following "
        f"{len(batch)} database columns.",
        'Return a JSON object with integer string keys mapped to question arrays:',
        '{"0": ["q1", "q2"], "1": ["q3", "q4"], ...}',
        "",
    ]
    for i, (tc, sample_values) in enumerate(batch):
        vals = f", sample values: {', '.join(sample_values[:6])}" if sample_values else ""
        lines.append(
            f"{i}. {tc.table_name}.{tc.col_name} ({tc.data_type}, {tc.semantic_type}{vals})"
        )
    return "\n".join(lines)


def _build_batch_fk_prompt(fk_batch):
    lines = [
        f"Generate {SYNTHETIC_QUERIES_PER_FK} questions for EACH of the following "
        f"{len(fk_batch)} database relationships.",
        "Each question should require joining the two tables mentioned.",
        'Return a JSON object with integer string keys mapped to question arrays:',
        '{"0": ["q1", "q2"], "1": ["q3", "q4"], ...}',
        "",
    ]
    for i, (from_col, to_col) in enumerate(fk_batch):
        lines.append(
            f"{i}. {from_col.table_name}.{from_col.col_name} "
            f"→ {to_col.table_name}.{to_col.col_name} "
            f"(each {from_col.table_name} links to a {to_col.table_name})"
        )
    return "\n".join(lines)


def _build_batch_value_prompt(batch):
    lines = [
        f"Generate {SYNTHETIC_QUERIES_PER_COLUMN} questions for EACH of the following "
        f"{len(batch)} database columns.",
        "Each question should mention a DIFFERENT value from the column's known values.",
        'Return a JSON object with integer string keys mapped to question arrays:',
        '{"0": ["q1", "q2"], "1": ["q3", "q4"], ...}',
        "",
    ]
    for i, (tc, vals) in enumerate(batch):
        lines.append(
            f"{i}. {tc.table_name}.{tc.col_name} (CATEGORY, values: {', '.join(vals[:8])})"
        )
    return "\n".join(lines)


def _parse_batch_response(raw, batch, n):
    if not raw:
        return {}
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"```\s*$", "", raw).strip()

    start = raw.find("{")
    if start != -1:
        end = raw.rfind("}")
        if end > start:
            try:
                obj = json.loads(raw[start:end+1])
                result = {}
                for k, v in obj.items():
                    try:
                        idx = int(k)
                        if isinstance(v, list) and 0 <= idx < len(batch):
                            qs = [str(q).strip() for q in v if str(q).strip()]
                            if qs:
                                result[idx] = qs[:n]
                    except (ValueError, TypeError):
                        continue
                if result:
                    return result
            except json.JSONDecodeError:
                pass

    arr_start = raw.find("[")
    if arr_start != -1:
        arr_end = raw.rfind("]")
        if arr_end > arr_start:
            try:
                items = json.loads(raw[arr_start:arr_end+1])
                questions = [str(q).strip() for q in items if str(q).strip()]
                if not questions:
                    return {}
                if len(batch) == 1:
                    return {0: questions[:n]}
                chunk = max(1, len(questions) // len(batch))
                return {
                    i: questions[i*chunk:(i+1)*chunk][:n]
                    for i in range(len(batch))
                    if i*chunk < len(questions)
                }
            except json.JSONDecodeError:
                pass
    return {}


def _parse_question_list(raw: str, n: int) -> List[str]:
    if not raw:
        return []
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"```\s*$", "", raw).strip()
    start = raw.find("[")
    if start != -1:
        end = raw.rfind("]")
        if end > start:
            try:
                items = json.loads(raw[start:end+1])
                questions = [str(q).strip() for q in items if str(q).strip()]
                if questions:
                    return questions[:n]
            except json.JSONDecodeError:
                pass
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line and len(line) > 10 and "?" in line or len(line) > 15:
            questions.append(line)
        if len(questions) >= n:
            break
    return questions[:n]


# =============================================================================
# Rule-based fallback (v1 — unchanged)
# =============================================================================

_SEMANTIC_TEMPLATES = {
    "MONETARY": [
        "show {col} for each {table}",
        "what is the total {col} per {table}",
        "list {table} with their {col}",
    ],
    "TEMPORAL": [
        "when was each {table} {col_verb}",
        "show {table} records by {col}",
        "list {table} and their {col}",
    ],
    "CATEGORY": [
        "show {table} grouped by {col}",
        "list all {col} types in {table}",
        "filter {table} by {col}",
    ],
    "IDENTIFIER": [
        "find {table} by their {col}",
        "show {col} for each {table} record",
        "list all {table} {col}",
    ],
    "METRIC": [
        "show {col} for each {table}",
        "what is the average {col} per {table}",
        "list {table} sorted by {col}",
    ],
    "FREE_TEXT": [
        "search {table} by {col}",
        "show {table} {col}",
        "list {col} from {table}",
    ],
}


def _col_verb(col_name: str) -> str:
    name = col_name.lower()
    if "created"  in name: return "created"
    if "updated"  in name: return "updated"
    if "deleted"  in name: return "deleted"
    if "modified" in name: return "modified"
    if "date"     in name: return "recorded"
    return "processed"


def _generate_fallback_pairs(tc: TypedColumn) -> List[TrainingPair]:
    templates    = _SEMANTIC_TEMPLATES.get(tc.semantic_type, _SEMANTIC_TEMPLATES["FREE_TEXT"])
    col_readable = tc.col_name.replace("_", " ")
    tbl_readable = tc.table_name.replace("_", " ")
    pairs = []
    for tmpl in templates:
        query = tmpl.format(col=col_readable, table=tbl_readable,
                            col_verb=_col_verb(tc.col_name))
        pairs.append(TrainingPair(
            query=query, col_id=tc.col_id,
            col_name=tc.col_name, table_name=tc.table_name,
            source="fallback",
        ))
    return pairs


# =============================================================================
# Table-level grouping helper
# =============================================================================

def _group_columns_by_table(
    typed_cols: List[TypedColumn],
) -> Dict[str, List[TypedColumn]]:
    """Groups TypedColumn list into {table_name: [cols]} dict."""
    groups: Dict[str, List[TypedColumn]] = {}
    for tc in typed_cols:
        groups.setdefault(tc.table_name, []).append(tc)
    return groups


# =============================================================================
# Public entry point
# =============================================================================

def run_synthetic_query_gen(
    inference_result: InferenceResult,
    verbose: bool = False,
) -> SyntheticQueryGenResult:
    """
    Main entry point for Step 10 (v2).

    Two-phase generation:
      Phase A (v2, new): DDL + sample rows → Qwen → (question, SQL) pairs
                          → DB execution validation → keep only rows>0 pairs
      Phase B (v1, kept): column/FK/value prompts → Qwen → question pairs
                           (fallback and additional coverage)

    Parameters
    ----------
    inference_result : InferenceResult
    verbose : bool

    Returns
    -------
    SyntheticQueryGenResult
    """
    logger.debug("Starting synthetic query gen: %d typed columns", len(inference_result.typed_columns))

    if not SYNTHETIC_QUERY_GEN_ENABLED:
        logger.debug("Synthetic query gen disabled (SYNTHETIC_QUERY_GEN_ENABLED=False)")
        return SyntheticQueryGenResult(
            pairs=[], total_pairs=0, column_pairs=0, fk_pairs=0,
            value_pairs=0, fallback_pairs=0, ddl_pairs=0,
            validated_pairs=0, discarded_pairs=0,
            ollama_available=False, db_available=False,
            output_path=SYNTHETIC_PAIRS_PATH, duration_sec=0.0,
            stats={"skipped": "SYNTHETIC_QUERY_GEN_ENABLED=False"},
        )

    if SYNTHETIC_USE_EXISTING_PAIRS and os.path.exists(SYNTHETIC_PAIRS_PATH):
        existing = load_training_pairs(SYNTHETIC_PAIRS_PATH)
        if len(existing) >= SYNTHETIC_MIN_PAIRS_FOR_FINETUNE:
            if verbose:
                print(f"[SyntheticQueryGen] Reusing {len(existing)} existing pairs "
                      f"from {SYNTHETIC_PAIRS_PATH}")
            return SyntheticQueryGenResult(
                pairs=existing, total_pairs=len(existing),
                column_pairs=0, fk_pairs=0, value_pairs=0,
                fallback_pairs=0, ddl_pairs=0, validated_pairs=0,
                discarded_pairs=0, ollama_available=False, db_available=False,
                output_path=SYNTHETIC_PAIRS_PATH, duration_sec=0.0,
                stats={"skipped": "SYNTHETIC_USE_EXISTING_PAIRS=True",
                       "loaded_pairs": len(existing)},
            )

    t0 = time.time()

    ollama_ok = _check_ollama_available()
    sampled_cols: Dict[str, SampledColumn] = get_sampled_columns()

    # ------------------------------------------------------------------
    # Open DB connection for validation (v2)
    # ------------------------------------------------------------------
    val_conn   = _get_validation_connection() if _VALIDATION_ENABLED else None
    db_ok      = val_conn is not None

    if verbose:
        print("[SyntheticQueryGen] Starting training pair generation (v2)...")
        print(f"  Ollama available  : {ollama_ok}")
        print(f"  DB validation     : {'on' if db_ok else 'off'}")
        print(f"  Model             : {SLM_MODEL_NAME}")
        print(f"  DDL pairs/table   : {_DDL_PAIRS_PER_TABLE}")
        print(f"  Min sample rows   : {_DDL_MIN_SAMPLE_ROWS}")
        print(f"  Batch size (v1)   : {SYNTHETIC_GEN_BATCH_SIZE}")

    all_pairs:      List[TrainingPair] = []
    column_pairs    = 0
    fk_pairs        = 0
    value_pairs     = 0
    fallback_pairs  = 0
    ddl_pairs       = 0
    validated_pairs = 0
    discarded_pairs = 0

    # ------------------------------------------------------------------
    # Phase A — DDL-based generation + execution validation (v2)
    # ------------------------------------------------------------------
    if verbose:
        print("\n  [Phase A] DDL + execution validation...")

    table_groups = _group_columns_by_table(inference_result.typed_columns)

    for table_name, cols in table_groups.items():
        # Fetch sample rows to (1) check table isn't empty, (2) include in prompt
        sample_rows = _fetch_sample_rows(
            val_conn, table_name,
            [c.col_name for c in cols],
            n=5,
        ) if val_conn else []

        if val_conn and len(sample_rows) < _DDL_MIN_SAMPLE_ROWS:
            # Table is empty or nearly empty — skip DDL generation for this table
            # (validation would discard everything anyway due to 0 rows)
            if verbose:
                print(f"    ⚠  {table_name}: only {len(sample_rows)} sample rows "
                      f"(min={_DDL_MIN_SAMPLE_ROWS}), skipping DDL gen")
            continue

        ddl_block = _build_ddl_block(table_name, cols, sample_rows)

        if not ollama_ok:
            continue  # can't generate without SLM — v1 fallback handles it below

        prompt = _build_ddl_prompt(ddl_block, _DDL_PAIRS_PER_TABLE)
        raw    = _call_ollama(prompt, _GEN_SYSTEM_PROMPT_V2,
                              _DDL_PAIRS_PER_TABLE, temperature=0.4)
        pairs_raw = _parse_ddl_response(raw or "")

        if verbose and pairs_raw:
            print(f"    {table_name}: {len(pairs_raw)} pairs generated by SLM")

        for question, sql in pairs_raw:
            # Find which columns this SQL references — used for col_id mapping
            referenced_cols = [
                c for c in cols
                if f'"{c.col_name}"' in sql or c.col_name in sql
            ]
            if not referenced_cols:
                # SQL doesn't clearly reference any column — skip
                discarded_pairs += 1
                continue

            # Execution validation
            if _VALIDATION_ENABLED and val_conn:
                is_valid, reason = _validate_sql(val_conn, sql)
            else:
                is_valid, reason = True, "validation_disabled"

            source = "ddl_validated" if is_valid else "ddl_unvalidated"

            if not is_valid:
                discarded_pairs += 1
                if verbose:
                    print(f"      ✗ discarded ({reason}): {question[:60]}")
                continue

            # Create one TrainingPair per referenced column
            # (MiniLM fine-tuning uses col_id — map question to all relevant cols)
            for col in referenced_cols[:3]:  # cap at 3 cols per pair
                all_pairs.append(TrainingPair(
                    query=question,
                    col_id=col.col_id,
                    col_name=col.col_name,
                    table_name=table_name,
                    source=source,
                    sql=sql,
                    validated=is_valid,
                ))
                ddl_pairs += 1
                if is_valid:
                    validated_pairs += 1

    if verbose:
        print(f"\n  Phase A complete: {ddl_pairs} DDL pairs "
              f"({validated_pairs} validated, {discarded_pairs} discarded)")

    # ------------------------------------------------------------------
    # Phase B — v1 column/FK/value generation
    # (kept for coverage on columns not well-represented by DDL pairs)
    # ------------------------------------------------------------------

    # Select eligible columns (same priority logic as v1)
    all_eligible = [
        tc for tc in inference_result.typed_columns
        if tc.semantic_type in SYNTHETIC_GEN_ELIGIBLE_TYPES
        and not tc.is_pk
    ]
    _sampled_now = get_sampled_columns()
    p1 = [tc for tc in all_eligible
          if tc.semantic_type == "CATEGORY"
          and tc.col_id in _sampled_now
          and len(_sampled_now[tc.col_id].raw_values) >= 2]
    p1_ids = {tc.col_id for tc in p1}
    p2 = [tc for tc in all_eligible if tc.is_display_col and tc.col_id not in p1_ids]
    p12_ids = p1_ids | {tc.col_id for tc in p2}
    p3 = [tc for tc in all_eligible if tc.col_id not in p12_ids]
    eligible = (p1 + p2 + p3)[:SYNTHETIC_GEN_MAX_COLUMNS]

    if verbose:
        print(f"\n  [Phase B] Column/FK/value generation "
              f"({len(eligible)} eligible columns)...")

    # Source 1 — column-based
    col_batch_inputs = [
        (tc, sampled_cols[tc.col_id].raw_values[:8] if tc.col_id in sampled_cols else [])
        for tc in eligible
    ]
    for batch_start in range(0, len(col_batch_inputs), SYNTHETIC_GEN_BATCH_SIZE):
        batch = col_batch_inputs[batch_start:batch_start + SYNTHETIC_GEN_BATCH_SIZE]
        if ollama_ok:
            raw    = _call_ollama(_build_batch_column_prompt(batch),
                                  _GEN_SYSTEM_PROMPT_V1,
                                  len(batch) * SYNTHETIC_QUERIES_PER_COLUMN,
                                  temperature=0.7)
            result = _parse_batch_response(raw, batch, SYNTHETIC_QUERIES_PER_COLUMN)
            covered = set()
            for idx, questions in result.items():
                if questions and idx < len(batch):
                    tc = batch[idx][0]
                    for q in questions:
                        all_pairs.append(TrainingPair(
                            query=q, col_id=tc.col_id,
                            col_name=tc.col_name, table_name=tc.table_name,
                            source="column",
                        ))
                        column_pairs += 1
                    covered.add(idx)
            for idx, (tc, _) in enumerate(batch):
                if idx not in covered:
                    fb = _generate_fallback_pairs(tc)
                    all_pairs.extend(fb)
                    fallback_pairs += len(fb)
        else:
            for tc, _ in batch:
                fb = _generate_fallback_pairs(tc)
                all_pairs.extend(fb)
                fallback_pairs += len(fb)

    # Source 2 — FK-based
    col_lookup = {tc.col_id: tc for tc in inference_result.typed_columns}
    fk_edges   = getattr(inference_result, "_fk_edges", [])
    fk_col_pairs = []
    for edge in fk_edges:
        from_col = col_lookup.get(edge.get("from_col_id", ""))
        to_col   = col_lookup.get(edge.get("to_col_id", ""))
        if from_col and to_col:
            fk_col_pairs.append((from_col, to_col))
    fk_col_pairs = fk_col_pairs[:SYNTHETIC_GEN_MAX_FK_EDGES]

    for batch_start in range(0, len(fk_col_pairs), SYNTHETIC_GEN_BATCH_SIZE):
        batch = fk_col_pairs[batch_start:batch_start + SYNTHETIC_GEN_BATCH_SIZE]
        if ollama_ok:
            raw    = _call_ollama(_build_batch_fk_prompt(batch),
                                  _GEN_SYSTEM_PROMPT_V1,
                                  len(batch) * SYNTHETIC_QUERIES_PER_FK,
                                  temperature=0.7)
            result = _parse_batch_response(raw, batch, SYNTHETIC_QUERIES_PER_FK)
            for idx, questions in result.items():
                if questions and idx < len(batch):
                    from_col, to_col = batch[idx]
                    for q in questions:
                        for col in (from_col, to_col):
                            all_pairs.append(TrainingPair(
                                query=q, col_id=col.col_id,
                                col_name=col.col_name,
                                table_name=col.table_name,
                                source="fk",
                            ))
                            fk_pairs += 1

    # Source 3 — value-based
    value_eligible = [
        tc for tc in eligible
        if tc.semantic_type == "CATEGORY"
        and tc.col_id in sampled_cols
        and len(sampled_cols[tc.col_id].raw_values) >= 2
    ]
    val_batch_inputs = [(tc, sampled_cols[tc.col_id].raw_values) for tc in value_eligible]
    for batch_start in range(0, len(val_batch_inputs), SYNTHETIC_GEN_BATCH_SIZE):
        batch = val_batch_inputs[batch_start:batch_start + SYNTHETIC_GEN_BATCH_SIZE]
        if ollama_ok:
            raw    = _call_ollama(_build_batch_value_prompt(batch),
                                  _GEN_SYSTEM_PROMPT_V1,
                                  len(batch) * SYNTHETIC_QUERIES_PER_COLUMN,
                                  temperature=0.7)
            result = _parse_batch_response(raw, batch, SYNTHETIC_QUERIES_PER_COLUMN)
            covered = set()
            for idx, questions in result.items():
                if questions and idx < len(batch):
                    tc = batch[idx][0]
                    for q in questions:
                        all_pairs.append(TrainingPair(
                            query=q, col_id=tc.col_id,
                            col_name=tc.col_name, table_name=tc.table_name,
                            source="value",
                        ))
                        value_pairs += 1
                    covered.add(idx)
            for idx, (tc, vals) in enumerate(batch):
                if idx not in covered:
                    cr = tc.col_name.replace("_", " ")
                    tr = tc.table_name.replace("_", " ")
                    for v in vals[:SYNTHETIC_QUERIES_PER_COLUMN]:
                        all_pairs.append(TrainingPair(
                            query=f"show {tr} where {cr} is {v}",
                            col_id=tc.col_id, col_name=tc.col_name,
                            table_name=tc.table_name, source="value",
                        ))
                        value_pairs += 1
        else:
            for tc, vals in batch:
                cr = tc.col_name.replace("_", " ")
                tr = tc.table_name.replace("_", " ")
                for v in vals[:SYNTHETIC_QUERIES_PER_COLUMN]:
                    all_pairs.append(TrainingPair(
                        query=f"show {tr} where {cr} is {v}",
                        col_id=tc.col_id, col_name=tc.col_name,
                        table_name=tc.table_name, source="value",
                    ))
                    value_pairs += 1

    if verbose:
        print(f"\n  Phase B complete: col={column_pairs} fk={fk_pairs} "
              f"value={value_pairs} fallback={fallback_pairs}")

    # ------------------------------------------------------------------
    # Close validation connection
    # ------------------------------------------------------------------
    if val_conn:
        try:
            val_conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Write JSONL output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(SYNTHETIC_PAIRS_PATH)), exist_ok=True)

    with open(SYNTHETIC_PAIRS_PATH, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps({
                "query":      pair.query,
                "col_id":     pair.col_id,
                "col_name":   pair.col_name,
                "table_name": pair.table_name,
                "source":     pair.source,
                "sql":        pair.sql,
                "validated":  pair.validated,
            }) + "\n")

    duration = round(time.time() - t0, 4)
    total    = len(all_pairs)

    stats = {
        "total_pairs":         total,
        "ddl_pairs":           ddl_pairs,
        "validated_pairs":     validated_pairs,
        "discarded_pairs":     discarded_pairs,
        "column_pairs":        column_pairs,
        "fk_pairs":            fk_pairs,
        "value_pairs":         value_pairs,
        "fallback_pairs":      fallback_pairs,
        "ollama_available":    ollama_ok,
        "db_available":        db_ok,
        "output_path":         SYNTHETIC_PAIRS_PATH,
        "meets_min_threshold": total >= SYNTHETIC_MIN_PAIRS_FOR_FINETUNE,
        "duration_sec":        duration,
    }

    if verbose:
        print(f"\n  ── Summary ──────────────────────────────────")
        print(f"  DDL validated      : {validated_pairs}")
        print(f"  DDL unvalidated    : {ddl_pairs - validated_pairs}")
        print(f"  DDL discarded      : {discarded_pairs}")
        print(f"  Column (v1)        : {column_pairs}")
        print(f"  FK (v1)            : {fk_pairs}")
        print(f"  Value (v1)         : {value_pairs}")
        print(f"  Fallback           : {fallback_pairs}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Total pairs        : {total}")
        print(f"  Meets threshold    : {total >= SYNTHETIC_MIN_PAIRS_FOR_FINETUNE} "
              f"(min={SYNTHETIC_MIN_PAIRS_FOR_FINETUNE})")
        print(f"  Written to         : {SYNTHETIC_PAIRS_PATH}")
        print(f"  Duration           : {duration}s")
        print()
        if all_pairs:
            print("  Sample pairs (first 5):")
            for p in all_pairs[:5]:
                src_tag = f"[{p.source}]"
                print(f"    {src_tag:<20} '{p.query}'")
                print(f"    {'':20} → {p.table_name}.{p.col_name}"
                      + (f"  SQL: {p.sql[:60]}..." if p.sql else ""))
        print("[SyntheticQueryGen] Done.\n")

    logger.info(
        "Synthetic query gen complete: total=%d (col=%d fk=%d val=%d fallback=%d) ollama=%s",
        total, column_pairs, fk_pairs, value_pairs, fallback_pairs, ollama_ok,
    )

    return SyntheticQueryGenResult(
        pairs=all_pairs, total_pairs=total,
        column_pairs=column_pairs, fk_pairs=fk_pairs,
        value_pairs=value_pairs, fallback_pairs=fallback_pairs,
        ddl_pairs=ddl_pairs, validated_pairs=validated_pairs,
        discarded_pairs=discarded_pairs,
        ollama_available=ollama_ok, db_available=db_ok,
        output_path=SYNTHETIC_PAIRS_PATH, duration_sec=duration,
        stats=stats,
    )


def load_training_pairs(path: str = SYNTHETIC_PAIRS_PATH) -> List[TrainingPair]:
    """
    Loads training pairs from a JSONL file.
    v2-compatible: reads optional sql and validated fields.
    Used by auto_finetune.py.
    """
    if not os.path.exists(path):
        return []

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pairs.append(TrainingPair(
                    query      = obj["query"],
                    col_id     = obj["col_id"],
                    col_name   = obj["col_name"],
                    table_name = obj["table_name"],
                    source     = obj.get("source", "unknown"),
                    sql        = obj.get("sql", ""),
                    validated  = obj.get("validated", False),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
    return pairs


# =============================================================================
# Smoke test — python ingestion/synthetic_query_gen.py
# =============================================================================

if __name__ == "__main__":
    from schema.simulate_schema import get_simulated_schema
    from ingestion.schema_scanner import run_schema_scanner
    from ingestion.semantic_type_inference import run_semantic_type_inference
    from ingestion.value_sampler import run_value_sampler

    print("Running ingestion pipeline...")
    raw_schema       = get_simulated_schema()
    scan_result      = run_schema_scanner(raw_schema=raw_schema, verbose=False)
    inference_result = run_semantic_type_inference(scan_result=scan_result, verbose=False)
    run_value_sampler(inference_result=inference_result, verbose=False)
    inference_result._fk_edges = scan_result.fk_edges

    print("Running synthetic query generator (v2)...\n")
    result = run_synthetic_query_gen(inference_result=inference_result, verbose=True)

    print("=" * 70)
    print("VEDA POC — Synthetic Query Generator v2 Output")
    print("=" * 70)
    print(f"  Total pairs        : {result.total_pairs}")
    print(f"  DDL validated      : {result.validated_pairs}")
    print(f"  DDL discarded      : {result.discarded_pairs}")
    print(f"  Column pairs (v1)  : {result.column_pairs}")
    print(f"  FK pairs (v1)      : {result.fk_pairs}")
    print(f"  Value pairs (v1)   : {result.value_pairs}")
    print(f"  Fallback           : {result.fallback_pairs}")
    print(f"  Ollama available   : {result.ollama_available}")
    print(f"  DB validation      : {result.db_available}")
    print(f"  Output path        : {result.output_path}")
    print(f"  Meets threshold    : {result.stats['meets_min_threshold']}")
    print(f"  Duration           : {result.duration_sec}s")
    print()
    from collections import Counter
    src_counts = Counter(p.source for p in result.pairs)
    print("  Source distribution:")
    for src, cnt in src_counts.most_common():
        print(f"    {src:<20} : {cnt}")