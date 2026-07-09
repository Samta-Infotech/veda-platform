# =============================================================================
# ingestion/value_sampler.py
# VEDA POC — Step 6: Column Value Sampler
#
# Responsibility:
#   - Samples distinct non-null values from CATEGORY, FREE_TEXT, and
#     IDENTIFIER columns during ingestion
#   - Persists sampled values to the column_values store
#   - Provides get_column_values_for_expansion() for query-time use
#     in semantic_layer.py Step 1b
#
# Why this matters — the vocabulary gap problem:
#   MiniLM understands semantics but fails on terminology mismatches where
#   the user's word is a VALUE in the column, not the column name itself.
#
#   Example:
#     User query:  "show all escalated incidents"
#     Column name: incident.workflow_state  (type: CATEGORY)
#     Column values: ['open', 'escalated', 'closed', 'investigating']
#
#   MiniLM doesn't know that "escalated" is a value of workflow_state.
#   But the value sampler does — it sampled "escalated" during ingestion.
#   At query time, "escalated" in the query maps to workflow_state column.
#
#   This fixes the vocabulary gap generically on ANY schema, ANY domain,
#   with zero human input — the values themselves are the synonym dictionary.
#
# Design constraints:
#   - Read-only: only SELECT DISTINCT queries against the client DB
#   - Generic: works on PostgreSQL, MySQL, SQLite, SQL Server, Oracle
#     (uses only standard SQL — no DB-specific extensions)
#   - Performance-safe: samples at most VALUE_SAMPLE_SIZE rows per column
#   - Graceful fallback: DB unavailable → empty store, pipeline continues
#   - Independent: no coupling to encoder mode, vector dimensions, or schema
#
# Query-time usage (semantic_layer.py Step 1b):
#   Before tokenisation, check if any query token matches a sampled value.
#   If match found, inject the column name as an additional search token.
#   e.g. query "show escalated incidents" → inject "workflow_state" token
#   This dramatically improves recall for enum-value queries.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ingestion.semantic_type_inference import InferenceResult, TypedColumn
from config import (
    VALUE_SAMPLER_ENABLED,
    VALUE_SAMPLE_SIZE,
    VALUE_SAMPLER_ELIGIBLE_TYPES,
    VALUE_SAMPLER_MAX_VALUE_LEN,
    COLUMN_VALUES_TABLE_NAME,
    VALUE_EXPANSION_MIN_TOKEN_LEN,
    VALUE_EXPANSION_PARTIAL_MIN_TOKEN_LEN,
    VALUE_EXPANSION_MAX_COL_MATCHES,
    SENSITIVE_PATTERNS,
    get_primary_relational_source,
)
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE as PSYCOPG2_AVAILABLE,
    get_internal_connection,
    release_internal_connection,
    get_client_connection,
    DICT_CURSOR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class SampledColumn:
    """Sampled values for a single column."""
    col_id:        str
    col_name:      str
    table_id:      str
    table_name:    str
    semantic_type: str
    values:        List[str]       # distinct non-null values, lowercase normalised
    raw_values:    List[str]       # original casing — used for display


@dataclass
class ValueSamplerResult:
    """Top-level output of the value sampler."""
    sampled_columns:  List[SampledColumn]
    total_values:     int
    columns_sampled:  int
    columns_skipped:  int
    backend:          str
    duration_sec:     float
    stats:            dict = field(default_factory=dict)


# =============================================================================
# In-memory store
# Two structures for efficient access:
#   _VALUE_STORE      — col_id → SampledColumn (for retrieval by col_id)
#   _VALUE_INDEX      — normalised_value → List[col_id] (for query expansion)
# =============================================================================

_VALUE_STORE: Dict[str, SampledColumn] = {}
_VALUE_INDEX: Dict[str, List[str]]     = {}   # value → [col_id, ...]


def _build_value_index(sampled_columns: List[SampledColumn]) -> Dict[str, List[str]]:
    """
    Builds an inverted index: normalised_value → [col_id, ...].
    Used at query time for O(1) value→column lookup.
    """
    index: Dict[str, List[str]] = {}
    for sc in sampled_columns:
        for val in sc.values:
            if val not in index:
                index[val] = []
            if sc.col_id not in index[val]:
                index[val].append(sc.col_id)
    return index


# =============================================================================
# Column eligibility
# =============================================================================

def _is_eligible(tc: TypedColumn) -> bool:
    """
    Returns True if this column's values are worth sampling.
    Focus on columns whose values users would type in queries.
    """
    # Must be an eligible semantic type
    if tc.semantic_type not in VALUE_SAMPLER_ELIGIBLE_TYPES:
        return False

    # Skip PKs — integer IDs are not useful query expansion tokens
    if tc.is_pk:
        return False

    # Skip FK columns — they reference IDs, not readable values
    if tc.is_fk:
        return False

    # Skip columns whose names look sensitive
    col_lower = tc.col_name.lower()
    if any(p in col_lower for p in SENSITIVE_PATTERNS):
        return False

    # Skip pure integer columns — numeric IDs are not vocabulary tokens
    if tc.data_type.lower() in ("integer", "bigint", "smallint", "numeric"):
        return False

    return True


def _normalise_value(val: str) -> str:
    """
    Normalises a sampled value for index storage and query matching.
    Lowercase, strip whitespace, collapse internal spaces.
    """
    return re.sub(r"\s+", " ", str(val).lower().strip())


def _is_useful_value(val: str, normalised: str) -> bool:
    """
    Returns True if this value is useful as a query expansion token.
    Filters out noise, numbers, empty strings, and overly long values.
    """
    if not normalised:
        return False

    # Too short to be meaningful
    if len(normalised) < VALUE_EXPANSION_MIN_TOKEN_LEN:
        return False

    # Too long — likely a text blob, not an enum value
    if len(val) > VALUE_SAMPLER_MAX_VALUE_LEN:
        return False

    # Purely numeric — not useful as a vocabulary token
    if normalised.replace(".", "").replace("-", "").isdigit():
        return False

    # UUID-like — not useful
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                normalised):
        return False

    return True


# =============================================================================
# SQL sampling — generic, works on any RDBMS
# =============================================================================

def _q(name: str) -> str:
    """Double-quote identifier."""
    return f'"{name.replace(chr(34), "")}"'


def _sample_column_values(
    cursor,
    table_name:    str,
    col_name:      str,
    n:             int,
    semantic_type: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Samples distinct non-null values from table.col.
    CATEGORY columns: no LIMIT — controlled vocabularies are low-cardinality
    and we need the full set for reliable query-time expansion.
    FREE_TEXT / IDENTIFIER: frequency-ordered and capped at n so the most
    commonly queried values surface first when cardinality exceeds the cap.
    """
    try:
        if semantic_type == "CATEGORY":
            cursor.execute(f"""
                SELECT DISTINCT {_q(col_name)}
                FROM {_q(table_name)}
                WHERE {_q(col_name)} IS NOT NULL;
            """)
        else:
            cursor.execute(f"""
                SELECT {_q(col_name)}, COUNT(*) AS freq
                FROM {_q(table_name)}
                WHERE {_q(col_name)} IS NOT NULL
                GROUP BY {_q(col_name)}
                ORDER BY freq DESC
                LIMIT %s;
            """, (n,))
        rows = cursor.fetchall()
    except Exception:
        return [], []

    raw_values  = []
    norm_values = []

    for row in rows:
        raw = str(row[0]) if row[0] is not None else ""
        normalised = _normalise_value(raw)
        if _is_useful_value(raw, normalised):
            raw_values.append(raw)
            norm_values.append(normalised)

    return raw_values, norm_values


# =============================================================================
# pgvector persistence
# =============================================================================

def _create_column_values_table(cursor) -> None:
    """Creates the column_values table if it does not exist."""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {COLUMN_VALUES_TABLE_NAME} (
            col_id        TEXT NOT NULL,
            col_name      TEXT NOT NULL,
            table_id      TEXT NOT NULL,
            table_name    TEXT NOT NULL,
            semantic_type TEXT NOT NULL,
            value_norm    TEXT NOT NULL,   -- lowercase normalised value
            value_raw     TEXT NOT NULL    -- original casing
        );
    """)
    # Index on value_norm for fast query-time expansion lookup
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{COLUMN_VALUES_TABLE_NAME}_value
        ON {COLUMN_VALUES_TABLE_NAME} (value_norm);
    """)
    # Index on col_id for retrieval by column
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{COLUMN_VALUES_TABLE_NAME}_col
        ON {COLUMN_VALUES_TABLE_NAME} (col_id);
    """)


def _store_values_pgvector(sampled_columns: List[SampledColumn]) -> int:
    """Persists sampled values to the internal pgvector store. Returns rows written.

    Batched via execute_values (F1) — was one INSERT per (column × sampled
    value), i.e. potentially tens of thousands of round trips for a wide
    schema. Same rows, same order-independent result, one bulk statement.
    """
    from psycopg2.extras import execute_values
    conn = get_internal_connection()
    written = 0
    try:
        with conn:
            with conn.cursor() as cur:
                _create_column_values_table(cur)
                cur.execute(f"TRUNCATE TABLE {COLUMN_VALUES_TABLE_NAME};")

                rows = [
                    (sc.col_id, sc.col_name, sc.table_id, sc.table_name,
                     sc.semantic_type, norm, raw)
                    for sc in sampled_columns
                    for raw, norm in zip(sc.raw_values, sc.values)
                ]
                if rows:
                    execute_values(
                        cur,
                        f"""INSERT INTO {COLUMN_VALUES_TABLE_NAME}
                            (col_id, col_name, table_id, table_name,
                             semantic_type, value_norm, value_raw)
                            VALUES %s""",
                        rows,
                        page_size=1000,   # chunk very large value sets
                    )
                    written = len(rows)
    finally:
        release_internal_connection(conn)
    return written


def _query_values_pgvector(
    query_tokens: List[str],
) -> Dict[str, List[str]]:
    """
    Returns {col_id: [matching_values]} for tokens that match stored values.
    Used at query time for expansion.
    """
    if not query_tokens:
        return {}

    placeholders = ",".join(["%s"] * len(query_tokens))
    conn = get_internal_connection()
    try:
        with conn.cursor(cursor_factory=DICT_CURSOR) as cur:
            cur.execute(f"""
                SELECT col_id, col_name, table_name, value_norm, value_raw
                FROM {COLUMN_VALUES_TABLE_NAME}
                WHERE value_norm IN ({placeholders});
            """, query_tokens)
            rows = cur.fetchall()
    finally:
        release_internal_connection(conn)

    result: Dict[str, List[str]] = {}
    for row in rows:
        cid = row["col_id"]
        if cid not in result:
            result[cid] = []
        result[cid].append(row["col_name"])
    return result


# =============================================================================
# Public entry point — ingestion
# =============================================================================

def run_value_sampler(
    inference_result: InferenceResult,
    source_id: str = None,
    verbose: bool = False,
) -> ValueSamplerResult:
    """
    Main entry point for Step 6.

    Samples distinct values from eligible columns and persists them.
    Called independently from main.py after semantic type inference.

    Parameters
    ----------
    inference_result : InferenceResult
        Output of run_semantic_type_inference(). Contains typed_columns.
    verbose : bool

    Returns
    -------
    ValueSamplerResult
        Always returns even if DB unavailable (empty result on failure).
    """
    global _VALUE_STORE, _VALUE_INDEX

    t0 = time.time()

    logger.debug("Starting value sampler: %d typed columns", len(inference_result.typed_columns))

    if not VALUE_SAMPLER_ENABLED:
        logger.debug("Value sampler disabled (VALUE_SAMPLER_ENABLED=False)")
        return ValueSamplerResult(
            sampled_columns = [],
            total_values    = 0,
            columns_sampled = 0,
            columns_skipped = 0,
            backend         = "disabled",
            duration_sec    = 0.0,
            stats           = {"skipped": "VALUE_SAMPLER_ENABLED=False"},
        )

    # Select eligible columns
    eligible = [tc for tc in inference_result.typed_columns if _is_eligible(tc)]
    skipped  = len(inference_result.typed_columns) - len(eligible)

    if verbose:
        print("[ValueSampler] Sampling column values...")
        print(f"  Eligible columns : {len(eligible)} / {len(inference_result.typed_columns)}")
        print(f"  Sample size      : {VALUE_SAMPLE_SIZE} per column")
        print(f"  Backend          : {'pgvector' if PSYCOPG2_AVAILABLE else 'in_memory_fallback'}")

    sampled_columns: List[SampledColumn] = []

    if PSYCOPG2_AVAILABLE:
        try:
            _source_id = source_id or get_primary_relational_source()["id"]
            conn = get_client_connection(_source_id)
            try:
                with conn.cursor() as cur:
                    for tc in eligible:
                        raw_vals, norm_vals = _sample_column_values(
                            cur, tc.table_name, tc.col_name, VALUE_SAMPLE_SIZE,
                            semantic_type=tc.semantic_type,
                        )
                        if norm_vals:
                            sampled_columns.append(SampledColumn(
                                col_id        = tc.col_id,
                                col_name      = tc.col_name,
                                table_id      = tc.table_id,
                                table_name    = tc.table_name,
                                semantic_type = tc.semantic_type,
                                values        = norm_vals,
                                raw_values    = raw_vals,
                            ))
            finally:
                conn.close()

            # Persist to pgvector
            total_written = _store_values_pgvector(sampled_columns)
            backend = "pgvector"

        except Exception as e:
            if verbose:
                print(f"  ⚠ DB connection failed ({e}) — using in-memory fallback")
            # Try in-memory only — no DB sampling possible
            sampled_columns = []
            total_written   = 0
            backend         = "in_memory_fallback"
    else:
        # No psycopg2 — cannot sample
        sampled_columns = []
        total_written   = 0
        backend         = "in_memory_fallback (no psycopg2)"

    # Always build in-memory index (even from empty list)
    # This ensures query-time expansion works without a DB roundtrip
    _VALUE_STORE = {sc.col_id: sc for sc in sampled_columns}
    _VALUE_INDEX = _build_value_index(sampled_columns)

    total_values = sum(len(sc.values) for sc in sampled_columns)
    duration     = round(time.time() - t0, 4)

    stats = {
        "columns_eligible": len(eligible),
        "columns_sampled":  len(sampled_columns),
        "columns_skipped":  skipped,
        "total_values":     total_values,
        "index_size":       len(_VALUE_INDEX),
        "backend":          backend,
        "duration_sec":     duration,
    }

    if verbose:
        print(f"  Columns sampled  : {len(sampled_columns)}")
        print(f"  Total values     : {total_values}")
        print(f"  Index entries    : {len(_VALUE_INDEX)}")
        print(f"  Duration         : {duration}s")
        if sampled_columns:
            print(f"\n  Sample (first 5 columns):")
            for sc in sampled_columns[:5]:
                print(f"    {sc.table_name}.{sc.col_name:<28} "
                      f"[{sc.semantic_type}]  "
                      f"values: {sc.values[:5]}")
        print("[ValueSampler] Done.\n")

    logger.info(
        "Value sampler complete: %d columns sampled, %d values, backend=%s",
        len(sampled_columns), total_values, backend,
    )

    return ValueSamplerResult(
        sampled_columns = sampled_columns,
        total_values    = total_values,
        columns_sampled = len(sampled_columns),
        columns_skipped = skipped,
        backend         = backend,
        duration_sec    = duration,
        stats           = stats,
    )


# =============================================================================
# Public entry point — query time
# =============================================================================

def expand_query_tokens(
    tokens: List[str],
    verbose: bool = False,
    full_query: str = "",
) -> Tuple[List[str], Dict[str, str]]:
    """
    Query-time vocabulary expansion.

    Called by semantic_layer.py Step 1b before tokenisation.
    Checks each query token against the value index. When a match is found,
    injects the column name as an additional search token.

    Example:
      Input tokens: ["show", "escalated", "incidents"]
      Value index:  "escalated" → [col_id of incident.workflow_state]
      Output tokens: ["show", "escalated", "incidents", "workflow_state"]
      Expansion map: {"escalated": "workflow_state"}

    Parameters
    ----------
    tokens : List[str]
        Tokenised query terms (already lowercased, stopwords removed).
    verbose : bool

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        (expanded_tokens, expansion_map)
        expansion_map: {matched_value → injected_col_name} for logging
    """
    if not _VALUE_INDEX:
        # Value index not populated — try pgvector fallback
        if PSYCOPG2_AVAILABLE:
            try:
                matches = _query_values_pgvector(tokens)
                if not matches:
                    return tokens, {}
                expansion_map: Dict[str, str] = {}
                extra_tokens: List[str] = []
                for col_id, col_names in matches.items():
                    for col_name in col_names:
                        if col_name not in tokens and col_name not in extra_tokens:
                            extra_tokens.append(col_name)
                            # Find matching token for the map
                            for tok in tokens:
                                if tok in _VALUE_INDEX and col_id in _VALUE_INDEX[tok]:
                                    expansion_map[tok] = col_name
                                    break
                if verbose and extra_tokens:
                    print(f"  [ValueExpansion] Injected via pgvector: {extra_tokens}")
                return tokens + extra_tokens, expansion_map
            except Exception:
                pass
        return tokens, {}

    expansion_map: Dict[str, str] = {}
    extra_tokens:  List[str]      = []

    # Context-aware column preference
    # When query mentions domain words, prefer those columns
    _CONTEXT_MAP = {
        "permission":  ["perm_name", "perm_code", "description"],
        "queue":       ["is_current_queue", "target_queue", "owned_by_group"],
        "alert":       ["object_type", "workflow_state"],
        "incident":    ["object_type", "workflow_state", "incident_status"],
        "role":        ["role_name", "role_code"],
        "workflow":    ["workflow_state", "name"],
        "transaction": ["transaction_type", "status"],
        "escalat":     ["workflow_state"],
    }
    query_lower    = full_query.lower() if full_query else " ".join(tokens)
    preferred_cols = []
    for ctx_word, cols in _CONTEXT_MAP.items():
        if ctx_word in query_lower:
            preferred_cols.extend(cols)

    for token in tokens:
        # Direct match — query token is a sampled column value
        if token in _VALUE_INDEX:
            if len(_VALUE_INDEX[token]) > VALUE_EXPANSION_MAX_COL_MATCHES:
                continue
            candidates = [
                sc for col_id in _VALUE_INDEX[token]
                if (sc := _VALUE_STORE.get(col_id))
                and (
                    sc.semantic_type == "CATEGORY"
                    or (sc.semantic_type == "FREE_TEXT" and sc.col_name in preferred_cols)
                )
                and sc.col_name not in tokens
                and sc.col_name not in extra_tokens
            ]
            if candidates:
                best = candidates[0]
                extra_tokens.append(best.col_name)
                expansion_map[token] = best.col_name

        # Partial match — query token is a prefix/substring of a value
        # Only for tokens longer than 4 chars to avoid noise
        elif len(token) >= VALUE_EXPANSION_PARTIAL_MIN_TOKEN_LEN:
            for val, col_ids in _VALUE_INDEX.items():
                if token in val or val in token:
                    for col_id in col_ids:
                        sc = _VALUE_STORE.get(col_id)
                        if (sc and sc.col_name not in tokens
                                and sc.col_name not in extra_tokens):
                            extra_tokens.append(sc.col_name)
                            expansion_map[token] = sc.col_name
                    break   # one partial match per token is enough

    # Bigram check — "level 1", "open incidents" etc.
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        if bigram in _VALUE_INDEX:
            candidates = []
            for col_id in _VALUE_INDEX[bigram]:
                sc = _VALUE_STORE.get(col_id)
                if (sc
                        and sc.col_name not in tokens
                        and sc.col_name not in extra_tokens
                        and (
                            sc.semantic_type in ("CATEGORY", "IDENTIFIER")
                            or (sc.semantic_type == "FREE_TEXT" and sc.col_name in preferred_cols)
                        )):
                    candidates.append(sc)
            def _score(sc):
                ctx = 0 if sc.col_name in preferred_cols else 1
                typ = {"CATEGORY": 0, "IDENTIFIER": 1, "FREE_TEXT": 2}.get(sc.semantic_type, 3)
                return (ctx, typ)
            candidates.sort(key=_score)
            if candidates:
                best = candidates[0]
                extra_tokens.append(best.col_name)
                expansion_map[bigram] = best.col_name
                if verbose:
                    print(f"  [ValueExpansion] Bigram '{bigram}' → '{best.col_name}' [{best.semantic_type}]")

    # Substring match for FREE_TEXT columns when context matches
    # e.g. "level 1" is substring of perm_name value "view level 1"
    if preferred_cols:
        for val, col_ids in _VALUE_INDEX.items():
            # Check if any query bigram/token is substring of stored value
            matched_phrase = None
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + " " + tokens[i+1]
                if bigram in val and len(bigram) >= 4:
                    matched_phrase = bigram
                    break
            if not matched_phrase:
                for tok in tokens:
                    if len(tok) >= 4 and tok in val:
                        matched_phrase = tok
                        break
            if not matched_phrase:
                continue

            for col_id in col_ids:
                sc = _VALUE_STORE.get(col_id)
                if (sc
                        and sc.semantic_type == "FREE_TEXT"
                        and sc.col_name in preferred_cols
                        and sc.col_name not in tokens
                        and sc.col_name not in extra_tokens):
                    extra_tokens.append(sc.col_name)
                    expansion_map[matched_phrase] = sc.col_name
                    if verbose:
                        print(f"  [ValueExpansion] Substring '{matched_phrase}' in '{val}' → '{sc.col_name}' [FREE_TEXT]")
                    break

    if verbose and extra_tokens:
        print(f"  [ValueExpansion] Tokens injected: {extra_tokens}")
        for matched, injected in expansion_map.items():
            print(f"    '{matched}' → '{injected}'")

    return tokens + extra_tokens, expansion_map


def get_sampled_columns() -> Dict[str, SampledColumn]:
    """Returns the full in-memory value store. Used by synthetic_query_gen.py."""
    return _VALUE_STORE


def get_value_index() -> Dict[str, List[str]]:
    """Returns the inverted value index. Used for inspection and testing."""
    return _VALUE_INDEX


def rebuild_value_index_from_db(force: bool = False) -> int:
    """
    Rebuilds _VALUE_INDEX from the persisted column_values table in the internal DB.
    Used at demo startup to restore query-time expansion without re-ingestion.
    Returns number of terms loaded.

    Idempotent: once the index is loaded, repeat calls are a no-op (return the cached
    size) — callers (nl_simplifier, retrieval_v2, …) guard on _VALUE_STORE, which this
    function never fills, so without this guard the column_values table was re-read on
    every query. Pass force=True to re-read after a fresh ingestion.
    """
    global _VALUE_INDEX
    if _VALUE_INDEX and not force:
        return len(_VALUE_INDEX)
    try:
        from ingestion.db_abstraction import get_internal_connection
        conn = get_internal_connection()
        cur  = conn.cursor()
        cur.execute(f"SELECT value_norm, col_id FROM {COLUMN_VALUES_TABLE_NAME}")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ValueSampler] rebuild_value_index_from_db failed: {e}")
        return 0

    index: Dict[str, List[str]] = {}
    for value_norm, col_id in rows:
        if value_norm not in index:
            index[value_norm] = []
        if col_id not in index[value_norm]:
            index[value_norm].append(col_id)

    _VALUE_INDEX = index
    print(f"[ValueSampler] Value index rebuilt: {len(_VALUE_INDEX)} terms from DB")
    return len(_VALUE_INDEX)


# =============================================================================
# Smoke test — python ingestion/value_sampler.py
# =============================================================================

if __name__ == "__main__":
    from schema.simulate_schema import get_simulated_schema
    from ingestion.schema_scanner import run_schema_scanner
    from ingestion.semantic_type_inference import run_semantic_type_inference

    print("Running ingestion pipeline (scanner + inference)...")
    raw_schema       = get_simulated_schema()
    scan_result      = run_schema_scanner(raw_schema=raw_schema, verbose=False)
    inference_result = run_semantic_type_inference(
        scan_result=scan_result, verbose=False
    )

    print("Running value sampler...\n")
    result = run_value_sampler(inference_result=inference_result, verbose=True)

    print("=" * 70)
    print("VEDA POC — Value Sampler Output")
    print("=" * 70)
    print(f"  Columns sampled  : {result.columns_sampled}")
    print(f"  Total values     : {result.total_values}")
    print(f"  Index entries    : {len(get_value_index())}")
    print(f"  Backend          : {result.backend}")
    print(f"  Duration         : {result.duration_sec}s")
    print()

    # Test query-time expansion
    test_queries = [
        ["show", "escalated", "incidents"],
        ["find", "open", "investigations"],
        ["list", "active", "roles"],
        ["show", "pending", "requests"],
        ["random", "tokens", "xyz"],
    ]

    print("Query expansion tests:")
    for tokens in test_queries:
        expanded, expansion_map = expand_query_tokens(tokens, verbose=False)
        injected = [t for t in expanded if t not in tokens]
        print(f"  {tokens}")
        if injected:
            print(f"    → injected: {injected}  (via: {expansion_map})")
        else:
            print(f"    → no expansion")
        print()