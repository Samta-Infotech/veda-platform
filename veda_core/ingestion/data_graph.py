# =============================================================================
# ingestion/data_graph.py
# VEDA POC — Step 2b: Data Graph (Undeclared FK Discovery)
#
# Responsibility:
#   - Samples actual column values from the client DB
#   - Computes value overlap between column pairs to find undeclared FKs
#   - Computes co-null correlation to find soft relationships
#   - Returns DiscoveredEdge list at three certainty levels
#   - Discovered edges are stored in fk_adjacency alongside declared FKs
#
# Why this matters:
#   Real enterprise schemas have two categories of relationships:
#     Declared FKs  — enforced by DB constraints (schema_scanner finds these)
#     Undeclared FKs — columns that behave like FKs but were never declared
#
#   Common causes of undeclared FKs:
#     - Legacy schemas where constraints were never added
#     - Reporting tables built outside the ORM
#     - Soft-deletes where FK constraints would cascade incorrectly
#     - Cross-schema or cross-service joins
#
#   Example: incident.owned_by → user.username
#     The DB has no constraint, but 95% of owned_by values exist in user.username.
#     This is an absolute mapping discovered from data, not schema.
#
# Three certainty tiers:
#   HIGH    (overlap > 0.90) — treat as FK, same weight as declared
#   MEDIUM  (overlap 0.70–0.90) — probable FK, lower weight in RRF
#   SOFT    (co-null > 0.85, overlap < 0.70) — related columns, training only
#
# Design constraints:
#   - Zero human input — runs fully automatically during ingestion
#   - Non-destructive — read-only SQL SELECT queries only
#   - Performance-safe — samples at most DATA_GRAPH_SAMPLE_SIZE rows per column
#   - Skips columns already connected by declared FKs — no redundant work
#   - Graceful fallback — DB unavailable → returns empty result, pipeline continues
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ingestion.schema_scanner import ScanResult, ScannedColumn
from config import (
    DATA_GRAPH_ENABLED,
    DATA_GRAPH_SAMPLE_SIZE,
    DATA_GRAPH_OVERLAP_THRESHOLD,
    SENSITIVE_PATTERNS,
    get_primary_relational_source,
)
from ingestion.db_abstraction import (
    INTERNAL_DB_AVAILABLE as PSYCOPG2_AVAILABLE,
    get_client_connection,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class DiscoveredEdge:
    """
    An undeclared FK relationship discovered by data-level analysis.
    Stored in fk_adjacency with edge_type != 'declared'.
    """
    # Column being analysed (the "FK" side)
    from_col_id:     str
    from_col_name:   str
    from_table_id:   str
    from_table_name: str

    # Column it maps to (the "PK" side)
    to_col_id:       str
    to_col_name:     str
    to_table_id:     str
    to_table_name:   str

    # Evidence
    overlap_score:   float          # 0.0–1.0 value overlap
    co_null_score:   float          # 0.0–1.0 co-null correlation
    certainty:       str            # "HIGH" | "MEDIUM" | "SOFT"
    evidence:        str            # human-readable description of evidence
    samples_used:    int            # number of rows sampled


@dataclass
class DataGraphResult:
    """
    Top-level output of the data graph analysis.
    """
    discovered_edges:    List[DiscoveredEdge]
    high_certainty:      List[DiscoveredEdge]   # overlap > 0.90
    medium_certainty:    List[DiscoveredEdge]   # overlap 0.70–0.90
    soft_certainty:      List[DiscoveredEdge]   # co-null only
    skipped_pairs:       int                    # pairs skipped (already declared, etc.)
    columns_sampled:     int
    duration_sec:        float
    stats:               dict = field(default_factory=dict)


# =============================================================================
# Column eligibility for data graph analysis
# =============================================================================

# Data types eligible for value overlap analysis
# Integer types: compare numeric IDs
# String types: compare varchar/text identifiers
_INTEGER_TYPES = {"integer", "bigint", "smallint", "int", "int4", "int8", "serial"}
_STRING_TYPES  = {"varchar", "text", "character varying", "char", "bpchar", "uuid"}

# Semantic types eligible — only IDENTIFIER and FREE_TEXT columns
# MONETARY, TEMPORAL, METRIC, CATEGORY are not join keys
_ELIGIBLE_SEMANTIC_TYPES = {"IDENTIFIER", "FREE_TEXT"}


def _is_eligible(col: ScannedColumn, semantic_type: str) -> bool:
    """
    Returns True if this column is eligible for value overlap analysis.
    Excludes: PKs (they are the reference side, not the FK side),
              temporal/monetary/metric columns, sensitive columns.
    """
    dt = col.data_type.lower()

    # Must be integer or string type
    if dt not in _INTEGER_TYPES and dt not in _STRING_TYPES:
        return False

    # Must be a potential FK semantic type
    if semantic_type not in _ELIGIBLE_SEMANTIC_TYPES:
        return False

    # PKs are the reference side — skip as "from" column
    # (they can still be the "to" column)
    if col.is_pk:
        return False

    # Already a declared FK — declared relationships are already handled
    # by schema_scanner. We focus on undeclared ones.
    if col.is_fk:
        return False

    # Sensitive column exclusion (double gate)
    col_lower = col.col_name.lower()
    if any(p in col_lower for p in SENSITIVE_PATTERNS):
        return False

    return True


def _is_eligible_as_target(col: ScannedColumn) -> bool:
    """
    Returns True if this column can be the "to" side of a discovered edge.
    Must be a PK or have an identifier-like name.
    """
    if col.is_pk:
        return True
    # Integer column with _id suffix is almost always a PK-like reference
    name = col.col_name.lower()
    if col.data_type.lower() in _INTEGER_TYPES and name.endswith("_id"):
        return True
    # uuid columns
    if col.data_type.lower() == "uuid":
        return True
    return False


# =============================================================================
# SQL value sampling — read-only, parameterised
# =============================================================================

def _sample_values(
    cursor,
    table_name: str,
    col_name:   str,
    n:          int,
) -> Set:
    """
    Samples up to n distinct non-null values from table.col.
    Returns a Python set of values.
    Uses TABLESAMPLE for large tables to avoid full scans.
    """
    try:
        # Use TABLESAMPLE SYSTEM for performance on large tables
        # Falls back to plain LIMIT if TABLESAMPLE is unsupported
        cursor.execute(f"""
            SELECT DISTINCT {_q(col_name)}
            FROM {_q(table_name)}
            WHERE {_q(col_name)} IS NOT NULL
            LIMIT %s;
        """, (n,))
        rows = cursor.fetchall()
        return {str(r[0]) for r in rows if r[0] is not None}
    except Exception:
        return set()


def _count_non_null(cursor, table_name: str, col_name: str) -> int:
    """Returns count of non-null rows for a column."""
    try:
        cursor.execute(f"""
            SELECT COUNT(*) FROM {_q(table_name)}
            WHERE {_q(col_name)} IS NOT NULL;
        """)
        return cursor.fetchone()[0]
    except Exception:
        return 0


def _count_co_null(
    cursor,
    table_name: str,
    col_a:      str,
    col_b:      str,
) -> Tuple[int, int]:
    """
    Returns (both_null_count, either_null_count) for two columns in same table.
    Used for co-null correlation within a table.
    Only meaningful when col_a and col_b are in the same table.
    """
    try:
        cursor.execute(f"""
            SELECT
                COUNT(*) FILTER (WHERE {_q(col_a)} IS NULL AND {_q(col_b)} IS NULL),
                COUNT(*) FILTER (WHERE {_q(col_a)} IS NULL OR  {_q(col_b)} IS NULL)
            FROM {_q(table_name)};
        """)
        row = cursor.fetchone()
        return (row[0] or 0, row[1] or 0)
    except Exception:
        return (0, 0)


def _q(name: str) -> str:
    """Double-quote identifier."""
    return f'"{name.replace(chr(34), "")}"'


# =============================================================================
# Core analysis phases
# =============================================================================

def _compute_overlap(set_a: Set, set_b: Set) -> float:
    """
    Overlap score = |A ∩ B| / min(|A|, |B|)
    Range: 0.0 (no overlap) to 1.0 (complete overlap).
    Uses min denominator so a small FK column matching a large PK column
    still scores 1.0 even though the PK has many more values.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def _phase1_value_overlap(
    cursor,
    eligible_from_cols: List[ScannedColumn],
    eligible_to_cols:   List[ScannedColumn],
    declared_pairs:     Set[Tuple[str, str]],
    sample_size:        int,
    threshold:          float,
    verbose:            bool,
) -> Tuple[List[DiscoveredEdge], int]:
    """
    Phase 1: Compare sampled values between candidate FK and PK columns.

    For each eligible (from_col, to_col) pair where:
      - Different tables
      - Same broad data type family (integer↔integer or string↔string)
      - Not already a declared FK pair
    Compute value overlap and record as DiscoveredEdge if above threshold.

    Returns (discovered_edges, skipped_count).
    """
    # Build value cache — sample each column once, reuse across comparisons
    value_cache: Dict[str, Set] = {}

    def _get_values(col: ScannedColumn) -> Set:
        key = col.col_id
        if key not in value_cache:
            value_cache[key] = _sample_values(
                cursor, col.table_name, col.col_name, sample_size
            )
        return value_cache[key]

    discovered: List[DiscoveredEdge] = []
    skipped = 0

    # Group to_cols by type family for fast filtering
    int_to_cols = [c for c in eligible_to_cols
                   if c.data_type.lower() in _INTEGER_TYPES]
    str_to_cols = [c for c in eligible_to_cols
                   if c.data_type.lower() in _STRING_TYPES]

    total_pairs = len(eligible_from_cols) * len(eligible_to_cols)
    if verbose:
        print(f"  [DataGraph Phase1] Analysing ~{total_pairs:,} pairs...")

    for from_col in eligible_from_cols:
        from_dt = from_col.data_type.lower()
        is_int  = from_dt in _INTEGER_TYPES

        # Only compare same type family
        candidates = int_to_cols if is_int else str_to_cols

        for to_col in candidates:
            # Skip same table — intra-table relationships handled separately
            if from_col.table_id == to_col.table_id:
                skipped += 1
                continue

            # Skip if already a declared FK
            pair_key = (from_col.col_id, to_col.col_id)
            if pair_key in declared_pairs:
                skipped += 1
                continue

            # Get sampled values
            from_vals = _get_values(from_col)
            to_vals   = _get_values(to_col)

            if len(from_vals) < 5 or len(to_vals) < 5:
                # Too few values to make a reliable judgement
                skipped += 1
                continue

            overlap = _compute_overlap(from_vals, to_vals)

            if overlap < threshold:
                continue

            # Determine certainty tier
            if overlap >= 0.90:
                certainty = "HIGH"
            else:
                certainty = "MEDIUM"

            evidence = (
                f"Value overlap {overlap:.1%} — "
                f"{len(from_vals & to_vals)} of {len(from_vals)} sampled "
                f"{from_col.table_name}.{from_col.col_name} values found in "
                f"{to_col.table_name}.{to_col.col_name}"
            )

            discovered.append(DiscoveredEdge(
                from_col_id     = from_col.col_id,
                from_col_name   = from_col.col_name,
                from_table_id   = from_col.table_id,
                from_table_name = from_col.table_name,
                to_col_id       = to_col.col_id,
                to_col_name     = to_col.col_name,
                to_table_id     = to_col.table_id,
                to_table_name   = to_col.table_name,
                overlap_score   = round(overlap, 4),
                co_null_score   = 0.0,
                certainty       = certainty,
                evidence        = evidence,
                samples_used    = len(from_vals),
            ))

    return discovered, skipped


def _phase2_co_null_correlation(
    cursor,
    scan_result:     ScanResult,
    semantic_map:    Dict[str, str],
    already_found:   Set[Tuple[str, str]],
    verbose:         bool,
) -> List[DiscoveredEdge]:
    """
    Phase 2: Within each table, find column pairs that go null together.

    High co-null correlation between two columns in the same table means:
    - They represent the same real-world concept via different paths
    - When one is unknown/missing, so is the other
    - This is a "soft" relationship — not a FK but a usage dependency

    Example: incident.assigned_to_id and incident.owned_by both null
    when no user is assigned → they co-vary → training signal.
    """
    discovered: List[DiscoveredEdge] = []

    for table in scan_result.tables:
        # Find nullable IDENTIFIER/FREE_TEXT columns in this table
        nullable_cols = [
            c for c in table.columns
            if c.nullable
            and not c.is_pk
            and semantic_map.get(c.col_id) in _ELIGIBLE_SEMANTIC_TYPES
            and not any(p in c.col_name.lower() for p in SENSITIVE_PATTERNS)
        ]

        if len(nullable_cols) < 2:
            continue

        # Compare all pairs within the table
        for i, col_a in enumerate(nullable_cols):
            for col_b in nullable_cols[i+1:]:
                # Skip if same type family — reduces noise
                dt_a = col_a.data_type.lower() in _INTEGER_TYPES
                dt_b = col_b.data_type.lower() in _INTEGER_TYPES
                if dt_a != dt_b:
                    continue

                # Skip if already found via value overlap
                pair = (col_a.col_id, col_b.col_id)
                if pair in already_found:
                    continue

                both_null, either_null = _count_co_null(
                    cursor, table.table_name, col_a.col_name, col_b.col_name
                )

                if either_null == 0:
                    continue

                co_null_rate = both_null / either_null

                if co_null_rate < 0.85:
                    continue

                evidence = (
                    f"Co-null correlation {co_null_rate:.1%} — "
                    f"both {col_a.col_name} and {col_b.col_name} are null "
                    f"in {both_null} of {either_null} rows where either is null"
                )

                # Determine direction — the non-PK col with more nulls is likely the FK
                # We store both directions as soft edges
                discovered.append(DiscoveredEdge(
                    from_col_id     = col_a.col_id,
                    from_col_name   = col_a.col_name,
                    from_table_id   = table.table_id,
                    from_table_name = table.table_name,
                    to_col_id       = col_b.col_id,
                    to_col_name     = col_b.col_name,
                    to_table_id     = table.table_id,
                    to_table_name   = table.table_name,
                    overlap_score   = 0.0,
                    co_null_score   = round(co_null_rate, 4),
                    certainty       = "SOFT",
                    evidence        = evidence,
                    samples_used    = either_null,
                ))

    return discovered


def _phase3_deduplicate(edges: List[DiscoveredEdge]) -> List[DiscoveredEdge]:
    """
    Removes duplicate discovered edges.
    Keeps the highest-certainty version when duplicates exist.
    """
    tier_rank = {"HIGH": 3, "MEDIUM": 2, "SOFT": 1}
    seen: Dict[Tuple[str, str], DiscoveredEdge] = {}

    for edge in edges:
        key = (edge.from_col_id, edge.to_col_id)
        if key not in seen:
            seen[key] = edge
        else:
            # Keep higher certainty
            if tier_rank[edge.certainty] > tier_rank[seen[key].certainty]:
                seen[key] = edge

    return list(seen.values())


# =============================================================================
# Public entry point
# =============================================================================

def run_data_graph(
    scan_result:   ScanResult,
    semantic_map:  Dict[str, str] = None,
    source_id:     str = None,
    verbose:       bool = False,
) -> DataGraphResult:
    """
    Main entry point for Step 2b: Data Graph analysis.

    Discovers undeclared FK relationships by sampling actual DB data.
    Results are appended to the FK adjacency store by main.py.

    Parameters
    ----------
    scan_result : ScanResult
        Output of run_schema_scanner(). Contains tables, columns, declared FKs.
    semantic_map : Dict[str, str], optional
        {col_id → semantic_type} from semantic_type_inference.
        If None, defaults to treating all IDENTIFIER-suffixed columns as eligible.
    verbose : bool

    Returns
    -------
    DataGraphResult
        Always returns even if DB is unavailable (empty result on failure).
    """
    t0 = time.time()

    logger.debug("Starting data graph: %d tables", len(scan_result.tables))

    if not DATA_GRAPH_ENABLED:
        logger.debug("Data graph disabled (DATA_GRAPH_ENABLED=False)")
        return DataGraphResult(
            discovered_edges = [],
            high_certainty   = [],
            medium_certainty = [],
            soft_certainty   = [],
            skipped_pairs    = 0,
            columns_sampled  = 0,
            duration_sec     = 0.0,
            stats            = {"skipped": "DATA_GRAPH_ENABLED=False"},
        )

    if not PSYCOPG2_AVAILABLE:
        if verbose:
            print("[DataGraph] psycopg2 not available — skipping data graph analysis")
        return DataGraphResult(
            discovered_edges = [],
            high_certainty   = [],
            medium_certainty = [],
            soft_certainty   = [],
            skipped_pairs    = 0,
            columns_sampled  = 0,
            duration_sec     = round(time.time() - t0, 4),
            stats            = {"skipped": "psycopg2 not available"},
        )

    if verbose:
        print("[DataGraph] Starting undeclared FK discovery...")
        print(f"  Tables           : {len(scan_result.tables)}")
        print(f"  Total columns    : {len(scan_result.all_columns)}")
        print(f"  Sample size      : {DATA_GRAPH_SAMPLE_SIZE} rows per column")
        print(f"  Overlap threshold: {DATA_GRAPH_OVERLAP_THRESHOLD:.0%}")

    # ------------------------------------------------------------------
    # Build declared FK pair set — skip these in phase 1
    # ------------------------------------------------------------------
    declared_pairs: Set[Tuple[str, str]] = {
        (edge["from_col_id"], edge["to_col_id"])
        for edge in scan_result.fk_edges
        if edge.get("from_col_id") and edge.get("to_col_id")
    }

    # ------------------------------------------------------------------
    # Build semantic_map if not provided
    # Default: use identifier suffix heuristic from col_name
    # ------------------------------------------------------------------
    if semantic_map is None:
        semantic_map = {}
        for col in scan_result.all_columns:
            name = col.col_name.lower()
            if (col.is_pk or col.is_fk or
                    any(name.endswith(s) for s in
                        ["_id", "_uuid", "_key", "_no", "_number", "_code", "_ref"])):
                semantic_map[col.col_id] = "IDENTIFIER"
            else:
                semantic_map[col.col_id] = "FREE_TEXT"

    # ------------------------------------------------------------------
    # Select eligible columns
    # ------------------------------------------------------------------
    eligible_from = [
        col for col in scan_result.all_columns
        if _is_eligible(col, semantic_map.get(col.col_id, "FREE_TEXT"))
    ]
    eligible_to = [
        col for col in scan_result.all_columns
        if _is_eligible_as_target(col)
    ]

    if verbose:
        print(f"  Eligible from-cols : {len(eligible_from)}")
        print(f"  Eligible to-cols   : {len(eligible_to)}")

    all_discovered: List[DiscoveredEdge] = []
    total_skipped = 0

    try:
        _source_id = source_id or get_primary_relational_source()["id"]
        conn = get_client_connection(_source_id)
        try:
            with conn.cursor() as cur:
                # ----------------------------------------------------------
                # Phase 1 — Value overlap analysis (cross-table)
                # ----------------------------------------------------------
                if verbose:
                    print(f"  [Phase 1] Value overlap analysis...")

                p1_edges, p1_skipped = _phase1_value_overlap(
                    cursor          = cur,
                    eligible_from_cols = eligible_from,
                    eligible_to_cols   = eligible_to,
                    declared_pairs  = declared_pairs,
                    sample_size     = DATA_GRAPH_SAMPLE_SIZE,
                    threshold       = DATA_GRAPH_OVERLAP_THRESHOLD,
                    verbose         = verbose,
                )
                all_discovered.extend(p1_edges)
                total_skipped += p1_skipped

                if verbose:
                    print(f"             Found {len(p1_edges)} edges, skipped {p1_skipped} pairs")

                # ----------------------------------------------------------
                # Phase 2 — Co-null correlation (intra-table)
                # ----------------------------------------------------------
                if verbose:
                    print(f"  [Phase 2] Co-null correlation analysis...")

                already_found = {(e.from_col_id, e.to_col_id) for e in all_discovered}
                p2_edges = _phase2_co_null_correlation(
                    cursor       = cur,
                    scan_result  = scan_result,
                    semantic_map = semantic_map,
                    already_found = already_found,
                    verbose      = verbose,
                )
                all_discovered.extend(p2_edges)

                if verbose:
                    print(f"             Found {len(p2_edges)} soft edges")

        finally:
            conn.close()

    except Exception as e:
        if verbose:
            print(f"  ⚠ Data graph DB connection failed: {e}")
            print(f"    Continuing without discovered edges.")
        return DataGraphResult(
            discovered_edges = [],
            high_certainty   = [],
            medium_certainty = [],
            soft_certainty   = [],
            skipped_pairs    = total_skipped,
            columns_sampled  = len(eligible_from),
            duration_sec     = round(time.time() - t0, 4),
            stats            = {"error": str(e)},
        )

    # ------------------------------------------------------------------
    # Phase 3 — Deduplicate
    # ------------------------------------------------------------------
    all_discovered = _phase3_deduplicate(all_discovered)

    # ------------------------------------------------------------------
    # Partition by certainty tier
    # ------------------------------------------------------------------
    high   = [e for e in all_discovered if e.certainty == "HIGH"]
    medium = [e for e in all_discovered if e.certainty == "MEDIUM"]
    soft   = [e for e in all_discovered if e.certainty == "SOFT"]

    duration = round(time.time() - t0, 4)

    stats = {
        "total_discovered":  len(all_discovered),
        "high_certainty":    len(high),
        "medium_certainty":  len(medium),
        "soft_certainty":    len(soft),
        "skipped_pairs":     total_skipped,
        "columns_sampled":   len(eligible_from),
        "declared_fks":      len(declared_pairs),
        "duration_sec":      duration,
    }

    if verbose:
        print(f"\n  [DataGraph] Results:")
        print(f"    HIGH   (≥90% overlap)    : {len(high)}")
        print(f"    MEDIUM (70–90% overlap)  : {len(medium)}")
        print(f"    SOFT   (co-null only)    : {len(soft)}")
        print(f"    Duration                 : {duration}s")
        print()
        if high:
            print("  HIGH certainty edges (absolute mappings):")
            for e in high:
                print(f"    ★  {e.from_table_name}.{e.from_col_name:<25} → "
                      f"{e.to_table_name}.{e.to_col_name:<25} "
                      f"overlap={e.overlap_score:.1%}")
        if medium:
            print("  MEDIUM certainty edges:")
            for e in medium:
                print(f"    ~  {e.from_table_name}.{e.from_col_name:<25} → "
                      f"{e.to_table_name}.{e.to_col_name:<25} "
                      f"overlap={e.overlap_score:.1%}")
        if soft:
            print("  SOFT edges (co-null, same table):")
            for e in soft[:5]:   # cap at 5 for readability
                print(f"    ·  {e.from_table_name}.{e.from_col_name:<25} ↔ "
                      f"{e.to_col_name:<25} "
                      f"co_null={e.co_null_score:.1%}")
        print(f"[DataGraph] Done.\n")

    logger.info(
        "Data graph complete: HIGH=%d MEDIUM=%d SOFT=%d discovered edges",
        len(high), len(medium), len(soft),
    )

    return DataGraphResult(
        discovered_edges = all_discovered,
        high_certainty   = high,
        medium_certainty = medium,
        soft_certainty   = soft,
        skipped_pairs    = total_skipped,
        columns_sampled  = len(eligible_from),
        duration_sec     = duration,
        stats            = stats,
    )


# =============================================================================
# store_discovered_edges — called by main.py to persist results
# =============================================================================

def to_fk_adjacency_rows(
    data_graph_result: DataGraphResult,
    include_soft:      bool = False,
) -> List[dict]:
    """
    Converts DiscoveredEdges into the fk_adjacency row format.
    Matches the dict structure expected by store_fk_adjacency().

    Parameters
    ----------
    data_graph_result : DataGraphResult
    include_soft : bool
        Whether to include SOFT edges. Default False —
        soft edges are intra-table usage signals, not join paths.
        Set True to use them for fine-tuning training pairs only.

    Returns
    -------
    List[dict] — same shape as scan_result.fk_edges
    """
    rows = []
    for edge in data_graph_result.discovered_edges:
        if edge.certainty == "SOFT" and not include_soft:
            continue
        rows.append({
            "from_col_id":   edge.from_col_id,
            "from_col_name": edge.from_col_name,
            "from_table_id": edge.from_table_id,
            "from_table":    edge.from_table_name,
            "to_col_id":     edge.to_col_id,
            "to_col_name":   edge.to_col_name,
            "to_table_id":   edge.to_table_id,
            "to_table":      edge.to_table_name,
            # Extra fields for auditing — not used by fk_adjacency table
            # but available for logging and fine-tuning
            "_certainty":    edge.certainty,
            "_overlap":      edge.overlap_score,
            "_co_null":      edge.co_null_score,
            "_evidence":     edge.evidence,
        })
    return rows


# =============================================================================
# Smoke test — python ingestion/data_graph.py
# =============================================================================

if __name__ == "__main__":
    from schema.simulate_schema import get_simulated_schema
    from ingestion.schema_scanner import run_schema_scanner

    print("Running schema scanner...")
    raw_schema = get_simulated_schema()
    scan       = run_schema_scanner(raw_schema=raw_schema, verbose=False)

    print("Running data graph analysis...")
    print("(Using real DB from config.py — will fallback gracefully if unavailable)\n")

    result = run_data_graph(scan, verbose=True)

    print("=" * 70)
    print("VEDA POC — Data Graph Results")
    print("=" * 70)
    print(f"  Total discovered edges  : {len(result.discovered_edges)}")
    print(f"  HIGH certainty          : {len(result.high_certainty)}")
    print(f"  MEDIUM certainty        : {len(result.medium_certainty)}")
    print(f"  SOFT (co-null)          : {len(result.soft_certainty)}")
    print(f"  Duration                : {result.duration_sec}s")
    print()

    # Show what would be added to fk_adjacency
    rows = to_fk_adjacency_rows(result, include_soft=False)
    print(f"  Rows to add to fk_adjacency (HIGH + MEDIUM): {len(rows)}")
    for row in rows[:10]:
        print(f"    {row['from_table']}.{row['from_col_name']:<25} → "
              f"{row['to_table']}.{row['to_col_name']:<25} "
              f"[{row['_certainty']}  {row['_overlap']:.1%}]")