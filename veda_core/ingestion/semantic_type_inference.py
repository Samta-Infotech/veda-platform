# =============================================================================
# ingestion/semantic_type_inference.py
# VEDA POC — Step 2: Semantic Type Inference
#
# Responsibility:
#   - Accepts ScanResult from schema_scanner.py
#   - Assigns one of six semantic types to every column
#   - Uses a three-layer rule engine (data type → name pattern → cardinality)
#   - Attaches a confidence score to every assignment
#   - Flags low-confidence columns for optional human review
#   - Identifies the PRIMARY DISPLAY COLUMN per table (post-processing pass)
#   - Produces a list of TypedColumn consumed by reg_builder.py
#
# Semantic types: MONETARY | TEMPORAL | CATEGORY | IDENTIFIER | METRIC | FREE_TEXT
#
# Rule priority (matches architecture document exactly):
#   Layer A — data type rules     (highest confidence)
#   Layer B — column name pattern matching via regex
#   Layer C — cardinality sampling fallback (lowest confidence)
#
# Primary Display Column (post-processing):
#   Every table has one human-readable identifier column users expect in results.
#   e.g. incident_no, order_number, username, role_name, tracking_code.
#   Identified by a three-layer rule — no domain knowledge required.
#   Tagged as is_display_col=True on TypedColumn and stored in table_metadata.
# =============================================================================

import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ingestion.schema_scanner import ScanResult, ScannedColumn, run_schema_scanner
from config import (
    SEMANTIC_TYPES,
    SEMANTIC_CONFIDENCE_THRESHOLD,
    MONETARY_KEYWORDS,
    METRIC_KEYWORDS,
    IDENTIFIER_SUFFIXES,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class TypedColumn:
    """
    A ScannedColumn enriched with semantic type and confidence score.
    Passed to reg_builder.py as the unit of graph node data.
    """
    # --- identity (carried forward from ScannedColumn) ---
    col_id:           str
    col_name:         str
    table_id:         str
    table_name:       str
    data_type:        str
    is_pk:            bool
    is_fk:            bool
    fk_ref_table_id:  Optional[str]
    fk_ref_table:     Optional[str]
    fk_ref_col:       Optional[str]
    nullable:         bool
    cardinality:      Optional[int]

    # --- semantic inference results ---
    semantic_type:    str            # one of SEMANTIC_TYPES
    confidence:       float          # 0.0 – 1.0
    inference_layer:  str            # "A" | "B" | "C"  — which layer assigned it
    flagged:          bool           # True if confidence < SEMANTIC_CONFIDENCE_THRESHOLD
    rule_matched:     str            # human-readable description of matched rule

    # --- display column tag ---
    # True for exactly one column per table — the human-readable identifier.
    # Set by _tag_display_columns() post-processing pass, not per-column inference.
    # Stored in table_metadata by vector_store.py for query-time injection.
    is_display_col:   bool = False


@dataclass
class InferenceResult:
    """
    Top-level output of semantic type inference.
    Consumed directly by reg_builder.py.
    """
    typed_columns:    List[TypedColumn]
    flagged:          List[TypedColumn]   # subset with confidence below threshold
    display_col_map:  dict = field(default_factory=dict)
    # {table_id → (col_id, col_name, table_name)} — one entry per table
    # that has an identifiable display column. Passed to vector_store.py
    # to populate the table_metadata store.
    stats:            dict = field(default_factory=dict)


# =============================================================================
# Layer A — Data type rules
# Highest priority. If a data type unambiguously maps to a semantic type,
# assign it here with high confidence and skip Layers B and C.
# =============================================================================

def _layer_a(col: ScannedColumn) -> Optional[Tuple[str, float, str]]:
    """
    Returns (semantic_type, confidence, rule_matched) or None if no match.
    """
    dt = col.data_type.lower()

    # timestamp / date / timestamptz / time → TEMPORAL
    if dt in ("timestamp", "date", "timestamptz", "time"):
        return ("TEMPORAL", 0.99, f"data_type='{dt}' → TEMPORAL")

    # boolean → CATEGORY (only 2 distinct values)
    if dt == "boolean":
        return ("CATEGORY", 0.99, f"data_type='boolean' → CATEGORY")

    # uuid → always an IDENTIFIER
    if dt == "uuid":
        return ("IDENTIFIER", 0.97, "data_type='uuid' → IDENTIFIER")

    # inet (IP address) → IDENTIFIER
    if dt == "inet":
        return ("IDENTIFIER", 0.80, "data_type='inet' → IDENTIFIER")

    # integer / bigint / smallint: PK or FK → IDENTIFIER, otherwise continue to B/C
    if dt in ("integer", "bigint", "smallint"):
        if col.is_pk:
            return ("IDENTIFIER", 0.97, f"data_type='{dt}' + is_pk=True → IDENTIFIER")
        if col.is_fk:
            return ("IDENTIFIER", 0.95, f"data_type='{dt}' + is_fk=True → IDENTIFIER")

    # double precision → always a numeric METRIC
    if dt == "double":
        return ("METRIC", 0.85, "data_type='double' → METRIC")

    # jsonb → structured blob, treat as FREE_TEXT
    if dt == "jsonb":
        return ("FREE_TEXT", 0.60, "data_type='jsonb' → FREE_TEXT")

    # text / character → FREE_TEXT
    if dt in ("character", "text"):
        return ("FREE_TEXT", 0.75, f"data_type='{dt}' → FREE_TEXT")

    return None


# =============================================================================
# Layer B — Column name pattern matching
# Regex-based. Uses keyword lists from config.py.
# Runs only if Layer A returned None.
# =============================================================================

# Pre-compile patterns for performance
_MONETARY_PATTERN  = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in MONETARY_KEYWORDS) + r")",
    re.IGNORECASE
)
_METRIC_PATTERN    = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in METRIC_KEYWORDS) + r")",
    re.IGNORECASE
)
_IDENTIFIER_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(s) for s in IDENTIFIER_SUFFIXES) + r")$",
    re.IGNORECASE
)
_CATEGORY_PATTERN = re.compile(
    r"(?:type|kind|category|state|condition|status|active|inactive|pending|approved|rejected)",
    re.IGNORECASE
)


def _layer_b(col: ScannedColumn) -> Optional[Tuple[str, float, str]]:
    """
    Returns (semantic_type, confidence, rule_matched) or None if no match.
    """
    name = col.col_name.lower()
    dt   = col.data_type.lower()

    # MONETARY: numeric + name contains monetary keyword
    if dt in ("numeric", "integer", "bigint", "smallint") and _MONETARY_PATTERN.search(name):
        matched = _MONETARY_PATTERN.search(name).group()
        return ("MONETARY", 0.92, f"data_type='{dt}' + name contains '{matched}' → MONETARY")

    # METRIC: numeric + name contains metric keyword
    if dt in ("numeric", "integer", "bigint", "smallint") and _METRIC_PATTERN.search(name):
        matched = _METRIC_PATTERN.search(name).group()
        return ("METRIC", 0.90, f"data_type='{dt}' + name contains '{matched}' → METRIC")

    # IDENTIFIER: name ends with _id / _uuid / _key suffix
    if _IDENTIFIER_PATTERN.search(name):
        return ("IDENTIFIER", 0.88, f"col_name ends with identifier suffix → IDENTIFIER")

    # CATEGORY: varchar/character varying with name match
    if dt in ("varchar", "character varying", "text", "character") and _CATEGORY_PATTERN.search(name):
        matched = _CATEGORY_PATTERN.search(name).group()
        return ("CATEGORY", 0.90, f"data_type='{dt}' + name contains '{matched}' → CATEGORY")

    # CATEGORY: varchar/character varying with low cardinality (< 100 distinct values)
    if dt in ("varchar", "character varying") and col.cardinality is not None and col.cardinality < 100:
        return ("CATEGORY", 0.85, f"data_type='{dt}' + cardinality={col.cardinality} < 100 → CATEGORY")

    return None


# =============================================================================
# Layer C — Cardinality sampling fallback
# Lowest priority. Runs only if Layers A and B both returned None.
# =============================================================================

_CATEGORY_CARDINALITY_THRESHOLD = 100


def _layer_c(col: ScannedColumn) -> Tuple[str, float, str]:
    """
    Always returns a result — this is the final fallback.
    Lower confidence reflects less certainty.
    """
    dt = col.data_type.lower()

    # varchar/character varying with known high cardinality → FREE_TEXT
    if dt in ("varchar", "character varying"):
        if col.cardinality is not None and col.cardinality >= _CATEGORY_CARDINALITY_THRESHOLD:
            return (
                "FREE_TEXT",
                0.78,
                f"data_type='{dt}' + cardinality={col.cardinality} >= 100 → FREE_TEXT"
            )
        # varchar with unknown cardinality — assume FREE_TEXT with lower confidence
        if col.cardinality is None:
            return (
                "FREE_TEXT",
                0.60,
                f"data_type='{dt}' + cardinality=None → FREE_TEXT (low confidence)"
            )
        # varchar with cardinality < 100 (should have been caught in Layer B)
        return (
            "CATEGORY",
            0.70,
            f"data_type='{dt}' + cardinality={col.cardinality} < 100 → CATEGORY (fallback)"
        )

    # numeric / integer / bigint / smallint with no name match → METRIC
    if dt in ("numeric", "integer", "bigint", "smallint"):
        return (
            "METRIC",
            0.55,
            f"data_type='{dt}' + no name match → METRIC (low confidence fallback)"
        )

    # anything else — FREE_TEXT as last resort (raised from 0.40 → 0.50)
    return (
        "FREE_TEXT",
        0.50,
        f"data_type='{dt}' → FREE_TEXT (last resort fallback)"
    )


# =============================================================================
# Per-column inference orchestrator
# =============================================================================

def _infer_column(col: ScannedColumn) -> TypedColumn:
    """
    Runs the three-layer rule engine for a single column.
    Returns a fully populated TypedColumn.
    """
    result = _layer_a(col)
    layer  = "A"

    if result is None:
        result = _layer_b(col)
        layer  = "B"

    if result is None:
        result = _layer_c(col)
        layer  = "C"

    semantic_type, confidence, rule_matched = result

    # Validate semantic_type is one of the six allowed values
    assert semantic_type in SEMANTIC_TYPES, (
        f"Unexpected semantic_type '{semantic_type}' for col '{col.col_name}'"
    )

    flagged = confidence < SEMANTIC_CONFIDENCE_THRESHOLD

    return TypedColumn(
        # carry forward all ScannedColumn fields
        col_id          = col.col_id,
        col_name        = col.col_name,
        table_id        = col.table_id,
        table_name      = col.table_name,
        data_type       = col.data_type,
        is_pk           = col.is_pk,
        is_fk           = col.is_fk,
        fk_ref_table_id = col.fk_ref_table_id,
        fk_ref_table    = col.fk_ref_table,
        fk_ref_col      = col.fk_ref_col,
        nullable        = col.nullable,
        cardinality     = col.cardinality,
        # semantic inference results
        semantic_type   = semantic_type,
        confidence      = confidence,
        inference_layer = layer,
        flagged         = flagged,
        rule_matched    = rule_matched,
    )



# =============================================================================
# Primary Display Column Inference
#
# Post-processing pass run AFTER per-column semantic typing.
# For each table, identifies the single column that humans use to identify
# rows — the display identifier. Examples across any domain:
#
#   incident     → incident_no
#   order        → order_number
#   user         → username
#   role         → role_name
#   shipment     → tracking_code
#   patient      → mrn
#   workflow     → name
#
# Three-layer rule (generic — zero domain knowledge required):
#
#   Layer 1 — Name suffix patterns (highest priority)
#     Matches: *_no, *_number, *_num, *_code, *_ref, *_key (non-FK)
#     Rationale: These suffixes universally indicate human-readable IDs
#     across all industries and ORMs.
#
#   Layer 2 — Name equality patterns
#     Matches: name, title, label (standalone, not as suffix)
#     Rationale: Tables named "role", "workflow", "state" typically use
#     "name" as their display field.
#
#   Layer 3 — First non-PK varchar column (fallback)
#     Rationale: Schema convention in most ORMs — PK first, then the
#     most important descriptive column. Lowest confidence but covers
#     tables that don't follow naming conventions.
#
# Only one column per table is tagged. PKs are excluded — they are system
# identifiers, not display identifiers.
# =============================================================================

# Pre-compiled display column patterns
_DISPLAY_SUFFIX_PATTERN = re.compile(
    r'(?:_no|_number|_num|_code|_ref|_key|_name|_title|_label)$',
    re.IGNORECASE,
)
_DISPLAY_EXACT_PATTERN = re.compile(
    r'^(?:name|title|label|code|reference|number|description)$',
    re.IGNORECASE,
)


def _infer_display_column(
    table_columns: List["TypedColumn"],
) -> Optional[str]:
    """
    Returns the col_id of the primary display column for a table,
    or None if no suitable column is found.

    Applies a three-layer rule in priority order:
      Layer 1 — suffix pattern match (_no, _number, _code, _ref, _key, _name, _title)
      Layer 2 — exact name match    (name, title, label, code, reference)
      Layer 3 — first non-PK varchar column (fallback)

    PKs and FK columns are excluded from all layers —
    they are system identifiers, not display identifiers.
    """
    # Exclude PKs and FKs — not display identifiers
    candidates = [
        tc for tc in table_columns
        if not tc.is_pk and not tc.is_fk
    ]
    if not candidates:
        return None

    # --- Layer 1: suffix pattern ---
    for tc in candidates:
        if _DISPLAY_SUFFIX_PATTERN.search(tc.col_name.lower()):
            return tc.col_id

    # --- Layer 2: exact name match ---
    for tc in candidates:
        if _DISPLAY_EXACT_PATTERN.match(tc.col_name.lower()):
            return tc.col_id

    # --- Layer 3: first non-PK varchar fallback ---
    for tc in candidates:
        if tc.data_type.lower() in ("varchar", "text", "character varying"):
            return tc.col_id

    return None


def _tag_display_columns(typed_columns: List["TypedColumn"]) -> dict:
    """
    Post-processing pass. Groups columns by table, runs _infer_display_column
    per table, tags the winning column with is_display_col=True.

    Returns display_col_map: {table_id → (col_id, col_name, table_name)}
    This map is consumed by vector_store.py to populate table_metadata.
    """
    # Group by table_id
    by_table: dict = {}
    for tc in typed_columns:
        by_table.setdefault(tc.table_id, []).append(tc)

    display_col_map: dict = {}

    for table_id, cols in by_table.items():
        winner_id = _infer_display_column(cols)
        if winner_id is None:
            continue
        for tc in cols:
            if tc.col_id == winner_id:
                tc.is_display_col = True
                display_col_map[table_id] = (
                    tc.col_id,
                    tc.col_name,
                    tc.table_name,
                )
                break

    return display_col_map


# =============================================================================
# Public entry point
# =============================================================================

def run_semantic_type_inference(
    scan_result: ScanResult = None,
    verbose: bool = False,
) -> InferenceResult:
    """
    Main entry point for Step 2.

    Parameters
    ----------
    scan_result : ScanResult, optional
        Output of run_schema_scanner(). If None, runs the scanner internally.
    verbose : bool
        Print progress and per-type breakdown to stdout if True.

    Returns
    -------
    InferenceResult
        All columns with semantic types, confidence scores, and flagged list.
    """
    if scan_result is None:
        scan_result = run_schema_scanner(verbose=verbose)

    logger.debug("Starting semantic type inference: %d columns", len(scan_result.all_columns))

    if verbose:
        print("[SemanticTypeInference] Starting inference...")
        print(f"  Columns to process : {len(scan_result.all_columns)}")

    typed_columns = [_infer_column(col) for col in scan_result.all_columns]
    flagged       = [c for c in typed_columns if c.flagged]

    # Post-processing: tag primary display column per table
    display_col_map = _tag_display_columns(typed_columns)

    # --- build stats ---
    type_counts = {t: 0 for t in SEMANTIC_TYPES}
    layer_counts = {"A": 0, "B": 0, "C": 0}
    for tc in typed_columns:
        type_counts[tc.semantic_type]   += 1
        layer_counts[tc.inference_layer] += 1

    avg_confidence = (
        sum(c.confidence for c in typed_columns) / len(typed_columns)
        if typed_columns else 0.0
    )

    stats = {
        "total_columns":        len(typed_columns),
        "flagged_count":         len(flagged),
        "avg_confidence":        round(avg_confidence, 4),
        "type_counts":           type_counts,
        "layer_counts":          layer_counts,
        "display_cols_found":    len(display_col_map),
        "tables_without_display": len(
            {tc.table_id for tc in typed_columns} - set(display_col_map.keys())
        ),
    }

    if verbose:
        print(f"  Avg confidence     : {stats['avg_confidence']}")
        print(f"  Flagged columns    : {stats['flagged_count']}")
        print(f"  Layer A assigned   : {layer_counts['A']}")
        print(f"  Layer B assigned   : {layer_counts['B']}")
        print(f"  Layer C assigned   : {layer_counts['C']}")
        print(f"  Type breakdown     :")
        for stype, count in type_counts.items():
            print(f"    {stype:<12} : {count}")
        print(f"  Display cols found : {stats['display_cols_found']}")
        print(f"  Tables without     : {stats['tables_without_display']}")
        print("[SemanticTypeInference] Done.\n")

    logger.info(
        "Semantic type inference complete: %d columns, flagged=%d, avg_conf=%.4f | "
        "MONETARY=%d TEMPORAL=%d CATEGORY=%d IDENTIFIER=%d METRIC=%d FREE_TEXT=%d",
        stats["total_columns"], stats["flagged_count"], stats["avg_confidence"],
        stats["type_counts"].get("MONETARY", 0),
        stats["type_counts"].get("TEMPORAL", 0),
        stats["type_counts"].get("CATEGORY", 0),
        stats["type_counts"].get("IDENTIFIER", 0),
        stats["type_counts"].get("METRIC", 0),
        stats["type_counts"].get("FREE_TEXT", 0),
    )

    return InferenceResult(
        typed_columns   = typed_columns,
        flagged         = flagged,
        display_col_map = display_col_map,
        stats           = stats,
    )


# =============================================================================
# Smoke test — python ingestion/semantic_type_inference.py
# =============================================================================

if __name__ == "__main__":
    result = run_semantic_type_inference(verbose=True)

    print("=" * 70)
    print("VEDA POC — Semantic Type Inference Output")
    print("=" * 70)

    # Print all columns grouped by table
    current_table = None
    for tc in result.typed_columns:
        if tc.table_name != current_table:
            current_table = tc.table_name
            print(f"\n  [{current_table}]")
        flag_marker = " ⚠ FLAGGED" if tc.flagged else ""
        print(
            f"    {tc.col_name:<28} "
            f"{tc.semantic_type:<12} "
            f"conf={tc.confidence:.2f}  "
            f"L{tc.inference_layer}  "
            f"{flag_marker}"
        )

    # Print display columns
    print(f"\n{'=' * 70}")
    print(f"Primary Display Columns ({len(result.display_col_map)} tables):")
    for table_id, (col_id, col_name, table_name) in result.display_col_map.items():
        print(f"  ★  {table_name:<30} display_col = {col_name}")

    # Print flagged columns separately
    if result.flagged:
        print(f"\n{'=' * 70}")
        print(f"Flagged for review ({len(result.flagged)} columns):")
        for tc in result.flagged:
            print(
                f"  ⚠  {tc.table_name}.{tc.col_name:<28} "
                f"conf={tc.confidence:.2f}  rule: {tc.rule_matched}"
            )