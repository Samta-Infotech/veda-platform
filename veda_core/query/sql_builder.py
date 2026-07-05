# =============================================================================
# query/sql_builder.py
# VEDA POC — Query Pipeline Layer 4: SQL Builder
#
# Responsibility:
#   Converts the IR JSON (output of L3 SLM) into a parameterised SQL string.
#   Deterministic: same IR input always produces the same SQL output.
#   No string interpolation — all values passed as psycopg2 %s parameters.
#   Table and column names are always double-quoted PostgreSQL identifiers.
#
# Input  : ir_json (SLMResult.ir_json) + top_k_columns (L2, for UUID→name)
# Output : SQLBuilderResult — sql string + params tuple
#
# Supports:
#   SELECT, COUNT, AGGREGATE
#   INNER / LEFT JOINs
#   WHERE filter_tree — AND/OR combinators, all operators
#   GROUP BY, ORDER BY, LIMIT
#   NOT_EXISTS subqueries (recursive, depth-limited)
#
# Resilience:
#   When L3 produces entity UUIDs that don't resolve against top_k_columns
#   (UUID grounding failure), the builder falls back to the most relevant
#   table inferred from top_k_columns rather than returning an error.
#   This makes L4 robust to L3 UUID hallucinations on any schema.
#
# Security constraints (from architecture doc):
#   - No string interpolation anywhere in SQL construction
#   - All user-supplied values go through %s parameterisation
#   - All identifiers double-quoted — prevents SQL injection via schema names
# =============================================================================

import sys
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Matches SLM-hallucinated placeholder values such as <literal>, <value>, <placeholder>.
# These appear when L3 cannot determine a concrete filter value and must not be
# emitted as SQL parameters — they would cause a type-mismatch error at execution time.
_PLACEHOLDER_RE = re.compile(r"^<[^>]+>$")

# Detects ISO-8601 date/datetime strings (used for temporal-mismatch detection).
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Operators that imply the filter column must hold a date/timestamp value.
_DATE_RANGE_OPERATORS: frozenset[str] = frozenset({"BETWEEN", "GTE", "LTE", "GT", "LT"})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.vector_store import RetrievalResult
from config import SQL_DEFAULT_LIMIT, SQL_MAX_SUBQUERY_DEPTH
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output structure
# =============================================================================

@dataclass
class SQLBuilderResult:
    sql:         str            # final SQL with %s placeholders
    params:      tuple          # positional values in order of appearance
    query_type:  str            # SELECT | COUNT | AGGREGATE
    tables_used: List[str]      # table names in appearance order
    error:       Optional[str]  # set on hard failure; sql = "" in that case
    warnings:    List[str]      = field(default_factory=list)
    duration_ms: float          = 0.0


# =============================================================================
# UUID resolution
# =============================================================================

def _build_uuid_maps(
    top_k_columns: List[RetrievalResult],
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Returns:
      col_map   : {col_id  → {col_name, table_id, table_name}}
      table_map : {table_id → table_name}

    Built from top_k_columns — the only UUID source L4 needs.
    """
    col_map:   Dict[str, Dict] = {}
    table_map: Dict[str, str]  = {}
    for r in top_k_columns:
        col_map[r.col_id] = {
            "col_name":     r.col_name,
            "table_id":     r.table_id,
            "table_name":   r.table_name,
            "semantic_type": r.semantic_type,
        }
        table_map[r.table_id] = r.table_name
    return col_map, table_map


def _resolve_ir_entities(
    ir_json:   Dict[str, Any],
    col_map:   Dict[str, Dict],
    table_map: Dict[str, str],
) -> None:
    """Mutate IR entities in-place: resolve table_name→table_id and col_name→col_id
    when the SLM emits NAMES instead of UUIDs (it sometimes disobeys the UUID-only
    prompt). Without this, an entity like {"table_name": "incident", "columns":
    [{"col_name": "decision_reason"}]} is unresolvable by downstream code that reads
    only table_id/col_id, silently degrading to SELECT *."""
    name_to_tid = {v: k for k, v in table_map.items()}
    for ent in ir_json.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        tid = ent.get("table_id", "") or ""
        if not tid:
            tname = ent.get("table_name", "") or ""
            tid = name_to_tid.get(tname, "") if tname else ""
            ent["table_id"] = tid
        for col in ent.get("columns") or []:
            if not isinstance(col, dict):
                continue
            cid = col.get("col_id", "") or ""
            if cid and cid in col_map:
                continue
            cname = col.get("col_name", "") or ""
            if cname and tid:
                col["col_id"] = next(
                    (c_id for c_id, info in col_map.items()
                     if info["col_name"] == cname and info["table_id"] == tid),
                    cid,
                )


def _infer_primary_table(
    top_k_columns: List[RetrievalResult],
) -> Optional[Tuple[str, str]]:
    """
    Fallback: when IR entities have unresolvable UUIDs, infer the most
    relevant table from top_k_columns by majority vote on table_id.
    Returns (table_id, table_name) of the most frequent table, or None.

    This makes L4 resilient to L3 UUID grounding failures without
    any schema-specific knowledge — pure structural fallback.
    """
    if not top_k_columns:
        return None
    counts: Dict[str, float] = {}
    names:  Dict[str, str] = {}
    for r in top_k_columns:
        if r.table_id and r.similarity > 0.0:  # exclude injected bridge cols
            counts[r.table_id] = counts.get(r.table_id, 0.0) + r.similarity
            names[r.table_id]  = r.table_name
    if not counts:
        return None
    best_tid = max(counts, key=counts.__getitem__)
    return best_tid, names[best_tid]


def _q(name: str) -> str:
    """Double-quote a PostgreSQL identifier. Strips any existing quotes first."""
    return f'"{name.replace(chr(34), "")}"'


def _remap_col_to_joined_table(
    col_id: str,
    col_map: Dict[str, Dict],
    joined_tids: set,
) -> Optional[str]:
    """Search col_map for a same-named column that belongs to a joined table.

    When a filter col_id resolves to an unjoined table, L4 calls this before
    discarding the condition.  A unique match means L3 grabbed the right
    *semantic concept* (e.g. ``created_datetime``) but from the wrong table
    (e.g. ``audit_log`` instead of ``incident``).  Remapping preserves the
    filter rather than silently dropping it.

    Args:
        col_id:      Original col_id that resolved to an unjoined table.
        col_map:     Full col_id → {col_name, table_id, table_name} map.
        joined_tids: Set of table_ids currently in the FROM/JOIN chain.

    Returns:
        A col_id from a joined table whose col_name matches exactly, or None
        when zero or multiple candidates are found (ambiguous → skip).
    """
    original_info = col_map.get(col_id, {})
    target_col_name = original_info.get("col_name", "")
    if not target_col_name:
        return None

    candidates = [
        cid
        for cid, info in col_map.items()
        if info.get("col_name") == target_col_name
        and info.get("table_id") in joined_tids
        and cid != col_id
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _is_datetime_value(value: Any) -> bool:
    """Return True when value looks like an ISO-8601 date/datetime string.

    Accepts a bare string or a list/tuple where at least one element matches.

    Args:
        value: The filter value from the IR node.

    Returns:
        True if value appears to encode a date or datetime.
    """
    if isinstance(value, str):
        return bool(_DATETIME_RE.match(value.strip()))
    if isinstance(value, (list, tuple)):
        return any(
            isinstance(v, str) and _DATETIME_RE.match(v.strip())
            for v in value
        )
    return False


def _remap_to_temporal_col(
    col_id: str,
    col_map: Dict[str, Dict],
    joined_tids: set,
) -> Optional[str]:
    """Find the unique TEMPORAL column for a date-range filter whose col_id
    resolved to a non-TEMPORAL column (e.g. incident.id instead of
    incident.created_datetime).

    Strategy:
      1. Look for TEMPORAL columns in the *same* table as col_id.
         Return the match only when exactly one exists (unambiguous).
      2. If the same table has no TEMPORAL columns, look across all joined
         tables and return the match only when exactly one exists.

    Args:
        col_id:      The original col_id (non-TEMPORAL) from the filter node.
        col_map:     Full col_id → {col_name, table_id, table_name,
                     semantic_type} map.
        joined_tids: Set of table_ids currently in the FROM/JOIN chain.

    Returns:
        A col_id whose semantic_type is TEMPORAL, or None when the result
        is ambiguous or no TEMPORAL column can be found.
    """
    origin_info  = col_map.get(col_id, {})
    # Match by table_name — uuid4 stores assign different random UUIDs to the
    # same physical table, so table_name is the only reliable cross-store key.
    origin_tname = origin_info.get("table_name", "")

    same_table_temporal = [
        cid
        for cid, info in col_map.items()
        if info.get("semantic_type") == "TEMPORAL"
        and info.get("table_name") == origin_tname
        and cid != col_id
    ]
    if len(same_table_temporal) == 1:
        return same_table_temporal[0]
    if len(same_table_temporal) > 1:
        return _pick_best_temporal(same_table_temporal, col_map)

    joined_tnames = {
        info.get("table_name", "")
        for info in col_map.values()
        if info.get("table_id", "") in joined_tids
    }
    any_joined_temporal = [
        cid
        for cid, info in col_map.items()
        if info.get("semantic_type") == "TEMPORAL"
        and info.get("table_name", "") in joined_tnames
        and cid != col_id
    ]
    if len(any_joined_temporal) == 1:
        return any_joined_temporal[0]
    if len(any_joined_temporal) > 1:
        return _pick_best_temporal(any_joined_temporal, col_map)
    return None


def _pick_best_temporal(candidates: list, col_map: Dict[str, Dict]) -> Optional[str]:
    """
    From a list of TEMPORAL col_ids, pick the best candidate for a date-range
    filter. Prefers columns whose name suggests event creation/occurrence time.
    Falls back to the first candidate alphabetically by column name.
    """
    _PREFER = ("created", "occurred", "reported", "raised", "opened", "started")
    _SECOND = ("date", "time", "timestamp", "at")
    for pref in _PREFER:
        for cid in candidates:
            if pref in col_map[cid].get("col_name", "").lower():
                return cid
    for pref in _SECOND:
        for cid in candidates:
            if pref in col_map[cid].get("col_name", "").lower():
                return cid
    return candidates[0]


# =============================================================================
# Builder context — shared state for a single SQL statement
# =============================================================================

class _Ctx:
    """
    Accumulates params, warnings, and alias assignments while building SQL.
    One instance per IR node (subqueries get a child context with shared maps).
    """

    def __init__(
        self,
        ir_json:   Dict[str, Any],
        col_map:   Dict[str, Dict],
        table_map: Dict[str, str],
        depth:     int = 0,
    ):
        self.ir        = ir_json
        self.col_map   = col_map
        self.table_map = table_map
        self.depth     = depth
        self.params:   List[Any] = []
        self.warnings: List[str] = []

        # Build table_id → alias (t1, t2, ...) from entities + join tables
        self.alias: Dict[str, str] = {}
        self._build_alias_map()

    def _build_alias_map(self) -> None:
        idx = 0
        for ent in self.ir.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            tid = ent.get("table_id", "")
            if tid and tid not in self.alias:
                idx += 1
                self.alias[tid] = f"t{idx}"
        # Also assign aliases for tables referenced only in joins
        for j in self.ir.get("joins") or []:
            if not isinstance(j, dict):
                continue
            for key in ("from_table_id", "to_table_id"):
                tid = j.get(key, "")
                if tid and tid not in self.alias:
                    idx += 1
                    self.alias[tid] = f"t{idx}"

    # ── UUID resolution ──────────────────────────────────────────────────────

    def col_ref(self, col_id: Any) -> Optional[str]:
        """Resolve col_id → alias."col_name". Returns None + warning on miss."""
        if not isinstance(col_id, str) or not col_id:
            return None
        info = self.col_map.get(col_id)
        if not info:
            self.warnings.append(f"Unknown col_id {col_id!r} — skipped")
            return None
        alias = self.alias.get(info["table_id"])
        if alias:
            return f'{alias}.{_q(info["col_name"])}'
        # Column from a table not in alias map (unusual but safe fallback)
        return f'{_q(info["table_name"])}.{_q(info["col_name"])}'

    def table_name(self, table_id: Any) -> Optional[str]:
        if not isinstance(table_id, str) or not table_id:
            return None
        name = self.table_map.get(table_id)
        if not name:
            self.warnings.append(f"Unknown table_id {table_id!r} — skipped")
        return name

    # ── Parameter handling ───────────────────────────────────────────────────

    def p(self, value: Any) -> str:
        """Record a parameter value and return its %s placeholder."""
        self.params.append(value)
        return "%s"


# =============================================================================
# Clause builders
# =============================================================================

def _resolve_gb_col_id(
    item: Any,
    col_map: Dict[str, Dict],
    entity_tids: Set[str],
) -> str:
    """Return a col_id from a group_by item that may be a dict or a plain string.

    L3 sometimes emits strings like "t1.workflow_state" instead of {"col_id": "<uuid>"}.
    Strips any alias prefix, matches by col_name — prefers entity tables.
    """
    if isinstance(item, dict):
        return item.get("col_id", "")
    if isinstance(item, str):
        col_name = item.split(".")[-1].strip()
        if not col_name:
            return ""
        entity_match = next(
            (cid for cid, info in col_map.items()
             if info.get("col_name") == col_name
             and info.get("table_id") in entity_tids),
            None,
        )
        if entity_match:
            return entity_match
        return next(
            (cid for cid, info in col_map.items()
             if info.get("col_name") == col_name),
            "",
        )
    return ""


def _build_select_cols(ctx: _Ctx, joined_table_ids: Set[str] = None) -> List[str]:
    """
    Returns the list of SELECT expressions.

    joined_table_ids: when provided, entity columns from tables NOT in this
    set are excluded — prevents SELECT references to aliases that were never
    added to the FROM chain (which causes PostgreSQL "missing FROM-clause" errors).

    Logic:
    - COUNT intent, no aggregations, no group_by → COUNT(*) AS total
    - GROUP BY present                           → group_by cols + agg expressions
    - AGGREGATE / COUNT with aggs               → agg expressions (+ entity cols)
    - SELECT                                     → entity columns
    """
    intent = (ctx.ir.get("intent") or "SELECT").upper()
    aggs   = ctx.ir.get("aggregations") or []
    gb     = ctx.ir.get("group_by")     or []

    # Simple row count
    if intent == "COUNT" and not aggs and not gb:
        return ["COUNT(*) AS total"]

    cols:      List[str] = []
    seen_refs: Set[str]  = set()

    def _add(ref: Optional[str]) -> None:
        if ref and ref not in seen_refs:
            cols.append(ref)
            seen_refs.add(ref)

    # GROUP BY columns appear first in SELECT
    entity_tids = set(ctx.alias.keys())
    for item in gb:
        col_id = _resolve_gb_col_id(item, ctx.col_map, entity_tids)
        if not col_id:
            continue
        if joined_table_ids is not None and col_id:
            info = ctx.col_map.get(col_id, {})
            tid  = info.get("table_id")
            if tid and tid not in joined_table_ids:
                remapped = _remap_col_to_joined_table(col_id, ctx.col_map, joined_table_ids)
                if remapped:
                    col_id = remapped
                else:
                    continue  # Warning emitted in _build_group_by
        _add(ctx.col_ref(col_id))

    # Aggregation expressions
    for agg in aggs:
        if not isinstance(agg, dict):
            continue
        func    = (agg.get("func") or agg.get("type") or agg.get("function") or "COUNT").upper()
        col_id  = agg.get("col_id", "")
        if col_id == "*":   # wildcard — treat same as no col_id
            col_id = ""
        alias   = agg.get("alias") or f"{func.lower()}_result"
        
        if joined_table_ids is not None and col_id:
            info = ctx.col_map.get(col_id, {})
            tid  = info.get("table_id")
            if tid and tid not in joined_table_ids:
                remapped = _remap_col_to_joined_table(col_id, ctx.col_map, joined_table_ids)
                if remapped:
                    col_id = remapped
                else:
                    ctx.warnings.append(f"Agg col_id {col_id!r} references unjoined table {tid!r} — skipped")
                    continue

        if col_id:
            ref = ctx.col_ref(col_id)
            if ref:
                expr = f"{func}({ref}) AS {_q(alias)}"
                if expr not in seen_refs:
                    cols.append(expr)
                    seen_refs.add(expr)
        else:
            expr = f"{func}(*) AS {_q(alias)}"
            if expr not in seen_refs:
                cols.append(expr)
                seen_refs.add(expr)

    # Entity columns — always included for SELECT; fill gap if no agg cols.
    # Skip columns from entities whose table never made it into the FROM chain.
    if intent == "SELECT" or not cols:
        for ent in ctx.ir.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            ent_tid = ent.get("table_id", "")
            if joined_table_ids is not None and ent_tid not in joined_table_ids:
                ctx.warnings.append(
                    f"Entity table_id {ent_tid!r} not in FROM chain — "
                    f"columns omitted from SELECT"
                )
                continue
            for col in ent.get("columns") or []:
                if not isinstance(col, dict):
                    continue
                cid = col.get("col_id", "")

                def _add_entity_col(col_id: str) -> None:
                    ref = ctx.col_ref(col_id)
                    if not ref:
                        return
                    c_info = ctx.col_map.get(col_id, {})
                    if gb and not aggs and c_info.get("semantic_type") == "METRIC":
                        expr = f"MAX({ref}) AS {_q(c_info.get('col_name', ''))}"
                        _add(expr)
                    else:
                        _add(ref)

                # Guard: L3 sometimes places a col_id from a different table inside
                # an entity.  The entity-level guard above only checks the entity's
                # declared table_id — we must also check each column's actual table.
                if joined_table_ids is not None and cid:
                    col_info = ctx.col_map.get(cid, {})
                    actual_tid = col_info.get("table_id", "")
                    if actual_tid and actual_tid not in joined_table_ids:
                        # Recovery 1: entity table has a column named after the
                        # blocked table (e.g. module.display_name → workflow.module).
                        blocked_tname = col_info.get("table_name", "")
                        blocked_cname = col_info.get("col_name", "")
                        recovered = next(
                            (rcid for rcid, rinfo in ctx.col_map.items()
                             if rinfo.get("table_id") == ent_tid
                             and rinfo.get("col_name") == blocked_tname),
                            None,
                        )
                        # Recovery 2: entity table has a column with the same name
                        # as the blocked column (e.g. x.last_logged_in → user.last_logged_in).
                        if not recovered and blocked_cname:
                            recovered = next(
                                (rcid for rcid, rinfo in ctx.col_map.items()
                                 if rinfo.get("table_id") == ent_tid
                                 and rinfo.get("col_name") == blocked_cname),
                                None,
                            )
                        if recovered:
                            rec_name = ctx.col_map[recovered].get("col_name", "")
                            ctx.warnings.append(
                                f"Column col_id {cid!r} (table {actual_tid!r}) not in FROM "
                                f"chain — recovered {rec_name!r} column on entity table"
                            )
                            _add_entity_col(recovered)
                        else:
                            ctx.warnings.append(
                                f"Column col_id {cid!r} belongs to table {actual_tid!r} "
                                f"which is not in FROM chain — skipped"
                            )
                        continue
                _add_entity_col(cid)

    # Recovery: when GROUP BY is present but L3 emitted no aggregations, scan
    # col_map for METRIC columns on entity tables and add them as MAX(col).
    # Satisfies PostgreSQL's "must appear in GROUP BY or aggregate" requirement
    # and recovers the metric value L3 forgot to include (e.g. sla_hours).
    if gb and not aggs:
        ent_tids = {
            ent.get("table_id", "")
            for ent in (ctx.ir.get("entities") or [])
            if isinstance(ent, dict)
        }
        for cid, info in ctx.col_map.items():
            if info.get("semantic_type") != "METRIC":
                continue
            tid = info.get("table_id", "")
            if tid not in ent_tids:
                continue
            if joined_table_ids is not None and tid not in joined_table_ids:
                continue
            ref = ctx.col_ref(cid)
            if not ref:
                continue
            expr = f"MAX({ref}) AS {_q(info['col_name'])}"
            if expr not in seen_refs:
                cols.append(expr)
                seen_refs.add(expr)

    if cols:
        return cols
    # Fallback: for COUNT intent never emit bare *, emit COUNT(*) instead
    return ["COUNT(*) AS total"] if intent == "COUNT" else ["*"]


def _col_ids_in_ir(ir: Dict[str, Any]) -> Set[str]:
    """Collect every col_id referenced anywhere in an IR JSON dict."""
    ids: Set[str] = set()

    for ent in ir.get("entities") or []:
        if isinstance(ent, dict):
            for col in ent.get("columns") or []:
                if isinstance(col, dict) and col.get("col_id"):
                    ids.add(col["col_id"])

    def _scan_filter(node: Any) -> None:
        if not isinstance(node, dict):
            return
        cid = node.get("col_id")
        if cid:
            ids.add(cid)
        for child in node.get("children") or []:
            _scan_filter(child)

    _scan_filter(ir.get("filter_tree"))

    for section in ("group_by", "order_by", "aggregations"):
        for item in ir.get(section) or []:
            if isinstance(item, dict) and item.get("col_id"):
                ids.add(item["col_id"])

    return ids


def _build_from_join(ctx: _Ctx, join_path: List[Any] = None) -> Tuple[str, List[str], Set[str]]:
    """
    Builds FROM + JOIN clauses using multi-pass ordering.
    Returns (clause_string, tables_used_in_order, joined_table_ids).

    Multi-pass handles L3 IRs where joins reference tables not yet in the
    FROM chain (jump joins). Each pass processes only joins whose driving
    side is already in `joined`; remaining joins are retried next pass.
    Direction-swap: if from_tid is absent from `joined` but to_tid is
    present, the join is reversed so the already-joined side drives the ON.
    """
    entities = ctx.ir.get("entities") or []
    joins    = ctx.ir.get("joins")    or []

    if not entities or not isinstance(entities[0], dict):
        fallback = _infer_primary_table(
            [RetrievalResult(
                col_id=cid, col_name=info['col_name'],
                table_id=info['table_id'], table_name=info['table_name'],
                semantic_type='', similarity=1.0, embedding=None,
            ) for cid, info in ctx.col_map.items()]
        )
        if fallback is None:
            return "", [], set()
        fb_tid, fb_nm = fallback
        fb_al = ctx.alias.get(fb_tid, "t1")
        if fb_tid not in ctx.alias:
            ctx.alias[fb_tid] = fb_al
        ctx.warnings.append(
            f"IR has no entities — synthesised primary table '{fb_nm}' "
            f"from top_k_columns"
        )
        return f'FROM {_q(fb_nm)} AS {fb_al}', [fb_nm], {fb_tid}

    # Base table — first entity
    base     = entities[0]
    base_tid = base.get("table_id", "")
    base_nm  = ctx.table_name(base_tid) if base_tid else None
    base_al  = ctx.alias.get(base_tid, "t1")

    if not base_nm:
        fallback = _infer_primary_table(
            [RetrievalResult(
                col_id=cid, col_name=info['col_name'],
                table_id=info['table_id'], table_name=info['table_name'],
                semantic_type='', similarity=1.0, embedding=None,
            ) for cid, info in ctx.col_map.items()]
        )
        if fallback is None:
            return "", [], set()
        base_tid, base_nm = fallback
        base_al = ctx.alias.get(base_tid, "t1")
        if base_tid not in ctx.alias:
            ctx.alias[base_tid] = base_al
        ctx.warnings.append(
            f"Entity table_id unresolvable — fell back to inferred "
            f"primary table '{base_nm}' from top_k_columns"
        )

    tables_used: List[str] = [base_nm]
    joined:      Set[str]  = {base_tid}
    join_parts:  List[str] = []
    # Helper to gather all column IDs used specifically in the filter_tree
    def _filter_col_ids_in_ir(ir: Dict[str, Any]) -> Set[str]:
        ids: Set[str] = set()
        def _scan(node: Any) -> None:
            if not isinstance(node, dict):
                return
            cid = node.get("col_id")
            if cid:
                ids.add(cid)
            for child in node.get("children", []):
                _scan(child)
        _scan(ir.get("filter_tree") or {})
        return ids

    ir_col_ids:   Set[str] = _col_ids_in_ir(ctx.ir)
    filter_cids:  Set[str] = _filter_col_ids_in_ir(ctx.ir)

    # Multi-pass: each pass adds any join whose driving side is now in `joined`
    pending = [j for j in joins if isinstance(j, dict)]
    for _pass in range(len(pending) + 1):
        if not pending:
            break
        still_pending: List[dict] = []
        made_progress = False

        for j in pending:
            from_cid = j.get("from_col_id", "")
            to_cid   = j.get("to_col_id",   "")
            from_tid = j.get("from_table_id", "")
            to_tid   = j.get("to_table_id",   "")
            jtype    = (j.get("join_type") or "INNER").upper()
            if jtype not in ("INNER", "LEFT", "RIGHT", "FULL"):
                jtype = "INNER"

            # Infer table IDs from col_map when IR omits them
            if not from_tid and from_cid:
                info = ctx.col_map.get(from_cid)
                if info:
                    from_tid = info["table_id"]
            if not to_tid and to_cid:
                info = ctx.col_map.get(to_cid)
                if info:
                    to_tid = info["table_id"]

            # Both sides already joined — nothing to add
            if to_tid in joined and from_tid in joined:
                continue

            def _resolve_join_refs(drv_cid: str, drv_tid: str, tgt_cid: str, tgt_tid: str) -> Tuple[Optional[str], Optional[str]]:
                drv_info = ctx.col_map.get(drv_cid, {}) if drv_cid else {}
                tgt_info = ctx.col_map.get(tgt_cid, {}) if tgt_cid else {}
                
                drv_ref = ctx.col_ref(drv_cid) if drv_cid else None
                tgt_ref = ctx.col_ref(tgt_cid) if tgt_cid else None

                # Guard 1: L3 hallucinated same UUID for both sides or wrong tables.
                if tgt_cid and tgt_info.get("table_id") != tgt_tid:
                    col_name = tgt_info.get("col_name", "")
                    remapped = next(
                        (cid for cid, inf in ctx.col_map.items()
                         if inf.get("col_name") == col_name and inf.get("table_id") == tgt_tid),
                        None,
                    )
                    if remapped:
                        tgt_ref = ctx.col_ref(remapped)
                        tgt_info = ctx.col_map.get(remapped, {})
                    else:
                        ctx.warnings.append(f"Guard 1: Irrecoverable table mismatch for tgt_cid {tgt_cid!r} — skipping join")
                        tgt_ref = None
                
                if drv_cid and drv_info.get("table_id") != drv_tid:
                    col_name = drv_info.get("col_name", "")
                    remapped = next(
                        (cid for cid, inf in ctx.col_map.items()
                         if inf.get("col_name") == col_name and inf.get("table_id") == drv_tid),
                        None,
                    )
                    if remapped:
                        drv_ref = ctx.col_ref(remapped)
                        drv_info = ctx.col_map.get(remapped, {})
                    elif drv_info.get("table_id") not in joined:
                        ctx.warnings.append(f"Guard 1: Irrecoverable table mismatch for drv_cid {drv_cid!r} — skipping join")
                        drv_ref = None

                # Guard 2: Semantic type mismatch
                sem_drv = drv_info.get("semantic_type", "")
                sem_tgt = tgt_info.get("semantic_type", "")
                
                if sem_drv and sem_tgt and sem_drv != sem_tgt and not {sem_drv, sem_tgt}.issubset({"ID", "FOREIGN_KEY", "IDENTIFIER"}):
                    if sem_drv in {"ID", "FOREIGN_KEY", "IDENTIFIER"}:
                        col_name = drv_info.get("col_name", "")
                        drv_tname = drv_info.get("table_name", "")
                        possible = [col_name]
                        if col_name.endswith("_id"): possible.append("id")
                        if col_name == "id" and drv_tname: possible.append(f"{drv_tname}_id")
                        remapped = next(
                            (cid for cid, inf in ctx.col_map.items()
                             if inf.get("col_name") in possible and inf.get("table_id") == tgt_tid),
                            None,
                        )
                        if remapped:
                            rec_name = ctx.col_map[remapped].get("col_name", "")
                            ctx.warnings.append(f"Semantic mismatch ({sem_drv} != {sem_tgt}) — recovered tgt to {rec_name!r}")
                            tgt_ref = ctx.col_ref(remapped)
                            
                    elif sem_tgt in {"ID", "FOREIGN_KEY", "IDENTIFIER"}:
                        col_name = tgt_info.get("col_name", "")
                        tgt_tname = tgt_info.get("table_name", "")
                        possible = [col_name]
                        if col_name.endswith("_id"): possible.append("id")
                        if col_name == "id" and tgt_tname: possible.append(f"{tgt_tname}_id")
                        remapped = next(
                            (cid for cid, inf in ctx.col_map.items()
                             if inf.get("col_name") in possible and inf.get("table_id") == drv_tid),
                            None,
                        )
                        if remapped:
                            rec_name = ctx.col_map[remapped].get("col_name", "")
                            ctx.warnings.append(f"Semantic mismatch ({sem_drv} != {sem_tgt}) — recovered drv to {rec_name!r}")
                            drv_ref = ctx.col_ref(remapped)

                return drv_ref, tgt_ref

            if from_tid in joined and to_tid not in joined:
                # Standard: from-side already in FROM chain, join to new table
                new_tid = to_tid
                drv_ref, new_ref = _resolve_join_refs(from_cid, from_tid, to_cid, to_tid)
            elif to_tid in joined and from_tid not in joined:
                # Reverse: to-side already in FROM chain, join from new table
                new_tid = from_tid
                drv_ref, new_ref = _resolve_join_refs(to_cid, to_tid, from_cid, from_tid)
            else:
                # Neither side joined yet — defer to next pass
                still_pending.append(j)
                continue

            if not new_tid:
                continue

            new_nm = ctx.table_name(new_tid)
            new_al = ctx.alias.get(new_tid)
            if not new_nm or not new_al or not drv_ref or not new_ref:
                continue

            # Skip joins whose target table has no columns referenced anywhere
            # in the IR — excluding the keys used to perform the join itself.
            # If a table only contributes its join key, it's a useless fan-out.
            join_col_ids = {j.get("to_col_id"), j.get("from_col_id")}
            if not any(
                ctx.col_map.get(cid, {}).get("table_id") == new_tid
                for cid in ir_col_ids if cid not in join_col_ids
            ):
                ctx.warnings.append(
                    f"Join to '{new_nm}' skipped — no non-join columns "
                    f"referenced in query"
                )
                continue

            tables_used.append(new_nm)
            joined.add(new_tid)
            
            # Safe-Join Guard: If the SLM hallucinates an INNER join for a table
            # that is strictly used for dimensional lookup (NOT in the WHERE clause),
            # safely downgrade it to a LEFT JOIN to prevent aggressive row dropping.
            if jtype == "INNER":
                has_filter = any(
                    ctx.col_map.get(cid, {}).get("table_id") == new_tid
                    for cid in filter_cids
                )
                if not has_filter:
                    jtype = "LEFT"
                    
            join_parts.append(
                f"{jtype} JOIN {_q(new_nm)} AS {new_al} ON {drv_ref}::text = {new_ref}::text"
            )
            made_progress = True

        pending = still_pending
        if not made_progress:
            break

    for j in pending:
        ctx.warnings.append(
            f"Join {j.get('from_table_id','?')} → {j.get('to_table_id','?')} "
            f"could not be connected to FROM chain — skipped"
        )

    # ── Auto-infer missing joins from L2 join_path ───────────────────────────
    # Recovers when L3 emits multiple entities but leaves joins[] empty.
    # Uses table/column names from join_path directly — FK cols may not be in col_map.
    if join_path:
        entity_tids_all = {
            ent.get("table_id", "")
            for ent in entities
            if isinstance(ent, dict) and ent.get("table_id")
        }
        for ent_tid in entity_tids_all - joined:
            ent_tname = ctx.table_map.get(ent_tid, "")
            joined_tnames = {ctx.table_map.get(t, "") for t in joined if t}

            edge = next(
                (e for e in join_path
                 if (e.from_table_id in joined and e.to_table_id == ent_tid) or
                    (e.to_table_id in joined and e.from_table_id == ent_tid)),
                None,
            )
            if not edge and ent_tname:
                edge = next(
                    (e for e in join_path
                     if (e.from_table_name in joined_tnames and e.to_table_name == ent_tname) or
                        (e.to_table_name in joined_tnames and e.from_table_name == ent_tname)),
                    None,
                )
            if not edge:
                continue

            if not any(
                ctx.col_map.get(cid, {}).get("table_id") == ent_tid or
                ctx.col_map.get(cid, {}).get("table_name") == ent_tname
                for cid in ir_col_ids
            ):
                continue

            new_nm = ctx.table_map.get(ent_tid) or ent_tname
            new_al = ctx.alias.get(ent_tid)
            if not new_nm or not new_al:
                continue

            if edge.from_table_id in joined or edge.from_table_name in joined_tnames:
                drv_tname, drv_cname = edge.from_table_name, edge.from_col_name
                new_cname = edge.to_col_name
            else:
                drv_tname, drv_cname = edge.to_table_name, edge.to_col_name
                new_cname = edge.from_col_name

            drv_tid = next(
                (tid for tid in joined if ctx.table_map.get(tid) == drv_tname),
                None,
            )
            drv_al = ctx.alias.get(drv_tid) if drv_tid else None
            if not drv_al:
                continue

            tables_used.append(new_nm)
            joined.add(ent_tid)
            ctx.warnings.append(
                f"Auto-inferred join to '{new_nm}' "
                f"({drv_tname}.{drv_cname} → {new_nm}.{new_cname}) — L3 omitted joins[]"
            )
            join_parts.append(
                f"LEFT JOIN {_q(new_nm)} AS {new_al} "
                f"ON {drv_al}.{_q(drv_cname)}::text = {new_al}.{_q(new_cname)}::text"
            )

    clause = f"FROM {_q(base_nm)} AS {base_al}"
    if join_parts:
        clause += "\n" + "\n".join(join_parts)

    return clause, tables_used, joined


def _build_filter(ctx: _Ctx, node: Any, joined_tids: Optional[set] = None) -> Optional[str]:
    """
    Recursively converts a filter_tree node to a SQL fragment.
    All literal values are added to ctx.params (%s placeholders).
    joined_tids: when provided, conditions referencing columns from unjoined tables are skipped.
    """
    if not isinstance(node, dict):
        return None

    node_type = (node.get("type") or "").upper()

    # ── AND / OR combinator ──────────────────────────────────────────────────
    if node_type in ("AND", "OR"):
        parts = []
        for child in node.get("children") or []:
            part = _build_filter(ctx, child, joined_tids)
            if part:
                parts.append(part)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        sep = f" {node_type} "
        return sep.join(f"({p})" for p in parts)

    # ── Leaf node ────────────────────────────────────────────────────────────
    col_id   = node.get("col_id", "")
    operator = (node.get("operator") or "EQ").upper()
    value    = node.get("value")

    # Skip conditions whose column belongs to a table not in the FROM/JOIN chain.
    # Before giving up, try to remap the col_id to a same-named column in a
    # joined table (e.g. L3 picked audit_log.created_by_id when the query
    # table incident.created_datetime has the same name prefix).
    if joined_tids is not None and col_id:
        info = ctx.col_map.get(col_id, {})
        tid  = info.get("table_id")
        if tid and tid not in joined_tids:
            remapped = _remap_col_to_joined_table(col_id, ctx.col_map, joined_tids)
            if remapped:
                remapped_info = ctx.col_map[remapped]
                ctx.warnings.append(
                    f"Filter col_id {col_id!r} (table {tid!r}) not in FROM chain — "
                    f"remapped to col_id {remapped!r} "
                    f"({remapped_info['table_name']}.{remapped_info['col_name']})"
                )
                col_id = remapped
            else:
                # Same-name remap failed.  If this is a date-range filter with
                # datetime values, try a TEMPORAL-type remap as a last resort
                # before discarding the condition entirely.
                # Example: L3 picks agent_registry.created_by_id (unjoined) for a
                # BETWEEN filter — no created_by_id in incident, but
                # incident.created_datetime is TEMPORAL and unique → use it.
                if operator in _DATE_RANGE_OPERATORS and _is_datetime_value(value):
                    temporal_remapped = _remap_to_temporal_col(
                        col_id, ctx.col_map, joined_tids
                    )
                    if temporal_remapped:
                        remapped_info = ctx.col_map[temporal_remapped]
                        ctx.warnings.append(
                            f"Filter col_id {col_id!r} (unjoined table {tid!r}, "
                            f"no same-named col in joined tables) — "
                            f"temporal remap to "
                            f"{remapped_info['table_name']}.{remapped_info['col_name']}"
                        )
                        col_id = temporal_remapped
                    else:
                        ctx.warnings.append(
                            f"Filter col_id {col_id!r} references unjoined table {tid!r} — "
                            f"no same-named column found in joined tables — condition skipped"
                        )
                        return None
                else:
                    ctx.warnings.append(
                        f"Filter col_id {col_id!r} references unjoined table {tid!r} — "
                        f"no same-named column found in joined tables — condition skipped"
                    )
                    return None

    # Temporal-type mismatch: a date-range operator with datetime values
    # was mapped to a non-TEMPORAL column (e.g. incident.id instead of
    # incident.created_datetime because L3 grabbed the first "created"
    # UUID it saw).  Find the unambiguous TEMPORAL column and remap.
    if joined_tids is not None and col_id and operator in _DATE_RANGE_OPERATORS:
        if _is_datetime_value(value):
            col_info = ctx.col_map.get(col_id, {})
            if col_info.get("semantic_type", "") != "TEMPORAL":
                remapped = _remap_to_temporal_col(col_id, ctx.col_map, joined_tids)
                if remapped:
                    remapped_info = ctx.col_map[remapped]
                    ctx.warnings.append(
                        f"Filter col_id {col_id!r} has semantic_type "
                        f"{col_info.get('semantic_type','?')!r} but operator "
                        f"{operator!r} expects TEMPORAL — remapped to "
                        f"{remapped_info['table_name']}.{remapped_info['col_name']}"
                    )
                    col_id = remapped
                else:
                    ctx.warnings.append(
                        f"Filter col_id {col_id!r} is not TEMPORAL for "
                        f"date-range operator {operator!r} — no unambiguous "
                        f"TEMPORAL column found — condition skipped"
                    )
                    return None

    ref = ctx.col_ref(col_id) if col_id else None

    # Reject SLM-hallucinated placeholder values (<literal>, <value>, etc.)
    if isinstance(value, str) and _PLACEHOLDER_RE.match(value.strip()):
        ctx.warnings.append(
            f"Placeholder value {value!r} in filter condition — condition skipped"
        )
        return None

    # IS_NULL — no parameter
    if operator == "IS_NULL":
        return f"{ref} IS NULL" if ref else None

    # NOT_EXISTS — recursive subquery
    if operator == "NOT_EXISTS":
        if ctx.depth >= SQL_MAX_SUBQUERY_DEPTH:
            ctx.warnings.append("NOT_EXISTS exceeds max subquery depth — skipped")
            return None
        if isinstance(value, dict) and value.get("type") == "SUBQUERY":
            sub_ir = value.get("ir") or {}
            sub    = _build_sql_from_ir(sub_ir, ctx.col_map, ctx.table_map, ctx.depth + 1)
            if sub.error:
                ctx.warnings.append(f"Subquery failed: {sub.error}")
                return None
            ctx.params.extend(sub.params)
            ctx.warnings.extend(sub.warnings)
            return f"NOT EXISTS ({sub.sql})"
        return None

    # IN — value is a list
    if operator == "IN":
        if not isinstance(value, list) or not value:
            return None
        clean = [v for v in value if not (isinstance(v, str) and _PLACEHOLDER_RE.match(v.strip()))]
        if not clean:
            ctx.warnings.append("IN list contains only placeholder values — condition skipped")
            return None
        placeholders = ", ".join(ctx.p(v) for v in clean)
        return f"{ref} IN ({placeholders})" if ref else None

    # BETWEEN — value is [start, end] or {start, end}
    if operator == "BETWEEN":
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return f"{ref} BETWEEN {ctx.p(value[0])} AND {ctx.p(value[1])}" if ref else None
        if isinstance(value, dict):
            start = value.get("start") or value.get("gte")
            end   = value.get("end")   or value.get("lte")
            if start and end and ref:
                return f"{ref} BETWEEN {ctx.p(start)} AND {ctx.p(end)}"
        return None

    # LIKE
    if operator == "LIKE":
        return f"{ref} LIKE {ctx.p(value)}" if ref else None

    # Standard comparison
    _OP_MAP = {"EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}
    sql_op = _OP_MAP.get(operator)
    if sql_op and ref:
        if value is None:
            return f"{ref} IS NULL" if sql_op == "=" else f"{ref} IS NOT NULL"
        return f"{ref} {sql_op} {ctx.p(value)}"

    ctx.warnings.append(f"Unknown operator {operator!r} — filter skipped")
    return None


def _build_group_by(ctx: _Ctx, joined_tids: Optional[set] = None) -> str:
    parts = []
    entity_tids = set(ctx.alias.keys())
    for item in ctx.ir.get("group_by") or []:
        col_id = _resolve_gb_col_id(item, ctx.col_map, entity_tids)
        if not col_id:
            continue
        if joined_tids is not None and col_id:
            info = ctx.col_map.get(col_id, {})
            tid  = info.get("table_id")
            if tid and tid not in joined_tids:
                remapped = _remap_col_to_joined_table(col_id, ctx.col_map, joined_tids)
                if remapped:
                    col_id = remapped
                else:
                    ctx.warnings.append(f"GROUP BY col_id {col_id!r} references unjoined table {tid!r} — skipped")
                    continue
        ref = ctx.col_ref(col_id)
        if ref:
            parts.append(ref)
    return f"GROUP BY {', '.join(parts)}" if parts else ""


def _build_order_by(ctx: _Ctx, joined_tids: Optional[set] = None) -> str:
    parts = []
    for item in ctx.ir.get("order_by") or []:
        if not isinstance(item, dict):
            continue
        col_id = item.get("col_id", "")
        if joined_tids is not None and col_id:
            info = ctx.col_map.get(col_id, {})
            tid  = info.get("table_id")
            if tid and tid not in joined_tids:
                remapped = _remap_col_to_joined_table(col_id, ctx.col_map, joined_tids)
                if remapped:
                    col_id = remapped
                else:
                    ctx.warnings.append(f"ORDER BY col_id {col_id!r} references unjoined table {tid!r} — skipped")
                    continue
        ref = ctx.col_ref(col_id)
        direction = (item.get("direction") or "ASC").upper()
        if direction not in ("ASC", "DESC"):
            direction = "ASC"
        if ref:
            parts.append(f"{ref} {direction}")
    return f"ORDER BY {', '.join(parts)}" if parts else ""


def _build_limit(ctx: _Ctx, is_subquery: bool) -> str:
    if is_subquery:
        return ""   # subqueries must not have LIMIT
    intent = (ctx.ir.get("intent") or "SELECT").upper()
    gb     = ctx.ir.get("group_by") or []
    if intent == "COUNT" and not gb:
        return ""   # COUNT(*) with no GROUP BY returns one row — no LIMIT needed
    limit = ctx.ir.get("limit")
    if limit is None:
        return f"LIMIT {SQL_DEFAULT_LIMIT}"
    try:
        n = int(limit)
        return f"LIMIT {min(n, SQL_DEFAULT_LIMIT)}"
    except (ValueError, TypeError):
        return f"LIMIT {SQL_DEFAULT_LIMIT}"


# =============================================================================
# Core recursive builder
# =============================================================================

def _build_sql_from_ir(
    ir_json:   Dict[str, Any],
    col_map:   Dict[str, Dict],
    table_map: Dict[str, str],
    depth:     int = 0,
    join_path: List[Any] = None,
) -> SQLBuilderResult:
    """Internal recursive SQL builder. Depth tracks subquery nesting level."""

    if not ir_json:
        return SQLBuilderResult(
            sql="", params=(), query_type="SELECT",
            tables_used=[], error="Empty IR JSON",
        )

    ctx = _Ctx(ir_json, col_map, table_map, depth)

    # FROM + JOINs
    from_clause, tables_used, joined_tids = _build_from_join(ctx, join_path)
    if not from_clause:
        return SQLBuilderResult(
            sql="", params=(), query_type="SELECT",
            tables_used=[], error="No resolvable tables in IR entities",
            warnings=ctx.warnings,
        )

    # SELECT
    select_cols   = _build_select_cols(ctx, joined_tids)
    select_clause = "SELECT " + ", ".join(select_cols)

    # WHERE
    filter_sql   = _build_filter(ctx, ir_json.get("filter_tree"), joined_tids)
    where_clause = f"WHERE {filter_sql}" if filter_sql else ""

    # GROUP BY / ORDER BY / LIMIT
    group_by_clause = _build_group_by(ctx, joined_tids)
    order_by_clause = _build_order_by(ctx, joined_tids)
    limit_clause    = _build_limit(ctx, is_subquery=(depth > 0))

    # Assemble — only include non-empty clauses
    parts = [select_clause, from_clause]
    for clause in (where_clause, group_by_clause, order_by_clause, limit_clause):
        if clause:
            parts.append(clause)

    sql = "\n".join(parts)

    return SQLBuilderResult(
        sql         = sql,
        params      = tuple(ctx.params),
        query_type  = (ir_json.get("intent") or "SELECT").upper(),
        tables_used = tables_used,
        error       = None,
        warnings    = ctx.warnings,
    )


# =============================================================================
# Public entry point
# =============================================================================

def run_sql_builder(
    ir_json:       Dict[str, Any],
    top_k_columns: List[RetrievalResult],
    verbose:       bool = False,
    join_path:     List[Any] = None,
) -> SQLBuilderResult:
    """
    Layer 4 entry point. Converts IR JSON to parameterised SQL.

    Parameters
    ----------
    ir_json        : SLMResult.ir_json — the structured IR from L3
    top_k_columns  : SemanticLayerResult.top_k_columns — for UUID→name lookup
    verbose        : Print per-step debug info

    Returns
    -------
    SQLBuilderResult — always returns, never raises.
    On failure: error field is set, sql = "".
    """
    t0 = time.time()

    logger.debug("L4 SQL builder: intent=%s, entities=%d, joins=%d",
                 ir_json.get("intent", "?") if ir_json else "NONE",
                 len(ir_json.get("entities") or []) if ir_json else 0,
                 len(ir_json.get("joins") or []) if ir_json else 0)

    if not ir_json:
        logger.warning("L4 SQL builder: ir_json is empty")
        return SQLBuilderResult(
            sql="", params=(), query_type="SELECT",
            tables_used=[], error="ir_json is empty — L3 produced no IR",
            duration_ms=0.0,
        )

    col_map, table_map = _build_uuid_maps(top_k_columns)
    _resolve_ir_entities(ir_json, col_map, table_map)   # name-only IR → UUIDs (anti SELECT-*)

    if verbose:
        print(f"[SQLBuilder] intent={ir_json.get('intent','SELECT')}  "
              f"entities={len(ir_json.get('entities') or [])}  "
              f"joins={len(ir_json.get('joins') or [])}  "
              f"aggs={len(ir_json.get('aggregations') or [])}  "
              f"uuid_cols={len(col_map)}")

    result = _build_sql_from_ir(ir_json, col_map, table_map, depth=0, join_path=join_path)
    result.duration_ms = round((time.time() - t0) * 1000, 2)

    if result.error:
        logger.warning("L4 SQL builder error: %s", result.error)
    else:
        logger.info(
            "L4 SQL: type=%s, tables=%s, params=%d, warnings=%d, %.1fms",
            result.query_type, result.tables_used, len(result.params),
            len(result.warnings), result.duration_ms,
        )
        logger.debug("L4 SQL text: %s", result.sql)
        if result.params:
            logger.debug("L4 params: %s", result.params)

    if verbose:
        if result.error:
            print(f"  [ERROR] {result.error}")
        else:
            print(f"  Tables    : {result.tables_used}")
            print(f"  Params    : {len(result.params)} values")
            print(f"  Duration  : {result.duration_ms}ms")
            print("  SQL:")
            for line in result.sql.split("\n"):
                print(f"    {line}")
        for w in result.warnings:
            print(f"  [WARN] {w}")

    return result


# =============================================================================
# Smoke test — python query/sql_builder.py
# =============================================================================

if __name__ == "__main__":
    import uuid
    import numpy as np
    from ingestion.vector_store import RetrievalResult

    def _col(col_id, col_name, table_id, table_name, semantic_type="IDENTIFIER"):
        return RetrievalResult(
            col_id=col_id, col_name=col_name,
            table_id=table_id, table_name=table_name,
            semantic_type=semantic_type, similarity=0.9, embedding=None,
        )

    # Stable UUIDs for the test
    INC_TID  = "11111111-0000-0000-0000-000000000001"
    INC_ID   = "11111111-0000-0000-0000-000000000010"
    INC_ST   = "11111111-0000-0000-0000-000000000011"
    INC_WS   = "11111111-0000-0000-0000-000000000012"
    INC_NO   = "11111111-0000-0000-0000-000000000013"
    INC_CDT  = "11111111-0000-0000-0000-000000000014"
    USR_TID  = "22222222-0000-0000-0000-000000000001"
    USR_UID  = "22222222-0000-0000-0000-000000000010"
    USR_UN   = "22222222-0000-0000-0000-000000000011"
    USR_EM   = "22222222-0000-0000-0000-000000000012"
    SLA_TID  = "33333333-0000-0000-0000-000000000001"
    SLA_ID   = "33333333-0000-0000-0000-000000000010"
    SLA_HRS  = "33333333-0000-0000-0000-000000000011"
    SLA_WS   = "33333333-0000-0000-0000-000000000012"

    top_k = [
        _col(INC_ID,  "id",               INC_TID, "incident"),
        _col(INC_ST,  "incident_status",  INC_TID, "incident"),
        _col(INC_WS,  "workflow_state",   INC_TID, "incident"),
        _col(INC_NO,  "incident_no",      INC_TID, "incident"),
        _col(INC_CDT, "created_datetime", INC_TID, "incident",   semantic_type="TEMPORAL"),
        _col(USR_UID, "user_id",          USR_TID, "user"),
        _col(USR_UN,  "username",         USR_TID, "user"),
        _col(USR_EM,  "email",            USR_TID, "user"),
        _col(SLA_ID,  "id",               SLA_TID, "sla_config"),
        _col(SLA_HRS, "sla_hours",        SLA_TID, "sla_config"),
        _col(SLA_WS,  "workflow_state",   SLA_TID, "sla_config"),
    ]

    print("=" * 70)
    print("VEDA POC — SQL Builder (L4) smoke test")
    print("=" * 70)

    # Test 1: Simple SELECT with WHERE
    ir1 = {
        "version": "1.0", "intent": "SELECT",
        "entities": [
            {"table_id": INC_TID, "alias": "t1", "columns": [
                {"col_id": INC_ST}, {"col_id": INC_WS}, {"col_id": INC_NO},
            ]}
        ],
        "filter_tree": {
            "type": "AND", "children": [
                {"col_id": INC_ST, "operator": "EQ", "value": "OPEN"},
            ]
        },
        "joins": [], "aggregations": [], "group_by": [], "order_by": [], "limit": None,
    }
    r1 = run_sql_builder(ir1, top_k, verbose=True)
    print(f"\n  → error: {r1.error}  params: {r1.params}")

    # Test 2: COUNT with GROUP BY
    ir2 = {
        "version": "1.0", "intent": "COUNT",
        "entities": [
            {"table_id": INC_TID, "alias": "t1", "columns": [
                {"col_id": INC_WS}, {"col_id": INC_ID},
            ]}
        ],
        "filter_tree": {"type": "AND", "children": []},
        "joins": [], "group_by": [{"col_id": INC_WS}],
        "aggregations": [{"function": "COUNT", "col_id": INC_ID, "alias": "total"}],
        "order_by": [], "limit": None,
    }
    print("\n" + "-" * 50)
    r2 = run_sql_builder(ir2, top_k, verbose=True)
    print(f"\n  → error: {r2.error}  params: {r2.params}")

    # Test 3: AGGREGATE (AVG) with GROUP BY
    ir3 = {
        "version": "1.0", "intent": "AGGREGATE",
        "entities": [
            {"table_id": SLA_TID, "alias": "t1", "columns": [
                {"col_id": SLA_WS}, {"col_id": SLA_HRS},
            ]}
        ],
        "filter_tree": {"type": "AND", "children": []},
        "joins": [], "group_by": [{"col_id": SLA_WS}],
        "aggregations": [{"function": "AVG", "col_id": SLA_HRS, "alias": "avg_sla"}],
        "order_by": [{"col_id": SLA_WS, "direction": "ASC"}],
        "limit": None,
    }
    print("\n" + "-" * 50)
    r3 = run_sql_builder(ir3, top_k, verbose=True)
    print(f"\n  → error: {r3.error}  params: {r3.params}")

    # Test 4: JOIN with BETWEEN filter (temporal)
    ir4 = {
        "version": "1.0", "intent": "SELECT",
        "entities": [
            {"table_id": INC_TID, "alias": "t1", "columns": [
                {"col_id": INC_NO}, {"col_id": INC_CDT},
            ]},
            {"table_id": USR_TID, "alias": "t2", "columns": [
                {"col_id": USR_UN}, {"col_id": USR_EM},
            ]},
        ],
        "filter_tree": {
            "type": "AND", "children": [
                {"col_id": INC_CDT, "operator": "BETWEEN",
                 "value": ["2024-01-01T00:00:00", "2024-03-31T23:59:59"]},
            ]
        },
        "joins": [{
            "from_table_id": INC_TID, "from_col_id": INC_ID,
            "to_table_id":   USR_TID, "to_col_id":   USR_UID,
            "join_type": "INNER",
        }],
        "aggregations": [], "group_by": [], "order_by": [], "limit": 50,
    }
    print("\n" + "-" * 50)
    r4 = run_sql_builder(ir4, top_k, verbose=True)
    print(f"\n  → error: {r4.error}  params: {r4.params}")

    print("\n" + "=" * 70)
    ok = sum(1 for r in [r1, r2, r3, r4] if not r.error)
    print(f"  {ok}/4 tests passed")