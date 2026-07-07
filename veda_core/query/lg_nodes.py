"""query/lg_nodes.py — LangGraph node functions for VEDA L3 pipeline."""

import re
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, TypedDict

from config import (
    SLM_OLLAMA_BASE_URL,
    SLM_MODEL_NAME,
    SLM_TEMPERATURE,
    SLM_TIMEOUT_SECS,
)
from query.lg_prompts import (
    INTENT_PROMPT,
    ENTITY_PROMPT,
    COLUMN_PROMPT,
    FILTER_PROMPT,
)

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _is_uuid(s: Any) -> bool:
    return isinstance(s, str) and bool(_UUID_RE.match(s))


# =============================================================================
# State
# =============================================================================

class VEDAQueryState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────────────
    query:            str
    temporal_filter:  Optional[Dict]       # {"start": iso, "end": iso} or None
    top_k_columns:    List[Dict]           # RetrievalResult as dicts
    join_path:        List[Dict]           # JoinEdge as dicts
    must_include:     List[Dict]           # {col_id, col_name, table_id, table_name}

    # ── node: classify_intent ─────────────────────────────────────────────────
    intent:               str              # SELECT | COUNT | AGGREGATE
    complexity:           str              # SIMPLE | MODERATE | COMPLEX
    needs_clarification:  bool
    clarification_reason: Optional[str]

    # ── node: select_entity ───────────────────────────────────────────────────
    primary_table_id:    str
    secondary_table_ids: List[str]

    # ── node: select_columns ─────────────────────────────────────────────────
    selected_col_ids:  List[str]
    group_by_col_id:   Optional[str]
    order_by_col_id:   Optional[str]
    order_direction:   str                 # ASC | DESC

    # ── node: build_filters ──────────────────────────────────────────────────
    filter_tree:       Optional[Dict]

    # ── node: assemble_ir ────────────────────────────────────────────────────
    ir_json:           Dict[str, Any]
    confidence:        float

    # ── meta ─────────────────────────────────────────────────────────────────
    errors:            List[str]
    node_times:        Dict[str, float]    # node_name → duration_ms


# =============================================================================
# Shared Ollama helper
# =============================================================================

def _call_node(system_prompt: str, user_msg: str) -> Optional[Dict]:
    """
    Calls Ollama with a focused system prompt and user message.
    Returns parsed dict or None on any failure.
    num_predict=256 — each node outputs a small JSON object.
    """
    try:
        from slm import call_slm
        content = call_slm(
            user_msg,
            system=system_prompt,
            purpose="lg_node",
            temperature=SLM_TEMPERATURE,
            num_predict=256,
            timeout=SLM_TIMEOUT_SECS,
        )
        # Strip markdown fences if present
        content = re.sub(r"```(?:json)?\s*", "", content)
        content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE).strip()
        start   = content.find("{")
        end     = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass
    return None


# =============================================================================
# Node 1 — classify_intent
# =============================================================================

def node_classify_intent(state: VEDAQueryState) -> Dict:
    t0    = time.time()
    query = state.get("query", "")
    errs  = list(state.get("errors", []))
    times = dict(state.get("node_times", {}))

    user_msg = f'Query: "{query}"'
    result   = _call_node(INTENT_PROMPT, user_msg)

    _VALID_INTENTS    = {"SELECT", "COUNT", "AGGREGATE"}
    _VALID_COMPLEXITY = {"SIMPLE", "MODERATE", "COMPLEX"}

    intent               = "SELECT"
    complexity           = "SIMPLE"
    needs_clarification  = False
    clarification_reason = None

    if result:
        raw_intent = result.get("intent", "SELECT")
        intent     = raw_intent if raw_intent in _VALID_INTENTS else "SELECT"
        raw_cx     = result.get("complexity", "SIMPLE")
        complexity = raw_cx if raw_cx in _VALID_COMPLEXITY else "SIMPLE"
        needs_clarification  = bool(result.get("needs_clarification", False))
        clarification_reason = result.get("clarification_reason")
    else:
        errs.append("classify_intent: Ollama call failed — using SELECT/SIMPLE fallback")

    times["classify_intent"] = round((time.time() - t0) * 1000, 1)
    return {
        "intent":               intent,
        "complexity":           complexity,
        "needs_clarification":  needs_clarification,
        "clarification_reason": clarification_reason,
        "errors":               errs,
        "node_times":           times,
    }


# =============================================================================
# Node 2 — select_entity
# =============================================================================

def node_select_entity(state: VEDAQueryState) -> Dict:
    t0           = time.time()
    query        = state.get("query", "")
    top_k        = state.get("top_k_columns", [])
    errs         = list(state.get("errors", []))
    times        = dict(state.get("node_times", {}))

    # Build table reference from unique table_ids
    seen_tables: Dict[str, str] = {}
    for col in top_k:
        tid   = col.get("table_id", "")
        tname = col.get("table_name", "")
        if tid and tid not in seen_tables:
            seen_tables[tid] = tname

    table_ref_lines = "\n".join(
        f"  {tid}  →  {tname}" for tid, tname in seen_tables.items()
    )
    user_msg = (
        f'Query: "{query}"\n\n'
        f"TABLE REFERENCE (copy table_id exactly):\n{table_ref_lines}"
    )

    result = _call_node(ENTITY_PROMPT, user_msg)

    # Fallback: most frequent table in top_k
    def _most_frequent_table() -> str:
        counts: Dict[str, int] = {}
        for col in top_k:
            tid = col.get("table_id", "")
            counts[tid] = counts.get(tid, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    primary_table_id    = ""
    secondary_table_ids = []
    valid_table_ids     = set(seen_tables.keys())

    if result:
        pid = result.get("primary_table_id", "")
        if _is_uuid(pid) and pid in valid_table_ids:
            primary_table_id = pid
        sids = result.get("secondary_table_ids", [])
        secondary_table_ids = [
            s for s in sids
            if _is_uuid(s) and s in valid_table_ids and s != primary_table_id
        ]

    if not primary_table_id:
        primary_table_id = _most_frequent_table()
        errs.append("select_entity: fallback to most-frequent table")

    times["select_entity"] = round((time.time() - t0) * 1000, 1)
    return {
        "primary_table_id":    primary_table_id,
        "secondary_table_ids": secondary_table_ids,
        "errors":              errs,
        "node_times":          times,
    }


# =============================================================================
# Node 3 — select_columns
# =============================================================================

def node_select_columns(state: VEDAQueryState) -> Dict:
    t0                  = time.time()
    query               = state.get("query", "")
    intent              = state.get("intent", "SELECT")
    top_k               = state.get("top_k_columns", [])
    primary_table_id    = state.get("primary_table_id", "")
    secondary_table_ids = state.get("secondary_table_ids", [])
    must_include        = state.get("must_include", [])
    errs                = list(state.get("errors", []))
    times               = dict(state.get("node_times", {}))

    entity_table_ids = {primary_table_id} | set(secondary_table_ids)

    # Filter columns to only entity tables
    filtered_cols = [c for c in top_k if c.get("table_id", "") in entity_table_ids]
    if not filtered_cols:
        filtered_cols = top_k  # fallback: show all

    valid_col_ids = {c["col_id"] for c in filtered_cols}

    col_ref_lines = "\n".join(
        f"  {c['col_id']}  →  {c['table_name']}.{c['col_name']}  [{c.get('semantic_type', '')}]"
        for c in filtered_cols
    )
    must_lines = "\n".join(
        f"  {m['col_id']}  →  {m['table_name']}.{m['col_name']}"
        for m in must_include
        if m.get("col_id") in valid_col_ids
    )
    user_msg = (
        f'Query: "{query}"\n'
        f"Intent: {intent}\n\n"
        f"COLUMN REFERENCE (copy col_id exactly):\n{col_ref_lines}"
    )
    if must_lines:
        user_msg += f"\n\nMust-include columns (MUST appear in selected_col_ids):\n{must_lines}"

    result = _call_node(COLUMN_PROMPT, user_msg)

    selected_col_ids = []
    group_by_col_id  = None
    order_by_col_id  = None
    order_direction  = "ASC"

    if result:
        raw_ids = result.get("selected_col_ids", [])
        selected_col_ids = [c for c in raw_ids if _is_uuid(c) and c in valid_col_ids]

        gb = result.get("group_by_col_id")
        if _is_uuid(gb) and gb in valid_col_ids:
            group_by_col_id = gb

        ob = result.get("order_by_col_id")
        if _is_uuid(ob) and ob in valid_col_ids:
            order_by_col_id = ob

        order_direction = result.get("order_direction", "ASC")
        if order_direction not in ("ASC", "DESC"):
            order_direction = "ASC"
    else:
        errs.append("select_columns: Ollama call failed — empty column selection")

    times["select_columns"] = round((time.time() - t0) * 1000, 1)
    return {
        "selected_col_ids": selected_col_ids,
        "group_by_col_id":  group_by_col_id,
        "order_by_col_id":  order_by_col_id,
        "order_direction":  order_direction,
        "errors":           errs,
        "node_times":       times,
    }


# =============================================================================
# Node 4 — build_filters
# =============================================================================

def node_build_filters(state: VEDAQueryState) -> Dict:
    t0               = time.time()
    query            = state.get("query", "")
    top_k            = state.get("top_k_columns", [])
    selected_col_ids = state.get("selected_col_ids", [])
    temporal_filter  = state.get("temporal_filter")
    errs             = list(state.get("errors", []))
    times            = dict(state.get("node_times", {}))

    # COLUMN REFERENCE: selected columns only (tightest valid set for filter)
    selected_set  = set(selected_col_ids)
    filter_cols   = [c for c in top_k if c.get("col_id", "") in selected_set]
    if not filter_cols:
        filter_cols = top_k  # fallback

    valid_col_ids = {c["col_id"] for c in filter_cols}

    col_ref_lines = "\n".join(
        f"  {c['col_id']}  →  {c['table_name']}.{c['col_name']}  [{c.get('semantic_type', '')}]"
        for c in filter_cols
    )
    tf_str = json.dumps(temporal_filter) if temporal_filter else "null"
    user_msg = (
        f'Query: "{query}"\n'
        f"Temporal filter: {tf_str}\n\n"
        f"COLUMN REFERENCE (copy col_id exactly):\n{col_ref_lines}"
    )

    result      = _call_node(FILTER_PROMPT, user_msg)
    filter_tree = {"type": "AND", "children": []}

    if result:
        ft = result.get("filter_tree")
        if isinstance(ft, dict):
            filter_tree = _clean_filter_tree(ft, valid_col_ids)

    times["build_filters"] = round((time.time() - t0) * 1000, 1)
    return {
        "filter_tree": filter_tree,
        "errors":      errs,
        "node_times":  times,
    }


def _clean_filter_tree(node: Any, valid_col_ids: set) -> Any:
    """Recursively strip filter conditions with invalid col_ids."""
    if not isinstance(node, dict):
        return {"type": "AND", "children": []}
    if "col_id" in node:
        return node if (_is_uuid(node.get("col_id")) and node["col_id"] in valid_col_ids) else None
    children = node.get("children", [])
    if isinstance(children, list):
        clean = [c for c in (_clean_filter_tree(ch, valid_col_ids) for ch in children) if c is not None]
        node["children"] = clean
    return node


# =============================================================================
# Node 5 — assemble_ir (DETERMINISTIC — zero LLM calls)
# =============================================================================

def node_assemble_ir(state: VEDAQueryState) -> Dict:
    t0 = time.time()

    query               = state.get("query", "")
    intent              = state.get("intent", "SELECT")
    complexity          = state.get("complexity", "SIMPLE")
    needs_clarification = state.get("needs_clarification", False)
    primary_table_id    = state.get("primary_table_id", "")
    secondary_table_ids = state.get("secondary_table_ids", [])
    selected_col_ids    = list(state.get("selected_col_ids", []))
    group_by_col_id     = state.get("group_by_col_id")
    order_by_col_id     = state.get("order_by_col_id")
    order_direction     = state.get("order_direction", "ASC")
    filter_tree         = state.get("filter_tree", {"type": "AND", "children": []})
    top_k               = state.get("top_k_columns", [])
    join_path           = state.get("join_path", [])
    must_include        = state.get("must_include", [])
    times               = dict(state.get("node_times", {}))

    # ── Build col_id → table_id mapping from top_k ───────────────────────────
    col_to_table: Dict[str, str] = {c["col_id"]: c["table_id"] for c in top_k if c.get("col_id")}

    # ── Inject must_include col_ids into selected_col_ids ────────────────────
    existing = set(selected_col_ids)
    for mi in must_include:
        cid = mi.get("col_id", "")
        if cid and cid not in existing:
            selected_col_ids.append(cid)
            existing.add(cid)

    # ── Determine entity set ──────────────────────────────────────────────────
    ordered_table_ids: List[str] = []
    if primary_table_id:
        ordered_table_ids.append(primary_table_id)
    for tid in secondary_table_ids:
        if tid not in ordered_table_ids:
            ordered_table_ids.append(tid)

    # Also add tables from must_include that aren't already included
    for mi in must_include:
        tid = mi.get("table_id", "")
        if tid and tid not in ordered_table_ids:
            ordered_table_ids.append(tid)

    # ── Build entities ────────────────────────────────────────────────────────
    entities: List[Dict] = []
    for alias_idx, tid in enumerate(ordered_table_ids):
        alias    = f"t{alias_idx + 1}"
        cols_for = [
            {"col_id": cid}
            for cid in selected_col_ids
            if col_to_table.get(cid) == tid
        ]
        entities.append({"table_id": tid, "alias": alias, "columns": cols_for})

    # ── Build joins from join_path ────────────────────────────────────────────
    entity_table_set = set(ordered_table_ids)
    joins: List[Dict] = [
        {
            "from_table_id": j["from_table_id"],
            "from_col_id":   j["from_col_id"],
            "to_table_id":   j["to_table_id"],
            "to_col_id":     j["to_col_id"],
            "join_type":     j.get("join_type", "INNER"),
        }
        for j in join_path
        if (
            j.get("from_table_id") in entity_table_set
            and j.get("to_table_id") in entity_table_set
            and _is_uuid(j.get("from_col_id", ""))
            and _is_uuid(j.get("to_col_id", ""))
            and j.get("from_col_id") != j.get("to_col_id")
        )
    ]
    # Dedupe joins by (from_table_id, to_table_id) pair
    seen_join_pairs: set = set()
    deduped_joins = []
    for j in joins:
        pair = (j["from_table_id"], j["to_table_id"])
        if pair not in seen_join_pairs:
            seen_join_pairs.add(pair)
            deduped_joins.append(j)
    joins = deduped_joins

    # ── Aggregations ─────────────────────────────────────────────────────────
    aggregations = []
    if intent in ("COUNT", "AGGREGATE"):
        aggregations = [{"func": "COUNT", "col_id": "*", "alias": "count_result"}]

    # ── Group by / order by ───────────────────────────────────────────────────
    group_by = [{"col_id": group_by_col_id}] if group_by_col_id and _is_uuid(group_by_col_id) else []
    order_by = (
        [{"col_id": order_by_col_id, "direction": order_direction}]
        if order_by_col_id and _is_uuid(order_by_col_id)
        else []
    )

    ir_json = {
        "version":        "1.0",
        "intent":         intent,
        "entities":       entities,
        "filter_tree":    filter_tree or {"type": "AND", "children": []},
        "joins":          joins,
        "aggregations":   aggregations,
        "group_by":       group_by,
        "order_by":       order_by,
        "limit":          None,
        "confidence":     0.9 if not needs_clarification else 0.3,
        "schema_version": 1,
    }

    times["assemble_ir"] = round((time.time() - t0) * 1000, 1)
    return {
        "ir_json":    ir_json,
        "confidence": ir_json["confidence"],
        "node_times": times,
    }


# =============================================================================
# Conditional edge
# =============================================================================

def should_continue(state: VEDAQueryState) -> str:
    """Skip to assemble if needs_clarification, else run full pipeline."""
    if state.get("needs_clarification"):
        return "assemble"
    return "select_entity"
