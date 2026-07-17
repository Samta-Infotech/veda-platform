# =============================================================================
# query/slm_layer.py
# VEDA POC — Query Pipeline Layer 3: SLM (Small Language Model)
#
# Responsibility:
#   Converts the enriched L2 result into a structured IR JSON.
#   Uses Qwen-2.5-Coder-7B served locally via Ollama.
#   Client data never leaves the server — no external API calls.
#
# Input  : raw query, temporal_filter (L1), top_k_columns + join_path (L2)
# Output : SLMResult containing intent, complexity, IR JSON
#
# Ambiguous queries (needs_clarification=True) are logged to
# SLM_AMBIGUOUS_LOG_PATH but the pipeline proceeds (best-effort).
#
# IR JSON uses UUIDs only — column/table names never appear in the IR.
# Validator checks every UUID against the L2 result before returning.
# =============================================================================

import json
import re
import sys
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    SLM_TEMPERATURE,
    SLM_TIMEOUT_SECS,
    SLM_MAX_RETRIES,
    SLM_AMBIGUOUS_LOG_PATH,
    SLM_MAX_TOKENS,
    SLM_NUM_CTX,
    SLM_ENABLED,
    TOP_K_TO_LLM,
)

try:
    from config import (
        VALUE_FILTER_ENABLED,
        VALUE_FILTER_VALUE_ONLY,
        VALUE_FILTER_SKIP_BOOLEAN,
        IR_JOIN_FREE_ENABLED,
        SLM_PROMPT_INCLUDE_VALUES,
        SLM_PROMPT_MAX_VALUES,
    )
except ImportError as _e:
    import warnings as _w
    _w.warn(f"[slm_layer] Config import failed: {_e}. Features disabled.", stacklevel=2)
    VALUE_FILTER_ENABLED = VALUE_FILTER_VALUE_ONLY = VALUE_FILTER_SKIP_BOOLEAN = False
    IR_JOIN_FREE_ENABLED = SLM_PROMPT_INCLUDE_VALUES = False
    SLM_PROMPT_MAX_VALUES = 8
from ingestion.vector_store import RetrievalResult
from query.semantic_layer import JoinEdge
from query.temporal_parser import TemporalFilter
from utils.logger import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

def _is_uuid(s: Any) -> bool:
    return isinstance(s, str) and bool(_UUID_RE.match(s))


_GROUPED_COUNT_RE = re.compile(
    r'\bwith their\b|\bper\b|\bby each\b|\band their\b|\bfor each\b|\bbreakdown by\b',
    re.IGNORECASE,
)

# Count intent must also be present before the grouped-count instruction fires.
# "show X with their Y" is a SELECT+JOIN, not a COUNT — guard against false positives.
_COUNT_INTENT_RE = re.compile(
    r'\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b|\btally\b',
    re.IGNORECASE,
)


# =============================================================================
# Output structures
# =============================================================================

@dataclass
class SLMResult:
    intent:               str            # SELECT | COUNT | AGGREGATE
    complexity:           str            # SIMPLE | MODERATE | COMPLEX
    needs_clarification:  bool
    clarification_reason: Optional[str]
    confidence:           float
    ir_json:              Dict[str, Any]
    raw_response:         str            # full model output for debugging
    duration_ms:          float
    error:                Optional[str]  # populated if Ollama call failed
    validation_warnings:  List[str]      = field(default_factory=list)
    # Advisory-only (rule 16): the model's own one-sentence statement of the
    # business question, emitted BEFORE the IR. Presentation metadata — never
    # read by sql_builder/validation, never affects SQL correctness.
    business_intent:      Optional[str]  = None


# =============================================================================
# Prompt builder
# =============================================================================

_SYSTEM_PROMPT = """\
CRITICAL — UUID COMPLIANCE (read before anything else):
The user message contains a REFERENCE TABLE with UUIDs.
For every table_id and col_id in your output:
  COPY character-for-character from REFERENCE TABLE only.
  NEVER generate, infer, guess, or approximate any UUID.
  If a UUID is not in REFERENCE TABLE → it does not exist → omit it.

SELF-CHECK before writing output:
  → Every table_id: is it in REFERENCE TABLE? If NO → remove.
  → Every col_id: is it in REFERENCE TABLE? If NO → remove.

You are VEDA, a database query planner. Convert natural language queries into structured IR JSON for SQL generation.

RULES:
1. Output ONLY a single JSON object. No markdown fences, no explanation, no commentary.
2. In ir_json, reference tables and columns using ONLY their UUIDs (table_id, col_id). Never use names.
3. Only include columns directly relevant to the query. Omit irrelevant ones.
4. Set needs_clarification=true only if essential filtering information is completely missing.
5. Use temporal_filter values directly for date-range filters (BETWEEN operator).
6. If a join path is provided, include it in the joins array using the exact UUIDs given.
7. Counting queries — two cases:
   a. Pure count with no grouping ("how many users", "total incidents"): set intent="COUNT", add {"func":"COUNT","col_id":"*"} to aggregations, include only the one primary entity table, no joins needed.
   b. Count grouped by another entity ("number of X per Y", "count of X and their Y", "X by Y", "number of X with their Y"): set intent="COUNT" (NOT SELECT), add {"func":"COUNT","col_id":"*"} to aggregations, include BOTH the primary entity table AND the grouping table as entities, add the join between them, and add the grouping column to group_by as {"col_id":"<uuid>"}. For the primary table prefer the direct entity table over junction tables. CRITICAL: a FK column like organization_id inside the primary table is NOT a substitute for the grouping entity — you MUST add the related table (e.g. organizations) as a separate entity and group by its display column.
8. Prefer direct columns over joins. If the primary entity table already has a column that directly answers the query (e.g. workflow.module answers "their modules"), use that column directly. Do NOT introduce a join to a separate table with the same concept name.
9. Every col_id inside an entity's "columns" array MUST belong to that entity's table_id. Never place a col_id from table B inside entity A's columns list. If you need a column from another table, add that table as a separate entity and join to it.
10. In every join, from_col_id and to_col_id MUST be different UUIDs that belong to different tables. Using the same UUID for both sides of a join is always wrong.
11. When the query explicitly names a column concept (e.g. "email", "name", "status", "timestamp"), you MUST include a column matching that exact concept in entities.columns. Never substitute an explicitly named concept with a different column. If the user message includes a "Must-include columns" section, every col_id listed there MUST appear in your IR.
12. Filter values MUST be real literal values — never use placeholder strings like "<literal>", "<value>", or "<placeholder>". For boolean columns (is_active, is_enabled, is_deleted, etc.) use true or false. For status/category strings use the exact string (e.g. "active", "pending"). For numbers use the number directly. If you genuinely do not know the filter value, omit the filter condition entirely rather than using a placeholder.
13. Whenever entities[] contains two or more tables, joins[] MUST NOT be empty. Copy the matching join from the join path provided in the user message — use the exact from_table_id, from_col_id, to_table_id, to_col_id UUIDs as given. An IR with multiple entities and empty joins is always wrong and will produce an incorrect SQL query.
14. Value-matched filters: When a CATEGORY column has example_values and one of them matches a query word (case-insensitive), you MUST add a filter condition for that column: {"col_id": "<col_uuid>", "operator": "EQ", "value": "<exact_example_value>"}. The user message may include a "Value filter hints" section with pre-computed matches — always follow those hints exactly. Example: query "escalated incidents", column incident_status has example_values ["Escalated","Open"] → add {"col_id": incident_status_uuid, "operator": "EQ", "value": "Escalated"} to filter_tree.
15. The user message may include a "Recommended Projection" section — the business-relevant columns that should appear in entities[].columns (the SELECT clause) for that table. Prefer these over other columns in the REFERENCE TABLE for entities.columns. You MAY still add a column outside this list, but ONLY when the query explicitly names it (rule 11), or it is structurally required for an aggregation, group_by, order_by, or join key. Do not add other REFERENCE TABLE columns to entities.columns "just in case" — the REFERENCE TABLE remains fully available for filter_tree/joins/group_by/order_by regardless of what's in Recommended Projection.
16. business_intent is ADVISORY ONLY: state, in one short sentence, the business question being answered — BEFORE deciding entities/filters. It is presentation metadata: it must NEVER change or justify any entity, filter, join, aggregation, group_by, order_by, or limit. When business_intent and the query text disagree, the query text wins. Priority is always: SQL correctness > user intent > schema correctness > business presentation.

OUTPUT FORMAT (all top-level fields are required; state business_intent FIRST, then derive the IR from the query text):
{
  "business_intent": "<one short sentence: the business question being answered>",
  "intent": "SELECT",
  "complexity": "SIMPLE",
  "needs_clarification": false,
  "clarification_reason": null,
  "confidence": 0.9,
  "ir_json": {
    "version": "1.0",
    "intent": "SELECT",
    "entities": [
      {"table_id": "<uuid>", "alias": "t1", "columns": [{"col_id": "<uuid>"}]}
    ],
    "filter_tree": {
      "type": "AND",
      "children": [
        {"col_id": "<uuid>", "operator": "EQ", "value": true}
      ]
    },
    "joins": [
      {
        "from_table_id": "<uuid>", "from_col_id": "<uuid>",
        "to_table_id": "<uuid>",   "to_col_id": "<uuid>",
        "join_type": "INNER"
      }
    ],
    "aggregations": [],
    "group_by": [],
    "order_by": [],
    "limit": null,
    "confidence": 0.9,
    "schema_version": 1
  }
}

OPERATORS: EQ | NEQ | GT | GTE | LT | LTE | BETWEEN | LIKE | IN | NOT_EXISTS | IS_NULL
INTENTS: SELECT | COUNT | AGGREGATE
COMPLEXITY: SIMPLE | MODERATE | COMPLEX
JOIN TYPES: INNER | LEFT
ALIASES: always lowercase t1, t2, t3, ... — never uppercase
FORBIDDEN KEYS: never use "aggregates" (use "aggregations"), never use "type" or "function" in aggregation objects (use "func"), never use "field" (use "col_id"), never use alias-qualified strings in group_by or order_by (use {"col_id":"<uuid>"} objects, not "t1.column_name")\
"""


# Maps domain query words to table name equivalents for entity reordering.
# Mirrors the table-name relevant entries from semantic_layer._DOMAIN_SYNONYMS.
_TABLE_SYNONYMS: Dict[str, List[str]] = {
    "alert":          ["incident"],
    "alerts":         ["incident"],
    "case":           ["incident"],
    "cases":          ["incident"],
    "investigation":  ["incident"],
    "investigations": ["incident"],
    "escalated":      ["incident"],
    "flagged":        ["incident", "transaction"],
    "trail":          ["audit"],
}

_MUST_INCLUDE_STOPWORDS: frozenset = frozenset({
    "show", "list", "get", "find", "give", "me", "all", "the", "a", "an",
    "and", "or", "for", "of", "in", "on", "at", "to", "with", "by",
    "from", "that", "their", "each", "per", "where", "which", "is", "are",
    "was", "were", "has", "have", "its", "user", "users",
})


def _compute_must_include(
    query: str,
    columns: List[RetrievalResult],
    join_path: List[JoinEdge] = None,
) -> List[RetrievalResult]:
    """Return columns whose col_name is an exact keyword match in the query.

    Matches either exact col_name equality or all underscore-parts present
    (e.g. last_logged_in matches 'last logged in'). Dedupes by col_name,
    preferring columns that belong to the primary table or are connected via join_path.
    """
    if join_path is None:
        join_path = []

    query_words = {
        w for w in re.sub(r"[^\w]", " ", query.lower()).split()
        if w not in _MUST_INCLUDE_STOPWORDS and len(w) > 2
    }
    # Also include singular forms so "incidents" matches column part "incident",
    # "types" matches "type", "names" matches "name" (E5 fix — mirrors E3 in semantic_layer).
    _query_words_depl = query_words | {w[:-1] for w in query_words if w.endswith("s") and len(w) > 3}
    primary_table_id = columns[0].table_id if columns else None
    valid_table_ids = set()
    if primary_table_id:
        valid_table_ids.add(primary_table_id)
    for edge in join_path:
        valid_table_ids.add(edge.from_table_id)
        valid_table_ids.add(edge.to_table_id)

    candidates_by_keyword: Dict[str, List[RetrievalResult]] = {}

    for r in columns:
        col_lc    = r.col_name.lower()
        col_parts = set(col_lc.split("_"))
        is_primary = (r.table_id == primary_table_id)

        matched_keyword = None

        non_trivial = {p for p in col_parts if len(p) > 2}
        if col_lc in query_words:
            matched_keyword = col_lc
        elif non_trivial and non_trivial <= _query_words_depl:
            # Use depluralized set so "incidents" matches column part "incident" (E5 fix)
            matched_keyword = col_lc
        elif is_primary and len(col_parts) > 1:
            suffix = col_lc.split("_")[-1]
            if suffix in _query_words_depl:
                matched_keyword = suffix
                
        if matched_keyword:
            if matched_keyword not in candidates_by_keyword:
                candidates_by_keyword[matched_keyword] = []
            candidates_by_keyword[matched_keyword].append(r)

    result:   List[RetrievalResult] = []
    seen_ids: set = set()
    
    for keyword, cands in candidates_by_keyword.items():
        primary_cands = [c for c in cands if c.table_id == primary_table_id]
        if primary_cands:
            best = primary_cands[0]
        else:
            valid_cands = [c for c in cands if c.table_id in valid_table_ids]
            best = valid_cands[0] if valid_cands else cands[0]
            
        if best.col_id not in seen_ids:
            result.append(best)
            seen_ids.add(best.col_id)
            
    return result


def _inject_must_include_cols(
    ir_json:      Dict[str, Any],
    must_include: List[RetrievalResult],
    query:        str = "",
) -> None:
    """Inject must-include col_ids into IR entities in-place.

    For each must-include column not already in any entity's columns:
      - If the entity for its table_id exists, append to its columns list.
      - Otherwise add a new entity.

    Promotion rule: when a new entity is created for a column that is a direct
    keyword match in the query (e.g. 'incident_status' ↔ 'incident status'), and
    the current entity[0] is a different table, the new entity is inserted at
    position 0. This prevents a noisy high-similarity column from a CSV export or
    analytics table from becoming the L4 FROM seed instead of the intended table.

    This runs after _normalize_ir so L3 omissions are corrected deterministically.
    """
    if not must_include:
        return

    query_words: set = set()
    if query:
        query_words = {
            w for w in re.sub(r"[^\w]", " ", query.lower()).split()
            if w not in _MUST_INCLUDE_STOPWORDS and len(w) > 2
        }
        for word in list(query_words):
            for expansion in _TABLE_SYNONYMS.get(word, []):
                query_words.add(expansion)

    existing: set = set()
    entity_by_tid: Dict[str, Dict] = {}
    for ent in ir_json.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        tid = ent.get("table_id", "")
        if tid:
            entity_by_tid[tid] = ent
        for col in ent.get("columns") or []:
            if isinstance(col, dict) and col.get("col_id"):
                existing.add(col["col_id"])

    for r in must_include:
        if r.col_id in existing:
            continue
        tid = r.table_id
        if tid in entity_by_tid:
            ent = entity_by_tid[tid]
            if not isinstance(ent.get("columns"), list):
                ent["columns"] = []
            ent["columns"].append({"col_id": r.col_id})
        else:
            new_alias = f"t{len(ir_json.get('entities', [])) + 1}"
            new_ent   = {
                "table_id": tid,
                "alias":    new_alias,
                "columns":  [{"col_id": r.col_id}],
            }
            entities = ir_json.setdefault("entities", [])
            col_lc    = r.col_name.lower()
            col_parts = set(col_lc.split("_"))
            non_triv  = {p for p in col_parts if len(p) > 2}
            is_direct = bool(query_words and (
                col_lc in query_words
                or (non_triv and non_triv <= query_words)
            ))
            if is_direct and entities and entities[0].get("table_id") != tid:
                entities.insert(0, new_ent)
            else:
                entities.append(new_ent)
            entity_by_tid[tid] = new_ent
        existing.add(r.col_id)

    # Reorder: if a non-primary entity's table name is explicitly mentioned in
    # the query, promote it to entity[0] so L4 uses it as the FROM seed.
    # This fixes cases where a generic export/CSV table ranks #1 in L2 and
    # becomes entity[0] in the L3 output, even though the query is about a
    # specific named table (e.g. "incident status" → FROM incident, not permissions_list_export).
    ents = ir_json.get("entities") or []
    if len(ents) > 1 and query_words:
        tid_to_tname: Dict[str, str] = {r.table_id: r.table_name for r in must_include}

        def _tname_score(ent: dict) -> int:
            tname  = tid_to_tname.get(ent.get("table_id", ""), "")
            tparts = {p for p in re.sub(r"[^\w]", "_", tname.lower()).split("_")
                      if p and len(p) > 2}
            return len(tparts & query_words)

        current = _tname_score(ents[0])
        for i in range(1, len(ents)):
            if _tname_score(ents[i]) > current:
                ents.insert(0, ents.pop(i))
                break


def _sampled_values_for(col_id: str) -> list:
    """Return raw sample values for a column (for inclusion in the L3 prompt)."""
    try:
        from ingestion.value_sampler import _VALUE_STORE
        sc = _VALUE_STORE.get(col_id)
        if sc:
            return list(sc.raw_values)
    except Exception:
        pass
    return []


def _build_user_message(
    query:                str,
    temporal_filter:      Optional[TemporalFilter],
    top_k_columns:        List[RetrievalResult],
    join_path:            List[JoinEdge],
    must_include_results: List[RetrievalResult] = None,
    is_hybrid:            bool = False,
    recommended_projection_results: List[RetrievalResult] = None,
) -> str:
    parts: List[str] = []

    parts.append(f'Query: "{query}"')
    parts.append("")

    # Temporal filter
    if temporal_filter and (temporal_filter.start or temporal_filter.end):
        tf_dict = {}
        if temporal_filter.start:
            tf_dict["start"] = temporal_filter.start
        if temporal_filter.end:
            tf_dict["end"] = temporal_filter.end
        parts.append(f"Temporal filter: {json.dumps(tf_dict)}")
    else:
        parts.append("Temporal filter: null")
    parts.append("")

    # Available columns — explicit reference table format to reduce UUID hallucination
    _seen_table_ids: dict = {}
    for r in top_k_columns:
        if r.table_id not in _seen_table_ids:
            _seen_table_ids[r.table_id] = r.table_name

    parts.append("=== REFERENCE TABLE — copy UUIDs ONLY from here ===")
    parts.append("")
    parts.append("TABLE IDs (copy exactly as table_id):")
    for tid, tname in _seen_table_ids.items():
        parts.append(f"  {tid}  →  {tname}")
    parts.append("")
    parts.append("COLUMN IDs (copy exactly as col_id):")
    for r in top_k_columns:
        parts.append(
            f"  {r.col_id}  →  {r.table_name}.{r.col_name}  [{r.semantic_type}]"
        )
    parts.append("=== END REFERENCE TABLE ===")
    parts.append("")

    # Recommended Projection (2026-07): the business-facing SELECT list VEDA already
    # computes for Tier1 (veda/routing.py::recommended_projection — default display
    # column + this query's own retrieval relevance + HIGH-importance columns,
    # composed deterministically, no LLM). Rendered as a SEPARATE section from the
    # REFERENCE TABLE, same principle as generation.py::_recommended_projection_block
    # (Tier1's own prompt): this does not restrict SQL correctness — every REFERENCE
    # TABLE column stays fully usable for filter_tree/joins/group_by/order_by, this
    # only steers entities.columns (SELECT) toward the smaller business-relevant set.
    # Absent entirely when the caller has nothing to recommend — an existing caller
    # that never passes this sees an unchanged prompt, byte for byte.
    if recommended_projection_results:
        proj = [{"col_id": r.col_id, "col_name": r.col_name, "table_name": r.table_name}
                for r in recommended_projection_results]
        parts.append(
            "Recommended Projection — prefer these columns for entities.columns "
            "(the SELECT clause) per rule 15:"
        )
        parts.append(json.dumps(proj, indent=2))
        parts.append("")

    # Must-include columns: pre-computed from the full top_k list in run_slm.
    # Surfaced as a grounding hint so the SLM knows which cols are mandatory.
    if must_include_results:
        hint = [
            {"col_id": r.col_id, "col_name": r.col_name, "table_name": r.table_name}
            for r in must_include_results
        ]
        parts.append(
            "Must-include columns (exact keyword match from query — "
            "per rule 11 these MUST appear in IR entities.columns):"
        )
        parts.append(json.dumps(hint, indent=2))
        parts.append("")

    # Value filter hints: explicitly surface CATEGORY columns whose example_values
    # match query words so the SLM generates a filter_tree condition (rule 14).
    _vf_enabled = VALUE_FILTER_ENABLED
    _vf_value_only = VALUE_FILTER_VALUE_ONLY
    _vf_skip_bool  = VALUE_FILTER_SKIP_BOOLEAN

    if _vf_enabled:
        _q_words = set(re.sub(r"[^\w]", " ", query.lower()).split())
        _BOOL_HINT_TOKENS = {"true", "false", "yes", "no"}
        _filter_hints = []
        for r in top_k_columns:
            if getattr(r, "semantic_type", None) == "CATEGORY":
                _table_tokens = set((r.table_name or "").lower().replace("_", " ").split())
                _col_tokens   = set((r.col_name  or "").lower().replace("_", " ").split())
                _is_bool_col = _vf_skip_bool and any(
                    (r.col_name or "").lower().startswith(p)
                    for p in ("is_", "has_", "can_", "should_", "was_", "will_")
                )
                _vals = _sampled_values_for(r.col_id)
                for v in _vals:
                    if isinstance(v, str):
                        v_lower = v.lower()
                        if v_lower in _table_tokens:
                            continue
                        if _vf_value_only and v_lower in _col_tokens:
                            continue
                        if _is_bool_col and v_lower not in _BOOL_HINT_TOKENS:
                            continue
                        if v_lower in _q_words:
                            _filter_hints.append({
                                "col_id":     r.col_id,
                                "col_name":   r.col_name,
                                "table_name": r.table_name,
                                "operator":   "EQ",
                                "value":      v,
                            })
                            break
        if _filter_hints:
            parts.append(
                "Value filter hints — query words match these column values. "
                "Per rule 14 you MUST add each of these to filter_tree using the EXACT col_id shown below. "
                "Do NOT substitute a different col_id:"
            )
            parts.append(json.dumps(_filter_hints, indent=2))
            parts.append("")

    # Join paths
    if join_path:
        parts.append("Join paths available:")
        join_list = []
        for e in join_path:
            join_list.append({
                "from_table_id": e.from_table_id,
                "from_col_id":   e.from_col_id,
                "to_table_id":   e.to_table_id,
                "to_col_id":     e.to_col_id,
                "join_type":     e.join_type,
            })
        parts.append(json.dumps(join_list, indent=2))
    else:
        parts.append("Join paths available: []")

    if _GROUPED_COUNT_RE.search(query) and _COUNT_INTENT_RE.search(query):
        parts.append(
            "\nGROUPED COUNT INSTRUCTION (Rule 7b applies): "
            "This query uses a 'with their / per / by' grouping pattern. "
            "You MUST: (1) add BOTH the primary entity table AND the grouping entity table to entities[]; "
            "(2) copy the matching join from the Join paths section above; "
            "(3) add COUNT(*) to aggregations; "
            "(4) add the grouping entity's display column to group_by. "
            "Do NOT use a FK column (e.g. organization_id inside the role table) as a substitute — "
            "the grouping entity (e.g. organizations) MUST appear as its own entry in entities[]."
        )
        parts.append("")

    if IR_JOIN_FREE_ENABLED:
        parts.append(
            "\nJOIN-FREE MODE (Rule 6 + 13 override): "
            "Do NOT populate joins[]. Leave joins=[] empty in the IR. "
            "Rule 6 is disabled — ignore the Join paths section for IR construction. "
            "Rule 13 is disabled — multiple entities with empty joins is CORRECT here. "
            "The pipeline derives join keys from the FK adjacency graph automatically. "
            "Only list the tables (entities[]) and their columns you need. Do not guess join columns."
        )
        parts.append("")

    if is_hybrid:
        parts.append("\nHYBRID QUERY INSTRUCTION:")
        parts.append(
            "CRITICAL INSTRUCTION: This is a hybrid query (SQL + Document). "
            "DO NOT attempt to perform text searches inside the SQL database using `LIKE` on text columns. "
            "Specifically, you MUST NEVER apply any filters (in `filter_tree`) to columns with `semantic_type: FREE_TEXT`. "
            "All unstructured text and document searching is handled by the downstream RAG system. "
            "You MAY apply SQL filters for structured data (columns of type ID, CATEGORICAL, TEMPORAL, etc.), but filtering on FREE_TEXT will break the pipeline."
        )

    return "\n".join(parts)


# =============================================================================
# Ollama HTTP client
# =============================================================================

def _call_ollama(user_message: str) -> str:
    """
    Calls the configured SLM backend (§10 seam — Ollama dev / vLLM prod) with the
    IR system prompt. Returns the raw assistant response string.
    Raises RuntimeError on network / API errors.
    """
    try:
        from config import SLM_IR_MAX_TOKENS as _IR_CAP
    except Exception:
        _IR_CAP = SLM_MAX_TOKENS
    from slm import call_slm
    return call_slm(
        user_message,
        system=_SYSTEM_PROMPT,
        purpose="ir_emit",
        temperature=SLM_TEMPERATURE,
        num_predict=_IR_CAP,        # IR JSON is small; smaller decode → lower latency
        num_ctx=SLM_NUM_CTX,
        timeout=SLM_TIMEOUT_SECS,
    )


# =============================================================================
# Response parser + validator
# =============================================================================

def _extract_json(text: str) -> str:
    """
    Extracts the first top-level JSON object from the model response.
    Handles cases where the model wraps output in markdown fences.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find first '{' and matching '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth   = 0
    in_str  = False
    escape  = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    raise ValueError("Malformed JSON — unmatched braces")


def _validate_ir(
    parsed:        Dict[str, Any],
    top_k_columns: List[RetrievalResult],
    ir_json:       Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Validates that every UUID referenced in ir_json exists in top_k_columns.
    Returns a list of warning strings (empty = clean).
    """
    valid_col_ids   = {r.col_id   for r in top_k_columns}
    valid_table_ids = {r.table_id for r in top_k_columns}
    warnings: List[str] = []

    # Use pre-resolved ir_json if provided, otherwise extract from parsed
    ir = ir_json if ir_json is not None else parsed.get("ir_json", {})
    if not ir:
        return ["ir_json field missing or empty"]

    def _check_col(col_id: Any, location: str) -> None:
        if not isinstance(col_id, str) or not col_id or col_id == "*":
            return
        if col_id not in valid_col_ids:
            warnings.append(f"Unknown col_id {col_id!r} in {location}")

    def _check_table(table_id: Any, location: str) -> None:
        if not isinstance(table_id, str) or not table_id:
            return
        if table_id not in valid_table_ids:
            warnings.append(f"Unknown table_id {table_id!r} in {location}")

    # entities
    for ent in ir.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        _check_table(ent.get("table_id", ""), "entities")
        for col in ent.get("columns") or []:
            if isinstance(col, dict):
                _check_col(col.get("col_id", ""), "entities.columns")

    # filter_tree
    def _walk_filter(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "col_id" in node:
                _check_col(node["col_id"], path)
            for child in node.get("children") or []:
                _walk_filter(child, path + ".children")

    _walk_filter(ir.get("filter_tree") or {}, "filter_tree")

    # joins
    for j in ir.get("joins") or []:
        if not isinstance(j, dict):
            continue
        _check_table(j.get("from_table_id", ""), "joins")
        _check_col(j.get("from_col_id", ""),   "joins")
        _check_table(j.get("to_table_id", ""), "joins")
        _check_col(j.get("to_col_id", ""),     "joins")

    # aggregations / group_by / order_by — model may return plain UUID strings or null
    for agg in ir.get("aggregations") or []:
        if isinstance(agg, dict):
            _check_col(agg.get("col_id", ""), "aggregations")
    for gb in ir.get("group_by") or []:
        if isinstance(gb, dict):
            _check_col(gb.get("col_id", ""), "group_by")
    for ob in ir.get("order_by") or []:
        if isinstance(ob, dict):
            _check_col(ob.get("col_id", ""), "order_by")

    return warnings


# =============================================================================
# UUID hallucination guard — prune IR to valid L2 UUIDs
# =============================================================================

def _prune_hallucinated_uuids(
    ir_json:       Dict[str, Any],
    top_k_columns: List[RetrievalResult],
) -> List[str]:
    """
    Remove IR nodes whose UUIDs are not present in top_k_columns (in-place).

    Runs after _inject_must_include_cols so legitimately injected must-include
    columns (which may come from ranks 11-20 of the full top-K) are retained.
    Returns a list of pruned-item descriptions for warning reporting.
    """
    valid_col_ids   = {r.col_id   for r in top_k_columns}
    valid_table_ids = {r.table_id for r in top_k_columns}
    pruned: List[str] = []

    # entities — drop unknown tables; strip unknown cols within kept entities
    kept_entities = []
    for ent in ir_json.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        tid = ent.get("table_id", "")
        if tid and tid not in valid_table_ids:
            pruned.append(f"entity table_id={tid!r}")
            continue
        kept_cols = []
        for c in ent.get("columns") or []:
            cid = c.get("col_id") if isinstance(c, dict) else None
            if cid and cid not in valid_col_ids:
                pruned.append(f"col_id={cid!r} in table {tid!r}")
            else:
                kept_cols.append(c)
        ent["columns"] = kept_cols
        kept_entities.append(ent)
    ir_json["entities"] = kept_entities

    # joins — drop any join with an unknown table or column UUID
    kept_joins = []
    for j in ir_json.get("joins") or []:
        if not isinstance(j, dict):
            continue
        from_tid = j.get("from_table_id", "")
        to_tid   = j.get("to_table_id", "")
        from_cid = j.get("from_col_id", "")
        to_cid   = j.get("to_col_id", "")
        if (
            (from_tid and from_tid not in valid_table_ids)
            or (to_tid   and to_tid   not in valid_table_ids)
            or (from_cid and from_cid not in valid_col_ids)
            or (to_cid   and to_cid   not in valid_col_ids)
        ):
            pruned.append(f"join {from_tid!r}→{to_tid!r}")
        else:
            kept_joins.append(j)
    ir_json["joins"] = kept_joins

    # filter_tree — recursively remove conditions with unknown col_ids
    def _prune_filter(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if "col_id" in node:
            cid = node.get("col_id")
            if cid and cid not in valid_col_ids:
                pruned.append(f"filter col_id={cid!r}")
                return None
            return node
        children = node.get("children")
        if isinstance(children, list):
            node["children"] = [c for c in (_prune_filter(ch) for ch in children) if c is not None]
        return node

    if ir_json.get("filter_tree"):
        _prune_filter(ir_json["filter_tree"])

    # aggregations — drop entries with unknown col_ids (keep COUNT(*))
    kept_aggs = []
    for agg in ir_json.get("aggregations") or []:
        if isinstance(agg, dict):
            cid = agg.get("col_id", "*")
            if cid and cid != "*" and cid not in valid_col_ids:
                pruned.append(f"aggregation col_id={cid!r}")
                continue
        kept_aggs.append(agg)
    ir_json["aggregations"] = kept_aggs

    # group_by / order_by
    for key in ("group_by", "order_by"):
        kept = []
        for entry in ir_json.get(key) or []:
            if isinstance(entry, dict):
                cid = entry.get("col_id")
                if cid and cid not in valid_col_ids:
                    pruned.append(f"{key} col_id={cid!r}")
                    continue
            kept.append(entry)
        ir_json[key] = kept

    return pruned


# =============================================================================
# Ambiguous query logger
# =============================================================================

def _log_ambiguous(query: str, reason: Optional[str]) -> None:
    try:
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            SLM_AMBIGUOUS_LOG_PATH,
        )
        log_path = os.path.normpath(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"AMBIGUOUS: {query!r}  reason={reason!r}\n"
            )
    except OSError:
        pass   # logging failures must never crash the pipeline


# =============================================================================
# Fallback result — returned when Ollama is unavailable
# =============================================================================

def _fallback_result(query: str, error: str, duration_ms: float) -> SLMResult:
    return SLMResult(
        intent               = "SELECT",
        complexity           = "SIMPLE",
        needs_clarification  = False,
        clarification_reason = None,
        confidence           = 0.0,
        ir_json              = {
            "version":        "1.0",
            "intent":         "SELECT",
            "entities":       [],
            "filter_tree":    {"type": "AND", "children": []},
            "joins":          [],
            "aggregations":   [],
            "group_by":       [],
            "order_by":       [],
            "limit":          None,
            "confidence":     0.0,
            "schema_version": 1,
        },
        raw_response  = "",
        duration_ms   = duration_ms,
        error         = error,
    )


def _normalize_ir(ir_json: dict, intent: str) -> None:
    """
    Normalizes known LLM hallucination variants in-place.
    Called after setdefault passes so structural keys always exist.
    """
    # Force ir_json.intent to match outer intent — model sometimes
    # sets SELECT inside ir_json even when outer intent is COUNT.
    # Also clamp to valid values in case ir_json.intent has same
    # hallucination problem as the outer intent field.
    _VALID = {"SELECT", "COUNT", "AGGREGATE"}
    intent = intent if intent in _VALID else "SELECT"
    ir_json["intent"] = intent

    # Map alternate aggregation container key emitted by some model runs
    if ir_json.get("aggregates") and not ir_json.get("aggregations"):
        ir_json["aggregations"] = ir_json.pop("aggregates")

    # Normalize individual aggregation objects
    normalized: list = []
    for agg in ir_json.get("aggregations") or []:
        if not isinstance(agg, dict):
            continue
        norm = dict(agg)
        # Canonicalize function key: "func" is canonical; accept "type" or "function"
        if "func" not in norm:
            norm["func"] = norm.pop("type", None) or norm.pop("function", None) or "COUNT"
        # Canonicalize column key: "col_id" is canonical; accept "field"
        if "col_id" not in norm and "field" in norm:
            norm["col_id"] = norm.pop("field")
        normalized.append(norm)
    ir_json["aggregations"] = normalized

    # Strip entities with non-UUID table_id and strip non-UUID col_ids
    for ent in ir_json.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        tid = ent.get("table_id", "")
        if tid and not _is_uuid(tid):
            ent["table_id"] = ""   # L4 fallback handles via _infer_primary_table
        cols = ent.get("columns")
        if isinstance(cols, list):
            ent["columns"] = [
                c for c in cols
                if isinstance(c, dict) and _is_uuid(c.get("col_id", ""))
            ]

    # Strip filter_tree conditions whose col_id is not a valid UUID
    def _clean_filter(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if "col_id" in node:
            return node if _is_uuid(node.get("col_id", "")) else None
        children = node.get("children")
        if isinstance(children, list):
            node["children"] = [c for c in (_clean_filter(ch) for ch in children) if c is not None]
        return node

    if ir_json.get("filter_tree"):
        _clean_filter(ir_json["filter_tree"])

    # Strip joins where any UUID field is not a valid UUID
    joins = ir_json.get("joins")
    if isinstance(joins, list):
        ir_json["joins"] = [
            j for j in joins
            if isinstance(j, dict)
            and _is_uuid(j.get("from_col_id", ""))
            and _is_uuid(j.get("to_col_id", ""))
            and _is_uuid(j.get("from_table_id", ""))
            and _is_uuid(j.get("to_table_id", ""))
        ]

    # Strip group_by / order_by entries that are strings instead of {"col_id": uuid} dicts
    for key in ("group_by", "order_by"):
        lst = ir_json.get(key)
        if isinstance(lst, list):
            ir_json[key] = [
                entry for entry in lst
                if isinstance(entry, dict) and _is_uuid(entry.get("col_id", ""))
            ]


# =============================================================================
# Public entry point
# =============================================================================

def run_slm_layer(
    query:           str,
    temporal_filter: Optional[TemporalFilter],
    top_k_columns:   List[RetrievalResult],
    join_path:       List[JoinEdge],
    verbose:         bool = False,
    is_hybrid:       bool = False,
    recommended_projection: List[RetrievalResult] = None,
    on_event=None,
) -> SLMResult:
    """
    Layer 3 entry point. Calls the local SLM (Qwen via Ollama) to produce IR JSON.

    Parameters
    ----------
    query            : Original raw NL query (not cleaned — full intent preserved)
    temporal_filter  : L1 output — date range or None
    top_k_columns    : L2 output — ranked candidate columns with UUIDs
    join_path        : L2 output — inferred JOIN edges
    verbose          : Print debug info
    recommended_projection : optional business-facing SELECT-list guidance (subset of
                       top_k_columns) — see _build_user_message's "Recommended
                       Projection" block. None (default) leaves the prompt unchanged.

    Returns
    -------
    SLMResult — always returns (never raises). On error, confidence=0 and
                error field is populated. Pipeline can still proceed.
    """
    from config import USE_LANGGRAPH
    if USE_LANGGRAPH:
        from query.slm_langgraph import run_langgraph_pipeline
        return run_langgraph_pipeline(
            query=query, temporal_filter=temporal_filter,
            top_k_columns=top_k_columns, join_path=join_path,
            verbose=verbose, is_hybrid=is_hybrid,
            recommended_projection=recommended_projection, on_event=on_event)
    t0 = time.time()

    logger.debug("L3 SLM: query=%r, top_k_cols=%d, temporal=%s, model=%s",
                 query[:120], len(top_k_columns),
                 temporal_filter is not None, SLM_MODEL_NAME)

    if not SLM_ENABLED:
        logger.debug("L3 SLM skipped (SLM_ENABLED=False)")
        dur = round((time.time() - t0) * 1000, 2)
        return _fallback_result(query, "SLM_ENABLED=False in config", dur)

    if not top_k_columns:
        dur = round((time.time() - t0) * 1000, 2)
        return _fallback_result(query, "No columns from L2 — cannot generate IR", dur)

    # Cap columns passed to L3 at TOP_K_TO_LLM (default 10).
    # L2 retrieves TOP_K=20 for recall; L3 only needs the top-N for UUID
    # grounding. Passing all 20 causes O(n²) attention over UUID tokens
    # and UUID grounding failures on 7B models. Top-10 by RRF/cosine score
    # contains all semantically relevant columns.
    llm_columns   = top_k_columns[:TOP_K_TO_LLM]

    # Filter join_path to only edges between tables represented in llm_columns.
    # The full join_path can contain 60+ edges (all FK edges across all L2 tables).
    # Passing the full set overwhelms the 7B model — it responds by describing
    # the join array instead of generating IR JSON. Scoping to llm_table_ids
    # aligns the join context with the column context already capped above.
    llm_table_ids = {r.table_id for r in llm_columns}
    llm_join_path = [
        e for e in join_path
        if e.from_table_id in llm_table_ids
    ]

    must_include  = _compute_must_include(query, top_k_columns, join_path)  # full list, not capped

    # Recommended Projection must reference only columns actually present in the
    # REFERENCE TABLE sent this round (llm_columns, capped at TOP_K_TO_LLM) — a
    # recommended column outside that cap would violate the UUID-compliance rule
    # (col_id not in REFERENCE TABLE → the model is told to omit it).
    llm_col_ids = {r.col_id for r in llm_columns}
    rec_proj = [r for r in (recommended_projection or []) if r.col_id in llm_col_ids] or None

    user_msg = _build_user_message(query, temporal_filter, llm_columns, llm_join_path, must_include,
                                   is_hybrid, recommended_projection_results=rec_proj)

    if verbose:
        print(f"[SLM] Calling {SLM_MODEL_NAME} via Ollama...")
        print(f"  Columns passed : {len(llm_columns)} (capped from {len(top_k_columns)})")
        print(f"  Joins passed   : {len(llm_join_path)} (filtered from {len(join_path)})")
        print(f"  Temporal filter: {temporal_filter}")

    raw_response = ""
    parsed:      Any = None
    last_error   = ""

    for attempt in range(1, SLM_MAX_RETRIES + 2):   # +2 = 1 initial + retries
        try:
            raw_response = _call_ollama(user_msg)
            json_str     = _extract_json(raw_response)
            parsed       = json.loads(json_str)
            break  # success — both call and parse succeeded
        except RuntimeError as exc:
            last_error = str(exc)
            if verbose:
                print(f"  [attempt {attempt}] Ollama error: {last_error}")
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"JSON parse failed: {exc}. Raw: {raw_response[:200]}"
            if verbose:
                print(f"  [attempt {attempt}] {last_error}")
        if attempt > SLM_MAX_RETRIES:
            dur = round((time.time() - t0) * 1000, 2)
            return _fallback_result(query, last_error, dur)

    if parsed is None:
        dur = round((time.time() - t0) * 1000, 2)
        return _fallback_result(query, last_error or "No response from Ollama", dur)

    # --- Extract and clamp top-level fields ---
    # Clamp intent — model sometimes emits a natural language sentence or
    # camelCase method name instead of SELECT | COUNT | AGGREGATE.
    # e.g. 'Retrieve the username...' or 'RetrieveWorkflowHistory'
    _VALID_INTENTS     = {"SELECT", "COUNT", "AGGREGATE"}
    _VALID_COMPLEXITY  = {"SIMPLE", "MODERATE", "COMPLEX"}

    raw_intent    = parsed.get("intent", "SELECT")
    intent        = raw_intent if raw_intent in _VALID_INTENTS else "SELECT"

    raw_complexity = parsed.get("complexity", "SIMPLE")
    complexity     = raw_complexity if raw_complexity in _VALID_COMPLEXITY else "SIMPLE"

    # Infer complexity from structural signals when model returns SIMPLE
    # for queries that clearly need multi-table joins or aggregations.
    # This corrects the deterministic collapse at low temperature where
    # every query gets classified SIMPLE.
    if complexity == "SIMPLE":
        n_cols   = len(llm_columns)
        n_tables = len({c.table_id for c in llm_columns})
        n_joins  = len(join_path)
        tf_present = temporal_filter is not None
        if n_tables >= 3 or (n_joins >= 2 and n_cols >= 6):
            complexity = "COMPLEX"
        elif n_tables >= 2 or n_joins >= 1 or tf_present:
            complexity = "MODERATE"
    needs_clarification  = bool(parsed.get("needs_clarification", False))
    clarification_reason = parsed.get("clarification_reason")
    confidence           = float(parsed.get("confidence", 0.5))
    ir_json              = parsed.get("ir_json") or {}

    # Ensure ir_json has required structure
    ir_json.setdefault("version", "1.0")
    ir_json.setdefault("schema_version", 1)
    ir_json.setdefault("entities", [])
    ir_json.setdefault("filter_tree", {"type": "AND", "children": []})
    ir_json.setdefault("joins", [])
    ir_json.setdefault("aggregations", [])
    ir_json.setdefault("group_by", [])
    ir_json.setdefault("order_by", [])
    ir_json.setdefault("limit", None)

    # Normalize LLM hallucination variants (alternate key names, wrong intent)
    _normalize_ir(ir_json, intent)

    # Inject must-include columns that L3 missed (e.g. col ranked below TOP_K_TO_LLM
    # or deprioritised despite being an exact keyword match in the query).
    _inject_must_include_cols(ir_json, must_include, query=query)

    # Model sometimes returns IR directly at top level without the outer wrapper.
    # Detect this by checking if entities/filter_tree live in parsed itself.
    if not ir_json and "entities" in parsed:
        ir_json = parsed
        ir_json.setdefault("version", "1.0")
        ir_json.setdefault("schema_version", 1)
        ir_json.setdefault("filter_tree", {"type": "AND", "children": []})
        ir_json.setdefault("joins", [])
        ir_json.setdefault("aggregations", [])
        ir_json.setdefault("group_by", [])
        ir_json.setdefault("order_by", [])
        ir_json.setdefault("limit", None)
        _normalize_ir(ir_json, intent)
        _inject_must_include_cols(ir_json, must_include, query=query)

    # --- UUID hallucination guard ---
    # Strip any entity/join/filter/agg/group_by reference whose UUID is not
    # present in top_k_columns (the full L2 result set, not the capped llm_columns).
    # _normalize_ir already removed non-UUID-format strings; this removes valid-format
    # UUIDs that the model hallucinated and that do not correspond to any real column.
    pruned_uuids = _prune_hallucinated_uuids(ir_json, top_k_columns)
    if pruned_uuids and verbose:
        print(f"  [SLM] Pruned {len(pruned_uuids)} hallucinated UUID refs: {pruned_uuids[:5]}")

    # --- Validate UUIDs ---
    # Validate against top_k_columns (full L2 set) so must-include cols injected
    # from ranks beyond TOP_K_TO_LLM do not produce false-positive warnings.
    warnings = _validate_ir(parsed, top_k_columns, ir_json)
    if warnings and verbose:
        print(f"  [SLM] Validation warnings: {warnings}")

    # --- Log ambiguous queries ---
    if needs_clarification:
        _log_ambiguous(query, clarification_reason)
        if verbose:
            print(f"  [SLM] Ambiguous query logged: {clarification_reason!r}")

    dur = round((time.time() - t0) * 1000, 2)

    if verbose:
        clamp_note = f" (clamped from '{raw_intent}')" if raw_intent != intent else ""
        print(f"  [SLM] intent={intent}{clamp_note}  complexity={complexity}  confidence={confidence}")
        print(f"  [SLM] Duration: {dur}ms")
        print(f"  [SLM] IR entities: {len(ir_json.get('entities', []))}")
        print(f"  [SLM] IR filters : {len((ir_json.get('filter_tree') or {}).get('children', []))}")
        print(f"  [SLM] IR joins   : {len(ir_json.get('joins', []))}")
        print(f"  [SLM] Pruned UUIDs: {len(pruned_uuids)}")
        print(f"  [SLM] Warnings   : {len(warnings)}")

    logger.info(
        "L3 SLM: intent=%s, complexity=%s, confidence=%.3f, entities=%d, joins=%d, "
        "warnings=%d, %dms",
        intent, complexity, confidence,
        len(ir_json.get("entities", [])),
        len(ir_json.get("joins", [])),
        len(warnings), dur,
    )
    logger.debug("L3 IR JSON: %s", json.dumps(ir_json)[:500])

    return SLMResult(
        intent               = intent,
        complexity           = complexity,
        needs_clarification  = needs_clarification,
        clarification_reason = clarification_reason,
        confidence           = confidence,
        ir_json              = ir_json,
        raw_response         = raw_response,
        duration_ms          = dur,
        error                = None,
        validation_warnings  = warnings,
        business_intent      = (str(parsed.get("business_intent") or "").strip() or None),
    )


# =============================================================================
# Compound-query decomposition (LLM-only; this is the ONLY file that may call
# Ollama). Splits ONE utterance that carries MULTIPLE INDEPENDENT questions into
# standalone sub-queries; the front door runs each through the existing single-
# query pipeline. See query/multi_result.py and veda_hybrid.run_hybrid_query.
#
# Contract (closed enum — anything else is coerced to "single"):
#   single           → one question (incl. ordinary joins / filtered counts).
#                      DO NOT split. sub_queries == [original].
#   independent      → ≥2 questions that DO NOT depend on each other's answer.
#                      THE ONLY case that splits. sub_queries are self-contained
#                      (pronouns/ellipsis resolved so each stands alone).
#   dependent_nested → ≥2 parts where one needs another's RESULT
#                      ("users in the team with the MOST open incidents").
#                      DO NOT split — these recompose into one query, which is
#                      out of scope for v1; the caller refuses (refuse-over-guess:
#                      a mis-split here is a silent WRONG answer).
#
# Discipline: WHEN UNSURE, return "single". A missed split is a rephrasable
# refusal; a mis-split silently answers the wrong question.
# =============================================================================

DECOMP_SINGLE = "single"
DECOMP_INDEPENDENT = "independent"
DECOMP_DEPENDENT = "dependent_nested"
_DECOMP_TYPES = frozenset({DECOMP_SINGLE, DECOMP_INDEPENDENT, DECOMP_DEPENDENT})


@dataclass
class DecomposeResult:
    type: str                                       # closed enum (_DECOMP_TYPES)
    sub_queries: List[str] = field(default_factory=list)
    error: Optional[str] = None                     # set when Ollama was unreachable etc.
    confidence: Optional[float] = None              # model self-report — LOGGED, not trusted

    @property
    def should_split(self) -> bool:
        return self.type == DECOMP_INDEPENDENT and len(self.sub_queries) >= 2


_DECOMPOSE_SYSTEM_PROMPT = """You split a user's database question into independent sub-questions.

Return ONLY a JSON object, no prose:
{"type": "<single|independent|dependent_nested>", "sub_queries": ["...", "..."], "confidence": 0.0}
(confidence: your 0..1 certainty in the type.)

Decide the type:
- "single": ONE question. This includes a question with several filters, or one that
  joins related tables ("users and their open incidents"). Do NOT split. Return the
  original question as the only element of sub_queries.
- "independent": TWO OR MORE questions joined in one sentence that DO NOT need each
  other's answer — they would be answered by separate queries. Put each as a complete,
  standalone question in sub_queries (rewrite pronouns/shared words so each stands on
  its own). Example:
    "how many incidents are open and list the active users"
    -> {"type":"independent","sub_queries":["how many incidents are open","list the active users"]}
- "dependent_nested": parts where one needs the RESULT of another to be answered
  ("the users in the team with the most open incidents" — you must first find the team).
  Do NOT split for execution, but in sub_queries list the parts IN THE ORDER they must
  be asked (the lookup first, then the dependent question) so the user can ask them
  separately. Example:
    "who are the users in the team with the most open incidents"
    -> {"type":"dependent_nested","sub_queries":["which team has the most open incidents","who are the users in that team"]}

When in doubt, choose "single". Never invent questions the user did not ask."""


def _call_ollama_decompose(query: str) -> str:
    """Decomposer Ollama call. Mirrors _call_ollama but uses the decomposer system
    prompt and a tiny decode budget (the output is a small JSON object). Raises
    RuntimeError on network/API errors — the caller degrades to 'single'."""
    from slm import call_slm
    return call_slm(
        query,
        system=_DECOMPOSE_SYSTEM_PROMPT,
        purpose="decompose",
        temperature=0.0,               # deterministic split decision
        num_predict=256,
        num_ctx=SLM_NUM_CTX,
        timeout=SLM_TIMEOUT_SECS,
    )


def _conf(parsed: Dict[str, Any]) -> Optional[float]:
    try:
        return float(parsed.get("confidence"))
    except (TypeError, ValueError):
        return None


def _coerce_decompose(parsed: Dict[str, Any], query: str) -> DecomposeResult:
    """Validate the LLM's split against the closed enum. Unknown type → single.
    'independent' with <2 usable subs is downgraded to single (nothing to split)."""
    dtype = parsed.get("type")
    conf = _conf(parsed)
    if dtype not in _DECOMP_TYPES:
        return DecomposeResult(type=DECOMP_SINGLE, sub_queries=[query], confidence=conf)

    if dtype == DECOMP_INDEPENDENT:
        subs = [str(s).strip() for s in (parsed.get("sub_queries") or [])
                if str(s).strip()]
        if len(subs) < 2:
            return DecomposeResult(type=DECOMP_SINGLE, sub_queries=[query], confidence=conf)
        return DecomposeResult(type=DECOMP_INDEPENDENT, sub_queries=subs, confidence=conf)

    if dtype == DECOMP_DEPENDENT:
        # Never split for execution, but KEEP the ordered parts the model identified —
        # the caller turns them into a guided refusal ("ask these in order") instead of
        # a dead-end. Falls back to the original if the model gave no usable parts.
        parts = [str(s).strip() for s in (parsed.get("sub_queries") or []) if str(s).strip()]
        return DecomposeResult(type=DECOMP_DEPENDENT,
                               sub_queries=parts if len(parts) >= 2 else [query],
                               confidence=conf)

    # single → carry the original through.
    return DecomposeResult(type=DECOMP_SINGLE, sub_queries=[query], confidence=conf)


def _log_decompose(query: str, res: "DecomposeResult") -> None:
    """Best-effort append to the split-decision log. This record IS the eval data:
    (query, predicted type, sub_queries) → the labelled set for measuring split
    accuracy from real traffic (evaluation/decompose_eval.py). Never raises."""
    try:
        from config import DECOMPOSE_LOG_ENABLED, DECOMPOSE_LOG_PATH
        if not DECOMPOSE_LOG_ENABLED:
            return
        try:
            from config import ROUTE_LOG_INCLUDE_QUERY as _inc
        except Exception:
            _inc = True
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = DECOMPOSE_LOG_PATH if os.path.isabs(DECOMPOSE_LOG_PATH) \
            else os.path.join(root, DECOMPOSE_LOG_PATH)
        rec = {"t": round(time.time(), 3), "type": res.type,
               "n_sub": len(res.sub_queries), "confidence": res.confidence,
               "error": res.error}
        if _inc:
            rec["query"] = query
            rec["sub_queries"] = res.sub_queries
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def run_decomposer(query: str, verbose: bool = False) -> DecomposeResult:
    """Decide whether `query` is one question or several independent ones.

    Safe by construction: any failure (Ollama down, malformed JSON, disabled SLM)
    returns type='single' so the caller's behaviour is EXACTLY as it is today.
    Every decision is logged (the eval data — see _log_decompose)."""
    if not SLM_ENABLED:
        res = DecomposeResult(type=DECOMP_SINGLE, sub_queries=[query], error="SLM disabled")
        _log_decompose(query, res)
        return res

    # Retry like the main SLM path (SLM_MAX_RETRIES): a transient blip — a remote Ollama
    # briefly unreachable ("no route to host"), a cold model load, a malformed first
    # response — shouldn't silently collapse a compound query to single. "No route to
    # host" fails fast, so retries are cheap when the host is genuinely down.
    res = None
    last_error = ""
    for attempt in range(1, SLM_MAX_RETRIES + 2):   # +2 = 1 initial + retries
        try:
            raw = _call_ollama_decompose(query)
            res = _coerce_decompose(json.loads(_extract_json(raw)), query)
            break
        except Exception as exc:                    # network, JSON, parse — retry, then single
            last_error = str(exc)
            if verbose:
                print(f"  [decompose attempt {attempt}] {type(exc).__name__}: {last_error[:120]}")
    if res is None:
        if verbose:
            print(f"  [decompose] unavailable after {SLM_MAX_RETRIES + 1} attempts "
                  f"— treating as single ({last_error[:80]})")
        res = DecomposeResult(type=DECOMP_SINGLE, sub_queries=[query], error=last_error[:140])
    _log_decompose(query, res)
    if verbose and res.type != DECOMP_SINGLE:
        print(f"  [decompose] {res.type}: {res.sub_queries}")
    return res