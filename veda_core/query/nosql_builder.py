# =============================================================================
# query/nosql_builder.py
# VEDA — NoSQL Query Builder (Phase 4)
#
# Responsibility:
#   - Accepts a natural language query + NoSQLCollection schema list
#   - Picks the most relevant collection from the schema
#   - Detects query intent: find | count | aggregate
#   - Extracts field-value filters from the query text
#   - Produces an engine-native query dict (MongoDB / Elasticsearch / DynamoDB)
#   - Returns the dict serialised as JSON for connector.execute_query()
#
# Phase 4 approach:
#   Primary path  — keyword extraction from the NL query against inferred field names
#   IR JSON path  — if ir_json is supplied (future: when nosql schema is in pgvector),
#                   filter_tree is translated directly to native predicates
#
# No LLM calls — query construction is deterministic.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from connectors.base import NoSQLCollection


# =============================================================================
# Output dataclass
# =============================================================================

@dataclass
class NoSQLBuilderResult:
    engine:          str
    source_id:       str
    collection_name: Optional[str]
    query_dict:      Optional[dict]   # native query as Python dict
    query_json:      Optional[str]    # JSON string for connector.execute_query()
    intent:          str              # "find" | "count" | "aggregate"
    field_hints:     List[str]        # field names the query targets
    warnings:        List[str]
    error:           Optional[str]
    duration_ms:     float


# =============================================================================
# Intent detection
# =============================================================================

_COUNT_SIGNALS = {
    "count", "how many", "total number", "number of",
    "tally", "how much",
}

_AGGREGATE_SIGNALS = {
    "sum", "average", "avg", "maximum", "minimum",
    "group by", "per ", "breakdown", "distribution",
}


def _detect_intent(query: str) -> str:
    q = query.lower()
    if any(sig in q for sig in _COUNT_SIGNALS):
        return "count"
    if any(sig in q for sig in _AGGREGATE_SIGNALS):
        return "aggregate"
    return "find"


# =============================================================================
# Collection selection
# =============================================================================

def _collection_score(query_lower: str, col: NoSQLCollection) -> float:
    """
    Scores a collection for relevance to the query.
    Checks collection name and inferred field names against query tokens.
    """
    score = 0.0
    tokens = set(re.split(r"\W+", query_lower)) - {"", "the", "a", "an", "of", "in", "for"}

    col_name_lower = col.collection_name.lower()
    if col_name_lower in query_lower:
        score += 3.0
    for tok in tokens:
        if len(tok) > 2 and tok in col_name_lower:
            score += 1.5

    field_names = {f["name"].lower() for f in col.inferred_fields}
    for tok in tokens:
        if tok in field_names:
            score += 1.0

    return score


def _pick_collection(
    query: str,
    collections: List[NoSQLCollection],
) -> Optional[NoSQLCollection]:
    if not collections:
        return None
    if len(collections) == 1:
        return collections[0]
    q = query.lower()
    scored = [(c, _collection_score(q, c)) for c in collections]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# =============================================================================
# Filter extraction from natural language
# =============================================================================

# Patterns: "<field> <op> <value>" in the query text
_FILTER_PATTERNS = [
    # equality: "status is open" / "status = open" / "status are active"
    (r"\b(\w+)\s+(?:is|are|=|equals?)\s+['\"]?(\w[\w\s-]*?)['\"]?\b",    "eq"),
    # not equal: "status is not closed"
    (r"\b(\w+)\s+(?:is\s+not|!=|not\s+equals?)\s+['\"]?(\w[\w\s-]*?)['\"]?\b", "ne"),
    # greater than: "age > 30" / "count greater than 5"
    (r"\b(\w+)\s+(?:>|greater\s+than)\s+(\d+(?:\.\d+)?)\b",               "gt"),
    # less than: "price < 100" / "price less than 100"
    (r"\b(\w+)\s+(?:<|less\s+than)\s+(\d+(?:\.\d+)?)\b",                  "lt"),
    # contains: "name contains John" / "description like %active%"
    (r"\b(\w+)\s+(?:contains?|like|includes?)\s+['\"]?(\w[\w\s-]*?)['\"]?\b", "like"),
]


def _extract_filters(
    query: str,
    collection: NoSQLCollection,
) -> List[Tuple[str, str, Any]]:
    """
    Extracts (field_name, operator, value) triples from the query.
    Only matches against fields that exist in the collection schema.
    """
    field_names = {f["name"].lower(): f["name"] for f in collection.inferred_fields}
    results: List[Tuple[str, str, Any]] = []
    seen: set = set()

    for pattern, op in _FILTER_PATTERNS:
        for m in re.finditer(pattern, query, re.IGNORECASE):
            raw_field = m.group(1).lower()
            raw_value = m.group(2).strip()

            # only use if field exists in schema
            if raw_field not in field_names:
                continue
            key = (raw_field, op, raw_value)
            if key in seen:
                continue
            seen.add(key)

            # coerce numeric values
            field_type = next(
                (f["type"] for f in collection.inferred_fields
                 if f["name"].lower() == raw_field),
                "varchar",
            )
            value: Any = raw_value
            if field_type in ("integer", "bigint", "smallint"):
                try:
                    value = int(raw_value)
                except ValueError:
                    pass
            elif field_type in ("numeric", "double"):
                try:
                    value = float(raw_value)
                except ValueError:
                    pass

            results.append((field_names[raw_field], op, value))

    return results


# =============================================================================
# MongoDB query construction
# =============================================================================

def _build_mongo_filter(filters: List[Tuple[str, str, Any]]) -> dict:
    if not filters:
        return {}
    clauses = []
    for field_name, op, value in filters:
        if op == "eq":
            clauses.append({field_name: {"$eq": value}})
        elif op == "ne":
            clauses.append({field_name: {"$ne": value}})
        elif op == "gt":
            clauses.append({field_name: {"$gt": value}})
        elif op == "lt":
            clauses.append({field_name: {"$lt": value}})
        elif op == "like":
            pattern = str(value).replace("%", ".*")
            clauses.append({field_name: {"$regex": pattern, "$options": "i"}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_mongo_query(
    collection: NoSQLCollection,
    intent:     str,
    filters:    List[Tuple[str, str, Any]],
    group_field: Optional[str] = None,
) -> dict:
    filter_doc = _build_mongo_filter(filters)
    if intent == "count":
        return {
            "collection": collection.collection_name,
            "intent":     "count",
            "filter":     filter_doc,
        }
    if intent == "aggregate" and group_field:
        return {
            "collection": collection.collection_name,
            "intent":     "aggregate",
            "pipeline": [
                {"$match":  filter_doc} if filter_doc else {"$match": {}},
                {"$group":  {"_id": f"${group_field}", "count": {"$sum": 1}}},
                {"$sort":   {"count": -1}},
                {"$limit":  50},
            ],
        }
    return {
        "collection": collection.collection_name,
        "intent":     "find",
        "filter":     filter_doc,
    }


# =============================================================================
# Elasticsearch query construction
# =============================================================================

def _build_es_filter(filters: List[Tuple[str, str, Any]]) -> dict:
    if not filters:
        return {"match_all": {}}
    must = []
    for field_name, op, value in filters:
        if op == "eq":
            must.append({"term": {field_name: value}})
        elif op == "ne":
            must.append({"bool": {"must_not": [{"term": {field_name: value}}]}})
        elif op == "gt":
            must.append({"range": {field_name: {"gt": value}}})
        elif op == "lt":
            must.append({"range": {field_name: {"lt": value}}})
        elif op == "like":
            must.append({"match": {field_name: str(value)}})
    if len(must) == 1:
        return must[0]
    return {"bool": {"must": must}}


def _build_es_query(
    collection: NoSQLCollection,
    intent:     str,
    filters:    List[Tuple[str, str, Any]],
    group_field: Optional[str] = None,
) -> dict:
    es_filter = _build_es_filter(filters)
    body: dict = {"query": es_filter}

    if intent == "count":
        # count intent: return 0 hits + total
        body["size"] = 0

    if intent == "aggregate" and group_field:
        body["size"] = 0
        body["aggs"] = {
            "group_by": {
                "terms": {"field": group_field, "size": 50}
            }
        }

    return {
        "index": collection.collection_name,
        "body":  body,
    }


# =============================================================================
# DynamoDB query construction
# =============================================================================

def _build_dynamodb_query(
    collection: NoSQLCollection,
    intent:     str,
    filters:    List[Tuple[str, str, Any]],
) -> dict:
    q: dict = {
        "table":  collection.collection_name,
        "intent": "count" if intent == "count" else "scan",
    }

    if filters and intent != "count":
        parts = []
        expr_values: dict = {}
        expr_names:  dict = {}
        for i, (field_name, op, value) in enumerate(filters):
            ph    = f":v{i}"
            alias = f"#f{i}"
            expr_names[alias] = field_name
            expr_values[ph]   = value
            if op == "eq":
                parts.append(f"{alias} = {ph}")
            elif op == "ne":
                parts.append(f"{alias} <> {ph}")
            elif op == "gt":
                parts.append(f"{alias} > {ph}")
            elif op == "lt":
                parts.append(f"{alias} < {ph}")
            elif op == "like":
                parts.append(f"contains({alias}, {ph})")
        if parts:
            q["FilterExpression"]          = " AND ".join(parts)
            q["ExpressionAttributeValues"] = expr_values
            q["ExpressionAttributeNames"]  = expr_names

    return q


# =============================================================================
# IR JSON → filter extraction (future: when nosql schema is in pgvector)
# =============================================================================

def _ir_filter_tree_to_filters(
    filter_tree:  dict,
    col_id_map:   Dict[str, Tuple[str, str]],  # col_id → (field_name, field_type)
) -> List[Tuple[str, str, Any]]:
    """
    Recursively converts IR JSON filter_tree into (field, op, value) triples.
    col_id_map maps UUID col_ids to (field_name, field_type) from top_k_columns.
    """
    if not filter_tree:
        return []

    node_type = filter_tree.get("type", "").upper()

    if node_type in ("AND", "OR"):
        results = []
        for child in filter_tree.get("conditions", []):
            results.extend(_ir_filter_tree_to_filters(child, col_id_map))
        return results

    col_id = filter_tree.get("col_id", "")
    if col_id not in col_id_map:
        return []

    field_name, field_type = col_id_map[col_id]
    value = filter_tree.get("value")

    # coerce type
    if field_type in ("integer", "bigint", "smallint"):
        try:
            value = int(value)
        except (TypeError, ValueError):
            pass
    elif field_type in ("numeric", "double"):
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass

    op_map = {
        "EQ":   "eq",  "NEQ": "ne",  "NE":   "ne",
        "GT":   "gt",  "LT":  "lt",
        "GTE":  "gt",  "LTE": "lt",
        "LIKE": "like", "CONTAINS": "like",
    }
    op = op_map.get(node_type, "eq")
    return [(field_name, op, value)]


# =============================================================================
# Public entry point
# =============================================================================

def run_nosql_builder(
    query:       str,
    source_id:   str,
    engine:      str,
    collections: List[NoSQLCollection],
    ir_json:     Optional[dict] = None,
    top_k_cols:  Optional[list] = None,
    verbose:     bool = False,
) -> NoSQLBuilderResult:
    """
    Builds a native NoSQL query from a natural language query string.

    Parameters
    ----------
    query       : raw NL query
    source_id   : VEDA_SOURCES entry id
    engine      : "mongodb" | "elasticsearch" | "dynamodb"
    collections : list of NoSQLCollection from connector.get_nosql_schema()
    ir_json     : optional IR JSON from L3 (used for structured filter extraction
                  when nosql schema is present in pgvector — Phase 5+)
    top_k_cols  : optional top-K column results from L2 (used with ir_json path)
    verbose     : print debug info

    Returns
    -------
    NoSQLBuilderResult — always returns; error field set on failure
    """
    t0       = time.time()
    warnings: List[str] = []

    if not collections:
        return NoSQLBuilderResult(
            engine=engine, source_id=source_id,
            collection_name=None, query_dict=None, query_json=None,
            intent="find", field_hints=[], warnings=[],
            error="No collections available from source schema",
            duration_ms=0.0,
        )

    # Pick best collection
    col = _pick_collection(query, collections)
    if col is None:
        return NoSQLBuilderResult(
            engine=engine, source_id=source_id,
            collection_name=None, query_dict=None, query_json=None,
            intent="find", field_hints=[], warnings=[],
            error="Could not select a collection for this query",
            duration_ms=0.0,
        )

    intent = _detect_intent(query)

    # ------------------------------------------------------------------
    # IR JSON path — structured filter extraction via UUID resolution
    # ------------------------------------------------------------------
    filters: List[Tuple[str, str, Any]] = []

    if ir_json and top_k_cols:
        col_id_map: Dict[str, Tuple[str, str]] = {
            r.col_id: (r.col_name, r.data_type if hasattr(r, "data_type") else "varchar")
            for r in top_k_cols
        }
        ft = ir_json.get("filter_tree")
        if ft:
            filters = _ir_filter_tree_to_filters(ft, col_id_map)
            if verbose:
                print(f"[NoSQLBuilder] IR JSON path: {len(filters)} filter(s) from filter_tree")

    # ------------------------------------------------------------------
    # Heuristic path — keyword extraction from query text
    # ------------------------------------------------------------------
    if not filters:
        filters = _extract_filters(query, col)
        if verbose:
            print(f"[NoSQLBuilder] Heuristic path: {len(filters)} filter(s) extracted")

    field_hints = [f for f, _, _ in filters]

    # Detect GROUP BY field for aggregation
    group_field: Optional[str] = None
    if intent == "aggregate":
        per_match = re.search(r"\bper\s+(\w+)\b", query, re.IGNORECASE)
        group_match = re.search(r"\bgroup(?:ed)?\s+by\s+(\w+)\b", query, re.IGNORECASE)
        m = per_match or group_match
        if m:
            candidate = m.group(1).lower()
            for f in col.inferred_fields:
                if f["name"].lower() == candidate:
                    group_field = f["name"]
                    break
        if not group_field:
            warnings.append(
                "Aggregate intent detected but no GROUP BY field found — falling back to find"
            )
            intent = "find"

    if verbose:
        print(f"[NoSQLBuilder] collection='{col.collection_name}'  "
              f"intent={intent}  filters={filters}")

    # ------------------------------------------------------------------
    # Build engine-native query
    # ------------------------------------------------------------------
    try:
        if engine == "mongodb":
            q_dict = _build_mongo_query(col, intent, filters, group_field)
        elif engine == "elasticsearch":
            q_dict = _build_es_query(col, intent, filters, group_field)
        elif engine == "dynamodb":
            q_dict = _build_dynamodb_query(col, intent, filters)
        else:
            return NoSQLBuilderResult(
                engine=engine, source_id=source_id,
                collection_name=col.collection_name,
                query_dict=None, query_json=None,
                intent=intent, field_hints=field_hints,
                warnings=warnings,
                error=f"Unsupported engine '{engine}'",
                duration_ms=round((time.time() - t0) * 1000, 2),
            )

        q_json = json.dumps(q_dict, ensure_ascii=False)
        return NoSQLBuilderResult(
            engine=engine, source_id=source_id,
            collection_name=col.collection_name,
            query_dict=q_dict, query_json=q_json,
            intent=intent, field_hints=field_hints,
            warnings=warnings, error=None,
            duration_ms=round((time.time() - t0) * 1000, 2),
        )

    except Exception as exc:
        return NoSQLBuilderResult(
            engine=engine, source_id=source_id,
            collection_name=col.collection_name,
            query_dict=None, query_json=None,
            intent=intent, field_hints=field_hints,
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
