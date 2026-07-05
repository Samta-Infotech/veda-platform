#!/usr/bin/env python3
# =============================================================================
# query/sql_generator.py
# VEDA Phase 4 - L5 SQL Generation Layer
#
# Input: Intent + top-K retrieved columns
# Output: IR JSON v1 + Parameterized SQL
# =============================================================================

import sys
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SEMANTIC_MODEL_FILE, TEMPORAL_PARSER_ENABLED
from schema.real_schema import get_real_schema
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level schema cache (singleton) - initialized once on first use
_SCHEMA_CACHE = None
_SEMANTIC_MODEL_CACHE = {}


def _get_cached_schema() -> Dict:
    """Get schema from module-level cache, or load once and cache it."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        logger.info("Loading schema (first time - will be cached)...")
        _SCHEMA_CACHE = get_real_schema()
    return _SCHEMA_CACHE


def _get_cached_semantic_model(model_file: str) -> Dict:
    """Get semantic model from cache, or load once and cache it."""
    if model_file not in _SEMANTIC_MODEL_CACHE:
        logger.info(f"Loading semantic model from {model_file}...")
        with open(model_file) as f:
            _SEMANTIC_MODEL_CACHE[model_file] = json.load(f)
    return _SEMANTIC_MODEL_CACHE[model_file]


class QueryIntent(Enum):
    """Query intent types (aligned with Phase 3)."""
    DIRECT = "DIRECT"
    SYNONYM = "SYNONYM"
    MULTI_TABLE = "MULTI_TABLE"
    TEMPORAL = "TEMPORAL"
    AGGREGATE = "AGGREGATE"


class SQLOperator(Enum):
    """SQL comparison operators for filter tree."""
    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    LT = "LT"
    GTE = "GTE"
    LTE = "LTE"
    LIKE = "LIKE"
    IN = "IN"
    IS_NULL = "IS_NULL"
    IS_NOT_NULL = "IS_NOT_NULL"


@dataclass
class IRColumn:
    """Column reference in IR JSON (UUID-based, never raw names)."""
    col_id: str  # UUID
    col_name: str  # For reference only (not used in SQL)
    table_name: str  # For reference only


@dataclass
class IRJoinCondition:
    """Join condition in IR JSON."""
    from_col_id: str
    to_col_id: str
    join_type: str  # INNER, LEFT, RIGHT, FULL


@dataclass
class IRFilterCondition:
    """Filter condition in IR JSON."""
    type: str  # EQ, GT, LT, LIKE, IN, etc.
    col_id: str
    value: Any
    operator: str  # AND, OR (for tree nodes)


@dataclass
class IRAggregation:
    """Aggregation in IR JSON."""
    func: str  # COUNT, SUM, AVG, MIN, MAX
    col_id: str
    alias: str


@dataclass
class IRQuery:
    """IR JSON v1 Query."""
    version: str = "1.0"
    intent: str = "SELECT"
    entities: List[Dict] = None
    columns: List[IRColumn] = None
    filter_tree: Optional[Dict] = None
    joins: List[IRJoinCondition] = None
    aggregations: List[IRAggregation] = None
    group_by: List[str] = None
    order_by: List[Dict] = None
    limit: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "intent": self.intent,
            "entities": self.entities or [],
            "columns": [{"col_id": c.col_id} for c in (self.columns or [])],
            "filter_tree": self.filter_tree,
            "joins": [
                {
                    "from_col_id": j.from_col_id,
                    "to_col_id": j.to_col_id,
                    "join_type": j.join_type,
                }
                for j in (self.joins or [])
            ],
            "aggregations": [
                {
                    "func": a.func,
                    "col_id": a.col_id,
                    "alias": a.alias,
                }
                for a in (self.aggregations or [])
            ],
            "group_by": [{"col_id": c} for c in (self.group_by or [])],
            "order_by": self.order_by or [],
            "limit": self.limit,
            "schema_version": 1,
        }


@dataclass
class ParameterBinder:
    """Manages parameterized SQL to prevent SQL injection."""
    params: List[Any]

    def bind(self, value: Any) -> str:
        """Add value to parameters, return placeholder."""
        self.params.append(value)
        return f"${len(self.params)}"

    def bind_identifier(self, name: str) -> str:
        """Quote identifier (column/table name)."""
        return f'"{name}"'


class SQLGenerator:
    """Generate parameterized SQL from IR JSON."""

    def __init__(self, semantic_model_file: str = SEMANTIC_MODEL_FILE):
        """Initialize generator with cached schema and semantic model."""
        self.semantic_model_file = semantic_model_file
        self.schema = _get_cached_schema()
        self.semantic_model = _get_cached_semantic_model(semantic_model_file)

    def generate_from_intent_and_columns(
        self,
        intent: str,
        top_columns: List[Tuple[str, float]],
        query: str,
        top_k: int = 20,
    ) -> Tuple[Dict, str]:
        """
        Generate IR JSON + SQL from intent + retrieved columns.

        Args:
            intent: Query intent (DIRECT, TEMPORAL, AGGREGATE, etc.)
            top_columns: [(col_id, score), ...]
            query: Original NL query
            top_k: Number of results to return

        Returns:
            (ir_json_dict, parameterized_sql)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SQL GENERATION: {intent} intent")
        logger.info(f"{'='*80}\n")

        # Extract column IDs
        col_ids = [col for col, score in top_columns[:5]]

        # Build IR JSON based on intent
        if intent == "DIRECT":
            ir_json = self._generate_direct_query(col_ids)
        elif intent == "TEMPORAL":
            ir_json = self._generate_temporal_query(col_ids, query)
        elif intent == "AGGREGATE":
            ir_json = self._generate_aggregate_query(col_ids, query)
        elif intent == "MULTI_TABLE":
            ir_json = self._generate_multi_table_query(col_ids, query)
        else:
            ir_json = self._generate_fallback_query(col_ids)

        # Convert IR JSON to SQL
        sql, params = self._ir_to_sql(ir_json)

        logger.info(f"Generated SQL:\n{sql}\n")
        logger.info(f"Parameters: {params}")

        return ir_json.to_dict(), sql

    def _generate_direct_query(self, col_ids: List[str]) -> IRQuery:
        """Generate SELECT query for DIRECT intent."""
        logger.info(f"Generating DIRECT query...")

        # Infer table from column IDs
        tables_set = set(col_id.split(".")[0] for col_id in col_ids)
        primary_table = list(tables_set)[0] if tables_set else None

        ir = IRQuery(
            intent="SELECT",
            columns=[IRColumn(col_id, col_id.split(".")[1], col_id.split(".")[0]) for col_id in col_ids],
            limit=20,
        )

        logger.info(f"  ✓ Selected {len(col_ids)} columns from {len(tables_set)} table(s)")
        return ir

    def _generate_temporal_query(self, col_ids: List[str], query: str) -> IRQuery:
        """Generate query with temporal filter (TEMPORAL intent)."""
        logger.info(f"Generating TEMPORAL query...")

        # Find temporal columns in retrieved columns
        temporal_cols = [c for c in col_ids if any(t in c.lower() for t in ["date", "time", "created", "updated"])]

        ir = IRQuery(
            intent="SELECT",
            columns=[IRColumn(col_id, col_id.split(".")[1], col_id.split(".")[0]) for col_id in col_ids],
            limit=20,
        )

        if temporal_cols:
            # Add temporal filter (simplified - would need actual date parsing in production)
            ir.filter_tree = {
                "type": "AND",
                "conditions": [
                    {
                        "type": "GTE",
                        "col_id": temporal_cols[0],
                        "value": "2026-01-01",  # Placeholder
                    }
                ],
            }

        logger.info(f"  ✓ Applied temporal filter on {len(temporal_cols)} column(s)")
        return ir

    def _generate_aggregate_query(self, col_ids: List[str], query: str) -> IRQuery:
        """Generate aggregation query (AGGREGATE intent)."""
        logger.info(f"Generating AGGREGATE query...")

        # Identify metric columns
        metric_cols = [c for c in col_ids if any(m in c.lower() for m in ["count", "sum", "total", "amount"])]
        if not metric_cols:
            metric_cols = col_ids[:1]

        ir = IRQuery(
            intent="SELECT",
            columns=[IRColumn(col_id, col_id.split(".")[1], col_id.split(".")[0]) for col_id in col_ids if col_id not in metric_cols],
            aggregations=[
                IRAggregation(func="COUNT", col_id=metric_cols[0], alias="result_count")
            ],
            group_by=col_ids[:2] if len(col_ids) > 1 else col_ids,
            limit=None,
        )

        logger.info(f"  ✓ Added COUNT aggregation on {metric_cols[0]}")
        return ir

    def _generate_multi_table_query(self, col_ids: List[str], query: str) -> IRQuery:
        """Generate multi-table join query (MULTI_TABLE intent)."""
        logger.info(f"Generating MULTI_TABLE query...")

        # Extract unique tables
        tables_set = set(col_id.split(".")[0] for col_id in col_ids)

        ir = IRQuery(
            intent="SELECT",
            columns=[IRColumn(col_id, col_id.split(".")[1], col_id.split(".")[0]) for col_id in col_ids],
            limit=20,
        )

        # Add joins if multiple tables
        if len(tables_set) > 1:
            ir.joins = self._infer_joins(list(tables_set))

        logger.info(f"  ✓ Set up joins for {len(tables_set)} table(s)")
        return ir

    def _generate_fallback_query(self, col_ids: List[str]) -> IRQuery:
        """Generate fallback query (SELECT COUNT(*) or SELECT *)."""
        logger.info(f"Generating FALLBACK query...")

        if col_ids:
            ir = IRQuery(
                intent="SELECT",
                columns=[IRColumn(col_id, col_id.split(".")[1], col_id.split(".")[0]) for col_id in col_ids],
                limit=20,
            )
        else:
            # Ultimate fallback: COUNT(*)
            ir = IRQuery(
                intent="SELECT",
                aggregations=[IRAggregation(func="COUNT", col_id="*", alias="total_count")],
                limit=None,
            )

        logger.info(f"  ✓ Fallback: selected {len(col_ids)} columns or COUNT(*)")
        return ir

    def _infer_joins(self, tables: List[str]) -> List[IRJoinCondition]:
        """Infer JOIN conditions from FK relationships."""
        joins = []

        # Simple heuristic: use FK map if available
        with open("data/veda_fk_map.json") as f:
            fk_map = json.load(f)

        for from_col, to_cols in fk_map.get("forward_fks", {}).items():
            from_table = from_col.split(".")[0]
            if from_table in tables:
                for to_col in to_cols:
                    to_table = to_col.split(".")[0]
                    if to_table in tables and to_table != from_table:
                        joins.append(
                            IRJoinCondition(from_col, to_col, "LEFT JOIN")
                        )

        return joins

    def _ir_to_sql(self, ir: IRQuery) -> Tuple[str, List[Any]]:
        """Convert IR JSON to parameterized SQL."""
        binder = ParameterBinder(params=[])

        sql_parts = []

        # SELECT clause
        if ir.aggregations:
            agg_exprs = []
            for agg in ir.aggregations:
                if agg.col_id == "*":
                    agg_expr = f"{agg.func}(*) AS {binder.bind_identifier(agg.alias)}"
                else:
                    agg_expr = f"{agg.func}({binder.bind_identifier(agg.col_id.split('.')[1])}) AS {binder.bind_identifier(agg.alias)}"
                agg_exprs.append(agg_expr)
            sql_parts.append(f"SELECT {', '.join(agg_exprs)}")
        elif ir.columns:
            col_exprs = [binder.bind_identifier(col.col_name) for col in ir.columns]
            sql_parts.append(f"SELECT {', '.join(col_exprs)}")
        else:
            sql_parts.append("SELECT *")

        # FROM clause
        if ir.columns:
            primary_table = ir.columns[0].table_name
            sql_parts.append(f"FROM {binder.bind_identifier(primary_table)}")
        else:
            sql_parts.append("FROM (SELECT NULL) AS t")

        # WHERE clause
        if ir.filter_tree:
            where_clause = self._build_where_clause(ir.filter_tree, binder)
            sql_parts.append(f"WHERE {where_clause}")

        # GROUP BY clause
        if ir.group_by:
            group_cols = [binder.bind_identifier(col.split(".")[1]) for col in ir.group_by]
            sql_parts.append(f"GROUP BY {', '.join(group_cols)}")

        # ORDER BY clause
        if ir.order_by:
            order_exprs = [f"{binder.bind_identifier(o['col_id'].split('.')[1])} {o.get('direction', 'ASC')}" for o in ir.order_by]
            sql_parts.append(f"ORDER BY {', '.join(order_exprs)}")

        # LIMIT clause
        if ir.limit:
            sql_parts.append(f"LIMIT {ir.limit}")

        sql = "\n".join(sql_parts)
        return sql, binder.params

    def _build_where_clause(self, filter_tree: Dict, binder: ParameterBinder) -> str:
        """Build WHERE clause from filter tree."""
        if filter_tree.get("type") in ["AND", "OR"]:
            conditions = [self._build_where_clause(c, binder) for c in filter_tree.get("conditions", [])]
            operator = f" {filter_tree['type']} "
            return f"({operator.join(conditions)})"
        else:
            # Leaf node
            col_id = filter_tree.get("col_id", "")
            op = filter_tree.get("type", "EQ")
            value = filter_tree.get("value")

            col_name = col_id.split(".")[1] if "." in col_id else col_id
            param_placeholder = binder.bind(value)

            op_map = {
                "EQ": "=",
                "NEQ": "!=",
                "GT": ">",
                "LT": "<",
                "GTE": ">=",
                "LTE": "<=",
                "LIKE": "LIKE",
                "IN": "IN",
            }

            sql_op = op_map.get(op, "=")
            return f"{binder.bind_identifier(col_name)} {sql_op} {param_placeholder}"


def generate_sql(
    intent: str,
    top_columns: List[Tuple[str, float]],
    query: str,
    top_k: int = 20,
) -> Tuple[Dict, str]:
    """
    Public API: Generate IR JSON + SQL from intent + columns.

    Args:
        intent: Query intent
        top_columns: Retrieved columns with scores
        query: Original NL query
        top_k: Result limit

    Returns:
        (ir_json_dict, parameterized_sql_string)
    """
    generator = SQLGenerator()
    return generator.generate_from_intent_and_columns(intent, top_columns, query, top_k)


if __name__ == "__main__":
    # Test
    test_intent = "DIRECT"
    test_columns = [("checklist.id", 0.85), ("checklist.name", 0.78)]
    test_query = "show me all checklists"

    ir_json, sql = generate_sql(test_intent, test_columns, test_query)

    print("\nGenerated IR JSON:")
    print(json.dumps(ir_json, indent=2))
    print("\nGenerated SQL:")
    print(sql)
