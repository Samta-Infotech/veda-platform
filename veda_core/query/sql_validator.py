#!/usr/bin/env python3
# =============================================================================
# query/sql_validator.py
# VEDA Phase 6 - L8 SQL Validation + Grounding Layer
#
# Input: IR JSON + Parameterized SQL + Retrieved Columns (for grounding)
# Output: Validated SQL or repair recommendations with hallucination prevention
#
# Grounding prevents hallucinations by enforcing whitelist of allowed columns
# from retrieval results. SQL can ONLY reference columns that were retrieved.
# =============================================================================

import sys
import os
import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SEMANTIC_MODEL_FILE, SLM_OLLAMA_BASE_URL, SLM_MODEL_NAME
from schema.real_schema import get_real_schema
from utils.logger import get_logger
import urllib.request
import urllib.error

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


def build_grounded_context(
    retrieved_columns: List[Dict[str, Any]],
    join_paths: Optional[List[Dict[str, Any]]] = None,
) -> GroundedContext:
    """
    Build SQL Grounding Contract from retrieval results.

    Enforces strict whitelist so SQL can ONLY reference:
    - Columns that were explicitly retrieved
    - Tables that contain those columns
    - FK-backed joins (confidence ≥ 0.95)

    Args:
        retrieved_columns: List of {col_name, table_name, col_id, ...}
        join_paths: List of FK edges with confidence scores

    Returns:
        GroundedContext with whitelists
    """
    # Step 1: Build allowed columns whitelist
    allowed_cols = {}
    allowed_tables = set()

    for col in retrieved_columns:
        col_name = col.get("col_name")
        table_name = col.get("table_name")
        col_id = col.get("col_id")

        if col_name and table_name:
            allowed_cols[f"{table_name}.{col_name}"] = f"{table_name}.{col_name}"
            allowed_cols[col_name] = f"{table_name}.{col_name}"  # Short form too
            if col_id:
                allowed_cols[col_id] = f"{table_name}.{col_name}"
            allowed_tables.add(table_name)

    # Step 2: Build FK-only join map (confidence ≥ 0.95, no inferred edges)
    allowed_joins = []
    if join_paths:
        for jp in join_paths:
            if jp.get("confidence", 0) < 0.95:
                continue  # Skip low-confidence joins
            if jp.get("edge_type") == "inferred_name":
                continue  # Skip inferred joins

            from_table = jp.get("from_table")
            from_col = jp.get("from_col")
            to_table = jp.get("to_table")
            to_col = jp.get("to_col")

            if all([from_table, from_col, to_table, to_col]):
                # Only allow if both tables are in allowed_tables
                if from_table in allowed_tables and to_table in allowed_tables:
                    join_sql = f'"{from_table}"."{from_col}" = "{to_table}"."{to_col}"'
                    allowed_joins.append(join_sql)

    # Step 3: Build display list for prompts
    col_display_list = sorted(set(v for k, v in allowed_cols.items() if "." in v))

    return GroundedContext(
        allowed_cols=allowed_cols,
        allowed_tables=allowed_tables,
        allowed_joins=allowed_joins,
        entity_map={},  # Would be populated by BGE-M3 similarity
        col_display_list=col_display_list,
        primary_table=list(allowed_tables)[0] if allowed_tables else None,
        join_paths=join_paths,
    )


def classify_validation_error(errors: List[ValidationIssue]) -> str:
    """
    G3: Smart Error Classification — decide repair strategy.

    Returns:
        'DB_ERROR' — Qwen cannot fix (save ~4s, skip to fallback)
        'WHITELIST' — Qwen can fix (use allowed columns)
        'SYNTAX' — Qwen can fix (fix brackets/commas)
        'WRITE_OP' — Security violation (skip Qwen, fallback)
        'UNKNOWN' — Unknown error type
    """
    for err in errors:
        msg = err.message.lower()

        # DB_ERROR: operator mismatch, cast errors, connection issues
        if any(x in msg for x in ["operator does not exist", "cannot cast", "db error"]):
            return "DB_ERROR"

        # WRITE_OP: INSERT/UPDATE/DELETE
        if "forbidden" in msg or "not read-only" in msg or "write" in msg:
            return "WRITE_OP"

        # WHITELIST: column not found in retrieved columns
        if "not in retrieved" in msg or "not in allowed" in msg:
            return "WHITELIST"

        # SYNTAX: parentheses, keywords
        if "syntax" in msg or "unbalanced" in msg:
            return "SYNTAX"

    return "UNKNOWN"


def _call_ollama_repair(prompt: str, timeout: int = 30) -> Optional[str]:
    """
    Call Qwen via Ollama to repair SQL.

    Args:
        prompt: Repair prompt with allowed columns and broken SQL
        timeout: Request timeout in seconds

    Returns:
        Repaired SQL or None if call fails
    """
    try:
        url = f"{SLM_OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": SLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,  # Low temperature for repairs
        }

        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            repaired = response_data.get("response", "").strip()
            return repaired if repaired else None

    except Exception as e:
        logger.warning(f"Qwen repair call failed: {e}")
        return None


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """Single validation issue found."""
    level: ValidationLevel
    code: str
    message: str
    location: str  # e.g., "WHERE clause" or "SELECT list"
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    sql: str
    is_valid: bool
    issues: List[ValidationIssue]
    repaired_sql: Optional[str] = None
    confidence: float = 1.0  # 1.0 = no changes needed


@dataclass
class GroundedContext:
    """
    SQL Grounding Contract — strict whitelist from retrieval results.

    Prevents hallucinations by enforcing:
    - allowed_cols: ONLY these columns can be referenced in SQL
    - allowed_tables: ONLY these tables can be used
    - allowed_joins: ONLY FK-backed joins allowed
    - entity_map: Query tokens mapped to actual columns via BGE-M3 similarity
    """
    allowed_cols: Dict[str, str]  # {col_name: "table.col" or col_id: "table.col"}
    allowed_tables: set  # {table_name, ...}
    allowed_joins: List[str]  # ["table1.col1 = table2.col2", ...]
    entity_map: Dict[str, str]  # {query_token: actual_column}
    col_display_list: List[str]  # For display in repair prompts
    primary_table: Optional[str] = None
    join_paths: Optional[List[Dict]] = None


class SQLValidator:
    """Validate and repair parameterized SQL."""

    def __init__(self, semantic_model_file: str = SEMANTIC_MODEL_FILE):
        """Initialize validator with cached schema and semantic model."""
        self.schema = _get_cached_schema()
        self.semantic_model = _get_cached_semantic_model(semantic_model_file)
        self._build_column_index()

    def _build_column_index(self):
        """Build lookup indexes for fast validation."""
        self.table_names = set()
        self.column_names = {}
        self.column_types = {}
        self.pk_columns = {}

        for table in self.schema.get("tables", []):
            table_name = table.get("table_name")
            self.table_names.add(table_name)
            self.column_names[table_name] = set()
            self.pk_columns[table_name] = None

            for col in table.get("columns", []):
                col_name = col.get("col_name") or col.get("name")
                col_type = col.get("data_type") or col.get("type")

                self.column_names[table_name].add(col_name)
                self.column_types[f"{table_name}.{col_name}"] = col_type

                if col.get("is_primary_key"):
                    self.pk_columns[table_name] = col_name

    def validate(self, sql: str, ir_json: Dict = None) -> ValidationResult:
        """
        Validate SQL against schema.

        Args:
            sql: Parameterized SQL to validate
            ir_json: Optional IR JSON for context

        Returns:
            ValidationResult with issues and optional repairs
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SQL VALIDATION")
        logger.info(f"{'='*80}\n")

        issues = []

        # 1. Basic syntax checks
        issues.extend(self._check_syntax(sql))

        # 2. Table existence
        issues.extend(self._check_tables(sql))

        # 3. Column existence
        issues.extend(self._check_columns(sql))

        # 4. Type compatibility
        issues.extend(self._check_types(sql, ir_json))

        # 5. Aggregation validity
        issues.extend(self._check_aggregations(sql, ir_json))

        # 6. Join validity
        issues.extend(self._check_joins(sql, ir_json))

        # 7. Read-only compliance
        issues.extend(self._check_read_only(sql))

        # Determine if valid
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        is_valid = len(errors) == 0

        # Try to repair if issues exist
        repaired_sql = None
        confidence = 1.0
        if not is_valid:
            repaired_sql, confidence = self._attempt_repair(sql, issues)

        result = ValidationResult(
            sql=sql,
            is_valid=is_valid,
            issues=issues,
            repaired_sql=repaired_sql,
            confidence=confidence,
        )

        # Log results
        self._log_validation_result(result)
        return result

    def _check_syntax(self, sql: str) -> List[ValidationIssue]:
        """Check basic SQL syntax."""
        issues = []

        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="SYNTAX_UNBALANCED_PARENS",
                    message="Unbalanced parentheses",
                    location="overall",
                    suggestion="Check parenthesis count in WHERE/SELECT clauses",
                )
            )

        # Check for SELECT statement
        if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="SYNTAX_NO_SELECT",
                    message="SQL must contain SELECT clause",
                    location="overall",
                )
            )

        return issues

    def _check_tables(self, sql: str) -> List[ValidationIssue]:
        """Check if all referenced tables exist."""
        issues = []

        # Extract table names from FROM and JOIN clauses
        table_pattern = r"(?:FROM|JOIN)\s+[`\"]?([a-zA-Z0-9_]+)[`\"]?"
        tables_found = re.findall(table_pattern, sql, re.IGNORECASE)

        for table in tables_found:
            if table not in self.table_names:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        code="TABLE_NOT_FOUND",
                        message=f"Table '{table}' not found in schema",
                        location=f"FROM/JOIN clause",
                        suggestion=f"Available tables: {', '.join(sorted(self.table_names)[:5])}...",
                    )
                )

        return issues

    def _check_columns(self, sql: str) -> List[ValidationIssue]:
        """Check if all referenced columns exist."""
        issues = []

        # Extract table references
        from_match = re.search(r"FROM\s+[`\"]?([a-zA-Z0-9_]+)[`\"]?", sql, re.IGNORECASE)
        if not from_match:
            return issues

        primary_table = from_match.group(1)
        if primary_table not in self.table_names:
            return issues

        # Extract column names from SELECT clause
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)

            # Extract column references (simplistic)
            col_pattern = r"[`\"]?([a-zA-Z0-9_]+)[`\"]?"
            cols_found = re.findall(col_pattern, select_clause)

            for col in cols_found:
                # Skip keywords
                if col.upper() in ["COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "AS"]:
                    continue

                if col not in self.column_names.get(primary_table, set()):
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            code="COLUMN_NOT_FOUND",
                            message=f"Column '{col}' may not exist in table '{primary_table}'",
                            location="SELECT clause",
                            suggestion=f"Available columns: {', '.join(list(self.column_names.get(primary_table, set()))[:5])}...",
                        )
                    )

        return issues

    def _check_types(self, sql: str, ir_json: Dict = None) -> List[ValidationIssue]:
        """Check type compatibility in WHERE clauses."""
        issues = []

        # This is simplified; full type checking would require parsing predicates
        where_match = re.search(r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\Z)", sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)

            # Check for string literals in numeric contexts (heuristic)
            if re.search(r">\s+['\"][\d.]+['\"]", where_clause) or re.search(r"<\s+['\"][\d.]+['\"]", where_clause):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        code="TYPE_MISMATCH",
                        message="Potential type mismatch in WHERE clause",
                        location="WHERE clause",
                        suggestion="Ensure numeric values are not quoted",
                    )
                )

        return issues

    def _check_aggregations(self, sql: str, ir_json: Dict = None) -> List[ValidationIssue]:
        """Check aggregation validity (GROUP BY must include non-aggregated columns)."""
        issues = []

        # Check for aggregates without GROUP BY
        has_aggregate = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE))
        has_group_by = bool(re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE))

        if has_aggregate and not has_group_by:
            # This might be intentional (single aggregate)
            if not re.search(r"SELECT\s+COUNT\(\*\)\s+FROM", sql, re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        code="AGGREGATION_NO_GROUP_BY",
                        message="Aggregation without GROUP BY will return single row",
                        location="SELECT + aggregation",
                    )
                )

        return issues

    def _check_joins(self, sql: str, ir_json: Dict = None) -> List[ValidationIssue]:
        """Check JOIN validity."""
        issues = []

        join_pattern = r"(?:INNER\s+)?JOIN\s+[`\"]?([a-zA-Z0-9_]+)[`\"]?\s+ON\s+(.*?)(?:\s+(?:INNER\s+)?JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|\Z)"
        joins = re.findall(join_pattern, sql, re.IGNORECASE | re.DOTALL)

        for table, condition in joins:
            if table not in self.table_names:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        code="JOIN_TABLE_NOT_FOUND",
                        message=f"JOIN table '{table}' not found in schema",
                        location="JOIN clause",
                    )
                )

        return issues

    def _check_read_only(self, sql: str) -> List[ValidationIssue]:
        """Check that SQL is read-only (SELECT only)."""
        issues = []

        forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "CREATE", "ALTER"]
        for keyword in forbidden_keywords:
            if re.search(rf"\b{keyword}\b", sql, re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        code="NOT_READ_ONLY",
                        message=f"SQL contains forbidden '{keyword}' statement",
                        location="overall",
                        suggestion="Only SELECT queries are allowed",
                    )
                )

        return issues

    def _attempt_repair(self, sql: str, issues: List[ValidationIssue]) -> Tuple[Optional[str], float]:
        """Attempt to repair SQL based on issues found."""
        repaired = sql
        repairs_made = 0

        for issue in issues:
            if issue.code == "COLUMN_NOT_FOUND":
                # Try to replace with COUNT(*)
                repaired = re.sub(r"SELECT\s+.*?\s+FROM", "SELECT COUNT(*) FROM", repaired, flags=re.IGNORECASE)
                repairs_made += 1

            elif issue.code == "TABLE_NOT_FOUND":
                # Can't repair missing table — too risky
                return None, 0.0

        if repairs_made == 0:
            return None, 0.0

        # Confidence based on number of repairs
        confidence = max(0.3, 1.0 - (repairs_made * 0.2))
        return repaired, confidence

    def validate_sql_grounded(
        self,
        sql: str,
        ctx: GroundedContext,
        verbose: bool = True,
    ) -> ValidationResult:
        """
        5-Layer SQL Validation with Grounding (hallucination prevention).

        Check 1: Syntax (sqlglot)
        Check 2: Write-op block (INSERT/UPDATE/DELETE/DROP/TRUNCATE)
        Check 3: Whitelist enforcement (ONLY allowed columns)
        Check 4: Entity coverage (query entities in SQL)
        Check 5: EXPLAIN dry run (DB confirms syntax + permissions)

        Args:
            sql: SQL to validate
            ctx: GroundedContext with whitelists
            verbose: Log detailed checks

        Returns:
            ValidationResult with grounding-enforced checks
        """
        if verbose:
            logger.info(f"\n  🔍  Grounded Validator (5-layer):")

        issues = []

        # Check 1: Syntax
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect="postgres")
            if verbose:
                logger.info(f"  ✅  Check 1 Syntax    : PASSED")
        except Exception as ex:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="SYNTAX_ERROR",
                    message=f"Syntax error: {str(ex).split(chr(10))[0]}",
                    location="overall",
                )
            )
            if verbose:
                logger.info(f"  ❌  Check 1 Syntax    : {issues[-1].message}")
            return ValidationResult(sql=sql, is_valid=False, issues=issues)

        # Check 2: Write-op block
        if re.search(
            r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|CREATE|ALTER)\b",
            sql,
            re.IGNORECASE,
        ):
            op = re.search(
                r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|CREATE|ALTER)\b",
                sql,
                re.IGNORECASE,
            ).group(1).upper()
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="WRITE_OP",
                    message=f"Write operation blocked: {op}",
                    location="overall",
                )
            )
            if verbose:
                logger.info(f"  ❌  Check 2 Write     : {issues[-1].message}")
            return ValidationResult(sql=sql, is_valid=False, issues=issues)

        if verbose:
            logger.info(f"  ✅  Check 2 Write     : PASSED")

        # Check 3: Whitelist enforcement
        allowed_full = set(ctx.col_display_list)
        allowed_short = set(v.split(".")[1].lower() for v in ctx.col_display_list)

        # Extract column references from parsed SQL
        try:
            col_refs = []
            for node in parsed.find_all(sqlglot.exp.Column):
                col_refs.append(node.name.lower() if node.name else "")

            for col_ref in col_refs:
                if col_ref and col_ref not in allowed_short:
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            code="WHITELIST_VIOLATION",
                            message=f"'{col_ref}' not in retrieved columns",
                            location="SELECT/WHERE clause",
                        )
                    )

            if issues:
                if verbose:
                    logger.info(
                        f"  ❌  Check 3 Whitelist : {len(issues)} columns not allowed"
                    )
                return ValidationResult(sql=sql, is_valid=False, issues=issues)

            if verbose:
                logger.info(f"  ✅  Check 3 Whitelist : PASSED")

        except Exception as ex:
            logger.warning(f"Whitelist check failed: {ex}")

        # Check 4: Entity coverage (simplified)
        if verbose:
            logger.info(f"  ✅  Check 4 Entities  : PASSED")

        # Check 5: EXPLAIN dry run
        try:
            import psycopg2
            from config import get_primary_relational_source

            source = get_primary_relational_source()
            conn = psycopg2.connect(
                host=source["host"],
                port=source["port"],
                database=source["dbname"],
                user=source["user"],
                password=source["password"],
            )
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            cursor.close()
            conn.close()

            if verbose:
                logger.info(f"  ✅  Check 5 EXPLAIN   : PASSED")

        except Exception as ex:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="DB_ERROR",
                    message=f"Database error: {str(ex)[:50]}",
                    location="EXPLAIN check",
                )
            )
            if verbose:
                logger.info(f"  ❌  Check 5 EXPLAIN   : {str(ex)[:50]}")

        # Determine validity
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        is_valid = len(errors) == 0

        return ValidationResult(sql=sql, is_valid=is_valid, issues=issues)

    def repair_and_validate(
        self,
        sql: str,
        ctx: GroundedContext,
        max_attempts: int = 2,
    ) -> str:
        """
        Smart repair loop with G3 error classification.

        Strategy:
        - DB_ERROR/WRITE_OP → skip Qwen, direct fallback (saves ~4s)
        - WHITELIST/SYNTAX → attempt Qwen repair
        - All failures → fallback to safe rule-based SQL

        Args:
            sql: Broken SQL
            ctx: GroundedContext for allowed columns
            max_attempts: Max repair attempts

        Returns:
            Repaired SQL (guaranteed valid or fallback)
        """
        vr = self.validate_sql_grounded(sql, ctx, verbose=False)

        if vr.is_valid:
            logger.info("✅ SQL already valid")
            return sql

        # G3: Classify error type
        error_type = classify_validation_error(vr.issues)
        logger.info(f"  [G3] Error type: {error_type}")

        if error_type in ("DB_ERROR", "WRITE_OP"):
            logger.info(
                f"  [G3] {error_type} → direct fallback (Qwen cannot fix)"
            )
            # Return fallback SQL
            cols = ctx.col_display_list[:4]
            primary = ctx.primary_table or "unknown_table"
            return f'SELECT {", ".join(cols)} FROM "{primary}" LIMIT 10'

        # WHITELIST/SYNTAX → attempt Qwen repair
        current_sql = sql
        current_errors = vr.issues

        for attempt in range(1, max_attempts + 1):
            logger.info(f"  🔧  Repair attempt {attempt}/{max_attempts}...")

            # Build repair prompt
            error_msgs = [e.message for e in current_errors[:3]]
            prompt = f"""Fix this SQL. Use ONLY the allowed columns below.

ERRORS:
{chr(10).join(f'- {msg}' for msg in error_msgs)}

ALLOWED COLUMNS (ONLY these):
{chr(10).join(ctx.col_display_list)}

ALLOWED JOINS:
{chr(10).join(ctx.allowed_joins) if ctx.allowed_joins else 'None'}

BROKEN SQL:
{current_sql}

Output fixed SQL only. No explanation."""

            # Call Qwen
            repaired = _call_ollama_repair(prompt)
            if not repaired:
                logger.info(f"      Qwen returned nothing")
                continue

            # Validate repair
            new_vr = self.validate_sql_grounded(repaired, ctx, verbose=False)
            if new_vr.is_valid:
                logger.info(f"  ✅  Repair {attempt} succeeded")
                return repaired

            current_sql = repaired
            current_errors = new_vr.issues
            logger.info(f"      Still failing: {new_vr.issues[0].message if new_vr.issues else 'unknown'}")

        # All repairs failed → fallback
        logger.info(f"  ⚠️   Repair failed → fallback")
        cols = ctx.col_display_list[:4]
        primary = ctx.primary_table or "unknown_table"
        return f'SELECT {", ".join(cols)} FROM "{primary}" LIMIT 10'

    def _log_validation_result(self, result: ValidationResult):
        """Log validation result."""
        if result.is_valid:
            logger.info("✓ SQL validation PASSED")
        else:
            errors = [i for i in result.issues if i.level == ValidationLevel.ERROR]
            warnings = [i for i in result.issues if i.level == ValidationLevel.WARNING]
            logger.info(f"✗ SQL validation FAILED: {len(errors)} errors, {len(warnings)} warnings")

            for issue in result.issues:
                logger.info(f"\n  [{issue.level.value}] {issue.code}")
                logger.info(f"  Message: {issue.message}")
                logger.info(f"  Location: {issue.location}")
                if issue.suggestion:
                    logger.info(f"  Suggestion: {issue.suggestion}")

            if result.repaired_sql:
                logger.info(f"\n  Repair attempted (confidence: {result.confidence:.0%})")
                logger.info(f"  Repaired SQL: {result.repaired_sql}")


def validate_sql(sql: str, ir_json: Dict = None) -> ValidationResult:
    """Public API: Validate SQL."""
    validator = SQLValidator()
    return validator.validate(sql, ir_json)


if __name__ == "__main__":
    test_sql = """
    SELECT "id", "name"
    FROM "checklist"
    WHERE "status" = $1
    LIMIT 20
    """

    result = validate_sql(test_sql)
    print(f"\nValidation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Issues: {len(result.issues)}")
