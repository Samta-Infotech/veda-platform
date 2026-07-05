#!/usr/bin/env python3
# =============================================================================
# query/executor.py
# VEDA Phase 7 - L7 Query Execution Engine
#
# Input: Parameterized SQL + parameters
# Output: Query results formatted
# =============================================================================

import sys
import os
import psycopg2
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import VEDA_INTERNAL_DB, EXECUTION_QUERY_TIMEOUT_SECS, EXECUTION_RESULT_LIMIT
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryExecutionResult:
    """Result of query execution."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    row_count: int = 0
    execution_time_ms: float = 0.0
    columns: Optional[List[str]] = None


class QueryExecutor:
    """Execute parameterized SQL against PostgreSQL."""

    def __init__(self):
        """Initialize executor with DB connection pool."""
        db_config = VEDA_INTERNAL_DB
        self.conn_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        self.timeout = EXECUTION_QUERY_TIMEOUT_SECS * 1000  # convert to ms
        self.max_rows = EXECUTION_RESULT_LIMIT

    def execute(
        self,
        sql: str,
        params: List[Any] = None,
        timeout_ms: Optional[int] = None,
    ) -> QueryExecutionResult:
        """
        Execute parameterized SQL.

        Args:
            sql: Parameterized SQL (uses $1, $2, ... placeholders)
            params: Parameter values to bind
            timeout_ms: Query timeout in milliseconds (default from config)

        Returns:
            QueryExecutionResult with data or error
        """
        timeout = timeout_ms or self.timeout
        start_time = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"QUERY EXECUTION")
        logger.info(f"{'='*80}\n")

        try:
            # Connect to database
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()

            # Set statement timeout
            cursor.execute(f"SET statement_timeout = {timeout};")

            logger.info(f"Executing query (timeout: {timeout}ms)...")
            logger.info(f"SQL: {sql}")
            if params:
                logger.info(f"Parameters: {params}")

            # Execute query
            cursor.execute(sql, params or [])

            # Fetch results
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            # Convert to list of dicts
            data = []
            for row in rows[: self.max_rows]:
                data.append(dict(zip(columns, row)))

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"\n✓ Query executed successfully")
            logger.info(f"Rows returned: {len(data)}")
            logger.info(f"Execution time: {execution_time:.1f}ms")

            cursor.close()
            conn.close()

            return QueryExecutionResult(
                success=True,
                data=data,
                row_count=len(data),
                execution_time_ms=execution_time,
                columns=columns,
            )

        except psycopg2.errors.QueryCanceled as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"\n✗ Query timeout after {execution_time:.1f}ms")
            return QueryExecutionResult(
                success=False,
                error=f"Query timeout: {timeout}ms exceeded",
                execution_time_ms=execution_time,
            )

        except psycopg2.ProgrammingError as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"\n✗ SQL error: {str(e)}")
            return QueryExecutionResult(
                success=False,
                error=f"SQL error: {str(e)}",
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"\n✗ Execution error: {str(e)}")
            return QueryExecutionResult(
                success=False,
                error=f"Execution error: {str(e)}",
                execution_time_ms=execution_time,
            )

    def execute_safe(
        self,
        sql: str,
        params: List[Any] = None,
        fallback_to_count: bool = True,
    ) -> QueryExecutionResult:
        """
        Execute SQL with automatic fallback to COUNT(*).

        Args:
            sql: Parameterized SQL
            params: Parameter values
            fallback_to_count: If query fails, try COUNT(*) instead

        Returns:
            QueryExecutionResult
        """
        result = self.execute(sql, params)

        if not result.success and fallback_to_count:
            logger.info("\nFallback: Attempting COUNT(*) query...")

            # Extract primary table from SQL
            import re

            from_match = re.search(r"FROM\s+[`\"]?([a-zA-Z0-9_]+)[`\"]?", sql, re.IGNORECASE)
            if from_match:
                table = from_match.group(1)
                count_sql = f'SELECT COUNT(*) as "count" FROM "{table}"'

                logger.info(f"Fallback SQL: {count_sql}")
                result = self.execute(count_sql, [])

        return result


def execute_sql(
    sql: str,
    params: List[Any] = None,
    timeout_ms: Optional[int] = None,
) -> QueryExecutionResult:
    """Public API: Execute SQL."""
    executor = QueryExecutor()
    return executor.execute(sql, params, timeout_ms)


def execute_sql_safe(
    sql: str,
    params: List[Any] = None,
    fallback_to_count: bool = True,
) -> QueryExecutionResult:
    """Public API: Execute SQL with fallback."""
    executor = QueryExecutor()
    return executor.execute_safe(sql, params, fallback_to_count)


if __name__ == "__main__":
    # Test
    test_sql = 'SELECT "id", "name" FROM "checklist" LIMIT 5'
    result = execute_sql(test_sql)

    print(f"\nExecution result:")
    print(f"  Success: {result.success}")
    print(f"  Rows: {result.row_count}")
    print(f"  Time: {result.execution_time_ms:.1f}ms")
    if result.data:
        print(f"  Data: {json.dumps(result.data[:2], indent=2, default=str)}")
