#!/usr/bin/env python3
# =============================================================================
# query/audit_logger.py
# VEDA Phase 8 - L8 Audit Logger
#
# Input: Query execution details
# Output: Append-only audit log
# =============================================================================

import sys
import os
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

AUDIT_LOG_DB = "data/veda_audit_log.db"

logger = get_logger(__name__)


@dataclass
class AuditLogEntry:
    """Single audit log entry."""
    timestamp: str
    query_hash: str
    original_query: str
    normalized_query: str
    intent: str
    top_columns: list
    sql_generated: str
    sql_valid: bool
    execution_success: bool
    row_count: int
    execution_time_ms: float
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AuditLogger:
    """Append-only audit log for all queries."""

    def __init__(self, db_path: str = AUDIT_LOG_DB):
        """Initialize audit logger with SQLite."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize audit log database and schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create audit log table (append-only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                original_query TEXT NOT NULL,
                normalized_query TEXT,
                intent TEXT,
                top_columns TEXT,
                sql_generated TEXT NOT NULL,
                sql_valid BOOLEAN,
                execution_success BOOLEAN,
                row_count INTEGER,
                execution_time_ms REAL,
                error_message TEXT,
                user_id TEXT,
                session_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for efficient querying
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_hash ON audit_log(query_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_intent ON audit_log(intent)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_success ON audit_log(execution_success)"
        )

        conn.commit()
        conn.close()

        logger.info(f"✓ Audit log database initialized: {self.db_path}")

    def log(
        self,
        original_query: str,
        normalized_query: str,
        intent: str,
        top_columns: list,
        sql_generated: str,
        sql_valid: bool,
        execution_success: bool,
        row_count: int,
        execution_time_ms: float,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log a query to audit trail.

        Args:
            original_query: User's original NL query
            normalized_query: Normalized query
            intent: Detected intent
            top_columns: Top-K retrieved columns
            sql_generated: Generated SQL
            sql_valid: Whether SQL passed validation
            execution_success: Whether execution succeeded
            row_count: Number of rows returned
            execution_time_ms: Total execution time
            error_message: Optional error message
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            AuditLogEntry that was logged
        """
        # Create entry
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            query_hash=self._hash_query(normalized_query),
            original_query=original_query,
            normalized_query=normalized_query,
            intent=intent,
            top_columns=top_columns,
            sql_generated=sql_generated,
            sql_valid=sql_valid,
            execution_success=execution_success,
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            user_id=user_id,
            session_id=session_id,
        )

        # Append to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO audit_log (
                    timestamp,
                    query_hash,
                    original_query,
                    normalized_query,
                    intent,
                    top_columns,
                    sql_generated,
                    sql_valid,
                    execution_success,
                    row_count,
                    execution_time_ms,
                    error_message,
                    user_id,
                    session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp,
                entry.query_hash,
                entry.original_query,
                entry.normalized_query,
                entry.intent,
                json.dumps(entry.top_columns),
                entry.sql_generated,
                entry.sql_valid,
                entry.execution_success,
                entry.row_count,
                entry.execution_time_ms,
                entry.error_message,
                entry.user_id,
                entry.session_id,
            ))

            conn.commit()
            conn.close()

            logger.info(f"✓ Audit log entry recorded: {entry.query_hash[:8]}")
            return entry

        except Exception as e:
            logger.error(f"✗ Failed to log audit entry: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get audit log statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total queries
            cursor.execute("SELECT COUNT(*) FROM audit_log")
            total = cursor.fetchone()[0]

            # Success rate
            cursor.execute("SELECT COUNT(*) FROM audit_log WHERE execution_success = 1")
            successful = cursor.fetchone()[0]

            # Average execution time
            cursor.execute("SELECT AVG(execution_time_ms) FROM audit_log WHERE execution_success = 1")
            avg_time = cursor.fetchone()[0] or 0

            # Intent distribution
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM audit_log
                GROUP BY intent
                ORDER BY count DESC
            """)
            intents = dict(cursor.fetchall())

            # Error summary
            cursor.execute("""
                SELECT error_message, COUNT(*) as count
                FROM audit_log
                WHERE error_message IS NOT NULL
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
            """)
            errors = dict(cursor.fetchall())

            conn.close()

            return {
                "total_queries": total,
                "successful_queries": successful,
                "success_rate": f"{100 * successful / total:.1f}%" if total > 0 else "0%",
                "avg_execution_time_ms": f"{avg_time:.1f}",
                "intent_distribution": intents,
                "top_errors": errors,
            }

        except Exception as e:
            logger.error(f"✗ Failed to get audit stats: {e}")
            return {}

    @staticmethod
    def _hash_query(query: str) -> str:
        """Hash normalized query for deduplication."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


def log_query(
    original_query: str,
    normalized_query: str,
    intent: str,
    top_columns: list,
    sql_generated: str,
    sql_valid: bool,
    execution_success: bool,
    row_count: int,
    execution_time_ms: float,
    error_message: Optional[str] = None,
) -> AuditLogEntry:
    """Public API: Log query to audit trail."""
    logger = AuditLogger()
    return logger.log(
        original_query=original_query,
        normalized_query=normalized_query,
        intent=intent,
        top_columns=top_columns,
        sql_generated=sql_generated,
        sql_valid=sql_valid,
        execution_success=execution_success,
        row_count=row_count,
        execution_time_ms=execution_time_ms,
        error_message=error_message,
    )


if __name__ == "__main__":
    # Test
    logger = AuditLogger()

    # Log test entry
    entry = logger.log(
        original_query="show me all checklists",
        normalized_query="show me all checklists",
        intent="DIRECT",
        top_columns=["checklist.id", "checklist.name"],
        sql_generated='SELECT "id", "name" FROM "checklist" LIMIT 20',
        sql_valid=True,
        execution_success=True,
        row_count=42,
        execution_time_ms=145.3,
    )

    print(f"\n✓ Logged entry: {entry.query_hash}")

    # Get stats
    stats = logger.get_stats()
    print(f"\nAudit log stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
