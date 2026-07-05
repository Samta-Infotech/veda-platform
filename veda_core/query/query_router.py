# =============================================================================
# query/query_router.py
# VEDA — Query Router (Phase 2)
#
# Responsibility:
#   - Classifies a user query as: "sql" | "rag" | "hybrid" | "nosql"
#   - Routes based on keyword signals first, then available source types
#   - Returns which source IDs to query for each intent
#
# Called before L1 when QUERY_ROUTER_ENABLED=True.
# Keyword signals are fast (no model inference). Falls back to
# embedding-based classification only when signals are ambiguous.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    QUERY_ROUTER_ENABLED,
    QUERY_ROUTER_CONFIDENCE_THRESHOLD,
    QUERY_ROUTER_INTENTS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class RouteResult:
    """Result of routing a user query to an intent and source set."""
    intent:      str            # "sql" | "rag" | "hybrid" | "nosql"
    source_ids:  List[str]      # which source IDs to query
    confidence:  float          # 0.0–1.0
    reason:      str            # human-readable explanation of the routing decision
    stats:       dict = field(default_factory=dict)


# =============================================================================
# Keyword signal tables
# =============================================================================

# Strong SQL signals — queries about counts, aggregations, filtering structured data
_SQL_KEYWORDS = {
    "count", "total", "sum", "average", "avg", "max", "min",
    "how many", "how much", "list", "show", "find", "get",
    "filter", "where", "group", "order", "sort", "top", "bottom",
    "last", "first", "recent", "latest", "oldest",
    "between", "since", "before", "after", "during",
    "per", "by", "from table", "from database",
    "revenue", "amount", "price", "status", "state", "type",
    "user", "users", "record", "records", "row", "rows",
    "transactions", "incidents", "tickets", "orders",
}

# Strong RAG signals — queries about document content, policies, agreements
_RAG_KEYWORDS = {
    "policy", "policies", "procedure", "procedures",
    "contract", "contracts", "agreement", "agreements",
    "document", "documents", "file", "files",
    "according to", "as per", "states that", "says that",
    "clause", "section", "paragraph", "article",
    "manual", "handbook", "guide", "guidelines",
    "report", "summary", "description", "definition",
    "what does", "explain", "describe", "tell me about",
    "regulations", "compliance", "terms", "conditions",
    "specification", "spec",
}

# Temporal words that lean strongly towards SQL
_TEMPORAL_KEYWORDS = {
    "yesterday", "today", "last week", "last month", "last year",
    "this week", "this month", "this year",
    "q1", "q2", "q3", "q4", "quarter", "fiscal",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

# NoSQL signals — queries about events, logs, time-series
_NOSQL_KEYWORDS = {
    "event", "events", "log", "logs", "stream", "streams",
    "document store", "collection", "collections",
    "mongodb", "elasticsearch", "nosql",
}


def _count_signal_hits(query_lower: str, keywords: set) -> int:
    """Returns the number of keyword matches in the query."""
    count = 0
    for kw in keywords:
        if kw in query_lower:
            count += 1
    return count



def _check_value_filter(query_lower: str) -> bool:
    """Return True if any query token matches a known DB column value.

    Signals a WHERE-clause filter query — when True the rag_score is
    discounted before threshold comparison so pure-SQL queries are not
    mis-routed to RAG.
    """
    try:
        from ingestion.value_sampler import expand_query_tokens
        tokens = query_lower.split()
        _, expansion_map = expand_query_tokens(tokens)
        return bool(expansion_map)
    except Exception:
        return False



# =============================================================================
# Source-type helpers
# =============================================================================

def _source_ids_by_type(sources: List[dict], source_type: str) -> List[str]:
    return [s["id"] for s in sources if s.get("type") == source_type and s.get("enabled", True)]


# =============================================================================
# Public entry point
# =============================================================================

def route_query(
    query:             str,
    available_sources: List[dict] = None,
    verbose:           bool = False,
) -> RouteResult:
    """
    Routes a user query to the appropriate pipeline intent.

    Parameters
    ----------
    query             : raw user query string
    available_sources : list of source config dicts from VEDA_SOURCES
                        (defaults to all enabled sources from config)
    verbose           : print routing decision

    Returns
    -------
    RouteResult with intent, source_ids, confidence, and reason
    """
    logger.debug("Router: query=%r", query[:120])

    if not QUERY_ROUTER_ENABLED:
        # Router disabled — always route to SQL on primary relational source
        from config import get_primary_relational_source
        try:
            primary = get_primary_relational_source()
            sql_ids = [primary["id"]]
        except ValueError:
            sql_ids = []
        result = RouteResult(
            intent     = "sql",
            source_ids = sql_ids,
            confidence = 1.0,
            reason     = "QUERY_ROUTER_ENABLED=False — defaulting to SQL",
        )
        logger.info("Router: intent=%s, confidence=%.2f (router disabled)", result.intent, result.confidence)
        return result

    if available_sources is None:
        from config import get_enabled_sources
        available_sources = get_enabled_sources()

    query_lower = query.lower().strip()

    # Collect available source types
    relational_ids = _source_ids_by_type(available_sources, "relational")
    datalake_ids   = _source_ids_by_type(available_sources, "datalake")
    document_ids   = _source_ids_by_type(available_sources, "document")
    nosql_ids      = _source_ids_by_type(available_sources, "nosql")

    sql_capable_ids = relational_ids + datalake_ids  # used for hybrid routing only

    # SQL queries target relational sources exclusively — datalake CSV columns
    # (e.g. permissions exports) pollute L2 top-K for any generic status/state query.
    # Datalake sources are included in HYBRID intent where SQL+doc fusion is explicit.
    sql_ids = relational_ids if relational_ids else sql_capable_ids

    # ------------------------------------------------------------------
    # No document or nosql sources → always SQL
    # ------------------------------------------------------------------
    if not document_ids and not nosql_ids:
        return RouteResult(
            intent     = "sql",
            source_ids = sql_ids,
            confidence = 1.0,
            reason     = "No document or NoSQL sources configured — routing to SQL",
        )


    # ------------------------------------------------------------------
    # Value filter check: if query tokens match known DB column values,
    # suppress RAG signal (it's a WHERE-clause filter query, not a doc search)
    # ------------------------------------------------------------------
    has_value_filter = _check_value_filter(query_lower)

    # ------------------------------------------------------------------
    # Keyword signal scoring
    # ------------------------------------------------------------------
    sql_hits   = _count_signal_hits(query_lower, _SQL_KEYWORDS)
    sql_hits  += _count_signal_hits(query_lower, _TEMPORAL_KEYWORDS) * 2  # temporal → strong SQL
    rag_hits   = _count_signal_hits(query_lower, _RAG_KEYWORDS)
    nosql_hits = _count_signal_hits(query_lower, _NOSQL_KEYWORDS)

    # Value filter discount: suppress RAG signal when tokens match DB values
    if has_value_filter:
        rag_hits = max(0, rag_hits - int(rag_hits * 0.4 + 0.5))

    total = sql_hits + rag_hits + nosql_hits
    if total == 0:
        # No keyword signals — fall back to relational/datalake if available
        intent     = "sql" if sql_ids else "rag"
        source_ids = sql_ids if sql_ids else document_ids
        return RouteResult(
            intent     = intent,
            source_ids = source_ids,
            confidence = 0.5,
            reason     = "No keyword signals — defaulting to SQL",
        )

    sql_score   = sql_hits   / total
    rag_score   = rag_hits   / total
    nosql_score = nosql_hits / total

    # ------------------------------------------------------------------
    # Decide intent based on dominant signal
    # ------------------------------------------------------------------

    # NoSQL: explicit signal + nosql sources available
    if nosql_score > 0.4 and nosql_ids:
        return RouteResult(
            intent     = "nosql",
            source_ids = nosql_ids,
            confidence = round(nosql_score, 3),
            reason     = f"NoSQL keyword signals ({nosql_hits} hits)",
            stats      = {"sql_hits": sql_hits, "rag_hits": rag_hits, "nosql_hits": nosql_hits},
        )

    # Hybrid: both SQL and RAG signals are significant
    if sql_hits >= 1 and rag_hits >= 1 and document_ids:
        hybrid_ids = sql_capable_ids + document_ids
        confidence = 1.0 - abs(sql_score - rag_score)   # higher when signals are balanced
        return RouteResult(
            intent     = "hybrid",
            source_ids = hybrid_ids,
            confidence = round(confidence, 3),
            reason     = f"Mixed SQL+RAG signals (sql={sql_hits}, rag={rag_hits})",
            stats      = {"sql_hits": sql_hits, "rag_hits": rag_hits},
        )

    # RAG: document signals dominate and doc sources are available
    if rag_score > sql_score and document_ids:
        return RouteResult(
            intent     = "rag",
            source_ids = document_ids,
            confidence = round(rag_score, 3),
            reason     = f"RAG keyword signals dominate ({rag_hits} hits)",
            stats      = {"sql_hits": sql_hits, "rag_hits": rag_hits},
        )

    # SQL: default for sql-capable sources
    result = RouteResult(
        intent     = "sql",
        source_ids = sql_ids or sql_capable_ids or (document_ids if not sql_capable_ids else []),
        confidence = round(sql_score if sql_hits > 0 else 0.6, 3),
        reason     = f"SQL signals dominate ({sql_hits} hits)",
        stats      = {"sql_hits": sql_hits, "rag_hits": rag_hits},
    )
    logger.info("Router: intent=%s, sources=%s, confidence=%.2f, reason=%s",
                result.intent, result.source_ids, result.confidence, result.reason)
    return result
