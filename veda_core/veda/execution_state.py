# veda/execution_state.py
# VEDA — lightweight Tier1→Tier2 execution context (Tier1/Tier2 propagation refactor).
#
# INTERNAL ONLY: this dataclass must never cross an HTTP boundary. Enforced (not just
# asserted in this comment) by inference/routes/hybrid.py::_serialize(), which strips
# the "context" key (and "trace") from every response before it's returned — that is
# the one place ALL head results (SQL/Tier-2/RAG/hybrid/NoSQL) funnel through on their
# way to becoming a wire response, so it's the correct enforcement point. If a new
# response path is ever added that does NOT go through that function, it must apply
# the same stripping — never assume a "d.get(known_key)" allowlist elsewhere is enough.
#
# Carries just enough for Tier2 to continue Tier1's work instead of restarting cold —
# not a copy of Tier1's full trace (that stays in run_query()'s existing "trace" key,
# for debugging).
#
# candidate_fields (not "candidate_columns"): VEDA's retrieval spans more than
# relational columns (files, Delta Lake, etc. are on the roadmap), so this uses a
# connector-agnostic name. Each entry is a plain dict — deliberately NOT a
# connector-specific RetrievalResult, so this module has zero import coupling to any
# single retrieval backend. Backward-compatible schema (older consumers that read
# only the first three keys keep working):
#   {"table_name", "col_name", "score",          # score = final (post-rerank, +boost)
#    "semantic_type",   # from the semantic model — lets Tier2 tell a resolved MEASURE
#                       # (METRIC/MONETARY) from a DIMENSION (CATEGORY) from an
#                       # IDENTIFIER, without re-reading metadata
#    "rrf_score",       # RAW 5-signal RRF score BEFORE cross-encoder rerank
#    "cross_encoder_score",  # reranked score, or None if this field wasn't reranked
#    "reranked"}        # bool — was this candidate rescored by the cross-encoder
# This preserves the retrieval PROVENANCE the audit found was flattened (RC-5): which
# score is raw vs reranked, and — via rerank_query below — against WHAT text.
#
# Consumption status (kept honest here so it can't drift from what veda_hybrid.py's
# reuse log claims): temporal_result and candidate_fields are ACTIVELY consumed by
# _tier2_sql. primary_table is consumed indirectly — pipeline.py boosts matching
# entries in candidate_fields by PRIMARY_TABLE_SEED_BOOST before Tier2 ever sees it,
# rather than Tier2 branching on primary_table directly. query_understanding and
# sql_planning are populated but NOT YET consumed by any Tier2 decision — reserved for
# future use; do not assume they influence behavior today.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionState:
    temporal_result:      object = None                 # TemporalParseResult (query.temporal_parser) — reused
    query_understanding:  Dict[str, Any] = field(default_factory=dict)   # intent/existence/aggregation — reserved, not yet consumed
    primary_table:        Optional[str] = None           # reused indirectly (see module docstring)
    candidate_tables:     List[str] = field(default_factory=list)        # informational only — table_name is already in candidate_fields
    candidate_fields:     List[Dict[str, Any]] = field(default_factory=list)  # enriched provenance dicts (see module docstring) — reused
    rerank_query:         Optional[str] = None            # the (enhanced) text the cross-encoder reranked against, or None if no rerank ran
    sql_planning:         Dict[str, Any] = field(default_factory=dict)   # action/anchor hints — reserved, not yet consumed
    refusal_reason:       Optional[str] = None            # reused (seeds the repair hint)
