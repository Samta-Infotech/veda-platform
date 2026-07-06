# =============================================================================
# retrieval/retrieval_engine_phase3.py
# VEDA Phase 3c - 5-Signal Hybrid Retrieval Orchestrator (NEW)
#
# Purpose:
#   Orchestrates all Phase 3 enhancements into single retrieval pipeline
#   - Phase 3a: Query enrichment (domain synonyms, concepts, glossary)
#   - Phase 3b: BGE-M3 semantic search (Signal 1)
#   - Phase 3d: Intent-aware boosting
#   - Phase 3e: Adaptive cutoff
#   - Phase 3f: Result caching
#
# Pipeline:
#   User query → enrich → 5 signals → RRF fusion → intent boost →
#   adaptive cutoff → cache → return top-15
#
# Status: Phase 3c (READY TO INTEGRATE)
# =============================================================================

import sys
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import psycopg2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SEMANTIC_MODEL_FILE
from utils.logger import get_logger

# Phase 3 components
from .query_enrichment import QueryEnricher
from .semantic_search import SemanticSearchEngine
from .intent_boosting import IntentBooster
from .adaptive_cutoff import AdaptiveCutoff
from .retrieval_cache import RetrievalCache
from .bm25_ranker import BM25Ranker
from .signal_builder import SignalBuilder
from .rrf_merger import RRFMerger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result."""

    col_id: str
    column_name: str
    table_name: str
    final_score: float

    # Signal scores (for debugging)
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    subgraph_score: float = 0.0
    fk_path_score: float = 0.0
    value_index_score: float = 0.0
    rrf_score: float = 0.0
    boosted_score: float = 0.0


class RetrievalEnginePhase3:
    """
    5-Signal Hybrid Retrieval Engine (Phase 3).

    Signals:
      1. BGE-M3 semantic search (1024-dim embeddings)
      2. BM25 keyword matching (enriched tokens)
      3. FK subgraph signals (table proximity)
      4. FK path bridges (high confidence joins)
      5. Value index (literal lookups)

    Fusion: Reciprocal Rank Fusion (equal weighting)
    Boost: Intent-aware score adjustments
    Cutoff: Adaptive (semantic cliff detection)
    Cache: 5-min TTL on enriched token hash
    """

    def __init__(
        self,
        semantic_model_file: str = SEMANTIC_MODEL_FILE,
        db_config: Optional[Dict] = None,
        use_cache: bool = True,
        use_redis: bool = False
    ):
        """
        Initialize 5-signal retrieval engine.

        Args:
            semantic_model_file: Path to L2 semantic model
            db_config: PostgreSQL config (for semantic search)
            use_cache: Enable result caching
            use_redis: Use Redis for cache (vs file-based)
        """
        self.semantic_model_file = semantic_model_file
        self.db_config = db_config or self._default_db_config()
        self.use_cache = use_cache
        self.use_redis = use_redis

        self.semantic_model = {}
        self.enricher = None
        self.semantic_searcher = None
        self.bm25_ranker = None
        self.signal_builder = None
        self.intent_booster = None
        self.rrf_merger = None
        self.adaptive_cutoff = None
        self.cache = None

        self._initialize()

    def _default_db_config(self) -> Dict:
        """Default PostgreSQL config."""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "veda_db"),
            "user": os.getenv("DB_USER", "veda_user"),
            "password": os.getenv("DB_PASSWORD", "password")
        }

    def _initialize(self):
        """Initialize all Phase 3 components."""
        logger.info("\n" + "="*70)
        logger.info("INITIALIZING PHASE 3 - 5-SIGNAL HYBRID RETRIEVAL")
        logger.info("="*70)

        # Load semantic model
        logger.info("\n[1/8] Loading semantic model...")
        with open(self.semantic_model_file) as f:
            self.semantic_model = json.load(f)
        logger.info(f"✓ Loaded semantic model")

        # Initialize components
        logger.info("\n[2/8] Initializing query enricher...")
        self.enricher = QueryEnricher()
        logger.info("✓ Query enricher ready")

        logger.info("\n[3/8] Initializing BGE-M3 semantic search...")
        try:
            self.semantic_searcher = SemanticSearchEngine(self.db_config)
            logger.info("✓ Semantic searcher ready")
        except Exception as e:
            logger.error(f"✗ Semantic search init failed: {e}")
            self.semantic_searcher = None

        logger.info("\n[4/8] Initializing BM25 ranker...")
        self.bm25_ranker = None
        # Q-2: load the persisted BM25 index (built at ingestion) instead of re-fitting
        # tokenization + IDF on every process warm. Scores are identical. Flag-gated;
        # falls back to a live fit() when the index is absent or loading fails.
        try:
            from config import BM25_PERSISTED_INDEX_ENABLED
            if BM25_PERSISTED_INDEX_ENABLED:
                from ingestion.bm25_index import load_bm25_index
                self.bm25_ranker = load_bm25_index()
                if self.bm25_ranker is not None:
                    logger.info("✓ BM25 ranker loaded from persisted index")
        except Exception as e:
            logger.warning(f"BM25 persisted-index load failed ({e}); falling back to fit()")
            self.bm25_ranker = None
        if self.bm25_ranker is None:
            self.bm25_ranker = BM25Ranker()
            self.bm25_ranker.fit(self.semantic_model)
        logger.info("✓ BM25 ranker ready")

        logger.info("\n[5/8] Initializing signal builder...")
        self.signal_builder = SignalBuilder()
        self.signal_builder.build_signals(self.semantic_model)
        logger.info("✓ Signal builder ready")

        logger.info("\n[6/8] Initializing intent booster...")
        self.intent_booster = IntentBooster(self.semantic_model)
        logger.info("✓ Intent booster ready")

        logger.info("\n[7/8] Initializing RRF merger (5-signal)...")
        self.rrf_merger = RRFMerger(k=60)
        logger.info("✓ RRF merger ready")

        logger.info("\n[8/8] Initializing adaptive cutoff...")
        self.adaptive_cutoff = AdaptiveCutoff(
            gap_threshold=0.28,
            min_k=5,
            max_k=20,
            hard_limit=15
        )
        logger.info("✓ Adaptive cutoff ready")

        if self.use_cache:
            logger.info("\nInitializing retrieval cache...")
            self.cache = RetrievalCache(use_redis=self.use_redis)
            logger.info("✓ Cache ready")

        logger.info("\n" + "="*70)
        logger.info("✓ PHASE 3 INITIALIZATION COMPLETE")
        logger.info("="*70 + "\n")

    def retrieve(
        self,
        query: str,
        intent: str = "SIMPLE",
        top_k: int = 15,
        use_cache: bool = True
    ) -> List[RetrievalResult]:
        """
        5-signal hybrid retrieval pipeline.

        Args:
            query: User natural language query
            intent: Query intent (AGGREGATE, TEMPORAL, MULTI_TABLE, DIRECT, SIMPLE)
            top_k: Number of results to return
            use_cache: Use cached results if available

        Returns:
            Top-K ranked columns
        """
        logger.info("\n" + "="*70)
        logger.info(f"RETRIEVAL: {query}")
        logger.info(f"Intent: {intent} | Caching: {use_cache}")
        logger.info("="*70)

        start_time = time.time()

        # STEP 1: QUERY ENRICHMENT
        logger.info("\n[STEP 1/7] Query enrichment...")
        enriched_tokens = self.enricher.enrich(query)

        # STEP 2: CHECK CACHE
        if use_cache and self.cache:
            logger.info("[STEP 2/7] Checking cache...")
            cached_results = self.cache.get(enriched_tokens)
            if cached_results:
                logger.info(f"✓ Cache hit! Returning {len(cached_results)} results")
                results = self._results_from_tuples(cached_results)
                elapsed = time.time() - start_time
                logger.info(f"\n✓ TOTAL TIME: {elapsed*1000:.0f}ms (cached)")
                return results

        logger.info("[STEP 2/7] Cache miss, running full retrieval...")

        # STEP 3: 5-SIGNAL RETRIEVAL
        logger.info("\n[STEP 3/7] 5-Signal Retrieval...")

        # Signal 1: BGE-M3 semantic
        signal1_semantic = []
        if self.semantic_searcher:
            logger.info("  - Signal 1: BGE-M3 semantic search (1024-dim)...")
            signal1_semantic = self.semantic_searcher.search(enriched_tokens, k=50)
            logger.info(f"    ✓ {len(signal1_semantic)} results")

        # Signal 2: BM25 keyword
        logger.info("  - Signal 2: BM25 keyword matching (enriched)...")
        signal2_bm25 = self.bm25_ranker.rank(query, top_k=50) or []
        logger.info(f"    ✓ {len(signal2_bm25)} results")

        # Signal 3: FK subgraph
        logger.info("  - Signal 3: FK subgraph signals...")
        signal3_subgraph_signals = {}
        for col_id in self.semantic_model.get("columns", {}).keys():
            score = self.signal_builder.get_signal(col_id, "subgraph_signal") or 0.0
            if score > 0:
                signal3_subgraph_signals[col_id] = score
        logger.info(f"    ✓ {len(signal3_subgraph_signals)} signals")

        # Signal 4: FK path bridges
        logger.info("  - Signal 4: FK path bridges...")
        signal4_fk_signals = {}
        for col_id in self.semantic_model.get("columns", {}).keys():
            score = self.signal_builder.get_signal(col_id, "fk_signal") or 0.0
            if score > 0:
                signal4_fk_signals[col_id] = score
        logger.info(f"    ✓ {len(signal4_fk_signals)} signals")

        # Signal 5: Value index (literals from query). When the user names a VALUE that
        # isn't a column NAME ("escalated", "level 1"), match it against each column's
        # sampled values so the column that HOLDS the value is surfaced — the same value
        # grounding the deterministic fast path uses. Keyed by the semantic-model col_id
        # so it fuses with signals 3/4. Multi-word values ("level 1") are caught via the
        # n-gram tokenizer; values equal to the column's own table name are skipped
        # ("incident" → object_type='Incident' is an entity reference, not a filter value).
        logger.info("  - Signal 5: Value index direct lookups...")
        signal5_value_signals = {}
        try:
            from query.value_filter import _query_value_tokens
            _qphrases = set(_query_value_tokens(query))
            if _qphrases:
                for _cid, _cm in self.semantic_model.get("columns", {}).items():
                    _tname = (_cm.get("table_name") or _cid.split(".")[0]).lower()
                    _tname_toks = set(_tname.replace("_", " ").split())
                    for _v in (_cm.get("sample_values") or []):
                        _vn = str(_v).lower().strip()
                        if _vn and _vn in _qphrases and _vn not in _tname_toks:
                            signal5_value_signals[_cid] = 1.0
                            break
        except Exception as _e:
            logger.warning(f"    value signal skipped: {_e}")
        logger.info(f"    ✓ {len(signal5_value_signals)} signals")

        # STEP 4: RRF FUSION
        logger.info("\n[STEP 4/7] RRF Fusion (5-signal)...")
        fused = self.rrf_merger.merge(
            signal1_semantic,
            signal2_bm25,
            signal4_fk_signals,
            signal3_subgraph_signals,
            signal5_value_signals,
            top_k=50
        )
        logger.info(f"✓ Merged {len(fused)} candidates")

        # STEP 5: INTENT-AWARE BOOSTING
        logger.info(f"\n[STEP 5/7] Intent-aware boosting ({intent})...")
        boosted = self.intent_booster.boost(fused, intent)
        logger.info(f"✓ Boosted {len(boosted)} results")

        # STEP 6: ADAPTIVE CUTOFF
        logger.info("\n[STEP 6/7] Adaptive cutoff (semantic cliff detection)...")
        cutoff_results = self.adaptive_cutoff.cutoff(boosted)
        logger.info(f"✓ Cut to {len(cutoff_results)} results")

        # STEP 7: CACHE RESULTS
        if use_cache and self.cache:
            logger.info("\n[STEP 7/7] Caching results...")
            self.cache.set(enriched_tokens, cutoff_results)
            logger.info(f"✓ Cached {len(cutoff_results)} results")

        # CONVERT TO RESULT OBJECTS
        results = self._results_from_tuples(cutoff_results)

        # TIMING
        elapsed = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info(f"✓ RETRIEVAL COMPLETE")
        logger.info(f"  Results: {len(results)}")
        logger.info(f"  Time: {elapsed*1000:.0f}ms")
        logger.info("="*70)

        return results

    def _results_from_tuples(
        self,
        tuples: List[Tuple[str, float]]
    ) -> List[RetrievalResult]:
        """Convert (col_id, score) tuples to RetrievalResult objects."""
        results = []
        for col_id, score in tuples:
            table_name, col_name = col_id.rsplit(".", 1) if "." in col_id else (col_id, col_id)
            result = RetrievalResult(
                col_id=col_id,
                column_name=col_name,
                table_name=table_name,
                final_score=score
            )
            results.append(result)
        return results

    def close(self):
        """Close connections."""
        if self.semantic_searcher:
            self.semantic_searcher.close()
        logger.info("✓ Retrieval engine closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def retrieve(
    query: str,
    intent: str = "SIMPLE",
    top_k: int = 15,
    semantic_model_file: str = SEMANTIC_MODEL_FILE,
    db_config: Optional[Dict] = None
) -> List[RetrievalResult]:
    """
    Standalone retrieval function.

    Args:
        query: User query
        intent: Query intent
        top_k: Number of results
        semantic_model_file: Path to semantic model
        db_config: PostgreSQL config

    Returns:
        Top-K ranked columns
    """
    engine = RetrievalEnginePhase3(
        semantic_model_file=semantic_model_file,
        db_config=db_config
    )
    try:
        return engine.retrieve(query, intent, top_k)
    finally:
        engine.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Initialize engine
    engine = RetrievalEnginePhase3(use_cache=True)

    try:
        # Example 1: AGGREGATE query
        results = engine.retrieve(
            query="show me total payments last 30 days",
            intent="AGGREGATE",
            top_k=15
        )
        print(f"\nFound {len(results)} columns:")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r.col_id} ({r.final_score:.3f})")

        # Example 2: TEMPORAL query
        results = engine.retrieve(
            query="payments from last month",
            intent="TEMPORAL",
            top_k=15
        )
        print(f"\nFound {len(results)} columns")

    finally:
        engine.close()
