# =============================================================================
# retrieval/semantic_search.py
# VEDA Phase 3b - BGE-M3 Semantic Search (Signal 1)
#
# Purpose:
#   Semantic similarity search using BGE-M3 (1024-dim embeddings)
#   Searches column embeddings in pgvector
#
# Input: Enriched query tokens + DB connection
# Output: Top-50 columns by semantic similarity
#
# Status: Phase 3b
# =============================================================================

import logging
import psycopg2
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_NAME   # unified model (cached); was hardcoded "BAAI/bge-m3"
from config import BIENCODER_COL_TABLE   # the 1024-dim BGE store that ensemble ingest writes

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """BGE-M3 semantic search over column embeddings."""

    def __init__(self, model_name: str = BGE_MODEL_NAME, device: str = "cpu"):
        """
        Initialize semantic searcher with BGE-M3 model.

        Args:
            model_name: HuggingFace model name
            device: torch device (cpu or cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load BGE-M3 model. Reuse the ONE shared instance (veda.runtime._get_bge) when it's
        the same model/device — avoids loading a second ~1.3GB copy of bge-large per process."""
        try:
            if self.device == "cpu":
                try:
                    from config import BGE_MODEL_NAME as _shared_name
                    if self.model_name == _shared_name:
                        from veda.runtime import _get_bge
                        self.model = _get_bge()
                        logger.info("✓ Model reused (shared BGE — no duplicate load)")
                        return
                except Exception:
                    pass   # any issue → fall back to an own load below
            logger.info(f"Loading {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device,
                                             local_files_only=True)
            logger.info(f"✓ Model loaded ({self.model.get_sentence_embedding_dimension()} dims)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def embed_query(self, tokens: List[str]) -> List[float]:
        """
        Embed enriched query tokens to 1024-dimensional vector.

        Args:
            tokens: List of query tokens (already enriched)

        Returns:
            1024-dimensional embedding as list of floats
        """
        # Join tokens into query text
        query_text = " ".join(tokens)

        logger.info(f"Embedding query: {query_text}")

        # Encode to 1024-dim vector
        embedding = self.model.encode(query_text, convert_to_numpy=True)

        logger.info(f"✓ Query embedding: {len(embedding)}-dim vector")

        return embedding.tolist()

    def retrieve_semantic(
        self,
        query_embedding: List[float],
        conn: psycopg2.extensions.connection,
        k: int = 50,
        schema: str = "public"
    ) -> List[Tuple[str, float]]:
        """
        Cosine similarity search in pgvector.

        Args:
            query_embedding: 1024-dim query vector
            conn: PostgreSQL connection
            k: Number of results to return
            schema: Database schema name

        Returns:
            List of (col_id, similarity_score) tuples
            Scores range from 0.0 (opposite) to 1.0 (identical)
        """
        # Phase 8 (B#6): Signal 1 routes through storage_adapters.ann_search → Django-owned
        # column_embeddings_bge (HNSW, per-(source,tenant), ef_search pinned to §7.1a
        # recall@k=1.0). This is the ACTIVE, multi-source-correct Signal-1 path: the store is
        # scoped to the request's ambient source_id, so one warm engine serves N sources.
        #
        # ON by default; set VEDA_ANN_VIA_ADAPTER=0 to fall back to the engine's own db_config
        # store (single-source, no source filter — dev/CLI only). NOTE (was gated off historically):
        # the engine's direct store returned 0 rows because db_config pointed at the L7 source
        # instead of the internal store, so retrieval ran on BM25 + 4 signals with adaptive-cutoff
        # tuned to an EMPTY Signal 1. Activating Signal 1 shifts the RRF/cutoff balance — recalibrate
        # ADAPTIVE_CUTOFF/RRF against eval traffic once there's data to tune on.
        import os as _os
        if _os.environ.get("VEDA_ANN_VIA_ADAPTER", "1") != "0":
            try:
                from veda_core.context import try_current
                if try_current() is not None:
                    from storage_adapters.reader import ann_search
                    rows = ann_search("bge", list(query_embedding), k)
                    out = [(str(cid), float(score)) for cid, score in rows]
                    logger.info(f"✓ Signal 1 via storage_adapters (Django HNSW): {len(out)} cols")
                    return out
            except Exception as _e:
                logger.warning(f"Signal 1 adapter unavailable ({_e}) — engine store")

        try:
            cur = conn.cursor()

            # Build embedding string for pgvector
            # pgvector expects format: "[1.0, 0.5, ...]"
            embedding_str = "[" + ", ".join(str(x) for x in query_embedding) + "]"

            # Execute cosine similarity search
            # <=> is pgvector's cosine distance operator
            # 1 - distance = similarity (0-1)
            query = f"""
                SELECT
                    col_id,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {schema}.{BIENCODER_COL_TABLE}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            cur.execute(query, (embedding_str, embedding_str, k))
            results = cur.fetchall()
            cur.close()

            # Convert results: (col_id, similarity)
            semantic_results = [(row[0], float(row[1])) for row in results]

            logger.info(f"✓ Found {len(semantic_results)} similar columns (Signal 1)")

            return semantic_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def batch_embed(self, text_list: List[str]) -> List[List[float]]:
        """
        Embed multiple texts (batch operation).

        Args:
            text_list: List of texts to embed

        Returns:
            List of embeddings
        """
        logger.info(f"Batch embedding {len(text_list)} texts...")

        embeddings = self.model.encode(text_list, convert_to_numpy=True)

        logger.info(f"✓ Batch embedded {len(embeddings)} texts")

        return [emb.tolist() for emb in embeddings]


class SemanticSearchEngine:
    """High-level API for semantic search."""

    def __init__(
        self,
        db_config: dict,
        model_name: str = BGE_MODEL_NAME,
        device: str = "cpu"
    ):
        """
        Initialize semantic search engine.

        Args:
            db_config: PostgreSQL connection config
            model_name: BGE model name
            device: torch device
        """
        self.db_config = db_config
        self.searcher = SemanticSearcher(model_name, device)
        self.conn = None

        self._connect_db()

    def _connect_db(self):
        """Connect to PostgreSQL."""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("database"),
                user=self.db_config.get("user"),
                password=self.db_config.get("password")
            )
            logger.info("✓ Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def search(self, enriched_tokens: List[str], k: int = 50) -> List[Tuple[str, float]]:
        """
        Semantic search given enriched tokens.

        Args:
            enriched_tokens: Enriched query tokens from query_enrichment
            k: Number of results to return

        Returns:
            List of (col_id, similarity_score) tuples
        """
        # Step 1: Embed query
        embedding = self.searcher.embed_query(enriched_tokens)

        # Step 2: Search in pgvector
        results = self.searcher.retrieve_semantic(embedding, self.conn, k)

        return results

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("✓ Disconnected from PostgreSQL")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# STANDALONE FUNCTIONS (for integration with retrieval_engine.py)
# ============================================================================

def retrieve_semantic(
    enriched_tokens: List[str],
    db_config: dict,
    k: int = 50
) -> List[Tuple[str, float]]:
    """
    Standalone semantic search function.

    Args:
        enriched_tokens: Enriched query tokens
        db_config: PostgreSQL config
        k: Number of results

    Returns:
        List of (col_id, similarity_score) tuples
    """
    engine = SemanticSearchEngine(db_config)
    try:
        results = engine.search(enriched_tokens, k)
        return results
    finally:
        engine.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example config (replace with actual)
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "veda_db",
        "user": "veda_user",
        "password": "password"
    }

    # Example enriched tokens (from phase 3a)
    enriched_tokens = [
        "show", "total", "payments", "amount", "sum", "last", "30", "days"
    ]

    # Search
    try:
        with SemanticSearchEngine(db_config) as engine:
            results = engine.search(enriched_tokens, k=50)
            print(f"\nSemantic search results:")
            for i, (col_id, sim_score) in enumerate(results[:5], 1):
                print(f"  {i}. {col_id}: {sim_score:.3f}")
    except Exception as e:
        print(f"Error: {e}")
