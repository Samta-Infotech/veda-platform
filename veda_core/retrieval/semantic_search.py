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
from config import BIENCODER_QUERY_PREFIX   # WP1: query prefix, matches how passages were stored

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
        """Use the ONE shared BGE-M3 dense facade (WP3) — the same underlying model that
        produced the stored column vectors, so query and passage share one space and the
        process holds a single copy."""
        try:
            from ingestion import m3_encoder
            self.model = m3_encoder.get_dense_encoder()
            logger.info("✓ Model reused (shared BGE-M3 — no duplicate load)")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 dense encoder: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed the RAW natural-language query to a 1024-dimensional vector.

        WP1 correctness fix: the prior path joined ENRICHED TOKENS with spaces and
        encoded them with no query prefix and no normalization — a train/serve
        mismatch against the column passages, which were doc-encoded and
        L2-normalized. We now encode BIENCODER_QUERY_PREFIX + the raw query with
        normalize_embeddings=True so the query vector lives in the SAME normalized
        space as the stored passages. Enrichment moves to the sparse/value signals.

        Args:
            query: Raw natural-language query (NOT enriched tokens)

        Returns:
            1024-dimensional embedding as list of floats
        """
        text = f"{BIENCODER_QUERY_PREFIX}{query}"

        logger.info(f"Embedding query: {text}")

        # Encode to 1024-dim vector, normalized to match stored passages.
        embedding = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True,
        )

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

            # HNSW: pin ef_search for this transaction so the served ANN ordering matches
            # the tuned recall target (WP2). Resolved per-source via the one shared helper;
            # SET LOCAL is released at COMMIT (transaction-pool-safe).
            from storage_adapters.reader import _resolve_ef_search
            from veda_core.context import try_current
            _c = try_current()
            _ef = _resolve_ef_search(_c.source_id if _c is not None else None)

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

            cur.execute("BEGIN")
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(_ef)}")
            cur.execute(query, (embedding_str, embedding_str, k))
            results = cur.fetchall()
            cur.execute("COMMIT")
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

    def search(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        """
        Semantic search given the RAW natural-language query (WP1).

        Args:
            query: Raw natural-language query (NOT enriched tokens — enrichment now
                lives only in the sparse/value signals)
            k: Number of results to return

        Returns:
            List of (col_id, similarity_score) tuples
        """
        # Step 1: Embed the raw query (prefixed + normalized)
        embedding = self.searcher.embed_query(query)

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
    query: str,
    db_config: dict,
    k: int = 50
) -> List[Tuple[str, float]]:
    """
    Standalone semantic search function.

    Args:
        query: Raw natural-language query (NOT enriched tokens)
        db_config: PostgreSQL config
        k: Number of results

    Returns:
        List of (col_id, similarity_score) tuples
    """
    engine = SemanticSearchEngine(db_config)
    try:
        results = engine.search(query, k)
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

    # Example raw query
    query = "show total payments amount last 30 days"

    # Search
    try:
        with SemanticSearchEngine(db_config) as engine:
            results = engine.search(query, k=50)
            print(f"\nSemantic search results:")
            for i, (col_id, sim_score) in enumerate(results[:5], 1):
                print(f"  {i}. {col_id}: {sim_score:.3f}")
    except Exception as e:
        print(f"Error: {e}")
