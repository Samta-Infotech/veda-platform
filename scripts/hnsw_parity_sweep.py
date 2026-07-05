"""Phase 7.1a — HNSW ef_search sweep to recall@k = 1.0 vs exact cosine.

Runs in the inference container (has BGE encoder + psycopg2). Steps:
  1. Populate the Django-owned HNSW table `column_embeddings_bge` (veda DB) from the
     engine's BGE store `column_embeddings_v2` (veda_engine) if empty.
  2. Encode a set of representative NL queries with the SAME BGE model retrieval uses.
  3. For each query vector, compute the EXACT cosine top-k (forced seqscan) and the
     HNSW top-k at several ef_search values; measure recall@k = |exact ∩ hnsw| / k.
  4. Report the lowest ef_search that reaches recall@k = 1.0 across all fixtures.

The pinned HNSW_EF_SEARCH (and build m/ef_construction) become the shipping params so
the gated index IS the shipped index (§7.1a).
"""
import os
import sys

import psycopg2

K = 20
QUERIES = [
    "how many users are there",
    "list all change requests",
    "total number of incidents",
    "show workflow states",
    "annotations created by editor",
    "checklist templates",
    "requests for information comments",
    "signal rules by priority",
    "documents and their versions",
    "users who created change requests",
]
EF_VALUES = [40, 100, 200, 400, 800]


def _veda():
    c = psycopg2.connect(host=os.environ.get("PGBOUNCER_HOST", "pgbouncer"),
                         port=int(os.environ.get("PGBOUNCER_PORT", "6432")),
                         dbname="veda", user=os.environ.get("POSTGRES_USER", "veda"),
                         password=os.environ.get("POSTGRES_PASSWORD", "change-me"))
    c.autocommit = True
    return c


def _engine():
    c = psycopg2.connect(host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
                         port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
                         dbname="veda_engine", user=os.environ.get("VEDA_INTERNAL_USER", "veda"),
                         password=os.environ.get("VEDA_INTERNAL_PASSWORD", "change-me"))
    c.autocommit = True
    return c


def populate():
    v = _veda()
    with v.cursor() as cur:
        cur.execute("SELECT count(*) FROM column_embeddings_bge")
        if cur.fetchone()[0] > 0:
            print("[populate] column_embeddings_bge already populated")
            return
    e = _engine()
    with e.cursor() as cur:
        cur.execute("SELECT col_id, embedding FROM column_embeddings_v2 WHERE col_id IS NOT NULL")
        rows = cur.fetchall()
    with v.cursor() as cur:
        for col_id, emb in rows:
            cur.execute(
                "INSERT INTO column_embeddings_bge (column_uuid, source_id, tenant, embedding) "
                "VALUES (%s::uuid, 1, 'default', %s::vector) ON CONFLICT DO NOTHING",
                (col_id, emb),
            )
    print(f"[populate] inserted {len(rows)} BGE embeddings into column_embeddings_bge")


def encode(queries):
    """Encode queries with the same BGE model retrieval uses (config.BGE_MODEL_NAME)."""
    from sentence_transformers import SentenceTransformer
    from veda_core.config import BGE_MODEL_NAME, BGE_DEVICE
    model = SentenceTransformer(BGE_MODEL_NAME, device=BGE_DEVICE)
    vecs = model.encode(queries, normalize_embeddings=True)
    return [("[" + ",".join(str(float(x)) for x in v) + "]") for v in vecs]


def topk_exact(cur, qvec, k):
    cur.execute("SET LOCAL enable_indexscan = off")
    cur.execute("SET LOCAL enable_bitmapscan = off")
    cur.execute(
        "SELECT column_uuid FROM column_embeddings_bge WHERE source_id=1 AND tenant='default' "
        "ORDER BY embedding <=> %s::vector LIMIT %s", (qvec, k))
    return [r[0] for r in cur.fetchall()]


def topk_hnsw(cur, qvec, k, ef):
    cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")
    cur.execute("SET LOCAL enable_seqscan = off")
    cur.execute(
        "SELECT column_uuid FROM column_embeddings_bge WHERE source_id=1 AND tenant='default' "
        "ORDER BY embedding <=> %s::vector LIMIT %s", (qvec, k))
    return [r[0] for r in cur.fetchall()]


def main():
    populate()
    print(f"[encode] encoding {len(QUERIES)} representative queries with BGE ...")
    qvecs = encode(QUERIES)

    v = _veda()
    exact = {}
    with v.cursor() as cur:
        for q, qv in zip(QUERIES, qvecs):
            exact[q] = set(topk_exact(cur, qv, K))

    print(f"\nrecall@{K} vs exact cosine, by ef_search:")
    pinned = None
    for ef in EF_VALUES:
        recalls = []
        with v.cursor() as cur:
            for q, qv in zip(QUERIES, qvecs):
                got = set(topk_hnsw(cur, qv, K, ef))
                recalls.append(len(got & exact[q]) / float(K))
        avg = sum(recalls) / len(recalls)
        mn = min(recalls)
        print(f"  ef_search={ef:>4}  recall@{K} avg={avg:.4f}  min={mn:.4f}")
        if pinned is None and mn >= 1.0:
            pinned = ef
    print(f"\n[result] lowest ef_search with recall@{K}=1.0 on all fixtures: {pinned}")
    if pinned is None:
        print("[result] recall@k=1.0 not reached in swept range — raise ef further / rebuild m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
