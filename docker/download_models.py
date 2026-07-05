"""Pre-pull query-time model weights into the model-cache volume (migration_plan.md §9).

Run once (online) so the inference container can warm-load offline (HF_HUB_OFFLINE=1).
Downloads into HF_HOME=/models. Models are the ones config.py actually loads:
  BAAI/bge-large-en-v1.5   (biencoder + Signal-1 BGE, 1024-dim)
  BAAI/bge-reranker-v2-m3  (cross-encoder reranker)
  all-MiniLM-L6-v2         (ensemble hybrid text encoder, 384-dim)
"""
import sys

from sentence_transformers import CrossEncoder, SentenceTransformer

BI = "BAAI/bge-large-en-v1.5"
RERANK = "BAAI/bge-reranker-v2-m3"
MINILM = "all-MiniLM-L6-v2"

print(f"[download] {BI} ...", flush=True)
SentenceTransformer(BI)
print(f"[download] {MINILM} ...", flush=True)
SentenceTransformer(MINILM)
print(f"[download] {RERANK} ...", flush=True)
CrossEncoder(RERANK)
print("[download] all three models cached OK", flush=True)
sys.exit(0)
