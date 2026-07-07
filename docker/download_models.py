"""Pre-pull query-time model weights into the model-cache volume (migration_plan.md §9).

Run once (online) so the inference container can warm-load offline (HF_HUB_OFFLINE=1).
Downloads into HF_HOME=/models. WP3 unified on ONE embedding model:
  BAAI/bge-m3              (unified dense + learned-sparse encoder, 1024-dim)
  BAAI/bge-reranker-v2-m3  (cross-encoder reranker — unchanged)

bge-large-en-v1.5 and all-MiniLM-L6-v2 are no longer used and are NOT baked.
"""
import sys

from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

M3 = "BAAI/bge-m3"
RERANK = "BAAI/bge-reranker-v2-m3"

print(f"[download] {M3} ...", flush=True)
BGEM3FlagModel(M3, use_fp16=False)   # pulls dense + sparse heads into the cache
print(f"[download] {RERANK} ...", flush=True)
CrossEncoder(RERANK)
print("[download] models cached OK", flush=True)
sys.exit(0)
