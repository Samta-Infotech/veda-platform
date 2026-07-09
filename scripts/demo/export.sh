#!/usr/bin/env bash
# Export everything a QUERY-ONLY demo needs from this (ingesting) machine:
#   1. veda_engine DB  — embeddings/graphs/metadata the query path reads
#   2. veda DB         — Django substrate (sources registry, auth, query log)
#   3. veda_core/data  — semantic model + graphs + per-source parquet (NOT in any DB)
#   4. model_cache vol — BGE-M3 + reranker HF cache
#   5. ollama vol      — the pulled SLM (so the demo box need not re-pull)
#
# Usage:  scripts/demo/export.sh [OUTPUT_DIR]
# Env:    PGHOST=localhost PGPORT=15432 PGUSER=veda PGPASSWORD=change-me
#         COMPOSE_PROJECT=veda-platform   (docker volume name prefix)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"        # repo root
OUT="${1:-$HERE/demo_bundle}"
PROJECT="${COMPOSE_PROJECT:-veda-platform}"
PGHOST="${PGHOST:-localhost}"; PGPORT="${PGPORT:-15432}"; PGUSER="${PGUSER:-veda}"
export PGPASSWORD="${PGPASSWORD:-change-me}"
mkdir -p "$OUT"

echo "==> [1/5] pg_dump veda_engine (direct, not through pgbouncer)"
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -Fc veda_engine > "$OUT/veda_engine.dump"

echo "==> [2/5] pg_dump veda (Django substrate)"
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -Fc veda > "$OUT/veda_django.dump"

echo "==> [3/5] tar veda_core/data (filesystem artifacts — not captured by pg_dump)"
tar czf "$OUT/veda_data.tgz" -C "$HERE/veda_core" data

echo "==> [4/5] export model_cache volume (${PROJECT}_model_cache)"
docker run --rm -v "${PROJECT}_model_cache:/models" -v "$OUT:/backup" alpine \
    tar czf /backup/model_cache.tgz -C /models .

echo "==> [5/5] export ollama_models volume (${PROJECT}_ollama_models)"
docker run --rm -v "${PROJECT}_ollama_models:/m" -v "$OUT:/backup" alpine \
    tar czf /backup/ollama_models.tgz -C /m . 2>/dev/null || echo "    (skipped: no ollama_models volume — demo box will pull the SLM instead)"

# Optional: fine-tuned BGE checkpoint, only if present (absent on some machines)
if [ -f "$HERE/veda_core/ingestion/client_bge/model.safetensors" ]; then
    echo "==> (extra) tar fine-tuned client_bge checkpoint"
    tar czf "$OUT/client_bge.tgz" -C "$HERE/veda_core/ingestion" client_bge
fi

echo ""
echo "==> done. Bundle at: $OUT"
du -sh "$OUT"/* 2>/dev/null || true
echo "    Ship the whole directory to the demo box, then run scripts/demo/restore.sh there."
