#!/usr/bin/env bash
# Restore a QUERY-ONLY demo bundle (produced by scripts/demo/export.sh) on the demo
# box, then bring up the query-only stack. Run from the repo root with .env and
# docker/userlist.txt copied from the source machine.
#
# Usage:  scripts/demo/restore.sh [BUNDLE_DIR]
# Env:    COMPOSE_PROJECT=veda-platform
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
IN="${1:-$HERE/demo_bundle}"
PROJECT="${COMPOSE_PROJECT:-veda-platform}"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.demo.yml"

[ -f "$IN/veda_engine.dump" ] || { echo "!! $IN/veda_engine.dump not found"; exit 1; }

echo "==> [1/5] import model_cache volume"
docker volume create "${PROJECT}_model_cache" >/dev/null
docker run --rm -v "${PROJECT}_model_cache:/models" -v "$IN:/backup" alpine \
    tar xzf /backup/model_cache.tgz -C /models

echo "==> [2/5] import ollama_models volume (if bundled)"
docker volume create "${PROJECT}_ollama_models" >/dev/null
if [ -f "$IN/ollama_models.tgz" ]; then
    docker run --rm -v "${PROJECT}_ollama_models:/m" -v "$IN:/backup" alpine \
        tar xzf /backup/ollama_models.tgz -C /m
else
    echo "    (no ollama bundle — will pull the SLM after startup)"
fi

echo "==> [3/5] restore filesystem artifacts into veda_core/data"
tar xzf "$IN/veda_data.tgz" -C "$HERE/veda_core"
[ -f "$IN/client_bge.tgz" ] && tar xzf "$IN/client_bge.tgz" -C "$HERE/veda_core/ingestion"

echo "==> [4/5] start postgres and load the dumps"
$COMPOSE up -d postgres
echo -n "    waiting for postgres"; until docker compose exec -T postgres pg_isready -U veda >/dev/null 2>&1; do echo -n .; sleep 2; done; echo " ready"
docker compose exec -T postgres createdb -U veda veda_engine 2>/dev/null || echo "    veda_engine exists — reusing"
docker compose exec -T postgres pg_restore -U veda -d veda_engine --clean --if-exists --no-owner < "$IN/veda_engine.dump"
docker compose exec -T postgres pg_restore -U veda -d veda        --clean --if-exists --no-owner < "$IN/veda_django.dump"

echo "==> source-kind preflight (relational sources need a LIVE source DB reachable from this box)"
NEEDS_LIVE_DB=0
while IFS='|' read -r sid dialect; do
    [ -z "$sid" ] && continue
    case "$dialect" in
        parquet|csv|csv_lake|xlsx|excel)
            echo "    source $sid ($dialect): self-contained — DuckDB over parquet ✓" ;;
        *)
            echo "    source $sid ($dialect): RELATIONAL — its live source DB must be reachable ⚠"
            NEEDS_LIVE_DB=1 ;;
    esac
done < <(docker compose exec -T postgres psql -U veda -d veda -tA -F'|' \
            -c "SELECT id, dialect FROM sources_source WHERE ready = true ORDER BY id;" 2>/dev/null)
[ "$NEEDS_LIVE_DB" = "1" ] && echo "    !! at least one ready source is relational — ensure its DB is reachable, or re-materialize it as parquet before export."

echo "==> [5/5] start the query-only stack"
$COMPOSE up -d postgres pgbouncer redis-broker redis-cache ollama inference api

# If the SLM wasn't bundled, pull it now (name from .env SLM_MODEL_NAME, default qwen2.5-coder:7b)
if [ ! -f "$IN/ollama_models.tgz" ]; then
    MODEL="$(grep -E '^SLM_MODEL_NAME=' "$HERE/.env" 2>/dev/null | cut -d= -f2)"; MODEL="${MODEL:-qwen2.5-coder:7b}"
    echo "==> pulling SLM $MODEL into ollama"
    docker compose exec ollama ollama pull "$MODEL"
fi

echo ""
echo "==> waiting for warm-load, then readiness check:"
sleep 8
curl -s http://localhost:8000/readyz | { command -v jq >/dev/null && jq || cat; }
echo ""
echo "==> demo up. Try:"
echo "    curl -s -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' \\"
echo "         -d '{\"query\": \"your natural-language question\"}' | jq"
