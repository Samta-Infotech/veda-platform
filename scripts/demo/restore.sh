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

echo "==> [2/5] pre-check the offloaded model hosts are reachable (mini 1 SLM, mini 2 BGE)"
OLLAMA_URL="$(grep -E '^OLLAMA_URL=' "$HERE/.env" 2>/dev/null | cut -d= -f2-)"
METAL_EMBED_URL="$(grep -E '^METAL_EMBED_URL=' "$HERE/.env" 2>/dev/null | cut -d= -f2-)"
[ -n "$OLLAMA_URL" ] || { echo "!! OLLAMA_URL not set in .env (mini-1 LAN IP)"; exit 1; }
[ -n "$METAL_EMBED_URL" ] || { echo "!! METAL_EMBED_URL not set in .env (mini-2 LAN IP)"; exit 1; }
if curl -sf --max-time 5 "${OLLAMA_URL%/}/api/tags" >/dev/null; then echo "    SLM (mini 1) $OLLAMA_URL ✓"; else echo "    !! SLM host unreachable at $OLLAMA_URL — open the firewall / check the IP"; fi
if curl -sf --max-time 5 "${METAL_EMBED_URL%/}/healthz" >/dev/null; then echo "    BGE (mini 2) $METAL_EMBED_URL ✓"; else echo "    !! BGE host unreachable at $METAL_EMBED_URL — queries will fall back to slow CPU (needs local model_cache)"; fi

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

echo "==> [5/5] start the query-only stack (SLM+BGE are offloaded to mini 1/2)"
$COMPOSE up -d postgres pgbouncer redis-broker redis-cache      # infra
$COMPOSE up -d --no-deps inference api                          # app (--no-deps: no local ollama)

echo ""
echo "==> waiting for warm-load, then readiness check:"
sleep 8
curl -s http://localhost:8000/readyz | { command -v jq >/dev/null && jq || cat; }
echo ""
echo "==> demo up. Try:"
echo "    curl -s -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' \\"
echo "         -d '{\"query\": \"your natural-language question\"}' | jq"
