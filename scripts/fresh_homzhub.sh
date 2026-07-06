#!/bin/bash
# FRESH homzhub setup — self-contained in the veda-platform repo (no veda-poc dependency).
# Wipes ALL derived state (stale files + engine store + Django substrate + verified cache),
# then runs the FULL L0 ingestion on the host using veda-platform's own .venv (Apple MPS /
# CUDA auto) + native Ollama (qwen). Writes into the platform's veda_engine (container
# Postgres @ localhost:15432) so the running api/inference containers serve queries after.
#
# Prereqs (one-time): .venv built from requirements/host-ingest.txt; the platform containers
# up (docker compose up -d); the homzhub Source registered in Django; `ollama serve` running.
#
# Usage:   bash scripts/fresh_homzhub.sh <SOURCE_ID> [TENANT]
# Logs to: logs/ingest_<SOURCE_ID>_<timestamp>.log  (also streamed to stdout via tee)

SOURCE_ID="${1:?usage: fresh_homzhub.sh <SOURCE_ID> [TENANT]}"
TENANT="${2:-default}"

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PY="$REPO/.venv/bin/python"     # veda-platform's OWN venv — no veda-poc
[ -x "$PY" ] || { echo "ERROR: $PY missing. Build it: python3 -m venv .venv && .venv/bin/pip install -r requirements/host-ingest.txt"; exit 1; }

mkdir -p logs
LOG="logs/ingest_${SOURCE_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "==> logging to $LOG"

# --- source connection from the Django Source row (config-driven, no hardcoding) ---
echo "==> [0/6] Resolve source connection from Django Source row"
CONN=$(docker compose exec -T api python -c "
import os,django,json; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from apps.sources.models import Source; c=Source.objects.get(pk=$SOURCE_ID).connection()
print(json.dumps({k:str(v) for k,v in c.items()}))")
eval "$($PY -c "
import json,shlex; c=json.loads('''$CONN''')
for k,v in [('VEDA_SOURCE_HOST',c['host']),('VEDA_SOURCE_PORT',c['port']),('VEDA_SOURCE_DBNAME',c['dbname']),('VEDA_SOURCE_USER',c['user']),('VEDA_SOURCE_PASSWORD',c['password'])]:
    print('export %s=%s'%(k,shlex.quote(v)))")"
echo "    source → $VEDA_SOURCE_USER@$VEDA_SOURCE_HOST:$VEDA_SOURCE_PORT/$VEDA_SOURCE_DBNAME"

export VEDA_INTERNAL_HOST=localhost VEDA_INTERNAL_PORT=15432 VEDA_INTERNAL_DBNAME=veda_engine
export VEDA_INTERNAL_USER=veda VEDA_INTERNAL_PASSWORD=change-me
export OLLAMA_URL=http://localhost:11434 SLM_MODEL_NAME=qwen2.5-coder:7b
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VEDA_DEVICE VEDA_RESUME 2>/dev/null || true

echo "==> [1/6] Verify source reachable + device"
$PY - <<PY
import os,sys,psycopg2; sys.path.insert(0,"veda_core")
from veda_core import config
print("    device:", config.resolve_device(), "| BGE_DEVICE:", config.BGE_DEVICE)
c=psycopg2.connect(host=os.environ["VEDA_SOURCE_HOST"],port=int(os.environ["VEDA_SOURCE_PORT"]),
    dbname=os.environ["VEDA_SOURCE_DBNAME"],user=os.environ["VEDA_SOURCE_USER"],password=os.environ["VEDA_SOURCE_PASSWORD"])
cur=c.cursor();cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
print("    source public tables:", cur.fetchone()[0]); c.close()
PY

echo "==> [2/6] DELETE all stale/derived files (fresh)"
rm -f veda_core/data/*.json 2>/dev/null || true
rm -f veda_core/ingestion/training_pairs.jsonl 2>/dev/null || true
rm -rf veda_core/ingestion/client_bge veda_core/ingestion/client_minilm 2>/dev/null || true
rm -f veda_core/schema/*.pkl 2>/dev/null || true
rm -f veda_core/logs/*.jsonl 2>/dev/null || true
echo "    removed data/*.json, training_pairs.jsonl, client_bge/client_minilm, schema/*.pkl, logs/*.jsonl"

echo "==> [3/6] Fresh internal store (veda_engine) + Django substrate + verified cache"
docker compose exec -T postgres psql -U veda -d postgres -c "DROP DATABASE IF EXISTS veda_engine;" >/dev/null
docker compose exec -T postgres psql -U veda -d postgres -c "CREATE DATABASE veda_engine OWNER veda;" >/dev/null
docker compose exec -T postgres psql -U veda -d veda_engine -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
docker compose exec -T postgres psql -U veda -d veda -c "
TRUNCATE substrate_fkedge, substrate_schemacolumn, substrate_schematable, substrate_columnvaluesample,
  substrate_synonym, substrate_glossaryentry, substrate_graphedge, substrate_graphnode,
  substrate_graphartifact, substrate_smcolumn, substrate_smtable, substrate_smretrievaldoc,
  substrate_smsynonym, substrate_smconcept, substrate_substrateversion, substrate_verifiedquerycache
  RESTART IDENTITY CASCADE; TRUNCATE column_embeddings_bge;" >/dev/null 2>&1 || true
echo "    veda_engine recreated + pgvector; Django substrate + cache truncated."

echo "==> [4/6] FULL ingestion (host, MPS encoders + Ollama qwen) — the long step"
( cd veda_core && $PY -u -c "import main; main.run_ingestion(verbose=True)" )

echo "==> [5/6] Warm: sync Django substrate + publish sm + rehydrate + flip ready"
docker compose exec -T -e PYTHONPATH=/app -w /app api python -c "
import os,django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from veda_core.context import RequestContext, set_context
from storage_adapters import writer
from apps.sources.models import Source, SourceStatus
set_context(RequestContext(source_id=$SOURCE_ID, tenant='$TENANT'))
print('    warm:', writer.warm())
Source.objects.filter(pk=$SOURCE_ID).update(ready=True, status=SourceStatus.READY)
"
docker compose exec -T inference sh -c "curl -s -X POST http://localhost:8001/v1/rehydrate -H 'Content-Type: application/json' -d '{\"source_id\":$SOURCE_ID,\"tenant\":\"$TENANT\",\"scope\":\"all\"}'" >/dev/null 2>&1 || true

echo "==> [6/6] DONE (log: $LOG). Query:"
echo "    curl -s -X POST http://localhost:8080/api/v1/query -H 'Content-Type: application/json' -d '{\"query\":\"how many customers are there\",\"source_id\":$SOURCE_ID}'"
