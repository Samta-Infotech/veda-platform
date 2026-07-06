#!/bin/bash
# FULL bare-metal ingestion on the Mac host for a REGISTERED source (by source_id).
# The source's DB connection is read from its Django Source row — no per-source env/code edits.
# Uses MPS (24GB unified memory) for the encoders + native host Ollama (Metal) for qwen.
# Writes into the platform's veda_engine (container Postgres @ localhost:15432) so the running
# api/inference containers serve queries from it.
#
# Usage:  bash scripts/ingest_baremetal.sh <SOURCE_ID> [TENANT]
SOURCE_ID="${1:?usage: ingest_baremetal.sh <SOURCE_ID> [TENANT]}"
TENANT="${2:-default}"

set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Users/ekesel/samta/veda-poc/venv/bin/python   # host venv: torch 2.12 + MPS + ML deps

echo "==> [1/5] Read source connection from the Django Source row (config-driven)"
CONN_JSON=$(docker compose exec -T api python -c "
import os,django,json; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from apps.sources.models import Source
s=Source.objects.get(pk=$SOURCE_ID); c=s.connection()
print(json.dumps({'name':s.name, **{k:str(v) for k,v in c.items()}}))
")
eval "$($PY -c "
import json,sys,shlex
c=json.loads('''$CONN_JSON''')
for k,v in [('VEDA_SOURCE_HOST',c['host']),('VEDA_SOURCE_PORT',c['port']),('VEDA_SOURCE_DBNAME',c['dbname']),('VEDA_SOURCE_USER',c['user']),('VEDA_SOURCE_PASSWORD',c['password']),('SRC_NAME',c['name'])]:
    print('export %s=%s' % (k, shlex.quote(v)))
")"
echo "   source '$SRC_NAME' → $VEDA_SOURCE_USER@$VEDA_SOURCE_HOST:$VEDA_SOURCE_PORT/$VEDA_SOURCE_DBNAME"

# internal store (host-exposed container Postgres) + host Ollama (Metal) + MPS auto
export VEDA_INTERNAL_HOST=localhost VEDA_INTERNAL_PORT=15432 VEDA_INTERNAL_DBNAME=veda_engine
export VEDA_INTERNAL_USER=veda VEDA_INTERNAL_PASSWORD=change-me
export OLLAMA_URL=http://localhost:11434 SLM_MODEL_NAME=qwen2.5-coder:7b
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VEDA_DEVICE 2>/dev/null || true

echo "==> [2/5] Verify source reachable from host + device=mps"
$PY - <<PY
import psycopg2, sys, os
sys.path.insert(0, "veda_core")
from veda_core import config
print("   device:", config.resolve_device(), "| BGE_DEVICE:", config.BGE_DEVICE)
c=psycopg2.connect(host=os.environ["VEDA_SOURCE_HOST"], port=int(os.environ["VEDA_SOURCE_PORT"]),
                   dbname=os.environ["VEDA_SOURCE_DBNAME"], user=os.environ["VEDA_SOURCE_USER"],
                   password=os.environ["VEDA_SOURCE_PASSWORD"])
cur=c.cursor(); cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
print("   source reachable; public tables:", cur.fetchone()[0]); c.close()
PY

echo "==> [3/5] Fresh internal store + Django substrate + derived files"
docker compose exec -T postgres psql -U veda -d postgres -c "DROP DATABASE IF EXISTS veda_engine;" >/dev/null
docker compose exec -T postgres psql -U veda -d postgres -c "CREATE DATABASE veda_engine OWNER veda;" >/dev/null
docker compose exec -T postgres psql -U veda -d veda_engine -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
docker compose exec -T postgres psql -U veda -d veda -c "
TRUNCATE substrate_fkedge, substrate_schemacolumn, substrate_schematable, substrate_columnvaluesample,
  substrate_synonym, substrate_glossaryentry, substrate_graphedge, substrate_graphnode,
  substrate_graphartifact, substrate_smcolumn, substrate_smtable, substrate_smretrievaldoc,
  substrate_smsynonym, substrate_smconcept, substrate_substrateversion, substrate_verifiedquerycache
  RESTART IDENTITY CASCADE; TRUNCATE column_embeddings_bge;" >/dev/null 2>&1 || true
rm -f veda_core/data/veda_semantic_model.json veda_core/data/veda_relationship_graph.json \
      veda_core/data/veda_verified_queries.json veda_core/data/veda_semantic_checkpoint.json \
      veda_core/data/veda_glossary.json 2>/dev/null || true
echo "   wiped."

echo "==> [4/5] FULL ingestion — bare-metal MPS encoders + host-Ollama qwen (the long one)"
( cd veda_core && $PY -u -c "import main; main.run_ingestion(verbose=True)" )

echo "==> [5/5] Warm (sync Django substrate + publish sm) + flip ready + reload inference"
docker compose exec -T -e PYTHONPATH=/app -w /app api python -c "
import os,django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from veda_core.context import RequestContext, set_context
from storage_adapters import writer
from apps.sources.models import Source, SourceStatus
set_context(RequestContext(source_id=$SOURCE_ID, tenant='$TENANT'))
print('   warm:', writer.warm())
Source.objects.filter(pk=$SOURCE_ID).update(ready=True, status=SourceStatus.READY)
"
docker compose exec -T inference sh -c "curl -s -X POST http://localhost:8001/v1/rehydrate -H 'Content-Type: application/json' -d '{\"source_id\":$SOURCE_ID,\"tenant\":\"$TENANT\",\"scope\":\"all\"}'" >/dev/null 2>&1 || true
echo "==> DONE. Query:  curl -s -X POST http://localhost:8080/api/v1/query -H 'Content-Type: application/json' -d '{\"query\":\"how many customers are there\"}'"
