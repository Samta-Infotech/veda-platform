#!/bin/bash
# Fresh-setup a NEW source DB end-to-end (wipe all derived state, re-point, re-ingest-ready).
# Run from the repo root:  bash scripts/onboard_source.sh
#
# Fill these in for your source (must be reachable FROM the containers — the Mac host is
# reachable as host.docker.internal). This does NOT touch your source data; it only reads it.
SRC_HOST="${SRC_HOST:-host.docker.internal}"
SRC_PORT="${SRC_PORT:-5432}"
SRC_DB="${SRC_DB:-homzhub}"
SRC_USER="${SRC_USER:-postgres}"
SRC_PASS="${SRC_PASS:-postgres}"
SRC_NAME="${SRC_NAME:-homzhub}"          # display name for the Django Source row
TENANT="${TENANT:-default}"

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/6] Point the engine at the new source (.env)"
python3 - "$SRC_HOST" "$SRC_PORT" "$SRC_DB" "$SRC_USER" "$SRC_PASS" <<'PY'
import sys, re, io
host, port, db, user, pw = sys.argv[1:6]
env = open(".env").read().splitlines()
def setkv(lines, k, v):
    out, seen = [], False
    for ln in lines:
        if ln.startswith(k + "="):
            out.append(f"{k}={v}"); seen = True
        else:
            out.append(ln)
    if not seen: out.append(f"{k}={v}")
    return out
for k, v in [("VEDA_SOURCE_HOST", host), ("VEDA_SOURCE_PORT", port),
             ("VEDA_SOURCE_DBNAME", db), ("VEDA_SOURCE_USER", user),
             ("VEDA_SOURCE_PASSWORD", pw)]:
    env = setkv(env, k, v)
open(".env", "w").write("\n".join(env) + "\n")
print("   .env updated: VEDA_SOURCE_* ->", host, port, db, user)
PY

echo "==> [2/6] Verify the source is reachable from a container"
docker compose exec -T worker python -c "
import psycopg2
c=psycopg2.connect(host='$SRC_HOST',port=$SRC_PORT,dbname='$SRC_DB',user='$SRC_USER',password='$SRC_PASS')
cur=c.cursor(); cur.execute(\"SELECT count(*) FROM information_schema.tables WHERE table_schema='public'\")
print('   OK — reachable; public tables:', cur.fetchone()[0]); c.close()
"

echo "==> [3/6] Wipe the internal engine store (fresh embeddings/graph/fk)"
docker compose exec -T postgres psql -U veda -d postgres -c \
  "DROP DATABASE IF EXISTS veda_engine;" >/dev/null
docker compose exec -T postgres psql -U veda -d postgres -c \
  "CREATE DATABASE veda_engine OWNER veda;" >/dev/null
docker compose exec -T postgres psql -U veda -d veda_engine -c \
  "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null
echo "   veda_engine recreated + pgvector."

echo "==> [4/6] Wipe Django substrate + verified cache + derived files"
docker compose exec -T postgres psql -U veda -d veda -c "
TRUNCATE substrate_fkedge, substrate_schemacolumn, substrate_schematable,
  substrate_columnvaluesample, substrate_synonym, substrate_glossaryentry,
  substrate_graphedge, substrate_graphnode, substrate_graphartifact,
  substrate_smcolumn, substrate_smtable, substrate_smretrievaldoc,
  substrate_smsynonym, substrate_smconcept, substrate_substrateversion,
  substrate_verifiedquerycache RESTART IDENTITY CASCADE;
TRUNCATE column_embeddings_bge;" >/dev/null 2>&1 || echo "   (some tables absent — ok)"
# derived engine files (semantic model, relationship graph, file caches)
docker compose exec -T ingest-worker sh -c \
  "rm -f /app/veda_core/data/veda_semantic_model.json /app/veda_core/data/veda_relationship_graph.json \
         /app/veda_core/data/veda_verified_queries.json /app/veda_core/data/veda_semantic_checkpoint.json \
         /app/veda_core/data/veda_glossary.json 2>/dev/null" || true
echo "   Django substrate + caches + derived files cleared."

echo "==> [5/6] Recreate the Source row + recreate containers (pick up .env)"
docker compose exec -T api python -c "
import os,django; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from apps.sources.models import Source, Dialect, SourceStatus
Source.objects.exclude(name='$SRC_NAME').update(ready=False, status=SourceStatus.REGISTERED)
s,created=Source.objects.update_or_create(name='$SRC_NAME', defaults=dict(
   dialect=Dialect.POSTGRES, connector_type='relational',
   connection_secret_ref='env:VEDA_SOURCE_*', status=SourceStatus.REGISTERED, ready=False))
print('   Source id=%d name=%s created=%s' % (s.id, s.name, created))
open('/app/.source_id','w').write(str(s.id))
"
SOURCE_ID=$(docker compose exec -T api cat /app/.source_id | tr -d '[:space:]')
echo "   SOURCE_ID=$SOURCE_ID"
docker compose up -d --force-recreate worker ingest-worker inference api >/dev/null
echo "   containers recreated."

echo "==> [6/6] READY. Kick off ingestion (choose one):"
echo "   FAST structural-only (no LLM glossary, ~minutes):"
echo "     docker compose exec -T ingest-worker python -c \"import os,django;os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev');django.setup();from apps.ingestion.tasks import task_ingest_source;print(task_ingest_source(source_id=$SOURCE_ID,tenant='$TENANT',force=True,skip_llm=True))\""
echo "   FULL (with LLM glossary/synonyms, hours on CPU — prefer GPU/bare-metal, P8-A):"
echo "     docker compose exec -T ingest-worker python -c \"...task_ingest_source(source_id=$SOURCE_ID,tenant='$TENANT',force=True)\""
echo "   Then query:  curl -s -X POST http://localhost:8080/api/v1/query -H 'Content-Type: application/json' -d '{\"query\":\"how many customers are there\"}'"
