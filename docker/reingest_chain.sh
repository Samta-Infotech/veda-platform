#!/bin/bash
# Dev helper: wait for the model download to finish, then re-run ingestion with the
# Ollama-URL fix + models present. Writes progress to /tmp/reingest_chain.log.
set -uo pipefail
cd /Users/ekesel/samta/veda-platform

echo "[chain] waiting for model download (veda_model_dl2)..."
docker wait veda_model_dl2 >/dev/null 2>&1 || true
echo "[chain] download container exited; model cache size:"
docker run --rm --entrypoint du -v veda-platform_model_cache:/models veda-platform-inference -sh /models 2>/dev/null | tail -1

NET=$(docker network ls --format '{{.Name}}' | grep -E 'veda.*net' | head -1)
docker rm -f veda_ingest_run >/dev/null 2>&1 || true

echo "[chain] launching re-ingestion (Ollama reachable, models cached)..."
docker run --rm --name veda_ingest_run \
  --entrypoint python \
  --network "$NET" \
  --add-host host.docker.internal:host-gateway \
  --env-file .env \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$PWD":/app -w /app/veda_core \
  -v veda-platform_model_cache:/models \
  veda-platform-inference -u -c "import main; r=main.run_ingestion(verbose=False); print('INGEST_DONE', r.get('source_id'))"
echo "[chain] re-ingestion finished with exit $?"
