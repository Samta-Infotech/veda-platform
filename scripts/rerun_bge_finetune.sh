#!/bin/bash
# Rerun ONLY the BGE fine-tune (step 11) standalone, after the config fix — reuses the
# already-built substrate. Rebuilds the cheap inference_result (schema scan + semantic types)
# and reuses the existing synthetic training pairs. Runs bare-metal on the host.
# Usage:  bash scripts/rerun_bge_finetune.sh <SOURCE_ID>
SOURCE_ID="${1:?usage: rerun_bge_finetune.sh <SOURCE_ID>}"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Users/ekesel/samta/veda-poc/venv/bin/python

# source connection from the Django Source row (config-driven)
CONN=$(docker compose exec -T api python -c "
import os,django,json; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.dev'); django.setup()
from apps.sources.models import Source; c=Source.objects.get(pk=$SOURCE_ID).connection()
print(json.dumps({k:str(v) for k,v in c.items()}))
")
eval "$($PY -c "
import json,shlex; c=json.loads('''$CONN''')
for k,v in [('VEDA_SOURCE_HOST',c['host']),('VEDA_SOURCE_PORT',c['port']),('VEDA_SOURCE_DBNAME',c['dbname']),('VEDA_SOURCE_USER',c['user']),('VEDA_SOURCE_PASSWORD',c['password'])]:
    print('export %s=%s'%(k,shlex.quote(v)))
")"
export VEDA_INTERNAL_HOST=localhost VEDA_INTERNAL_PORT=15432 VEDA_INTERNAL_DBNAME=veda_engine
export VEDA_INTERNAL_USER=veda VEDA_INTERNAL_PASSWORD=change-me
export OLLAMA_URL=http://localhost:11434 SLM_MODEL_NAME=qwen2.5-coder:7b
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE VEDA_DEVICE 2>/dev/null || true

( cd veda_core && $PY -u -c "
import time
from ingestion.schema_scanner import run_schema_scanner
from ingestion.semantic_type_inference import run_semantic_type_inference
from ingestion.auto_finetune import run_bge_finetune

t0=time.time()
print('[1/3] schema scan …', flush=True)
scan = run_schema_scanner(verbose=False)
print('[2/3] semantic type inference …', flush=True)
inf = run_semantic_type_inference(scan_result=scan, verbose=False)
# fine-tune reads fk edges off inference_result (same as main.py step 11)
inf._fk_edges = scan.fk_edges
print('[3/3] BGE fine-tune (the fixed step) …', flush=True)
res = run_bge_finetune(inf, verbose=True)
print('BGE_FINETUNE_RESULT:', getattr(res,'__dict__',res))
print('total %.1fs' % (time.time()-t0))
" )
