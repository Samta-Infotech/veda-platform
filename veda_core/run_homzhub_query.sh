#!/bin/bash
# Query the ingested homzhub source through the hybrid engine.
# Usage: ./run_homzhub_query.sh "how many users are there" [--debug]
set -euo pipefail
cd "$(dirname "$0")"

export VEDA_SOURCE_ID=1
export VEDA_SOURCE_HOST=homzhub-dev-do-user-7308632-0.i.db.ondigitalocean.com
export VEDA_SOURCE_PORT=25060
export VEDA_SOURCE_DBNAME=homzhub_prod
export VEDA_SOURCE_USER=homzhub_dev
export VEDA_SOURCE_PASSWORD=''
export VEDA_SOURCE_SCHEMA=homzhub
export VEDA_EXCLUDE_TABLES='[]'

export VEDA_INTERNAL_HOST=localhost
export VEDA_INTERNAL_PORT=15432
export VEDA_INTERNAL_DBNAME=veda
export VEDA_INTERNAL_USER=veda
export VEDA_INTERNAL_PASSWORD=change-me

export OLLAMA_URL=http://localhost:11434
export SLM_OLLAMA_BASE_URL=http://localhost:11434
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$(dirname "$(pwd)")"

source ../.venv/bin/activate
python3 main.py --query "$1" "${@:2}"
