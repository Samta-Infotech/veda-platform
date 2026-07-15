#!/bin/bash
# entrypoint for the inference image (ASGI query/retrieval service, migration_plan.md
# §8.1). Must be chmod +x (already set in Dockerfile.inference).
set -euo pipefail

# WORKERS is deliberately left as a plain env var, not a computed default: §8.1 requires
# sizing this from a MEASURED per-worker RSS on the target hardware
# (workers_per_replica = floor((replica_RAM_GB * safety_fraction - overhead) / PER_WORKER_RSS_GB)),
# not from CPU count. Do not default this to `nproc` — that ignores the model-RAM ceiling
# that actually bounds this service and risks an OOM on first load.
WORKERS="${WORKERS:?set WORKERS explicitly — sized from measured per-worker RSS, see migration_plan.md §8.1}"


# reload with a warning rather than pass an invalid flag combo if it isn't.
RELOAD_ARGS=()
if [ "${DEV_AUTORELOAD:-0}" = "1" ]; then
    if [ "${WORKERS}" = "1" ]; then
        # --reload-dir /app (not just the default cwd, /app/veda_core below):
        # uvicorn --reload with no --reload-dir only watches the CURRENT
        # WORKING DIRECTORY, which this service sets to /app/veda_core
        # (engine cwd-relative paths). Edits under sibling directories at the
        # same PYTHONPATH root — inference/ (this image's own routes),
        # apps/, chatbot/ — were silently never picked up (confirmed: a fix
        # to inference/routes/hybrid.py stayed dead until a manual restart,
        # 2026-07). Watching the whole /app tree covers all of them.
        RELOAD_ARGS=(--reload --reload-dir /app)
    else
        echo "entrypoint.inference.sh: DEV_AUTORELOAD=1 but WORKERS=${WORKERS} (>1)" \
             "— uvicorn --reload only supports a single worker, skipping autoreload" >&2
    fi
fi

exec uvicorn inference.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers "${WORKERS}" \
    --timeout-keep-alive "${UVICORN_KEEPALIVE:-30}" \
    "${RELOAD_ARGS[@]}"
