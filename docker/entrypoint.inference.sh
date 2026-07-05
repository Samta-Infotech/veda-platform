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

exec uvicorn inference.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers "${WORKERS}" \
    --timeout-keep-alive "${UVICORN_KEEPALIVE:-30}"
