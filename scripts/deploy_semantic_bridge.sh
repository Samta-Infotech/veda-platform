#!/usr/bin/env bash
# =============================================================================
# scripts/deploy_semantic_bridge.sh
# VEDA — Deploy the semantic bridge (docs/SEMANTIC_ENTITY_BRIDGE.md) to a running stack.
#
# WHAT IT DOES
#   1. (code)   Assumes the new/changed engine files are already on the host (git pull /
#               rsync). It then makes the workers pick them up:
#                 - default: restart inference + ingest-worker (works when the repo is
#                   bind-mounted into the containers, as in docker-compose.yml).
#                 - REBUILD=1: rebuild those images first (use when code is baked in).
#   2. (wait)   Waits for ingest-worker to be running.
#   3. (backfill) Runs scripts/backfill_semantic_bridge.py INSIDE ingest-worker:
#                 EMBED (structured column/table embeddings — the missing piece),
#                 VALUES (Tier B value index), RELINK (recreate bridge edges from
#                 existing chunks — no source re-ingest), VERIFY (counts + safety check).
#
# The backfill is IDEMPOTENT and re-runnable. It does NOT touch customer source DBs.
#
# USAGE (from the repo root on the deploy host):
#   ./scripts/deploy_semantic_bridge.sh                 # restart + full backfill
#   REBUILD=1 ./scripts/deploy_semantic_bridge.sh       # rebuild images first
#   PHASE=verify ./scripts/deploy_semantic_bridge.sh    # just re-run verification
#   SOURCES=2,3 ./scripts/deploy_semantic_bridge.sh     # limit backfill to source ids
#   SKIP_RESTART=1 ./scripts/deploy_semantic_bridge.sh  # code already live; backfill only
#
# ENV KNOBS
#   COMPOSE       docker compose invocation      (default: "docker compose")
#   COMPOSE_FILE  optional -f file               (default: repo docker-compose.yml)
#   WORKER        ingest worker service name      (default: ingest-worker)
#   INFER         inference service name          (default: inference)
#   PHASE         all|embed|values|relink|verify  (default: all)
#   SOURCES       comma-separated source ids       (default: all discovered)
#   TENANT        tenant id                        (default: default)
#   REBUILD=1     rebuild worker/inference images before restart
#   SKIP_RESTART=1  skip the restart/rebuild step (backfill only)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE="${COMPOSE:-docker compose}"
COMPOSE_FILE_ARG=""
[ -n "${COMPOSE_FILE:-}" ] && COMPOSE_FILE_ARG="-f ${COMPOSE_FILE}"
DC="${COMPOSE} ${COMPOSE_FILE_ARG}"

WORKER="${WORKER:-ingest-worker}"
INFER="${INFER:-inference}"
PHASE="${PHASE:-all}"
SOURCES="${SOURCES:-}"
TENANT="${TENANT:-default}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- 1. code refresh -------------------------------------------------------
if [ "${SKIP_RESTART:-0}" = "1" ]; then
  log "SKIP_RESTART=1 — assuming new code is already live in the workers"
elif [ "${REBUILD:-0}" = "1" ]; then
  log "Rebuilding images for ${INFER} + ${WORKER} (REBUILD=1)"
  ${DC} build "${INFER}" "${WORKER}"
  log "Recreating ${INFER} + ${WORKER}"
  ${DC} up -d --no-deps "${INFER}" "${WORKER}"
else
  log "Restarting ${INFER} + ${WORKER} to pick up bind-mounted code"
  ${DC} up -d --no-deps "${INFER}" "${WORKER}"
  ${DC} restart "${INFER}" "${WORKER}"
fi

# --- 2. wait for worker ----------------------------------------------------
log "Waiting for ${WORKER} to be running"
for i in $(seq 1 60); do
  state="$(${DC} ps --format '{{.Name}} {{.State}}' 2>/dev/null | grep -E "${WORKER}" | head -1 || true)"
  if echo "$state" | grep -qi 'running\|up'; then
    echo "  ${WORKER} is up ($state)"; break
  fi
  sleep 3
  [ "$i" = "60" ] && { echo "  ${WORKER} did not come up in time"; exit 1; }
done

# --- 3. backfill -----------------------------------------------------------
# Run with the ENGINE config on sys.path (the script inserts /app/veda_core itself; we
# also cd there so relative artifact paths resolve like a real ingest).
SRC_ARG=""
[ -n "${SOURCES}" ] && SRC_ARG="--sources ${SOURCES}"

log "Running semantic-bridge backfill inside ${WORKER} (phase=${PHASE}, tenant=${TENANT}${SOURCES:+, sources=${SOURCES}})"
${DC} exec -T "${WORKER}" sh -lc \
  "cd /app/veda_core && python /app/scripts/backfill_semantic_bridge.py --phase ${PHASE} --tenant ${TENANT} ${SRC_ARG}"

log "Done. Re-run any time — the backfill is idempotent. Use PHASE=verify to re-check."
