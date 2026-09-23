#!/usr/bin/env bash
# =============================================================================
# scripts/env_endpoints.sh — switch the SLM + embedding endpoints between the
#   office Metal hosts and this laptop's own, in one command.
#
# WHY THIS EXISTS
#   Three keys in .env decide where every LLM call and every embedding goes. Two of
#   them point at machines that only exist on the office LAN (192.168.1.x). Carry the
#   laptop home and an ingestion mid-run does not fail fast — the SLM stage retries,
#   trips its cooldown, and aborts hours later; the embedder falls back to CPU silently.
#   Editing .env by hand is also only half the job: container env is bound at CREATE
#   time, so a plain `restart` keeps the old values and nothing says so (that is how
#   METAL_EMBED_URL sat empty in ingest-worker all of 2026-09-22 while .env had it set).
#
#   This does both halves, then PROVES the result by probing from inside a container.
#
# PROFILES
#   office   OLLAMA_URL        = http://192.168.1.35:11500     (Metal box, GPU)
#            METAL_EMBED_URL   = http://192.168.1.39:11435     (BGE-M3 server, mps)
#            VEDA_SLM_CHAT_URL = http://192.168.1.35:11500/api/chat
#
#   home     OLLAMA_URL        = http://host.docker.internal:11434   (this Mac's own
#              ollama — native, so it IS Metal-GPU accelerated; only the embeddings lose
#              their GPU, not the SLM)
#            METAL_EMBED_URL   = (blank) → BGE-M3 loads in-process on CPU. Correct, and
#              roughly 4x slower per vector (measured 0.37s remote vs ~1.5s/node local).
#            VEDA_SLM_CHAT_URL = http://host.docker.internal:11434/api/chat
#
# USAGE
#   bash scripts/env_endpoints.sh show      # what is set, and what actually answers
#   bash scripts/env_endpoints.sh home
#   bash scripts/env_endpoints.sh office
#   NO_RECREATE=1 bash scripts/env_endpoints.sh home    # edit .env only, recreate later
#
# Always backs .env up first. Never touches postgres/pgbouncer/redis.
# =============================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMPOSE="${COMPOSE:-docker compose -f docker-compose.yml}"
APP_SERVICES="${APP_SERVICES:-ingest-worker inference api worker beat}"
NO_RECREATE="${NO_RECREATE:-0}"

OFFICE_OLLAMA="http://192.168.1.35:11500"
OFFICE_EMBED="http://192.168.1.39:11435"
HOME_OLLAMA="http://host.docker.internal:11434"
HOME_EMBED=""            # blank = in-process CPU BGE-M3 (m3_encoder falls back cleanly)

ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[1;31m✗\033[0m %s\n' "$*"; }
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

set_key() {  # $1 = key, $2 = value. Rewrites in place, appends when absent.
  python3 - "$1" "$2" <<'PY'
import sys
key, val = sys.argv[1], sys.argv[2]
lines = open(".env").read().splitlines()
out, seen = [], False
for ln in lines:
    if ln.startswith(key + "="):
        out.append(f"{key}={val}"); seen = True
    else:
        out.append(ln)
if not seen:
    out.append(f"{key}={val}")
open(".env", "w").write("\n".join(out) + "\n")
PY
}

probe() {  # report what each endpoint ACTUALLY answers, from inside a container
  local svc="$1"
  $COMPOSE ps --format '{{.Service}} {{.State}}' 2>/dev/null | grep -qE "^$svc (running|up)" || {
    warn "$svc is not running — cannot probe from inside it"; return 0; }

  # NO `</dev/null` here: the heredoc IS this command's stdin, and a later redirect wins —
  # python would read an empty program and print nothing, which is exactly what it did.
  $COMPOSE exec -T "$svc" python - <<'PY' || warn "probe failed"
import json, os, urllib.request

def get(url, timeout=6):
    return json.load(urllib.request.urlopen(url, timeout=timeout))

ollama = (os.environ.get("OLLAMA_URL") or "").rstrip("/")
embed  = (os.environ.get("METAL_EMBED_URL") or "").strip().rstrip("/")
chat   = os.environ.get("VEDA_SLM_CHAT_URL") or "(unset)"
model  = os.environ.get("SLM_MODEL_NAME") or "(unset)"

print(f"    OLLAMA_URL        {ollama or '(unset)'}")
if ollama:
    try:
        names = [m.get("name") for m in get(ollama + "/api/tags").get("models", [])]
        mark = "OK " if model in names else "MISSING "
        print(f"      -> {mark}pinned model {model!r}; serves {', '.join(names)}")
    except Exception as e:
        print(f"      -> UNREACHABLE ({type(e).__name__}) — every SLM call will fail")

print(f"    METAL_EMBED_URL   {embed or '(blank — in-process CPU BGE-M3)'}")
if embed:
    try:
        req = urllib.request.Request(embed + "/encode_dense",
                                     data=json.dumps({"texts": ["probe"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=15))
        v = (r.get("vecs") or r.get("vectors") or r.get("embeddings") or [[]])[0]
        print(f"      -> OK {len(v)}-dim")
    except Exception as e:
        # A SET-but-dead offload is the dangerous state: it is not a clean fallback.
        print(f"      -> UNREACHABLE ({type(e).__name__}) — BLANK IT or embeddings stall")

print(f"    VEDA_SLM_CHAT_URL {chat}")
PY
}

profile="${1:-show}"

case "$profile" in
  show)
    log "Current endpoints (.env)"
    grep -E '^(OLLAMA_URL|METAL_EMBED_URL|VEDA_SLM_CHAT_URL|SLM_MODEL_NAME)=' .env | sed 's/^/    /'
    log "What actually answers, probed from inside ingest-worker"
    probe ingest-worker
    exit 0 ;;
  office) ollama="$OFFICE_OLLAMA"; embed="$OFFICE_EMBED" ;;
  home)   ollama="$HOME_OLLAMA";   embed="$HOME_EMBED" ;;
  *) echo "usage: $0 [show|home|office]" >&2; exit 2 ;;
esac

backup=".env.bak-$(date +%Y%m%d-%H%M%S)"
cp -p .env "$backup"
log "Switching to the '$profile' profile (backup: $backup)"

set_key OLLAMA_URL        "$ollama"
set_key METAL_EMBED_URL   "$embed"
set_key VEDA_SLM_CHAT_URL "${ollama}/api/chat"
ok "OLLAMA_URL        = $ollama"
ok "METAL_EMBED_URL   = ${embed:-(blank — CPU embeddings)}"
ok "VEDA_SLM_CHAT_URL = ${ollama}/api/chat"

if [ "$NO_RECREATE" = "1" ]; then
  warn "NO_RECREATE=1 — .env edited but the containers still hold the OLD values."
  warn "apply with:  $COMPOSE up -d --force-recreate $APP_SERVICES"
  exit 0
fi

# --force-recreate, not plain `up -d`: compose does not always notice an env_file edit
# and will report "Running" while leaving the stale env in place (seen 2026-09-22).
log "Recreating $APP_SERVICES (env is bound at container CREATE time)"
$COMPOSE up -d --force-recreate $APP_SERVICES >/dev/null 2>&1 || {
  bad "recreate failed — run it yourself: $COMPOSE up -d --force-recreate $APP_SERVICES"; exit 1; }
sleep 6
ok "containers recreated"

log "Verifying from inside ingest-worker"
probe ingest-worker

if [ "$profile" = "home" ]; then
  log "Note for the home profile"
  printf '    %s\n' \
    "This Mac's own ollama must be running (\`ollama serve\`, or the menu-bar app) —" \
    "it is native, so the SLM keeps its Metal GPU. Only embeddings move to CPU." \
    "Embeddings are ~4x slower there (0.37s remote vs ~1.5s/node local), which shows" \
    "up in L4 (graph_embed + biencoder), not in the L3 semantic layer."
fi
