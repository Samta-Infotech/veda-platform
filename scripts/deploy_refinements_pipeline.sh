#!/usr/bin/env bash
# =============================================================================
# scripts/deploy_refinements_pipeline.sh
# VEDA — deploy branch `feat/refinements-pipeline` onto a running stack and do
#        exactly the data work its changes require. NOT a blanket re-ingest.
#
# WHY A SCRIPT AND NOT A RUNBOOK
#   Three of this branch's changes silently invalidate data that is already in the
#   deployment, and all three fail QUIETLY — a missing artifact is treated as an
#   empty artifact, a stale sparse vector just scores badly. Nothing raises. Each
#   check below exists because the failure it catches is invisible at runtime:
#
#   1. ARTIFACTS ARE PER-SOURCE NOW, WITH NO FLAT FALLBACK.
#      config.resolve_source_artifact() (M1 close-out, 2026-09-15) resolves ONLY
#      `<artifact_root>/<tenant>/<source_id>/<name>` — the old "scoped if it exists,
#      else the flat data/<name>" contract is gone. A source whose artifacts still
#      sit at flat paths loses its semantic model, relationship graph, join paths,
#      enrichment index, unified graph and HNSW settings, and answers anyway.
#      => preflight refuses to finish while a relational source has no scoped model.
#
#   2. DOC-CHUNK SPARSE VECTORS ARE CASE-NORMALISED NOW.
#      ingestion/chunk_embedder.py lowercases before BGE-M3 sparse encoding and
#      query/rag_layer.py lowercases the query to match (both 2026-09-16). Chunks
#      embedded before that hold case-sensitive weights; after this deploy the query
#      side no longer matches them. There is no backfill for it — the chunks have to
#      be re-embedded, i.e. the DOCUMENT sources have to be re-ingested.
#      => the `reingest` phase targets exactly those sources and nothing else.
#
#   3. ROUTING CARDS ARE A NEW L5 STAGE.
#      Sources ingested before it have no card. This one does NOT need a re-ingest:
#      scripts/backfill_routing_cards.py rebuilds the card as a pure transform of
#      artifacts that already exist.
#      => the `backfill` phase runs it for any source missing a card.
#
#   Everything else this branch changes is code + one migration + rebuilt images.
#
# WHAT IT DOES (phases, in order — select with PHASES=)
#   preflight  read-only. Branch/clone identity, .env keys, Postgres major version vs
#              the compose pin, ingest-worker presence, unapplied migrations, and a
#              per-source inventory of scoped artifacts / routing cards / stale chunks.
#   build      docker compose build (REQUIRED: requirements/{inference,host-ingest}.txt
#              gained mysql-connector-python — a restart will not install it).
#   migrate    manage.py migrate — substrate 0009 adds VerifiedQueryCache.substrate_version
#              and rekeys the cache. NOTE: DEPLOYMENT.md's one-shot `release` service does
#              NOT exist in either compose file (B6 outstanding), so this runs in `api`.
#   up         docker compose up -d  — recreate, never `restart`: env is read at container
#              CREATE time, so `restart` keeps the old .env while looking healthy.
#   backfill   routing cards for any source missing one (idempotent, re-runnable).
#   reingest   re-ingest the document sources whose chunks predate the sparse fix, and
#              (only with REINGEST_MISSING_ARTIFACTS=1) sources with no scoped artifacts.
#   verify     migrations applied, /readyz all-ok, inventory clean, per-source semantic
#              models describe their OWN tables.
#
# WHAT IT DELIBERATELY DOES NOT DO
#   No git commit / push / checkout / pull / stash / reset — this repo's CLAUDE.md
#   forbids it and the containers execute the working tree, so a stray checkout would
#   change what is live. When the branch is wrong the script prints the command and
#   stops; you run it.
#
# USAGE (from the repo root, on the deploy host)
#   ./scripts/deploy_refinements_pipeline.sh --check          # preflight only, changes nothing
#   ./scripts/deploy_refinements_pipeline.sh                  # full deploy, prompts before ingesting
#   ./scripts/deploy_refinements_pipeline.sh --prod --yes     # prod overlay, no prompts
#   ./scripts/deploy_refinements_pipeline.sh --dry-run        # print every mutating command
#   PHASES=backfill,verify ./scripts/deploy_refinements_pipeline.sh
#   SOURCES=3 ./scripts/deploy_refinements_pipeline.sh --phases reingest
#
# FLAGS / ENV KNOBS  (flags win; every flag has an env equivalent)
#   --check                 PHASES=preflight
#   --prod                  PROD=1            add docker-compose.prod.yml
#   --yes                   ASSUME_YES=1      no confirmation before re-ingesting
#   --dry-run               DRY_RUN=1         print mutating commands, run none
#   --skip-build            skip the build phase
#   --skip-ingest           skip the reingest phase
#   --phases a,b,c          PHASES=a,b,c
#   --sources 3,4           SOURCES=3,4       re-ingest exactly these, skip detection
#   --tenant default        TENANT=default
#   COMPOSE                 docker compose invocation      (default: "docker compose")
#   COMPOSE_FILE            single -f override             (default: the repo's files)
#   API/INFER/WORKER/PG     service names   (api / inference / ingest-worker / postgres)
#   EXPECT_BRANCH           branch this deploy is for      (feat/refinements-pipeline)
#   SPARSE_FIX_UTC          chunk-embedder case-fix cutoff (2026-09-16 00:00:00+00)
#   REINGEST_MISSING_ARTIFACTS=1   also re-ingest sources with no scoped artifacts
#   INGEST_TIMEOUT          seconds to wait per source     (default 7200)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- configuration
COMPOSE="${COMPOSE:-docker compose}"
PROD="${PROD:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
DRY_RUN="${DRY_RUN:-0}"
TENANT="${TENANT:-default}"
SOURCES="${SOURCES:-}"
PHASES="${PHASES:-preflight,build,migrate,up,backfill,reingest,verify}"
API="${API:-api}"; INFER="${INFER:-inference}"; WORKER="${WORKER:-ingest-worker}"; PG="${PG:-postgres}"
EXPECT_BRANCH="${EXPECT_BRANCH:-feat/refinements-pipeline}"
# The commit that made chunk_embedder lowercase text before sparse encoding. Chunks
# whose ingestion finished before this are case-mismatched against the (now lowercased)
# query. Override only if you cherry-picked that change onto a different timeline.
SPARSE_FIX_UTC="${SPARSE_FIX_UTC:-2026-09-16 00:00:00+00}"
REINGEST_MISSING_ARTIFACTS="${REINGEST_MISSING_ARTIFACTS:-0}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-7200}"

while [ $# -gt 0 ]; do
  case "$1" in
    --check)        PHASES="preflight" ;;
    --prod)         PROD=1 ;;
    --yes|-y)       ASSUME_YES=1 ;;
    --dry-run)      DRY_RUN=1 ;;
    --skip-build)   PHASES="$(echo "$PHASES" | sed 's/\bbuild\b,\?//')" ;;
    --skip-ingest)  PHASES="$(echo "$PHASES" | sed 's/\breingest\b,\?//')" ;;
    --phases)       PHASES="$2"; shift ;;
    --sources)      SOURCES="$2"; shift ;;
    --tenant)       TENANT="$2"; shift ;;
    -h|--help)      sed -n '2,/^# =\{20,\}$/p' "$0" | sed -n '2,$p' | tail -r | sed -n '2,$p' | tail -r; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

COMPOSE_FILES="-f docker-compose.yml"
[ "$PROD" = "1" ] && COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
[ -n "${COMPOSE_FILE:-}" ] && COMPOSE_FILES="-f ${COMPOSE_FILE}"
DC="${COMPOSE} ${COMPOSE_FILES}"

# .env keys with no code default that matter to this deployment. A missing one is a
# warning, not a blocker: some hosts legitimately set them in the environment instead.
REQUIRED_ENV_KEYS=(
  DJANGO_SETTINGS_MODULE POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD
  VEDA_INTERNAL_HOST VEDA_INTERNAL_PORT VEDA_INTERNAL_DBNAME VEDA_INTERNAL_USER
  SLM_MODEL_NAME SLM_TEMPERATURE
  MULTISOURCE_ROUTING_ENABLED MULTISOURCE_ROUTING_SHADOW
  REQUIRED_SOURCE_ESCALATION_ENABLED ROUTING_PERMISSION_DENY_GAP
)

# ---------------------------------------------------------------------- output
BLOCKERS=(); WARNINGS=(); ACTIONS=()
log()   { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[1;33m!\033[0m %s\n' "$*"; WARNINGS+=("$*"); }
bad()   { printf '  \033[1;31m✗\033[0m %s\n' "$*"; BLOCKERS+=("$*"); }
info()  { printf '    %s\n' "$*"; }
act()   { ACTIONS+=("$*"); }

run() {  # mutating command; honours --dry-run
  # %q, not %s: the echoed line is meant to be copy-pasteable, and a `sh -c "cd x && y"`
  # argument printed unquoted is a different command from the one actually run.
  local shown; shown="$(printf '%q ' "$@")"
  if [ "$DRY_RUN" = "1" ]; then printf '  \033[0;35m[dry-run]\033[0m %s\n' "$shown"; return 0; fi
  printf '  \033[0;90m$ %s\033[0m\n' "$shown"
  "$@"
}

confirm() {  # $1 = prompt; honours --yes and --dry-run
  [ "$ASSUME_YES" = "1" ] && return 0
  [ "$DRY_RUN" = "1" ] && return 0
  printf '\n  %s [y/N] ' "$1"
  read -r reply </dev/tty || return 1
  case "$reply" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

want_phase() { case ",${PHASES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# ------------------------------------------------------------------- container
# $DC is intentionally unquoted everywhere: it is a command prefix with arguments.
dc()      { $DC "$@"; }

# Both of these capture first and match second, deliberately. `docker compose ... | grep -q`
# is a race under `set -o pipefail`: grep exits on its first match, docker compose takes
# SIGPIPE, and the PIPELINE reports 141 even though the match succeeded — so the same check
# passed in preflight and failed in the reingest phase minutes later.
svc_up() {
  local ps_out; ps_out="$(dc ps --format '{{.Service}} {{.State}}' 2>/dev/null || true)"
  printf '%s\n' "$ps_out" | grep -E "^$1 (running|up)" >/dev/null 2>&1
}
svc_defined() {
  local svcs; svcs="$(dc config --services 2>/dev/null || true)"
  printf '%s\n' "$svcs" | grep -x "$1" >/dev/null 2>&1
}

psql_q() {  # $1 = database, $2 = SQL -> "|"-separated rows on stdout
  dc exec -T "$PG" sh -c 'exec psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$0" -tAF"|" -c "$1"' "$1" "$2" </dev/null
}

# Artifact root, resolved the way veda_core/config.py resolves it (VEDA_ARTIFACT_ROOT
# may be absolute; when relative it hangs off <VEDA_APP_DIR>/veda_core). Read from a
# container, not the host: in prod the code is baked in and there is no bind mount.
ART_ROOT=""
art_root() {
  [ -n "$ART_ROOT" ] && { printf '%s' "$ART_ROOT"; return 0; }
  # </dev/null on every helper exec: `docker compose exec` drains the caller's stdin,
  # which silently ate the rest of the inventory's row loop after the first source.
  ART_ROOT="$(dc exec -T "$1" sh -c '
      root="${VEDA_ARTIFACT_ROOT:-data}"
      case "$root" in /*) ;; *) root="${VEDA_APP_DIR:-/app}/veda_core/$root" ;; esac
      printf "%s" "$root"' </dev/null 2>/dev/null | tr -d '\r')"
  printf '%s' "$ART_ROOT"
}

has_artifact() {  # $1 = service, $2 = source id, $3 = artifact filename
  local root; root="$(art_root "$1")"
  [ -n "$root" ] || return 1
  dc exec -T "$1" sh -c 'test -f "$0/$1/$2/$3"' "$root" "$TENANT" "$2" "$3" </dev/null >/dev/null 2>&1
}

kind_of() {  # dialect -> engine source kind (apps/sources/models.py::_DIALECT_TO_ENGINE)
  case "$1" in
    postgres|mysql|sqlite|oracle|sqlserver|duckdb) echo relational ;;
    mongo|es|dynamo)                               echo nosql ;;
    filesystem|s3_docs)                            echo document ;;
    delta|parquet|csv_lake|iceberg)                echo datalake ;;
    *)                                             echo relational ;;   # the silent default
  esac
}

PG_DB=""            # Django database name, read from the postgres container's own env
STALE_DOC_SOURCES=""   # document sources needing a re-embed (set by inventory)
NOCARD_SOURCES=""      # sources with no routing card (set by inventory)
NOARTIFACT_SOURCES=""  # relational sources with no scoped model (set by inventory)
NEVER_INGESTED=""      # sources with no successful ingestion at all (set by inventory)

# =============================================================================
# preflight
# =============================================================================
preflight() {
  log "Preflight — read-only; nothing below changes the deployment"

  # --- the clone and the branch ------------------------------------------------
  local branch; branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  if [ "$branch" = "$EXPECT_BRANCH" ]; then
    ok "on branch $branch ($(git rev-parse --short HEAD))"
  else
    bad "on branch '$branch', expected '$EXPECT_BRANCH'"
    info "run it yourself — this script never moves your branch:"
    info "    git checkout $EXPECT_BRANCH"
  fi
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    warn "working tree has uncommitted changes — the containers execute the working tree, so they deploy too"
    git status --porcelain | head -10 | sed 's/^/      /'
  else
    ok "working tree clean"
  fi

  # The two-clone trap (CLAUDE.md): only one of them is what the containers mount.
  local cid mount
  cid="$(dc ps -q "$API" 2>/dev/null | head -1 || true)"
  if [ -n "$cid" ]; then
    mount="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/app"}}{{.Source}}{{end}}{{end}}' "$cid" 2>/dev/null || true)"
    if [ -n "$mount" ] && [ "$mount" != "$REPO_ROOT" ]; then
      bad "the $API container bind-mounts a DIFFERENT clone: $mount"
      info "you are in $REPO_ROOT — deploying from here changes nothing the containers run"
    elif [ -n "$mount" ]; then
      ok "containers bind-mount this clone ($mount)"
    else
      ok "no /app bind mount (code is baked into the image — a rebuild is mandatory)"
    fi
  else
    warn "$API is not running — skipped the clone-identity check"
  fi

  # --- .env --------------------------------------------------------------------
  if [ -f .env ]; then
    local missing=""
    for k in "${REQUIRED_ENV_KEYS[@]}"; do
      grep -qE "^${k}=" .env || missing="$missing $k"
    done
    if [ -n "$missing" ]; then
      warn "keys absent from .env (fine only if set in the environment):$missing"
    else
      ok ".env carries every key this deployment needs"
    fi
    # Values with a wrong code default — worth eyeballing, never auto-changed.
    grep -qE '^MULTISOURCE_ROUTING_SHADOW=1' .env \
      || warn "MULTISOURCE_ROUTING_SHADOW is not 1 — SHADOW=0 regressed single-source queries here (MULTI_SOURCE_DEPLOYMENT.md §3)"
    grep -qE '^REQUIRED_SOURCE_ESCALATION_ENABLED=0' .env \
      || warn "REQUIRED_SOURCE_ESCALATION_ENABLED is not 0 — the code default 1 forces un-granted sources into scope (RBAC leak)"
  else
    warn ".env not found — it is gitignored and does not travel; every flag falls back to its code default"
  fi

  # --- Postgres major version vs the compose pin -------------------------------
  # This branch pins pgvector:pg16. Postgres refuses to start a newer major against an
  # older data dir AND vice versa, so a mismatch here means the data tier will not come
  # back up after `up -d` — the one failure in this script that is not silent, but is
  # much cheaper to catch before the build than after.
  local pin pgv
  local cfg; cfg="$(dc config 2>/dev/null || true)"
  pin="$(printf '%s\n' "$cfg" | grep -oE 'pgvector/pgvector:pg[0-9]+' | head -1 | grep -oE '[0-9]+$' || true)"
  if svc_up "$PG"; then
    pgv="$(dc exec -T "$PG" cat /var/lib/postgresql/data/PG_VERSION 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -n "$pin" ] && [ -n "$pgv" ] && [ "$pin" != "$pgv" ]; then
      bad "compose pins Postgres pg${pin} but this host's pg_data is PG${pgv} — it will not start"
      info "either keep pg${pgv} in docker-compose.yml for this host, or pg_dumpall + restore first"
    elif [ -n "$pgv" ]; then
      ok "Postgres pg${pgv} matches the compose pin (pg${pin:-?})"
    fi
  else
    warn "$PG is not running — could not compare pg_data's major version against the compose pin (pg${pin:-?})"
  fi

  # --- who consumes the ingestion queue ----------------------------------------
  if svc_defined "$WORKER"; then
    svc_up "$WORKER" && ok "$WORKER is running (consumes the 'ingestion' queue)" \
                     || warn "$WORKER is defined but not running — re-ingestion will queue and never start"
  else
    warn "no '$WORKER' service in this compose configuration"
    info "docker-compose.prod.yml has no ingest worker (PRODUCTION_READINESS_PLAN B7 is outstanding)."
    info "Ingestion enqueued here would sit in Redis unconsumed — the reingest phase will refuse to run."
  fi

  # --- migrations ---------------------------------------------------------------
  if svc_up "$API"; then
    local pending
    local plan; plan="$(dc exec -T "$API" python manage.py showmigrations --plan </dev/null 2>/dev/null || true)"
    pending="$(printf '%s\n' "$plan" | grep -c '^\[ \]' || true)"
    if [ "${pending:-0}" -gt 0 ]; then
      ok "${pending} migration(s) pending — the migrate phase will apply them"
      printf '%s\n' "$plan" | grep '^\[ \]' | sed 's/^/      /' || true
    else
      ok "no pending migrations"
    fi
  else
    warn "$API is not running — could not list pending migrations"
  fi

  inventory
}

# =============================================================================
# inventory — the per-source "does this need re-ingesting?" table
# =============================================================================
inventory() {
  log "Per-source inventory (tenant=$TENANT)"

  if ! svc_up "$PG"; then
    warn "$PG is not running — cannot inventory sources"
    return 0
  fi
  PG_DB="$(dc exec -T "$PG" sh -c 'printf "%s" "$POSTGRES_DB"' </dev/null | tr -d '\r')"

  local probe_svc="$INFER"
  svc_up "$probe_svc" || probe_svc="$API"
  svc_up "$probe_svc" || { warn "neither $INFER nor $API is running — cannot read artifact paths"; return 0; }

  local sql rows
  sql="SELECT s.id, s.dialect, COALESCE(s.name,''),
              COALESCE(to_char(MAX(j.finished_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI'), 'never'),
              CASE WHEN MAX(j.finished_at) > TIMESTAMPTZ '${SPARSE_FIX_UTC}' THEN 'fresh' ELSE 'stale' END
         FROM sources_source s
         LEFT JOIN ingestion_ingestionjob j
                ON j.source_id = s.id AND j.status = 'success' AND j.tenant = '${TENANT}'
        GROUP BY s.id, s.dialect, s.name
        ORDER BY s.id"
  rows="$(psql_q "$PG_DB" "$sql" || true)"
  [ -n "$rows" ] || { warn "no sources found in $PG_DB"; return 0; }

  printf '  %-4s %-12s %-16s %-17s %-9s %-6s %s\n' id dialect name "last ingest(UTC)" artifacts card "chunks"
  printf '  %s\n' "-------------------------------------------------------------------------------------------"

  STALE_DOC_SOURCES=""; NOCARD_SOURCES=""; NOARTIFACT_SOURCES=""; NEVER_INGESTED=""
  local id dialect name last freshness kind a_state c_state k_state
  while IFS='|' read -r id dialect name last freshness <&3; do
    [ -n "$id" ] || continue
    kind="$(kind_of "$dialect")"

    # Artifacts: a relational source answers THROUGH its on-disk scoped model; a
    # datalake/document source builds a lite model at query time and legitimately has
    # none, so only the relational case is a blocker.
    if has_artifact "$probe_svc" "$id" veda_semantic_model.json; then
      a_state="ok"
    elif [ "$last" = "never" ]; then
      # Never ingested at all — not the artifact-scoping migration this script is about.
      # Registering a source and never ingesting it is a legitimate state (a stub row, a
      # source retired before the platform moved), so it warns rather than blocks.
      a_state="-"; NEVER_INGESTED="${NEVER_INGESTED}${id},"
    elif [ "$kind" = "relational" ]; then
      a_state="MISSING"; NOARTIFACT_SOURCES="${NOARTIFACT_SOURCES}${id},"
    else
      a_state="n/a"
    fi

    if has_artifact "$probe_svc" "$id" veda_routing_card.json; then
      c_state="ok"
    elif [ "$last" = "never" ]; then
      c_state="n/a"
    else
      c_state="none"; NOCARD_SOURCES="${NOCARD_SOURCES}${id},"
    fi

    # Doc chunks: only document sources store doc_chunks sparse weights.
    if [ "$kind" != "document" ] || [ "$last" = "never" ]; then
      k_state="n/a"
    elif [ "$freshness" = "fresh" ]; then
      k_state="ok"
    else
      k_state="STALE"; STALE_DOC_SOURCES="${STALE_DOC_SOURCES}${id},"
    fi

    printf '  %-4s %-12s %-16s %-17s %-9s %-6s %s\n' \
           "$id" "$dialect" "${name:0:16}" "$last" "$a_state" "$c_state" "$k_state"
  done 3<<< "$rows"

  STALE_DOC_SOURCES="${STALE_DOC_SOURCES%,}"; NOCARD_SOURCES="${NOCARD_SOURCES%,}"
  NOARTIFACT_SOURCES="${NOARTIFACT_SOURCES%,}"; NEVER_INGESTED="${NEVER_INGESTED%,}"

  echo
  if [ -n "$NOARTIFACT_SOURCES" ]; then
    bad "no scoped semantic model for relational source(s): $NOARTIFACT_SOURCES"
    info "resolve_source_artifact() has no flat fallback — these sources will plan SQL with an EMPTY model"
    info "fix: re-ingest them (REINGEST_MISSING_ARTIFACTS=1), or regenerate their artifacts into"
    info "     <artifact_root>/$TENANT/<id>/ before serving traffic"
  else
    ok "every ingested relational source has its own scoped semantic model"
  fi
  [ -n "$NEVER_INGESTED" ] && warn "source(s) never ingested in this deployment: $NEVER_INGESTED — nothing to migrate or re-embed"
  if [ -n "$NOCARD_SOURCES" ]; then
    ok "routing card missing for source(s): $NOCARD_SOURCES — the backfill phase rebuilds them (no re-ingest)"
    act "backfill routing cards for sources $NOCARD_SOURCES"
  else
    ok "every ingested source has a routing card"
  fi
  if [ -n "$STALE_DOC_SOURCES" ]; then
    ok "document source(s) ingested before ${SPARSE_FIX_UTC%% *}: $STALE_DOC_SOURCES"
    info "their doc_chunks hold case-SENSITIVE sparse weights; the query side now lowercases."
    info "no backfill exists — the reingest phase re-embeds them."
    act "re-ingest document sources $STALE_DOC_SOURCES (sparse case-normalisation)"
  else
    ok "no document source needs re-embedding"
  fi
}

# =============================================================================
# mutating phases
# =============================================================================
do_build() {
  log "Build — requirements changed on this branch (mysql-connector-python); a restart would not install it"
  run $DC build
}

do_migrate() {
  log "Migrate — substrate 0009 (VerifiedQueryCache.substrate_version)"
  info "old verified-cache rows carry substrate_version='' and stop matching: the cache goes cold and refills."
  svc_up "$API" || run $DC up -d "$API"
  run $DC exec -T "$API" python manage.py migrate --noinput
}

do_up() {
  log "Recreate — 'up -d', never 'restart' (env is read at container CREATE time)"
  run $DC up -d
  [ "$DRY_RUN" = "1" ] && return 0
  local i
  for i in $(seq 1 40); do
    svc_up "$API" && svc_up "$INFER" && break
    sleep 3
  done
  dc ps --format '  {{.Service}}\t{{.State}}\t{{.Status}}' || true
}

do_backfill() {
  log "Backfill — routing cards (pure transform of existing artifacts; idempotent)"
  local targets="${SOURCES:-$NOCARD_SOURCES}"
  if [ -z "$targets" ]; then
    ok "nothing to backfill — every ingested source already has a card"
    return 0
  fi
  local svc="$WORKER"
  svc_up "$svc" || svc="$INFER"
  if ! svc_up "$svc"; then
    bad "neither $WORKER nor $INFER is running — cannot run the backfill"
    return 0
  fi
  info "sources: $targets (in $svc)"
  run $DC exec -T "$svc" sh -c \
    "cd /app/veda_core && python /app/scripts/backfill_routing_cards.py --sources '$targets' --tenant '$TENANT'"
}

do_reingest() {
  log "Re-ingest — only the sources whose STORED data this branch invalidated"

  local targets
  if [ -n "$SOURCES" ]; then
    targets="$SOURCES"
    info "explicit --sources: $targets (detection skipped)"
  else
    targets="$STALE_DOC_SOURCES"
    if [ "$REINGEST_MISSING_ARTIFACTS" = "1" ] && [ -n "$NOARTIFACT_SOURCES" ]; then
      targets="${targets:+$targets,}$NOARTIFACT_SOURCES"
      info "REINGEST_MISSING_ARTIFACTS=1 — also re-ingesting $NOARTIFACT_SOURCES"
    fi
  fi

  if [ -z "$targets" ]; then
    ok "no source needs re-ingesting"
    return 0
  fi
  if ! svc_defined "$WORKER"; then
    bad "no '$WORKER' service — an enqueued job would sit in Redis with no consumer. Refusing to enqueue."
    info "sources that still need it: $targets"
    return 0
  fi
  if ! svc_up "$WORKER"; then
    bad "$WORKER is not running — refusing to enqueue $targets"
    return 0
  fi

  echo
  info "about to re-ingest source(s): $targets   (force=True — a resume would skip exactly the stages we need)"
  info "this reads the source systems and rewrites their embeddings; it is the expensive step."
  confirm "proceed?" || { warn "re-ingestion declined — sources still stale: $targets"; return 0; }

  local sid
  for sid in ${targets//,/ }; do
    ingest_one "$sid" || bad "ingestion failed for source $sid"
  done
}

ingest_one() {
  local sid="$1" baseline jid status stages waited=0
  log "Ingesting source $sid (tenant=$TENANT)"

  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m enqueue task_ingest_source(source_id=%s, tenant=%s, force=True) and poll\n' "$sid" "$TENANT"
    return 0
  fi

  baseline="$(psql_q "$PG_DB" "SELECT COALESCE(MAX(id),0) FROM ingestion_ingestionjob" | tr -d '[:space:]')"
  dc exec -T "$API" python manage.py shell -c \
    "from apps.ingestion.tasks import task_ingest_source; print(task_ingest_source.delay(source_id=${sid}, tenant='${TENANT}', force=True).id)" \
    | sed 's/^/    celery task /'

  # Poll the job row rather than the celery result: the row is what the platform (and
  # /admin) considers authoritative, and it carries per-stage progress.
  while [ "$waited" -lt "$INGEST_TIMEOUT" ]; do
    sleep 10; waited=$((waited + 10))
    jid="$(psql_q "$PG_DB" "SELECT id FROM ingestion_ingestionjob WHERE source_id=${sid} AND id > ${baseline} ORDER BY id DESC LIMIT 1" | tr -d '[:space:]')"
    [ -n "$jid" ] || { printf '\r    waiting for the worker to pick the job up (%ss)' "$waited"; continue; }
    status="$(psql_q "$PG_DB" "SELECT status FROM ingestion_ingestionjob WHERE id=${jid}" | tr -d '[:space:]')"
    stages="$(psql_q "$PG_DB" "SELECT string_agg(name || ':' || status, ' ' ORDER BY \"order\") FROM ingestion_ingestionstage WHERE job_id=${jid} AND status <> 'pending'" | tr -d '\r')"
    printf '\r    job %s [%s] %-70.70s' "$jid" "$status" "${stages:-starting}"
    case "$status" in
      success) echo; ok "source $sid ingested (job $jid)"; return 0 ;;
      failed)  echo
               bad "source $sid FAILED (job $jid)"
               psql_q "$PG_DB" "SELECT name, status, left(replace(error_traceback, chr(10), ' '), 300) FROM ingestion_ingestionstage WHERE job_id=${jid} AND status='failed'" | sed 's/^/      /'
               return 1 ;;
    esac
  done
  echo; bad "source $sid timed out after ${INGEST_TIMEOUT}s (job ${jid:-none}) — it may still be running"
  return 1
}

# =============================================================================
# verify
# =============================================================================
do_verify() {
  log "Verify"

  if dc exec -T "$API" python manage.py migrate --check >/dev/null 2>&1; then
    ok "all migrations applied"
  else
    bad "migrations still pending after the migrate phase"
  fi

  local ready
  ready="$(dc exec -T "$API" python -c \
    "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/readyz', timeout=15).read().decode()[:500])" 2>&1 || true)"
  if echo "$ready" | grep -q '"status": *"ready"'; then
    ok "/readyz: ready"
    info "$ready"
  else
    bad "/readyz is not ready"
    info "$ready"
  fi

  # Each source's model must describe its OWN tables — the semantic-model bleed in
  # MULTI_SOURCE_DEPLOYMENT.md §2 passed the firewall and answered from the wrong schema.
  if svc_up "$INFER"; then
    log "Per-source semantic models (datalake/document resolve through Redis; relational is on-disk and prints [])"
    local sql rows id dialect
    sql="SELECT id, dialect FROM sources_source ORDER BY id"
    rows="$(psql_q "$PG_DB" "$sql" 2>/dev/null || true)"
    while IFS='|' read -r id dialect <&3; do
      [ -n "$id" ] || continue
      [ "$(kind_of "$dialect")" = "relational" ] && continue
      dc exec -T -w /app/veda_core -e PYTHONPATH=/app:/app/veda_core "$INFER" python -c "
from veda_hybrid import _load_sm_from_redis
sm = _load_sm_from_redis(scope=($id, '$TENANT'))
print('    source $id ($dialect):', sorted({k.split('.')[0] for k in (sm or {}).get('columns', {})}))" </dev/null 2>/dev/null \
        || info "source $id ($dialect): could not read the published model"
    done 3<<< "$rows"
  fi

  inventory

  log "Still yours to do by hand (they need real credentials and real questions)"
  info "1. the three permission cases in MULTI_SOURCE_DEPLOYMENT.md §5 — case 3 must REFUSE without naming the source"
  info "2. one question per source, checking each answers from its OWN data"
  info "3. the same question against /api/v1/query AND the streaming endpoint (contextvars are lost separately there)"
}

# =============================================================================
# main
# =============================================================================
printf '\033[1m VEDA — deploy %s\033[0m\n' "$EXPECT_BRANCH"
printf ' compose : %s\n phases  : %s\n tenant  : %s%s\n' \
       "$DC" "$PHASES" "$TENANT" "$([ "$DRY_RUN" = 1 ] && echo '   (DRY RUN)')"

want_phase preflight && preflight
# Every mutating phase needs the inventory the preflight builds; rebuild it when the
# preflight was skipped (PHASES=backfill,verify and friends).
if ! want_phase preflight && { want_phase backfill || want_phase reingest; }; then
  inventory
fi

if [ ${#BLOCKERS[@]} -gt 0 ]; then
  log "Blocked — ${#BLOCKERS[@]} issue(s) must be resolved first"
  printf '  \033[1;31m✗\033[0m %s\n' "${BLOCKERS[@]}"
  # A missing scoped artifact is the one blocker with an in-script answer.
  case " ${BLOCKERS[*]} " in
    *"no scoped semantic model"*)
      info "re-run with REINGEST_MISSING_ARTIFACTS=1 to have this script re-ingest them" ;;
  esac
  exit 1
fi

want_phase build    && do_build
want_phase migrate  && do_migrate
want_phase up       && do_up
want_phase backfill && do_backfill
want_phase reingest && do_reingest
want_phase verify   && do_verify

# ------------------------------------------------------------------- summary
log "Summary"
if [ "${#ACTIONS[@]}" -gt 0 ] && want_phase preflight && [ "$PHASES" = "preflight" ]; then
  printf '  data work this deployment needs:\n'
  printf '    - %s\n' "${ACTIONS[@]}"
fi
if [ "${#WARNINGS[@]}" -gt 0 ]; then
  uniq_warnings="$(printf '%s\n' "${WARNINGS[@]}" | awk '!seen[$0]++')"
  printf '  \033[1;33m%s warning(s)\033[0m\n' "$(printf '%s\n' "$uniq_warnings" | wc -l | tr -d ' ')"
  printf '%s\n' "$uniq_warnings" | sed 's/^/    ! /'
fi
if [ "${#BLOCKERS[@]}" -gt 0 ]; then
  printf '  \033[1;31m%s blocker(s)\033[0m\n' "${#BLOCKERS[@]}"
  printf '    ✗ %s\n' "${BLOCKERS[@]}"
  exit 1
fi
printf '  \033[1;32mno blockers\033[0m\n'
