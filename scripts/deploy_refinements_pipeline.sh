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
#   artifacts  adopt a pre-scoping source's FLAT artifacts into its scoped directory
#              (the cheap alternative to re-ingesting it — see ARTIFACT_OWNER below).
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
#   --file f.yml            EXTRA_COMPOSE     extra `-f` overlay, repeatable. Use this to
#                                             hold a host-local difference (e.g. a pg17
#                                             data dir) WITHOUT editing the tracked
#                                             docker-compose.yml, which another host reads.
#   --all-sources           ALL_SOURCES=1     re-ingest EVERY previously-ingested source, not
#                                             just the ones this branch invalidated. Ordered
#                                             cheap-first (document/datalake, then relational)
#                                             so a failure surfaces before the multi-hour one.
#   --full                  FULL_INGEST=1     defeat the auto-resume for the sources being
#                                             re-ingested: apps/ingestion/tasks.py::_should_resume
#                                             sets VEDA_RESUME=1 when the source has ANY prior
#                                             FAILED job in its history — and under resume, L3
#                                             skips when a semantic model exists on disk and L4
#                                             skips when the source already has column
#                                             embeddings. Without --full such a run LOOKS like a
#                                             full ingest and silently reuses the old model and
#                                             the old vectors. --full moves the model aside
#                                             (timestamped, reversible) and clears that source's
#                                             embedding rows so both stages genuinely re-run.
#   --artifact-owner 2      ARTIFACT_OWNER=2  the source that PRODUCED the flat artifacts in
#                                             <artifact_root>/*.json. Never inferred: those
#                                             files came from exactly one source's ingestion,
#                                             and handing them to the wrong source is the
#                                             semantic-model bleed in MULTI_SOURCE_DEPLOYMENT
#                                             §2 (wrong-schema answers that pass the firewall).
#                                             Set it and the `artifacts` phase copies them into
#                                             <artifact_root>/<tenant>/<id>/, which is what the
#                                             M1 close-out expects — no re-ingest.
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
PHASES="${PHASES:-preflight,build,migrate,up,artifacts,backfill,reingest,verify}"
API="${API:-api}"; INFER="${INFER:-inference}"; WORKER="${WORKER:-ingest-worker}"; PG="${PG:-postgres}"
EXPECT_BRANCH="${EXPECT_BRANCH:-feat/refinements-pipeline}"
# The commit that made chunk_embedder lowercase text before sparse encoding. Chunks
# whose ingestion finished before this are case-mismatched against the (now lowercased)
# query. Override only if you cherry-picked that change onto a different timeline.
SPARSE_FIX_UTC="${SPARSE_FIX_UTC:-2026-09-16 00:00:00+00}"
REINGEST_MISSING_ARTIFACTS="${REINGEST_MISSING_ARTIFACTS:-0}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-7200}"
EXTRA_COMPOSE="${EXTRA_COMPOSE:-}"
ARTIFACT_OWNER="${ARTIFACT_OWNER:-}"
# --check trims PHASES to `preflight`, so "will a later phase fix this?" cannot be answered
# from PHASES alone: a preview that reports a blocker the run it previews would not hit is
# worse than no preview. CHECK_MODE says "judge against the full run".
CHECK_MODE="${CHECK_MODE:-0}"
SKIP_PHASES=""
ALL_SOURCES="${ALL_SOURCES:-0}"
FULL_INGEST="${FULL_INGEST:-0}"

# Derived artifacts a reader resolves per-source (every name passed to
# config.resolve_source_artifact / source_artifact_path in the tree). Deliberately EXCLUDES
# veda_semantic_checkpoint.json (a resume checkpoint — adopting it would make a later resume
# skip stages it should run) and the caches (veda_verified_queries/parity_baseline/
# synonym_enrich_cache), which are not per-source artifacts.
SCOPED_ARTIFACTS="veda_semantic_model.json veda_relationship_graph.json veda_join_paths.json \
veda_unified_graph.json veda_enrichment_index.json veda_rerank_docs.json veda_value_referents.json \
veda_concept_graph.json veda_domain_synonyms.json veda_entity_aliases.json veda_glossary.json \
veda_hnsw.json veda_profiling.json concepts.json dimensions.json metrics.json MANIFEST.json"

while [ $# -gt 0 ]; do
  case "$1" in
    --check)        PHASES="preflight"; CHECK_MODE=1 ;;
    --prod)         PROD=1 ;;
    --yes|-y)       ASSUME_YES=1 ;;
    --dry-run)      DRY_RUN=1 ;;
    --skip-build)   SKIP_PHASES="${SKIP_PHASES}build," ;;
    --skip-ingest)  SKIP_PHASES="${SKIP_PHASES}reingest," ;;
    --phases)       PHASES="$2"; shift ;;
    --sources)      SOURCES="$2"; shift ;;
    --tenant)       TENANT="$2"; shift ;;
    --file)         EXTRA_COMPOSE="$EXTRA_COMPOSE -f $2"; shift ;;
    --artifact-owner) ARTIFACT_OWNER="$2"; shift ;;
    --all-sources)  ALL_SOURCES=1 ;;
    --full)         FULL_INGEST=1 ;;
    -h|--help)      sed -n '2,/^# =\{20,\}$/p' "$0" | sed -n '2,$p' | tail -r | sed -n '2,$p' | tail -r; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# Drop the --skip-* phases in bash, not sed: `\b` is a GNU extension, so on macOS the old
# `sed 's/\breingest\b,\?//'` matched nothing and --skip-ingest silently did nothing at all
# — the run still reached the phase and asked to ingest.
if [ -n "$SKIP_PHASES" ]; then
  _kept=""; _ifs_save="$IFS"; IFS=','
  for _p in $PHASES; do
    case ",${SKIP_PHASES}" in *",${_p},"*) continue ;; esac
    _kept="${_kept:+$_kept,}$_p"
  done
  IFS="$_ifs_save"; PHASES="$_kept"
fi

if [ "$ALL_SOURCES" = "1" ] && [ "$INGEST_TIMEOUT" = "7200" ]; then
  INGEST_TIMEOUT=28800   # source 2 is 178 tables; its LLM stage alone runs for hours
fi

COMPOSE_FILES="-f docker-compose.yml"
[ "$PROD" = "1" ] && COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
# COMPOSE_FILE is docker compose's OWN env var, and its value is path-separator delimited
# ("a.yml:b.yml"). Honour that shape instead of passing it as one -f, so a host that exports
# COMPOSE_FILE=docker-compose.yml:docker-compose.pg17.yml to make every bare `docker compose`
# safe does not silently break this script.
if [ -n "${COMPOSE_FILE:-}" ]; then
  COMPOSE_FILES=""
  _ifs_save="$IFS"; IFS=":,"
  for _cf in $COMPOSE_FILE; do
    [ -n "$_cf" ] && COMPOSE_FILES="$COMPOSE_FILES -f $_cf"
  done
  IFS="$_ifs_save"
fi
# Overlays come LAST so they win, and they are never written back to the tracked compose
# file: the pg pin differs per host (one machine's pg_data is PG16, another's PG17) and
# editing docker-compose.yml to suit this host silently breaks the other one.
[ -n "$EXTRA_COMPOSE" ] && COMPOSE_FILES="$COMPOSE_FILES $EXTRA_COMPOSE"
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

has_flat_artifact() {  # $1 = service — is there a pre-scoping flat semantic model to adopt?
  local root; root="$(art_root "$1")"
  [ -n "$root" ] || return 1
  dc exec -T "$1" sh -c 'test -f "$0/veda_semantic_model.json"' "$root" </dev/null >/dev/null 2>&1
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
INGESTED_SOURCES=""    # every previously-ingested source, cheap kinds first (set by inventory)

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
    if svc_up "$WORKER"; then
      ok "$WORKER is running (consumes the 'ingestion' queue)"
    elif want_phase up; then
      ok "$WORKER is defined but not running — the up phase starts it before anything is enqueued"
    else
      warn "$WORKER is defined but not running — re-ingestion would queue and never start"
    fi
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
  INGESTED_SOURCES=""; local _relational_last=""
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

    # Cheap kinds first, relational last: a broken connector or a missing SLM shows up in
    # minutes instead of after the multi-hour relational source has already run.
    if [ "$last" != "never" ]; then
      if [ "$kind" = "relational" ]; then _relational_last="${_relational_last}${id},"
      else INGESTED_SOURCES="${INGESTED_SOURCES}${id},"; fi
    fi

    printf '  %-4s %-12s %-16s %-17s %-9s %-6s %s\n' \
           "$id" "$dialect" "${name:0:16}" "$last" "$a_state" "$c_state" "$k_state"
  done 3<<< "$rows"

  INGESTED_SOURCES="${INGESTED_SOURCES}${_relational_last}"; INGESTED_SOURCES="${INGESTED_SOURCES%,}"
  STALE_DOC_SOURCES="${STALE_DOC_SOURCES%,}"; NOCARD_SOURCES="${NOCARD_SOURCES%,}"
  NOARTIFACT_SOURCES="${NOARTIFACT_SOURCES%,}"; NEVER_INGESTED="${NEVER_INGESTED%,}"

  echo
  if [ -n "$NOARTIFACT_SOURCES" ]; then
    if { [ "$ALL_SOURCES" = "1" ] || [ "$REINGEST_MISSING_ARTIFACTS" = "1" ]; } \
       && { want_phase reingest || [ "$CHECK_MODE" = "1" ]; }; then
      ok "no scoped model for source(s) $NOARTIFACT_SOURCES — the reingest phase rebuilds it from scratch"
      act "re-ingest $NOARTIFACT_SOURCES (rebuilds its scoped artifacts)"
    elif [ -n "$ARTIFACT_OWNER" ] && { want_phase artifacts || [ "$CHECK_MODE" = "1" ]; } \
       && has_flat_artifact "$probe_svc"; then
      # A remedy is already selected for this run, so it is an action, not a blocker.
      ok "no scoped model for source(s) $NOARTIFACT_SOURCES — the artifacts phase adopts source $ARTIFACT_OWNER's flat files"
      act "adopt flat artifacts into <artifact_root>/$TENANT/$ARTIFACT_OWNER/"
      case ",$NOARTIFACT_SOURCES," in
        *",$ARTIFACT_OWNER,"*) ;;
        *) warn "ARTIFACT_OWNER=$ARTIFACT_OWNER is not among the sources missing a model ($NOARTIFACT_SOURCES) — check that id" ;;
      esac
    else
      bad "no scoped semantic model for relational source(s): $NOARTIFACT_SOURCES"
      info "resolve_source_artifact() has no flat fallback — these sources will plan SQL with an EMPTY model"
      if has_flat_artifact "$probe_svc"; then
        info "flat artifacts ARE present at <artifact_root>/*.json — they were produced by ONE source's"
        info "ingestion. Re-run with --artifact-owner <that source id> to adopt them (seconds, no re-ingest)."
      else
        info "no flat artifacts to adopt either — these sources must be re-ingested"
        info "(REINGEST_MISSING_ARTIFACTS=1)"
      fi
    fi
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

do_artifacts() {
  log "Artifacts — adopt the flat, pre-scoping artifacts into their owner's scoped directory"

  if [ -z "$ARTIFACT_OWNER" ]; then
    ok "no --artifact-owner given — nothing to adopt (this phase never guesses whose files those are)"
    return 0
  fi

  local svc="$INFER"
  svc_up "$svc" || svc="$API"
  svc_up "$svc" || { bad "neither $INFER nor $API is running — cannot adopt artifacts"; return 0; }

  local root; root="$(art_root "$svc")"
  [ -n "$root" ] || { bad "could not resolve the artifact root inside $svc"; return 0; }
  info "root=$root  owner=source $ARTIFACT_OWNER  tenant=$TENANT"

  # cp, never mv: the flat files stay put as the rollback, and nothing else reads them any
  # more anyway (resolve_source_artifact stopped falling back to them). Executed directly
  # rather than through run(): %q-quoting a multi-line shell body prints an unreadable
  # line, so the intent is stated instead.
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m cp %s/{%s,...} -> %s/%s/%s/  (never overwrites)\n' \
           "$root" "veda_semantic_model.json" "$root" "$TENANT" "$ARTIFACT_OWNER"
  else
    dc exec -T "$svc" sh -c '
      root="$1"; tenant="$2"; sid="$3"; shift 3
      mkdir -p "$root/$tenant/$sid"
      for f in "$@"; do
        if [ -f "$root/$tenant/$sid/$f" ]; then
          echo "    kept    $f (already scoped, not overwritten)"
        elif [ -f "$root/$f" ]; then
          cp -p "$root/$f" "$root/$tenant/$sid/$f" && echo "    adopted $f"
        else
          echo "    absent  $f (no flat copy)"
        fi
      done' sh "$root" "$TENANT" "$ARTIFACT_OWNER" $SCOPED_ARTIFACTS </dev/null
  fi

  [ "$DRY_RUN" = "1" ] && return 0

  # Prove the adopted model actually belongs to this source: its table count must match the
  # table nodes THIS source has in the engine store. A mismatch is the bleed — a model
  # describing another schema passes the firewall and answers from the wrong data.
  local model_tables node_tables engine_db
  engine_db="$(dc exec -T "$API" sh -c 'printf "%s" "${VEDA_INTERNAL_DBNAME:-veda_engine}"' </dev/null | tr -d '\r')"
  model_tables="$(dc exec -T "$svc" python -c \
    "import json;print(len(json.load(open('$root/$TENANT/$ARTIFACT_OWNER/veda_semantic_model.json')).get('tables',{})))" \
    </dev/null 2>/dev/null | tr -d '[:space:]' || true)"
  node_tables="$(psql_q "$engine_db" "SELECT count(*) FROM graph_nodes WHERE source_id='$ARTIFACT_OWNER' AND node_type='table'" | tr -d '[:space:]' || true)"

  if [ -n "$model_tables" ] && [ -n "$node_tables" ] && [ "$model_tables" = "$node_tables" ]; then
    ok "adopted model describes $model_tables tables — matches source $ARTIFACT_OWNER's own table nodes"
  elif [ -n "$model_tables" ] && [ -n "$node_tables" ]; then
    bad "adopted model has $model_tables tables but source $ARTIFACT_OWNER owns $node_tables in graph_nodes"
    info "that is the semantic-model bleed — the flat files likely belong to a DIFFERENT source."
    info "remove $root/$TENANT/$ARTIFACT_OWNER/ and re-check the owner id before serving traffic."
  else
    warn "could not cross-check the adopted model against graph_nodes (model=$model_tables nodes=$node_tables)"
  fi
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
  elif [ "$ALL_SOURCES" = "1" ]; then
    targets="$INGESTED_SOURCES"
    info "--all-sources: every previously-ingested source, cheap kinds first -> $targets"
    [ -n "$NEVER_INGESTED" ] && info "excluded (never ingested here, so this is not a RE-ingest): $NEVER_INGESTED"
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

  # An ingestion that cannot reach its SLM does not fail fast — the LLM stage degrades and
  # the run still reports success, hours later, with a poorer semantic model. Checking and
  # then continuing anyway (which this did) is the worst of both: the operator is warned
  # about a degraded run they are then allowed to start. Abort unless waived.
  if ! check_ingest_slm && [ "${IGNORE_SLM_CHECK:-0}" != "1" ]; then
    info "refusing to ingest with a degraded SLM. Pull the model, or waive with IGNORE_SLM_CHECK=1"
    info "(a document source's chunking/embedding does not use the SLM — waiving is reasonable there;"
    info " a relational source's semantic layer is LLM-authored, so waiving costs you model quality)."
    return 0
  fi

  # Auto-resume turns a "full re-ingest" into a partial one, silently (see --full in the
  # header). Report it per source BEFORE asking for confirmation, because it changes what
  # the operator is actually agreeing to.
  local sid resuming=""
  for sid in ${targets//,/ }; do
    if [ "$(prior_failed_jobs "$sid")" -gt 0 ]; then resuming="${resuming}${sid},"; fi
  done
  resuming="${resuming%,}"
  if [ -n "$resuming" ]; then
    if [ "$FULL_INGEST" = "1" ]; then
      warn "source(s) $resuming have a prior FAILED job -> VEDA_RESUME=1; --full will clear the skip preconditions"
    else
      warn "source(s) $resuming have a prior FAILED job -> VEDA_RESUME=1: L3 (semantic model) and L4"
      info "    (biencoder) will SKIP for them and the old artifacts/vectors survive. That is not a"
      info "    full ingest. Re-run with --full to make those stages actually run."
    fi
  fi

  echo
  info "about to re-ingest source(s): $targets   (force=True, timeout ${INGEST_TIMEOUT}s per source)"
  info "this reads the source systems and rewrites their embeddings; it is the expensive step."
  [ "$FULL_INGEST" = "1" ] && [ -n "$resuming" ] && \
    info "--full will move aside the scoped semantic model and DELETE the column embeddings of: $resuming"
  confirm "proceed?" || { warn "re-ingestion declined — sources still stale: $targets"; return 0; }

  for sid in ${targets//,/ }; do
    [ "$FULL_INGEST" = "1" ] && defeat_resume "$sid"
    ingest_one "$sid" || bad "ingestion failed for source $sid"
  done
}

prior_failed_jobs() {  # $1 = source id -> count of FAILED jobs in its history
  # _should_resume() looks at the source's WHOLE history, not just the previous job, so one
  # failure years ago still forces every later run into resume mode.
  psql_q "$PG_DB" "SELECT count(*) FROM ingestion_ingestionjob WHERE source_id=$1 AND status='failed'" \
    2>/dev/null | tr -d '[:space:]' || echo 0
}

check_ingest_slm() {
  local probe
  probe="$(dc exec -T "$WORKER" python -c "
import os, json, urllib.request
url = os.environ.get('OLLAMA_URL', 'http://ollama:11434').rstrip('/')
want = os.environ.get('SLM_MODEL_NAME', '')
try:
    have = [m.get('name', '') for m in json.load(urllib.request.urlopen(url + '/api/tags', timeout=8)).get('models', [])]
except Exception as e:
    print('UNREACHABLE', url, type(e).__name__); raise SystemExit
print('OK' if want in have else 'MISSING', want, url, '|', ','.join(have)[:160])
" </dev/null 2>/dev/null || true)"

  case "$probe" in
    OK*)          ok "ingestion SLM reachable and the pinned model is pulled (${probe#OK })"
                  return 0 ;;
    MISSING*)     bad "the ingestion SLM host does not serve the pinned model: ${probe#MISSING }"
                  info "    pull it on the Ollama host it points at, e.g.:  ollama pull <SLM_MODEL_NAME>"
                  info "    or re-pin SLM_MODEL_NAME in .env to a model that host already serves"
                  return 1 ;;
    UNREACHABLE*) bad "the ingestion SLM is unreachable: ${probe#UNREACHABLE }"
                  info "    ingest-worker points OLLAMA_URL at the HOST's native Ollama; start it before ingesting"
                  return 1 ;;
    *)            warn "could not probe the ingestion SLM — proceeding blind"
                  return 0 ;;
  esac
}

defeat_resume() {  # $1 = source id — make L3/L4 actually re-run under VEDA_RESUME=1
  local sid="$1" root svc="$INFER" engine_db stamp
  [ "$(prior_failed_jobs "$sid")" -gt 0 ] || return 0
  svc_up "$svc" || svc="$API"
  root="$(art_root "$svc")"; [ -n "$root" ] || { warn "no artifact root — cannot clear the resume skip for $sid"; return 0; }
  stamp="$(date +%Y%m%d-%H%M%S)"

  log "Clearing the resume skip for source $sid (--full)"
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m mv veda_semantic_model.json -> .bak-%s ; DELETE column_embeddings_v2 WHERE source_id=%s\n' "$stamp" "$sid"
    return 0
  fi
  # Moved, not deleted: if the run dies before L3 rewrites it, the old model is one mv away.
  dc exec -T "$svc" sh -c '
      f="$1/$2/$3/veda_semantic_model.json"
      [ -f "$f" ] && mv "$f" "$f.bak-$4" && echo "    moved aside $f -> $f.bak-$4" || echo "    no scoped model to move (L3 will run)"
    ' sh "$root" "$TENANT" "$sid" "$stamp" </dev/null
  engine_db="$(dc exec -T "$API" sh -c 'printf "%s" "${VEDA_INTERNAL_DBNAME:-veda_engine}"' </dev/null | tr -d '\r')"
  psql_q "$engine_db" "DELETE FROM column_embeddings_v2 WHERE source_id='$sid'" >/dev/null 2>&1 \
    && ok "cleared source $sid's column embeddings — L4 will re-embed them this run" \
    || warn "could not clear source $sid's column embeddings; L4 may still skip"
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
               info "    NOTE: this failure now forces every LATER ingest of source $sid into resume mode"
               info "    (L3/L4 skip). Retry with --full, or those stages will not re-run."
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
    local engine_db sm_tables n_sm n_nodes
    engine_db="$(dc exec -T "$API" sh -c 'printf "%s" "${VEDA_INTERNAL_DBNAME:-veda_engine}"' </dev/null | tr -d '\r')"
    while IFS='|' read -r id dialect <&3; do
      [ -n "$id" ] || continue
      [ "$(kind_of "$dialect")" = "relational" ] && continue
      sm_tables="$(dc exec -T -w /app/veda_core -e PYTHONPATH=/app:/app/veda_core "$INFER" python -c "
from veda_hybrid import _load_sm_from_redis
sm = _load_sm_from_redis(scope=($id, '$TENANT'))
print(','.join(sorted({k.split('.')[0] for k in (sm or {}).get('columns', {})})))" </dev/null 2>/dev/null | tr -d '\r' || true)"
      info "source $id ($dialect): [${sm_tables}]"

      # The bleed is a COUNT mismatch, not an empty model: before the per-source build
      # existed every source published homzhub's model, so a 4-column datalake source came
      # back describing 8 foreign tables — and answered from them, past the firewall.
      # awk NF, not `tr | wc -l`: the list has no trailing newline, so wc undercounts by one
      # (a two-table model read as one, which then "detected" a bleed that was not there).
      n_sm=0; [ -n "$sm_tables" ] && n_sm="$(printf '%s' "$sm_tables" | awk -F, '{print NF}')"
      n_nodes="$(psql_q "$engine_db" "SELECT count(*) FROM graph_nodes WHERE source_id='$id' AND node_type='table'" 2>/dev/null | tr -d '[:space:]' || true)"
      if [ -n "$n_nodes" ] && [ "${n_nodes:-0}" -gt 0 ] && [ "$n_sm" -ne "${n_nodes:-0}" ]; then
        warn "source $id publishes a model of $n_sm table(s) but owns $n_nodes in graph_nodes — possible semantic-model bleed"
        info "    fix without re-ingesting:  $DC exec $API python scripts/backfill_semantic_model.py --source-ids $id --tenant $TENANT"
      fi
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
      info "cheapest remedy first: --artifact-owner <id of the source that produced the flat files>"
      info "only if there are no flat artifacts to adopt: REINGEST_MISSING_ARTIFACTS=1 (re-ingests, hours)" ;;
  esac
  exit 1
fi

want_phase build     && do_build
want_phase migrate   && do_migrate
want_phase up        && do_up
want_phase artifacts && do_artifacts
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
