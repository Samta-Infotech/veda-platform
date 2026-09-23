#!/usr/bin/env bash
# =============================================================================
# scripts/seed_and_reingest.sh
# VEDA — seed the source registry and rebuild EVERY derived artifact for every
#        source: schema, graph, embeddings (dense + sparse), semantics, the
#        cross-source links, the semantic bridge, routing cards, the uniform
#        item layer, source descriptions and the RBAC catalog projection.
#
# WHAT THIS IS FOR
#   `scripts/deploy_refinements_pipeline.sh` is a TARGETED deploy: it re-ingests
#   only what one branch invalidated. This script is the other one — the blanket
#   "rebuild everything from the sources" pass you want after a wipe, a restore,
#   a new machine, an embedding-model change, or when you no longer trust what is
#   in the stores. It is deliberately safe to run on a live deployment too: with
#   no --reset it never deletes anything the ingestion would not overwrite anyway.
#
# THE PART A RUNBOOK ALWAYS MISSES
#   `task_ingest_source` does NOT produce a complete deployment. Five enrichment
#   passes sit outside it, each one silent when absent — nothing raises, answers
#   just get worse:
#
#   1. THE THREE DJANGO POST-INGEST HOOKS ARE FLAG-GATED, DEFAULT OFF.
#      VEDA_AUTO_SYNC_CATALOG, SOURCE_PROFILER_ENABLED and
#      SOURCE_ITEM_PROFILER_ENABLED all default to "0" (config/settings/base.py)
#      and none of them is set in this repo's .env. So a by-the-book ingestion
#      leaves: no CatalogResource rows (absent == DENIED for every RBAC check),
#      no Source.description (the routing prior reads it), and no SourceItem rows
#      at all (the per-item routing prior is simply never primed).
#      => the `enrich` phase runs all three EXPLICITLY, flag or no flag.
#
#   2. THE SEMANTIC BRIDGE IS ORDER-DEPENDENT.
#      A document source's entity_linker matches chunks against the structured
#      sources' column vectors in graph_node_embeddings (ingestion/semantic_linker.py).
#      Ingest the documents BEFORE the databases and every semantic_about edge is
#      silently missing — the bridge degrades to a no-op and nobody is told.
#      => `ingest` orders datalake → nosql → relational → document, and `enrich`
#         re-runs the relink phase afterwards regardless, because the last
#         relational source to finish changes what the bridge could have matched.
#
#   3. CROSS-SOURCE LINKS NEED THE JOIN-KEY COLUMNS THE SAMPLER SKIPS.
#      value_sampler deliberately skips PK/FK/id columns, which are exactly the
#      columns cross-source discovery needs sketches for. The in-pipeline sketch
#      pass inherits that gap; scripts/backfill_cross_source.py samples the join
#      keys straight from the source instead.
#      => `enrich` runs it tenant-wide after every source is in.
#
#   4. ROUTING CARDS EXIST ONLY FOR RELATIONAL SOURCES AFTER AN INGEST.
#      The card is an L5 stage, and L5 only runs on the relational path — the
#      document/datalake dispatcher never reaches it. Same for the per-source
#      semantic model of a non-relational source.
#      => `enrich` backfills cards for every source, and rebuilds a non-relational
#         source's model when `verify` finds it publishing another source's tables.
#
#   5. column_values AND fk_adjacency ARE GLOBAL, AND TRUNCATED ON EVERY
#      RELATIONAL INGEST (value_sampler.py:299, vector_store.py:253).
#      With two or more relational sources, whichever one finishes LAST owns those
#      two tables for the whole deployment. That is not fixable from a script — so
#      the script picks the owner deliberately (--owner, default: the relational
#      source with the most table nodes) and says who it is, instead of letting
#      ingest order decide it by accident.
#
# PHASES  (--phases to pick, comma-separated; default: everything except reset)
#   preflight  read-only. Clone, compose, services, SLM, embed host, engine DB,
#              migrations, manifest, source reachability, data folders. Blocks.
#   seed       upsert the Source rows from the manifest. Idempotent, never deletes
#              a source, never clears a credential it cannot supply.
#   reset      OPT-IN (--reset). Scope-delete every derived row + artifact of the
#              targeted sources so the re-ingest is a true rebuild, not a resume.
#   ingest     force re-ingest, correctly ordered, polling the job rows.
#   enrich     the five passes above + re-warm the query-tier caches.
#   verify     a per-source matrix of every store, plus the bleed check + /readyz.
#
# USAGE
#   bash scripts/seed_and_reingest.sh --check          # preflight only, changes nothing
#   bash scripts/seed_and_reingest.sh --dry-run        # print every mutation, run none
#   bash scripts/seed_and_reingest.sh                  # seed + ingest + enrich + verify
#   bash scripts/seed_and_reingest.sh --reset --yes    # full rebuild, unattended
#   bash scripts/seed_and_reingest.sh --export         # write the manifest FROM the live DB
#   bash scripts/seed_and_reingest.sh --phases enrich,verify
#   bash scripts/seed_and_reingest.sh --sources 3,4 --reset
#
# FLAGS / ENV
#   --manifest PATH      source manifest      (default scripts/sources.seed.json)
#   --sources LIST       ids or names to act on (default: every enabled manifest source)
#   --tenant NAME        tenant               (default: default)
#   --owner ID           relational source that ends up owning column_values/fk_adjacency
#   --reset              add the reset phase (scope-delete derived state first)
#   --reset-global       also truncate the global column_values/fk_adjacency
#   --recreate-engine    DROP + CREATE veda_engine before anything (nuclear; implies --reset)
#   --with-synonyms      also run the SLM synonym enrichment (hours; opt-in)
#   --with-row-counts    routing cards open a read-only connection for real row counts
#   --resume-ok          allow VEDA_RESUME to skip L3/L4 (default: this script defeats it)
#   --skip-ingest        seed + enrich + verify only
#   --prod / --file F    compose overlays
#   -y/--yes  --dry-run  --check  -h/--help
#   INGEST_TIMEOUT       seconds per source   (default 7200; 28800 with a relational source)
#   IGNORE_SLM_CHECK=1   ingest anyway with a degraded/unreachable SLM
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- configuration
COMPOSE="${COMPOSE:-docker compose}"
PROD="${PROD:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_MODE="${CHECK_MODE:-0}"
EXPORT_MODE=0
TENANT="${TENANT:-default}"
MANIFEST="${MANIFEST:-scripts/sources.seed.json}"
SOURCES="${SOURCES:-}"
OWNER="${OWNER:-}"
DO_RESET=0
# Default ON (2026-09-23). column_values + fk_adjacency have no source_id and are TRUNCATEd
# by every relational ingest, and storage_adapters/writer.py::sync_from_engine reads them
# UNFILTERED — so leaving stale rows there made each source's warm() adopt the previous
# relational source's entire schema. That polluted three sources with homzhub's 116 tables
# and then failed homzhub's own warm on a duplicate substrate PK, after a 4-hour build.
# --keep-global restores the old behaviour for the rare case where you are resetting ONE
# source and deliberately want another source's sampled values left in place.
RESET_GLOBAL=1
RECREATE_ENGINE=0
WITH_SYNONYMS=0
WITH_ROW_COUNTS=0
RESUME_OK=0
SKIP_PHASES=""
PHASES="${PHASES:-preflight,seed,ingest,enrich,verify}"
API="${API:-api}"; INFER="${INFER:-inference}"; WORKER="${WORKER:-ingest-worker}"; PG="${PG:-postgres}"
INGEST_TIMEOUT="${INGEST_TIMEOUT:-7200}"
EXTRA_COMPOSE="${EXTRA_COMPOSE:-}"

# Per-source derived artifacts, resolved by config.resolve_source_artifact(). The reset
# phase moves these aside so L3/L5 genuinely rebuild them. Deliberately EXCLUDES
# veda_semantic_checkpoint.json: keeping it would make the next run skip stages it must run.
SCOPED_ARTIFACTS="veda_semantic_model.json veda_relationship_graph.json veda_join_paths.json \
veda_unified_graph.json veda_enrichment_index.json veda_rerank_docs.json veda_value_referents.json \
veda_concept_graph.json veda_domain_synonyms.json veda_entity_aliases.json veda_glossary.json \
veda_hnsw.json veda_profiling.json veda_routing_card.json concepts.json dimensions.json \
metrics.json MANIFEST.json"

# Engine tables carrying a source_id — everything a source's ingestion writes into
# veda_engine. The reset phase deletes only THIS source's rows from each.
# (column_values + fk_adjacency have no source_id: they are global and get TRUNCATEd by
# the next relational ingest anyway — see --reset-global.)
ENGINE_SCOPED_TABLES="column_embeddings_v2 table_embeddings_v2 column_sparse_v1 table_sparse_v1 \
chunk_sparse_v1 doc_chunks table_metadata graph_nodes graph_edges graph_node_embeddings \
column_sketches entity_value_embeddings"

# Django substrate tables carrying a source_id. Truncating the SUBSTRATE alone would be
# wrong (warm() rebuilds it from the engine store), but a reset must clear it too, or a
# table dropped upstream lingers in the projection forever.
DJANGO_SCOPED_TABLES="substrate_columnprofile substrate_columnvaluesample substrate_fkedge \
substrate_glossaryentry substrate_graphartifact substrate_graphedge substrate_graphnode \
substrate_schemacolumn substrate_schematable substrate_semanticconcept substrate_semantictype \
substrate_smcolumn substrate_smconcept substrate_smretrievaldoc substrate_smsynonym \
substrate_smtable substrate_substrateversion substrate_synonym substrate_syntheticpair \
substrate_tablemetadata substrate_verifiedquerycache"

# .env keys with no usable code default for a full rebuild. Missing is a warning, not a
# blocker: a host may legitimately export them into the environment instead.
REQUIRED_ENV_KEYS=(
  DJANGO_SETTINGS_MODULE POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD
  VEDA_INTERNAL_HOST VEDA_INTERNAL_PORT VEDA_INTERNAL_DBNAME VEDA_INTERNAL_USER
  SLM_MODEL_NAME SLM_TEMPERATURE METAL_EMBED_URL OLLAMA_URL
)

while [ $# -gt 0 ]; do
  case "$1" in
    --check)            PHASES="preflight"; CHECK_MODE=1 ;;
    --export)           EXPORT_MODE=1 ;;
    --prod)             PROD=1 ;;
    --yes|-y)           ASSUME_YES=1 ;;
    --dry-run)          DRY_RUN=1 ;;
    --reset)            DO_RESET=1 ;;
    --reset-global)     DO_RESET=1; RESET_GLOBAL=1 ;;
    --keep-global)      RESET_GLOBAL=0 ;;
    --recreate-engine)  DO_RESET=1; RECREATE_ENGINE=1 ;;
    --with-synonyms)    WITH_SYNONYMS=1 ;;
    --with-row-counts)  WITH_ROW_COUNTS=1 ;;
    --resume-ok)        RESUME_OK=1 ;;
    --skip-ingest)      SKIP_PHASES="${SKIP_PHASES}ingest," ;;
    --skip-enrich)      SKIP_PHASES="${SKIP_PHASES}enrich," ;;
    --phases)           PHASES="$2"; shift ;;
    --sources)          SOURCES="$2"; shift ;;
    --manifest)         MANIFEST="$2"; shift ;;
    --tenant)           TENANT="$2"; shift ;;
    --owner)            OWNER="$2"; shift ;;
    --file)             EXTRA_COMPOSE="$EXTRA_COMPOSE -f $2"; shift ;;
    -h|--help)          sed -n '2,/^# =\{20,\}$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# The reset phase is never in the default PHASES — it deletes rows, so it has to be asked
# for by name. --reset splices it in after seed rather than making the operator retype the
# whole phase list to get it.
if [ "$DO_RESET" = "1" ]; then
  case ",${PHASES}," in
    *",reset,"*) ;;
    *) PHASES="$(printf '%s' "$PHASES" | sed 's/seed/seed,reset/')"
       case ",${PHASES}," in *",reset,"*) ;; *) PHASES="reset,${PHASES}" ;; esac ;;
  esac
fi

# Drop --skip-* phases in bash: `\b` is a GNU sed extension, so the obvious sed form
# silently does nothing on macOS (the same trap deploy_refinements_pipeline.sh hit).
if [ -n "$SKIP_PHASES" ]; then
  _kept=""; _ifs_save="$IFS"; IFS=','
  for _p in $PHASES; do
    case ",${SKIP_PHASES}" in *",${_p},"*) continue ;; esac
    _kept="${_kept:+$_kept,}$_p"
  done
  IFS="$_ifs_save"; PHASES="$_kept"
fi

# `docker compose exec -T` disables the pseudo-TTY but still forwards stdin. Backgrounded
# (nohup ... &), the first such read raises SIGTTIN and the shell SUSPENDS the whole run —
# silently, with the log frozen mid-preflight. Detaching stdin here covers every exec call
# site at once. confirm() still prompts correctly: it reads /dev/tty, not stdin.
exec < /dev/null

COMPOSE_FILES="-f docker-compose.yml"
[ "$PROD" = "1" ] && COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
# COMPOSE_FILE is docker compose's own env var and is path-separator delimited. Honour that
# shape, so a host that exports COMPOSE_FILE=a.yml:b.yml to make every bare `docker compose`
# safe (the pg16/pg17 pin differs per machine) does not silently get a single bogus -f.
if [ -n "${COMPOSE_FILE:-}" ]; then
  COMPOSE_FILES=""
  _ifs_save="$IFS"; IFS=":,"
  for _cf in $COMPOSE_FILE; do
    [ -n "$_cf" ] && COMPOSE_FILES="$COMPOSE_FILES -f $_cf"
  done
  IFS="$_ifs_save"
fi
[ -n "$EXTRA_COMPOSE" ] && COMPOSE_FILES="$COMPOSE_FILES $EXTRA_COMPOSE"
DC="${COMPOSE} ${COMPOSE_FILES}"

# ---------------------------------------------------------------------- output
BLOCKERS=(); WARNINGS=(); NOTES=()
log()   { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[1;33m!\033[0m %s\n' "$*"; WARNINGS+=("$*"); }
bad()   { printf '  \033[1;31m✗\033[0m %s\n' "$*"; BLOCKERS+=("$*"); }
info()  { printf '    %s\n' "$*"; }
note()  { NOTES+=("$*"); }

run() {  # mutating command; honours --dry-run. %q so the echoed line is copy-pasteable.
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
# $DC is intentionally unquoted: it is a command prefix with arguments.
dc() { $DC "$@"; }

# Capture first, match second. `docker compose ... | grep -q` is a race under pipefail:
# grep exits on its first match, compose takes SIGPIPE, and the pipeline reports 141 even
# though the match succeeded — the same check then "passes" in one phase and fails in the next.
svc_up() {
  local ps_out; ps_out="$(dc ps --format '{{.Service}} {{.State}}' 2>/dev/null || true)"
  printf '%s\n' "$ps_out" | grep -E "^$1 (running|up)" >/dev/null 2>&1
}
svc_defined() {
  local svcs; svcs="$(dc config --services 2>/dev/null || true)"
  printf '%s\n' "$svcs" | grep -x "$1" >/dev/null 2>&1
}

# </dev/null on every exec: `docker compose exec` drains the caller's stdin, which otherwise
# eats the rest of a `while read` loop after its first iteration.
psql_q() {  # $1 = database, $2 = SQL -> "|"-separated rows
  dc exec -T "$PG" sh -c 'exec psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$0" -tAF"|" -c "$1"' "$1" "$2" </dev/null
}

psql_x() {  # $1 = database, $2 = mutating SQL. Honours --dry-run; never aborts the run.
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m [%s] %s\n' "$1" "$2"
    return 0
  fi
  psql_q "$1" "$2" >/dev/null 2>&1 || true
}

PG_DB=""; ENGINE_DB=""
resolve_dbs() {
  [ -n "$PG_DB" ] && return 0
  PG_DB="$(dc exec -T "$PG" sh -c 'printf "%s" "$POSTGRES_DB"' </dev/null | tr -d '\r')"
  ENGINE_DB="$(dc exec -T "$API" sh -c 'printf "%s" "${VEDA_INTERNAL_DBNAME:-veda_engine}"' </dev/null | tr -d '\r')"
  [ -n "$PG_DB" ] && [ -n "$ENGINE_DB" ]
}

# Artifact root exactly as veda_core/config.py resolves it (VEDA_ARTIFACT_ROOT may be
# absolute; relative hangs off <VEDA_APP_DIR>/veda_core). Read from a container, never the
# host: in prod the code is baked into the image and there is no bind mount to look at.
ART_ROOT=""
art_root() {
  [ -n "$ART_ROOT" ] && { printf '%s' "$ART_ROOT"; return 0; }
  ART_ROOT="$(dc exec -T "$1" sh -c '
      root="${VEDA_ARTIFACT_ROOT:-data}"
      case "$root" in /*) ;; *) root="${VEDA_APP_DIR:-/app}/veda_core/$root" ;; esac
      printf "%s" "$root"' </dev/null 2>/dev/null | tr -d '\r')"
  printf '%s' "$ART_ROOT"
}

has_artifact() {  # $1 = service, $2 = source id, $3 = filename
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

engine_svc() {  # the service with veda_core on its path, preferred order
  if svc_up "$WORKER"; then echo "$WORKER"
  elif svc_up "$INFER"; then echo "$INFER"
  else echo ""; fi
}

# Inventory, filled by resolve_targets(): ids in INGEST ORDER, plus the per-kind splits.
TARGET_IDS=""          # everything this run acts on, ingest-ordered
REL_IDS=""; DOC_IDS=""; LAKE_IDS=""; NOSQL_IDS=""
INVENTORY_DONE=0

# =============================================================================
# manifest
# =============================================================================
# read_manifest is called from $( ), where a bad()/warn() would be written into a SUBSHELL's
# copy of BLOCKERS and silently lost — so it stays quiet and check_manifest does the reporting.
read_manifest() {  # -> manifest JSON on stdout; nothing on stderr; non-zero when unusable
  [ -f "$MANIFEST" ] || return 1
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$MANIFEST" >/dev/null 2>&1 || return 1
  cat "$MANIFEST"
}

check_manifest() {  # the reporting half, called from the phase body itself
  if [ ! -f "$MANIFEST" ]; then
    bad "manifest not found: $MANIFEST"
    info "write one from the live registry:  bash scripts/seed_and_reingest.sh --export"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); d['sources']" "$MANIFEST" >/dev/null 2>&1; then
    bad "manifest is not valid JSON, or has no 'sources' list: $MANIFEST"
    return 1
  fi
  return 0
}

do_export() {
  log "Export — writing the manifest FROM the live registry ($MANIFEST)"
  svc_up "$API" || { bad "$API is not running — cannot read the registry"; return 1; }

  # Credentials are never exported. A source whose password lives inline in the row is
  # exported with "password_keep": true, which tells the seed phase to leave that column
  # exactly as it found it. Exporting the secret into a file that lands in git is the one
  # failure here that cannot be undone by re-running anything.
  local out
  out="$(dc exec -T "$API" python - <<'PY' 2>/dev/null
import json, django
django.setup()
from apps.sources.models import Source

rows = []
for s in Source.objects.all().order_by("pk"):
    row = {
        "name": s.name, "dialect": s.dialect, "connector_type": s.connector_type,
        "enabled": True,
    }
    if s.host:
        row.update(host=s.host, port=s.port, dbname=s.dbname, db_user=s.db_user)
    if s.password_env:
        row["password_env"] = s.password_env
    elif s.password_inline:
        row["password_keep"] = True          # seed must not clear the working credential
    if s.connection_secret_ref:
        row["connection_secret_ref"] = s.connection_secret_ref
    if s.source_path:
        row.update(source_path=s.source_path, doc_recursive=s.doc_recursive)
        if s.doc_formats:
            row["doc_formats"] = list(s.doc_formats)
        if s.doc_max_file_mb:
            row["doc_max_file_mb"] = s.doc_max_file_mb
    if s.exclude_tables:
        row["exclude_tables"] = list(s.exclude_tables)
    if s.schema_filter:
        row["schema_filter"] = s.schema_filter
    if s.domain_tags:
        row["domain_tags"] = list(s.domain_tags)
    # Only a HUMAN-written description belongs in the manifest; an auto-generated one is
    # rebuilt by the enrich phase and pinning it here would freeze it forever.
    if s.description and not s.description_generated:
        row["description"] = s.description
    if s.is_canonical:
        row["is_canonical"] = True
    # A registered-but-never-connectable stub (no host, no path) is exported disabled:
    # ingesting it can only fail, and a failed job forces every later run into resume mode.
    if not s.host and not s.source_path:
        row["enabled"] = False
        row["note"] = "stub row: no connection and no path — nothing to ingest"
    rows.append(row)

print(json.dumps({"version": 1, "sources": rows}, indent=2))
PY
)"
  if [ -z "$out" ]; then bad "export produced nothing — is $API healthy?"; return 1; fi
  if [ "$DRY_RUN" = "1" ]; then printf '%s\n' "$out"; return 0; fi
  printf '%s\n' "$out" > "$MANIFEST"
  ok "wrote $MANIFEST ($(printf '%s' "$out" | grep -c '"name"') sources)"
  info "credentials were NOT exported; sources with an inline password carry \"password_keep\": true"
}

# =============================================================================
# preflight
# =============================================================================
preflight() {
  log "Preflight — read-only; nothing here changes the deployment"

  # --- which clone -------------------------------------------------------------
  # There are two clones of this repo on this machine and only one of them has the
  # complete .env; a change made in the other is invisible to the containers.
  local mounted
  mounted="$(dc config 2>/dev/null | grep -oE '^[[:space:]]*source: .*' | head -1 | awk '{print $2}' || true)"
  ok "repo: $REPO_ROOT ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git repo'))"
  if [ -n "$mounted" ] && [ "$mounted" != "$REPO_ROOT" ]; then
    bad "the containers bind-mount $mounted, not this clone ($REPO_ROOT)"
    info "every edit and every artifact this script writes would land in a tree nothing reads."
    info "run it from $mounted instead."
  fi

  # --- .env --------------------------------------------------------------------
  if [ -f .env ]; then
    local missing="" k
    for k in "${REQUIRED_ENV_KEYS[@]}"; do
      grep -qE "^${k}=" .env || missing="${missing}${k} "
    done
    if [ -n "$missing" ]; then
      warn ".env does not set: ${missing%% }"
      info "fine if the host exports them; SLM_MODEL_NAME/SLM_TEMPERATURE exist ONLY in .env"
      info "and without them the engine asks Ollama for a model it will not serve."
    else
      ok ".env carries every key this rebuild needs"
    fi
  else
    bad "no .env in $REPO_ROOT — the engine cannot resolve its model or its internal DB"
  fi

  # --- services ----------------------------------------------------------------
  local svc
  for svc in "$PG" "$API" "$WORKER" "$INFER"; do
    if svc_up "$svc"; then ok "service $svc is up"
    elif svc_defined "$svc"; then bad "service $svc is defined but not running ($DC up -d $svc)"
    else bad "no '$svc' service in this compose project"; fi
  done
  if ! svc_up "$WORKER"; then
    info "ingest-worker is the ONLY consumer of the ingestion queue — an enqueued job with no"
    info "worker sits in Redis forever and the script would poll a job row that never appears."
  fi

  svc_up "$PG" || return 0
  if ! resolve_dbs; then bad "could not resolve the database names from the containers"; return 0; fi
  ok "databases: django=$PG_DB engine=$ENGINE_DB (tenant=$TENANT)"

  # --- engine store ------------------------------------------------------------
  if psql_q "$ENGINE_DB" "SELECT 1 FROM pg_extension WHERE extname='vector'" | grep -q 1; then
    ok "$ENGINE_DB exists with pgvector installed"
  else
    bad "$ENGINE_DB has no 'vector' extension — every embedding write will fail"
    info "CREATE EXTENSION IF NOT EXISTS vector;  (inside $ENGINE_DB)"
  fi

  # --- migrations --------------------------------------------------------------
  if svc_up "$API"; then
    if dc exec -T "$API" python manage.py migrate --check >/dev/null 2>&1; then
      ok "all Django migrations applied"
    else
      bad "migrations are pending — seed would write into a stale schema"
      info "    $DC exec $API python manage.py migrate"
    fi
  fi

  # --- the SLM the semantic layer is authored by --------------------------------
  check_slm || true

  # --- the embedding path -------------------------------------------------------
  # Two DIFFERENT things resolve an embedding endpoint here and they fail differently:
  #
  #   the engine (ingest-worker)  METAL_EMBED_URL is an OPTIONAL host-GPU offload with an
  #     in-process CPU fallback (ingestion/m3_encoder.py:54). Unset is CORRECT — just much
  #     slower. Only a SET-but-unreachable URL is a problem, and that one is silent.
  #   the item profiler (api)     resolves its OWN two endpoints, each with a HARDCODED IP
  #     default (apps/sources/item_profiler.py:30-32). Unreachable, every item fails one by
  #     one, is logged and swallowed, and the rows end up with empty summaries and no
  #     routing vector — an unprimed prior that looks exactly like a primed one from outside.
  if svc_up "$WORKER"; then
    local embed
    embed="$(dc exec -T "$WORKER" python -c "$(probe_embed_py)" </dev/null 2>/dev/null || true)"
    case "$embed" in
      OK*)          ok "engine embedding offload reachable (${embed#OK })" ;;
      SHAPE*)       warn "the embedding host answered but in an unexpected shape: ${embed#SHAPE }"
                    info "    expected {\"vecs\": [[...]]} from scripts/metal_embed_server.py" ;;
      UNSET)        if [ -f .env ] && grep -qE '^METAL_EMBED_URL=.+' .env; then
                      # `docker compose restart` does NOT re-read .env — env is bound at container
                      # CREATE time — so a container started before the edit keeps the old value
                      # forever and nothing but its own startup log says so.
                      warn "ENV DRIFT: .env sets METAL_EMBED_URL but $WORKER has it EMPTY"
                      info "    every embedding runs in-process on CPU instead of the host GPU."
                      info "    measured here: ~1.5s per graph node on CPU — for a 2000-node source"
                      info "    that is hours of extra wall clock for byte-identical vectors."
                      info "    fix before ingesting:  $DC up -d $WORKER   (NOT restart)"
                    else
                      ok "METAL_EMBED_URL unset in $WORKER and in .env — BGE-M3 runs on CPU (deliberate)"
                    fi ;;
      UNREACHABLE*) bad "METAL_EMBED_URL is SET but unreachable from $WORKER: ${embed#UNREACHABLE }"
                    info "    a set-but-dead offload does not fall back cleanly — check the IP,"
                    info "    it has moved before (.43 -> .39), or unset it to embed on CPU." ;;
      *)            warn "could not probe the engine embedding path — proceeding blind" ;;
    esac
  fi

  if svc_up "$API"; then
    local item_probe
    item_probe="$(dc exec -T "$API" python - <<'PY' 2>/dev/null || true
import json, urllib.request, django
django.setup()
from apps.sources import item_profiler as ip

def probe(url, payload):
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=20).read(1)
        return "OK"
    except Exception as e:
        return type(e).__name__

print("SLM", ip._SLM_URL, probe(ip._SLM_URL, {
    "model": ip._SLM_MODEL, "stream": False,
    "messages": [{"role": "user", "content": "ping"}]}))
print("EMBED", ip._METAL_URL, probe(ip._METAL_URL, {"texts": ["veda preflight"]}))
PY
)"
    local line kind url state
    while IFS=' ' read -r kind url state <&4; do
      [ -n "$kind" ] || continue
      if [ "$state" = "OK" ]; then
        ok "item profiler $kind endpoint reachable ($url)"
      else
        warn "item profiler $kind endpoint unreachable: $url ($state)"
        [ "$kind" = "SLM" ] &&           info "    VEDA_SLM_CHAT_URL is unset, so it fell back to a hardcoded IP. The SourceItem"
        [ "$kind" = "SLM" ] &&           info "    rows will be created with EMPTY summaries and no routing vector — silently."
        [ "$kind" = "SLM" ] &&           info "    set VEDA_SLM_CHAT_URL=\${OLLAMA_URL}/api/chat in .env, then \`up -d api\` (not restart)."
        [ "$kind" = "EMBED" ] &&           info "    set METAL_EMBED_URL in .env and recreate the api container."
      fi
    done 4<<< "$item_probe"
  fi

  # --- lock blockers on the engine store ------------------------------------------
  check_lock_blockers "preflight"

  # --- the manifest and what it points at ----------------------------------------
  local man=""
  check_manifest && man="$(read_manifest || true)"
  if [ -n "$man" ]; then
    local n; n="$(printf '%s' "$man" | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for s in d['sources'] if s.get('enabled', True)))")"
    ok "manifest $MANIFEST: $n enabled source(s)"
    preflight_manifest_targets "$man"
  fi

  resolve_targets || true
  preflight_global_store
}

# Every ingestion begins by ensuring its schema — "ALTER TABLE column_embeddings_v2 ADD
# COLUMN IF NOT EXISTS ...", which needs an ACCESS EXCLUSIVE lock. Any connection sitting in
# `idle in transaction` after so much as a SELECT on that table holds ACCESS SHARE and blocks
# it indefinitely. Nothing times out, nothing logs: the job just stops producing output.
# Worse, once the ALTER is queued every NEW reader queues behind it, so the table goes dark
# for the query tier too. Observed live 2026-09-22: a backend idle since an hour before the
# run stalled it for 39 minutes, and the work itself then took seconds.
_IDLE_TXN_WARN_SECONDS="${_IDLE_TXN_WARN_SECONDS:-120}"

has_lock_blockers() {  # silent, side-effect-free: is anything blocked or idle-in-txn right now?
  resolve_dbs >/dev/null 2>&1 || return 1
  local n
  n="$(psql_q "$ENGINE_DB" "
      SELECT count(*) FROM pg_stat_activity
       WHERE datname = current_database()
         AND (cardinality(pg_blocking_pids(pid)) > 0
              OR (state = 'idle in transaction'
                  AND now() - xact_start > interval '${_IDLE_TXN_WARN_SECONDS} seconds'))" \
      2>/dev/null | tr -d '[:space:]' || echo 0)"
  [ "${n:-0}" -gt 0 ]
}

check_lock_blockers() {  # $1 = context label; returns 1 when something is actually blocked
  resolve_dbs >/dev/null 2>&1 || return 0
  local idle blocked found=0

  idle="$(psql_q "$ENGINE_DB" "
      SELECT pid, round(extract(epoch FROM now() - xact_start))::int,
             left(replace(coalesce(query,''), chr(10), ' '), 60)
        FROM pg_stat_activity
       WHERE datname = current_database() AND state = 'idle in transaction'
         AND now() - xact_start > interval '${_IDLE_TXN_WARN_SECONDS} seconds'
       ORDER BY xact_start" 2>/dev/null || true)"

  local pid age q
  while IFS='|' read -r pid age q <&5; do
    [ -n "$pid" ] || continue
    found=1
    warn "backend $pid has been IDLE IN TRANSACTION for ${age}s on $ENGINE_DB"
    info "    last statement: $q"
    info "    it holds a read lock that blocks the schema-ensure ALTER every ingestion runs,"
    info "    which then blocks every later reader of that table too. Nothing times out."
    info "    clear it:  $DC exec $PG psql -U \$POSTGRES_USER -d $ENGINE_DB -c 'SELECT pg_terminate_backend($pid)'"
  done 5<<< "$idle"

  # Anything actually waiting right now, with who is holding it.
  blocked="$(psql_q "$ENGINE_DB" "
      SELECT a.pid, array_to_string(pg_blocking_pids(a.pid), ','),
             round(extract(epoch FROM now() - a.query_start))::int,
             left(replace(coalesce(a.query,''), chr(10), ' '), 55)
        FROM pg_stat_activity a
       WHERE cardinality(pg_blocking_pids(a.pid)) > 0" 2>/dev/null || true)"
  local bpid holders bage bq
  while IFS='|' read -r bpid holders bage bq <&5; do
    [ -n "$bpid" ] || continue
    found=1
    bad "backend $bpid has been BLOCKED for ${bage}s by pid(s) $holders"
    info "    waiting on: $bq"
  done 5<<< "$blocked"

  [ "$found" = "0" ] && [ "$1" = "preflight" ] && ok "no lock blockers on $ENGINE_DB"
  [ "$found" = "0" ]
}

probe_embed_py() {
  cat <<'PY'
import os, json, urllib.request
url = os.environ.get('METAL_EMBED_URL', '').strip().rstrip('/')
if not url:
    print('UNSET'); raise SystemExit
try:
    req = urllib.request.Request(url + '/encode_dense',
                                 data=json.dumps({'texts': ['veda preflight']}).encode(),
                                 headers={'Content-Type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=20))
    # scripts/metal_embed_server.py answers {"vecs": [[...1024 floats...]]}; the other two
    # keys are only here in case that server's shape ever changes.
    v = (r.get('vecs') or r.get('vectors') or r.get('embeddings') or [[]])[0]
    print('OK' if v else 'SHAPE', url, str(len(v)) + '-dim')
except Exception as e:
    print('UNREACHABLE', url, type(e).__name__)
PY
}

check_slm() {
  svc_up "$WORKER" || return 0
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
    OK*)          ok "ingestion SLM reachable, pinned model pulled (${probe#OK })"; return 0 ;;
    MISSING*)     bad "the SLM host does not serve the pinned model: ${probe#MISSING }"
                  info "    the LLM semantic layer then DEGRADES silently: the run still reports"
                  info "    success, hours later, with a poorer model. Pull it, or re-pin SLM_MODEL_NAME."
                  return 1 ;;
    UNREACHABLE*) bad "the ingestion SLM is unreachable: ${probe#UNREACHABLE }"; return 1 ;;
    *)            warn "could not probe the SLM — proceeding blind"; return 0 ;;
  esac
}

preflight_manifest_targets() {  # $1 = manifest json — is each source actually reachable?
  local man="$1" name dialect kind host port dbname user path
  printf '  %-18s %-12s %s\n' source kind reachability
  printf '  %s\n' "--------------------------------------------------------------------"
  while IFS='|' read -r name dialect host port dbname user path <&3; do
    [ -n "$name" ] || continue
    kind="$(kind_of "$dialect")"
    case "$kind" in
      relational|nosql)
        if [ -z "$host" ]; then
          printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "no host in the manifest — cannot ingest"
          bad "source '$name' has no connection in the manifest"
          continue
        fi
        # Probed FROM a container: the host's own reachability says nothing about the
        # container network, which is where ingestion actually runs.
        local probe
        probe="$(dc exec -T "$WORKER" python -c "
import sys, psycopg2
try:
    c = psycopg2.connect(host=sys.argv[1], port=int(sys.argv[2] or 5432), dbname=sys.argv[3],
                         user=sys.argv[4], password=sys.argv[5], connect_timeout=8)
    cur = c.cursor(); cur.execute(\"SELECT count(*) FROM information_schema.tables WHERE table_schema='public'\")
    print('OK', cur.fetchone()[0]); c.close()
except Exception as e:
    print('FAIL', type(e).__name__)
" "$host" "${port:-5432}" "$dbname" "$user" "$(source_password "$name")" </dev/null 2>/dev/null || echo 'FAIL probe')"
        case "$probe" in
          OK*) printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "reachable, ${probe#OK } public tables" ;;
          *)   printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "UNREACHABLE (${probe#FAIL })"
               bad "source '$name' is not reachable from $WORKER" ;;
        esac ;;
      document|datalake)
        if [ -z "$path" ]; then
          printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "no source_path"
          bad "source '$name' is a $kind source with no source_path"
          continue
        fi
        local files
        files="$(dc exec -T "$WORKER" sh -c 'test -d "$0" && find "$0" -type f | wc -l || echo -1' "$path" </dev/null 2>/dev/null | tr -d '[:space:]')"
        if [ "${files:--1}" = "-1" ]; then
          printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "$path does not exist in the container"
          bad "source '$name': $path is not present inside $WORKER"
        elif [ "${files:-0}" = "0" ]; then
          printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "$path is EMPTY"
          # An empty folder ingests "successfully" with zero chunks and answers nothing.
          warn "source '$name': $path holds no files — its ingestion will succeed with nothing in it"
        else
          printf '  %-18s %-12s %s\n' "${name:0:18}" "$kind" "$path ($files file(s))"
        fi ;;
    esac
  done 3<<< "$(printf '%s' "$man" | python3 -c "
import json, sys
for s in json.load(sys.stdin)['sources']:
    if not s.get('enabled', True):
        continue
    print('|'.join(str(s.get(k, '') or '') for k in
                   ('name', 'dialect', 'host', 'port', 'dbname', 'db_user', 'source_path')))
")"
}

source_password() {  # $1 = source name -> the password to probe with, never printed
  # Manifest-declared env ref first; otherwise ask the live row (a "password_keep" source).
  local name="$1"
  dc exec -T "$API" python - "$name" <<'PY' 2>/dev/null || true
import sys, django
django.setup()
from apps.sources.models import Source
s = Source.objects.filter(name=sys.argv[1]).first()
print(s.resolve_password() if s else "", end="")
PY
}

preflight_global_store() {
  # column_values and fk_adjacency have no source_id and are TRUNCATEd by every relational
  # ingestion. This is invisible at runtime — the loser just has no sampled values.
  local n=0 id
  for id in ${REL_IDS//,/ }; do [ -n "$id" ] && n=$((n + 1)); done
  [ "$n" -le 1 ] && { [ "$n" = "1" ] && ok "one relational source — column_values/fk_adjacency have a single unambiguous owner"; return 0; }
  warn "$n relational sources: column_values + fk_adjacency are GLOBAL and truncated per ingest"
  info "whichever finishes LAST owns them deployment-wide (value_sampler.py:299, vector_store.py:253)."
  info "this run ingests relational sources in this order: $REL_IDS — so source ${REL_IDS##*,} wins."
  info "pick deliberately with --owner <id>; per-source artifacts (value_referents, the Redis"
  info "value mirror) are built inside each source's own L5 and stay correct either way."
}

# =============================================================================
# targets — which sources, in which order
# =============================================================================
resolve_targets() {
  [ "$INVENTORY_DONE" = "1" ] && return 0
  resolve_dbs || return 1

  local man names sql rows
  man="$(read_manifest 2>/dev/null || true)"
  if [ -n "$man" ]; then
    names="$(printf '%s' "$man" | python3 -c "
import json, sys
print(','.join(\"'\" + s['name'].replace(\"'\", \"''\") + \"'\"
               for s in json.load(sys.stdin)['sources'] if s.get('enabled', True)))")"
  fi

  if [ -n "$SOURCES" ]; then
    # --sources takes ids or names; resolve both against the registry.
    local quoted; quoted="$(printf '%s' "$SOURCES" | tr ',' '\n' | sed "s/'/''/g; s/^/'/; s/$/'/" | paste -sd, -)"
    sql="SELECT id, dialect, name FROM sources_source WHERE name IN ($quoted) OR id::text IN ($quoted) ORDER BY id"
  elif [ -n "$names" ]; then
    sql="SELECT id, dialect, name FROM sources_source WHERE name IN ($names) ORDER BY id"
  else
    sql="SELECT id, dialect, name FROM sources_source ORDER BY id"
  fi
  rows="$(psql_q "$PG_DB" "$sql" 2>/dev/null || true)"

  REL_IDS=""; DOC_IDS=""; LAKE_IDS=""; NOSQL_IDS=""
  local id dialect name kind
  while IFS='|' read -r id dialect name <&3; do
    [ -n "$id" ] || continue
    kind="$(kind_of "$dialect")"
    case "$kind" in
      relational) REL_IDS="${REL_IDS}${id}," ;;
      document)   DOC_IDS="${DOC_IDS}${id}," ;;
      datalake)   LAKE_IDS="${LAKE_IDS}${id}," ;;
      nosql)      NOSQL_IDS="${NOSQL_IDS}${id}," ;;
    esac
  done 3<<< "$rows"
  REL_IDS="${REL_IDS%,}"; DOC_IDS="${DOC_IDS%,}"; LAKE_IDS="${LAKE_IDS%,}"; NOSQL_IDS="${NOSQL_IDS%,}"

  REL_IDS="$(order_relational "$REL_IDS")"

  # THE ORDER IS THE POINT, not an optimisation:
  #   datalake/nosql first — cheap, so a broken connector or a dead embed host surfaces in
  #     minutes rather than after the multi-hour relational run,
  #   relational next     — it produces the column vectors in graph_node_embeddings,
  #   document LAST       — because its entity_linker matches chunks against exactly those
  #     vectors. Documents first means every semantic_about edge is silently absent.
  TARGET_IDS="$(printf '%s' "${LAKE_IDS:+$LAKE_IDS,}${NOSQL_IDS:+$NOSQL_IDS,}${REL_IDS:+$REL_IDS,}${DOC_IDS}" | sed 's/,$//')"
  INVENTORY_DONE=1
  [ -n "$TARGET_IDS" ] || { warn "no sources matched — nothing to do"; return 1; }
  return 0
}

order_relational() {  # $1 = csv of relational ids -> same ids, the column_values owner last
  local ids="$1"
  [ -z "$ids" ] && return 0
  if [ -n "$OWNER" ]; then
    local rest="" id
    for id in ${ids//,/ }; do [ "$id" = "$OWNER" ] || rest="${rest}${id},"; done
    case ",$ids," in *",$OWNER,"*) printf '%s' "${rest}${OWNER}"; return 0 ;; esac
    warn "--owner $OWNER is not one of the relational sources ($ids) — ignoring it"
  fi
  # Default owner: the source with the most table nodes. It is the one whose sampled values
  # the most queries depend on, so if the global store can only hold one, hold that one.
  local ordered
  ordered="$(psql_q "$ENGINE_DB" "
      SELECT source_id FROM graph_nodes
       WHERE node_type='table' AND source_id IN ($(printf "'%s'," ${ids//,/ } | sed 's/,$//'))
       GROUP BY source_id ORDER BY count(*) ASC" 2>/dev/null | tr -d '\r' | paste -sd, - || true)"
  if [ -z "$ordered" ]; then printf '%s' "$ids"; return 0; fi
  # A source with no nodes yet (never ingested) is absent from that ranking — prepend it.
  local pre="" id
  for id in ${ids//,/ }; do
    case ",$ordered," in *",$id,"*) ;; *) pre="${pre}${id}," ;; esac
  done
  printf '%s' "${pre}${ordered}"
}

# =============================================================================
# seed
# =============================================================================
do_seed() {
  log "Seed — upsert the Source registry from $MANIFEST"
  svc_up "$API" || { bad "$API is not running — cannot seed"; return 0; }
  check_manifest || return 0
  local man; man="$(read_manifest)" || return 0

  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m upsert %s source row(s) from %s\n' \
           "$(printf '%s' "$man" | grep -c '"name"')" "$MANIFEST"
    return 0
  fi

  # The manifest is handed over as an env var, not read from /app/scripts: in prod the code
  # is baked into the image and this file is not in it.
  dc exec -T -e VEDA_SEED_MANIFEST_JSON="$man" "$API" python - <<'PY'
import json, os, django
django.setup()
from apps.sources.models import Source, SourceStatus

man = json.loads(os.environ["VEDA_SEED_MANIFEST_JSON"])
# Fields the manifest owns. `password_inline` is deliberately absent: a credential is never
# read from, nor written by, a file that lives in the repo.
FIELDS = ("dialect", "connector_type", "host", "port", "dbname", "db_user", "password_env",
          "connection_secret_ref", "source_path", "doc_formats", "doc_recursive",
          "doc_max_file_mb", "exclude_tables", "schema_filter", "domain_tags",
          "description", "is_canonical")

created = updated = unchanged = skipped = 0
for entry in man["sources"]:
    name = entry["name"]
    if not entry.get("enabled", True):
        if Source.objects.filter(name=name).exists():
            print(f"    skip     {name}  (disabled in the manifest; the existing row is left alone)")
        skipped += 1
        continue

    source = Source.objects.filter(name=name).first()
    values = {f: entry[f] for f in FIELDS if f in entry}
    # A description in the manifest is a HUMAN one: mark it so the profiler never overwrites it.
    if "description" in values:
        values["description_generated"] = False

    if source is None:
        # A brand-new source starts not-ready: the query path reads only ready=True sources,
        # so it stays invisible until its ingestion actually succeeds.
        Source.objects.create(name=name, status=SourceStatus.REGISTERED, ready=False, **values)
        print(f"    created  {name}")
        # "password_keep" means "the row already has an inline credential, leave it alone".
        # On a FRESH deployment there is no row to keep it from, so the source is created
        # with NO password and its ingestion fails at connect time — the one way this
        # manifest does not transplant cleanly to a new server. Say so at seed time rather
        # than letting it surface as a connection error an hour later.
        if entry.get("password_keep") and not entry.get("password_env"):
            print(f"    !        {name}: manifest says password_keep but this is a NEW row —")
            print(f"    !        there is no existing credential to keep. Set password_env in")
            print(f"    !        the manifest and export that variable, or the connection will fail.")
        created += 1
        continue

    changes = []
    for field, want in values.items():
        have = getattr(source, field)
        if have != want:
            changes.append(f"{field}: {have!r} -> {want!r}" if field != "exclude_tables"
                           else f"exclude_tables: {len(have or [])} -> {len(want or [])} entries")
            setattr(source, field, want)
    # password_keep is the whole reason seeding cannot be a blind update: clearing a working
    # inline credential would leave the source connectable-looking and un-ingestable.
    if entry.get("password_keep") and not source.password_inline:
        print(f"    !        {name}: manifest says password_keep, but the row has no inline password")
    if changes:
        source.save(update_fields=list(values.keys()) + ["updated_at"])
        print(f"    updated  {name}")
        for c in changes:
            print(f"               {c}")
        updated += 1
    else:
        unchanged += 1

print(f"\n    created={created} updated={updated} unchanged={unchanged} skipped={skipped}")
for s in Source.objects.all().order_by("pk"):
    print(f"    id={s.pk:<4} {s.name:<18} {s.dialect:<12} ready={str(s.ready):<5} {s.status}")
PY
  ok "registry seeded"
  INVENTORY_DONE=0     # ids may have changed; re-resolve before ingesting
  resolve_targets || true
}

# =============================================================================
# reset
# =============================================================================
do_reset() {
  log "Reset — scope-delete the derived state of: ${TARGET_IDS:-none}"
  resolve_targets || return 0
  resolve_dbs || { bad "cannot resolve the databases"; return 0; }

  info "this deletes DERIVED data only: engine embeddings/graph/chunks, the Django substrate"
  info "projection, the RBAC catalog rows, the item layer and the scoped artifacts."
  info "it never touches a source system, and never drops a Source row."
  info "prior ingestion JOB rows go too — otherwise _should_resume() sees an old failure and"
  info "forces the whole rebuild into resume mode, skipping the L3/L4 stages you came for."
  [ "$RESET_GLOBAL" = "1" ] && info "--reset-global: column_values + fk_adjacency will also be truncated"
  [ "$RECREATE_ENGINE" = "1" ] && info "--recreate-engine: $ENGINE_DB will be DROPPED and recreated (every source, not just these)"
  confirm "proceed with the reset?" || { warn "reset declined — the ingest phase will run against existing state"; return 0; }

  if [ "$RECREATE_ENGINE" = "1" ]; then
    log "Recreating $ENGINE_DB"
    # Dropping the engine DB is deployment-wide, so it is the one reset step that ignores
    # --sources: there is no way to drop "part of" a database.
    run dc exec -T "$PG" sh -c \
      'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$0\"" \
       && psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"$0\" OWNER \"$POSTGRES_USER\"" \
       && psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$0" -c "CREATE EXTENSION IF NOT EXISTS vector"' \
      "$ENGINE_DB"
    ok "$ENGINE_DB recreated with pgvector"
  fi

  local sid t root svc stamp
  svc="$INFER"; svc_up "$svc" || svc="$API"
  root="$(art_root "$svc")"
  stamp="$(date +%Y%m%d-%H%M%S)"

  for sid in ${TARGET_IDS//,/ }; do
    log "Reset source $sid"

    if [ "$RECREATE_ENGINE" != "1" ]; then
      for t in $ENGINE_SCOPED_TABLES; do
        # source_id is TEXT in the engine store — quoting it is not cosmetic: an unquoted
        # integer comparison against a text column is the exact int/text mismatch that made
        # ann_search return zero rows while reporting success.
        psql_x "$ENGINE_DB" "DELETE FROM $t WHERE source_id = '$sid'"
      done
      ok "engine rows cleared for source $sid"
    fi

    for t in $DJANGO_SCOPED_TABLES; do
      psql_x "$PG_DB" "DELETE FROM $t WHERE source_id = $sid"
    done
    psql_x "$PG_DB" "DELETE FROM access_management_catalogresource WHERE source_id = $sid"
    psql_x "$PG_DB" "DELETE FROM sources_sourceitem WHERE source_id = $sid"
    ok "Django substrate + catalog + item rows cleared for source $sid"

    # Job rows last: while they exist, _should_resume() turns the next run into a resume.
    psql_x "$PG_DB" "DELETE FROM ingestion_ingestionstage WHERE job_id IN (SELECT id FROM ingestion_ingestionjob WHERE source_id = $sid)"
    psql_x "$PG_DB" "DELETE FROM ingestion_ingestionjob WHERE source_id = $sid"
    psql_x "$PG_DB" "UPDATE sources_source SET ready = false, status = 'registered', last_ingested_at = NULL WHERE id = $sid"
    ok "ingestion history cleared for source $sid (the rebuild will not resume)"

    # Artifacts are MOVED, not deleted: if the rebuild dies before L3/L5 rewrite them, the
    # previous deployment is one `mv` away instead of gone.
    if [ -n "$root" ] && [ "$DRY_RUN" = "1" ]; then
      printf '  \033[0;35m[dry-run]\033[0m mv %s/%s/%s/{%s,...} -> *.bak-%s\n' "$root" "$TENANT" "$sid" "veda_semantic_model.json" "$stamp"
    elif [ -n "$root" ]; then
      dc exec -T "$svc" sh -c '
        root="$1"; tenant="$2"; sid="$3"; stamp="$4"; shift 4
        d="$root/$tenant/$sid"
        [ -d "$d" ] || { echo "    no artifact dir at $d"; exit 0; }
        for f in "$@"; do
          [ -f "$d/$f" ] && mv "$d/$f" "$d/$f.bak-$stamp" && echo "    moved aside $f"
        done
        exit 0' sh "$root" "$TENANT" "$sid" "$stamp" $SCOPED_ARTIFACTS </dev/null || true
    fi
  done

  if [ "$RESET_GLOBAL" = "1" ] && [ "$RECREATE_ENGINE" != "1" ]; then
    psql_x "$ENGINE_DB" "TRUNCATE TABLE column_values; TRUNCATE TABLE fk_adjacency;"
    ok "global column_values + fk_adjacency truncated (the relational ingest rewrites both)"
  fi
}

# =============================================================================
# ingest
# =============================================================================
do_ingest() {
  log "Ingest — full re-ingestion, ordered so the semantic bridge has something to bridge to"
  resolve_targets || return 0
  resolve_dbs || { bad "cannot resolve the databases"; return 0; }

  if ! svc_defined "$WORKER"; then
    bad "no '$WORKER' service — an enqueued job would sit in Redis with no consumer. Refusing."
    return 0
  fi
  if ! svc_up "$WORKER"; then bad "$WORKER is not running — refusing to enqueue $TARGET_IDS"; return 0; fi

  # A degraded SLM does not fail an ingestion; it quietly produces a poorer semantic model,
  # hours later. Warning the operator and then starting anyway is the worst of both.
  if ! check_slm && [ "${IGNORE_SLM_CHECK:-0}" != "1" ]; then
    info "refusing to ingest with a degraded SLM. Pull the model, or waive with IGNORE_SLM_CHECK=1."
    info "(a document/datalake source's chunking and embedding never call the SLM — waiving is"
    info " reasonable when those are all you are rebuilding; a relational source's semantic"
    info " layer is LLM-authored, so waiving there costs you model quality you cannot see.)"
    return 0
  fi

  # A relational source's LLM stage runs for hours (178 tables on this deployment); the
  # default per-source timeout is for the cheap kinds.
  if [ -n "$REL_IDS" ] && [ "$INGEST_TIMEOUT" = "7200" ]; then
    INGEST_TIMEOUT=28800
    info "relational source in scope — per-source timeout raised to ${INGEST_TIMEOUT}s"
  fi

  echo
  info "order: ${TARGET_IDS}   (datalake/nosql → relational → document)"
  [ -n "$DOC_IDS" ] && [ -n "$REL_IDS" ] && \
    info "documents ($DOC_IDS) run after the databases so entity_linker can match their chunks"
  info "each source is ingested with force=True; this reads the source systems and rewrites"
  info "their embeddings — it is the expensive step (timeout ${INGEST_TIMEOUT}s per source)."
  confirm "proceed with the re-ingestion?" || { warn "ingestion declined"; return 0; }

  local sid failed=""
  for sid in ${TARGET_IDS//,/ }; do
    [ "$RESUME_OK" = "1" ] || defeat_resume "$sid"
    ingest_one "$sid" || { bad "ingestion failed for source $sid"; failed="${failed}${sid},"; }
  done
  [ -n "$failed" ] && info "failed source(s): ${failed%,} — the enrich phase still runs for the rest"
  return 0
}

prior_failed_jobs() {  # $1 = source id -> FAILED jobs in its whole history
  # _should_resume() looks at the entire history, not the previous job: one failure years
  # ago still forces every later run into resume mode.
  psql_q "$PG_DB" "SELECT count(*) FROM ingestion_ingestionjob WHERE source_id=$1 AND status='failed'" \
    2>/dev/null | tr -d '[:space:]' || echo 0
}

defeat_resume() {  # $1 = source id — make L3/L4 actually run under VEDA_RESUME=1
  local sid="$1" root svc="$INFER" stamp
  [ "$(prior_failed_jobs "$sid")" -gt 0 ] || return 0
  svc_up "$svc" || svc="$API"
  root="$(art_root "$svc")"
  stamp="$(date +%Y%m%d-%H%M%S)"

  log "Source $sid has a prior failed job -> VEDA_RESUME=1; clearing the skip preconditions"
  info "under resume, L3 (the LLM semantic layer) and L4 (the biencoder) SKIP when their"
  info "output already exists — which would make this 'full rebuild' rebuild neither."
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m mv veda_semantic_model.json -> .bak-%s ; DELETE column_embeddings_v2 WHERE source_id=%s\n' "$stamp" "$sid"
    return 0
  fi
  [ -n "$root" ] && dc exec -T "$svc" sh -c '
      f="$1/$2/$3/veda_semantic_model.json"
      if [ -f "$f" ]; then mv "$f" "$f.bak-$4" && echo "    moved aside $f"; else echo "    no scoped model (L3 will run)"; fi
    ' sh "$root" "$TENANT" "$sid" "$stamp" </dev/null || true
  psql_q "$ENGINE_DB" "DELETE FROM column_embeddings_v2 WHERE source_id = '$sid'" >/dev/null 2>&1 \
    && ok "cleared source $sid's column embeddings — L4 will re-embed" \
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

  # Poll the job ROW, not the celery result: the row is what the platform and /admin treat
  # as authoritative, and it carries the per-stage progress.
  while [ "$waited" -lt "$INGEST_TIMEOUT" ]; do
    sleep 10; waited=$((waited + 10))
    jid="$(psql_q "$PG_DB" "SELECT id FROM ingestion_ingestionjob WHERE source_id=${sid} AND id > ${baseline} ORDER BY id DESC LIMIT 1" | tr -d '[:space:]')"
    [ -n "$jid" ] || { printf '\r    waiting for the worker to pick the job up (%ss)' "$waited"; continue; }
    status="$(psql_q "$PG_DB" "SELECT status FROM ingestion_ingestionjob WHERE id=${jid}" | tr -d '[:space:]')"
    stages="$(psql_q "$PG_DB" "SELECT string_agg(name || ':' || status, ' ' ORDER BY \"order\") FROM ingestion_ingestionstage WHERE job_id=${jid} AND status <> 'pending'" | tr -d '\r')"
    # \r is for a terminal. Redirected to a file it produces ONE enormous line that
    # needs sed to read back, so when stdout is not a tty print a real line, and only
    # every 5th poll so an 8-hour ingest does not write 2880 of them.
    if [ -t 1 ]; then
      printf '\r    job %s [%s] %-70.70s' "$jid" "$status" "${stages:-starting}"
    elif [ $((waited % 50)) -eq 0 ]; then
      printf '    [%ss] job %s [%s] %s\n' "$waited" "$jid" "$status" "${stages:-starting}"
    fi
    # The dispatcher path (document/datalake) emits no [[STAGE]] events, so the stage rows
    # stay `pending` and this line reads the same whether the job is working or wedged.
    # Every 5 minutes, check whether it is blocked on a lock and say so.
    if [ $((waited % 300)) -eq 0 ] && has_lock_blockers; then
      echo
      warn "source $sid is not progressing — something is blocked on $ENGINE_DB"
      check_lock_blockers "detail" || true
    fi
    case "$status" in
      success) echo; ok "source $sid ingested (job $jid)"; return 0 ;;
      failed)  echo
               bad "source $sid FAILED (job $jid)"
               info "    this failure now forces every LATER ingest of source $sid into resume mode;"
               info "    re-running this script clears that again (it moves the artifacts aside first)."
               psql_q "$PG_DB" "SELECT name, status, left(replace(error_traceback, chr(10), ' '), 300) FROM ingestion_ingestionstage WHERE job_id=${jid} AND status='failed'" | sed 's/^/      /'
               return 1 ;;
    esac
  done
  echo; bad "source $sid timed out after ${INGEST_TIMEOUT}s (job ${jid:-none}) — it may still be running"
  return 1
}

# =============================================================================
# enrich — everything task_ingest_source does not do for you
# =============================================================================
do_enrich() {
  log "Enrich — the passes that sit OUTSIDE the ingestion task"
  resolve_targets || return 0
  resolve_dbs || { bad "cannot resolve the databases"; return 0; }

  local esvc; esvc="$(engine_svc)"
  if [ -z "$esvc" ]; then bad "neither $WORKER nor $INFER is running — cannot run the engine-side backfills"; return 0; fi
  info "engine-side steps run in: $esvc     Django-side steps in: $API"

  # --- 1. cross-source sketches + links ---------------------------------------
  # The in-pipeline sketch pass inherits value_sampler's blind spot: it skips PK/FK/id
  # columns, which are precisely the join keys cross-source discovery needs. This one
  # samples them straight from the source, read-only.
  log "[1/8] Cross-source sketches + link discovery"
  # backfill_cross_source.py samples join keys from the SOURCE ITSELF, so it needs that
  # source's connection — which normally only exists because apps.ingestion.tasks injects
  # VEDA_SOURCE_* from the Source row. Invoked bare it dies with "VEDA_SOURCE_HOST is not
  # set" (2026-09-22). Inject per source, one at a time, then run discovery once at the end
  # (discovery reads the persisted sketches and needs no source connection).
  local sid
  for sid in ${TARGET_IDS//,/ }; do
    local senv
    senv="$(dc exec -T "$API" python - "$sid" <<'PY' 2>/dev/null || true
import sys, django
django.setup()
from apps.sources.models import Source
s = Source.objects.filter(pk=int(sys.argv[1])).first()
# Only relational/nosql sources have a connection to sample join keys from.
print(" ".join(f"-e {k}={v}" for k, v in (s.as_engine_env() if s and s.host else {}).items()))
PY
)"
    if [ -z "$senv" ]; then
      info "source $sid has no DB connection — skipping its sketch pass (nothing to sample)"
      continue
    fi
    run dc exec -T $senv "$esvc" sh -c \
      "cd /app/veda_core && python /app/scripts/backfill_cross_source.py --source-ids '$sid' --tenant '$TENANT' --sketch-only" || \
      warn "sketch pass failed for source $sid"
  done
  run dc exec -T "$esvc" sh -c \
    "cd /app/veda_core && python -c \"import sys; sys.path.insert(0,'/app/veda_core'); from ingestion.cross_source_graph import discover_and_persist; print('cross_source_fk:', discover_and_persist('$TENANT', source_ids=None, verbose=True))\"" || \
    warn "cross-source discovery failed — sources stay unlinked (no cross_source_fk edges)"

  # --- 2. the semantic bridge --------------------------------------------------
  # Re-run unconditionally even after a correctly ordered ingest: EMBED covers relational
  # sources whose column nodes never reached graph_node_embeddings, VALUES rebuilds the
  # Tier-B value index, and RELINK re-links the chunks that were embedded before the last
  # relational source finished. All three are idempotent.
  log "[2/8] Semantic bridge — column embeddings, value index, chunk relink"
  run dc exec -T "$esvc" sh -c \
    "cd /app/veda_core && python /app/scripts/backfill_semantic_bridge.py --phase all --tenant '$TENANT'" || \
    warn "semantic bridge backfill failed — chunks keep only their exact value_of links"

  # --- 3. routing cards --------------------------------------------------------
  # The card is an L5 stage, so only the relational path emits one; every document and
  # datalake source needs this. Pure transform — no LLM, no re-scan.
  log "[3/8] Routing cards"
  local rc_flags=""
  [ "$WITH_ROW_COUNTS" = "1" ] && rc_flags="--with-row-counts"
  run dc exec -T "$esvc" sh -c \
    "cd /app/veda_core && python /app/scripts/backfill_routing_cards.py --tenant '$TENANT' $rc_flags" || \
    warn "routing card backfill failed — those sources route the way they did before cards existed"

  # --- 4. per-source semantic model for the non-relational sources -------------
  # warm() builds a lite model for these from their own graph columns, so this is a REPAIR,
  # not a routine step: it runs only when verify's bleed check found a source publishing a
  # model that describes another source's tables.
  local bleeding; bleeding="$(detect_sm_bleed)"
  if [ -n "$bleeding" ]; then
    log "[4/8] Rebuilding the per-source semantic model of: $bleeding (bleed detected)"
    # PYTHONPATH=/app: unlike its siblings, this script does not insert the repo root itself,
    # and `python scripts/x.py` puts /app/scripts on sys.path — not /app — so the bare
    # invocation dies on `from storage_adapters import assembler`.
    run dc exec -T -e PYTHONPATH=/app "$API" python scripts/backfill_semantic_model.py \
        --source-ids "$bleeding" --tenant "$TENANT" || \
      warn "semantic model backfill failed for $bleeding"
  else
    log "[4/8] Per-source semantic models — every source publishes its own tables, nothing to repair"
  fi

  # --- 5. source descriptions --------------------------------------------------
  # SOURCE_PROFILER_ENABLED defaults to 0 and is not in this repo's .env, so the post-ingest
  # hook did nothing. The description is a routing input (SOURCE_DESC_PRIOR_ENABLED=1), and a
  # blank one just quietly weakens routing. profile_source() is manual-wins: a human-written
  # description is never overwritten.
  log "[5/8] Source descriptions (the routing prior reads these)"
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m profile_source() for %s\n' "$TARGET_IDS"
  else
    dc exec -T -e VEDA_TARGET_IDS="$TARGET_IDS" -e VEDA_TENANT="$TENANT" "$API" python - <<'PY' || \
      warn "source profiling failed"
import os, django
django.setup()
from apps.sources.source_profiler import profile_source

tenant = os.environ.get("VEDA_TENANT", "default")
for sid in [s for s in os.environ["VEDA_TARGET_IDS"].split(",") if s]:
    r = profile_source(int(sid), tenant=tenant)
    print(f"    source {sid}: {r.reason} — {(r.description or '')[:110]}")
PY
  fi

  # --- 6. the uniform item layer ----------------------------------------------
  # SOURCE_ITEM_PROFILER_ENABLED also defaults to 0, so SourceItem is empty on a by-the-book
  # deployment and the per-item routing prior is never primed at all. Needs the SLM chat
  # endpoint and the Metal embed endpoint — both probed in preflight.
  log "[6/8] SourceItem layer — structural rows + SLM summary + routing embedding"
  local sid item_flags=""
  [ "$DO_RESET" = "1" ] && item_flags="--force"
  for sid in ${TARGET_IDS//,/ }; do
    run dc exec -T "$API" python manage.py build_source_items --source "$sid" $item_flags || \
      warn "source item build failed for source $sid"
  done

  # --- 7. the RBAC catalog projection ------------------------------------------
  # CatalogResource has no foreign key to the substrate and is recreated per ingestion.
  # Between "substrate rebuilt" and "catalog re-synced" every resource of that source is
  # ABSENT — and absent means DENIED. With VEDA_AUTO_SYNC_CATALOG=0 nothing closes that gap
  # but this command.
  log "[7/8] RBAC catalog projection"
  run dc exec -T "$API" python manage.py sync_catalog || \
    bad "catalog sync failed — every RBAC-checked resource of these sources reads as DENIED"

  # --- 8. re-warm the query tier ------------------------------------------------
  # Steps 1-4 rewrote artifacts that the query tier holds in Redis from ingestion time.
  log "[8/8] Re-warming the query tier (persist + publish + rehydrate fan-out)"
  if [ "$DRY_RUN" = "1" ]; then
    printf '  \033[0;35m[dry-run]\033[0m task_warm_caches(source_id=…, tenant=%s) for %s\n' "$TENANT" "$TARGET_IDS"
  else
    for sid in ${TARGET_IDS//,/ }; do
      dc exec -T "$API" python manage.py shell -c \
        "from apps.ingestion.tasks import task_warm_caches; print('    source $sid warm:', task_warm_caches(source_id=$sid, tenant='$TENANT'))" \
        </dev/null || warn "warm failed for source $sid"
    done
  fi

  # --- optional: SLM synonym enrichment ----------------------------------------
  if [ "$WITH_SYNONYMS" = "1" ]; then
    log "[opt] Business-vocabulary synonyms (SLM pass over entities + measures)"
    info "resumable: every generated item is cached, so a killed run continues rather than restarts."
    run dc exec -T "$INFER" sh -c \
      "cd /app/veda_core && python3 /app/scripts/enrich_synonyms.py" || \
      warn "synonym enrichment failed"
    info "re-run this script with --phases enrich to republish the recompiled registries"
  else
    note "synonym enrichment skipped (--with-synonyms enables it; it is an hours-long SLM pass)"
  fi
}

check_substrate_ownership() {
  # storage_adapters/writer.py reads the GLOBAL fk_adjacency/column_values, so before the
  # 2026-09-23 fix a source's warm() filed the last relational source's whole schema under
  # ITSELF. Nothing raised: docs_contracts (a filesystem source) silently owned 116 homzhub
  # tables for hours, and the collision only surfaced when the real owner tried to sync and
  # hit a duplicate substrate PK. Compare each source's substrate tables against the tables
  # it actually owns in the engine store.
  log "Substrate ownership (each source's tables must be its own)"
  local rows sid n_sub n_own foreign
  rows="$(psql_q "$PG_DB" "SELECT id FROM sources_source ORDER BY id" 2>/dev/null || true)"
  while IFS='|' read -r sid <&4; do
    [ -n "$sid" ] || continue
    n_sub="$(psql_q "$PG_DB" "SELECT count(*) FROM substrate_schematable WHERE source_id=$sid" 2>/dev/null | tr -d '[:space:]' || echo 0)"
    n_own="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM graph_nodes WHERE source_id='$sid' AND node_type='table'" 2>/dev/null | tr -d '[:space:]' || echo 0)"
    [ "${n_sub:-0}" = "0" ] && [ "${n_own:-0}" = "0" ] && continue
    if [ "${n_sub:-0}" -gt "${n_own:-0}" ]; then
      foreign=$(( n_sub - n_own ))
      bad "source $sid: $n_sub substrate table(s) but it owns only $n_own — $foreign belong to another source"
      info "    that is the warm() global-table bleed; re-warm it after clearing the foreign rows."
    else
      ok "source $sid: $n_sub substrate table(s), owns $n_own"
    fi
  done 4<<< "$rows"
}

detect_sm_bleed() {  # -> csv of non-relational sources publishing a foreign model
  local bleeding="" id n_sm n_nodes
  svc_up "$INFER" || { printf ''; return 0; }
  for id in ${LAKE_IDS//,/ } ${NOSQL_IDS//,/ } ${DOC_IDS//,/ }; do
    [ -n "$id" ] || continue
    n_sm="$(dc exec -T -w /app/veda_core -e PYTHONPATH=/app:/app/veda_core "$INFER" python -c "
from veda_hybrid import _load_sm_from_redis
sm = _load_sm_from_redis(scope=($id, '$TENANT'))
print(len({k.split('.')[0] for k in (sm or {}).get('columns', {})}))" </dev/null 2>/dev/null | tr -d '[:space:]' || echo "")"
    n_nodes="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM graph_nodes WHERE source_id='$id' AND node_type='table'" 2>/dev/null | tr -d '[:space:]' || echo "")"
    [ -n "$n_sm" ] && [ -n "$n_nodes" ] && [ "${n_nodes:-0}" -gt 0 ] && [ "$n_sm" != "$n_nodes" ] \
      && bleeding="${bleeding}${id},"
  done
  printf '%s' "${bleeding%,}"
}

# =============================================================================
# verify
# =============================================================================
do_verify() {
  log "Verify"
  resolve_targets || return 0
  resolve_dbs || { bad "cannot resolve the databases"; return 0; }

  if dc exec -T "$API" python manage.py migrate --check >/dev/null 2>&1; then
    ok "all migrations applied"
  else
    bad "migrations pending"
  fi

  local ready
  ready="$(dc exec -T "$API" python -c \
    "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/readyz', timeout=15).read().decode()[:400])" </dev/null 2>&1 || true)"
  if printf '%s' "$ready" | grep -q '"status": *"ready"'; then ok "/readyz: ready"; else bad "/readyz is not ready"; info "$ready"; fi

  # --- the matrix ---------------------------------------------------------------
  log "Per-source store matrix (tenant=$TENANT)"
  printf '  %-4s %-16s %-11s %-7s %-7s %-7s %-7s %-6s %-6s %-6s %-5s %s\n' \
         id name kind cols sparse chunks nodes nodeEmb sketch items cat art
  printf '  %s\n' "---------------------------------------------------------------------------------------------------------"

  local probe_svc="$INFER"; svc_up "$probe_svc" || probe_svc="$API"
  local rows id dialect name kind cols sparse chunks nodes nemb sketch items cat art card
  rows="$(psql_q "$PG_DB" "SELECT id, dialect, name FROM sources_source WHERE id IN ($TARGET_IDS) ORDER BY id" 2>/dev/null || true)"
  while IFS='|' read -r id dialect name <&3; do
    [ -n "$id" ] || continue
    kind="$(kind_of "$dialect")"
    cols="$(count_engine column_embeddings_v2 "$id")"
    # The learned-sparse index lives in a different table per kind: a document source's
    # sparse weights are in chunk_sparse_v1, never column_sparse_v1. Reading the column one
    # for a document reports 0 and looks like a missing index that was never meant to exist.
    if [ "$kind" = "document" ]; then
      sparse="$(count_engine chunk_sparse_v1 "$id")"
    else
      sparse="$(count_engine column_sparse_v1 "$id")"
    fi
    chunks="$(count_engine doc_chunks "$id")"
    nodes="$(count_engine graph_nodes "$id")"
    nemb="$(count_engine graph_node_embeddings "$id")"
    sketch="$(count_engine column_sketches "$id")"
    items="$(psql_q "$PG_DB" "SELECT count(*) FROM sources_sourceitem WHERE source_id=$id" 2>/dev/null | tr -d '[:space:]' || echo '?')"
    cat="$(psql_q "$PG_DB" "SELECT count(*) FROM access_management_catalogresource WHERE source_id=$id AND is_active" 2>/dev/null | tr -d '[:space:]' || echo '?')"
    art="-"; has_artifact "$probe_svc" "$id" veda_semantic_model.json && art="sm"
    card="";  has_artifact "$probe_svc" "$id" veda_routing_card.json && card="+card"
    printf '  %-4s %-16s %-11s %-7s %-7s %-7s %-7s %-6s %-6s %-6s %-5s %s\n' \
           "$id" "${name:0:16}" "$kind" "$cols" "$sparse" "$chunks" "$nodes" "$nemb" "$sketch" "$items" "$cat" "${art}${card}"

    # Each of these is a store that answers questions. Empty is never "fine" — it is a
    # signal that answers relying on it will come back thin, with nothing raised anywhere.
    case "$kind" in
      relational|datalake|nosql)
        [ "${cols:-0}" = "0" ]   && bad "source $id ($name): no column embeddings — retrieval Signal 1 is dead for it"
        [ "${nodes:-0}" = "0" ]  && bad "source $id ($name): no graph nodes — nothing to plan a join over"
        [ "${nemb:-0}" = "0" ]   && warn "source $id ($name): no graph node embeddings — PPR seeding and the semantic bridge cannot match it"
        [ "${sketch:-0}" = "0" ] && warn "source $id ($name): no column sketches — it can never be cross-source linked"
        [ "$kind" = "relational" ] && [ "$art" = "-" ] && \
          bad "source $id ($name): no scoped semantic model — resolve_source_artifact() has no flat fallback, so it plans SQL with an EMPTY model"
        ;;
      document)
        [ "${chunks:-0}" = "0" ] && bad "source $id ($name): no doc_chunks — the document source holds nothing"
        [ "${sparse:-0}" = "0" ] && warn "source $id ($name): no chunk sparse vectors — retrieval is dense-only for it"
        ;;
    esac
    [ -z "$card" ] && warn "source $id ($name): no routing card — the router infers what this source is from whichever columns a query happens to hit"
    [ "${items:-0}" = "0" ] && warn "source $id ($name): no SourceItem rows — the per-item routing prior is unprimed"
    [ "${cat:-0}" = "0" ] && bad "source $id ($name): no active catalog resources — every RBAC check against it reads as DENIED"
  done 3<<< "$rows"

  # --- cross-source + bridge ----------------------------------------------------
  local xedges bedges vedges
  xedges="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM graph_edges WHERE edge_type='cross_source_fk'" 2>/dev/null | tr -d '[:space:]' || echo '?')"
  bedges="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM graph_edges WHERE edge_type='semantic_about'" 2>/dev/null | tr -d '[:space:]' || echo '?')"
  vedges="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM graph_edges WHERE edge_type IN ('value_of','semantic_value_of')" 2>/dev/null | tr -d '[:space:]' || echo '?')"
  log "Cross-source + bridge"
  info "cross_source_fk edges : $xedges"
  info "semantic_about edges  : $bedges   (chunk → column, the fuzzy lane)"
  info "value_of edges        : $vedges   (chunk → column, exact + semantic value)"
  local n_sources=0 _i
  for _i in ${TARGET_IDS//,/ }; do n_sources=$((n_sources + 1)); done
  [ "$n_sources" -gt 1 ] && [ "${xedges:-0}" = "0" ] && \
    warn "more than one source but zero cross_source_fk edges — federated questions cannot join across them"
  [ -n "$DOC_IDS" ] && [ "${bedges:-0}" = "0" ] && \
    warn "document sources present but zero semantic_about edges — the bridge is a no-op (was a structured source embedded?)"

  # The bridge's safety invariant: a semantic edge must never be column↔column, or it could
  # authorise a SQL join between two sources on a fuzzy match.
  local unsafe
  unsafe="$(psql_q "$ENGINE_DB" "
      SELECT count(*) FROM graph_edges e
       JOIN graph_nodes a ON a.node_id = e.src_node_id
       JOIN graph_nodes b ON b.node_id = e.dst_node_id
      WHERE e.edge_type IN ('semantic_about','semantic_value_of')
        AND a.node_type='column' AND b.node_type='column'" 2>/dev/null | tr -d '[:space:]' || echo '')"
  if [ "${unsafe:-0}" = "0" ]; then
    ok "bridge safety invariant holds — no semantic edge is column↔column, so none can drive a join"
  elif [ -n "$unsafe" ]; then
    bad "$unsafe semantic edge(s) are column↔column — a fuzzy match could authorise a real SQL join"
  fi

  # --- the global store's owner --------------------------------------------------
  local cv; cv="$(psql_q "$ENGINE_DB" "SELECT count(*) FROM column_values" 2>/dev/null | tr -d '[:space:]' || echo '?')"
  info "column_values holds $cv row(s) — global, owned by the last relational source to finish (${REL_IDS##*,})"

  check_substrate_ownership

  local bleeding; bleeding="$(detect_sm_bleed)"
  if [ -n "$bleeding" ]; then
    bad "semantic-model bleed: source(s) $bleeding publish a model that is not their own schema"
    info "    $DC exec -e PYTHONPATH=/app $API python scripts/backfill_semantic_model.py --source-ids $bleeding --tenant $TENANT"
  else
    ok "every source publishes a semantic model of its own tables"
  fi

  log "Still yours to do by hand"
  info "1. one real question per source, checking each answers from its OWN data"
  info "2. the permission cases in MULTI_SOURCE_DEPLOYMENT.md §5 — the deny case must refuse"
  info "   WITHOUT naming the source it is hiding"
  info "3. the same question against /api/v1/query AND the streaming endpoint (the request"
  info "   context is carried separately there, and is lost separately too)"
}

count_engine() {  # $1 = table, $2 = source id -> row count, or "?" if the table is absent
  psql_q "$ENGINE_DB" "SELECT count(*) FROM $1 WHERE source_id = '$2'" 2>/dev/null | tr -d '[:space:]' || echo '?'
}

# =============================================================================
# main
# =============================================================================
printf '\033[1m VEDA — seed + full re-ingestion\033[0m\n'
printf ' compose  : %s\n manifest : %s\n phases   : %s\n tenant   : %s%s\n' \
       "$DC" "$MANIFEST" "$PHASES" "$TENANT" "$([ "$DRY_RUN" = 1 ] && echo '   (DRY RUN)')"

if [ "$EXPORT_MODE" = "1" ]; then
  do_export
  exit $?
fi

want_phase preflight && preflight

# Every mutating phase needs the target list the preflight builds.
if ! want_phase preflight; then resolve_targets || true; fi

if [ ${#BLOCKERS[@]} -gt 0 ]; then
  log "Blocked — ${#BLOCKERS[@]} issue(s) must be resolved before anything is rebuilt"
  printf '  \033[1;31m✗\033[0m %s\n' "${BLOCKERS[@]}"
  exit 1
fi

want_phase seed    && do_seed
want_phase reset   && do_reset
want_phase ingest  && do_ingest
want_phase enrich  && do_enrich
want_phase verify  && do_verify

# ------------------------------------------------------------------- summary
log "Summary"
[ -n "$TARGET_IDS" ] && info "sources: $TARGET_IDS (order: datalake/nosql → relational → document)"
if [ "${#NOTES[@]}" -gt 0 ]; then
  printf '%s\n' "${NOTES[@]}" | sed 's/^/    · /'
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
