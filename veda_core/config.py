# =============================================================================
# config.py
# VEDA — Universal Natural Language Data Platform
# Central Configuration — all parameters live here. Nothing hardcoded elsewhere.
#
# ENCODER_MODE switch:
#   "relgt_only"  — RELGT 256-dim structural only        (POC run 1)
#   "light_text"  — TF-IDF + SVD 256-dim                 (POC run 2)
#   "hybrid"      — MiniLM 384 + RELGT 256 = 640-dim     (POC run 3)
#   "ensemble"    — Light Text + Hybrid dual-store + RRF  (POC run 4, default)
# =============================================================================

# -----------------------------------------------------------------------------
# *** EXPERIMENT SWITCH — only line you need to change between encoder runs ***
# -----------------------------------------------------------------------------
ENCODER_MODE = "ensemble"   # "relgt_only" | "light_text" | "hybrid" | "ensemble"

# =============================================================================
# SOURCE REGISTRY
#
# VEDA_SOURCES — list of all client data sources VEDA should ingest and query.
# Each source is a dict describing one connected data system.
# The ingestion dispatcher (ingestion/source_dispatcher.py) reads this list
# and routes each source to the correct connector.
#
# Source types:
#   "relational" — PostgreSQL, MySQL, SQLite, Oracle, SQL Server
#   "document"   — PDF, Word, HTML, TXT, Markdown files
#   "datalake"   — Delta Lake, Parquet, CSV, Iceberg
#   "nosql"      — MongoDB, Elasticsearch, DynamoDB, Cassandra
#
# Each source MUST have:
#   id    — unique string identifier used throughout the pipeline
#   type  — one of the four source types above
#   role  — "queryable" (SQL/query generation) | "searchable" (RAG/vector search)
#
# Add as many sources as needed. The pipeline processes all enabled sources.
# =============================================================================

VEDA_SOURCES = [
    # ------------------------------------------------------------------
    # Primary relational database (client's main DB)
    # ------------------------------------------------------------------
    {
        "id":       "primary_db",
        "type":     "relational",
        "enabled":  True,
        "engine":   "postgresql",      # postgresql | mysql | sqlite | oracle | sqlserver
        # Env-overridable (§9) so containers reach the launchpad DB via
        # host.docker.internal:5433 while a bare-metal run keeps localhost.
        "host":     __import__("os").environ.get("VEDA_SOURCE_HOST", "localhost"),
        "port":     int(__import__("os").environ.get("VEDA_SOURCE_PORT", "5433")),
        "dbname":   __import__("os").environ.get("VEDA_SOURCE_DBNAME", "launchpad"),
        "user":     __import__("os").environ.get("VEDA_SOURCE_USER", "postgres"),
        "password": __import__("os").environ.get("VEDA_SOURCE_PASSWORD", "admin"),
        "role":     "queryable",       # generates SQL against this source
        # Optional: schema to restrict scanning to (None = all public schemas)
        "schema":   None,
        # Optional: tables to exclude from ingestion (in addition to VEDA internal)
        "exclude_tables": [
            "auth_group",
            "auth_group_permissions",
            "auth_permission",
            "django_admin_log",
            "django_content_type",
            "django_migrations",
            "django_session",
            "django_celery_beat_clockedschedule",
            "django_celery_beat_crontabschedule",
            "django_celery_beat_intervalschedule",
            "django_celery_beat_periodictask",
            "django_celery_beat_periodictasks",
            "django_celery_beat_solarschedule",
            "celery_task",
            "audit_log",
            "job_execution_log",
            "task_execution_log",
            "document_ingestion_audit_log",
            "table_ingestion_audit_log",
            "sampling_history",
            "workflow_history",
            "identity_management_historicaluser",
            "risk_scoring_historicalsignalrule",
            "counterparty_merge_history",
            "pipeline_state",
            "entity_cfg",
            "entity_schedule_cfg",
            "entity_schedule_status",
            "entity_source_cfg",
            "source_ocs_column_map_cfg",
            "source_type",
            "tenant_config",
            "tenant_connection_config",
            "tenant_data_config",
            "tenant_source_config",
            "sync_item",
            "sync_status",
            "table_tracker",
            "ml_config",
            "model_registry",
            "prompt_registry",
            "message_config",
            "notification_property",
            "password_attempt_log",
            "common_password",
            "mfa_delivery_method",
            "mfa_provider",
            "user_login_session",
            "user_mfa_settings",
            "user_mfa_settings_delivery_methods",
            "dashboard_available_filters",
            "dashboard_global_filters",
            "dashboard_item_filters",
            "dashboard_item_layouts",
            "dashboard_item_permissions",
            "document_chunks",
            "agent_registry",
            "agent_message_documents",
            "ocs_agent",
            "ocs_agent_message",
            "notification_preference",
            "notification_template",
            "scheduled_notification",
            "subsupervisor_registry",
            "incident_processing_status",
            "workflow_group"
        ],
    },

    # ------------------------------------------------------------------
    # Document store example (PDF/Word/HTML files)
    # Uncomment and configure to enable document RAG
    # ------------------------------------------------------------------
    # {
    #     "id":      "contracts",
    #     "type":    "document",
    #     "enabled": False,
    #     "engine":  "filesystem",    # filesystem | s3 | azure_blob | gcs
    #     "path":    "/data/contracts",
    #     "formats": ["pdf", "docx", "txt", "html", "md"],
    #     "role":    "searchable",    # chunk retrieval + LLM synthesis
    #     # Optional: recursive directory scanning
    #     "recursive": True,
    #     # Optional: file size limit in MB
    #     "max_file_mb": 50,
    # },
    {
        "id":      "dmt",
        "type":    "document",
        "enabled": False,           # disabled: stale veda-poc CSV doc source, not part of homzhub
        "engine":  "filesystem",    # filesystem | s3 | azure_blob | gcs
        "path":    "",              # was /Users/ekesel/samta/veda-poc — removed (self-contained)
        "formats": ["csv"],
        "role":    "searchable",    # chunk retrieval + LLM synthesis
        # Optional: recursive directory scanning
        "recursive": False,
        # Optional: file size limit in MB
        "max_file_mb": 50,
    },

    # ------------------------------------------------------------------
    # Data lake example (Delta / Parquet)
    # Uncomment and configure to enable data lake querying
    # ------------------------------------------------------------------
    # {
    #     "id":      "analytics_lake",
    #     "type":    "datalake",
    #     "enabled": True,
    #     "engine":  "csv",           # delta | parquet | csv | iceberg
    #     "path":    "/Users/ekesel/samta/veda-poc",
    #     "role":    "queryable",     # generates DuckDB/Spark SQL against this source
    #     # Optional: AWS/GCS credentials
    #     "aws_access_key": None,
    #     "aws_secret_key": None,
    #     "aws_region":     "us-east-1",
    # },

    # ------------------------------------------------------------------
    # NoSQL example (MongoDB)
    # Uncomment and configure to enable NoSQL querying
    # ------------------------------------------------------------------
    # {
    #     "id":      "events_db",
    #     "type":    "nosql",
    #     "enabled": False,
    #     "engine":  "mongodb",       # mongodb | elasticsearch | dynamodb | cassandra
    #     "host":    "localhost",
    #     "port":    27017,
    #     "dbname":  "events",
    #     "role":    "queryable",     # generates native MongoDB query
    #     "user":    None,
    #     "password": None,
    # },
]


# =============================================================================
# VEDA INTERNAL DATABASE
#
# VEDA's own pgvector index — always PostgreSQL.
# Completely separate from VEDA_SOURCES.
# Stores: column embeddings, FK adjacency, table metadata,
#         column values, document chunks, source registry.
#
# The client's data NEVER flows into this database.
# This database is VEDA's internal index only.
# =============================================================================
import os as _os_env  # env overrides for containerized deploy (migration_plan.md §9)

VEDA_INTERNAL_DB = {
    "host":     _os_env.environ.get("VEDA_INTERNAL_HOST", "localhost"),
    "port":     int(_os_env.environ.get("VEDA_INTERNAL_PORT", "5433")),
    "dbname":   _os_env.environ.get("VEDA_INTERNAL_DBNAME", "veda"),  # embeddings + v2 tables
    "user":     _os_env.environ.get("VEDA_INTERNAL_USER", "postgres"),
    "password": _os_env.environ.get("VEDA_INTERNAL_PASSWORD", "admin"),
}


# =============================================================================
# BACKWARD COMPATIBILITY SHIM
#
# DB_CONFIG is kept for all existing files that reference it directly.
#
# Rule:
#   DB_CONFIG  → VEDA_INTERNAL_DB  for vector_store, data_graph, value_sampler
#   Use VEDA_SOURCES[n] for client DB connections via the DAL
# =============================================================================
DB_CONFIG = VEDA_INTERNAL_DB   # shim — points to internal DB for existing code


# =============================================================================
# SOURCE HELPERS — used by ingestion/source_dispatcher.py
# =============================================================================

def get_source(source_id: str) -> dict:
    """Returns the source config dict for the given source_id. Raises if not found."""
    for src in VEDA_SOURCES:
        if src["id"] == source_id:
            return src
    raise KeyError(f"No source with id='{source_id}' found in VEDA_SOURCES")


def get_enabled_sources(source_type: str = None) -> list:
    """
    Returns all enabled sources, optionally filtered by type.
    source_type: "relational" | "document" | "datalake" | "nosql" | None (all)
    """
    sources = [s for s in VEDA_SOURCES if s.get("enabled", True)]
    if source_type:
        sources = [s for s in sources if s["type"] == source_type]
    return sources


def get_primary_relational_source() -> dict:
    """
    Returns the first enabled relational source.
    """
    sources = get_enabled_sources("relational")
    if not sources:
        raise ValueError("No enabled relational source found in VEDA_SOURCES")
    return sources[0]


# =============================================================================
# VEDA INTERNAL TABLE NAMES
# All tables VEDA creates in VEDA_INTERNAL_DB.
# Never scan these as part of the client schema.
# =============================================================================
BIENCODER_COL_TABLE   = "column_embeddings_v2"
BIENCODER_TABLE_TABLE = "table_embeddings_v2"

VEDA_INTERNAL_TABLES = {
    # Embedding stores
    "column_embeddings",
    "column_embeddings_lt",
    "column_embeddings_hybrid",
    # V2 retrieval stores
    BIENCODER_COL_TABLE,
    BIENCODER_TABLE_TABLE,
    # Metadata stores
    "fk_adjacency",
    "table_metadata",
    "column_values",
    # Document RAG store
    "doc_chunks",
    # Source registry
    "source_registry",
    # Unified data graph
    "graph_nodes", "graph_edges", "graph_node_embeddings", "graph_node_embeddings_gnn",
}

# -----------------------------------------------------------------------------
# Vector store table names (unchanged)
# -----------------------------------------------------------------------------
VECTOR_TABLE_NAME            = "column_embeddings"
VECTOR_TABLE_NAME_LIGHT_TEXT = "column_embeddings_lt"
VECTOR_TABLE_NAME_HYBRID     = "column_embeddings_hybrid"

# Delete stale rows scoped to source_id before re-inserting (idempotent re-ingestion)
VECTOR_STORE_TRUNCATE_ON_INGEST = True

# -----------------------------------------------------------------------------
# FK Adjacency Store
# -----------------------------------------------------------------------------
FK_ADJACENCY_TABLE_NAME    = "fk_adjacency"
FK_MAX_HOP_DEPTH           = 2
FK_BRIDGE_INJECTION_ENABLED = True
FK_MAX_INJECTED_COLS       = 5

# -----------------------------------------------------------------------------
# Embedding dimensions — driven by ENCODER_MODE
# -----------------------------------------------------------------------------
RELGT_EMBEDDING_DIM      = 256
LIGHT_TEXT_EMBEDDING_DIM = 256
MINILM_EMBEDDING_DIM     = 384
HYBRID_EMBEDDING_DIM     = MINILM_EMBEDDING_DIM + RELGT_EMBEDDING_DIM  # 640

_ENCODER_DIM_MAP = {
    "relgt_only": RELGT_EMBEDDING_DIM,
    "light_text": LIGHT_TEXT_EMBEDDING_DIM,
    "hybrid":     HYBRID_EMBEDDING_DIM,
    "ensemble":   LIGHT_TEXT_EMBEDDING_DIM,
}
VECTOR_DIM = _ENCODER_DIM_MAP[ENCODER_MODE]

# -----------------------------------------------------------------------------
# Semantic type inference
# -----------------------------------------------------------------------------
SEMANTIC_TYPES = [
    "MONETARY", "TEMPORAL", "CATEGORY",
    "IDENTIFIER", "METRIC", "FREE_TEXT",
]
SEMANTIC_CONFIDENCE_THRESHOLD = 0.75
MONETARY_KEYWORDS   = ["price", "amount", "revenue", "cost", "rent", "fee", "value", "paid", "earning"]
METRIC_KEYWORDS     = ["count", "quantity", "score", "area", "size", "sqft", "period", "number", "floor"]
IDENTIFIER_SUFFIXES = ["_id", "_uuid", "_key", "_no", "_number", "_num", "_code", "_ref"]
SENSITIVE_PATTERNS  = [
    "password", "passwd", "secret", "token", "api_key", "private_key",
    "ssn", "aadhar", "pan_number", "credit_card", "cvv", "otp",
    "salt", "hash",
]

# -----------------------------------------------------------------------------
# REG Builder
# -----------------------------------------------------------------------------
TABLE_NAME_EMBED_DIM  = 64
MAX_TABLES            = 500
NUM_TABLES            = MAX_TABLES   # alias used by schema/simulate_schema.py

# -----------------------------------------------------------------------------
# RELGT Encoder
# -----------------------------------------------------------------------------
RELGT_HIDDEN_DIM    = 128
RELGT_NUM_LAYERS    = 3
RELGT_OUTPUT_DIM    = RELGT_EMBEDDING_DIM
RELGT_FK_EDGE_WEIGHT = 3.0   # FK cross-table edges weighted 3x sibling edges before normalisation


# -----------------------------------------------------------------------------
# Generic compute-device resolution (no hardcoding — works on CUDA / Apple MPS / CPU)
# -----------------------------------------------------------------------------
def resolve_device() -> str:
    """Pick the inference device at RUNTIME, portably. Order: explicit VEDA_DEVICE env
    override → CUDA → Apple MPS → CPU. Returns 'cpu' if torch is absent (thin api image)."""
    import os as _o
    forced = _o.environ.get("VEDA_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch as _t
        if _t.cuda.is_available():
            return "cuda"
        if getattr(_t.backends, "mps", None) is not None and _t.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def available_memory_gb() -> float:
    """Best-effort available device/host memory in GB, read at runtime (never hardcoded).
    CUDA → free VRAM; MPS/CPU → host RAM. Falls back to a conservative 4.0 GB."""
    import os as _o
    dev = resolve_device()
    try:
        import torch as _t
        if dev == "cuda":
            free, _total = _t.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    try:  # host RAM (Linux/macOS) without adding a dependency
        page = _o.sysconf("SC_PAGE_SIZE")
        avail = _o.sysconf("SC_AVPHYS_PAGES")
        if page > 0 and avail > 0:
            return (page * avail) / (1024 ** 3)
    except Exception:
        pass
    return 4.0


def scaled_batch_size(base: int, per_gb: int = 8, cap: int = 256) -> int:
    """Scale a base batch size by a FRACTION of available memory — bigger box → bigger
    batches, small box → base. Never references a literal machine size (§P8-A)."""
    import os as _o
    mem = available_memory_gb()
    scaled = int(base * max(1.0, (mem * 0.5) / per_gb))
    return max(base, min(scaled, cap)) if _o.environ.get("VEDA_SCALE_BATCH", "1") == "1" else base


# The resolved device drives every encoder below; each stays env-overridable via VEDA_DEVICE.
_RESOLVED_DEVICE = resolve_device()

# -----------------------------------------------------------------------------
# MiniLM Encoder
# -----------------------------------------------------------------------------
MINILM_MODEL_NAME         = "all-MiniLM-L6-v2"
MINILM_SENTENCE_TEMPLATE  = "{col_name} {table_name} {semantic_type}"
MINILM_BATCH_SIZE         = 64
MINILM_DEVICE             = _RESOLVED_DEVICE

# -----------------------------------------------------------------------------
# Light Text Encoder
# -----------------------------------------------------------------------------
LIGHT_TEXT_SENTENCE_TEMPLATE  = "{col_name} {table_name} {semantic_type}"
LIGHT_TEXT_TFIDF_MAX_FEATURES = 512
LIGHT_TEXT_TFIDF_NGRAM_RANGE  = (1, 2)
LIGHT_TEXT_SVD_COMPONENTS     = LIGHT_TEXT_EMBEDDING_DIM
LIGHT_TEXT_CHAR_SPLIT         = True

# -----------------------------------------------------------------------------
# Ensemble Retrieval (RRF)
# -----------------------------------------------------------------------------
ENSEMBLE_RRF_K                = 60
ENSEMBLE_LIGHT_TEXT_WEIGHT    = 1.0
ENSEMBLE_HYBRID_WEIGHT        = 1.8
ENSEMBLE_CANDIDATES_PER_STORE = 60

# -----------------------------------------------------------------------------
# Query pipeline
# -----------------------------------------------------------------------------
TOP_K      = 15
NUM_FK_RELATIONS = 10

# -----------------------------------------------------------------------------
# Graph store — persisted REG graph for query-time subgraph RELGT
# -----------------------------------------------------------------------------
REG_GRAPH_PATH  = "schema/reg_graph.pkl"
COL_ID_IDX_PATH = "schema/col_id_to_idx.pkl"

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
_ENCODER_LABEL_MAP = {
    "relgt_only": "POC Run 1 — RELGT structural only (256-dim)",
    "light_text": "POC Run 2 — Light Text TF-IDF+SVD (256-dim)",
    "hybrid":     "POC Run 3 — MiniLM + RELGT Hybrid (640-dim)",
    "ensemble":   "POC Run 4 — Ensemble Light Text + Hybrid + RRF",
}
POC_LABEL      = _ENCODER_LABEL_MAP[ENCODER_MODE]

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_VALID_ENCODER_MODES = {"relgt_only", "light_text", "hybrid", "ensemble"}
if ENCODER_MODE not in _VALID_ENCODER_MODES:
    raise ValueError(
        f"config.py: ENCODER_MODE='{ENCODER_MODE}' is invalid. "
        f"Must be one of: {_VALID_ENCODER_MODES}"
    )

# -----------------------------------------------------------------------------
# NL Simplifier — Layer 0
# -----------------------------------------------------------------------------
NL_SIMPLIFIER_ENABLED = True

# -----------------------------------------------------------------------------
# SLM — Layer 3
# -----------------------------------------------------------------------------
SLM_MODEL_NAME       = "qwen2.5-coder:7b"
SLM_OLLAMA_BASE_URL  = __import__("os").environ.get("OLLAMA_URL", "http://localhost:11434")  # env-overridable (§9): container reaches ollama:11434
SLM_TEMPERATURE      = 0.3
SLM_TIMEOUT_SECS     = 240
SLM_MAX_RETRIES      = 2
SLM_MAX_TOKENS       = 2048
# IR-JSON output is small (~150–500 tokens); a dedicated, smaller cap than the shared
# SLM_MAX_TOKENS (which RAG synthesis needs for prose) shortens decode → lower latency
# on the hybrid/IR path without starving RAG answers.
SLM_IR_MAX_TOKENS    = 512
SLM_NUM_CTX          = 4096
SLM_AMBIGUOUS_LOG_PATH = "query/ambiguous_query.log"
SLM_ENABLED          = True

# -----------------------------------------------------------------------------
# Data Graph — Step 3
# -----------------------------------------------------------------------------
DATA_GRAPH_ENABLED           = True
DATA_GRAPH_SAMPLE_SIZE       = 200
DATA_GRAPH_OVERLAP_THRESHOLD = 0.70

# -----------------------------------------------------------------------------
# L3 prompt engineering
# -----------------------------------------------------------------------------
TOP_K_TO_LLM = 6

# -----------------------------------------------------------------------------
# SQL Builder — Layer 4
# -----------------------------------------------------------------------------
SQL_DEFAULT_LIMIT     = 1000
SQL_MAX_SUBQUERY_DEPTH = 3

# -----------------------------------------------------------------------------
# Value Sampler — Step 6
# -----------------------------------------------------------------------------
VALUE_SAMPLER_ENABLED        = True
VALUE_SAMPLE_SIZE            = 100
VALUE_SAMPLER_ELIGIBLE_TYPES = ["CATEGORY", "FREE_TEXT", "IDENTIFIER"]
VALUE_SAMPLER_MAX_VALUE_LEN  = 64
COLUMN_VALUES_TABLE_NAME     = "column_values"
VALUE_EXPANSION_MIN_TOKEN_LEN = 3
VALUE_EXPANSION_PARTIAL_MIN_TOKEN_LEN = 7  # partial substring match requires longer tokens to avoid noise
VALUE_EXPANSION_MAX_COL_MATCHES = 8        # skip value tokens matching >N columns (too generic to be useful)

# =============================================================================
# Column Embedding Enrichment
# =============================================================================
EMBED_USE_COLUMN_DESCRIPTIONS = True
# Passage text for the BGE column store — EXPERIMENTAL; A/B before promoting.
#   "structural" : build_enriched_column_text (schema facts)  ← current default, unchanged
#   "doc"        : the semantic model's retrieval_document (rich NL — definition,
#                  synonyms, example questions). Strongest NL match, but inherits any
#                  LLM mistakes in the doc, and a longer passage can blur the embedding.
#   "hybrid"     : doc + structural concatenated — rich NL signal AND grounding tokens.
# Default "doc" to MATCH what's already in the live store (column_embeddings_v2 was
# ingested with the rich retrieval_documents). Keeping this at "structural" would make
# a re-embed silently DOWNGRADE the column embeddings to thin name/type text. Missing
# doc → falls back to structural. ("hybrid" = doc + structural, if A/B justifies it.)
EMBED_TEXT_STRATEGY = "doc"
EMBED_SENTENCE_MAX_VALUES     = 8
EMBED_SENTENCE_MAX_VALUE_LEN  = 40

# -----------------------------------------------------------------------------
# Synthetic Query Generator — Step 10
# -----------------------------------------------------------------------------
SYNTHETIC_QUERY_GEN_ENABLED     = True
SYNTHETIC_QUERIES_PER_COLUMN    = 3
SYNTHETIC_QUERIES_PER_FK        = 2
SYNTHETIC_GEN_ELIGIBLE_TYPES    = ["MONETARY", "TEMPORAL", "CATEGORY", "IDENTIFIER", "METRIC"]
SYNTHETIC_PAIRS_PATH            = "ingestion/training_pairs.jsonl"
SYNTHETIC_MIN_PAIRS_FOR_FINETUNE = 50
SYNTHETIC_GEN_BATCH_SIZE        = 10
SYNTHETIC_GEN_MAX_COLUMNS       = 100
SYNTHETIC_GEN_MAX_FK_EDGES      = 50
# If True and SYNTHETIC_PAIRS_PATH already contains >= SYNTHETIC_MIN_PAIRS_FOR_FINETUNE
# pairs, skip re-generation and reuse the existing file.
SYNTHETIC_USE_EXISTING_PAIRS    = True

# -----------------------------------------------------------------------------
# Auto Fine-Tuning — Step 11
# -----------------------------------------------------------------------------
AUTO_FINETUNE_ENABLED        = False   # fine-tune chain removed; both tiers use base weights
AUTO_FINETUNE_EPOCHS         = 3
AUTO_FINETUNE_BATCH_SIZE     = 16
BGE_FINETUNE_BATCH_SIZE      = 8    # CPU training — no MPS memory limit
BGE_FINETUNE_DEVICE          = "cpu"  # BGE-large OOMs on MPS at 6.77 GB; always train on CPU
BGE_FINETUNE_EPOCHS          = 1    # optional fine-tune step; referenced by main.py step 11
BGE_FINETUNE_MAX_SEQ_LEN     = 128  # NL/column training pairs are short; matches AUTO_FINETUNE_MAX_SEQ_LEN
AUTO_FINETUNE_WARMUP_STEPS   = 10
AUTO_FINETUNE_MAX_SEQ_LEN    = 128
AUTO_FINETUNE_CHECKPOINT_DIR = "ingestion/client_minilm"
AUTO_FINETUNE_LR             = 2e-5

# MiniLM fine-tuning mode
# "synthetic_only" — synthetic_query_gen.py pairs only (baseline)
# "glossary_only"  — glossary-derived query pairs only
# "combined"       — synthetic + glossary + paraphrase (default, best)
import os as _os
MINILM_FINETUNE_MODE = _os.environ.get("MINILM_FINETUNE_MODE", "combined")
USE_LANGGRAPH        = _os.environ.get("USE_LANGGRAPH", "true").lower() == "true"

# =============================================================================
# QUERY ROUTER
# Classifies user queries into SQL | RAG | hybrid | nosql intent.
# Added here now so config is the single source of truth.
# =============================================================================

# Intent types the router can return
QUERY_ROUTER_INTENTS = ["sql", "rag", "hybrid", "nosql"]

# Confidence threshold below which the router asks for clarification
QUERY_ROUTER_CONFIDENCE_THRESHOLD = 0.6

# Toggle automatic routing — if False, always routes to SQL (backward compat)
QUERY_ROUTER_ENABLED = True

# =============================================================================
# DOCUMENT INGESTION
# Config for connectors/document.py and ingestion/chunk_embedder.py
# =============================================================================

# Supported document formats
DOC_SUPPORTED_FORMATS = ["pdf", "docx", "txt", "html", "md"]

# Chunk size in tokens (approximate — based on whitespace split)
DOC_CHUNK_SIZE        = 512

# Overlap between consecutive chunks in tokens
DOC_CHUNK_OVERLAP     = 64

# Table name for document chunk embeddings (in VEDA_INTERNAL_DB)
DOC_CHUNKS_TABLE_NAME = "doc_chunks"

# Max file size in MB — larger files are skipped with a warning
DOC_MAX_FILE_MB       = 50

# =============================================================================
# DATA LAKE
# Config for connectors/datalake.py
# =============================================================================

# Execution backend for data lake queries
# "duckdb" — in-process, reads Parquet/Delta directly (no cluster needed)
# "spark"  — requires running Spark cluster
# "trino"  — requires running Trino cluster
DATALAKE_QUERY_ENGINE = "duckdb"

# DuckDB memory limit
DATALAKE_DUCKDB_MEMORY_LIMIT = "4GB"

# =============================================================================
# NOSQL
# Config for connectors/nosql.py
# =============================================================================

# Number of documents to sample for NoSQL schema inference
NOSQL_SCHEMA_SAMPLE_SIZE = 100

# Max nested depth to flatten in NoSQL documents
NOSQL_MAX_NESTING_DEPTH = 3

# =============================================================================
# RAG + HYBRID QUERY (Improvements 1–3)
# =============================================================================

# Top-K chunks to retrieve per RAG query
RAG_TOP_K = 5

# RRF smoothing constant for hybrid SQL + RAG fusion
# Same role as ENSEMBLE_RRF_K — higher = smoother rank weighting
HYBRID_RRF_K = 60

# Weight given to SQL column ranks in hybrid RRF merge
# Higher = SQL results ranked more prominently than doc chunks
HYBRID_SQL_WEIGHT = 1.0

# Weight given to RAG chunk ranks in hybrid RRF merge
HYBRID_RAG_WEIGHT = 1.0

# Run L3→L4→L6 SQL execution in the hybrid path and include results in synthesis prompt
HYBRID_EXECUTE_SQL = True

# Maximum rows from SQL execution to include in the hybrid synthesis context
HYBRID_MAX_RESULT_ROWS = 20

# --- Synthetic Query Gen v2 ---
SYNTHETIC_DDL_PAIRS_PER_TABLE  = 5    # how many (question, SQL) pairs per table
SYNTHETIC_DDL_MIN_SAMPLE_ROWS  = 3    # skip tables with fewer real rows than this
SYNTHETIC_VALIDATION_ENABLED   = True # set False to skip DB execution check
SYNTHETIC_VALIDATION_ROW_LIMIT = 5    # LIMIT used during validation (keep small)

# =============================================================================
# UNIFIED DATA GRAPH  (structured + unstructured)
# Master switch + per-phase flags. All default OFF until validated.
# =============================================================================

UNIFIED_GRAPH_ENABLED = True

GRAPH_PERSIST_ENABLED = True
GRAPH_NODES_TABLE     = "graph_nodes"
GRAPH_EDGES_TABLE     = "graph_edges"
GRAPH_EDGE_WEIGHTS = {
    "has_column":    1.0,
    "fk_to":         3.0,
    "discovered_fk": 2.0,
    "mentions":      1.0,
    "name_match":    0.6,   # lower ceiling than value-overlap so it never dominates (B7 fix)
    "about":         1.5,
}
GRAPH_DISCOVERED_FK_TIER_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.6}

GRAPH_CHUNK_LINKING_ENABLED   = True
GRAPH_LINK_VALUE_OVERLAP_MIN  = 0.15
GRAPH_LINK_NAME_MIN_TOKEN_LEN = 4
GRAPH_LINK_EMBED_SIM_MIN      = 0.45
GRAPH_LINK_MAX_EDGES_PER_CHUNK = 8

GRAPH_EMBED_ENABLED       = True
GRAPH_NODE_EMB_TABLE      = "graph_node_embeddings"
GRAPH_NODE_EMB_DIM        = 1024   # BGE large — matches BIENCODER_DIM
GRAPH_COLUMN_SENTENCE_TEMPLATE = "{col_name} {table_name} {semantic_type}"
GRAPH_TABLE_SENTENCE_TEMPLATE  = "{table_name}"

GRAPH_RETRIEVAL_ENABLED   = True
GRAPH_SEED_TOP_K          = 12
GRAPH_EXPAND_HOPS         = 2
GRAPH_EXPAND_MAX_NODES    = 40
GRAPH_HOP_DECAY           = 0.6

GRAPH_GNN_ENABLED         = False

# --- Phase 4 retrieval tuning (regression fixes) ---
GRAPH_HUB_DEGREE_CAP        = 30    # do not expand THROUGH nodes with degree > this (hub guard)
GRAPH_SIBLING_SCORE_FACTOR  = 0.5   # sibling score = min(expanded_scores) * this
GRAPH_SIBLING_MAX_PER_TABLE = 6     # cap sibling columns added per seed table
GRAPH_SINGLE_TABLE_SIM      = 0.9   # top-seed sim that triggers single-table short-circuit
GRAPH_SINGLE_TABLE_TOPN     = 3     # how many top seeds must share one table
GRAPH_SINGLE_TABLE_GAP      = 0.35  # min sim gap seed#1→seed#2 to call seed#1 dominant
GRAPH_SEED_SIM_FLOOR        = 0.50  # ignore seeds below this when testing table agreement
GRAPH_MAX_COLS_TO_L3        = TOP_K     # truncate graph columns before they reach L3 (=15)
GRAPH_MAX_CHUNKS            = RAG_TOP_K # cap cross-modal chunks returned

# --- Phase 2 name-match hardening ---
GRAPH_LINK_NAME_STOPWORDS = {
    "name", "type", "status", "state", "id", "date", "time", "user", "data",
    "code", "flag", "text", "note", "info", "list", "item", "link", "path",
    "file", "mode", "role", "form", "base", "size", "rank", "sort", "key",
    "hash", "value", "count", "total", "bool", "json", "uuid", "null",
    "created", "updated", "deleted", "modified", "last", "first", "next",
    "prev", "from", "with", "this", "that", "have", "been",
    "number", "org", "object", "field",
}

# =============================================================================
# BIENCODER  (BGE large — two-stage retrieval)
# =============================================================================

BIENCODER_ENABLED      = True
BIENCODER_MODEL        = "BAAI/bge-large-en-v1.5"
BIENCODER_DIM          = 1024
BIENCODER_DEVICE       = _RESOLVED_DEVICE
BIENCODER_BATCH_SIZE   = 32
BIENCODER_QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
BIENCODER_PASSAGE_PREFIX = ""
BIENCODER_CANDIDATE_COLS   = 80
BIENCODER_CANDIDATE_TABLES = 10

# =============================================================================
# RERANKER  (cross-encoder — final stage)
# =============================================================================

RERANKER_ENABLED       = True
# Run the cross-encoder reranker on the PRIMARY path (veda/pipeline.py, after Phase3
# retrieve, before anchor selection) — not just the Tier-2 fallback. Reorders candidates +
# updates final_score so anchor/grain selection ranks off reranked scores. Safe to promote
# now that the reranker carries NO hardcoded business map (uses generated domain_synonyms).
# NOTE: after enabling, the anchor override-margin (ANCHOR_CONFIDENCE_MARGIN) must be
# recalibrated to the reranker score scale — see Step 3 / warm-env eval.
PRIMARY_RERANK_ENABLED = True
RERANKER_MODEL         = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE        = _RESOLVED_DEVICE
RERANKER_BATCH_SIZE    = 64
RERANKER_MAX_TEXT_LEN  = 512
RERANKER_TOP_COLS      = 15
RERANKER_TOP_TABLES    = 5

# --- Reranker: enriched text + dynamic cutoff (Gaps 1, 2, 3) ---
# A/B data: bare names score sharper (0.89) than enriched text (0.15) for the cross-encoder;
# enriched text helps the bi-encoder first stage but hurts cross-encoder pooling.
RERANKER_USE_ENRICHED_TEXT = True  # bare names give sharper cross-encoder scores
RERANKER_DYNAMIC_CUTOFF    = True   # cut by score gap, not fixed top-N
RERANKER_RELATIVE_DROP     = 0.15   # loosened to keep more synonym-matched columns
RERANKER_SCORE_MIN         = 0.03   # lower absolute floor to preserve low-scoring filter columns
RERANKER_MIN_COLS          = 3      # most queries need >= 3 output cols (status, id, primary)
RERANKER_MAX_COLS          = 20     # increased cap to match TOP_K

# Intent-aware cutoff thresholds (Fix 2)
RERANKER_CUTOFF_BY_INTENT      = True
RERANKER_RELATIVE_DROP_DIRECT  = 0.15  # loosened: synonym queries score lower but are still relevant
RERANKER_RELATIVE_DROP_MULTI   = 0.05  # very loose: multi-table / synonym queries
RERANKER_MAX_COLS_MULTI        = 20    # match TOP_K for multi-table queries

# =============================================================================
# VALUE FILTER  (query-time value matching)
# =============================================================================

# --- Value-aware filter retrieval + prompting (Gap 4) ---
VALUE_FILTER_ENABLED             = True   # detect query tokens that match sampled column values
VALUE_FILTER_MAX_COLS            = 3      # max value-matched filter cols to force-include
VALUE_FILTER_MIN_TOKEN_LEN       = 4      # ignore short tokens when matching values
VALUE_FILTER_SCOPE_TO_CANDIDATES = True   # only keep value-matches whose table is in the candidate set
VALUE_FILTER_ALLOW_CROSS_TABLE   = True  # if no in-scope match, add nothing (safer)
VALUE_FILTER_DETERMINISTIC       = True   # build WHERE condition from value-match; don't rely on SLM
SLM_PROMPT_INCLUDE_VALUES        = True   # put sample values of candidate cols in the L3 prompt
SLM_PROMPT_MAX_VALUES            = 8      # sample values per column shown to the SLM

BIDIRECTIONAL_ENABLED  = True

# PART 2 — Value filter exact-match guards (query-time; no re-ingestion)
VALUE_FILTER_VALUE_ONLY       = True    # require exact (ci) token→value; skip col-name-token matches
VALUE_FILTER_SKIP_BOOLEAN     = True    # skip is_*/has_* cols from non-boolean tokens

# =============================================================================
# ACRONYM EXPANSION  (ingestion-time; re-ingestion required when changed)
# =============================================================================

ACRONYM_EXPANSION_ENABLED = True
ACRONYM_MAP = {
    "rfi":    "request for information",
    "sla":    "service level agreement",
    "kyc":    "know your customer",
    "pii":    "personally identifiable information",
    "aml":    "anti money laundering",
    "sar":    "suspicious activity report",
    "ctr":    "currency transaction report",
    "str":    "suspicious transaction report",
    "cdd":    "customer due diligence",
    "edd":    "enhanced due diligence",
    "pep":    "politically exposed person",
    "ubo":    "ultimate beneficial owner",
    "ofac":   "office of foreign assets control",
    "bsa":    "bank secrecy act",
    "txn":    "transaction",
    "ml":     "machine learning",
    "fk":     "foreign key",
    "pk":     "primary key",
    "org":    "organization",
    "config": "configuration",
    "auth":   "authentication",
    "perm":   "permission",
}

# =============================================================================
# SCHEMA LINKER  (query-time; no re-ingestion)
# =============================================================================

SCHEMA_LINK_ENABLED        = True
SPACY_MODEL                = "en_core_web_sm"
SCHEMA_LINK_MIN_TOKEN_LEN  = 3
SCHEMA_LINK_SHORTCIRCUIT_MIN_COLS   = 1
SCHEMA_LINK_SHORTCIRCUIT_MAX_TABLES = 2
SCHEMA_LINK_SYNONYMS = {
    "number": ["no", "num"], "identifier": ["id"], "amount": ["amt"],
    "quantity": ["qty"], "description": ["desc"], "category": ["cat", "type"],
}

# =============================================================================
# JOIN-FREE IR + NL ANSWER
# =============================================================================
IR_JOIN_FREE_ENABLED = True   # SLM omits joins[]; sql_builder derives from fk_adjacency

NL_ANSWER_ENABLED      = True
NL_ANSWER_MAX_ROWS     = 50

# =============================================================================
# BGE fine-tune mode — mirrors MINILM_FINETUNE_MODE
# =============================================================================
BGE_FINETUNE_MODE = _os.environ.get("BGE_FINETUNE_MODE", "combined")

# RELGT in hybrid embedding (always True in merged branch)
RELGT_IN_HYBRID = True

# Retrieval V2 — off by default, enable when BGE indexed
RETRIEVAL_V2_ENABLED = _os.environ.get("RETRIEVAL_V2_ENABLED", "true").lower() == "true"

# Legacy retrieval — keep True so existing ensemble path works
LEGACY_RETRIEVAL_DISABLED = False

# =============================================================================
# FINAL ARCHITECTURE (L1–L6) — Hybrid POC + Homzhub
# =============================================================================

# --- L1: Temporal Parser ---
TEMPORAL_PARSER_ENABLED = True

# --- L2: Semantic Layer (5-stage pipeline) ---
SEMANTIC_LAYER_V2_ENABLED = True

# Post-ingestion derived artifacts (LLM-free transforms of veda_semantic_model.json):
#   - data/veda_relationship_graph.json        (join planner / fast path / graph guard)
#   - semantic/{concepts,dimensions,metrics,MANIFEST}.json  (fast-path Phase-1 registry)
# Regenerated on every ingestion so the fast path never reads a stale graph/model.
DERIVED_ARTIFACTS_ENABLED = True

# Semantic-layer checkpointing — the Qwen Stage 3/4 work is the slow part (hours on a
# big DB). Flush the generated table+column metadata to a JSON checkpoint every N tables
# so a crash/Ctrl-C resumes from the last checkpoint instead of restarting. The
# checkpoint is fingerprint-guarded (ignored if the schema changed) and deleted on a
# successful full build.
SEMANTIC_CHECKPOINT_ENABLED = True
SEMANTIC_CHECKPOINT_EVERY   = 5                                  # flush every N tables
SEMANTIC_CHECKPOINT_FILE    = "data/veda_semantic_checkpoint.json"

# Parallel Qwen execution for the semantic layer (Stage 3 table + Stage 4 column
# understanding). OFF by default → the exact sequential behaviour is preserved. When
# ON, independent table-level LLM calls run concurrently via a bounded thread pool;
# results are merged in schema order so the output is identical to a sequential run.
# Only scheduling changes — prompts, parsing, overrides, checkpointing are untouched.
#   Recommended max_parallel_requests:
#     • low-memory laptop ............ 1–2
#     • 32–64 GB workstation ......... 4–8
#     • Mac Mini / Mac Studio (ample RAM) → tune to Ollama throughput
#   Workers are always capped at the number of tables (never one thread per table).
SEMANTIC_PARALLEL_QWEN_ENABLED = False   # enable_parallel_qwen
SEMANTIC_MAX_PARALLEL_REQUESTS = 2       # max_parallel_requests (validated: min 1) — 2 fits single-GPU compute

# Resilience for Qwen/Ollama calls (applies to sequential AND parallel).
#   • Retry with exponential backoff recovers transient timeouts so a single slow call
#     no longer silently degrades a table to deterministic-only metadata.
#   • Circuit breaker (parallel only): after N consecutive failures it serializes calls
#     so a saturated single-slot Ollama can recover, then resumes concurrency on success.
#     Set threshold <= 0 to disable. NOTE: real speedup still requires Ollama itself to
#     allow concurrency (start it with OLLAMA_NUM_PARALLEL>=workers); these knobs make
#     parallel SAFE, not fast.
SEMANTIC_QWEN_MAX_RETRIES = 2            # retries AFTER the first attempt (0 = legacy: no retry)
SEMANTIC_QWEN_BACKOFF_BASE_SEC = 2.0     # backoff delay = base * 2**attempt + jitter
SEMANTIC_CIRCUIT_BREAKER_THRESHOLD = 3   # consecutive failures → trip (0 = disabled)
# Sequential fail-fast breaker: after THRESHOLD consecutive failures, cool down and probe
# Ollama. If it recovers (model reload / brief restart), resume; if still unreachable after
# MAX_COOLDOWNS probes, ABORT the run (progress is checkpointed → rerun resumes). This stops
# the pipeline from grinding through every remaining table when Ollama is systemically down.
SEMANTIC_CIRCUIT_COOLDOWN_SEC = 15.0     # pause between health probes after tripping
SEMANTIC_CIRCUIT_MAX_COOLDOWNS = 3       # probes before aborting (0 = never abort, just cooldown once)

# Stage 1: Data Profiling
PROFILING_ENABLED = True
PROFILING_NULL_SAMPLE_SIZE = 1000   # sample this many rows to compute null%
PROFILING_DISTINCT_LIMIT = 100      # cap distinct value computation at this
PROFILING_TOP_VALUES_LIMIT = 10     # keep top N values per column

# Stage 2: Domain Glossary (Qwen)
GLOSSARY_GENERATION_ENABLED = True
GLOSSARY_TEMPERATURE = 0.5
GLOSSARY_TIMEOUT = 120
GLOSSARY_DOMAIN_DESCRIPTION = "Compliance and risk management, fraud detection, AML/KYC, incident investigation"

# One-line domain primer injected into the LLM SQL-generation prompt so it interprets
# business terminology correctly. DESCRIPTIVE only — the prompt explicitly forbids
# inventing filters/rules from it, and the IR-equivalence firewall enforces that.
# Set to "" to disable.
DOMAIN_CONTEXT = GLOSSARY_DOMAIN_DESCRIPTION

# Per-column glossary injected into the SQL prompt: the in-scope columns' real
# business_definition + aliases (vocab→column), so the LLM maps "the handler" →
# assigned_to_id. Scoped to in-scope cols, capped + truncated (token budget), and
# framed "interpret only — no invented filters" (firewall still backstops). This is
# the column-level replacement for the near-useless generic DOMAIN_CONTEXT line.
SQL_COLUMN_GLOSSARY_ENABLED  = True
SQL_COLUMN_GLOSSARY_MAX_COLS = 12
SQL_COLUMN_GLOSSARY_DEF_LEN  = 80

# Stage 3: Table Understanding (Qwen)
TABLE_UNDERSTANDING_ENABLED = True
TABLE_UNDERSTANDING_TEMPERATURE = 0.3
TABLE_UNDERSTANDING_TIMEOUT = 120

# Stage 4: Column Understanding (Qwen, batched)
COLUMN_UNDERSTANDING_ENABLED = True
COLUMN_UNDERSTANDING_BATCH_SIZE = 5  # Optimal batch size (10 was slower, 5 is sweet spot)
COLUMN_UNDERSTANDING_TEMPERATURE = 0.3
COLUMN_UNDERSTANDING_TIMEOUT = 240   # generous: concurrent generations share GPU compute → slower per call

# Stage 5: Retrieval Document Builder
RETRIEVAL_DOCUMENT_BUILDER_ENABLED = True
RETRIEVAL_DOCUMENT_TEMPLATE = """COLUMN: {col_name} | ROLE: {analytics_role} | TYPE: {semantic_type} |
DEFINITION: {business_definition} |
ALIASES: {aliases_str} |
TABLE: {table_name} ({table_purpose}) |
LINKS TO: {fk_links_str} |
VALUES: {top_values_str} |
RANGE: {min_val} to {max_val} | AVG: {avg_val} |
NULL: {null_percentage}% | DISTINCT: {distinct_count}"""

# Post-processing: Domain Synonyms + Concept Graph
DOMAIN_SYNONYMS_ENABLED = True
CONCEPT_GRAPH_ENABLED = True
CONCEPT_GRAPH_CONCEPTS = ["PERSON", "AMOUNT", "DATE", "METRIC"]

# Cache invalidation: fingerprint hash of schema

# --- L3: Retrieval (4-signal RRF + cross-encoder) ---

# Signal 1: BGE-M3 semantic
# Unified on bge-large-en-v1.5 — the SAME model BIENCODER_MODEL uses for the column
# store (1024-dim), so query encoder and stored embeddings share one vector space.
# (Was "BAAI/bge-m3", which isn't in the local HF cache → offline load failed.)
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_DEVICE = _RESOLVED_DEVICE

# Schema-grounding gate: a query concept is "grounded" if its best cosine to any
# column/table embedding is >= this floor. Concepts below it (e.g. "AML risk score"
# when no such field exists) trigger refusal instead of hallucinated SQL.
# Single tunable knob — no hardcoded vocabulary. Calibrate per deployment.

# Query-grammar operators — the LANGUAGE layer (shared across ALL databases, NOT
# per-schema). These are universal NL query semantics (negation/existence/quantity/
# grouping), the only signal that can flip EXISTS↔NOT EXISTS. Edit per language, never
# per table/column. All DB-specific knowledge lives in the semantic model + relationship
# graph, not here.
QUERY_GRAMMAR = {
    "negation":  ["without", "no", "not", "missing", "never", "lacking", "absent"],
    "existence": ["with", "have", "has", "having", "contains", "containing",
                  "associated", "linked"],
    "counting":  ["how many", "count", "number of"],
    "quantity":  ["more than", "at least", "fewer than", "less than", "greater than",
                  "exactly", "over"],
    "grouping":  ["per", "each", "grouped by"],
}

# Query LANGUAGE layer — the ONLY word-lists in the system. These are CLOSED
# LINGUISTIC CLASSES (command verbs, ranking operators, function words, temporal
# words, number words), NOT schema vocabulary. They do not grow per query and carry
# no DB knowledge. All entity/column/value knowledge is data-derived (semantic model
# + registry). Override per deployment language; never add table/column/value words
# here (those would be hardcoding). Consumed by the qualifier-completeness gate and
# the fast-path residual guard to separate "what the user is asking FOR" (content,
# must appear in the SQL) from "how they phrased it" (language, ignorable).
QUERY_LANGUAGE = {
    # imperative verbs that introduce a request
    "command_verbs": ["show", "list", "give", "get", "find", "display", "return",
                      "fetch", "tell", "select", "retrieve", "provide", "see",
                      "count", "rank", "sort", "order", "view", "pull"],
    # ranking operators → become ORDER BY ... LIMIT, never a literal in the SQL
    "ranking": ["largest", "biggest", "greatest", "highest", "smallest", "lowest",
                "oldest", "newest", "latest", "earliest", "maximum", "minimum",
                "max", "min", "most", "least", "greater", "fewer", "fewest", "top",
                "bottom", "first", "last"],
    # function words (articles, prepositions, conjunctions, auxiliaries, pronouns)
    "stopwords": ["the", "all", "any", "each", "and", "for", "with", "from", "into",
                  "out", "that", "those", "these", "this", "are", "was", "were",
                  "what", "how", "why", "when",   # interrogatives (which/who/… already below)
                  "which", "who", "whom", "whose", "where", "there", "here", "please",
                  "their", "its", "about", "also", "only", "just", "much", "more",
                  "less", "than", "exactly", "them", "they", "not", "but", "have",
                  "has", "had", "been", "being", "does", "per", "give"],
    # temporal words (the temporal window itself is resolved by L1, so these tokens
    # are already consumed before SQL generation)
    "temporal": ["last", "past", "previous", "recent", "recently", "ago", "since",
                 "before", "after", "during", "year", "month", "week", "day",
                 "quarter", "today", "yesterday", "date", "time", "range", "current",
                 "january", "february", "march", "april", "june", "july", "august",
                 "september", "october", "november", "december"],
    # cardinality words (the count/threshold is handled by the quantity/grain path)
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "single", "multiple", "several", "number", "total",
                "amount"],
    # generic collection / relation nouns — how people refer to rows or a junction
    # ("mappings", "entries", "records") regardless of the table's real name
    "collection_nouns": ["mapping", "record", "entry", "item", "row", "listing",
                         "collection", "association", "detail", "info", "field",
                         "attribute", "value", "set", "thing"],
    # relationship verbs — describe HOW entities relate (the join), never a value
    "relation_verbs": ["assigned", "owned", "belong", "belonging", "related", "linked",
                      "associated", "mapped", "connected", "tied", "attached", "held",
                      "containing", "contains", "including", "registered", "stored",
                      "tracked", "named", "called", "based", "using"],
}
# --- Target selection (evidence-based, Stage 1 of the join pipeline) -------------
# Feature-flagged so it can be benchmarked OLD-vs-NEW before adoption. When False the
# legacy boolean token-match target selection runs (behaviour-preserving). When True,
# candidates are scored by a confidence blend and ambiguous/low-confidence cases REFUSE
# (correctness > recall). Thresholds are tuned from benchmark data, not intuition.
USE_NEW_TARGET_SELECTION = True
TARGET_SELECTION = {
    "W_LEX": 0.5,      # weight on lexical name-coverage (0..1)
    "W_RET": 0.5,      # weight on normalized retrieval score (0..1)
    "ACCEPT": 0.65,    # confidence ≥ ACCEPT  → requested target
    "REJECT": 0.30,    # confidence <  REJECT → dropped
    "DELTA": 0.05,     # two token-competing candidates within DELTA → ambiguous → refuse
}

# Anchor (subject/grain) selection is a MULTI-SIGNAL SCORE, not a binary heuristic.
# Each signal is normalized 0..1; position (word order) is ONE feature among several,
# never the sole decider. No vocabulary — all four signals are data/structure-derived:
#   lexical   : how fully the query names this table (matched name tokens / table tokens)
#   position  : how early the table is mentioned (1 - first_word_index / len) — subject prior
#   retrieval : normalized retrieval score for the table
#   graph     : fraction of the OTHER co-mentioned candidates this table can reach
#               (the query's structural hub — the entity the others hang off)
# confidence = top_score - second_score. Below ANCHOR_CONFIDENCE_MARGIN the grain is
# AMBIGUOUS → clarify/refuse instead of silently emitting SQL at the wrong grain.
ANCHOR_SCORING = {
    "lexical":   0.40,
    "position":  0.20,
    "retrieval": 0.25,
    "graph":     0.15,
}
ANCHOR_CONFIDENCE_MARGIN = 0.06    # min top-second gap to commit; below → ambiguous
ANCHOR_CONFIDENCE_GATE = True      # when False, commit to top candidate without abstaining
# When the composite winner is NOT the earliest-mentioned candidate, the subject
# prior (position) disagrees with the lexical pick — a SIGNAL CONFLICT. Require this
# multiple of the normal margin to commit; otherwise abstain ("comment with their
# investigation_and_research_counter_party": position says comment, the 4-token
# object name says otherwise → don't silently pick either).
ANCHOR_CONFLICT_MULT = 3.0
# The table router (select_primary_table) blends lexical+semantic+column score but has
# NO word-order, grain-hint, or junction awareness — so it mis-picks the grain on
# "comments with their incident" (→incident) and "signal score for each incident"
# (→incident_signal_score). When True, vet_primary lets the multi-signal score_anchors
# OVERRIDE the router's primary, but only when it wins by ≥ ANCHOR_CONFIDENCE_MARGIN
# (so the router still decides the many cases where it's right). Covers BOTH the
# single-table and join paths.
ANCHOR_VET_ROUTER = True

# ── Explainability Trace (veda/explain.py) ────────────────────────────────────
# ENABLED: collect a structured per-query trace (decisions + confidences + why) —
#   cheap (dict appends), attached to the result + persisted compact to the trace log.
# VERBOSE: ALSO collect heavy candidate lists / rejected paths (debug only).
# PERSIST: append the compact trace to logs/explain_trace.jsonl.
EXPLAIN_TRACE_ENABLED = True
EXPLAIN_TRACE_VERBOSE = True
EXPLAIN_TRACE_PERSIST = True

# IR Equivalence Validation (veda/ir_equivalence.py): refuse LLM-generated SQL that
# introduces filters/grouping/ordering/joins/DISTINCT the query never licensed
# (catches "how many workflow state" → WHERE is_final=…). Only runs on LLM SQL;
# deterministic builders are trusted. Correctness > recall.
IR_EQUIVALENCE_ENABLED = True

# Query Enhancement (veda/query_enhancement.py): additive retrieval-recall sidecar.
# The original query is immutable and stays the input to routing/planning/validation;
# enhancement only widens what RETRIEVAL searches for (typo/plural/alias/synonym).
# Adds NO filters/dates/intent. LLM follow-up resolution is separately gated + off.
QUERY_ENHANCEMENT_ENABLED = False
QUERY_ENHANCEMENT_LLM_FOLLOWUP = False

# Failure feedback (veda/feedback.py): on a refusal/error, emit a plain-language WHY +
# WHAT's-needed + concrete suggestions (valid column values, closest tables) instead of a
# terse rejection. Deterministic + always-on. FEEDBACK_LLM_POLISH (default OFF) optionally
# routes the structured facts through the SLM to rephrase them — rephrase-only, never
# invent; the deterministic text is the guaranteed fallback.
FEEDBACK_ENABLED = True
FEEDBACK_LLM_POLISH = True


# Signal 2: BM25 keyword

# Signal 3: Subgraph

# Signal 4: FK Path

# RRF Parameters

# Cross-encoder reranking — the ONE reranker config is RERANKER_MODEL / RERANKER_DEVICE /
# RERANKER_TOP_COLS above (used by query/reranker.py). The former CROSS_ENCODER_* triple here
# was an unused duplicate of the same model string and has been removed to avoid drift.

# Intent detection

# Intent-aware boosting

# --- L4: Intent & Cache ---
# Single source of truth for the Phase-3 RETRIEVAL result cache (retrieval_cache.py, 5-min
# TTL). Previously hardcoded per call-site — main.py used True, the hybrid engine used False —
# so the SAME query could return cached-vs-fresh (and stale, if data changed) depending on
# entry point and timing. All call-sites now read this flag. Default False = always fresh =
# deterministic + no staleness (the canonical hybrid path's behaviour); the real repeat-query
# speedup is the verified-query cache, not this retrieval cache.
RETRIEVAL_CACHE_ENABLED = False

# --- L5: SQL Generation (Qwen + fallback) ---

# =============================================================================
# DETERMINISTIC FAST PATH (Phase-1 semantic-layer slice)
# Count / aggregate / dimension-list questions resolve directly against the
# compiled registries (semantic/*.json) — no retrieval, no join planner, no LLM.
# Built offline by: python3 -m semantic.compile_semantic_layer
# Fast-path SQL is still value-grounded + AST-validated before execution.
# =============================================================================
FAST_PATH_ENABLED = True
# Store the raw NL query text in the route log. Useful for tuning; turn OFF in
# deployments where users may type sensitive values into questions — the log
# then carries only route/table/latency (no query content).
ROUTE_LOG_INCLUDE_QUERY = True

# =============================================================================
# JOIN PLANNER V2 (Steiner join tree + occurrence aliasing)
# =============================================================================
# Tree planner connects ALL requested entities with one minimal join tree
# (replaces pairwise anchor->target paths and the 2-target cap). Flag is a
# rollback switch: False restores the legacy pairwise planner exactly.
JOIN_TREE_PLANNER_ENABLED = True
JOIN_MAX_TARGETS   = 5    # sanity cap on requested entities per query (was hardcoded 2)
# Cost-based traversal: weights are business=1, reference/poly=2, audit=10, so a
# cost ceiling of 8 permits deep business chains (up to ~6 hops) while still
# making any audit-edge chain unreachable. Hops is a secondary backstop.
JOIN_MAX_HOPS      = 6
JOIN_MAX_PATH_COST = 8
# Grain planner: "X with their Y count" / "X with more than N Y" gets deterministic
# pre-aggregation CTE SQL (aggregate each child by FK first, then join to the anchor
# grain) — no LLM, no fan-out, converts the guard's refusals into correct answers.
GRAIN_PLANNER_ENABLED = True
# Apply each metric's stored soft-delete filter ("live rows only") to fast-path
# counts. OFF by default: filtering changes answer semantics, so a human enables
# it per deployment after reviewing MANIFEST.grain_suspects / soft_delete_metrics.
COUNT_EXCLUDE_SOFT_DELETED = False
# Live-DB reverse value resolver: when the registry's sampled values can't match a
# filter token, probe the live client DB to find which column holds that value
# (sample-independent, grounded). OFF by default — resolution quality depends on the
# real data, so validate on the real env before enabling.
VALUE_RESOLVER_LIVE_DB = True
# Value-vs-Column Arbitration: before retrieval expansion / SQL generation, classify
# query spans as SCHEMA_REF | VALUE | NEGATED_VALUE | ENTITY using the sampled
# `column_values` store (data-driven, EXACT match — no word lists). A token that matches
# a sampled categorical value is grounded as a filter VALUE and excluded from
# column-name retrieval ("open" -> status value, not open_date). Adds negation
# (`unresolved` -> status != resolved) which value_resolver cannot express.
VALUE_ARBITER_ENABLED = True
# Arbiter consumes the same column_values store the value_sampler writes; no new store.
# Retrieval-side use of the arbiter: exclude tokens grounded as categorical VALUES from
# the keyword name-match injection (Step 4a) so e.g. "open" cannot inject `open_date`.
# This is the ONLY arbiter behaviour that can move retrieval recall, so it ships OFF by
# default — flip to True and A/B on the real-env recall suite before enabling.
VALUE_ARBITER_RETRIEVAL_FILTER = True
# Answer-Entity Discovery: when a query asks WHO (person answer), project the person's
# display column reached over a FK (incident.assigned_to_id → user.<name>) instead of the
# raw id. Reuses concept_graph["PERSON"] + the FK graph + _resolve_display_column. Emits a
# deterministic JOIN. Ships OFF — it builds a JOIN in the hot path, so validate on the real
# env (SLM+DB) before enabling.
ANSWER_ENTITY_DISCOVERY_ENABLED = True
# Multi-hop FK resolution: when 1-hop value resolution finds nothing, resolve an entity
# filter through ONE unambiguous junction-membership path (tags on a document via
# document_tags). Multiple paths (RBAC direct+role), shared dimensions, or provenance FKs →
# refuse → LLM (never guesses/unions — path choice is semantic, not structural). Builds a
# nested IN-subquery. Ships OFF — validate on the real env (SLM+DB) before enabling.
# TEMP off: misreads the subject word ("incidents") as a value → emits invalid SQL
# (unknown columns incident_id/level_id/signal_id/target_table) → only causes refusals. Fix later.
MULTIHOP_FK_RESOLUTION_ENABLED = True
# Author-agnostic graph guards in the firewall: every JOIN key must be a real FK edge
# and the query must be connected (no cartesian). Verified not to reject the
# deterministic head's own SQL shapes; ON by default.
GRAPH_GUARD_ENABLED = True
# Tier-2 SQL fallback: when the deterministic head can't answer, let the LLM emit IR →
# deterministic builder → graph-guarded firewall → execute. ON: this is the clean
# phrasing-robustness fallback (LLM never writes SQL; firewall validates every join +
# parameterizes). Needs Ollama reachable; if it isn't, _tier2_sql degrades to None and the
# original refusal stands — so enabling is safe. NOTE: the IR schema models only
# SELECT/COUNT/AGGREGATE — ratio/trend/compare paraphrases are NOT expressible here and
# stay fast-path-only until the IR is extended.
TIER2_LLM_FALLBACK = True
# Phase 2 unification — ONE JOIN ENGINE. When the LLM (LangGraph) path identifies a
# MULTI-table query, build joins via the deterministic graph planner (plan_join_tree
# + build_skeleton) instead of sql_builder's retrieval-provided join_path. The LLM
# only NAMES entities; it never invents joins. Single-table still uses sql_builder.
# Activates only inside the Tier-2 fallback (itself gated by TIER2_LLM_FALLBACK), so
# production is unaffected until both are on + validated in a real env (Ollama+DB).
LANGGRAPH_SHARED_PLANNER = True

# Compound-query decomposition — when ONE utterance carries MULTIPLE INDEPENDENT
# questions ("how many incidents are open AND list active users"), split it into
# independent sub-queries and run EACH through the existing single-query pipeline
# (query.slm_layer.run_decomposer + veda_hybrid fan-out). The LLM decides the split
# (no lexical rules); when unsure it returns a SINGLE query (refuse-over-guess — a
# missed split is a rephrasable refusal, a mis-split is a silent wrong answer).
# The deterministic head self-certifies completeness (qualifier_completeness), so a
# clean SQL answer SKIPS the decomposer (zero added latency on the hot path); only a
# non-deterministic head or a deterministic refusal triggers it. Needs Ollama; if
# unreachable run_decomposer degrades to "single" and behaviour is exactly as today.
QUERY_DECOMPOSE_ENABLED = False   # TEMP off: splits join queries ("X and their Y") wrongly — fix later
# Independent sub-queries of a compound query are I/O-bound (DB / Ollama / RAG) and
# share NO state — execute_sql opens a fresh connection per call — so fan them out
# concurrently. Output is captured per sub-query and emitted IN ORDER; results keep
# query order regardless of completion order. Set 1 to force sequential (debugging).
QUERY_DECOMPOSE_MAX_WORKERS = 1
# The split decision IS the eval data: every decomposition is logged (query, predicted
# type, sub_queries) so production traffic becomes the labelled set for measuring
# split accuracy (evaluation/decompose_eval.py). Best-effort; never raises. Turn the
# query text OFF (like ROUTE_LOG_INCLUDE_QUERY) where questions may carry sensitive values.
DECOMPOSE_LOG_PATH = "logs/decompose_log.jsonl"
DECOMPOSE_LOG_ENABLED = True

# Fallback SQL (rule-based, no LLM)

# --- L6: Validation & Repair ---

# 5-layer checks



# Repair loop

# Repair strategies

# --- Execution + Audit ---
EXECUTION_QUERY_TIMEOUT_SECS = 30
EXECUTION_RESULT_LIMIT = 1000


# =============================================================================
# OUTPUT FILES (Final Architecture)
# =============================================================================
# Absolute-overridable (§9) so the inference container finds it regardless of cwd.
SEMANTIC_MODEL_FILE = __import__("os").environ.get(
    "VEDA_SEMANTIC_MODEL_FILE", "data/veda_semantic_model.json"
)
GLOSSARY_FILE = "data/veda_glossary.json"
DOMAIN_SYNONYMS_FILE = "data/veda_domain_synonyms.json"
CONCEPT_GRAPH_FILE = "data/veda_concept_graph.json"
RELATIONSHIP_GRAPH_FILE = "data/veda_relationship_graph.json"
PROFILING_FILE = "data/veda_profiling.json"

# ── Unified Knowledge Graph (fuses FK + concept + semantic + synonyms into one) ──
# Built by ingestion/unified_graph_builder.py; queried by graph/query_graph.py.
UNIFIED_GRAPH_FILE = "data/veda_unified_graph.json"
# graph_expand() in retrieval_v2 is ADDITIVE + flag-guarded — OFF keeps retrieval byte-identical.
GRAPH_EXPAND_ENABLED = True
GRAPH_EXPAND_MAX     = 12   # cap columns added per query (token/latency bound; reranker still cuts)
