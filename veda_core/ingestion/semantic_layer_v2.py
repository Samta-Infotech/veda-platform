# =============================================================================
# ingestion/semantic_layer_v2.py
# VEDA Final Architecture — L2 Semantic Layer (Stages 1-5 + Post-processing)
#
# Stage 1: Data Profiling (SQL-based, pure)
# Stage 2: Database Glossary (Qwen × 1 call for entire DB)
# Stage 3: Table Understanding (Qwen × 1 per table)
# Stage 4: Column Understanding (Qwen × batches/5)
# Stage 5: Retrieval Document Builder (deterministic, optimized field ordering)
# Post 1: Domain Synonyms (smart rule-based extraction)
# Post 2: Concept Graph (concept→column mapping)
#
# Output:
#   veda_semantic_model.json — master semantic metadata file
#   veda_glossary.json — business glossary (13 files total)
#   veda_domain_synonyms.json — synonym expansion
#   veda_concept_graph.json — concept→column mapping
# =============================================================================

import sys
import os
import json
import time
import re
import random
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    TABLE_UNDERSTANDING_ENABLED,
    TABLE_UNDERSTANDING_TEMPERATURE,
    TABLE_UNDERSTANDING_TIMEOUT,
    COLUMN_UNDERSTANDING_ENABLED,
    COLUMN_UNDERSTANDING_BATCH_SIZE,
    COLUMN_UNDERSTANDING_TEMPERATURE,
    COLUMN_UNDERSTANDING_TIMEOUT,
    RETRIEVAL_DOCUMENT_BUILDER_ENABLED,
    RETRIEVAL_DOCUMENT_TEMPLATE,
    SEMANTIC_MODEL_FILE,
    DOMAIN_SYNONYMS_FILE,
    CONCEPT_GRAPH_FILE,
)
from utils.logger import get_logger
from ingestion import data_profiler
from ingestion import glossary_builder
import urllib.request
import urllib.error

logger = get_logger(__name__)


# --- Qwen/Ollama resilience: exponential-backoff retry + adaptive circuit breaker -----
# Retry recovers transient timeouts (so a slow call no longer silently degrades a table).
# The circuit breaker is active ONLY during parallel runs (set by the orchestrator): after
# N consecutive failures it serializes calls so a saturated single-slot Ollama can recover,
# then resumes concurrency on the next success. Sequential mode is unchanged except that a
# failed call is now retried before giving up.
def _qwen_retry_config():
    try:
        from config import SEMANTIC_QWEN_MAX_RETRIES as _r, SEMANTIC_QWEN_BACKOFF_BASE_SEC as _b
        return max(0, int(_r)), float(_b)
    except Exception:
        return 0, 2.0


class _QwenCircuit:
    """Thread-safe adaptive limiter for concurrent Ollama calls. Normally a pass-through;
    after `threshold` consecutive failures it TRIPS and serializes calls (a global lock)
    so an overwhelmed Ollama can recover. A success after tripping RESETS it. threshold<=0
    disables tripping. Used only inside parallel orchestration."""
    def __init__(self, threshold: int):
        self._threshold = max(0, int(threshold))
        self._lock = threading.Lock()
        self._serial = threading.Lock()
        self._consec = 0
        self._tripped = False

    def run(self, fn):
        if self._tripped:
            with self._serial:          # tripped → one Ollama call at a time
                return self._invoke(fn)
        return self._invoke(fn)

    def _invoke(self, fn):
        r = fn()
        ok = r is not None
        with self._lock:
            if ok:
                self._consec = 0
                if self._tripped:
                    self._tripped = False
                    logger.info("⚡ Circuit breaker RESET — Ollama healthy, resuming concurrency")
            else:
                self._consec += 1
                if self._threshold and self._consec >= self._threshold and not self._tripped:
                    self._tripped = True
                    logger.warning(f"⚡ Circuit breaker TRIPPED after {self._consec} consecutive "
                                   f"Ollama failures — serializing calls until recovery")
        return r


_ACTIVE_CIRCUIT: Optional[_QwenCircuit] = None   # set by the parallel orchestrator only


def _set_qwen_circuit(cb: Optional[_QwenCircuit]):
    global _ACTIVE_CIRCUIT
    _ACTIVE_CIRCUIT = cb


def _new_circuit() -> _QwenCircuit:
    try:
        from config import SEMANTIC_CIRCUIT_BREAKER_THRESHOLD as _t
    except Exception:
        _t = 0
    return _QwenCircuit(_t)


class OllamaUnavailableError(RuntimeError):
    """Raised by the sequential fail-fast breaker when Ollama is systemically down."""


def _probe_ollama() -> bool:
    """Lightweight health check — is the Ollama server reachable?"""
    try:
        import urllib.request as _u
        _u.urlopen(f"{SLM_OLLAMA_BASE_URL}/api/tags", timeout=5).read()
        return True
    except Exception:
        return False


_SEQ_FAIL = {"consecutive": 0}   # sequential-mode consecutive-failure counter


def _seq_breaker_record(ok: bool):
    """Sequential fail-fast breaker. Counts consecutive Ollama failures; on THRESHOLD it
    cools down and PROBES the server — recovering (a brief restart / model reload) resets
    and resumes, but if Ollama stays unreachable after MAX_COOLDOWNS it ABORTS so the run
    doesn't grind through every remaining table producing degraded metadata. Progress is
    checkpointed, so a rerun resumes. No-op while a parallel circuit is active (that path
    has its own breaker), and disabled when threshold <= 0."""
    try:
        from config import (SEMANTIC_CIRCUIT_BREAKER_THRESHOLD as _thr,
                            SEMANTIC_CIRCUIT_COOLDOWN_SEC as _cd,
                            SEMANTIC_CIRCUIT_MAX_COOLDOWNS as _k)
    except Exception:
        return
    if _thr <= 0:
        return
    if ok:
        _SEQ_FAIL["consecutive"] = 0
        return
    _SEQ_FAIL["consecutive"] += 1
    if _SEQ_FAIL["consecutive"] < _thr:
        return
    logger.warning(f"⚡ Sequential circuit breaker TRIPPED — {_SEQ_FAIL['consecutive']} consecutive "
                   f"Ollama failures; cooling down to check if the server is alive")
    for i in range(max(1, _k)):
        time.sleep(_cd)
        if _probe_ollama():
            logger.info(f"⚡ Ollama reachable again after cooldown {i + 1} — resuming")
            _SEQ_FAIL["consecutive"] = 0
            return
    raise OllamaUnavailableError(
        f"Ollama unreachable after {_SEQ_FAIL['consecutive']} consecutive failures and "
        f"{_k} cooldown probes — aborting ingestion. Progress is checkpointed; restart "
        f"Ollama and rerun to resume.")


def _call_ollama(prompt: str, model: str = None, temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
    """Call Ollama with exponential-backoff retry; routed through the circuit breaker when
    one is active (parallel mode). Returns the response text, or None after all retries."""
    _max_retries, _base = _qwen_retry_config()

    def _attempted():
        for attempt in range(_max_retries + 1):
            r = _raw_call_ollama(prompt, model, temperature, timeout)
            if r is not None:
                return r
            if attempt < _max_retries:
                delay = _base * (2 ** attempt) + random.uniform(0, _base)
                logger.warning(f"Ollama call failed — retry {attempt + 1}/{_max_retries} in {delay:.1f}s")
                time.sleep(delay)
        return None

    cb = _ACTIVE_CIRCUIT
    if cb is not None:
        return cb.run(_attempted)        # parallel mode: serialize-on-trip breaker
    r = _attempted()                     # sequential mode: fail-fast breaker
    _seq_breaker_record(r is not None)
    return r


def _raw_call_ollama(prompt: str, model: str = None, temperature: float = 0.3, timeout: int = 120) -> Optional[str]:
    """Single Ollama HTTP attempt (no retry). Returns response text or None on any failure."""
    if model is None:
        model = SLM_MODEL_NAME

    url = f"{SLM_OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    request_data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            return response_data.get("response", "").strip()

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return None


@dataclass
class TableMetadata:
    """Metadata for a single table."""
    table_name: str
    business_purpose: str
    primary_entity: str
    table_type: str  # TRANSACTION|MASTER|REFERENCE|EVENT|BRIDGE
    candidate_temporal_columns: List[str]
    candidate_measure_columns: List[str]


@dataclass
class ColumnMetadata:
    """Metadata for a single column.

    Structural fields (semantic_type, analytics_role, sql_usage, importance_class)
    are deterministic — the LLM never emits them, so it cannot corrupt the SQL
    contract. Business fields (business_definition, aliases, query patterns,
    examples, column_domain) come from the LLM. field_confidence/validation track
    per-field trust for tuning.
    """
    col_name: str
    table_name: str
    semantic_type: str  # MONETARY|TEMPORAL|CATEGORICAL|IDENTIFIER|FLAG|TEXT
    analytics_role: str  # MEASURE|DIMENSION|TIME_DIMENSION|IDENTIFIER|ATTRIBUTE
    business_definition: str
    aliases: List[str]
    null_meaning: str
    allowed_aggregations: List[str]
    confidence: float
    # --- v2 additive fields (defaults keep older callers working) ---
    business_role: str = ""
    business_domain: str = ""        # final, after reconciliation
    column_domain: str = ""          # LLM suggestion (entity the column refers to)
    user_query_patterns: List[str] = field(default_factory=list)
    negative_aliases: List[str] = field(default_factory=list)
    business_examples: List[str] = field(default_factory=list)
    importance_class: str = "MEDIUM"  # HIGH | MEDIUM | LOW (deterministic)
    sql_usage: Dict[str, bool] = field(default_factory=dict)  # deterministic
    value_pattern: str = ""           # set when raw values were withheld (leakage-safe)
    value_handling: str = "stats"     # keep | stats | pattern | remove
    sample_values: List[str] = field(default_factory=list)  # ONLY safe enum values
    contains_pii: bool = False
    related_columns: List[str] = field(default_factory=list)
    field_confidence: Dict[str, float] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checkpointing — flush the (slow) Qwen Stage 3/4 work every N tables so a big-DB
# ingestion resumes from the last checkpoint instead of restarting. The checkpoint
# stores the actual generated metadata (tables + columns), is keyed by a schema
# fingerprint (stale checkpoints from a changed schema are ignored), and is deleted on
# a successful full build (see _clear_checkpoint at the end of run_full_semantic_layer).
# ---------------------------------------------------------------------------
def _schema_fingerprint(schema_dict: Dict[str, Any]) -> str:
    import hashlib
    parts = []
    for tname in sorted(schema_dict):
        cols = schema_dict[tname].get("columns", [])
        names = sorted((c.get("col_name") or c.get("name") or "") for c in cols)
        parts.append(tname + ":" + ",".join(names))
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _checkpoint_file() -> str:
    try:
        from config import SEMANTIC_CHECKPOINT_FILE
        return SEMANTIC_CHECKPOINT_FILE
    except Exception:
        return "data/veda_semantic_checkpoint.json"


def _load_checkpoint(fp: str):
    """Return the saved checkpoint dict iff it exists and matches the schema fingerprint
    `fp`; otherwise None (start fresh)."""
    try:
        from config import SEMANTIC_CHECKPOINT_ENABLED
        if not SEMANTIC_CHECKPOINT_ENABLED:
            return None
    except Exception:
        pass
    path = _checkpoint_file()
    if not os.path.exists(path):
        return None
    try:
        ckpt = json.load(open(path))
        if ckpt.get("fingerprint") != fp:
            logger.info("  Checkpoint found but schema changed — ignoring (will rebuild).")
            return None
        n_t = len(ckpt.get("tables", {}))
        n_c = len(ckpt.get("columns", []))
        logger.info(f"  ✓ Resuming from checkpoint: {n_t} tables, {n_c} columns already done.")
        return ckpt
    except Exception as e:
        logger.warning(f"  Checkpoint unreadable ({e}) — starting fresh.")
        return None


def _save_checkpoint(fp: str, table_metadata: dict, all_columns: list) -> None:
    """Flush generated work to JSON. table_metadata: {name: TableMetadata};
    all_columns: [ColumnMetadata]. Both are dataclasses → asdict for JSON."""
    try:
        from config import SEMANTIC_CHECKPOINT_ENABLED
        if not SEMANTIC_CHECKPOINT_ENABLED:
            return
    except Exception:
        pass
    path = _checkpoint_file()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "fingerprint": fp,
            "tables":  {n: asdict(m) for n, m in table_metadata.items()},
            "columns": [asdict(c) for c in all_columns],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)   # atomic — a crash mid-write can't corrupt the checkpoint
        logger.info(f"  💾 checkpoint: {len(payload['tables'])} tables, {len(payload['columns'])} columns")
    except Exception as e:
        logger.warning(f"  Checkpoint save failed (non-fatal): {e}")


def _clear_checkpoint() -> None:
    path = _checkpoint_file()
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("  ✓ semantic build complete — checkpoint cleared.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parallel Qwen execution (OPTIONAL — Stage 3/4 only)
#
# Why optional: the per-table LLM calls are independent, so they parallelize
# cleanly — but concurrency is bounded by Ollama throughput and host RAM (each
# concurrent request holds model context). Default OFF preserves the exact
# sequential behaviour. When ON, tasks run in a bounded ThreadPoolExecutor and
# results are merged on the MAIN thread in SCHEMA ORDER, so the output is
# identical to a sequential run. Only scheduling changes — prompts, parsing,
# overrides, checkpointing, and the semantic model are untouched.
#
# Recommended workers:
#   • Low-memory laptop ............. 1–2
#   • 32–64 GB workstation .......... 4–8
#   • Mac Mini / Mac Studio (ample RAM) tune to Ollama throughput
# Worker count is always capped at the number of tasks (never one thread/table).
# ---------------------------------------------------------------------------
def _parallel_qwen_enabled() -> bool:
    try:
        from config import SEMANTIC_PARALLEL_QWEN_ENABLED as _p
        return bool(_p)
    except Exception:
        return False


def _resolve_qwen_workers(n_tasks: int) -> int:
    """Bounded worker count: at least 1, never more than the number of tasks."""
    try:
        from config import SEMANTIC_MAX_PARALLEL_REQUESTS as _w
    except Exception:
        _w = 4
    try:
        _w = int(_w)
    except Exception:
        _w = 4
    return max(1, min(_w, max(1, n_tasks)))


def _stage3_batch_worker(batch_tuples, batch_names, glossary):
    """Thread worker — one Stage-3 batch (the UNCHANGED batched call, so output is
    identical to sequential). Returns [(table_name, TableMetadata | None)]. Never
    mutates shared state; the caller merges on the main thread."""
    metadatas = stage3_batch_table_understanding(batch_tuples, glossary)
    return list(zip(batch_names, metadatas))


def _stage4_table_worker(table_name, columns, table_meta, glossary, profiling):
    """Thread worker — one table's Stage-4 column understanding, using the UNCHANGED
    per-batch calls + per-batch deterministic overrides (exactly as the sequential
    inner loop). Returns (table_name, [ColumnMetadata]). No shared-state mutation."""
    out: List[ColumnMetadata] = []
    for i in range(0, len(columns), COLUMN_UNDERSTANDING_BATCH_SIZE):
        batch = columns[i:i + COLUMN_UNDERSTANDING_BATCH_SIZE]
        cm = stage4_column_understanding(
            columns_batch=batch, table_name=table_name,
            table_metadata=table_meta, glossary=glossary, profiling=profiling,
        )
        if cm:
            cm = _apply_deterministic_overrides(cm)
            out.extend(cm)
    return table_name, out


def stage3_table_understanding(
    table_name: str,
    columns: List[Dict],
    profiling: Dict[str, Any],
    glossary: Dict[str, str],
) -> Optional[TableMetadata]:
    """
    Stage 3: Call Qwen to understand a single table.

    Args:
        table_name: Table name
        columns: List of {name, type} dicts
        profiling: Profiling dict from Stage 1
        glossary: Business glossary from Stage 2

    Returns:
        TableMetadata or None
    """
    col_names = ", ".join([c.get("col_name") or c.get("name") for c in columns[:10]])
    if len(columns) > 10:
        col_names += f", ... and {len(columns) - 10} more"

    glossary_str = "\n".join([f"  {k}: {v}" for k, v in list(glossary.items())[:15]])

    prompt = f"""You are a database schema analyst.
Given a table definition and business glossary, provide understanding of the table.

Table: {table_name}
Columns: {col_names}

Business Glossary:
{glossary_str}

Provide:
1. business_purpose (one sentence)
2. primary_entity (what does each row represent)
3. table_type (one of: TRANSACTION, MASTER, REFERENCE, EVENT, BRIDGE)
4. candidate_temporal_columns (list of timestamp/date columns)
5. candidate_measure_columns (list of numeric/metric columns)

Output JSON format:
{{
  "business_purpose": "...",
  "primary_entity": "...",
  "table_type": "...",
  "candidate_temporal_columns": [...],
  "candidate_measure_columns": [...]
}}

Output ONLY JSON, no explanation.
"""

    response = _call_ollama(
        prompt=prompt,
        temperature=TABLE_UNDERSTANDING_TEMPERATURE,
        timeout=TABLE_UNDERSTANDING_TIMEOUT,
    )

    if response is None:
        return None

    # Strip markdown code blocks if present
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    try:
        data = json.loads(response)
        return TableMetadata(
            table_name=table_name,
            business_purpose=data.get("business_purpose", ""),
            primary_entity=data.get("primary_entity", ""),
            table_type=data.get("table_type", "TRANSACTION"),
            candidate_temporal_columns=data.get("candidate_temporal_columns", []),
            candidate_measure_columns=data.get("candidate_measure_columns", []),
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse table understanding for {table_name}: {e}")
        return None


def stage3_batch_table_understanding(
    table_batch: List[tuple],
    glossary: Dict[str, str],
) -> List[Optional[TableMetadata]]:
    """
    Stage 3 (Optimized): Call Qwen to understand 2-3 tables in ONE call.

    Args:
        table_batch: List of (table_name, columns) tuples
        glossary: Business glossary from Stage 2

    Returns:
        List of TableMetadata (one per table, with None for failures)
    """
    if not table_batch:
        return []

    # Build table descriptions
    table_descs = []
    for table_name, columns in table_batch:
        col_names = ", ".join([c.get("col_name") or c.get("name") for c in columns[:8]])
        if len(columns) > 8:
            col_names += f", ... ({len(columns)-8} more)"
        table_descs.append(f"  {table_name}: {col_names}")

    glossary_str = "\n".join([f"  {k}: {v}" for k, v in list(glossary.items())[:10]])

    prompt = f"""You are a database schema analyst.
Given {len(table_batch)} table definitions and business glossary, provide understanding of each table.

Tables:
{chr(10).join(table_descs)}

Business Glossary:
{glossary_str}

For EACH table, provide:
1. business_purpose (one sentence)
2. primary_entity (what does each row represent)
3. table_type (one of: TRANSACTION, MASTER, REFERENCE, EVENT, BRIDGE)
4. candidate_temporal_columns (list of timestamp/date columns)
5. candidate_measure_columns (list of numeric/metric columns)

Output JSON array format:
[
  {{
    "table_name": "{table_batch[0][0]}",
    "business_purpose": "...",
    "primary_entity": "...",
    "table_type": "...",
    "candidate_temporal_columns": [...],
    "candidate_measure_columns": [...]
  }},
  ...
]

Output ONLY JSON array, no explanation.
"""

    response = _call_ollama(
        prompt=prompt,
        temperature=TABLE_UNDERSTANDING_TEMPERATURE,
        timeout=TABLE_UNDERSTANDING_TIMEOUT,
    )

    if response is None:
        return [None] * len(table_batch)

    # Strip markdown code blocks if present
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    try:
        data = json.loads(response)
        if not isinstance(data, list):
            return [None] * len(table_batch)

        result = []
        table_names_in_batch = [t[0] for t in table_batch]

        for item in data:
            table_name = item.get("table_name", "")
            if table_name in table_names_in_batch:
                metadata = TableMetadata(
                    table_name=table_name,
                    business_purpose=item.get("business_purpose", ""),
                    primary_entity=item.get("primary_entity", ""),
                    table_type=item.get("table_type", "TRANSACTION"),
                    candidate_temporal_columns=item.get("candidate_temporal_columns", []),
                    candidate_measure_columns=item.get("candidate_measure_columns", []),
                )
                result.append(metadata)

        # Fill in missing results with None
        while len(result) < len(table_batch):
            result.append(None)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse batch table understanding: {e}")
        return [None] * len(table_batch)


def _det_semantic_type(col_name, data_type, is_pk, is_fk, cardinality):
    """Deterministic semantic_type via the authoritative 3-layer rule engine
    (reused from semantic_type_inference — no duplication)."""
    from types import SimpleNamespace
    from ingestion.semantic_type_inference import _layer_a, _layer_b, _layer_c
    sc = SimpleNamespace(col_name=col_name, data_type=(data_type or "").lower(),
                         is_pk=bool(is_pk), is_fk=bool(is_fk), cardinality=cardinality or 0)
    for layer in (_layer_a, _layer_b, _layer_c):
        try:
            r = layer(sc)
        except Exception:
            r = None
        if r:
            return r[0]
    return "FREE_TEXT"


def _safe_value_context(value_handling, value_info, profile):
    """Human-readable, leakage-safe value hint for the LLM prompt."""
    if value_handling == "keep" and value_info.get("values"):
        return "example values: " + ", ".join(str(v) for v in value_info["values"][:8])
    if value_handling in ("pattern", "remove"):
        return f"value format: {value_info.get('value_pattern', 'REDACTED')} (values withheld)"
    # stats only
    bits = []
    if profile:
        if profile.get("distinct_count") is not None:
            bits.append(f"distinct={profile['distinct_count']}")
        if profile.get("min") is not None and profile.get("max") is not None:
            bits.append(f"range={profile['min']}..{profile['max']}")
    return ("stats: " + ", ".join(bits)) if bits else "statistics only"


# --- Relationship-graph enrichment for the stage-4 prompt (schema-aware meanings) ---
# The graph is built offline (ingestion/relationship_graph.py). We only READ it here to
# tell the LLM how a column connects to other tables. No link is ever inferred or invented.

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REL_GRAPH_CACHE: Dict[str, Any] = {}


def _load_relationship_graph() -> Dict[str, Any]:
    """Load the relationship graph once per process, CWD-independently.

    Missing/unreadable/empty file → empty graph so the stage degrades to its
    pre-enrichment behaviour with no crash.
    """
    if "graph" in _REL_GRAPH_CACHE:
        return _REL_GRAPH_CACHE["graph"]
    graph: Dict[str, Any] = {"tables": [], "edges": []}
    try:
        from config import RELATIONSHIP_GRAPH_FILE as _rel_file
    except Exception:
        _rel_file = "data/veda_relationship_graph.json"
    path = _rel_file if os.path.isabs(_rel_file) else os.path.join(_REPO_ROOT, _rel_file)
    try:
        with open(path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            graph = loaded
    except Exception:
        pass
    _REL_GRAPH_CACHE["graph"] = graph
    return graph


def _build_edge_index(graph: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Index graph edges by column and by table for fast per-column lookup.

    edges_by_col[(table, column)] = [ {direction, other_table, other_col, rel_type,
                                       cardinality, polymorphic, discovery, confidence} ]
    neighbours_by_table[table]    = [ (other_table, linking_column, best_weight) ]  (ordered)
    """
    if "index" in _REL_GRAPH_CACHE:
        return _REL_GRAPH_CACHE["index"]
    from collections import defaultdict
    edges_by_col: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    neigh: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)  # table -> {other: (col, weight)}

    for e in graph.get("edges", []):
        st, sc = e.get("source_table"), e.get("source_column")
        tt, tc = e.get("target_table"), e.get("target_column")
        meta = {
            "rel_type": e.get("relationship_type"),
            "cardinality": e.get("cardinality"),
            "polymorphic": bool(e.get("polymorphic")),
            "discovery": e.get("discovery"),
            "confidence": e.get("confidence"),
        }
        weight = e.get("weight") or 1
        if st and sc:
            edges_by_col[(st, sc)].append({"direction": "references", "other_table": tt,
                                           "other_col": tc, **meta})
        if tt and tc:
            edges_by_col[(tt, tc)].append({"direction": "referenced_by", "other_table": st,
                                           "other_col": sc, **meta})
        # undirected neighbour set; remember the strongest linking edge per neighbour
        if st and tt:
            if tt not in neigh[st] or weight > neigh[st][tt][1]:
                neigh[st][tt] = (sc, weight)
            if st not in neigh[tt] or weight > neigh[tt][st][1]:
                neigh[tt][st] = (tc, weight)

    neighbours_by_table: Dict[str, List[Tuple[str, str, float]]] = {}
    for t, m in neigh.items():
        ordered = sorted(((other, col, w) for other, (col, w) in m.items()),
                         key=lambda x: (-x[2], x[0]))
        neighbours_by_table[t] = ordered

    index = (dict(edges_by_col), neighbours_by_table)
    _REL_GRAPH_CACHE["index"] = index
    return index


def _relationship_clause(edges: List[Dict]) -> str:
    """Render up to the top-3 edges for one column as a deterministic prompt clause.

    Order: declared FKs before data-inferred, then higher confidence, then name —
    stable across runs. Returns "" for columns with no edges (row stays unchanged).
    """
    if not edges:
        return ""
    ordered = sorted(edges, key=lambda x: (
        0 if x.get("discovery") == "declared_fk" else 1,
        -(x.get("confidence") or 0.0),
        x.get("other_table") or "",
        x.get("other_col") or "",
    ))[:3]
    parts = []
    for e in ordered:
        card = f" ({e['cardinality']})" if e.get("cardinality") else ""
        if e["direction"] == "references":
            s = f"→ references {e['other_table']}.{e['other_col']}{card}"
        else:
            s = f"← referenced by {e['other_table']}{card}"
        if e.get("polymorphic"):
            s += " [polymorphic ref resolved by a discriminator]"
        parts.append(s)
    return " | " + "; ".join(parts)


def _neighbours_line(neighbours: List[Tuple[str, str, float]]) -> str:
    """One-line table connectivity summary, capped at the top 6 neighbours."""
    if not neighbours:
        return ""
    parts = [f"{t} ({c})" for (t, c, _w) in neighbours[:6]]
    return "This table connects to: " + ", ".join(parts)


def _entity_label(table: str) -> str:
    """Generic table-name → business-entity label (light singularization, title case).

    Derives the label purely from the schema identifier — no hardcoded names — so it
    works for any database. e.g. 'organizations' → 'Organization', 'signal_rules' →
    'Signal Rule', 'user' → 'User'.
    """
    words = table.replace("_", " ").strip().split()
    if words:
        w = words[-1]
        if w.endswith("ies") and len(w) > 4:
            words[-1] = w[:-3] + "y"
        elif w.endswith("ses") and len(w) > 4:
            words[-1] = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            words[-1] = w[:-1]
    return " ".join(x[:1].upper() + x[1:] for x in words)


def _fk_target_entity(edges: List[Dict]) -> str:
    """For a column with FK-out edges to a SINGLE unambiguous, non-polymorphic target,
    return that target's entity label. Empty string for no-FK / polymorphic / multi-target
    columns (those keep the LLM's own domain — we only override what we know for certain).
    """
    refs = [e for e in edges
            if e.get("direction") == "references" and not e.get("polymorphic")]
    if not refs:
        return ""
    targets = {e.get("other_table") for e in refs if e.get("other_table")}
    if len(targets) != 1:
        return ""
    return _entity_label(next(iter(targets)))


def stage4_column_understanding(
    columns_batch: List[Dict],
    table_name: str,
    table_metadata: TableMetadata,
    glossary: Dict[str, str],
    profiling: Optional[Dict[str, Any]] = None,
) -> Optional[List[ColumnMetadata]]:
    """Stage 4 (hybrid): deterministic structure + leakage-safe value handling +
    slim LLM business pass + merge with per-field confidence.

    Structural fields are computed by rules (the LLM never emits them). Raw values
    are sanitized per column before anything reaches the LLM. If the LLM call
    fails, deterministic-only records are still returned (always usable).
    """
    from ingestion.deterministic_metadata import compute_deterministic

    profiling = profiling or {}
    table_domain = (table_metadata.primary_entity or table_name.replace("_", " ")).strip()
    try:
        from config import GLOSSARY_DOMAIN_DESCRIPTION as _DOMAIN_DESC
    except Exception:
        _DOMAIN_DESC = ""

    # ---- 1. Deterministic pass + value sanitization (no LLM, no raw leakage) ----
    edges_by_col, neighbours_by_table = _build_edge_index(_load_relationship_graph())

    det = {}            # col_name -> deterministic dict
    prompt_rows = []
    for c in columns_batch:
        cname = c.get("col_name") or c.get("name") or ""
        dtype = c.get("data_type") or c.get("type") or ""
        is_pk, is_fk = bool(c.get("is_pk")), bool(c.get("is_fk"))
        card = c.get("cardinality")
        prof = profiling.get(f"{table_name}.{cname}", {})
        samples = prof.get("top_values", []) if isinstance(prof, dict) else []
        distinct = prof.get("distinct_count") if isinstance(prof, dict) else None

        sem = _det_semantic_type(cname, dtype, is_pk, is_fk, card)
        d = compute_deterministic(cname, dtype, sem, is_pk, is_fk,
                                  table_name, samples, distinct)
        d["semantic_type"] = sem
        det[cname] = d
        prompt_rows.append(
            f"- {cname} | type={sem} | role={d['analytics_role']} | "
            f"{_safe_value_context(d['value_handling'], d['value_info'], prof)}"
            f"{_relationship_clause(edges_by_col.get((table_name, cname), []))}")

    # ---- 2. Slim LLM pass — business meaning only (structure is FIXED context) ----
    _neighbours = _neighbours_line(neighbours_by_table.get(table_name, []))
    _connectivity = f"\n{_neighbours}" if _neighbours else ""
    prompt = f"""You are an enterprise data analyst working in this business domain:
{_DOMAIN_DESC}

For each column below, generate ONLY business-meaning metadata, grounded in that
domain. The semantic_type and role are FIXED (do not change them).
When a column references another table (→/← clause), ground its meaning in that
relationship — describe the real-world link it represents, not just "an identifier".

Table: {table_name} — {table_metadata.business_purpose}
Table domain: {table_domain}{_connectivity}

Columns (name | fixed type | fixed role | value hint | relationships):
{chr(10).join(prompt_rows)}

For EACH column output an object with:
  "col_name"
  "business_definition"  (one business-language sentence, consistent with the fixed type)
  "business_role"        (short label e.g. Transaction Amount, Customer Identifier, Status Flag)
  "extra_aliases"        (business synonyms a user might say; 0-6; no keyword stuffing)
  "user_query_patterns"  (3-6 realistic natural-language search phrases)
  "negative_aliases"     (0-5 confusable terms that belong to a DIFFERENT column)
  "business_examples"    (0-3 example questions a user might ask involving this column)
  "column_domain"        (the business entity this column refers to, e.g. Customer, Payment, Document)
  "confidence"           (0.0-1.0)

Rules: realistic language only; no SQL, no statistics, no raw values in any field.
Output ONLY a JSON array, no explanation."""

    response = _call_ollama(prompt=prompt,
                            temperature=COLUMN_UNDERSTANDING_TEMPERATURE,
                            timeout=COLUMN_UNDERSTANDING_TIMEOUT)

    llm_by_col = {}
    if response:
        if response.startswith("```"):
            response = response.split("```")[1]
            response = response[4:] if response.startswith("json") else response
            response = response.strip()
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                llm_by_col = {it.get("col_name", ""): it for it in parsed if isinstance(it, dict)}
        except json.JSONDecodeError as e:
            logger.warning(f"Stage 4 LLM parse failed for {table_name}: {e} — using deterministic-only")

    # ---- 3. Merge: deterministic authoritative + LLM business fields ----
    result = []
    for c in columns_batch:
        cname = c.get("col_name") or c.get("name") or ""
        d = det[cname]
        llm = llm_by_col.get(cname, {})

        llm_conf = float(llm.get("confidence", 0.0) or 0.0)
        # When a column is an unambiguous FK, we KNOW the entity it points to from the
        # relationship graph — use that instead of hoping the LLM echoes it (small models
        # do so inconsistently: same clause yields "User" for one col, "Audit Trail" for
        # another). Falls back to the LLM's own domain for non-FK / polymorphic columns.
        fk_domain = _fk_target_entity(edges_by_col.get((table_name, cname), []))
        col_domain = fk_domain or (llm.get("column_domain") or "").strip()
        # Trust the LLM's column-level domain at a lower bar so domains reflect
        # business concepts (editor_name → User), not just table ownership.
        final_domain = col_domain if (col_domain and (fk_domain or llm_conf >= 0.5)) else table_domain

        from ingestion.deterministic_metadata import _GENERIC_ALIASES
        base_aliases = d["base_aliases"]
        extra = [a.lower().strip() for a in (llm.get("extra_aliases") or []) if isinstance(a, str)]
        # filter generic single-token aliases from the LLM too
        extra = [a for a in extra if a and (" " in a or a not in _GENERIC_ALIASES)]
        merged_aliases = sorted(set(base_aliases) | set(extra))

        # conflict: LLM gave nothing or very low confidence on a business-facing column
        suspect = (not llm and d["importance_class"] != "LOW") or (llm and llm_conf < 0.4)

        result.append(ColumnMetadata(
            col_name=cname,
            table_name=table_name,
            semantic_type=d["semantic_type"],          # deterministic
            analytics_role=d["analytics_role"],        # deterministic
            business_definition=(llm.get("business_definition") or "").strip(),
            aliases=merged_aliases,
            null_meaning="",
            allowed_aggregations=(["SUM", "AVG", "COUNT", "MIN", "MAX"]
                                  if d["analytics_role"] == "MEASURE" else ["COUNT", "GROUP_BY"]),
            confidence=llm_conf or 0.5,
            business_role=(llm.get("business_role") or "").strip(),
            business_domain=final_domain,
            column_domain=col_domain,
            user_query_patterns=[p for p in (llm.get("user_query_patterns") or []) if isinstance(p, str)][:6],
            negative_aliases=[n for n in (llm.get("negative_aliases") or []) if isinstance(n, str)][:5],
            business_examples=[e for e in (llm.get("business_examples") or []) if isinstance(e, str)][:3],
            importance_class=d["importance_class"],    # deterministic
            sql_usage=d["sql_usage"],                  # deterministic
            value_pattern=d["value_info"].get("value_pattern", ""),
            value_handling=d["value_handling"],         # keep | stats | pattern | remove
            sample_values=[str(v) for v in d["value_info"].get("values", [])][:8],  # safe enums only
            contains_pii=(d["value_handling"] == "remove"),
            related_columns=[],                         # filled by post-pass (FK + siblings)
            field_confidence={
                # independent per-field signals, not a uniform blanket score
                "business_definition": round(llm_conf, 2),
                "aliases": round(0.9 if len(base_aliases) >= 2 else max(llm_conf, 0.4), 2),
                "domain": round(llm_conf if col_domain else 0.4, 2),
            },
            validation={"review_status": "suspect" if suspect else "approved",
                        "issues_found": (["low/absent LLM signal"] if suspect else []),
                        "quality_score": round(
                            0.4 * (min(llm_conf, 1.0) if llm.get("business_definition") else 0.3)
                            + 0.3 * (1.0 if len(base_aliases) >= 2 else 0.5)
                            + 0.3 * (0.0 if suspect else 1.0), 2)},
        ))
    return result


def build_domain_synonyms(
    semantic_model: Dict[str, Any],
    typed_cols: Optional[List[Any]] = None,
) -> Dict[str, List[str]]:
    """
    Post-Processing 1: Auto-generate domain synonyms from semantic model.

    Rules (production-tested, zero hardcoding):
    1. Broad terms (payment, amount, date, type) → map to MAX 2 most relevant cols
    2. Specific terms (credit, debit, overdue) → keep all matches
    3. Skip audit columns (_created_at, _updated_by, etc.)
    4. Skip history tables (_history, _log, _audit, _archive, _temp)
    5. Works for ANY domain (no hardcoding)

    Returns:
        {synonym: [table.col, ...]}
    """
    logger.info("Post-Processing 1: Building domain synonyms...")

    from ingestion.deterministic_metadata import _GENERIC_ALIASES

    SKIP_SUFFIX = ('_history', '_log', '_audit', '_archive', '_temp')
    AUDIT_PATTERNS = ('_created_', '_updated_', '_modified_', '_deleted_')
    # Single-token terms that are inherently ambiguous regardless of frequency.
    REJECT = _GENERIC_ALIASES | {
        "summary", "note", "notes", "comment", "comments", "person", "editor",
        "description", "descriptions", "title", "label", "amount", "general",
    }
    MAX_COLS_PER_TERM = 2   # production rule: a term mapping to >2 columns is too broad

    columns = semantic_model.get('columns', {})

    # 1) collect candidate term → set(columns) from each column's (cleaned) aliases
    term_to_cols: Dict[str, set] = {}
    for col_key, col_meta in columns.items():
        table_name = col_meta.get('table_name', '')
        col_name = col_meta.get('col_name', '')
        if any(table_name.lower().endswith(s) for s in SKIP_SUFFIX):
            continue
        if any(p in col_name.lower() for p in AUDIT_PATTERNS):
            continue
        for alias in col_meta.get('aliases', []):
            a = alias.lower().strip()
            if len(a) < 2:
                continue
            # reject generic SINGLE-token terms; keep specific multi-word phrases
            if " " not in a and a in REJECT:
                continue
            term_to_cols.setdefault(a, set()).add(col_key)

    # 2) keep only specific terms that map to ≤ MAX_COLS_PER_TERM columns
    domain_syns = {t: sorted(cs) for t, cs in term_to_cols.items()
                   if len(cs) <= MAX_COLS_PER_TERM}

    dropped = len(term_to_cols) - len(domain_syns)
    logger.info(f"Domain synonyms built: {len(domain_syns)} terms "
                f"(dropped {dropped} over-broad/generic)")
    return domain_syns


def build_concept_graph(
    semantic_model: Dict[str, Any],
    glossary: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Post-Processing 2: Build concept→column mapping graph.

    Used at query time to resolve business concepts like:
    - "PERSON" → payer_name, receiver_name, owner_name
    - "AMOUNT" → amount, price, fee, total, payment
    - "DATE" → date, created_at, timestamp, when

    Deterministic — no LLM involved.

    Returns:
        {concept: [{table, column, role}, ...]}
    """
    logger.info("Post-Processing 2: Building concept graph...")
    concept_graph = {}

    PERSON_PATTERNS = ['name', 'person', 'user', 'owner', 'tenant', 'landlord', 'payer', 'receiver', 'client', 'borrower']
    AMOUNT_PATTERNS = ['amount', 'price', 'fee', 'cost', 'value', 'total', 'sum', 'balance', 'paid', 'due']
    DATE_PATTERNS = ['date', 'time', 'at', 'on', 'when', 'month', 'year', 'created', 'updated', 'timestamp']

    columns = semantic_model.get('columns', {})

    for col_key, col_meta in columns.items():
        table_name = col_meta.get('table_name', '')
        col_name = col_meta.get('col_name', '')
        role = col_meta.get('analytics_role', 'ATTRIBUTE')

        col_lower = col_name.lower()

        # Map patterns to concepts
        for pat in PERSON_PATTERNS:
            if pat in col_lower:
                if 'PERSON' not in concept_graph:
                    concept_graph['PERSON'] = []
                concept_graph['PERSON'].append({
                    'table': table_name,
                    'column': col_name,
                    'role': role
                })
                break

        for pat in AMOUNT_PATTERNS:
            if pat in col_lower and role == 'MEASURE':
                if 'AMOUNT' not in concept_graph:
                    concept_graph['AMOUNT'] = []
                concept_graph['AMOUNT'].append({
                    'table': table_name,
                    'column': col_name,
                    'role': role
                })
                break

        for pat in DATE_PATTERNS:
            if pat in col_lower and role == 'TIME_DIMENSION':
                if 'DATE' not in concept_graph:
                    concept_graph['DATE'] = []
                concept_graph['DATE'].append({
                    'table': table_name,
                    'column': col_name,
                    'role': role
                })
                break

    logger.info(f"Concept graph built: {len(concept_graph)} concepts")
    return concept_graph


def _apply_deterministic_overrides(columns: List[ColumnMetadata]) -> List[ColumnMetadata]:
    """Apply deterministic semantic type overrides (always after Qwen)."""
    for col in columns:
        col_name_lower = col.col_name.lower()

        # Rule 1: _id suffix → IDENTIFIER
        if col_name_lower.endswith("_id"):
            col.analytics_role = "IDENTIFIER"
            col.semantic_type = "IDENTIFIER"

        # Rule 2: _status/_type/_flag suffix → DIMENSION
        if col_name_lower.endswith(("_status", "_type", "_flag")):
            col.analytics_role = "DIMENSION"
            col.allowed_aggregations = ["COUNT", "GROUP_BY"]

        # Rule 3: is_* / has_* prefix → FLAG
        if col_name_lower.startswith(("is_", "has_")):
            col.semantic_type = "FLAG"
            col.analytics_role = "DIMENSION"
            col.allowed_aggregations = ["COUNT", "GROUP_BY"]

    return columns


def stage5_build_retrieval_documents(
    columns: List[ColumnMetadata],
    profiling: Dict[str, Any],
) -> Dict[str, str]:
    """
    Stage 5: Build retrieval documents for BGE-M3 embedding.

    CRITICAL: Field ordering by importance for BGE-M3 truncation.
    BGE-M3 has ~512 token limit, truncates from END.
    Most important fields MUST come FIRST.

    Order:
    1. COLUMN (most important)
    2. ROLE (critical)
    3. DEFINITION (critical)
    4. TERMS/ALIASES (critical for retrieval)
    5. LINKS (FK relationships)
    6. TABLE (context)
    7. VALUES (examples)
    8. RANGE (numeric context)
    9. NULL/DISTINCT (statistics)

    Args:
        columns: List of ColumnMetadata from Stage 4
        profiling: Profiling dict from Stage 1

    Returns:
        {table.column: verbalization_string}
    """
    documents = {}

    for col in columns:
        key = f"{col.table_name}.{col.col_name}"
        prof_key = key

        if prof_key not in profiling:
            prof = {
                "null_percentage": 0,
                "distinct_count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "top_values": [],
            }
        else:
            prof = profiling[prof_key]

        # Build parts in PRIORITY ORDER (most important first)
        parts = []

        # 1. COLUMN (CRITICAL - comes first)
        parts.append(f"COLUMN: {col.col_name.replace('_', ' ')}")

        # 2. ROLE (CRITICAL)
        parts.append(f"ROLE: {col.analytics_role}")

        # 3. DEFINITION (CRITICAL)
        if col.business_definition:
            parts.append(f"DEFINITION: {col.business_definition}")

        # 4. ALIASES/TERMS (CRITICAL for retrieval)
        if col.aliases:
            parts.append(f"TERMS: {', '.join(col.aliases)}")

        # 4b. USER QUERY PATTERNS — real search language (CRITICAL for retrieval recall)
        patterns = getattr(col, "user_query_patterns", None)
        if patterns:
            parts.append(f"SEARCH: {', '.join(patterns)}")

        # 4c. BUSINESS ROLE + DOMAIN (concept context, improves routing)
        if getattr(col, "business_role", ""):
            parts.append(f"BUSINESS ROLE: {col.business_role}")
        if getattr(col, "business_domain", ""):
            parts.append(f"DOMAIN: {col.business_domain}")

        # 5. RELATED COLUMNS (relationship context — often beats more aliases)
        related = getattr(col, "related_columns", None)
        if related:
            parts.append(f"RELATED: {', '.join(related[:5])}")

        # 6. TABLE CONTEXT
        parts.append(f"TABLE: {col.table_name.replace('_', ' ')}")

        # 7. SEMANTIC TYPE (metadata)
        parts.append(f"TYPE: {col.semantic_type}")

        # 8. VALUES — LEAKAGE-SAFE: only sanitized enum values; never raw PII /
        #    identifiers / free-text. Withheld columns expose only a format label.
        handling = getattr(col, "value_handling", "stats")
        if handling == "keep" and getattr(col, "sample_values", None):
            parts.append(f"VALUES: {', '.join(str(v) for v in col.sample_values[:8])}")
        elif getattr(col, "value_pattern", ""):
            parts.append(f"FORMAT: {col.value_pattern}")

        # 9. NUMERIC RANGE (for MEASURE columns)
        if col.analytics_role == "MEASURE":
            min_val = prof.get("min")
            max_val = prof.get("max")
            avg_val = prof.get("avg")
            if min_val is not None and max_val is not None:
                parts.append(f"RANGE: {min_val} to {max_val}")
            if avg_val is not None:
                parts.append(f"AVG: {avg_val}")

        # 10. STATISTICS (least important - may be truncated)
        null_pct = prof.get("null_percentage", 0)
        distinct_cnt = prof.get("distinct_count", 0)
        parts.append(f"NULL: {null_pct}%")
        parts.append(f"DISTINCT: {distinct_cnt}")

        # Join all parts with pipe separator
        doc = " | ".join(parts)
        documents[key] = doc

    return documents


def run_full_semantic_layer(
    schema_dict: Dict[str, Any],
    profiling: Optional[Dict[str, Any]] = None,
    glossary: Optional[Dict[str, str]] = None,
    force_glossary: bool = False,
) -> Dict[str, Any]:
    """
    Run full L2 semantic layer (Stages 1-5 + Post-processing).

    Args:
        schema_dict: {table_name: {columns: [...]}}
        profiling: Optional profiling dict (if None, Stage 1 runs)
        glossary: Optional glossary dict (if None, Stage 2 runs)
        force_glossary: Force regeneration of glossary

    Returns:
        {table: {...}, columns: [...], retrieval_documents: {...},
         domain_synonyms: {...}, concept_graph: {...}}
    """
    logger.info(f"Starting L2 semantic layer (stages 1-5 + post-processing) on {len(schema_dict)} tables...")
    start_time = time.time()

    # STAGE 1: Data Profiling (if not provided)
    if profiling is None:
        logger.info("Stage 1: Data Profiling...")
        try:
            conn = data_profiler._get_db_connection()
            profiling = data_profiler.run_profiling(conn, schema_dict)
            conn.close()
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}, using empty profiling")
            profiling = {}
    else:
        logger.info("Stage 1: Using provided profiling")

    # STAGE 2: Database Glossary (if not provided)
    if glossary is None:
        logger.info("Stage 2: Generating Database Glossary...")
        try:
            table_names = list(schema_dict.keys())
            glossary = glossary_builder.load_or_generate_glossary(
                table_names=table_names, force=force_glossary
            )
            if glossary is None:
                glossary = {}
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}, using empty glossary")
            glossary = {}
    else:
        logger.info("Stage 2: Using provided glossary")

    semantic_model = {
        "version": "2.0",
        "tables": {},
        "columns": {},
        "retrieval_documents": {},
    }

    # --- Checkpoint resume: restore generated work from a prior (interrupted) run ---
    _fp = _schema_fingerprint(schema_dict)
    _ckpt = _load_checkpoint(_fp)
    try:
        from config import SEMANTIC_CHECKPOINT_EVERY as _ckpt_every
    except Exception:
        _ckpt_every = 5

    _parallel    = _parallel_qwen_enabled()
    _t_stage34   = time.time()   # Stage 3+4 timer (perf comparison: sequential vs parallel)

    # Stage 3: Table Understanding (Qwen × batches of 3 tables)
    logger.info("Stage 3: Table Understanding (batched, 3 tables per call)...")
    table_metadata = {}
    all_columns = []   # may be pre-filled from a checkpoint below (consumed by Stage 4)

    if _ckpt:
        for _n, _d in _ckpt.get("tables", {}).items():
            try:
                _m = TableMetadata(**_d)
                table_metadata[_n] = _m
                semantic_model["tables"][_n] = asdict(_m)
            except Exception:
                pass
        for _cd in _ckpt.get("columns", []):
            try:
                all_columns.append(ColumnMetadata(**_cd))
            except Exception:
                pass

    tables_list = list(schema_dict.items())
    batch_size = 3
    _pending3   = [(n, info) for n, info in tables_list if n not in table_metadata]
    _since_ckpt = 0

    def _stage3_default(table_name):
        logger.warning(f"  Failed to understand table {table_name}, using defaults")
        return TableMetadata(
            table_name=table_name, business_purpose="", primary_entity="",
            table_type="TRANSACTION", candidate_temporal_columns=[],
            candidate_measure_columns=[],
        )

    if not _parallel:
        # ── Sequential (default — unchanged behaviour) ──────────────────────────
        for i in range(0, len(_pending3), batch_size):
            batch = _pending3[i : i + batch_size]
            batch_tuples = [(name, info.get("columns", [])) for name, info in batch]
            batch_names = [name for name, _ in batch]

            logger.info(f"  Understanding tables {len(table_metadata)+1}..{len(table_metadata)+len(batch)} of {len(tables_list)}: {', '.join(batch_names[:3])}")

            metadatas = stage3_batch_table_understanding(batch_tuples, glossary)

            for table_name, metadata in zip(batch_names, metadatas):
                if metadata:
                    table_metadata[table_name] = metadata
                    semantic_model["tables"][table_name] = asdict(metadata)
                else:
                    table_metadata[table_name] = _stage3_default(table_name)

            _since_ckpt += len(batch)
            if _since_ckpt >= _ckpt_every:
                _save_checkpoint(_fp, table_metadata, all_columns)
                _since_ckpt = 0
    elif _pending3:
        # ── Parallel (opt-in) — same batch composition, run concurrently ────────
        _batches = [_pending3[i:i + batch_size] for i in range(0, len(_pending3), batch_size)]
        _workers = _resolve_qwen_workers(len(_batches))
        logger.info(f"  ⚡ Parallel Qwen ENABLED — Workers: {_workers}  ·  Processing {len(_pending3)} tables  ·  {len(_batches)} batches")
        try:
            _build_edge_index(_load_relationship_graph())   # pre-warm shared caches on main thread
        except Exception:
            pass
        _done = _ck_mark = 0
        _set_qwen_circuit(_new_circuit())   # breaker active only during this parallel block
        from veda_core.context import try_current as _try_ctx, with_context as _with_ctx
        _pctx = _try_ctx()  # carry ambient (source, tenant) into worker threads (§4.1)
        with ThreadPoolExecutor(max_workers=_workers) as _ex:
            _futs = {}
            for _b in _batches:
                _bt = [(name, info.get("columns", [])) for name, info in _b]
                _bn = [name for name, _ in _b]
                _futs[_ex.submit(_with_ctx(_pctx, _stage3_batch_worker), _bt, _bn, glossary)] = _bn
            for _fut in as_completed(_futs):
                _bn = _futs[_fut]
                try:
                    _pairs = _fut.result()
                except Exception as e:
                    logger.error(f"  Stage 3 batch failed {_bn}: {type(e).__name__}: {e}")
                    _pairs = [(n, None) for n in _bn]
                for _name, _md in _pairs:
                    table_metadata[_name] = _md if _md else _stage3_default(_name)
                    logger.info(f"  Completed {_name}")
                    _done += 1
                if _done - _ck_mark >= _ckpt_every:        # checkpoint on the main thread only
                    _save_checkpoint(_fp, table_metadata, all_columns)
                    _ck_mark = _done
        _set_qwen_circuit(None)             # deactivate breaker after the parallel block
        # Deterministic write order: rebuild tables in schema order (parallel-safe)
        for _name, _info in tables_list:
            if _name in table_metadata:
                semantic_model["tables"][_name] = asdict(table_metadata[_name])

    # Stage 4: Column Understanding (Qwen × batches/5)
    logger.info("Stage 4: Column Understanding...")
    # all_columns may already hold checkpoint-restored columns — skip those tables.
    _done_cols_tables = {c.table_name for c in all_columns}
    if _done_cols_tables:
        logger.info(f"  Resuming Stage 4 — {len(_done_cols_tables)} tables already have columns.")
    _since_ckpt = 0

    if not _parallel:
        # ── Sequential (default — unchanged behaviour) ──────────────────────────
        for table_name, table_info in schema_dict.items():
            if table_name in _done_cols_tables:
                continue
            columns = table_info.get("columns", [])

            # Process in batches
            for i in range(0, len(columns), COLUMN_UNDERSTANDING_BATCH_SIZE):
                batch = columns[i : i + COLUMN_UNDERSTANDING_BATCH_SIZE]
                logger.info(f"  Understanding {table_name} columns {i}-{i + len(batch)}")

                col_metadata = stage4_column_understanding(
                    columns_batch=batch,
                    table_name=table_name,
                    table_metadata=table_metadata[table_name],
                    glossary=glossary,
                    profiling=profiling,
                )

                if col_metadata:
                    # Apply deterministic overrides
                    col_metadata = _apply_deterministic_overrides(col_metadata)
                    all_columns.extend(col_metadata)

            # Checkpoint per-table (a table is fully done here), every N tables.
            _since_ckpt += 1
            if _since_ckpt >= _ckpt_every:
                _save_checkpoint(_fp, table_metadata, all_columns)
                _since_ckpt = 0
    else:
        # ── Parallel (opt-in) — one task per table, merged on the MAIN thread in
        # SCHEMA ORDER so the output is byte-identical to the sequential run. Each
        # worker uses the UNCHANGED per-batch calls + deterministic overrides. ──
        _pending4 = [(n, info) for n, info in schema_dict.items()
                     if n not in _done_cols_tables]
        _workers4 = _resolve_qwen_workers(len(_pending4)) if _pending4 else 1
        logger.info(f"  ⚡ Parallel Qwen (Stage 4) ENABLED — Workers: {_workers4}  ·  "
                    f"Processing {len(_pending4)} tables")
        _results4: Dict[str, List[ColumnMetadata]] = {}
        _done4 = _ck4 = 0
        if _pending4:
            _set_qwen_circuit(_new_circuit())   # breaker active only during this parallel block
            from veda_core.context import try_current as _try_ctx4, with_context as _with_ctx4
            _pctx4 = _try_ctx4()  # carry ambient (source, tenant) into worker threads (§4.1)
            with ThreadPoolExecutor(max_workers=_workers4) as _ex:
                _futs = {_ex.submit(_with_ctx4(_pctx4, _stage4_table_worker), _n, _info.get("columns", []),
                                    table_metadata[_n], glossary, profiling): _n
                         for _n, _info in _pending4}
                for _fut in as_completed(_futs):
                    _n = _futs[_fut]
                    try:
                        _tn, _cols = _fut.result()
                    except Exception as e:
                        logger.error(f"  Stage 4 failed {_n}: {type(e).__name__}: {e}")
                        _tn, _cols = _n, []
                    _results4[_tn] = _cols
                    logger.info(f"  Completed {_tn}")
                    _done4 += 1
                    # Checkpoint on the MAIN thread only, using a schema-ordered partial.
                    if _done4 - _ck4 >= _ckpt_every:
                        _partial = list(all_columns)
                        for _pn, _ in _pending4:
                            if _pn in _results4:
                                _partial.extend(_results4[_pn])
                        _save_checkpoint(_fp, table_metadata, _partial)
                        _ck4 = _done4
            _set_qwen_circuit(None)             # deactivate breaker after the parallel block
        # Deterministic merge: extend in schema order (identical to sequential).
        for _n, _ in _pending4:
            if _n in _results4:
                all_columns.extend(_results4[_n])

    # Stage 3+4 timing — log mode + elapsed so sequential vs parallel runs are comparable.
    _stage34_elapsed = time.time() - _t_stage34
    logger.info(f"  ⏱  Stage 3+4 [{'parallel' if _parallel else 'sequential'}] finished in "
                f"{_stage34_elapsed / 60:.1f} min ({_stage34_elapsed:.0f}s)"
                + (f"  ·  workers={_resolve_qwen_workers(len(tables_list))}" if _parallel else ""))

    # POST-PASS A: alias precision filter — drop aliases shared by too many
    # columns (they hurt precision). Each column keeps its specific full phrase.
    from collections import Counter
    alias_freq = Counter(a for col in all_columns for a in set(col.aliases))
    freq_threshold = max(8, int(0.10 * max(len(all_columns), 1)))
    dropped_generic = 0
    for col in all_columns:
        kept = [a for a in col.aliases if alias_freq[a] <= freq_threshold]
        dropped_generic += len(col.aliases) - len(kept)
        col.aliases = kept or col.aliases[:1]  # never leave a column alias-less
    logger.info(f"  Alias filter: dropped {dropped_generic} over-frequent aliases "
                f"(threshold > {freq_threshold} columns)")

    # POST-PASS B: related columns — FK targets + same-table HIGH-importance siblings.
    fk_map = {}
    by_table = {}
    for tname, tinfo in schema_dict.items():
        for c in tinfo.get("columns", []):
            cn = c.get("col_name") or c.get("name")
            ref_t, ref_c = c.get("fk_ref_table"), c.get("fk_ref_col")
            if ref_t:
                fk_map[f"{tname}.{cn}"] = f"{ref_t}.{ref_c or 'id'}"
    for col in all_columns:
        by_table.setdefault(col.table_name, []).append(col)
    for col in all_columns:
        rel = []
        fk = fk_map.get(f"{col.table_name}.{col.col_name}")
        if fk:
            rel.append(fk)
        sibs = [f"{col.table_name}.{s.col_name}" for s in by_table.get(col.table_name, [])
                if s.col_name != col.col_name and s.importance_class == "HIGH"][:3]
        col.related_columns = (rel + sibs)[:5]

    # Rebuild the columns dict from the post-processed records.
    semantic_model["columns"] = {f"{c.table_name}.{c.col_name}": asdict(c) for c in all_columns}

    # Stage 5: Build Retrieval Documents (ENHANCED - priority-ordered fields)
    logger.info("Stage 5: Building Retrieval Documents...")
    retrieval_documents = stage5_build_retrieval_documents(all_columns, profiling)
    semantic_model["retrieval_documents"] = retrieval_documents

    # POST-PROCESSING 1: Domain Synonyms
    logger.info("Post-Processing 1: Building Domain Synonyms...")
    domain_synonyms = build_domain_synonyms(semantic_model)
    semantic_model["domain_synonyms"] = domain_synonyms

    # POST-PROCESSING 2: Concept Graph
    logger.info("Post-Processing 2: Building Concept Graph...")
    concept_graph = build_concept_graph(semantic_model, glossary)
    semantic_model["concept_graph"] = concept_graph

    # Save glossary
    logger.info("Saving glossary...")
    glossary_builder.save_glossary(glossary)

    # Save domain synonyms
    logger.info("Saving domain synonyms...")
    os.makedirs(os.path.dirname(DOMAIN_SYNONYMS_FILE) or ".", exist_ok=True)
    with open(DOMAIN_SYNONYMS_FILE, "w") as f:
        json.dump(domain_synonyms, f, indent=2)
    logger.info(f"Domain synonyms saved to {DOMAIN_SYNONYMS_FILE}")

    # Save concept graph
    logger.info("Saving concept graph...")
    os.makedirs(os.path.dirname(CONCEPT_GRAPH_FILE) or ".", exist_ok=True)
    with open(CONCEPT_GRAPH_FILE, "w") as f:
        json.dump(concept_graph, f, indent=2)
    logger.info(f"Concept graph saved to {CONCEPT_GRAPH_FILE}")

    elapsed = time.time() - start_time
    logger.info(f"L2 semantic layer complete: {len(all_columns)} columns in {elapsed:.1f}s")
    logger.info(f"  Stage 1: Profiling ({len(profiling)} columns profiled)")
    logger.info(f"  Stage 2: Glossary ({len(glossary)} terms)")
    logger.info(f"  Stages 3-5: {len(semantic_model['tables'])} tables, {len(all_columns)} columns")
    logger.info(f"  Post-proc: {len(domain_synonyms)} synonyms, {len(concept_graph)} concepts")

    _clear_checkpoint()   # full build succeeded — drop the resume checkpoint
    return semantic_model


def save_semantic_model(semantic_model: Dict[str, Any], output_file: str = None):
    """Save semantic model to JSON file."""
    if output_file is None:
        output_file = SEMANTIC_MODEL_FILE

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(semantic_model, f, indent=2)

    logger.info(f"Semantic model saved to {output_file}")


def load_semantic_model(input_file: str = None) -> Dict[str, Any]:
    """Load semantic model from JSON file."""
    if input_file is None:
        input_file = SEMANTIC_MODEL_FILE

    if not os.path.exists(input_file):
        logger.warning(f"Semantic model file not found: {input_file}")
        return {}

    with open(input_file, "r") as f:
        model = json.load(f)

    logger.info(f"Semantic model loaded from {input_file}")
    return model
