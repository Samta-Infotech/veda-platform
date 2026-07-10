# =============================================================================
# semantic/registry.py
# VEDA — in-memory loader + matchers for the compiled semantic registries.
#
# Loads concepts.json / dimensions.json / metrics.json once per process and
# exposes pure-lookup matchers (no embeddings, no LLM). Everything here is
# deterministic and microsecond-cheap; the registries are the source of truth.
# =============================================================================

import os
import sys
import json
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Resolve the compiled registries through config's scoped artifact paths so the
# reader and the writer (compile_semantic_layer) always agree. Falls back to the
# legacy flat _HERE location if config is unavailable, so the loader can never
# crash on import.
try:
    from config import CONCEPTS_FILE, DIMENSIONS_FILE, METRICS_FILE
    _REG_FILES = {
        "concepts.json":   CONCEPTS_FILE,
        "dimensions.json": DIMENSIONS_FILE,
        "metrics.json":    METRICS_FILE,
    }
except Exception:
    _REG_FILES = {}

# Per-(source, tenant) registry cache. The query tier is a warm worker that serves
# N ready sources, so the registries must be scope-keyed exactly like the semantic
# model (veda_hybrid._SM) — a single global would serve one source's concepts to
# every source. `_STATE` is kept as a LIVE MIRROR of the active scope so external
# readers that reference `registry._STATE[...]` (query/intent_envelope, query/fast_path)
# keep working; _active() refreshes it on every access.
_CACHE: dict = {}   # (source_id, tenant) → {"concepts","dimensions","metrics","source_hash"}
_STATE = {"loaded": False, "concepts": {}, "dimensions": {}, "metrics": {},
          "source_hash": None}

_CONNECTIVES = {"and", "or", "of", "to", "by", "the", "a", "an", "in", "on",
                "for", "with", "per", "each", "show", "list", "give", "me",
                "all", "get", "find", "how", "many", "much", "what", "is",
                "are", "number", "count", "total", "average", "sum"}


def _singularize(word: str) -> str:
    try:
        from retrieval.query_enrichment import _singularize as _s
        out = _s(word)
        if out:                       # guard: never let a None/empty slip through
            return out
    except Exception:
        pass
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def query_tokens(query: str) -> set:
    """Return BOTH raw and singularized tokens. Carrying both forms makes concept /
    dimension / value matching robust to any singularizer drift between compile-time
    (where registry tokens were built) and runtime — a mismatch there would otherwise
    empty every intersection and make the whole fast path silently fall through."""
    # Keep pure-digit tokens regardless of length: a value like "Level 1" / "Tier 2"
    # carries its meaning in the digit, and dropping it collapses "Level 1" ≡ "Level 2".
    raw = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2 or w.isdigit()}
    return raw | {_singularize(w) for w in raw}


def _reg_scope():
    """(source_id, tenant) for the registry cache / redis key. Mirrors
    veda_hybrid._sm_scope so the SQL head and the fast path agree on scope: prefers
    the ambient per-request context, falling back to the env pin (single-source dev)."""
    try:
        from context import try_current
        ctx = try_current()
        if ctx is not None:
            return (str(ctx.source_id), str(ctx.tenant))
    except Exception:
        pass
    return (os.environ.get("VEDA_SM_SOURCE_ID", "1"),
            os.environ.get("VEDA_SM_TENANT", "default"))


def _load_file(name):
    path = _REG_FILES.get(name) or os.path.join(_HERE, name)
    if not os.path.exists(path):
        return {}, None
    blob = json.load(open(path))
    return blob.get("items", {}), blob.get("source_hash")


def _load_from_redis(scope):
    """Load {concepts,dimensions,metrics} for this scope from redis-cache (published by
    storage_adapters.assembler.publish_registry at warm time), keyed by (source, tenant).
    Gated by the same VEDA_SM_REDIS flag as the semantic model. Returns a state dict, or
    None to fall back to the on-disk files (dev / cache miss / flag off)."""
    if os.environ.get("VEDA_SM_REDIS", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        import redis as _redis
        url = os.environ.get("REDIS_CACHE_URL", "redis://redis-cache:6379/0")
        source_id, tenant = scope
        raw = _redis.Redis.from_url(url).get(f"veda:registry:{source_id}:{tenant}")
        if not raw:
            return None
        blob = json.loads(raw)
        c = blob.get("concepts", {}) or {}
        d = blob.get("dimensions", {}) or {}
        m = blob.get("metrics", {}) or {}
        return {"loaded": True,
                "concepts":   c.get("items", {}) or {},
                "dimensions": d.get("items", {}) or {},
                "metrics":    m.get("items", {}) or {},
                "source_hash": c.get("source_hash")}
    except Exception:
        return None


def _load_scope(scope):
    """Build the state dict for a scope: redis first (scoped), on-disk files as fallback."""
    st = _load_from_redis(scope)
    if st is not None:
        return st
    c, h1 = _load_file("concepts.json")
    d, _  = _load_file("dimensions.json")
    m, _  = _load_file("metrics.json")
    return {"loaded": True, "concepts": c, "dimensions": d, "metrics": m, "source_hash": h1}


def _active() -> dict:
    """Return the registry state for the current (source, tenant) scope, loading and
    caching it on first access. Also refreshes the `_STATE` mirror so external readers
    of `registry._STATE[...]` see the active scope."""
    scope = _reg_scope()
    st = _CACHE.get(scope)
    if st is None:
        st = _load_scope(scope)
        _CACHE[scope] = st
    if _STATE is not st:            # keep the legacy mirror pointed at the active scope
        _STATE.clear()
        _STATE.update(st)
    return st


def load(force: bool = False) -> bool:
    """Ensure the current scope's registries are loaded. Returns True if a non-empty
    layer loaded. `force` drops this scope's cache so the next access reloads."""
    if force:
        _CACHE.pop(_reg_scope(), None)
    st = _active()
    return bool(st["concepts"] or st["metrics"])


def clear() -> None:
    """Drop every cached scope — called on rehydrate / re-ingest so the next query
    reloads the fresh registries from redis (or file)."""
    _CACHE.clear()
    _STATE.clear()
    _STATE.update({"loaded": False, "concepts": {}, "dimensions": {}, "metrics": {},
                   "source_hash": None})


def is_ready() -> bool:
    st = _active()
    return bool(st["concepts"] and st["metrics"])


def active() -> dict:
    """Public accessor for the CURRENT (source, tenant) scope's registry state —
    {"concepts","dimensions","metrics","source_hash"}. Loads/caches on first access.
    External readers must use this (not the `_STATE` mirror) to be scope-correct."""
    return _active()


# ---------------------------------------------------------------------------
# Matchers — all return explainable, scored results; ambiguity is surfaced, not hidden
# ---------------------------------------------------------------------------

def match_concepts(qtoks: set) -> list:
    """Return [(concept, score)] for entity concepts whose tokens appear in the query,
    best first. score = (#matched tokens, coverage)."""
    hits = []
    for c in _active()["concepts"].values():
        ctoks = set(c.get("match_tokens", []))
        if not ctoks:
            continue
        matched = ctoks & qtoks
        if not matched:
            continue
        coverage = len(matched) / len(ctoks)
        hits.append((c, (len(matched), round(coverage, 3))))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def get_metric(metric_id: str):
    return _active()["metrics"].get(metric_id)


def match_metric_labels(query_l: str) -> list:
    """Direct label match for non-count metrics (SUM/AVG) that have no entity noun.
    Returns [(metric, n_label_tokens_matched)]."""
    out = []
    for m in _active()["metrics"].values():
        if m.get("kind") == "COUNT":
            continue                      # COUNT resolved via concept, not labels
        for lab in m.get("labels", []):
            if lab and lab in query_l:
                out.append((m, len(lab.split())))
                break
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def dimensions_for_table(table: str) -> list:
    return [d for d in _active()["dimensions"].values() if d.get("owner_table") == table]


def match_dimension_in_table(table: str, qtoks: set, query_l: str):
    """Best dimension on `table` named by the query (by label/token), or None."""
    best, best_score = None, 0
    for d in dimensions_for_table(table):
        score = 0
        for lab in d.get("labels", []):
            if lab and lab in query_l:
                score = max(score, len(lab.split()) + 1)
        dtoks = {_singularize(t) for t in d["col_name"].split("_") if len(t) > 2}
        score = max(score, len(dtoks & qtoks))
        if score > best_score:
            best, best_score = d, score
    return best if best_score > 0 else None


def match_dimensions_in_table(table: str, qtoks: set, query_l: str, k: int = 2):
    """Up to `k` DISTINCT dimensions on `table` named by the query, best-scoring first.
    Powers multi-dimension grouping ("<metric> per X by Y") in the deterministic fast
    path — a strict superset of match_dimension_in_table (k=1 returns the same single
    best dimension). Only dimensions that actually score (named by the query) are
    returned, so a single-dimension query still yields exactly one."""
    scored = []
    for d in dimensions_for_table(table):
        score = 0
        for lab in d.get("labels", []):
            if lab and lab in query_l:
                score = max(score, len(lab.split()) + 1)
        dtoks = {_singularize(t) for t in d["col_name"].split("_") if len(t) > 2}
        score = max(score, len(dtoks & qtoks))
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda sd: sd[0], reverse=True)
    out, seen = [], set()
    for _, d in scored:
        if d["col_name"] in seen:
            continue
        seen.add(d["col_name"])
        out.append(d)
        if len(out) >= k:
            break
    return out


def match_values_in_table(table: str, qtoks: set):
    """ALL dimension VALUES on `table` named by the query → (dimension, [values]).
    Multiple matched values on one dimension = an OR filter ("open or new" →
    IN ('Open','New')). Case-insensitive token match; exact DB casing emitted."""
    for d in dimensions_for_table(table):
        if not d.get("filterable", True):
            continue
        matched = []
        for v in d.get("values", []):
            vl = str(v).lower()
            vtoks = {_singularize(t) for t in re.findall(r"[a-z0-9]+", vl) if len(t) > 2 or t.isdigit()}
            if vtoks and vtoks <= qtoks:
                matched.append(str(v))
        if matched:
            return d, matched
    return None


def match_value_in_table(table: str, qtoks: set):
    """Single-value form of match_values_in_table (first matched value)."""
    hit = match_values_in_table(table, qtoks)
    return (hit[0], hit[1][0]) if hit else None
