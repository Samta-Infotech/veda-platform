"""query/datalake_values.py — QUERY-TIME datalake value grounding (flag-gated, ingestion-safe).

The value-arbiter (query/value_arbiter.py) grounds a filter token as a VALUE only if it appears in the
`column_values` store, which ingestion builds from the RELATIONAL DB only. Datalake parquet values
(e.g. vendors.city='Kochi') are never sampled there, so a datalake filter/group query gets refused
("is 'Kochi' a column or a value?"). This module samples a datalake source's parquet TEXT columns'
DISTINCT values ON DEMAND (read-only DuckDB) and WRAPS the arbiter's typed value lookup so those
values ground for THIS query — without touching ingestion, the `column_values` store, or any shared
engine cache.

Flag: DATALAKE_VALUE_GROUNDING_ENABLED (default OFF → this module is never entered).
"""
from __future__ import annotations
from query.cross_source_composer import resolve_surface
import re
from typing import Callable, Dict, List, Optional, Tuple
import duckdb

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:                                    # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

# token -> [(table, col, semantic_type, value_raw)] — the arbiter's TypedLookup shape.
_ValueIndex = Dict[str, List[Tuple[str, str, str, str]]]

# Cache the sampled index per (sorted source-id set, tenant) so a request samples parquet once, not
# per token. Small bounded dict — cleared implicitly by process lifetime; never mutated across scopes.
_CACHE: Dict[Tuple[Tuple[str, ...], str], _ValueIndex] = {}
_CACHE_MAX = 32





def is_enabled() -> bool:
    try:
        from config import DATALAKE_VALUE_GROUNDING_ENABLED
        return bool(DATALAKE_VALUE_GROUNDING_ENABLED)
    except Exception:
        return False


def _sample_limit() -> int:
    try:
        from config import DATALAKE_VALUE_SAMPLE_LIMIT
        return int(DATALAKE_VALUE_SAMPLE_LIMIT)
    except Exception:
        return 500


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())

def _current_scope() -> Tuple[List[str], str, dict]:
    """(source_ids, tenant, profiles) from the ambient request context. Empty on any failure."""
    for modname in ("veda_core.context", "context"):
        try:
            mod = __import__(modname, fromlist=["try_current", "current_source_profiles"])
            ctx = mod.try_current()
            if ctx is None:
                continue
            sids = [str(s) for s in (getattr(ctx, "source_ids", ()) or ())]
            tenant = str(getattr(ctx, "tenant", "default"))
            profiles = mod.current_source_profiles() or {}
            return sids, tenant, profiles
        except Exception:
            continue
    return [], "default", {}


def _datalake_source_ids(source_ids: List[str], profiles: dict) -> List[str]:
    """The in-scope source ids whose profile kind is a tabular datalake."""
    out = []
    for sid in source_ids:
        st = str((profiles or {}).get(str(sid), {}).get("source_type", "")).lower()
        if st == "datalake":
            out.append(str(sid))
    return out


def _sample_source(source_id: str, tenant: str, limit: int) -> _ValueIndex:
    """DuckDB-sample DISTINCT values of a datalake source's TEXT parquet columns → value index.
    Read-only; best-effort → {} on any failure. Only TEXT/VARCHAR columns are sampled (numeric
    columns like amount/rating are not filter *values* in this sense) and treated as CATEGORY so the
    arbiter grounds them as a filter VALUE."""
    idx: _ValueIndex = {}
    try:
        
        surf = resolve_surface(str(source_id), tenant)
        if surf is None or getattr(surf, "kind", "") != "parquet":
            return {}
        tables = getattr(surf, "tables", {}) or {}
        if not tables:
            return {}
        
        try:
            from config import DATALAKE_DUCKDB_MEMORY_LIMIT as _mem
        except Exception:
            _mem = "1GB"
        conn = duckdb.connect()
        try:
            try:
                conn.execute(f"SET memory_limit='{_mem}';")
            except Exception:
                pass
            for tname, path in tables.items():
                # column names + types
                try:
                    cols = conn.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
                except Exception:
                    continue
                for cinfo in cols:
                    cname = cinfo[0]
                    ctype = str(cinfo[1] or "").upper()
                    if "CHAR" not in ctype and "STRING" not in ctype and "TEXT" not in ctype:
                        continue                      # only text columns are filter values here
                    try:
                        rows = conn.execute(
                            f'SELECT DISTINCT "{cname}" FROM read_parquet(\'{path}\') '
                            f'WHERE "{cname}" IS NOT NULL LIMIT {int(limit)}').fetchall()
                    except Exception:
                        continue
                    for (val,) in rows:
                        if val is None:
                            continue
                        vn = _norm(val)
                        if not vn:
                            continue
                        idx.setdefault(vn, []).append((tname, cname, "CATEGORY", str(val)))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        logger.debug("datalake_values: sample failed for src %s (%s)", source_id, e)
        return {}
    return idx


def datalake_value_index(source_ids: List[str], tenant: str, profiles: dict) -> _ValueIndex:
    """Merged value index for all in-scope datalake sources, cached per (source-set, tenant)."""
    dl = _datalake_source_ids(source_ids, profiles)
    if not dl:
        return {}
    key = (tuple(sorted(dl)), tenant)
    if key in _CACHE:
        return _CACHE[key]
    merged: _ValueIndex = {}
    limit = _sample_limit()
    for sid in dl:
        for vn, cands in _sample_source(sid, tenant, limit).items():
            merged.setdefault(vn, []).extend(cands)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = merged
    return merged


def augment_lookup(base_lookup: Callable[[str], list]) -> Callable[[str], list]:
    """Wrap the arbiter's typed value lookup: return base results first; when the base finds nothing,
    fall back to the datalake parquet-sampled values for the in-scope datalake sources. When the flag
    is off or no datalake source is in scope, returns ``base_lookup`` unchanged (byte-identical)."""
    if not is_enabled():
        return base_lookup
    source_ids, tenant, profiles = _current_scope()
    if not _datalake_source_ids(source_ids, profiles):
        return base_lookup
    idx = datalake_value_index(source_ids, tenant, profiles)
    if not idx:
        return base_lookup

    def _wrapped(token: str):
        base = base_lookup(token) or []
        if base:
            return base
        return list(idx.get(_norm(token), []))

    return _wrapped


def clear_cache():
    """Test helper — drop the per-scope sample cache."""
    _CACHE.clear()
