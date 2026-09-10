# =============================================================================
# veda/source_names.py
# VEDA — safe source display-name projection (traceability Phase 1, Part 6).
#
# THE PROBLEM
#   The engine only ever sees a numeric source id. `Source.name` lives in the
#   Django `apps.sources` registry, which veda_core must never import (the api
#   tier / engine boundary — see apps/query/inference_client.py's docstring). So
#   every explanation the engine could build said "2".
#
# THE BOUNDARY
#   Names arrive the way every other cross-tier fact already does: the api tier
#   resolves them ONCE (apps.query.scope.source_profiles_for) and forwards them
#   in the X-Veda-Source-Profiles header, which the inference middleware binds to
#   the `current_source_profiles()` ContextVar. This module is the ONLY reader.
#   No DB join is spread through the pipeline — the execution layer keeps using
#   source_id, and only the user-facing projection resolves a name.
#
# AUTHORIZATION
#   The profile map contains exactly the sources the api tier already authorised
#   for THIS request (resolve_query_scope -> permitted_source_ids). An id absent
#   from it is either out of scope or not permitted, and either way this module
#   must not invent a name for it — `display_name` falls back to a generic
#   "a data source", never to the raw id, so a projection can never disclose the
#   existence or naming of a source the caller was not told about.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Engine source KIND -> the word a user sees. The engine's kinds are an internal
#: grammar (relational/datalake/document/nosql); these are product copy.
TYPE_LABELS: Dict[str, str] = {
    "relational": "Database",
    "datalake": "Data Lake",
    "document": "Documents",
    "nosql": "Records",
}

#: What we call a source we cannot safely name. Deliberately NOT the id.
GENERIC_NAME = "a data source"
GENERIC_TYPE = "Data source"


def _profiles() -> Dict[str, dict]:
    """The request-scoped {source_id: profile} map, or {} when nothing was bound."""
    try:
        from veda_core.context import current_source_profiles
    except Exception:
        try:
            from context import current_source_profiles
        except Exception:
            return {}
    try:
        return current_source_profiles() or {}
    except Exception:
        return {}


def profile_for(source_id) -> dict:
    """The raw profile dict for a source id, or {} when it is unknown/unauthorised."""
    if source_id is None:
        return {}
    return _profiles().get(str(source_id)) or {}


def display_name(source_id, default: Optional[str] = None) -> str:
    """The user-facing name for a source.

    Falls back to a GENERIC label — never the numeric id and never a schema or
    connection detail — when the source is unknown to this request's profile map.
    """
    name = str((profile_for(source_id) or {}).get("name") or "").strip()
    return name or (default if default is not None else GENERIC_NAME)


def display_type(source_id, default: Optional[str] = None) -> str:
    """The user-facing type label ("Database", "Documents", …)."""
    kind = str((profile_for(source_id) or {}).get("source_type") or "").strip().lower()
    return TYPE_LABELS.get(kind) or (default if default is not None else GENERIC_TYPE)


def is_known(source_id) -> bool:
    """Whether this request is allowed to name this source at all."""
    return bool(profile_for(source_id))


def describe(source_id) -> Dict[str, Any]:
    """One source as a safe, user-facing dict.

    `id` is retained because the client needs a stable key to correlate an entry
    with a per-source execution record; it is an opaque handle, not a secret (the
    caller already sent it as part of its own request scope). Nothing else about
    an unknown source is emitted.
    """
    return {
        "id": str(source_id) if source_id is not None else "",
        "name": display_name(source_id),
        "type": display_type(source_id),
        "known": is_known(source_id),
    }


def describe_all(source_ids) -> List[Dict[str, Any]]:
    """Safe projections for a list of ids, order preserved, duplicates dropped.

    Only sources that ACTUALLY participated should be passed in — this function
    does not filter by participation, it only makes the naming safe.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for sid in (source_ids or []):
        key = str(sid)
        if key in seen:
            continue
        seen.add(key)
        out.append(describe(sid))
    return out


def summarize(source_ids) -> str:
    """A one-line "which data was used" sentence, safe for any audience.

        []                      -> "No data source was used."
        [Sales DB]              -> "Used Sales Database."
        [Sales DB, Contracts]   -> "Used 2 data sources: Sales Database and Contracts."
    """
    described = describe_all(source_ids)
    if not described:
        return "No data source was used."
    names = [d["name"] for d in described]
    if len(names) == 1:
        return f"Used {names[0]}."
    if len(names) == 2:
        joined = f"{names[0]} and {names[1]}"
    else:
        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
    return f"Used {len(names)} data sources: {joined}."
