"""L4 INDEX · merged enrichment index (Q-3).

Builds ONE pre-tokenized, pre-inverted index (domain synonyms + concept graph +
glossary) at ingestion so the query-time ``QueryEnricher`` loads a single artifact
instead of parsing four JSON files and re-inverting them on every warm. Also a
correctness win: the glossary is now guaranteed fresh (L3 rebuilds it — I-5).

Pure transform of the on-disk enrichment files → no source-DB touch, non-fatal.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _index_path(source_id=None, tenant: str = "default") -> str:
    """Per-(tenant, source) when source_id is given (P0-5, 2026-09-10) — unconditionally,
    via config.source_artifact_path(), not gated behind VEDA_ARTIFACT_SCOPE (P0-7). The
    INPUTS this index is built from (DOMAIN_SYNONYMS_FILE/CONCEPT_GRAPH_FILE/GLOSSARY_FILE)
    are still process-global/flat — this scopes the OUTPUT, not (yet) those three."""
    if source_id is not None:
        from config import source_artifact_path
        return source_artifact_path("veda_enrichment_index.json", source_id, tenant)
    from config import artifact_path
    return artifact_path("veda_enrichment_index.json")


def _load_json(path: str) -> dict:
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def build_enrichment_index(source_id: str = "", tenant: str = "default",
                          verbose: bool = False) -> Dict:
    """Merge synonyms + concepts + glossary into one lowercased token→expansions map.

    Inputs resolved via config.resolve_source_artifact() (P0-5, 2026-09-11) — THIS
    source's own domain synonyms/concept graph/glossary when they exist, else the
    legacy flat files (unblocks this artifact from needing all three inputs
    migrated at once)."""
    from config import (DOMAIN_SYNONYMS_FILE, CONCEPT_GRAPH_FILE, GLOSSARY_FILE,
                       resolve_source_artifact)

    _sid = source_id or None
    synonyms = _load_json(resolve_source_artifact("veda_domain_synonyms.json", _sid, tenant,
                                                  flat_default=DOMAIN_SYNONYMS_FILE))
    concepts = _load_json(resolve_source_artifact("veda_concept_graph.json", _sid, tenant,
                                                  flat_default=CONCEPT_GRAPH_FILE))
    glossary = _load_json(resolve_source_artifact("veda_glossary.json", _sid, tenant,
                                                  flat_default=GLOSSARY_FILE))

    # token -> sorted list of expansion terms (dedup, lowercased, order-stable)
    inverted: Dict[str, set] = {}

    def _add(term, expansions):
        key = str(term).strip().lower()
        if not key:
            return
        bucket = inverted.setdefault(key, set())
        for e in expansions:
            e = str(e).strip().lower()
            if e and e != key:
                bucket.add(e)

    if isinstance(synonyms, dict):
        for term, syns in synonyms.items():
            _add(term, syns if isinstance(syns, (list, tuple)) else [syns])
    if isinstance(concepts, dict):
        for concept, members in concepts.items():
            members = members if isinstance(members, (list, tuple)) else [members]
            _add(concept, members)
            for m in members:
                _add(m, [concept])
    if isinstance(glossary, dict):
        for term, meaning in glossary.items():
            # regex, not .split()+isalnum() — punctuation-adjacent tokens ("incident,",
            # "state-changes") fail isalnum() and were silently dropped from the index.
            toks = [t for t in _TOKEN_RE.findall(str(meaning).lower()) if len(t) > 2]
            _add(term, toks)

    index = {"inverted": {k: sorted(v) for k, v in inverted.items()},
             "counts": {"synonyms": len(synonyms), "concepts": len(concepts),
                        "glossary": len(glossary)}}

    path = _index_path(source_id or None, tenant)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    if verbose:
        print(f"  [enrichment_index] {len(index['inverted'])} terms → {path}")

    # P0-6: drop this source's cached index so the next read sees the fresh one.
    try:
        invalidate_enrichment_index_cache(source_id or None)
    except Exception:
        pass

    return {"terms": len(index["inverted"]), "path": path}


# (tenant, str(source_id) | "") -> {token: [expansions]} | None (checked, absent).
_ENRICHMENT_INDEX_CACHE: dict = {}


def load_enrichment_index(source_id=None, tenant=None):
    """Query-tier warm loader: return the merged {token: [expansions]} map, or None
    if it was never built (caller falls back to parsing the individual files).
    Resolves source_id/tenant from the ambient request context when not given, and
    memoizes per (tenant, source) (P0-5, 2026-09-10) — a single global cache here
    meant source B's enricher could silently warm from source A's merged index."""
    if source_id is None:
        from veda_core import context
        ctx = context.try_current()
        if ctx is not None:
            source_id = ctx.source_id
            tenant = ctx.tenant or tenant
    key = (tenant or "default", str(source_id) if source_id is not None else "")
    if key in _ENRICHMENT_INDEX_CACHE:
        return _ENRICHMENT_INDEX_CACHE[key]
    path = _index_path(source_id, tenant or "default")
    if not os.path.exists(path):
        _ENRICHMENT_INDEX_CACHE[key] = None
        return None
    try:
        with open(path) as f:
            data = json.load(f).get("inverted", {})
        _ENRICHMENT_INDEX_CACHE[key] = data
        return data
    except Exception:
        return None


def invalidate_enrichment_index_cache(source_id=None):
    """Drop the cached merged index for one source, or every source when source_id
    is None. Call after build_enrichment_index() rewrites the artifact and from both
    rehydrate paths (P0-6)."""
    if source_id is None:
        _ENRICHMENT_INDEX_CACHE.clear()
        return
    sid = str(source_id)
    for key in [k for k in _ENRICHMENT_INDEX_CACHE if k[1] == sid]:
        del _ENRICHMENT_INDEX_CACHE[key]
