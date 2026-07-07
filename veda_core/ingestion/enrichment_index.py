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
from typing import Dict


def _index_path() -> str:
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


def build_enrichment_index(source_id: str = "", verbose: bool = False) -> Dict:
    """Merge synonyms + concepts + glossary into one lowercased token→expansions map."""
    from config import DOMAIN_SYNONYMS_FILE, CONCEPT_GRAPH_FILE, GLOSSARY_FILE

    synonyms = _load_json(DOMAIN_SYNONYMS_FILE)
    concepts = _load_json(CONCEPT_GRAPH_FILE)
    glossary = _load_json(GLOSSARY_FILE)

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
            toks = [t for t in str(meaning).lower().split() if len(t) > 2 and t.isalnum()]
            _add(term, toks)

    index = {"inverted": {k: sorted(v) for k, v in inverted.items()},
             "counts": {"synonyms": len(synonyms), "concepts": len(concepts),
                        "glossary": len(glossary)}}

    path = _index_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    if verbose:
        print(f"  [enrichment_index] {len(index['inverted'])} terms → {path}")
    return {"terms": len(index["inverted"]), "path": path}


def load_enrichment_index():
    """Query-tier warm loader: return the merged {token: [expansions]} map, or None
    if it was never built (caller falls back to parsing the individual files)."""
    path = _index_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("inverted", {})
    except Exception:
        return None
