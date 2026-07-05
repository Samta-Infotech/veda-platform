# query/schema_linker.py
# VEDA — Step 0: spaCy query understanding + deterministic schema linking
# Gate: RETRIEVAL_V2_ENABLED and SCHEMA_LINK_ENABLED
# When the query clearly names its schema objects, link deterministically
# and set short_circuit=True so retrieval/injection is skipped entirely.

import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from config import (
    SPACY_MODEL,
    SCHEMA_LINK_MIN_TOKEN_LEN,
    SCHEMA_LINK_SHORTCIRCUIT_MIN_COLS,
    SCHEMA_LINK_SHORTCIRCUIT_MAX_TABLES,
    SCHEMA_LINK_SYNONYMS,
    ACRONYM_EXPANSION_ENABLED,
    ACRONYM_MAP,
)
from ingestion.vector_store import RetrievalResult, retrieve_cols_by_name_keywords

# Inverse acronym map: spelled-out phrase → short token (e.g. "request for information" → "rfi")
# Built once at module load; used to expand query phrases to schema tokens.
_ACRONYM_INVERSE: Dict[str, str] = {v: k for k, v in ACRONYM_MAP.items()} if ACRONYM_MAP else {}

# ---------------------------------------------------------------------------
# Availability guard — spaCy is optional
# ---------------------------------------------------------------------------
SPACY_AVAILABLE = False
_NLP = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass

# Module-level singleton NLP model (loaded once)
_NLP_CACHE: Dict[str, object] = {}


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    if not SPACY_AVAILABLE:
        return None
    try:
        import spacy
        _NLP = spacy.load(SPACY_MODEL, disable=["ner", "textcat"])
        return _NLP
    except Exception as e:
        import warnings
        warnings.warn(f"[SchemaLinker] Could not load spaCy model '{SPACY_MODEL}': {e}")
        return None


def _split_identifier(name: str) -> List[str]:
    tokens = name.replace("_", " ")
    tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", tokens)
    return [t.lower().strip() for t in tokens.split() if len(t.strip()) >= SCHEMA_LINK_MIN_TOKEN_LEN]


def _expand_with_synonyms(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for tok in tokens:
        # SCHEMA_LINK_SYNONYMS: abbreviation expansions (e.g. number → no, num)
        for e in SCHEMA_LINK_SYNONYMS.get(tok, []):
            if e not in expanded:
                expanded.append(e)
        if ACRONYM_EXPANSION_ENABLED:
            # Forward: schema token → spelled-out phrase tokens
            # (e.g. 'rfi' in schema name → add 'request', 'for', 'information')
            fwd = ACRONYM_MAP.get(tok)
            if fwd:
                for phrase_tok in fwd.split():
                    if phrase_tok not in expanded and len(phrase_tok) >= SCHEMA_LINK_MIN_TOKEN_LEN:
                        expanded.append(phrase_tok)
            # Inverse: query phrase tokens → short schema token
            # (e.g. query contains 'information' → add 'rfi')
            for phrase, short in _ACRONYM_INVERSE.items():
                phrase_tokens = phrase.split()
                if all(pt in tokens for pt in phrase_tokens):
                    if short not in expanded:
                        expanded.append(short)
    return expanded


@dataclass
class SchemaLinkResult:
    matched_columns: List[RetrievalResult] = field(default_factory=list)
    matched_tables:  List[str]             = field(default_factory=list)
    short_circuit:   bool                  = False
    primary_table_id: Optional[str]        = None
    chunks_used:     List[str]             = field(default_factory=list)
    stats:           dict                  = field(default_factory=dict)


def run_schema_linker(
    query:      str,
    source_ids: Optional[List[str]] = None,
    verbose:    bool = False,
) -> SchemaLinkResult:
    """
    Deterministically links query noun-chunks to schema columns using spaCy lemmatisation
    and PhraseMatcher.

    Returns SchemaLinkResult. short_circuit=True when the query clearly names
    <= SCHEMA_LINK_SHORTCIRCUIT_MAX_TABLES tables with >= SCHEMA_LINK_SHORTCIRCUIT_MIN_COLS
    matched columns — the caller can then skip retrieval entirely.

    Falls back gracefully (short_circuit=False, matched_*=[]) if spaCy is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        if verbose:
            print("  [SchemaLinker] spaCy unavailable — falling through to retrieval")
        return SchemaLinkResult(stats={"spacy_available": False})

    # ------------------------------------------------------------------
    # 1. Process query: lemmatise, extract noun chunks
    # ------------------------------------------------------------------
    doc = nlp(query)

    _noun_pos = {"NOUN", "PROPN", "ADJ"}
    _skip_lemmas = {
        "show", "list", "get", "find", "give", "tell", "display", "return",
        "fetch", "retrieve", "pull", "all", "me", "us",
    }

    content_lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in _noun_pos
        and token.lemma_.lower() not in _skip_lemmas
        and len(token.lemma_) >= SCHEMA_LINK_MIN_TOKEN_LEN
        and not token.is_stop
        and not token.is_punct
    ]

    noun_chunks = [
        chunk.text.lower().strip()
        for chunk in doc.noun_chunks
        if len(chunk.text.strip()) >= SCHEMA_LINK_MIN_TOKEN_LEN
    ]

    if verbose:
        print(f"  [SchemaLinker] Query lemmas     : {content_lemmas}")
        print(f"  [SchemaLinker] Noun chunks      : {noun_chunks}")

    # ------------------------------------------------------------------
    # 2. Keyword-based name matching via existing infrastructure
    # ------------------------------------------------------------------
    search_tokens = _expand_with_synonyms(content_lemmas)
    if not search_tokens:
        return SchemaLinkResult(
            chunks_used=noun_chunks,
            stats={"spacy_available": True, "content_lemmas": content_lemmas},
        )

    candidates = retrieve_cols_by_name_keywords(search_tokens)

    # Filter by source_ids if provided
    if source_ids:
        candidates = [
            c for c in candidates
            if not c.source_id or c.source_id in source_ids
        ]

    if not candidates:
        return SchemaLinkResult(
            chunks_used=noun_chunks,
            stats={
                "spacy_available":  True,
                "content_lemmas":   content_lemmas,
                "search_tokens":    search_tokens,
                "candidates_found": 0,
            },
        )

    # ------------------------------------------------------------------
    # 3. Short-circuit decision
    # ------------------------------------------------------------------
    matched_table_ids = list(dict.fromkeys(c.table_id for c in candidates))
    matched_tables    = list(dict.fromkeys(c.table_name for c in candidates))

    short_circuit = (
        len(candidates) >= SCHEMA_LINK_SHORTCIRCUIT_MIN_COLS
        and len(matched_table_ids) <= SCHEMA_LINK_SHORTCIRCUIT_MAX_TABLES
    )

    primary_table_id = matched_table_ids[0] if matched_table_ids else None

    # Set similarity=1.0 for exact name-match results
    for c in candidates:
        c.similarity = 1.0

    if verbose:
        print(f"  [SchemaLinker] Candidates       : {len(candidates)}")
        print(f"  [SchemaLinker] Matched tables   : {matched_tables}")
        print(f"  [SchemaLinker] Short-circuit    : {short_circuit}")

    return SchemaLinkResult(
        matched_columns  = candidates,
        matched_tables   = matched_tables,
        short_circuit    = short_circuit,
        primary_table_id = primary_table_id,
        chunks_used      = noun_chunks,
        stats            = {
            "spacy_available":  True,
            "content_lemmas":   content_lemmas,
            "search_tokens":    search_tokens,
            "candidates_found": len(candidates),
            "tables_spanned":   len(matched_table_ids),
        },
    )
