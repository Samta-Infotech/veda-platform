# =============================================================================
# ingestion/column_text.py
# VEDA — Shared Column Embedding Text Builder
#
# Responsibility:
#   Builds enriched natural-language embedding text for a single column,
#   drawing on all available metadata: glossed name, location, semantic type,
#   sample values, PK/FK relationships, DB description, and sibling columns.
#
# Used by all three embedding sites so they stay consistent:
#   - ingestion/relgt_encoder.py  _build_column_sentence  (light_text / TF-IDF)
#   - ingestion/relgt_encoder.py  _build_minilm_sentence  (MiniLM 384-dim)
#   - ingestion/graph_embedder.py embed_graph_nodes        (graph node 384-dim)
#
# Design constraints:
#   - Graceful degradation: missing sampled values or description → skip those
#     parts silently; name + table + type + gloss always produce a valid sentence.
#   - Two styles:
#       "light_text" — space-joined tokens for TF-IDF (sklearn tokenises on whitespace)
#       "minilm"     — comma-joined natural phrases for MiniLM / sentence-transformers
#   - Content is identical between styles; only formatting differs.
#
# Why this fixes retrieval:
#   The original 3-token string ("{col_name} {table_name} {semantic_type}") had
#   near-zero semantic signal — cosine similarity was noise. Adding sample values
#   means "incident_status" now contains "Open, Escalated, Closed" and will rank
#   near the query "incident status" while "sla_scheduled_action.db_name" won't.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import List, Optional

from config import (
    EMBED_SENTENCE_MAX_VALUES,
    EMBED_SENTENCE_MAX_VALUE_LEN,
    ACRONYM_EXPANSION_ENABLED,
    ACRONYM_MAP,
)


_TYPE_PHRASES = {
    "CATEGORY":   "a category field",
    "MONETARY":   "a monetary amount",
    "TEMPORAL":   "a date or time",
    "METRIC":     "a numeric metric",
    "IDENTIFIER": "an identifier",
    "FREE_TEXT":  "a free text field",
}


def _split_identifier(name: str) -> str:
    """Split snake_case / camelCase identifiers into lowercase words."""
    tokens = name.replace("_", " ")
    tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", tokens)
    return tokens.lower().strip()


def _expand_acronyms(gloss: str) -> str:
    """Append spelled-out expansions of any acronym tokens present in gloss.
    e.g. 'rfi objects' → 'rfi objects request for information'
    Only fires when ACRONYM_EXPANSION_ENABLED=True.
    """
    if not ACRONYM_EXPANSION_ENABLED:
        return gloss
    extras = []
    for tok in gloss.split():
        exp = ACRONYM_MAP.get(tok)
        if exp and exp not in gloss:
            extras.append(exp)
    if extras:
        return gloss + " " + " ".join(extras)
    return gloss


def build_enriched_column_text(
    col_name:      str,
    table_name:    str,
    semantic_type: str,
    *,
    is_pk:         bool = False,
    is_fk:         bool = False,
    fk_ref_table:  Optional[str] = None,
    fk_ref_col:    Optional[str] = None,
    sampled=None,
    description:   Optional[str] = None,
    sibling_names: Optional[List[str]] = None,
    style:         str = "minilm",
) -> str:
    """
    Build enriched embedding text for one column from all available metadata.

    Parameters
    ----------
    col_name, table_name, semantic_type : str
        Always-available column identity fields.
    is_pk, is_fk, fk_ref_table, fk_ref_col : structural metadata
    sampled : SampledColumn | None
        From value_sampler.get_sampled_columns()[col_id]. raw_values used.
    description : str | None
        DB column comment (Postgres col_description). Used when EMBED_USE_COLUMN_DESCRIPTIONS.
    sibling_names : List[str] | None
        Other CATEGORY/TEMPORAL/METRIC column names in the same table.
    style : "minilm" | "light_text"
        "minilm"     — comma-joined natural phrases (MiniLM, graph_embedder)
        "light_text" — space-joined tokens         (TF-IDF)
    """
    gloss     = _expand_acronyms(_split_identifier(col_name))
    tbl_gloss = _expand_acronyms(_split_identifier(table_name))
    sem_upper = semantic_type.upper() if semantic_type else ""
    type_phrase = _TYPE_PHRASES.get(sem_upper, sem_upper.lower().replace("_", " "))

    raw_values: List[str] = []
    if sampled is not None:
        rv = getattr(sampled, "raw_values", None) or []
        for v in rv[:EMBED_SENTENCE_MAX_VALUES]:
            vs = str(v)[:EMBED_SENTENCE_MAX_VALUE_LEN].strip()
            if vs:
                raw_values.append(vs)

    if style == "light_text":
        parts: List[str] = []
        parts.append(f"{gloss} {tbl_gloss}")
        parts.append(f"column {gloss} in table {tbl_gloss}")
        parts.append(type_phrase)
        if raw_values:
            parts.append("example values " + " ".join(raw_values))
        if is_pk:
            parts.append("primary key")
        if is_fk and fk_ref_table:
            parts.append(f"references {_split_identifier(fk_ref_table)} {_split_identifier(fk_ref_col or 'id')}")
        if sibling_names:
            parts.append("also " + " ".join(_split_identifier(s) for s in sibling_names))
        if description:
            parts.append(description)
        return " ".join(parts)

    else:
        # MiniLM / graph_embedder: comma-joined natural phrases
        parts = [gloss]
        parts.append(tbl_gloss)   # table name gloss carries acronym expansions
        parts.append(f"column {col_name} in table {table_name}")
        parts.append(type_phrase)
        if raw_values:
            parts.append(f"example values: {', '.join(raw_values)}")
        if is_pk:
            parts.append("primary key")
        if is_fk and fk_ref_table:
            parts.append(f"references {fk_ref_table}.{fk_ref_col or 'id'}")
        if sibling_names:
            parts.append("table also has " + " ".join(sibling_names))
        if description:
            parts.append(description)
        return ", ".join(parts)
