"""query/doc_data_planner.py — bounded DOCUMENT_FACT + DATA_GROUNDING cross-source strategy.

For a MULTI query spanning a DOCUMENT source and a DATA source where the answer is "the entities the
document names that ALSO exist in our data" (e.g. "which amenities mentioned in the maintenance policy
exist in our asset amenities"):

    1. RAG-retrieve the document chunks (existing retrieval).
    2. ONE bounded SLM call: extract the LIST of entities the question is about FROM THE CHUNKS
       (grounded — must appear in the chunk text), and PICK the data-side column to match against from
       a supplied candidate list (an index, never a free column name).
    3. Query DISTINCT that data column through the existing data pipeline.
    4. Deterministic case-insensitive INTERSECTION (code, not SLM).
    5. Grounded answer with document + data provenance.

The SLM never invents a source/table/column (picks a candidate index) and never invents an entity
(must be quoted from the chunks; a post-filter drops any not found in the chunk text). Flag:
DOC_DATA_GROUNDING_ENABLED (default OFF).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from slm._call_slm import call_slm


def doc_data_enabled() -> bool:
    try:
        from config import DOC_DATA_GROUNDING_ENABLED
        return bool(DOC_DATA_GROUNDING_ENABLED)
    except Exception:
        return False


_SYSTEM = (
    "You ground a DOCUMENT question against structured DATA. Given the question, passages from a "
    "document, and a NUMBERED list of candidate data columns, do TWO things and reply with STRICT "
    "JSON only: {\"entities\": [<the CORE entity names the document mentions that the question asks "
    "about>], \"data_column\": <the candidate NUMBER whose values are the same KIND of thing>}. "
    "STRICT RULES (to avoid making things up):\n"
    "1. Every entity you output MUST be a word/phrase that actually appears in the passages — copy it, "
    "never invent, guess, or add one from outside the passages.\n"
    "2. Output the CORE name only, WITHOUT surrounding descriptive words: from 'Football court upkeep' "
    "output 'Football'; from 'the Swimming Pool facility' output 'Swimming Pool'. The core name must "
    "still be a substring of the passage text.\n"
    "3. If the passages name no relevant entities, return \"entities\": [].\n"
    "4. Pick the data_column number whose meaning matches the entities (e.g. amenity names ↔ an "
    "amenity-name column), or -1 if none fits.\n"
    "Do NOT write any prose, explanation, or values from the data — only the JSON."
)


def _build_user(query: str, chunks: List[str], data_cols: List[dict]) -> str:
    lines = [f"QUESTION: {query}", "", "DOCUMENT PASSAGES:"]
    for c in chunks[:6]:
        lines.append(f"- {str(c)[:300]}")
    lines.append("")
    lines.append("CANDIDATE DATA COLUMNS (pick one number, or -1):")
    for i, dc in enumerate(data_cols, 1):
        lines.append(f"{i}: {dc['table']}.{dc['col']} (source {dc['source_id']})")
    lines.append("")
    lines.append("Reply with STRICT JSON only.")
    return "\n".join(lines)


def _parse(raw) -> Optional[dict]:
    if raw is None:
        return None
    s = str(raw).strip()
    if "```" in s and s.count("```") >= 2:
        s = s.split("```")[1].lstrip("json").strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        s = s[a:b + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def classify(query: str, chunks: List[str], data_cols: List[dict], slm_call=None
             ) -> Optional[Tuple[List[str], dict]]:
    """Bounded SLM: return (grounded entity list, chosen data column dict) or None. Every entity is
    re-checked against the chunk text (drop hallucinations); the column choice is validated to be a
    supplied candidate. Empty entities or an invalid column → None (defer)."""
    if not chunks or not data_cols:
        return None
    slm_call = slm_call or (lambda system, user: call_slm(user, system=system,
                                                          purpose="doc_data_ground",
                                                          temperature=0.0, json_format=True))
    parsed = _parse(slm_call(_SYSTEM, _build_user(query, chunks, data_cols)))
    if not isinstance(parsed, dict):
        return None
    ents = parsed.get("entities")
    if not isinstance(ents, list) or not ents:
        return None
    # grounding: keep only entities that actually appear in the chunk text (case-insensitive)
    blob = " \n ".join(str(c) for c in chunks).lower()
    grounded = [str(e).strip() for e in ents if str(e).strip() and str(e).strip().lower() in blob]
    if not grounded:
        return None
    try:
        n = int(parsed.get("data_column"))
    except Exception:
        return None
    if n < 1 or n > len(data_cols):
        return None
    return grounded, data_cols[n - 1]


# Only universal grammatical stopwords — NOT domain words. The token-SUBSET match already handles
# descriptive words ('Football' ⊆ 'Football court') without listing domain terms, so nothing here is
# tuned to any dataset.
_STOP = {"the", "a", "an", "of", "and", "or", "for", "in", "on", "to", "with"}


def _tokens(s: str) -> set:
    """Significant lowercase word tokens (alnum), minus a small generic stop set. General, not
    per-query: it just splits words so 'Football court' ⊇ 'Football' can match by token containment."""
    import re
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t and t not in _STOP}


def intersect(doc_entities: List[str], data_values: List[str]) -> List[str]:
    """Deterministic intersection: which document entities correspond to a real data value. A match is
    TOKEN-SUBSET (case-insensitive) — the significant words of one are all contained in the other's
    (so 'Football court' ↔ 'Football'). No fuzzy scoring, no per-value rules: pure set containment on
    words. Returns the DATA value (its canonical form) for each match, deduped."""
    dv_tok = [(str(v).strip(), _tokens(v)) for v in (data_values or []) if v is not None and str(v).strip()]
    seen, out = set(), []
    for e in doc_entities:
        et = _tokens(e)
        if not et:
            continue
        for canon, vt in dv_tok:
            if not vt:
                continue
            # one side's significant words are a subset of the other's → same entity
            if (et <= vt or vt <= et) and canon.lower() not in seen:
                seen.add(canon.lower())
                out.append(canon)
                break
    return out
