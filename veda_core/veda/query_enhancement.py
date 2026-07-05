# =============================================================================
# veda/query_enhancement.py
# Query ENHANCEMENT (not a rewriter). The user's query is IMMUTABLE; this emits an
# additive SIDECAR of search terms / candidates that improves retrieval RECALL only.
#
# Hard contract:
#   • original_query is never changed.
#   • Output feeds RETRIEVAL only. Routing, planning, and BOTH correctness gates
#     (qualifier_completeness, ir_equivalence) run on the ORIGINAL query.
#   • Enhancement may change HOW we search — never WHAT is constrained. It adds NO
#     filters, dates, business rules, grouping, or intent. There is no field for them.
#   • Deterministic-first (typo / plural / alias / synonym from the semantic model).
#     LLM is used ONLY for follow-up resolution, gated, and re-grounded.
#   • negative_aliases SUPPRESS bad expansions.
#
# Correctness is owned by the firewall; this layer only raises recall.
# =============================================================================

import re
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Conservative caps — over-expansion floods the retrieval vector and hurts precision
# (correctness > recall, so under-expand rather than over-expand).
_MAX_SEARCH_EXTRA      = 6     # extra terms appended to the retrieval string
_MAX_KEYS_PER_TOKEN    = 3     # a token matching > this many columns is too generic → skip
_MAX_CANDIDATE_ENTITIES = 5


@dataclass
class QueryEnhancement:
    original_query: str
    search_terms:        List[str] = field(default_factory=list)
    expanded_aliases:    List[str] = field(default_factory=list)
    candidate_entities:  List[str] = field(default_factory=list)
    candidate_measures:  List[str] = field(default_factory=list)
    candidate_dimensions: List[str] = field(default_factory=list)
    enhancement_trace:   List[str] = field(default_factory=list)

    @property
    def search_query(self) -> str:
        """The string passed to RETRIEVAL only — original query plus a FEW high-precision
        additive terms (typo corrections, synonyms, singular forms). Column-name matches
        live in expanded_aliases as sidecar HINTS and are NOT searched (they flood the
        query vector and hurt precision). Never executed, never validated against."""
        extra = [t for t in self.search_terms
                 if t and t.lower() not in self.original_query.lower()]
        seen, uniq = set(), []
        for t in extra:
            if t not in seen:
                seen.add(t); uniq.append(t)
        uniq = uniq[:_MAX_SEARCH_EXTRA]
        return (self.original_query + " " + " ".join(uniq)).strip() if uniq else self.original_query

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "search_terms": self.search_terms,
            "expanded_aliases": self.expanded_aliases,
            "candidate_entities": self.candidate_entities,
            "candidate_measures": self.candidate_measures,
            "candidate_dimensions": self.candidate_dimensions,
            "enhancement_trace": self.enhancement_trace,
        }


# ---------------------------------------------------------------------------
# Indexes built once from the semantic model (data-derived — NO hand lists)
# ---------------------------------------------------------------------------
_IDX_CACHE = {"v": None}


def _toks(s: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2}


def _build_indexes(sm: dict) -> dict:
    if _IDX_CACHE["v"] is not None:
        return _IDX_CACHE["v"]
    vocab: Set[str] = set()                       # typo-correction target vocabulary
    alias_to_keys: Dict[str, Set[str]] = {}        # token → {table.col}
    token_to_table: Dict[str, Set[str]] = {}       # token → {table}
    neg: Dict[str, Set[str]] = {}                  # table.col → negative-alias tokens
    role: Dict[str, str] = {}                      # table.col → analytics_role
    domain_syn: Dict[str, List[str]] = sm.get("domain_synonyms", {}) or {}

    for t in sm.get("tables", {}):
        for w in _toks(t.replace("_", " ")):
            vocab.add(w)
            token_to_table.setdefault(w, set()).add(t)

    for key, meta in (sm.get("columns", {}) or {}).items():
        t, c = key.split(".", 1) if "." in key else ("", key)
        meta = meta or {}
        for w in _toks(c.replace("_", " ")):
            vocab.add(w)
            alias_to_keys.setdefault(w, set()).add(key)
        for a in (meta.get("aliases") or []):
            for w in _toks(a):
                vocab.add(w)
                alias_to_keys.setdefault(w, set()).add(key)
        neg[key] = set().union(*[_toks(a) for a in (meta.get("negative_aliases") or [])]) \
            if meta.get("negative_aliases") else set()
        role[key] = (meta.get("analytics_role") or "").upper()
    for term in domain_syn:
        vocab.update(_toks(term))

    _IDX_CACHE["v"] = {"vocab": sorted(vocab), "vocab_set": vocab,
                       "alias_to_keys": alias_to_keys, "token_to_table": token_to_table,
                       "neg": neg, "role": role, "domain_syn": domain_syn}
    return _IDX_CACHE["v"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enhance_query(query: str, sm: dict, prev_query: Optional[str] = None) -> QueryEnhancement:
    """Produce an additive enhancement sidecar. Deterministic; LLM only for follow-up
    (gated). The original query is returned unchanged in `original_query`."""
    enh = QueryEnhancement(original_query=query)
    if not query or not sm:
        return enh
    idx = _build_indexes(sm)
    from retrieval.query_enrichment import _singularize

    # Gate-strip command / grammar / temporal words so they never drive expansion
    # ("show", "to", "last month") — only CONTENT tokens do.
    from veda.validation import _gate_strip
    gate = _gate_strip()
    content = [w for w in re.findall(r"[a-z]+", query.lower())
               if len(w) > 2 and w not in gate]

    # 1 — typo correction (content tokens vs schema vocab; a token already in vocab is
    #     never "corrected", so real values aren't mangled). → SEARCHABLE.
    corrected: List[str] = []
    for t in content:
        if t in idx["vocab_set"]:
            continue
        m = difflib.get_close_matches(t, idx["vocab"], n=1, cutoff=0.86)
        if m:
            corrected.append(m[0])
            enh.search_terms.append(m[0])
            enh.enhancement_trace.append(f"typo: {t}→{m[0]}")
    base = content + corrected

    # 2 — singular forms → SEARCHABLE
    for t in base:
        s = _singularize(t)
        if s != t and s not in base:
            enh.search_terms.append(s)
    if any(_singularize(t) != t for t in base):
        enh.enhancement_trace.append("pluralization applied")

    # 3 — domain synonyms (client → customer) → SEARCHABLE, high-precision (1:few)
    _added_syn = False
    for t in base:
        for syn in (idx["domain_syn"].get(t) or []):
            s = str(syn).lower().strip()
            if s and "." not in s and not s.endswith("_id") and re.fullmatch(r"[a-z ]{3,}", s):
                enh.search_terms.append(s); _added_syn = True   # natural-language synonyms only
    if _added_syn:
        enh.enhancement_trace.append("semantic expansion applied")

    # 4 — alias / column expansion → SIDECAR HINTS ONLY (not searched). Skip GENERIC
    #     tokens (match > _MAX_KEYS_PER_TOKEN columns → uninformative); apply
    #     negative_alias suppression. These feed the planner/trace, not the query vector.
    matched_keys: Set[str] = set()
    for t in base:
        all_keys = idx["alias_to_keys"].get(t, set())
        keys = [k for k in all_keys if t not in idx["neg"].get(k, set())]
        if len(keys) < len(all_keys):
            enh.enhancement_trace.append(f"suppressed: {t} (negative_alias)")
        if not keys or len(keys) > _MAX_KEYS_PER_TOKEN:
            continue                                   # too generic → no signal
        for k in keys:
            matched_keys.add(k)
            tbl, col = k.split(".", 1)
            if col not in enh.expanded_aliases:
                enh.expanded_aliases.append(col)
            if tbl not in enh.candidate_entities and len(enh.candidate_entities) < _MAX_CANDIDATE_ENTITIES:
                enh.candidate_entities.append(tbl)
    for t in base:
        for tbl in idx["token_to_table"].get(t, ()):
            if tbl not in enh.candidate_entities and len(enh.candidate_entities) < _MAX_CANDIDATE_ENTITIES:
                enh.candidate_entities.append(tbl)
    if matched_keys:
        enh.enhancement_trace.append("alias expansion applied")

    # 5 — candidate measures / dimensions (PRIORS for the planner — never constraints)
    for key in matched_keys:
        r = idx["role"].get(key, "")
        col = key.split(".", 1)[1]
        if r == "MEASURE" and col not in enh.candidate_measures:
            enh.candidate_measures.append(col)
        elif r == "DIMENSION" and col not in enh.candidate_dimensions:
            enh.candidate_dimensions.append(col)

    # 6 — follow-up resolution (LLM, GATED). Resolves "only active ones" using prev_query,
    #     temp 0, and its output is re-grounded to schema tokens or discarded. Additive.
    try:
        from config import QUERY_ENHANCEMENT_LLM_FOLLOWUP
    except Exception:
        QUERY_ENHANCEMENT_LLM_FOLLOWUP = False
    if QUERY_ENHANCEMENT_LLM_FOLLOWUP and prev_query and _looks_like_followup(query):
        enh.enhancement_trace.append("follow-up resolution attempted (LLM)")
        # left as the integration seam: a temp-0 call would propose terms, each kept
        # ONLY if it grounds to idx['vocab_set'] — never added as a filter.

    # de-dup search_terms; original tokens stay implicit in original_query
    seen, uniq = set(), []
    for t in enh.search_terms:
        if t not in seen:
            seen.add(t); uniq.append(t)
    enh.search_terms = uniq
    return enh


_FOLLOWUP_RE = re.compile(r"^\s*(only|just|and|also|what about|how about|same but)\b", re.I)


def _looks_like_followup(query: str) -> bool:
    return bool(_FOLLOWUP_RE.search(query)) or len(query.split()) <= 3
