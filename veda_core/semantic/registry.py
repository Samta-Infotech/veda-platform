# =============================================================================
# semantic/registry.py
# VEDA — in-memory loader + matchers for the compiled semantic registries.
#
# Loads concepts.json / dimensions.json / metrics.json once per process and
# exposes pure-lookup matchers (no embeddings, no LLM). Everything here is
# deterministic and microsecond-cheap; the registries are the source of truth.
# =============================================================================

import os
import json
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

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


def _load_file(name):
    path = os.path.join(_HERE, name)
    if not os.path.exists(path):
        return {}, None
    blob = json.load(open(path))
    return blob.get("items", {}), blob.get("source_hash")


def load(force: bool = False) -> bool:
    """Load registries into memory once. Returns True if a non-empty layer loaded."""
    if _STATE["loaded"] and not force:
        return bool(_STATE["concepts"] or _STATE["metrics"])
    c, h1 = _load_file("concepts.json")
    d, _  = _load_file("dimensions.json")
    m, _  = _load_file("metrics.json")
    _STATE.update({"loaded": True, "concepts": c, "dimensions": d,
                   "metrics": m, "source_hash": h1})
    return bool(c or m)


def is_ready() -> bool:
    load()
    return bool(_STATE["concepts"] and _STATE["metrics"])


# ---------------------------------------------------------------------------
# Matchers — all return explainable, scored results; ambiguity is surfaced, not hidden
# ---------------------------------------------------------------------------

def match_concepts(qtoks: set) -> list:
    """Return [(concept, score)] for entity concepts whose tokens appear in the query,
    best first. score = (#matched tokens, coverage)."""
    load()
    hits = []
    for c in _STATE["concepts"].values():
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
    load()
    return _STATE["metrics"].get(metric_id)


def match_metric_labels(query_l: str) -> list:
    """Direct label match for non-count metrics (SUM/AVG) that have no entity noun.
    Returns [(metric, n_label_tokens_matched)]."""
    load()
    out = []
    for m in _STATE["metrics"].values():
        if m.get("kind") == "COUNT":
            continue                      # COUNT resolved via concept, not labels
        for lab in m.get("labels", []):
            if lab and lab in query_l:
                out.append((m, len(lab.split())))
                break
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def dimensions_for_table(table: str) -> list:
    load()
    return [d for d in _STATE["dimensions"].values() if d.get("owner_table") == table]


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
