"""veda/intent_sql_alignment.py — shared Intent↔SQL referent-alignment guard (Option B, increment 1).

The remaining silent-wrong failures share ONE invariant: the schema element the SQL actually uses must
correspond to what the query REFERS to. This module is the first increment of a GENERALIZED comparator
(not one-off guards): it derives the query's referents deterministically and compares them against the
SQL's ACTUAL referents (``business_explain.extract_sql_facts``) using schema metadata
(``semantic_type`` / ``analytics_role`` / table-ownership). NO LLM, no hardcoded table/column/business
vocabulary — grammar-signal + AST + metadata only.

Two referent classes in this increment (each fully/strongly deterministic per the feasibility audit):
  A. TEMPORAL      — a per-time-period BREAKDOWN intent ("leads per month", "monthly …", "over time")
                     whose SQL groups by a NON-temporal column / has no date bucketing.
  B. ENTITY-ANCHOR — the query NAMES a measure column that lives on table T, but the SQL measures /
                     orders by a different column not on T (anchored on the wrong entity/reference table).

Each check returns (ok, reason). ``ok=False`` → the caller refuses (an aligned answer it cannot build is
safer refused than answered wrong). Flag-gated (INTENT_SQL_ALIGNMENT_ENABLED, default OFF → always ok,
byte-identical). Fires ONLY on a clear mismatch; anything it cannot decide → ok (no over-refusal).
"""
from __future__ import annotations

import re

# ── language-level temporal-BREAKDOWN signals (not schema vocabulary) ──────────────────────────────
_TIME_UNITS = ("hour", "day", "week", "month", "quarter", "year")
_TIME_ADVERBS = ("hourly", "daily", "weekly", "monthly", "quarterly", "yearly")
_TIME_GROUP_PREPS = ("per ", "each ", "every ", "by ")   # "per month", "by year" (a breakdown, not a filter)


def _enabled() -> bool:
    try:
        from config import INTENT_SQL_ALIGNMENT_ENABLED
        return bool(INTENT_SQL_ALIGNMENT_ENABLED)
    except Exception:
        return False


def _facts(sql):
    try:
        from veda.business_explain import extract_sql_facts
        return extract_sql_facts(sql) or {}
    except Exception:
        return {}


# ── A. TEMPORAL ────────────────────────────────────────────────────────────────────────────────────
def _adverb_modifies_measure(query, adverb, sm):
    """True when a time-adverb ("monthly") is part of a MEASURE column's NAME rather than a breakdown
    intent — e.g. "monthly rent" / "highest monthly rent" references `expected_monthly_rent`, not a
    per-month grouping. Signal: some MEASURE column's name contains the adverb AND ≥1 of that column's
    OTHER name-words also appears in the query (so the query is naming that measure). Data-driven (the
    schema's own MEASURE column names), no hardcoded vocabulary."""
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", (query or "").lower()) + " "
    for k, c in (sm or {}).get("columns", {}).items():
        if (c.get("analytics_role") or "").upper() != "MEASURE":
            continue
        words = set(k.split(".", 1)[1].lower().split("_"))
        if adverb in words and any((" " + w + " ") in ql for w in words - {adverb} if len(w) > 2):
            return True
    return False


def _wants_time_bucket(query, sm=None):
    """A per-time-period BREAKDOWN intent (grouped OVER time), NOT a time FILTER ("this month") and NOT
    a measure NAME that happens to contain a time-adverb ("monthly rent"). The `per/each/by <unit>`
    preposition signal and explicit trend phrases are unambiguous breakdowns; a bare `<unit>ly` adverb
    is a breakdown ONLY when it is not modifying a MEASURE noun (else "highest monthly rent" over-refuses)."""
    ql = " " + (query or "").lower().strip() + " "
    if " over time " in ql or " trend " in ql or " time series " in ql:
        return True
    if any((p + u) in ql for p in _TIME_GROUP_PREPS for u in _TIME_UNITS):
        return True                                          # "per month", "by year" — unambiguous
    for a in _TIME_ADVERBS:
        if (" " + a + " ") in ql and not _adverb_modifies_measure(query, a, sm):
            return True
    return False


def _sql_has_temporal_bucket(sql: str, sm, facts) -> bool:
    su = (sql or "").upper()
    if "DATE_TRUNC" in su or "DATE_PART" in su or re.search(r"\bEXTRACT\s*\(", su):
        return True                                      # explicit date bucketing
    cols = (sm or {}).get("columns", {})
    tcols = {k.split(".", 1)[1] for k, c in cols.items()
             if (c.get("semantic_type") or "").upper() == "TEMPORAL"}
    return any(g in tcols for g in facts.get("groupings", []))


def temporal_alignment_ok(query, sql, sm):
    """(ok, reason). Refuse when a per-time BREAKDOWN intent is answered by SQL with no temporal
    grouping/bucketing (the "leads per month" grouped-by-lead_stage silent-wrong)."""
    if not _enabled() or not sql:
        return True, ""
    if not _wants_time_bucket(query, sm):
        return True, ""
    if _sql_has_temporal_bucket(sql, sm, _facts(sql)):
        return True, ""
    return False, ("this asks for a per-time-period breakdown, but the SQL groups by a non-temporal "
                   "column — I can't produce a time-bucketed result for it")


# ── B. ENTITY-ANCHOR ────────────────────────────────────────────────────────────────────────────────
def _col_table(col, sm):
    for k in (sm or {}).get("columns", {}):
        if k.split(".", 1)[1] == col:
            return k.split(".", 1)[0]
    return None


def _named_measure_columns(query, sm):
    """{(table, col)} for MEASURE columns whose full word-phrase the query names verbatim — data-driven
    (the schema's own column names vs the user's own words), no alias list. A column 'carpet_area' is
    matched only when the phrase 'carpet area' appears in the query, so it is precise (low false-fire)."""
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", (query or "").lower()) + " "
    ql = re.sub(r"\s+", " ", ql)
    out = set()
    for k, c in (sm or {}).get("columns", {}).items():
        if (c.get("analytics_role") or "").upper() != "MEASURE":
            continue
        tbl, _, col = k.partition(".")
        words = [w for w in col.lower().split("_") if len(w) > 2]
        if words and (" " + " ".join(words) + " ") in ql:
            out.add((tbl, col))
    return out


def entity_anchor_ok(query, sql, sm):
    """(ok, reason). Refuse when the query NAMES a measure column on table T but the SQL measures/orders
    by a different column not on T (the "highest carpet area" → assets_carpetareaunit silent-wrong).
    Fires only when the query names a specific measure column AND the SQL's measure/order columns are
    entirely elsewhere — otherwise silent (no over-refusal)."""
    if not _enabled() or not sql:
        return True, ""
    named = _named_measure_columns(query, sm)
    if not named:
        return True, ""                                  # query names no specific measure → cannot misalign
    facts = _facts(sql)
    sql_cols = {c for (_f, c) in facts.get("aggregations", []) if c} \
        | {o[0] for o in facts.get("orderings", [])}
    if not sql_cols:
        return True, ""
    named_cols = {c for (_t, c) in named}
    named_tables = {t for (t, _c) in named}
    aligned = any(c in named_cols for c in sql_cols) \
        or any(_col_table(c, sm) in named_tables for c in sql_cols)
    if aligned:
        return True, ""
    return False, ("this measures a different column than the one the question names — the query's "
                   "measure lives on another table than the one the SQL ranks/aggregates")


def alignment_ok(query, sql, sm):
    """Shared Option-B comparator entry point: run both referent-alignment checks. Returns (ok, reason);
    the first violation wins. ok=True when the flag is off or no mismatch is found."""
    ok, why = temporal_alignment_ok(query, sql, sm)
    if not ok:
        return False, why
    return entity_anchor_ok(query, sql, sm)


# ── Aggregate-OMISSION guard (Increment 3A) ─────────────────────────────────────────────────────────
# Language-level scalar-aggregate intent signals — a figure (count/total/average) is expected, NOT a row
# list. Superlatives ("highest"/"lowest"/"maximum"/"minimum") are deliberately EXCLUDED: they have a valid
# ORDER BY … LIMIT form, so a missing aggregate there is not an omission.
_AGG_INTENT = (" how many ", " number of ", " count of ", " count the ", " total number ",
               " total ", " sum of ", " average ", " avg ", " mean ")


def _agg_presence_enabled():
    try:
        from config import INTENT_SQL_AGG_PRESENCE_ENABLED
        return bool(INTENT_SQL_AGG_PRESENCE_ENABLED)
    except Exception:
        return False


def aggregate_presence_ok(query, sql, sm=None):
    """(ok, reason). Refuse when the query has a scalar-aggregate INTENT ("how many"/"total"/"average") but
    the SQL has ZERO aggregate functions — it returns a row list (often projection + LIMIT 100) that the
    summariser reports as the requested count/total (the "how many projects → 100" omission silent-wrong).
    Fires only on intent-present + no-aggregate; a grouped or scalar aggregate passes. Covers OMISSION only,
    not wrong-value aggregates. No-op when the flag is off."""
    if not _agg_presence_enabled() or not sql:
        return True, ""
    ql = " " + (query or "").lower().strip() + " "
    if not any(s in ql for s in _AGG_INTENT):
        return True, ""                                  # no scalar-aggregate intent
    # Per-entity attribute listing ("number of bedrooms FOR EACH property", "total area OF EACH property"):
    # the aggregate word modifies a stored attribute the query wants listed per entity, not a scalar
    # aggregate — a row projection is legitimate here, so do not treat a missing aggregate as an omission.
    if " for each " in ql or " of each " in ql:
        return True, ""
    if _facts(sql).get("aggregations"):
        return True, ""                                  # SQL computes an aggregate → not omitted
    return False, ("this asks for a count/total/average, but the SQL returns rows without an aggregate — "
                   "the result would be a row list, not the requested figure")


# ── C. DIMENSION referent alignment (increment 2) ──────────────────────────────────────────────────
# Four outcomes — a boolean can't express the CLARIFY (genuine ambiguity) case distinctly from REFUSE.
DIM_ALIGNED = "ALIGNED"
DIM_REFUSE = "REFUSE"
DIM_CLARIFY = "CLARIFY"
DIM_NOT_APPLICABLE = "NOT_APPLICABLE"

_DIM_GROUP_PREPS = ("by ", "per ", "each ")
_DIM_STOP = {"the", "a", "an", "and", "or", "of", "for", "in", "on", "their", "its", "our",
             "all", "with", "each", "every", "any", "some"}


def _dim_enabled():
    try:
        from config import INTENT_SQL_DIMENSION_ALIGNMENT_ENABLED
        return bool(INTENT_SQL_DIMENSION_ALIGNMENT_ENABLED)
    except Exception:
        return False


def _dimension_phrase_words(query):
    """Significant words of the requested grouping dimension — the phrase after the LAST by/per/each,
    up to a clause boundary. Grammar-level (the user's own words), no schema vocabulary.
    'leads by status' → {status}; 'leads by furnishing status' → {furnishing, status}."""
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", (query or "").lower()) + " "
    ql = re.sub(r"\s+", " ", ql)
    pos = -1
    for p in _DIM_GROUP_PREPS:
        i = ql.rfind(" " + p)
        if i > pos:
            pos = i + 1 + len(p)
    if pos < 0:
        return set()
    tail = ql[pos:].strip().split()
    words = []
    for w in tail:
        if w in _DIM_STOP:
            if words:               # stop at a boundary once we've collected the phrase
                break
            continue
        if len(w) > 2:
            words.append(w)
    return set(words)


def _dimension_candidates(phrase_words, sql_tables, sm):
    """ACCEPTABLE_CANDIDATES: {col} for CATEGORY/DIMENSION columns on the SQL's own tables whose name
    contains EVERY phrase word (data-driven — the schema's own dimension columns vs the user's words)."""
    if not phrase_words:
        return set()
    out = set()
    for k, c in (sm or {}).get("columns", {}).items():
        tbl, _, col = k.partition(".")
        if tbl not in sql_tables:
            continue
        role = (c.get("analytics_role") or "").upper()
        sem = (c.get("semantic_type") or "").upper()
        if sem not in ("CATEGORY", "CATEGORICAL") and role != "DIMENSION":
            continue
        cl = col.lower()
        if all(w in cl for w in phrase_words):
            out.add(col)
    return out


def dimension_alignment(query, sql, sm):
    """(outcome, reason). Validate the SQL GROUP BY dimension against the requested dimension referent.
    ALIGNED → allow; REFUSE → grouped outside the requested dimension family; CLARIFY → ≥2 acceptable
    candidates (genuine ambiguity — ask, never pick arbitrarily); NOT_APPLICABLE → no dimension phrase,
    no GROUP BY, or an empty candidate set (synonym / no name-match → decline to judge, never refuse)."""
    if not _dim_enabled() or not sql:
        return DIM_NOT_APPLICABLE, ""
    phrase = _dimension_phrase_words(query)
    if not phrase:
        return DIM_NOT_APPLICABLE, ""                     # no "by <dimension>" grouping intent
    facts = _facts(sql)
    group_cols = list(facts.get("groupings", []))
    if not group_cols:
        return DIM_NOT_APPLICABLE, ""                     # no GROUP BY — grouped-shape guard's concern
    acceptable = _dimension_candidates(phrase, set(facts.get("entities", [])), sm)
    if not acceptable:
        return DIM_NOT_APPLICABLE, ""                     # can't confidently build candidates → decline
    in_set = [g for g in group_cols if g in acceptable]
    if not in_set:
        return DIM_REFUSE, ("the SQL groups by a column outside the dimension you asked for "
                            f"({', '.join(sorted(group_cols))}) — expected one of "
                            f"{', '.join(sorted(acceptable))}")
    if len(acceptable) >= 2:
        return DIM_CLARIFY, ("this dimension is ambiguous — did you mean "
                             f"{' or '.join(sorted(acceptable))}?")
    return DIM_ALIGNED, ""
