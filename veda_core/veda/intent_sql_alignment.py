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

    def _mentioned(w: str) -> bool:
        """Is this name-word in the query, allowing the plural the user naturally writes?

        The column is `monthly_fee` but people ask about "monthly fees" — and an exact
        " fee " probe misses " fees ", so the adverb looked like a time breakdown and the
        temporal guard refused the whole question. Measured: "highest monthly fee" passed
        while "…and monthly fees" was refused, on the same column. A trailing "s" either
        way is enough here; this stays schema-driven (the column's own words) with no
        vocabulary list, exactly as the docstring promises.
        """
        base = w[:-1] if w.endswith("s") and len(w) > 3 else w
        return any((" " + f + " ") in ql for f in (w, base, base + "s"))

    for k, c in (sm or {}).get("columns", {}).items():
        if (c.get("analytics_role") or "").upper() != "MEASURE":
            continue
        words = set(k.split(".", 1)[1].lower().split("_"))
        if adverb in words and any(_mentioned(w) for w in words - {adverb} if len(w) > 2):
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
    return False, ("this asks for a breakdown over time, but the information it would be grouped "
                   "by isn't a date — so a per-period breakdown can't be produced for it")


def _plain(name: str) -> str:
    """An internal identifier rendered as ordinary words.

    These refusal strings are shown to the USER verbatim as the reply, so they may
    not carry `column`, `table`, `SQL` or a raw identifier — the same rule the whole
    safe-projection layer enforces on the explainability payload. Measured live on a
    verified-cache clarify: the reply read "the query's measure lives on another
    table than the one the SQL ranks/aggregates".
    """
    return " ".join(w for w in str(name).replace("_", " ").split() if w)


def _plain_list(names) -> str:
    vals = [_plain(n) for n in sorted(names)]
    if len(vals) <= 1:
        return vals[0] if vals else ""
    return ", ".join(vals[:-1]) + " or " + vals[-1]


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
    # A measure the query names as a THRESHOLD ("carpet area above 1000") is a filter, not
    # the figure to rank by — the SQL applies it as `"carpet_area" > 1000`. Measured
    # 2026-09-26: that drill was refused here although the filter was in the SQL.
    named = {(t, c) for (t, c) in named
             if not re.search(rf'"{re.escape(c)}"\s*(?:>=|<=|>|<)', sql)}
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
    # Plain language (see the aggregate-omission message below for the same reasoning).
    return False, ("I couldn't match the figure you asked about to the data I'd have to rank it "
                   "by, so I'd rather not show a number that might be wrong")


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
    # Plain language — an end user reads this, and "the SQL returns rows without an aggregate"
    # tells them nothing they can act on while exposing how the query was built. What matters to
    # them is that we could not produce the single figure they asked for and are not going to
    # show a number we do not trust.
    return False, ("I couldn't work out a reliable total for this, so I'd rather not show a "
                   "figure that might be wrong")


# ── D. FILTER presence ─────────────────────────────────────────────────────────────────────────────
# Comparison words the query uses to express a numeric predicate. A closed grammar list, the same
# shape as _AGG_INTENT / _TIME_UNITS above — not a semantic vocabulary.
_CMP_WORDS = (" above ", " over ", " below ", " under ", " more than ", " less than ",
              " greater than ", " fewer than ", " at least ", " at most ", " higher than ",
              " lower than ")


def _filter_presence_enabled() -> bool:
    try:
        import config as _cfg
        return bool(getattr(_cfg, "INTENT_SQL_FILTER_PRESENCE_ENABLED", True))
    except Exception:
        return True


def _boolean_flag_named(query, sm, tables=None):
    """A BOOLEAN/FLAG column the query NAMES, e.g. "gated" for `is_gated`. Schema-driven (the
    column's words), no vocabulary list.

    Two conditions, both learned the hard way (measured 2026-09-26): "What is the distribution
    of properties by city?" was refused because the single word "city" matched
    `services_valuebundlepricing.is_city_dependent` — a flag of an unrelated table.
      * only columns of the tables the SQL actually reads (`tables`, when given);
      * EVERY distinctive word of the column's name must be in the query ("city dependent"),
        not any one of them."""
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", (query or "").lower()) + " "
    for k, c in (sm or {}).get("columns", {}).items():
        st = (c.get("semantic_type") or "").upper()
        if st not in ("BOOLEAN", "FLAG", "BOOL"):
            continue
        tbl, _, col = k.partition(".")
        if tables is not None and tbl not in tables:
            continue
        words = [w for w in col.lower().split("_")
                 if len(w) > 3 and w not in ("flag", "is", "has")]
        if words and all((" " + w + " ") in ql for w in words):
            return k
    return None


def filter_presence_ok(query, sql, sm):
    """(ok, reason). Refuse when the question states a FILTER the SQL never applied.

    The silent-wrong this exists for: "Only those rated above 4.0" produced
    `SELECT rating, vendor_id, city FROM vendors LIMIT 100` — no WHERE at all — and the answer
    layer then reported "5 vendors have ratings above 4.0" off six unfiltered rows. Same shape on
    "Only the ones that are gated": the SQL merely PROJECTED is_gated and the summary became
    "60% of assets are gated". Both read as confident answers and both are invented.
    qualifier_completeness does not catch it because the column IS present in the SQL — as a
    projection — so the qualifier looks represented.

    Two filter intents are detected, both cheap and both evidence-based:
      * a comparison phrase plus a number ("above 4.0", "over 250");
      * a BOOLEAN column named by the query ("gated" -> is_gated).
    Either one with ZERO filters in the SQL is an omission. A SQL that filters anything at all
    passes — this only catches "the predicate vanished entirely", never a wrong predicate."""
    if not _filter_presence_enabled() or not sql:
        return True, ""
    ql = " " + re.sub(r"[^a-z0-9.]", " ", (query or "").lower()) + " "
    wants = False
    if any(w in ql for w in _CMP_WORDS) and re.search(r"\d", ql):
        wants = True                                     # "above 4.0" / "over 250"
    elif _boolean_flag_named(query, sm, tables=set(re.findall(
            r'(?:FROM|JOIN)\s+(?:[A-Za-z0-9_]+\.)*"?([A-Za-z_][A-Za-z0-9_]*)"?', sql or "",
            re.I))):
        wants = True                                     # "gated" -> is_gated
    if not wants:
        return True, ""
    if _facts(sql).get("filters"):
        return True, ""                                  # SQL filters something -> not omitted
    return False, ("I couldn't apply the condition you asked for, so I'd rather not show numbers "
                   "that ignore it")


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
        return DIM_REFUSE, ("this would be broken down by "
                            f"{_plain_list(group_cols)}, which isn't the grouping you asked for "
                            f"— the available groupings here are {_plain_list(acceptable)}")
    if len(acceptable) >= 2:
        return DIM_CLARIFY, ("more than one grouping fits what you asked for — did you mean "
                             f"{_plain_list(acceptable)}?")
    return DIM_ALIGNED, ""
