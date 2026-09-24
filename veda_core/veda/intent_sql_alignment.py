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
    # Word-boundary on the UNIT (2026-09-15): the plain substring test matched " by month"
    # inside "top 3 amenities BY MONTHLY fee" and refused a correct ranking as a missing
    # time breakdown — before the adverb-modifies-measure check below could ever run.
    _preps = "|".join(re.escape(p.strip()) for p in _TIME_GROUP_PREPS)
    _units = "|".join(_TIME_UNITS)
    if re.search(rf"\b(?:{_preps})\s+(?:{_units})s?\b", ql):
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
        if not words or (" " + " ".join(words) + " ") not in ql:
            continue
        if len(words) == 1:
            # A ONE-word column ("amount") is not "named" by that word alone — it fired on
            # every question containing "amount" and vetoed a correct AVG(paid_amount)
            # because reminders_reminder.amount / accounts_generalledger.amount weren't the
            # anchor (2026-09-16, "average payment amount broken down by currency"). It
            # counts only when the question also names the column's table ("reminder amount").
            _ttoks = [t for t in re.split(r"[_\s]+", tbl.lower()) if len(t) > 2]
            _ttoks = [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in _ttoks]
            if not any((" " + t) in ql for t in _ttoks):
                continue
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
              " lower than ",
              # two-sided ranges state a comparison just as much as a one-sided one;
              # without them "between 100 and 50,000" was not even seen as a filter
              # request, so the filter-presence guard had nothing to check (2026-09-23).
              " between ", " range ", " in the range ", " from ")


def _filter_presence_enabled() -> bool:
    try:
        import config as _cfg
        return bool(getattr(_cfg, "INTENT_SQL_FILTER_PRESENCE_ENABLED", True))
    except Exception:
        return True


def _value_named_in_query(query, sm, sql_tables=None):
    """A query span that value-grounds to a column of one of the SQL's OWN tables (the
    same value-resolution the anchor scorer uses — query/resolution.resolve), e.g. "Kochi"
    → vendors.city. Returns "table.column" or None. 2026-09-15: "vendors in Kochi" and
    "amenities in the Sports category" were answered as UNFILTERED row lists; the value
    grounded (L6a passed) and the column was projected (qualifier gate passed), so nothing
    noticed the predicate had vanished. Same evidence-based shape as the boolean-flag case
    above — no vocabulary list."""
    try:
        from query.resolution import resolve
        for tr in resolve(query, sm):
            if getattr(tr, "grammar", False):
                continue
            for r in (tr.values or {}).get("direct", []):
                if sql_tables and r.get("table") not in sql_tables:
                    continue
                return f"{r.get('table')}.{r.get('column')}"
    except Exception:
        return None
    return None


def _boolean_flag_named(query, sm, sql_tables=None):
    """A BOOLEAN/FLAG column whose own distinctive name-word the query uses, e.g. "gated" for
    `is_gated`. Schema-driven (the column's words), no vocabulary list. `sql_tables` (2026-09-15):
    only flags on tables the SQL actually reads count — a flag on some unrelated table cannot
    be "the filter this SQL forgot" ("users created LAST month" used to match
    users_userpreference.is_LAST_name_obfuscated and refuse a correct temporal query)."""
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", (query or "").lower()) + " "
    cols = (sm or {}).get("columns", {})
    # A word that is ALSO a name-part of a NON-flag column on the same tables is explained by
    # that column, not by the flag: "total PAID amount" names paid_amount (a measure), not
    # is_paid — the guard used to refuse a correct grouped SUM as a "forgotten condition"
    # (2026-09-16, "total paid amount per currency").
    _non_flag_words = set()
    for k, c in cols.items():
        if (c.get("semantic_type") or "").upper() in ("BOOLEAN", "FLAG", "BOOL"):
            continue
        if sql_tables and k.split(".", 1)[0] not in sql_tables:
            continue
        _non_flag_words.update(w for w in k.split(".", 1)[1].lower().split("_") if len(w) > 3)
    for k, c in cols.items():
        st = (c.get("semantic_type") or "").upper()
        if st not in ("BOOLEAN", "FLAG", "BOOL"):
            continue
        if sql_tables and k.split(".", 1)[0] not in sql_tables:
            continue
        for w in k.split(".", 1)[1].lower().split("_"):
            if (len(w) > 3 and w not in ("flag", "is", "has") and w not in _non_flag_words
                    and (" " + w + " ") in ql):
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
    facts = _facts(sql)
    sql_tables = set(facts.get("entities") or [])
    wants = False
    if any(w in ql for w in _CMP_WORDS) and re.search(r"\d", ql):
        wants = True                                     # "above 4.0" / "over 250"
    elif _boolean_flag_named(query, sm, sql_tables):
        wants = True                                     # "gated" -> is_gated
    elif _value_named_in_query(query, sm, sql_tables):
        wants = True                                     # "vendors in KOCHI" -> city='Kochi'
    if not wants:
        return True, ""
    if facts.get("filters"):
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


# ── C. ENTITY COVERAGE — the question named several entities, the SQL answers a subset ─────────────
# The third referent class, and the one the other two cannot see: nothing about the SQL is WRONG, it
# is INCOMPLETE. "Audit of ticket updates, assignees and attachments by category" produced a correct
# COUNT over ticket updates alone, while validation reported "no requested filters were ignored" and
# the answer shipped at confidence 1.0 — the two entities that were dropped are not filters, so
# qualifier_completeness (filters + attributes of the QUERIED tables) has no opinion on them.
#
# Unlike A and B this NEVER refuses: a correct answer to part of the question is worth shipping, it
# just must not present itself as the whole answer. The caller records the miss as a failed check and
# drops the confidence, so the gap is visible in the explainability panel instead of silent.
#
# Deliberately conservative — it reports a gap only on evidence of ALL of:
#   (1) the noun grounds, through the SAME deterministic resolver the planner uses
#       (understanding.grounding.ground_entity: curated aliases → model vocabulary → name tokens),
#   (2) to a table JOINABLE to the SQL's own tables (the ingested join-path artifact) — so an
#       unrelated same-named table elsewhere in the schema is never reported, and
#   (3) at least one other named entity IS in the SQL — the SQL answers a SUBSET. A SQL that shares
#       no entity with the question is a wrong-anchor problem, which the guards above own.
# A noun the model has no vocabulary for stays unresolved and is simply not reported: this makes the
# check quiet-when-unsure (an under-report), never a false accusation.

def _coverage_enabled() -> bool:
    try:
        from config import INTENT_SQL_ENTITY_COVERAGE_ENABLED
        return bool(INTENT_SQL_ENTITY_COVERAGE_ENABLED)
    except Exception:
        return False


# Direct FK neighbours only. The ingested artifact reaches 4 hops, but at 2 the neighbourhood is
# most of the schema (everything meets at users_user) and the nouns start grounding to unrelated
# look-alikes — measured on the trigger query: 1 hop reports the attachment/activity entities the
# question named, 2 reports assets_amenitycategory. An entity the user expects in a "ticket audit"
# hangs directly off the tables being queried.
_MAX_COVERAGE_HOPS = 1


def _join_neighbourhood(sql_tables):
    """Tables reachable from the SQL's tables in the ingested join-path artifact (the same
    veda_join_paths.json the join planner uses). Empty set when the artifact is absent — the
    check then reports nothing, rather than judging joinability by guesswork."""
    try:
        import json as _json
        import os as _os
        from config import resolve_source_artifact
        p = resolve_source_artifact("veda_join_paths.json")
        pairs = (_json.load(open(p)) or {}).get("pairs", {}) if (p and _os.path.exists(p)) else {}
    except Exception:
        return set()
    near = set()
    for key, meta in pairs.items():
        a, _, b = str(key).partition("|")
        if not b:
            continue
        hops = (meta or {}).get("hops")
        if hops is not None and hops > _MAX_COVERAGE_HOPS:
            continue
        if a in sql_tables:
            near.add(b)
        if b in sql_tables:
            near.add(a)
    return near - set(sql_tables)


def _query_nouns(query):
    """Content words of the question, singularized, minus the query-LANGUAGE layer — the same
    content/grammar split validation.qualifier_completeness applies, reused rather than restated."""
    try:
        from veda.validation import _gate_strip, _singularize
        strip = _gate_strip()
    except Exception:
        # The query-LANGUAGE layer is unavailable. ground_entity applies its own stop-word
        # set anyway, and a grammar word that slips through simply grounds to nothing — so
        # fall back to that rather than silently skipping the check entirely.
        from veda.understanding.grounding import _STOP as strip, _singularize
    out, seen = [], set()
    for w in re.findall(r"[a-z]+", (query or "").lower()):
        if len(w) <= 2:
            continue
        s = _singularize(w)
        if w in strip or s in strip or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


_COVERAGE_STEM = 6


def _description_referent(noun, candidates, sm):
    """Fallback for a noun the NAME-based resolver can't place: match it against the
    candidate tables' own descriptions, and accept the match ONLY when exactly one table
    in the neighbourhood carries it.

    "assignees" appears nowhere in `worklists_ticketuser`'s name; that table describes
    itself as "a single ticket assignment record", and assignee/assignment share a stem
    no substring test finds — so the coverage check stayed silent about the dropped
    entity. Uniqueness is what makes this safe: "ticket" matches seven neighbours here,
    so it is ambiguous and ignored, while "assign" matches exactly one. A shared stem of
    """ + str(_COVERAGE_STEM) + """ characters is long enough that unrelated words don't
    collide, and short enough to cross the morphology (assign|ee/ment)."""
    if len(noun) < _COVERAGE_STEM:
        return None
    stem = noun[:_COVERAGE_STEM]
    tables = (sm or {}).get("tables", {}) or {}
    hits = set()
    for t in candidates:
        meta = tables.get(t) or {}
        text = f"{meta.get('primary_entity') or ''} {meta.get('business_purpose') or ''}".lower()
        if any(w[:_COVERAGE_STEM] == stem
               for w in re.findall(r"[a-z]+", text) if len(w) >= _COVERAGE_STEM):
            hits.add(t)
    return next(iter(hits)) if len(hits) == 1 else None


def entity_coverage(query, sql, sm=None):
    """(ok, missing_tables, missing_terms). ok=False means the SQL answers a STRICT SUBSET of the
    entities the question named — never a reason to refuse, only to stop claiming completeness.

    `missing_tables` are real table names, for the caller to label and display.
    `missing_terms` are the USER'S OWN WORDS for them ("assignees"), which is what the summariser
    must not claim to have measured — it never sees a table name, only the question and the
    numbers, so without these it papers over the gap in prose ("…in ticket updates and assignee
    activities", about a result containing no assignee figures at all)."""
    if not _coverage_enabled() or not sql:
        return True, [], []
    facts = _facts(sql)
    sql_tables = set(facts.get("entities") or [])
    if not sql_tables:
        return True, [], []
    near = _join_neighbourhood(sql_tables)
    if not near:
        return True, [], []                               # no join artifact → nothing to judge against
    try:
        from veda.understanding.grounding import ground_entity
    except Exception:
        return True, [], []
    candidates = near | sql_tables
    grounded = {}
    for noun in _query_nouns(query):
        try:
            t = ground_entity(noun, candidates, set(), sm)
        except Exception:
            t = None
        if t is None:
            t = _description_referent(noun, candidates, sm)
        if t:
            grounded.setdefault(t, noun)                  # the word the user used for this table
    missing = sorted(set(grounded) - sql_tables)
    if not missing or not (set(grounded) & sql_tables):
        return True, [], []                               # nothing dropped, or no shared entity
    return False, missing, [grounded[t] for t in missing]
