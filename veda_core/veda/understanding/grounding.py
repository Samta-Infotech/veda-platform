"""veda.understanding.grounding — the DETERMINISTIC anti-hallucination firewall.

Turns an (untrusted) RawIntent's CONCEPTS into a (trusted) GroundedIntent whose every
table/column is a real, schema-validated artifact — or a Refusal when a required
concept can't be grounded. NO LLM here: pure, testable, reproducible. This is the layer
that guarantees "never invent a table/column" no matter what the LLM said.

Grounding sources, in priority order (all existing VEDA infra, reused not rebuilt):
  1. exact/subset name-token match against real graph tables (most specific wins)
  2. the curated entity glossary (business noun → table)
  3. retrieval evidence (optional, when `results` supplied)
A concept that matches none of these does NOT get a guessed table — it goes to
`unresolved`, and the orchestrator turns a required unresolved into a Refusal.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from veda.understanding.schema import (
    RawIntent, GroundedIntent, GroundedMeasure, GroundedFilter, Refusal,
)
from veda.routing import _name_toks as _routing_name_toks
from query.entity_resolver import _entity_glossary

# language connectives that are NEVER entities/filters (schema-agnostic)
_STOP = frozenset({
    "the", "a", "an", "their", "them", "they", "with", "and", "or", "of", "for", "by",
    "per", "each", "all", "to", "from", "in", "on", "across", "relate", "related",
    "show", "list", "give", "get", "number", "count", "total", "amount", "sum",
    "average", "avg", "how", "many", "much", "value", "values",
    # generic record nouns an LLM appends to an entity ("maintenance record", "payment
    # entries") — never part of a table's name (2026-09-16)
    "record", "records", "entry", "entries", "row", "rows", "item", "items", "data", "detail", "details",
})
# Column phrases keep the measure nouns an entity phrase drops: "amount", "count", "value"
# ARE column names (maintenance.amount resolved to nothing while they were stop words).
_COL_STOP = _STOP - {"amount", "count", "number", "total", "sum", "average", "avg", "value", "values"}

_AGG_WORDS = {  # measure phrase → aggregation kind
    "count": "count", "number": "count", "how many": "count",
    "total": "sum", "sum": "sum",
    "average": "avg", "avg": "avg", "mean": "avg",
    "highest": "max", "max": "max", "maximum": "max", "largest": "max",
    "lowest": "min", "min": "min", "minimum": "min", "smallest": "min",
}


def _singularize(w: str) -> str:
    """Singular form of a token. Delegates to the shared enrichment singularizer; if that
    module is unavailable, falls back to a naive trailing-'s' strip (only for words > 3
    chars, so short tokens like 'is'/'as' are left intact)."""
    try:
        from retrieval.query_enrichment import _singularize as _s
        return _s(w)
    except ImportError:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _name_toks(t: str, sm=None):
    """Schema-aware name tokens for a table, via the shared routing tokenizer; falls back to
    a naive underscore split (parts > 2 chars) if that path errors."""
    try:
        toks = _routing_name_toks(t, sm)
    except Exception:
        toks = {p for p in t.split("_") if len(p) > 2}
    # singularised, like the concept side (M2, 2026-09-16): "amenity" must meet
    # amenities_catalog's {amenities, catalog} — it refused the anchor before this.
    return {_singularize(p) for p in toks}


def _glossary() -> Dict[str, str]:
    """The curated business-noun → table glossary (empty dict if unavailable/erroring)."""
    try:
        return _entity_glossary() or {}
    except Exception:
        return {}


def _concept_tokens(concept: str) -> List[str]:
    """Lowercase alpha tokens of a concept phrase, singularized and stop-word filtered."""
    return [_singularize(w) for w in re.findall(r"[a-z]+", (concept or "").lower())
            if len(w) > 2 and w not in _STOP]


# Grounding METHODS, in trust order (pre-M3 item 1, 2026-09-16). The first four are
# evidence the user NAMED the table (exact name / curated glossary / the model's own
# per-table vocabulary / name tokens); "retrieval" is a ranked guess. Only the named
# methods make a grounded anchor RE-ENTRY-ELIGIBLE; a retrieval grounding stays a
# candidate — see pipeline.py's re-entry override and GroundedIntent.reentry_eligible.
GROUND_EXACT, GROUND_GLOSSARY, GROUND_VOCAB, GROUND_NAME_TOKENS, GROUND_RETRIEVAL = (
    "exact_name", "glossary", "table_vocabulary", "name_tokens", "retrieval")
REENTRY_METHODS = frozenset({GROUND_EXACT, GROUND_GLOSSARY, GROUND_VOCAB, GROUND_NAME_TOKENS})
_LAST_METHOD: Dict[str, str] = {}      # concept → method of the most recent ground_entity()


def anchor_named_in_query(query: str, table: str, sm=None) -> Optional[str]:
    """Does the QUESTION carry name evidence for `table`? Returns the method
    (GROUND_NAME_TOKENS | GROUND_VOCAB) or None. Same evidence ground_entity accepts for
    re-entry eligibility, applied to a router-chosen primary (pre-M3 item 1 follow-up,
    2026-09-18): a deterministic branch with nothing else anchoring it — the bare
    "how many X are there" count — must not count whatever table retrieval ranked first
    ("how many gizmos are there" → COUNT(*) FROM worklists_ticketuser). Tokens: the
    question's content words vs the table's name tokens and its L3 `primary_entity`."""
    if not query or not table:
        return None
    qtoks = {_singularize(w) for w in re.findall(r"[a-z]+", query.lower())
             if len(w) > 2 and w not in _STOP}
    if not qtoks:
        return None
    ttoks = _name_toks(table, sm)
    if ttoks and (ttoks & qtoks):
        return GROUND_NAME_TOKENS
    # the curated per-source glossary (veda_entity_aliases.json: business noun → table) is
    # user-supplied evidence, the same it is for ground_entity — "properties" → assets_asset
    gl = _glossary()
    if any(gl.get(t) == table for t in qtoks) or any(gl.get(t) == table for t in
                                                     re.findall(r"[a-z]+", query.lower())):
        return GROUND_GLOSSARY
    meta = ((sm or {}).get("tables", {}) or {}).get(table) or {}
    for n in (str(meta.get("primary_entity") or ""), str(meta.get("business_name") or ""),
              str(meta.get("display_name") or "")):
        ntoks = {_singularize(w) for w in re.findall(r"[a-z]+", n.lower()) if len(w) > 2 and w not in _STOP}
        if ntoks and (ntoks & qtoks):
            return GROUND_VOCAB
    return None


def ground_entity(concept: str, graph_tables, junctions, sm=None,
                  retrieval_scores: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Concept → the single best REAL table, or None if it can't be grounded.
    Deterministic: name-token match (most specific), then glossary, then retrieval.
    Never invents — None means 'the firewall must decide (refuse if required)'.
    Records HOW it grounded in _LAST_METHOD[concept] (read by ground())."""
    t = _ground_entity(concept, graph_tables, junctions, sm, retrieval_scores)
    return t


def _ground_entity(concept, graph_tables, junctions, sm, retrieval_scores):
    def _hit(table, method):
        _LAST_METHOD[(concept or "").strip().lower()] = method
        return table
    # 0. exact real-table match — the extractor's entity_catalog contains real table
    #    names (when a business_name is absent), so the LLM often echoes one verbatim.
    #    An exact hit on a real table IS valid grounding (not a guess) — and it sidesteps
    #    the compound-token mismatch ('accounts_paymenttransaction' tokenizes to
    #    {account, paymenttransaction} but the table's name_toks are {account, payment,
    #    transaction}). Case/space-insensitive.
    if concept:
        _c = concept.strip().lower().replace(" ", "_")
        for t in graph_tables:
            if t.lower() == _c and t not in junctions:
                return _hit(t, GROUND_EXACT)
    toks = _concept_tokens(concept)
    if not toks:
        return None
    concat = "".join(toks)
    # 1. curated glossary FIRST (business noun → table). It is a deliberate, human-verified
    #    per-source mapping, so it must OUT-PRIORITIZE a coincidental name-token match:
    #    "tenant" → users_user (glossary), NOT assets_leasetenant (which merely shares the
    #    'tenant' token). Checked before name-tokens for exactly this class.
    gl = _glossary()
    for key in (concat, *toks):
        if key in gl and gl[key] in graph_tables and gl[key] not in junctions:
            return _hit(gl[key], GROUND_GLOSSARY)
    # 1b. the L3 model's OWN table-level vocabulary. What semantic_layer_v2 actually emits
    #     per table (verified 2026-09-16): business_purpose, primary_entity ("what does each
    #     row represent"), table_type — there is NO business_name/aliases field, and its
    #     "glossary" is a generic domain vocabulary, not table names. So the usable
    #     vocabulary is primary_entity ("property listing" → assets_asset); business_name /
    #     display_name / aliases are read too for any model that carries them.
    #     Exact token-set match, unique winner only.
    _bn_hits = []
    for t, meta in ((sm or {}).get("tables", {}) or {}).items():
        if t not in graph_tables or t in junctions or not isinstance(meta, dict):
            continue
        names = [str(meta.get("business_name") or ""), str(meta.get("display_name") or ""),
                 str(meta.get("primary_entity") or "")]
        al = meta.get("aliases") or meta.get("synonyms") or []
        names += [str(a) for a in (al if isinstance(al, (list, tuple)) else [al])]
        for n in names:
            ntoks = {_singularize(w) for w in re.findall(r"[a-z]+", n.lower()) if len(w) > 2 and w not in _STOP}
            if ntoks and ntoks == set(toks):
                _bn_hits.append(t)
                break
    if len(set(_bn_hits)) == 1:
        return _hit(_bn_hits[0], GROUND_VOCAB)
    # 2a. EXACT name-token equality wins — even for a table the junction heuristic
    #     flagged. When the concept's tokens EXACTLY equal a table's name tokens, the user
    #     named that entity precisely (it's a query TARGET, not a fuzzy intermediate
    #     bridge): "asset type" {asset,type} → assets_assettype {asset,type}, even though
    #     assettype is (mis)classified as a junction. Junction-exclusion only applies to
    #     the fuzzy subset match below, never to an exact name.
    _toks = set(toks)
    exact = [t for t in graph_tables if _name_toks(t, sm) == _toks]
    if exact:
        return _hit(min(exact, key=len), GROUND_NAME_TOKENS)
    # 2b. fuzzy name-token match (subset / concat) — junctions excluded here (they're
    #     bridges, not the named target). Prefer the FEWEST name tokens (most exact:
    #     "user" → users_user {user}, not users_userrole {user,role}).
    matches = [t for t in graph_tables
               if t not in junctions and (set(toks) <= _name_toks(t, sm)
                                          or concat in _name_toks(t, sm))]
    if matches:
        return _hit(min(matches, key=lambda t: (len(_name_toks(t, sm)), len(t))), GROUND_NAME_TOKENS)
    # 3. retrieval evidence (optional): highest-scored table whose name shares a token
    if retrieval_scores:
        cand = [(s, t) for t, s in retrieval_scores.items()
                if t in graph_tables and t not in junctions and (set(toks) & _name_toks(t, sm))]
        if cand:
            return _hit(max(cand)[1], GROUND_RETRIEVAL)
        # 3b. (2026-09-16) no table shares a token with the concept and the model has no
        #     business vocabulary for it ("property" vs assets_asset: 0/178 tables carry a
        #     business_name on this source) — take retrieval's top table. Safe ONLY because
        #     a grounded intent is a CANDIDATE: its SQL is used when this anchor equals the
        #     router's primary, so retrieval's own ranking remains the authority.
        top = [(s, t) for t, s in retrieval_scores.items() if t in graph_tables and t not in junctions]
        if top:
            return _hit(max(top)[1], GROUND_RETRIEVAL)
    return None


def ground_measure(concept: Optional[str], anchor: Optional[str], graph_tables, junctions,
                   sm=None, intent: Optional[str] = None) -> Optional[GroundedMeasure]:
    """Measure concept → GroundedMeasure. "number of payment transactions" → COUNT of
    the payments table; "total paid amount" → SUM (column grounded downstream). Returns
    None when there's no measure (a plain list) or it can't be grounded to a kind.
    The INTENT's aggregate wins over words inside the phrase (2026-09-16): intent=avg with
    measure='total paid amount' is an average — the phrase word "total" is not the op."""
    if not concept:
        return None
    low = concept.lower()
    kind = intent if intent in ("count", "sum", "avg", "max", "min") else None
    if kind is None:
        for w, k in _AGG_WORDS.items():
            if w in low:
                kind = k
                break
    if kind is None:
        return None
    # "number/count of <entity>" → count that entity's table
    if kind == "count":
        tbl = ground_entity(concept, graph_tables, junctions, sm)
        return GroundedMeasure(kind="count", table=tbl, column=None, concept=concept)
    # sum/avg/max/min of a column — column resolution is done later against `anchor`'s
    # columns by the planner/generator; here we record the kind + concept (no guess).
    return GroundedMeasure(kind=kind, table=anchor, column=None, concept=concept)


# ── M2 (2026-09-16): column-level grounding with TYPE checks (L2 semantic_type) ─────────
# Dimensions must be groupable, measures numeric, time TEMPORAL; filters carry a typed op.
_DIM_TYPES = frozenset({"CATEGORY", "FLAG", "IDENTIFIER"})
_NUM_TYPES = frozenset({"METRIC", "MONETARY"})
# comparator phrase → SQL op; checked longest-first so "no more than" wins over "more than"
_COMPARATORS = (
    (" no more than ", "<="), (" no fewer than ", ">="), (" at least ", ">="), (" at most ", "<="),
    (" greater than ", ">"), (" more than ", ">"), (" fewer than ", "<"), (" less than ", "<"),
    (" over ", ">"), (" above ", ">"), (" under ", "<"), (" below ", "<"),
)
_AGG_STRIP_RE = re.compile(
    r"\b(?:total|sum|number|count|amount\s+of|no\.?\s+of|average|avg|mean|max|maximum|highest|"
    r"greatest|largest|min|minimum|lowest|smallest|least|most|of|distinct|unique|different)\b")


def anchor_columns(anchor: str, sm) -> Dict[str, dict]:
    """{col_name: meta} for the anchor's columns in the (scoped) semantic model."""
    cols = (sm or {}).get("columns", {}) or {}
    return {k.split(".", 1)[1]: (cols[k] or {}) for k in cols if k.split(".", 1)[0] == anchor}


def column_kind(meta: dict) -> Optional[str]:
    """'numeric' | 'dimension' | 'temporal' | 'text' | None, from L2 semantic_type first,
    then the lite model's analytics role, then the physical data_type. None = unknown."""
    st = str((meta or {}).get("semantic_type") or "").upper()
    if st in _NUM_TYPES:
        return "numeric"
    if st in _DIM_TYPES:
        return "dimension"
    if st == "TEMPORAL":
        return "temporal"
    if st == "FREE_TEXT":
        return "text"
    role = str((meta or {}).get("role") or (meta or {}).get("analytics_role") or "").upper()
    if role == "MEASURE":
        return "numeric"
    if role == "DIMENSION":
        return "dimension"
    dt = str((meta or {}).get("data_type") or (meta or {}).get("type") or "").lower()
    if any(x in dt for x in ("int", "numeric", "decimal", "float", "double", "money", "real")):
        return "numeric"
    if any(x in dt for x in ("date", "time")):
        return "temporal"
    return None


def resolve_anchor_column(concept: str, anchor: str, sm, *, numeric: bool = False,
                          kinds: Optional[set] = None, query: str = "") -> Optional[str]:
    """Ground a concept phrase to ONE real column of `anchor`, or None. Deterministic:
    (1) exact / underscore-joined name match, (2) every concept token in the column name,
    (3) business_role / aliases text, (4) a UNIQUE column sharing any token (ambiguous → None).
    `numeric=True` accepts only numeric columns (the M2 measure rule); `kinds` restricts the
    accepted column_kind set (dimensions: {"dimension"}). Never guesses."""
    if not concept or not anchor:
        return None
    cols = anchor_columns(anchor, sm)
    if not cols:
        return None
    anchor_toks = {_singularize(w) for w in re.findall(r"[a-z]+", anchor.lower().replace("_", " "))}
    toks = [_singularize(w) for w in re.findall(r"[a-z]+", _AGG_STRIP_RE.sub(" ", concept.lower()))
            if len(w) > 2 and w not in _COL_STOP]
    toks = [t for t in toks if t not in anchor_toks] or toks
    if not toks:
        return None

    def _ok(c):
        k = column_kind(cols.get(c, {}))
        if numeric:
            return k == "numeric"
        if kinds:
            return k in kinds or k is None          # unknown type: allow (lite models), never numeric
        return True

    joined = "_".join(toks)
    # the user's OWN word forms (question first — the LLM's concept may say "floor count"
    # where the question says "floors"), used only to break ties between candidates
    raw_toks = ({w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2 and w not in _COL_STOP}
                if query else
                {w for w in re.findall(r"[a-z]+", _AGG_STRIP_RE.sub(" ", concept.lower())) if len(w) > 2})
    exact = [c for c in cols if c.lower() == joined and _ok(c)]
    if exact:
        return exact[0]
    subset = [c for c in cols
              if set(toks) <= {_singularize(p) for p in c.lower().split("_")} and _ok(c)]
    if len(subset) == 1:
        return subset[0]
    if len(subset) > 1:
        # several columns share the concept's tokens ("floor" → floor_number, total_floors):
        # prefer the ones whose name carries the user's RAW word form ("floors" → total_floors);
        # still ambiguous → None (typed clarify upstream), never the first in dict order.
        raw_hit = [c for c in subset if raw_toks & set(c.lower().split("_"))]
        if len(raw_hit) == 1:
            return raw_hit[0]
        return None
    for c, meta in cols.items():
        blob = (str(meta.get("business_role") or "") + " " + str(meta.get("aliases") or "")
                + " " + str(meta.get("business_definition") or "")).lower()
        if blob and all(t in blob for t in toks) and _ok(c):
            return c
    shared = [c for c in cols
              if any(t in {_singularize(p) for p in c.lower().split("_")} for t in toks) and _ok(c)]
    return shared[0] if len(shared) == 1 else None


def ground_dimensions(concepts: List[str], anchor: str, sm):
    """'per X' concepts → real groupable anchor columns. Returns (grounded, unresolved)."""
    out, bad = [], []
    for d in (concepts or []):
        col = resolve_anchor_column(d, anchor, sm, kinds={"dimension"})
        if col:
            out.append(GroundedFilter(table=anchor, column=col, concept=d,
                                      semantic_type=(anchor_columns(anchor, sm).get(col) or {}).get("semantic_type")))
        elif d:
            bad.append(f"dimension:{d}")
    return out, bad


def _comparator_for(query: str) -> str:
    q = " " + (query or "").lower() + " "
    for phrase, op in _COMPARATORS:
        if phrase in q:
            return op
    return "="


def ground_filters(raw_filters: List[Dict[str, Any]], query: str, anchor: str, sm):
    """Filters are grounded against DATA, not the LLM's word: (1) the value arbiter
    (query/value_arbiter + the scoped typed value lookup) classifies query spans that are
    sampled values of the anchor's columns → typed '='/'!=' predicates; (2) a numeric
    literal whose concept resolves to a NUMERIC anchor column becomes a comparator
    predicate with the op read from the question ("more than 3 floors" → total_floors > 3);
    (3) anything the LLM listed that neither grounds → unresolved (typed clarify upstream).
    Returns (grounded, unresolved)."""
    out: List[GroundedFilter] = []
    bad: List[str] = []
    cols = anchor_columns(anchor, sm)
    covered_norm = set()
    try:
        from query.value_arbiter import arbitrate, anchor_filters, build_schema_terms
        from query.resolution import typed_value_lookup
        arb = arbitrate(query, typed_value_lookup(), build_schema_terms(sm))
        for f in anchor_filters(arb, anchor):
            vn = str(f.get("value_norm") or "").lower()
            out.append(GroundedFilter(table=anchor, column=f["column"], value=vn,
                                      concept=str(f.get("value") or vn),
                                      op=("!=" if f.get("kind") == "negated_value" else "="),
                                      semantic_type=(cols.get(f["column"]) or {}).get("semantic_type")))
            covered_norm.add(vn)
    except Exception:
        pass                                     # arbiter unavailable → only numeric/unresolved below
    # EXISTENCE phrasing the LLM tends to drop ("how many users HAVE A last login", "assets
    # WITH A society"): the concept after have/has/with + article, when it resolves to a
    # column of the anchor, is an IS NOT NULL predicate — the "existence_count" gap
    # (2026-09-16). Data-validated (the column must exist on the anchor), no value needed.
    for m in re.finditer(r"\b(?:have|has|with)\s+(?:a|an|any)\s+([a-z][a-z ]{2,30}?)\s*(?:$|\?|,|\band\b|\bor\b|\bthat\b|\bwho\b|\bwhich\b)",
                         (query or "").lower()):
        concept = m.group(1).strip()
        col = resolve_anchor_column(concept, anchor, sm, query=query)
        if col and col not in {f.column for f in out}:
            out.append(GroundedFilter(table=anchor, column=col, value=None, concept=concept,
                                      op="IS NOT NULL",
                                      semantic_type=(cols.get(col) or {}).get("semantic_type")))
    for f in (raw_filters or []):
        if not isinstance(f, dict):
            continue
        concept = str(f.get("concept") or "").strip()
        value = str(f.get("value") or "").strip()
        vn = re.sub(r"\s+", " ", value.lower())
        if not value:
            continue
        if any(x.op == "IS NOT NULL" and x.concept == concept.lower() for x in out):
            continue                             # already grounded as an existence predicate
        if vn in covered_norm or any(vn in c or c in vn for c in covered_norm if c):
            continue                             # the arbiter already grounded it on data
        # numeric literal — bare ("500") or with the comparator folded into the value by
        # the LLM ("above 500", "> 3"): the number is the value, the op comes from the
        # value text first, else the question ("more than 3 floors").
        num = re.search(r"-?\d+(?:\.\d+)?", vn)
        if num and re.fullmatch(r"[a-z<>=!\s]*-?\d+(?:\.\d+)?[a-z\s]*", vn):
            col = resolve_anchor_column(concept, anchor, sm, numeric=True, query=query)
            if col:
                ns = num.group(0)
                op = _comparator_for(vn)
                if op == "=":
                    op = {">=": ">=", "<=": "<=", ">": ">", "<": "<"}.get(
                        re.sub(r"[^<>=]", "", vn) or "", None) or _comparator_for(query)
                out.append(GroundedFilter(table=anchor, column=col, value=float(ns) if "." in ns else int(ns),
                                          concept=concept, op=op, numeric=True,
                                          semantic_type=(cols.get(col) or {}).get("semantic_type")))
                continue
        bad.append(f"filter:{concept}={value}")
    return out, bad


def ground_time(anchor: str, sm, tf) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """L1 temporal window → the anchor's canonical TEMPORAL column (pipeline's own
    chooser, one source of truth). (None, None) when there is no window; (None,
    'time:<window>') when the anchor has no TEMPORAL column at all."""
    start = getattr(tf, "start", None) if tf else None
    end = getattr(tf, "end", None) if tf else None
    if not (start or end):
        return None, None
    try:
        from veda.pipeline import _resolve_temporal_column
        col = _resolve_temporal_column(anchor, sm)
    except Exception:
        col = None
    if not col:
        return None, f"time:{start or ''}..{end or ''}"
    return {"column": col, "start": start, "end": end}, None


def ground_distinct(raw: RawIntent, query: str, anchor: str, sm) -> Optional[str]:
    """'how many amenity categories are there' → the DIMENSION column a COUNT is over
    (COUNT(DISTINCT col)); None for a plain row count. Only for count intents whose
    measure/grain phrase names a groupable column of the anchor (never the anchor itself)."""
    if raw.intent != "count":
        return None
    anchor_toks = {_singularize(w) for w in re.findall(r"[a-z]+", anchor.lower().replace("_", " "))}
    cols = anchor_columns(anchor, sm)

    def _residue(phrase):
        toks = [_singularize(w) for w in re.findall(r"[a-z]+", _AGG_STRIP_RE.sub(" ", (phrase or "").lower()))
                if len(w) > 2 and w not in _STOP]
        return [t for t in toks if t not in anchor_toks]

    # (1) the measure phrase ("number of amenity categories"): residual tokens ⊆ column parts
    for c, meta in cols.items():
        r = _residue(raw.measure)
        if r and set(r) <= {_singularize(p) for p in c.lower().split("_")} and column_kind(meta) == "dimension":
            return c
    # (2) the grain phrase ("amenity category" on amenities_catalog): the residue must EQUAL
    #     the column's parts — "property" on assets_asset must not become corner_property.
    r = _residue(raw.grain)
    if r:
        for c, meta in cols.items():
            if set(r) == {_singularize(p) for p in c.lower().split("_")} and column_kind(meta) == "dimension":
                return c
    return None


def ground(raw: RawIntent, sm, graph, junctions,
           retrieval_scores: Optional[Dict[str, float]] = None,
           min_confidence: float = 0.5, query: str = "", tf=None):
    """RawIntent → GroundedIntent | Refusal | None.

    None  = degrade (low confidence / nothing to ground) → caller uses existing path.
    Refusal = a REQUIRED concept couldn't be grounded, or intent is refuse/clarify.
    GroundedIntent = every artifact validated real.

    M2 (2026-09-16): dimensions, filters and the time window are grounded here too —
    with type checks (dimension ⇒ groupable, measure ⇒ numeric, time ⇒ TEMPORAL) and
    value checks through the value arbiter. A dimension or filter the question names
    that cannot be grounded is a typed CLARIFY (reason "ambiguous", the unresolved
    concepts listed) — never a silently narrower answer.
    """
    # Real tables = the relationship graph's ∪ the scoped semantic model's. A single-table
    # (lite) source has NO graph edges and therefore 0 graph tables — every concept on
    # source 5 refused as "ungrounded" until the model's own table list counted (2026-09-16).
    graph_tables = set((graph or {}).get("tables", []) or []) \
        | set(((sm or {}).get("tables", {}) or {}).keys())
    if not raw or not raw.is_valid_shape():
        return None
    if raw.intent in ("refuse", "clarify"):
        return Refusal(reason="impossible" if raw.intent == "refuse" else "ambiguous",
                       message=("This question can't be answered from the available data."
                                if raw.intent == "refuse"
                                else "This question is ambiguous — please specify."),
                       evidence={"llm_intent": raw.intent})
    if raw.confidence and raw.confidence < min_confidence:
        return None                          # not confident enough → degrade, don't guess

    unresolved: List[str] = []
    ev: Dict[str, Any] = {"grounded": {}}

    # grain → anchor (the single most important grounding — fixes grain-inversion)
    anchor = ground_entity(raw.grain, graph_tables, junctions, sm, retrieval_scores) \
        if raw.grain else None
    anchor_method = _LAST_METHOD.get((raw.grain or "").strip().lower()) if anchor else None
    if raw.grain and not anchor:
        unresolved.append(f"grain:{raw.grain}")
    else:
        ev["grounded"]["grain"] = anchor
        ev["grounded"]["grain_method"] = anchor_method

    # other entities → secondaries (distinct, real, not the anchor)
    secondaries: List[str] = []
    for e in (raw.entities or []):
        t = ground_entity(e, graph_tables, junctions, sm, retrieval_scores)
        if t and t != anchor and t not in secondaries:
            secondaries.append(t)
        elif not t and e:
            unresolved.append(f"entity:{e}")

    # anchor fallback: if grain didn't ground but an entity did, use the first entity
    if not anchor and secondaries:
        anchor = secondaries.pop(0)
        anchor_method = _LAST_METHOD.get(next((str(e).strip().lower() for e in (raw.entities or [])
                                               if ground_entity(e, graph_tables, junctions, sm, retrieval_scores) == anchor), ""))
        ev["grounded"]["grain"] = anchor
        ev["grounded"]["grain_method"] = anchor_method

    measure = ground_measure(raw.measure, anchor, graph_tables, junctions, sm, intent=raw.intent)

    # FIREWALL: an answer intent with NO grounded anchor at all cannot proceed → refuse
    if anchor is None:
        return Refusal(reason="ungrounded",
                       message=("Couldn't identify which data this question is about. "
                                "Please name the entity (e.g. properties, payments, users)."),
                       unresolved=unresolved or ([f"grain:{raw.grain}"] if raw.grain else []),
                       evidence=ev)

    # ── M2: column-level grounding on the anchor ─────────────────────────────
    # the LLM often lists a filtered concept under BOTH dimensions and filters
    # ("floor count" → dims=['floor count'], filters=[{concept:'floor count', value:'> 3'}]);
    # a constrained concept is a filter, not a breakdown.
    _fconcepts = {str(f.get("concept") or "").strip().lower() for f in (raw.filters or []) if isinstance(f, dict)}
    _dim_concepts = [d for d in (raw.dimensions or []) if str(d).strip().lower() not in _fconcepts]
    dims, bad_dims = ground_dimensions(_dim_concepts, anchor, sm)
    filters, bad_filters = ground_filters(raw.filters, query, anchor, sm)
    # A "dimension" the LLM named that is really a sampled VALUE ("properties in Mumbai"
    # → dimensions=["Mumbai"]) is a filter the arbiter already grounded — data wins over
    # the LLM's label; only a concept that is neither a column nor a value stays unresolved.
    _fvals = {str(f.value).lower() for f in filters} | {str(f.concept).lower() for f in filters}
    bad_dims = [b for b in bad_dims
                if b.split(":", 1)[1].strip().lower() not in _fvals
                and not any(b.split(":", 1)[1].strip().lower() in v for v in _fvals if v)]
    time_win, bad_time = ground_time(anchor, sm, tf)
    distinct_col = ground_distinct(raw, query, anchor, sm)
    ev["grounded"].update({
        "dimensions": [d.column for d in dims],
        "filters": [(f.column, f.op, f.value) for f in filters],
        "time": time_win, "distinct_column": distinct_col,
    })
    required_bad = bad_dims + bad_filters + ([bad_time] if bad_time else [])
    if required_bad:
        # A named breakdown / constraint / window that grounds to nothing is a typed
        # clarify: the answer without it would be a different question.
        what = "; ".join(required_bad)
        return Refusal(reason="ambiguous",
                       message=(f"I couldn't match part of the question to the data "
                                f"({what}). Which column did you mean?"),
                       unresolved=required_bad, evidence=ev)

    return GroundedIntent(
        intent=raw.intent, anchor=anchor, secondaries=secondaries, measure=measure,
        dimensions=dims, filters=filters, time=time_win, distinct_column=distinct_col,
        confidence=raw.confidence, anchor_method=anchor_method or GROUND_RETRIEVAL,
        evidence={**ev, "unresolved_nonfatal": unresolved},
    )
