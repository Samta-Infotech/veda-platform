"""chatbot.memory.frame — QueryFrame: structured analytical memory (§3/§6/§7/§9
of docs/MEMORY_ARCHITECTURE.md).

Everything in this module is a PURE function — no I/O, no LLM calls, no Redis,
no veda_core import (chatbot/ runs in the api container, which must never
import veda_core directly — same boundary chatbot/llm.py and
apps/query/inference_client.py already document). That boundary is exactly
why validation here is deliberately light: real schema/column/value
validation already happens server-side, in the inference tier's own
deterministic pipeline (veda_core/veda/pipeline.py's L6a-L6c checks), on
EVERY call regardless of whether memory supplied the query text. Memory's job
is to hand that pipeline a BETTER-GROUNDED input than raw ellipsis text —
never to bypass or duplicate its validation.

Every field that ends up in a stored QueryFrame comes from ONE of exactly two
places:
  1. harvest_frame() — extracted from an engine_result that already has
     status == "answered", i.e. it already passed L6a-L6c server-side.
  2. The user's own current message, copied verbatim (render_frame_as_query
     never invents a filter value that isn't either an old proven fact or a
     substring the user just typed).
Nothing here is ever an LLM's free invention.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class FilterFact(TypedDict, total=False):
    field: str          # business-facing field name (from business_explain.py)
    column: str         # the RAW column the executed SQL filtered on. `field` is a LABEL
                        # ("Location") and the table has no such column (it is city_name),
                        # so a remembered filter could only ever travel as a bare value —
                        # which does not always name its own column. This does.
    operator: str
    value: Any
    source: str          # "executed_sql" — the only source memory ever writes


# Fields whose filters are MUTUALLY EXCLUSIVE at equality: a second `equals` on one of
# them cannot coexist with the first ("Year = 2025 AND Year = 2024" selects nothing), so
# a new value REPLACES rather than joins. Everything else stays additive, which is what
# keeps "Country = India AND CustomerType = Enterprise" representable at the same time.
#
# Detected structurally, from the operator, rather than from a list of field names: any
# field is single-valued under `equals`, and no field is under `in`/`between`/`>=`. That
# is a property of SQL, not of this schema, so it needs no per-schema configuration and
# works on a source this code has never seen.
_SINGLE_VALUED_OPERATORS = frozenset({"equals", "is", "=", "=="})

# Temporal fields, recognised by name — the only case where the REPLACEMENT VALUE may not
# be a literal the user typed ("last year", "previous quarter"). See apply_context_delta.
_TEMPORAL_FIELD_WORDS = frozenset({
    "year", "years", "date", "dates", "month", "months", "quarter", "quarters",
    "week", "weeks", "day", "days", "period", "periods", "time", "times",
    "fy", "financial", "fiscal", "yr", "dt", "timestamp", "datetime",
})


def _tokens(field: Optional[str]) -> frozenset:
    """A field name split into WORDS — on separators and on camelCase boundaries.

    Matching on whole words rather than raw substrings is the difference between a
    working guard and a silent one. Measured 2026-09-17: substring matching made
    "candidate_name" temporal (it contains "date"), "notify_email" temporal (it contains
    "fy"), and made the field "City" match a frame holding "Capacity" — each of which
    DELETED a filter the user had asked to keep, with no signal that anything was gone.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(field or ""))
    return frozenset(t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t)


def is_temporal_field(field: Optional[str]) -> bool:
    """True when a WORD of the field name is a time word — never when one merely occurs
    inside another word. See _tokens for what the substring version cost."""
    return bool(_tokens(field) & _TEMPORAL_FIELD_WORDS)


def _same_field(a: Optional[str], b: Optional[str]) -> bool:
    """Do these two names refer to the same field?

    Business field names arrive from business_explain's field_of() on one side and from
    a model's echo of the user's words on the other, so case and separators differ while
    the concept does not: "Order Year" / "order_year" / "orderYear" are one field, and a
    bare "Year" legitimately names a frame's "Fiscal Year".

    Word-level, not substring: one name matches the other when its WORDS are all present
    in the other's. "city" and "capacity" share no word and no longer collide, nor do
    "id" and "paid" — both of which previously matched and removed the wrong filter.
    """
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


class DrillLevel(TypedDict, total=False):
    dimension: str      # the humanised LABEL, kept because the path is also shown to people
    column: str         # the RAW column the executed SQL filtered on
    value: Any
    # WHY BOTH. A level used to record only the label, and rebuild_frame_from_stack
    # therefore re-derived a column-less filter after a "go back". ConversationContext
    # drops a filter that cannot name its own column — deliberately, because re-grounding
    # from a bare value is the guess that once put an is_gated filter onto all_day_access —
    # so the remaining level never reached the engine, the answer came back unfiltered, and
    # memory_write_node's prune (which keeps only levels still present in the frame's
    # filters) then dropped EVERY level. One "go back" erased the whole path, measured
    # 2026-09-24 at depth 2. Carrying the column makes the rebuilt filter usable again.


class OrderFact(TypedDict, total=False):
    field: str          # raw result column the executed SQL ordered on
    desc: bool          # True = highest first


class QueryFrame(TypedDict, total=False):
    version: int
    tenant: str
    session_id: str
    entity: Optional[str]           # raw table name (engine_result["table"])
    entity_display: Optional[str]   # humanized dataset name (explain.data_used.datasets[0])
    base_query: Optional[str]       # the user's own question before any narrowing — replayed
                                    # when "go back" pops the last drill level (a table name
                                    # alone is not a question the engine can answer)
    understanding: Optional[str]    # engine's own deterministic summary sentence
                                     # (business_explain.build_explain, NOT LLM prose)
    filters: List[FilterFact]
    group_by: List[str]
    aggregation: Optional[str]     # "count" / "sum" / "average" / … — WHICH aggregate the
                                   # previous turn computed, so replaying its shape does not
                                   # have to assume one
    # What the previous turn MEASURED and how it was RANKED. Without these the frame
    # remembered only "which rows" (entity + filters) and never "what about them", so
    # a follow-up that changes only the ranking — "the most expensive instead",
    # "top 10 rather than 100" — reached the engine with no measure to change and had
    # to re-guess one from the words alone. All three are harvested from the executed
    # SQL's own analytics (never an LLM), same evidence-only rule as `filters`.
    measures: List[str]            # analytics["query_measures"] — raw column names
    available_measures: List[str]  # analytics["available_measures"] — the TABLE's measure
                                   # columns, used ONLY to ground a measure addition
    order_by: List[OrderFact]      # analytics["orderings"]
    limit: Optional[int]           # analytics["limit"]
    # Which source the entity above was resolved in. An entity name alone is not
    # unique across a multi-source scope, so a follow-up could silently re-resolve
    # the same table name in a DIFFERENT source than the turn it is following up on.
    source_id: Optional[int]
    # Which engine head produced this frame — "rag" for a document answer, a SQL head
    # otherwise. Kept explicit rather than inferred from which fields are empty.
    route: str
    drill_path: List[DrillLevel]
    last_sql: Optional[str]
    last_row_count: Optional[int]
    last_status: str
    confidence: float
    updated_at: str
    turn_index: int


_MAX_DRILL_DEPTH = 10


def empty_frame(tenant: str, session_id: str) -> QueryFrame:
    return {
        "version": 0,
        "tenant": tenant,
        "session_id": session_id,
        "entity": None,
        "entity_display": None,
        "understanding": None,
        "filters": [],
        "group_by": [],
        "measures": [],
        "order_by": [],
        "limit": None,
        "source_id": None,
        "route": "",
        "drill_path": [],
        "last_sql": None,
        "last_row_count": None,
        "last_status": "",
        "confidence": 0.0,
        "updated_at": _now(),
        "turn_index": 0,
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# Names the SQL builder generates for itself — CTEs, derived tables, aliases. A real
# table name from an ingested source never looks like this.
_ENGINE_ALIAS_RE = re.compile(
    r"^(?:agg|cte|sub|tmp|temp|t|q|src|derived|anon)[_\-]?\d*$", re.IGNORECASE)

# Names the ENGINE uses for a way of answering rather than for a thing in the data.
# "federated" is emitted verbatim by veda_hybrid when a cross-source plan has no single
# group table (`table = ... if _group_table else "federated"`), and it reached the frame
# as if it were an entity: every federated turn wrote `entity="federated"` under ONE
# source's key. Downstream that is worse than no memory — is_topic_switch compares it,
# the next turn's resolved query names it, and recall tells the user their last answer
# came from "federated". Same refusal as an engine alias, different vocabulary: an alias
# is a name the SQL invented, this is a name the ROUTE invented.
_ROUTE_NAMES = frozenset({"federated", "hybrid", "rag", "deterministic", "multi_source",
                          "multisource", "cross_source", "unknown", "none"})


def _looks_like_an_engine_alias(table: Optional[str]) -> bool:
    """Is this a name the engine made up, rather than something in the customer's data?"""
    name = str(table or "").strip()
    if not name:
        return False
    return bool(_ENGINE_ALIAS_RE.match(name)) or name.lower() in _ROUTE_NAMES


def _cited_first(answer: Any, datasets: List[Any]) -> List[Any]:
    """Reorder `datasets` so the documents the answer's own "Sources:" line names come
    first, in the order it names them. Only the text after the LAST "Sources:" is read —
    the document prompt puts citations on the final line and nowhere else — and a name
    counts as cited when its letters and digits appear there. No "Sources:" line, or
    none of the datasets named in it, changes nothing.
    """
    text = answer if isinstance(answer, str) else ""
    at = text.lower().rfind("sources:")
    if at < 0:
        return datasets
    cited = _normalise_document_name(text[at + len("sources:"):])
    hits = []
    for d in datasets:
        pos = cited.find(_normalise_document_name(str(d))) if d else -1
        if pos >= 0 and _normalise_document_name(str(d)):
            hits.append((pos, d))
    if not hits:
        return datasets
    first = [d for _, d in sorted(hits, key=lambda h: h[0])]
    return first + [d for d in datasets if d not in first]


def harvest_frame(engine_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pure extraction, NO LLM: turn an "answered" engine_result (the full
    dict chatbot/nodes.py::call_engine_node stores at state["engine_result"],
    i.e. veda_core/veda/pipeline.py's _done() payload forwarded verbatim over
    the wire) into the harvestable facts for a QueryFrame. Returns None when
    there's nothing safe to harvest (wrong status, or business_explain itself
    failed server-side — pipeline.py already logs+degrades that case, we just
    skip the write rather than harvest partial/absent data).
    """
    if not engine_result or engine_result.get("status") != "answered":
        return None
    explain = engine_result.get("explain")
    if not explain:
        return None

    data_used = explain.get("data_used") or {}
    datasets = data_used.get("datasets") or []
    filters_applied = (explain.get("filters") or {}).get("applied") or []
    operations = explain.get("operations") or []
    understanding = (explain.get("understanding") or {}).get("summary")

    # The RAW column when explain provides it, falling back to the label parsed out of the
    # summary for entries written before it did (a frame lives 7 days, so those are still
    # in flight). Only the column can be replayed as SQL; the label is display text.
    group_by = [op.get("column") or op["summary"][len("Group by "):]
                for op in operations
                if op.get("type") == "group" and op.get("summary", "").startswith("Group by ")]

    # WHICH aggregate the previous turn computed. Without it, replaying a remembered shape
    # has to assume one, and assuming SUM turned a COUNT(DISTINCT id) distribution into
    # SUM("id") — summing a primary key, which the summariser then reported as a real
    # figure ("id_total above the average of 291,689", measured 2026-09-23). The kinds come
    # from explain's own operation types, which are derived from the executed SQL.
    aggregation = next((op.get("type") for op in operations
                        if op.get("type") in ("count", "sum", "average", "avg",
                                              "minimum", "maximum", "min", "max")), None)

    rows = engine_result.get("rows")
    row_count = len(rows) if isinstance(rows, list) else None

    # The measure/ordering/limit facts come from `analytics` rather than `explain`:
    # analytics carries them STRUCTURED (raw column names + a desc flag), whereas
    # explain renders the same facts as prose for humans ("Sort by Expected Price
    # (lowest first)"), which would have to be parsed back. Absent on a federated
    # result that carries no analytics — the frame simply keeps fewer facts then,
    # exactly as it did before these fields existed.
    analytics = engine_result.get("analytics") or {}
    order_by = [{"field": entry[0], "desc": bool(entry[1])}
                for entry in (analytics.get("orderings") or [])
                if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0]]

    # An engine-internal name is not a business entity. Measured 2026-09-17: a query
    # whose SQL used a CTE came back with table="agg_0", the frame stored it, and the
    # NEXT turn was sent to the engine as "what about Pune (for Agg 0s (agg_0))" — a
    # string this layer invented, handed to a pipeline that parses every word as data
    # ("Could you clarify if 'agg' is a column name?", 135s). Nothing downstream can
    # tell an invented entity from a real one, so it is refused here.
    # Which head answered. Recorded because a document answer and a SQL answer are
    # different KINDS of memory, and inferring the difference from absent fields
    # ("no sql, so probably retrieval") is the sort of guess that goes stale.
    route = engine_result.get("_route") or engine_result.get("route") or ""

    entity = engine_result.get("table")
    if not entity and datasets:
        # datasets arrive in RETRIEVAL order (the document of the most similar passage
        # first), which is not the document the answer came from. Measured 2026-09-25
        # (demo D2): the reply said "Sources: (msa_green_tower.pdf)" while datasets[0] was
        # the employee handbook, so the frame filed the handbook and the follow-up searched
        # it instead of the MSA. The document(s) the answer itself cites go first.
        datasets = _cited_first(engine_result.get("answer"), datasets)
    if not entity and datasets:
        # A retrieval answer has no table — it has a DOCUMENT. Measured 2026-09-21:
        # without this the frame was written with entity=None, and since every
        # downstream check reads frame["entity"], the whole memory layer treated a
        # document conversation as if nothing had ever been answered: no recall, no
        # re-present, no follow-up grounding. The dataset name is not invented here —
        # it is the engine's own business-facing name, already used for entity_display.
        entity = datasets[0]
    # WHERE the entity came from, recorded as a fact instead of inferred later from the
    # route name. Measured 2026-09-22: the HYBRID head tries SQL first and answers from
    # the documents when SQL finds nothing, but records route="hybrid" — which is not in
    # _DOCUMENT_ROUTES, so every frame built by that head was treated as relational.
    # The employee handbook is answered by the hybrid head, so in a whole live document
    # conversation none of the document logic (anchoring, shape/drill refusals) ever
    # ran. A route NAME describes which head executed; this describes what the answer
    # was actually built from, which is the thing the memory layer needs to know.
    entity_is_document = not engine_result.get("table") and bool(datasets)
    if _looks_like_an_engine_alias(entity):
        logger.info("harvest_frame: table=%r is an engine-internal alias, not a business "
                    "entity — not recording it as the frame's entity", entity)
        entity = None

    return {
        "entity": entity,
        "route": route,
        "entity_is_document": entity_is_document,
        "measures": [m for m in (analytics.get("query_measures") or []) if m],
        # The TABLE's measure columns, as the engine's own semantic model classifies
        # them — not this query's. Kept solely so an "also include profit" follow-up can
        # be GROUNDED against real columns instead of this layer inventing one. Nothing
        # renders it; it is vocabulary, not a fact about the answer.
        "available_measures": [m for m in (analytics.get("available_measures") or []) if m],
        "order_by": order_by,
        "limit": analytics.get("limit"),
        "entity_display": datasets[0] if datasets else None,
        # EVERY document this answer drew on, not just the first. datasets[0] alone
        # cannot answer "was the document we were already discussing actually used
        # here?", which is what keeps a document conversation from wandering —
        # see stabilise_document_entity.
        "datasets": [d for d in datasets if d],
        "understanding": understanding,
        "filters": [
            {"field": f.get("field"), "column": f.get("column"),
             "operator": f.get("operator"), "value": f.get("value"),
             "source": "executed_sql"}
            for f in filters_applied if f.get("field")
        ],
        "group_by": group_by,
        "aggregation": aggregation,
        "last_sql": engine_result.get("sql"),
        "last_row_count": row_count,
    }


def is_topic_switch(frame: Optional[QueryFrame], harvested: Optional[Dict[str, Any]]) -> bool:
    """Deterministic reset detector (§10) — reuses the engine's OWN independent
    table-routing decision (already computed for every query by the 5-signal
    retrieval engine in veda_core/retrieval/, with zero knowledge of the
    current frame) instead of asking an LLM "is this a new topic". No new
    embedding/vector lookup — this is a free byproduct of a call that already
    happened."""
    if not frame or not frame.get("entity") or not harvested or not harvested.get("entity"):
        return False
    return harvested["entity"] != frame["entity"]


def merge_frame_post_execution(
    prev: Optional[QueryFrame], harvested: Dict[str, Any], delta_type: str,
    tenant: str, session_id: str,
) -> QueryFrame:
    """POST-execution write (§6, memory_write_node). `harvested` always wins
    over `prev` for every field it supplies — it is strictly newer, executed,
    validated evidence; `prev` only fills in what harvested doesn't carry
    (e.g. drill_path, which harvest_frame can't derive on its own)."""
    reset = delta_type == "new_topic" or is_topic_switch(prev, harvested) or not prev
    base = empty_frame(tenant, session_id) if reset else dict(prev)  # type: ignore[arg-type]

    merged: QueryFrame = {**base, **harvested}  # type: ignore[typeddict-item]
    merged["tenant"] = tenant
    merged["session_id"] = session_id
    merged["version"] = (base.get("version") or 0) + 1
    merged["turn_index"] = (base.get("turn_index") or 0) + 1
    merged["last_status"] = "answered"
    merged["confidence"] = 1.0
    merged["updated_at"] = _now()
    merged["drill_path"] = [] if reset else list(base.get("drill_path") or [])
    return merged


def push_drill(stack: List[DrillLevel], harvested: Dict[str, Any]) -> List[DrillLevel]:
    """Append the most granular NEW filter as a drill level. Heuristic: the
    LAST filter in the freshly-executed query's filter list (deterministic,
    not LLM-chosen) — good enough for the linear drill-down shape in the
    prompt's own example (Region -> NA -> California -> Los Angeles)."""
    filters = harvested.get("filters") or []
    if not filters:
        return stack
    last = filters[-1]
    if not last.get("field"):
        return stack
    level: DrillLevel = {"dimension": last["field"], "column": last.get("column"),
                         "value": last.get("value")}
    return (stack + [level])[-_MAX_DRILL_DEPTH:]


def newly_added_filter(prev: Optional[Dict[str, Any]],
                       harvested: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The filter THIS turn added relative to `prev`, or None if it added none.

    Deterministic drill detection, and the reason the DrillStack was always empty. The LLM
    delta classifier labels the ordinary value-narrowing follow-ups — "only the open ones",
    "just the Repair ones" — as `refine`, not `drill_down` (measured on this deployment), and
    memory_write_node only pushed a level for `drill_down`. So no breadcrumb was ever recorded
    and "go back" had nothing to pop, even though the user had plainly drilled in.

    Evidence beats the label here, which is the same principle harvest_frame already applies:
    if the query we actually EXECUTED carries a filter the previous one did not, this turn
    narrowed — whatever the classifier chose to call it. Compares (field, operator, value) so
    re-running the same filter is not mistaken for a new level.

    A filter on a field the previous turn ALREADY constrained is a REPLACEMENT, not a new
    level: the drill path is one step deep either way. This used to compare the full
    (field, operator, value) key alone, so swapping a value pushed a second level —
    measured 2026-09-22 on a live chain, "how many assets" -> "only the ones in Pune" ->
    "what about Mumbai" left a stack of depth 2 for what the user sees as one narrowing,
    and "go back" then returned to Pune instead of to all assets. The stack is a path of
    narrowings (see DrillLevel), not an undo history of edits.
    """
    def _key(f):
        return (f.get("field"), f.get("operator"), str(f.get("value")))
    prev_filters = (prev or {}).get("filters") or []
    prev_keys = {_key(f) for f in prev_filters}
    for f in (harvested.get("filters") or []):
        if not f.get("field") or _key(f) in prev_keys:
            continue
        if any(_same_field(f.get("field"), p.get("field")) for p in prev_filters):
            continue        # same field, different value — a replacement
        return f
    return None


def update_drill_level(stack: List[DrillLevel], field: Optional[str],
                       value: Any) -> List[DrillLevel]:
    """Re-point an existing drill level at a new value, leaving the path's DEPTH alone.

    The companion to newly_added_filter's replacement case: "what about Mumbai" changes
    which rows the current level selects, not how far in the user has drilled. Without
    this the level kept the old value, and a later "go back" rebuilt the frame from the
    stack (rebuild_frame_from_stack) and resurrected the filter the user had replaced.
    """
    if not field:
        return stack
    return [{**lvl, "value": value} if _same_field(lvl.get("dimension"), field) else lvl
            for lvl in stack]


def push_drill_level(stack: List[DrillLevel], filt: Dict[str, Any]) -> List[DrillLevel]:
    """Push ONE explicitly-identified filter as a drill level (see newly_added_filter).

    push_drill() below guesses "the last filter" because a `drill_down` turn does not say which
    filter is new; when the caller already KNOWS which one was added, record that instead of
    guessing — with several filters in play the newest is not necessarily last.
    """
    if not filt or not filt.get("field"):
        return stack
    level: DrillLevel = {"dimension": filt["field"], "column": filt.get("column"),
                         "value": filt.get("value")}
    return (stack + [level])[-_MAX_DRILL_DEPTH:]


def pop_drill(stack: List[DrillLevel]) -> List[DrillLevel]:
    """"Go back" — pop one level. No-op (never errors) at the root."""
    return stack[:-1] if stack else stack


def _describe_frame(frame: QueryFrame) -> str:
    """Deterministic text reconstruction from ALREADY-PROVEN facts only —
    never an LLM paraphrase. Filters are copied verbatim from the last
    EXECUTED query's own explain output (business_explain.py, zero LLM).

    Deliberately a compact "entity + filters" noun phrase, NOT the full
    `frame["understanding"]` sentence — that sentence also carries the PRIOR
    turn's own operation verbs ("Find the top 100 ... by ...", sort/limit
    language), and concatenating a full second sentence ahead of the new
    message was observed (live testing) to confuse the engine's join planner
    into treating the repeated entity mention as a second, unnecessary join
    target. A short noun-phrase prefix carries the same grounding (which
    table, which filters already apply) without re-asserting operations the
    new message isn't asking to repeat.

    NOTE (audit C2 fix): this used to take its own `drop_last_filter` flag
    for the drill_up case, applied ON TOP OF the caller (context_resolve_node)
    already popping the drill stack and calling rebuild_frame_from_stack()
    first — a double decrement that silently dropped TWO context levels for
    a single "go back". `frame.filters` is now trusted as-is: whoever calls
    this (render_frame_as_query) is responsible for handing in an
    ALREADY-correct frame (rebuilt first, on drill_up), never for this
    function to second-guess it.
    """
    # Both the business-facing display name AND the raw table name — display
    # name alone (e.g. "Payment Transactions") was observed (2026-07 live
    # testing) to be ambiguous enough that a bare "go back" re-resolved to a
    # DIFFERENT, similarly-named table (reminders_reminderpaymenttransaction
    # instead of accounts_paymenttransaction) — the raw table name is the
    # exact, unambiguous identifier retrieval already indexes on; the display
    # name stays too since it's what makes the phrase read naturally.
    raw_entity, display = frame.get("entity"), frame.get("entity_display")
    if display and raw_entity and display.lower() != raw_entity.lower():
        entity = f"{display} ({raw_entity})"
    else:
        entity = display or raw_entity or ""
    filters = list(frame.get("filters") or [])
    # VALUES, not "Field equals Value". The field name in a frame is the engine's own
    # humanised LABEL from explain.filters.applied ("Location"), and the table has no
    # such column — assets_asset stores it as `city_name`. Echoing the label back made
    # Tier-2 try to ground `assets_asset.location`, fail, and fall through to an LLM-IR
    # plan that joined through users_useraddress and reported a completely different
    # table; harvest_frame then moved the QueryFrame's entity there and every later turn
    # of the conversation asked about the wrong thing.
    #
    # Measured 2026-09-22, same question, everything else identical:
    #   "... Location equals Mumbai"    -> accounts_generalledger, 3 joins   WRONG
    #   "... city_name equals Mumbai"   -> filtered on building_name         WRONG
    #   "... Mumbai"                    -> assets_asset, WHERE city_name     RIGHT
    #   "... " (no filter at all)       -> assets_asset, WHERE city_name     RIGHT
    # Naming the field is what breaks it, and the REAL column name breaks it too — so
    # this is not a label-vs-column bug to be fixed by storing better names. The engine
    # resolves a bare value to its own column correctly; it is the side that knows the
    # schema, and this layer should not pretend to.
    #
    # Non-equality comparisons keep their operator: "greater than 5000" carries meaning
    # a bare "5000" loses, and those are not the shape that broke here.
    filter_phrases = [
        str(f["value"]) if str(f.get("operator") or "equals").lower() == "equals"
        else f"{f.get('operator')} {f['value']}"
        for f in filters if f.get("field") and f.get("value") is not None
    ]
    # ENTITY + FILTERS ONLY. The frame also remembers measures/order_by/limit, and a
    # first cut appended them here as "measuring <col>, ranked by <col> (lowest first),
    # top 100". Live test, 2026-09-17: the engine read the word "measuring" as a data
    # term and spent 54s before asking "Could you clarify if 'measuring' is a column
    # name or a value to filter on?" — the exact failure this function's own note above
    # describes. Operation language belongs where a decision is made about it (the
    # classifier prompt, which now shows the frame's ranking) and NOT in text handed to
    # a pipeline whose job is to parse every word of it as data.
    parts: List[str] = ([entity] if entity else []) + filter_phrases
    return ", ".join(p for p in parts if p)


# The frame's SHAPE slots — what the previous turn measured, grouped and ranked by, as
# opposed to WHICH ROWS it looked at (that is `filters`). A delta may now name one of
# these instead of a filter field: "make it top 10" acts on `limit`, "by month instead"
# on `group_by`, "don't sort by amount" on `order_by`.
#
# Keys are what a model might actually emit for each; the canonical name is the frame
# key. Matched on a normalised form (case, spaces and underscores folded) so "group by",
# "group_by" and "Group By" are one thing. This is a closed set on purpose — anything
# outside it falls through to the filter path exactly as before.
_SHAPE_SLOT_ALIASES = {
    "limit": "limit", "rowlimit": "limit", "top": "limit", "topn": "limit",
    "count": "limit", "howmany": "limit",
    "groupby": "group_by", "grouping": "group_by", "groupedby": "group_by",
    "dimension": "group_by", "breakdown": "group_by",
    "orderby": "order_by", "ordering": "order_by", "sort": "order_by",
    "sortby": "order_by", "sortedby": "order_by", "ranking": "order_by",
    "rankedby": "order_by",
    "measures": "measures", "measure": "measures", "metric": "measures",
    "metrics": "measures", "measuredby": "measures",
}

# Slots holding a LIST, as opposed to `limit`, which holds one number.
_LIST_SHAPE_SLOTS = frozenset({"group_by", "order_by", "measures"})


def _shape_slot_named(field: Optional[str]) -> Optional[str]:
    """Which shape slot this delta_field names, or None if it names none."""
    key = re.sub(r"[^a-z0-9]+", "", str(field or "").lower())
    return _SHAPE_SLOT_ALIASES.get(key)


def _shape_element_field(element) -> str:
    """The comparable name of one element of a list-valued shape slot. `order_by`
    holds {"field": ..., "desc": ...}; `group_by` and `measures` hold bare strings."""
    if isinstance(element, dict):
        return str(element.get("field") or "")
    return str(element or "")


def _apply_shape_delta(frame: QueryFrame, delta_type: str, slot: str,
                       value: Optional[str], message: str) -> QueryFrame:
    """The shape half of apply_context_delta. Same three guards as the filter half, for
    the same reasons: the slot must already hold something (a turn cannot replace a
    ranking the previous turn never had), a replacement value must appear VERBATIM in
    the user's own message, and anything that does not typecheck carries the context
    forward unchanged rather than writing a guess.

    One rule of its own: `replace` acts only when the slot holds exactly ONE value.
    With "group_by: [Year, Region]" and "show it by month instead", nothing in the
    message says WHICH of the two is being swapped, and picking one would be this layer
    guessing. `remove` has no such limit because the user names the thing they are
    dropping ("don't sort by amount"), so the named element can be matched exactly.
    """
    current = frame.get(slot)
    if not current:
        logger.debug("apply_context_delta: %s names slot=%r which the frame does not "
                     "hold — carrying context unchanged", delta_type, slot)
        return frame

    if delta_type == "remove":
        if slot not in _LIST_SHAPE_SLOTS:
            return {**frame, slot: None}
        if not value:
            return {**frame, slot: []}
        # A named element is dropped by name; an unrecognised name drops nothing, which
        # is the same refuse-over-guess posture the filter path takes.
        kept = [e for e in current if not _same_field(_shape_element_field(e), value)]
        if len(kept) == len(current):
            logger.debug("apply_context_delta: remove names %r which slot=%r does not "
                         "hold — carrying context unchanged", value, slot)
            return frame
        return {**frame, slot: [dict(e) if isinstance(e, dict) else e for e in kept]}

    # replace
    if not (value and str(value).lower() in (message or "").lower()):
        logger.debug("apply_context_delta: replace value=%r is not present in the "
                     "message — carrying context unchanged", value)
        return frame

    if slot == "limit":
        digits = re.sub(r"[^0-9]", "", str(value))
        if not digits or int(digits) <= 0:
            logger.debug("apply_context_delta: replace limit=%r carries no usable "
                         "number — carrying context unchanged", value)
            return frame
        return {**frame, "limit": int(digits)}

    if len(current) != 1:
        logger.debug("apply_context_delta: slot=%r holds %d values and the message "
                     "names no one of them — carrying context unchanged",
                     slot, len(current))
        return frame

    if slot == "order_by":
        # Only the field changes. The DIRECTION stays whatever the executed SQL proved
        # last turn: "sort by amount instead" says nothing about ascending/descending,
        # and inventing one here would silently reverse a ranking the user never asked
        # to reverse.
        previous = current[0] if isinstance(current[0], dict) else {}
        return {**frame, "order_by": [{"field": str(value),
                                       "desc": bool(previous.get("desc", True))}]}

    return {**frame, slot: [str(value)]}


# ── deterministic shape-delta detection ──────────────────────────────────────
#
# WHY THIS IS NOT A MODEL CALL. classify_node merges the action and delta decisions
# into ONE SLM round-trip (chatbot/prompts/supervisor.py), and that merged call was
# measured on 2026-09-18 to collapse every shape change onto "refine":
#
#   frame limit 100      "make it top 10"        -> refine, delta_field "limit"
#   frame group_by year  "show it by month instead" -> refine, delta_field "group_by"
#   frame order_by amount "don't sort by amount"  -> refine, delta_field "order_by"
#
# It identifies the SLOT correctly every time and the OPERATION never. That is fatal
# on its own, because parse_delta_response drops delta_field for any type outside
# DELTA_TYPES_WITH_FIELD — so on a "refine" the slot name is discarded and no signal
# survives at all. The focused standalone prompt gets all four right, but reaching it
# costs a second SLM call on turns that are already the common case.
#
# Teaching the merged prompt did not work either: adding shape rules and examples to
# it left the refine bias exactly as it was AND regressed an ordinary filter swap
# ("what about Mumbai" went from replace/City to refine), so those edits were reverted.
# A 7B asked to do two classifications in one pass has a budget, and this was over it.
#
# So the operation is decided here instead, from the message's own shape. This is the
# same deterministic-fast-path pattern the conversation layer already uses for reset,
# drill-up, recall, presentation and runtime-context (chatbot/nodes.py) — and it is
# strictly safer than those, because a match still has to survive every guard in
# apply_context_delta: the slot must already hold a value, and the new value must be a
# literal the user typed. It costs no model call, cannot drift between runs, and
# behaves identically on a larger model.
#
# Every pattern is ANCHORED TO THE WHOLE MESSAGE. That is what separates "top 10" (a
# re-ranking of the current question) from "top 5 cities by assets" (a new question
# that happens to start the same way) — measured across all 299 real user messages in
# evaluation/conversation/corpus_real.jsonl, where it fires zero times.
_SHAPE_PATTERNS = (
    # (delta_type, slot, explicit, compiled pattern) — group 1, when present, is the new
    # value. `explicit` means the message names an OPERATION, not just a value: "make it
    # top 5" versus a bare "top 5". Only an explicit one may override a pending
    # clarification, because a bare value is exactly what a clarifying question asks for
    # ("which city?" -> "Pune") and stealing it would break the round-trip.
    ("replace", "limit", True, re.compile(
        r"^(?:make\s+it|just|only|show\s+me|show|give\s+me|i\s+want)\s+"
        r"(?:the\s+)?(?:top|first|bottom)\s+(\d{1,6})"
        r"(?:\s+(?:rows?|entries|results|records))?\s*[.!?]*$", re.IGNORECASE)),
    ("replace", "limit", False, re.compile(
        r"^(?:the\s+)?(?:top|first|bottom)\s+(\d{1,6})"
        r"(?:\s+(?:rows?|entries|results|records))?\s*[.!?]*$", re.IGNORECASE)),
    ("remove", "limit", True, re.compile(
        r"^(?:(?:show|list|give)\s+)?(?:me\s+)?all"
        r"(?:\s+of\s+them|\s+rows|\s+results|\s+records|\s+of\s+it)?\s*[.!?]*$"
        r"|^(?:remove|drop|without)\s+the\s+limit\s*[.!?]*$", re.IGNORECASE)),
    ("replace", "group_by", True, re.compile(
        r"^(?:show\s+|group\s+|break\s*(?:it\s+)?down\s+)?(?:it\s+)?"
        r"by\s+([\w-]+(?:\s+[\w-]+)?)\s+instead\s*[.!?]*$", re.IGNORECASE)),
    ("replace", "order_by", True, re.compile(
        r"^(?:sort|order|rank)\s+(?:it\s+)?by\s+([\w-]+(?:\s+[\w-]+)?)"
        r"\s+instead\s*[.!?]*$", re.IGNORECASE)),
    ("remove", "order_by", True, re.compile(
        r"^(?:do\s*n'?t|do\s+not|stop|no\s+longer)\s+(?:sort|order|rank)(?:ing)?"
        r"\s+by\s+([\w-]+(?:\s+[\w-]+)?)?\s*[.!?]*$", re.IGNORECASE)),
)


def matches_shape_phrase(message: str) -> bool:
    """Does this message ASK for a different shape, regardless of whether any frame
    could satisfy it? `detect_shape_delta` answers the narrower question — "and is
    there a slot to change?" — and returns None when there is not, which is correct
    for the merge but leaves the caller unable to tell "no such request" from "no slot
    to apply it to". Those need different answers on a document conversation, where
    there are never any slots at all."""
    text = (message or "").strip()
    if not text:
        return False
    return any(pattern.match(text) for _, _, _, pattern in _SHAPE_PATTERNS)


def detect_shape_delta(frame: Optional[QueryFrame], message: str,
                       explicit_only: bool = False):
    """-> (delta_type, slot, value) for a message that changes the SHAPE of the
    question the frame already holds, or None.

    `explicit_only` restricts this to messages that name an OPERATION rather than just a
    value — "make it top 5", not a bare "top 5". classify_node uses it to decide whether a
    message outranks a PENDING CLARIFICATION: a clarifying question asks for a bare value,
    so stealing one would break the round-trip, while "by month instead" cannot be an
    answer to any clarifying question the engine asks.

    Only ever returns a slot the frame ACTUALLY HOLDS, so "make it top 10" after a
    question that had no row limit is left alone rather than invented into one — the
    user's own words still reach the engine either way, which is the recoverable
    outcome. The caller decides when to ask (chatbot/nodes.py::context_resolve_node
    asks only when the model did not already produce a usable replace/remove, and
    never when the model called the turn a new topic — correcting a known operation
    bias is not the same as overruling a topic decision).
    """
    if not frame or not (message or "").strip():
        return None
    text = message.strip()
    for delta_type, slot, explicit, pattern in _SHAPE_PATTERNS:
        if explicit_only and not explicit:
            continue
        match = pattern.match(text)
        if not match:
            continue
        if not frame.get(slot):
            logger.debug("detect_shape_delta: %r names slot=%r which the frame does "
                         "not hold — no delta", text, slot)
            continue
        value = (match.group(1) or "").strip() if match.re.groups else ""
        return delta_type, slot, value
    return None


# How English names a SET of rows, not a value in one. "ones", "records", "rows" are
# never data: no column holds the value "ones". A replacement value ending in one of
# these is the model having read an ADDITIVE phrase as a swap.
#
# Measured 2026-09-21, an 18-turn conversation. Frame held City=Pune and Status=ACTIVE;
# the user said "just the verified ones" — a NEW dimension. The model returned
# replace/Status/"verified ones", so ACTIVE was overwritten with a value no column has,
# and the user's "active" constraint vanished with no signal. Both existing guards
# passed it: Status IS a field the frame holds, and "verified ones" IS verbatim in the
# message. Nothing else could have caught it.
#
# The asymmetry is the whole argument. A wrongly-declined replace costs nothing — the
# frame carries forward and the user's own words still reach the engine, which then
# sees "verified" alongside Pune and ACTIVE and can resolve all three. A wrongly-
# ACCEPTED replace destroys a remembered fact silently. When the two are not equally
# likely, prefer the recoverable one.
_COLLECTIVE_NOUNS = frozenset({
    "one", "ones", "row", "rows", "record", "records", "item", "items",
    "entry", "entries", "value", "values", "result", "results",
})


def _names_a_set_not_a_value(value: Optional[str]) -> bool:
    """Does this replacement value END in a word that names a collection?"""
    words = re.split(r"[^a-z0-9]+", str(value or "").lower())
    words = [w for w in words if w]
    return bool(words) and words[-1] in _COLLECTIVE_NOUNS


# Adding a MEASURE is not adding a filter, and the closed delta set has no "add": the
# model is never asked for one. Detected here from the user's own words, the same way
# detect_shape_delta handles "make it top 10" — deterministic, so a 7B's label cannot
# put a column into memory.
def detect_measure_addition(frame: Optional[QueryFrame], message: str) -> Optional[str]:
    """The measure this message asks to ADD to what is already being measured.

    Returns the real column name, or None. Two conditions:

      1. the message names something the TABLE offers as a measure
         (`available_measures`, from the engine's own semantic model), and
      2. that measure is not already in the frame.

    WHAT IS DELIBERATELY *NOT* HERE: a list of addition phrases. The first version
    required the message to match "also include|add|along with|as well|too" before it
    would look at all. That was redundant and fragile at once — the caller already gates
    on delta_type == "refine", whose definition in the prompt IS addition ("ADDS a
    filter/grouping, keeps everything in the frame"), while "what about profit" is
    classified `replace` and never reaches here. So the model already draws the
    add-vs-replace line, and a second English word-list could only disagree with it, and
    would need a new phrasing forever ("as well" was missed on the first pass).

    The split that matters: DETECTION is the model's job, GROUNDING is this function's.
    Condition 1 is the deterministic half — the user's words are matched against real
    columns, and a term that grounds against nothing returns None. This layer never
    invents a column name; that is the failure mode that sent "Location equals Mumbai"
    to the engine and relocated a whole conversation onto another table.

    SCOPE: this updates MEMORY, not the request. Measures are deliberately not rendered
    into the resolved query (see _describe_frame), so the engine learns about "profit"
    from the user's own message, passed through verbatim. What the frame gains is knowing
    the NEXT turn's "make it top 10" applies to a two-measure question.
    """
    if not frame or not message:
        return None
    available = [m for m in (frame.get("available_measures") or []) if m]
    if not available:
        return None
    current = {str(m).strip().lower() for m in (frame.get("measures") or [])}
    words = _content_words_for_measures(message)
    if not words:
        return None
    for col in available:
        if str(col).strip().lower() in current:
            continue                      # already measured — nothing to add
        col_words = _content_words_for_measures(col)
        if col_words and col_words <= words:
            return col
    return None


def _content_words_for_measures(text) -> set:
    """Words of a message or a column name, folded to a comparable form.

    Deliberately the same crude folding the rest of this module uses (split on
    separators and camelCase, drop tokens under three characters, fold a trailing "s"):
    a real stemmer would also fold unrelated words together, and here that would mean
    binding the user's words to a column they did not name.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    out = set()
    for token in re.split(r"[^A-Za-z0-9]+", spaced.lower()):
        if len(token) < 3:
            continue
        out.add(token)
        out.add(token[:-1] if token.endswith("s") and len(token) > 3 else token + "s")
    return out


def add_measure(frame: QueryFrame, column: str) -> QueryFrame:
    """Append one grounded measure. Returns a NEW frame; touches nothing else.

    Same discipline as apply_context_delta: never mutates in place, and returns the
    caller's own object when there is nothing to do, so identity stays an exact
    "did anything change" signal for the caller.
    """
    if not column:
        return frame
    current = list(frame.get("measures") or [])
    if any(str(m).strip().lower() == str(column).strip().lower() for m in current):
        return frame
    return {**frame, "measures": current + [column]}


class ComparisonSide(TypedDict, total=False):
    """One half of a comparison — the facts needed to resolve a follow-up against it.

    NOT a second QueryFrame. A frame also carries bookkeeping (version, tenant,
    session_id, updated_at, last_sql, drill_path, base_query) that means nothing for a
    comparand, and duplicating it would leave two things to keep in step. This is the
    subset that answers "which rows, measured how" — composition, not a copy.
    """
    entity: Optional[str]
    entity_display: Optional[str]
    source_id: Optional[int]
    filters: List[FilterFact]
    measures: List[str]
    group_by: List[str]
    order_by: List[OrderFact]
    limit: Optional[int]
    label: str              # what identifies this side to a reader: "2025", "homzhub"


class ComparisonContext(TypedDict, total=False):
    """Two sides plus what varies between them.

    Stored at SESSION level, not under a source key: a comparison can span sources, so
    filing it under one of them would be the same mistake the single session-wide frame
    made before source scoping — one side silently owning the other.

    `dimension` records WHAT is being compared, because the follow-up resolution differs:
      "value"  — same entity and source, different filter value  ("2025" vs "2024")
      "source" — same question, different source                 (A vs B)
      "entity" — different entities
    It is derived from the two sides, never asked of a model.
    """
    primary: ComparisonSide
    comparison: ComparisonSide
    dimension: str
    created_at: str
    turn_index: int


def _comparison_side(frame: Optional[QueryFrame], label: str = "") -> ComparisonSide:
    """Project a frame down to a comparand. Pure; the frame is not modified."""
    f = frame or {}
    return {
        "entity": f.get("entity"),
        "entity_display": f.get("entity_display"),
        "source_id": f.get("source_id"),
        "filters": [dict(x) for x in (f.get("filters") or [])],
        "measures": list(f.get("measures") or []),
        "group_by": list(f.get("group_by") or []),
        "order_by": [dict(x) for x in (f.get("order_by") or [])],
        "limit": f.get("limit"),
        "label": label or _side_label(f),
    }


def _side_label(frame: Optional[QueryFrame]) -> str:
    """A reader-facing name for a side, taken from what actually distinguishes it.

    Filter VALUES first (that is what a "2025 vs 2024" comparison varies), then the
    entity. Never a field name — see _describe_frame for why naming fields is unsafe.
    """
    f = frame or {}
    values = [str(x.get("value")) for x in (f.get("filters") or [])
              if x.get("value") is not None]
    if values:
        return ", ".join(values)
    return str(f.get("entity_display") or f.get("entity") or "")


def _comparison_dimension(primary: ComparisonSide, comparison: ComparisonSide) -> str:
    """What varies between the two sides — derived from the sides themselves."""
    if primary.get("source_id") != comparison.get("source_id"):
        return "source"
    if (primary.get("entity") or "") != (comparison.get("entity") or ""):
        return "entity"
    return "value"


def build_comparison(primary_frame: Optional[QueryFrame],
                     comparison_frame: Optional[QueryFrame],
                     turn_index: int = 0) -> Optional[ComparisonContext]:
    """Assemble a comparison from the two frames that produced its sides.

    Returns None when either side has no entity — a comparand with nothing identifying
    it cannot ground a follow-up, and a half-built comparison is worse than none: the
    next turn would resolve against a side that names nothing.
    """
    if not (primary_frame or {}).get("entity") or not (comparison_frame or {}).get("entity"):
        return None
    primary = _comparison_side(primary_frame)
    comparison = _comparison_side(comparison_frame)
    return {
        "primary": primary,
        "comparison": comparison,
        "dimension": _comparison_dimension(primary, comparison),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "turn_index": turn_index,
    }


def render_comparison_as_query(comparison: Optional[ComparisonContext],
                               message: str) -> str:
    """PRE-execution text for a follow-up while a comparison is active.

    Carries BOTH sides so the follow-up cannot collapse into one of them — that collapse
    is the specific failure this structure exists to prevent.

    Only VALUES and business entity names are emitted, never field names. That rule is
    measured, not stylistic: rendering the engine's own column LABEL ("Location equals
    Mumbai") made Tier-2 try to ground a column that does not exist, fall back to an
    LLM-IR plan, and answer from an unrelated table. A bare value resolved correctly.
    """
    if not comparison:
        return message
    a = (comparison.get("primary") or {}).get("label") or ""
    b = (comparison.get("comparison") or {}).get("label") or ""
    if not a or not b:
        return message
    entity = ((comparison.get("primary") or {}).get("entity_display")
              or (comparison.get("primary") or {}).get("entity") or "")
    subject = f"{entity}, " if entity else ""
    return f"{message} (for {subject}comparing {a} with {b})".strip()


def comparison_is_stale(comparison: Optional[ComparisonContext],
                        frame: Optional[QueryFrame]) -> bool:
    """Has the conversation moved off what this comparison was about?

    Evidence-based, same rule as is_topic_switch: if the answered entity no longer
    matches either side, the comparison describes a question nobody is asking. Kept
    conservative — an unrelated new topic drops it, a refinement of either side does not.
    """
    if not comparison:
        return False
    entity = (frame or {}).get("entity")
    if not entity:
        return False
    sides = {(comparison.get("primary") or {}).get("entity"),
             (comparison.get("comparison") or {}).get("entity")}
    return entity not in sides


# The word after a comparison preposition — "compare that with Mumbai" -> "Mumbai",
# "compare with 2024" -> "2024", "compare 2025 vs 2024" -> "2024". Anchored to the END
# of the message: the comparand is what the user is asking to see ALONGSIDE the current
# context, and it is normally the last thing named.
_COMPARISON_TARGET_RE = re.compile(
    r"\b(?:with|against|versus|vs\.?|to)\s+(.+?)\s*[.,!?]*$", re.IGNORECASE)


def detect_comparison_target(message: str) -> Optional[str]:
    """The comparand, extracted VERBATIM from the message — never invented.

    `delta_type == "compare"` is the model's job, and it is reliable: it is a
    closed-set classification with clear examples. WHICH VALUE is being compared to is
    a different question, and the model's own `slot_candidates` answer for it turned out
    NOT to be reliable — measured 2026-09-22, live: three separate turns of
    "compare that with Mumbai" all classified correctly as `compare` and all three came
    back with an empty slot list, so nothing populated `delta_value` and the render
    fell through to sending the raw message (including the word "compare" itself) to the
    engine, which then asked whether "compare" was a column name.

    So this is the same choice already made for shape ops and measure additions:
    detection stays the model's job, extraction becomes deterministic Python. The
    returned text is a literal substring of `message` — nothing here can invent a value
    that was not typed.
    """
    if not message:
        return None
    m = _COMPARISON_TARGET_RE.search(message)
    if not m:
        return None
    target = m.group(1).strip()
    return target or None


def apply_context_delta(frame: Optional[QueryFrame], delta_type: str,
                       field: Optional[str] = None, value: Optional[str] = None,
                       message: str = "") -> QueryFrame:
    """Apply ONE structured context operation to the frame, in Python. Returns a new
    frame; never mutates the one passed in.

    This is the half of the design the model is deliberately kept out of. The model's
    entire job upstream is to emit {delta_type, field, value}; deciding WHICH remembered
    fact that displaces, and removing it, is arithmetic on a dict — a 7B model asked to
    restate the whole context correctly every turn will eventually not, whereas this
    cannot drift.

    Two guards, both structural, so a wrong or invented classification degrades to "carry
    the context unchanged" rather than to a wrong answer:

    · `field` must already exist in the frame. A field the previous turn never filtered
      on is not something this turn can replace or remove; the delta is dropped. `field`
      may name either a remembered FILTER or one of the frame's SHAPE slots — `limit`,
      `group_by`, `order_by`, `measures` — so "make it top 10" and "by month instead"
      are structured operations rather than words the engine has to re-derive. A filter
      of the same name always wins, keeping this additive. See _apply_shape_delta.
    · `value` must appear VERBATIM in the user's own message. The same vocabulary gate
      chatbot/memory/classify.py already applies to slot_candidates — memory may narrow
      what the engine is told, never invent it.

    TEMPORAL REPLACEMENT is the one case where a literal value is not required. "What
    about last year?" carries no year to copy, so the old temporal filter is REMOVED and
    the user's own words are left to the engine's existing L1 temporal parser
    (veda/pipeline.py), which already resolves relative periods. Removing beats guessing:
    the alternative is this layer computing a date, which is exactly the kind of derived
    fact the frame is built never to hold.
    """
    if not frame:
        return frame or {}
    if delta_type not in ("replace", "remove"):
        return frame                            # add/refine/drill/compare are unchanged
    filters = list(frame.get("filters") or [])

    target = next((f for f in filters if _same_field(f.get("field"), field)), None)

    # A shape slot is considered only when no real filter answers to the same name, so a
    # source that genuinely has a column called "Limit" or "Ranking" keeps the behaviour
    # it has always had and this stays purely additive.
    slot = _shape_slot_named(field)
    if slot and target is None:
        return _apply_shape_delta(frame, delta_type, slot, value, message)

    if not filters:
        return frame
    if target is None:
        logger.debug("apply_context_delta: %s names field=%r which the frame does not "
                     "hold — carrying context unchanged", delta_type, field)
        return frame

    # Every returned filter is a COPY, including the ones this delta does not touch.
    # Returning the caller's own dicts made the "new" frame share them with the old one,
    # so a later edit to either reached back into the other — action at a distance that
    # no test of a single call can see. Found by an adversarial sweep, 2026-09-17.
    if delta_type == "remove":
        return {**frame, "filters": [dict(f) for f in filters if f is not target]}

    # replace. Ungrounded values never reach the frame at all — carrying the previous
    # context unchanged is always recoverable; writing a value the user never typed is
    # not, and the engine still sees the user's own words either way.
    if not (value and str(value).lower() in (message or "").lower()):
        logger.debug("apply_context_delta: replace value=%r is not present in the "
                     "message — carrying context unchanged", value)
        return frame

    if _names_a_set_not_a_value(value):
        # See _COLLECTIVE_NOUNS. Declining leaves the frame intact and lets the user's
        # words reach the engine unchanged, which is the recoverable outcome.
        logger.info("apply_context_delta: replace value=%r names a SET, not a value — "
                    "declining the swap and carrying %r forward", value,
                    target.get("field"))
        return frame

    if is_temporal_field(target.get("field")) and not any(ch.isdigit() for ch in str(value)):
        # A RELATIVE period — "last year", "previous quarter". There is no literal to
        # substitute, and resolving one here would mean this layer computing a date: a
        # derived fact, which the frame is built never to hold. Drop the stale filter
        # instead and leave the user's own words to the engine's L1 temporal parser,
        # which already resolves exactly these phrases (veda/pipeline.py).
        return {**frame, "filters": [dict(f) for f in filters if f is not target]}

    return {**frame, "filters": [{**f, "value": value} if f is target else dict(f)
                                 for f in filters]}


_DOCUMENT_ROUTES = frozenset({"rag", "doc", "document"})

# Extensions carry no subject matter — "policy.docx" and "policy.pdf" name the same
# document as far as "did the user already mention it" is concerned.
_FILE_EXTENSION_WORDS = frozenset({"pdf", "docx", "doc", "txt", "csv", "xlsx", "pptx",
                                   "md", "html", "htm", "rtf"})


def is_document_frame(frame: Optional[QueryFrame]) -> bool:
    """Was this frame built from documents rather than from an executed query?

    Prefers the fact harvest_frame records (`entity_is_document`) over the route name.
    The route is still consulted, both for frames written before that fact existed —
    Redis holds them for 7 days — and because a pure retrieval head names itself
    honestly. The route alone was NOT enough: the hybrid head answers from documents
    under the name "hybrid" (see harvest_frame), and reading only the route silently
    excluded every answer it produced.
    """
    if not frame:
        return False
    if frame.get("entity_is_document") is not None:
        return bool(frame["entity_is_document"])
    return str(frame.get("route") or "").lower() in _DOCUMENT_ROUTES


def _already_names(message: str, document: str) -> bool:
    """Does the message itself already point at this document?

    Compared on content words, not the raw string: the frame's entity is a file name
    ("maintenance_policy.docx", "Samta-Employee Handbook April_2026.pdf") while the user
    writes prose ("the maintenance policy"), so an exact substring test would almost
    never fire. A document is considered named when every distinctive word of its name
    appears in the message.
    """
    doc_words = {w for w in re.findall(r"[a-z0-9]+", document.lower())
                 if len(w) > 2 and w not in _FILE_EXTENSION_WORDS}
    if not doc_words:
        return False
    msg_words = set(re.findall(r"[a-z0-9]+", message.lower()))
    return doc_words <= msg_words


def _normalise_document_name(name: str) -> str:
    """Compare document names by their letters and digits alone.

    The same document reaches this layer spelled several ways — "maintenance policy",
    "maintenance_policy.docx", "Samta-Employee Handbook April 2026" vs
    "Samta-Employee Handbook April_2026.pdf" — so an exact string match would report a
    document as absent from an answer that cited it.
    """
    stem = re.sub(r"\.(%s)$" % "|".join(sorted(_FILE_EXTENSION_WORDS)), "",
                  name.strip().lower())
    return re.sub(r"[^a-z0-9]+", "", stem)


def keep_entity_on_lane_change(prev_frame: Optional[QueryFrame], harvested: Dict[str, Any],
                              referential: bool) -> Dict[str, Any]:
    """Do not let a FOLLOW-UP move the conversation to an entity from a different LANE.

    The mirror of stabilise_document_entity, for the case that one does not cover: a
    conversation anchored on a TABLE, whose follow-up gets answered from the documents.

    Measured 2026-09-23/24. After "What is the distribution of properties by furnishing?"
    (frame: assets_asset, route deterministic), the follow-up "only the Nagpur ones" was
    answered by the RAG head — "The provided context does not contain information specific
    to Nagpur", citing an employee handbook. That turn's status is `answered`, so memory
    wrote it, and the frame became `Samta-Employee Handbook April 2026`. Every later turn
    in that conversation was then anchored to a document nobody had asked about.

    The write itself was correct by its own rules — it recorded what the engine returned.
    What was wrong is ADOPTING it as the conversation's subject: a follow-up is, by
    definition, a continuation of what came before, so an answer from a different lane is
    evidence that the turn went astray, not that the topic changed.

    So the ENTITY is held: the rest of the harvest (row count, sql, status) is written
    unchanged, because those are facts about what just ran. Only the claim "this is what
    the conversation is about now" is refused.

    Deliberately narrow:
      · follow-ups only — a NEW topic is allowed to move anywhere, which is what makes it
        a new topic.
      · route change only — a follow-up answered by the SAME lane is the normal case and
        is untouched, however much the table moved within it.
      · no frame, or no route recorded on either side -> nothing to compare, nothing to do.
    """
    if not referential or not prev_frame or not harvested:
        return harvested
    prev_route = str(prev_frame.get("route") or "").strip().lower()
    new_route = str(harvested.get("route") or "").strip().lower()
    if not prev_route or not new_route or prev_route == new_route:
        return harvested
    # Only the SQL-to-document direction. The reverse (a document conversation that finds
    # a table) is already handled by stabilise_document_entity's own evidence rule, and
    # re-deciding it here could fight with it.
    if not (prev_route not in _DOCUMENT_ROUTES and new_route in _DOCUMENT_ROUTES):
        return harvested
    if not prev_frame.get("entity"):
        return harvested
    out = dict(harvested)
    out["entity"] = prev_frame.get("entity")
    out["entity_display"] = prev_frame.get("entity_display")
    out["route"] = prev_frame.get("route")
    if prev_frame.get("source_id") is not None:
        out["source_id"] = prev_frame.get("source_id")
    logger.info("keep_entity_on_lane_change: follow-up answered via %r but the conversation "
                "is on %r — keeping entity %r instead of adopting %r",
                new_route, prev_route, prev_frame.get("entity"), harvested.get("entity"))
    return out


def stabilise_document_entity(prev_frame: Optional[QueryFrame], harvested: Dict[str, Any],
                              referential: bool) -> Dict[str, Any]:
    """Keep a document conversation in its document when the follow-up stayed there.

    The frame's entity is taken from `datasets[0]`, and on a follow-up that ordering is
    not a statement about what the answer was ABOUT. Measured 2026-09-22: after an
    answer from maintenance_policy.docx, the follow-up "and what about response times"
    was answered — correctly, and citing maintenance_policy.docx — but the engine listed
    the employee handbook first, so the frame moved to the handbook and every later turn
    in that conversation was anchored to the wrong document.

    The correction is evidence-only, the same discipline as the rest of this module: the
    previous document is kept ONLY if this answer actually used it, which is exactly
    what `datasets` records. It is not carried over a document that was not consulted.

    Gated on `referential` so it cannot hold a genuinely new question in the old
    document merely because that document is cited again — a real risk here, where one
    file holds 92% of the corpus and is cited constantly.
    """
    if not referential:
        return harvested
    if not is_document_frame(prev_frame) or not is_document_frame(harvested):
        return harvested
    previous = str((prev_frame or {}).get("entity") or "").strip()
    if not previous or previous == str(harvested.get("entity") or "").strip():
        return harvested
    wanted = _normalise_document_name(previous)
    if not wanted or not any(_normalise_document_name(str(d)) == wanted
                             for d in (harvested.get("datasets") or [])):
        # The document we were discussing was not among the ones this answer read, so
        # the conversation genuinely moved. Let the new entity stand.
        return harvested
    logger.info("stabilise_document_entity: follow-up still drew on %r — keeping it as "
                "the frame's document instead of %r (datasets order is not aboutness)",
                previous, harvested.get("entity"))
    return {**harvested, "entity": previous,
            "entity_display": (prev_frame or {}).get("entity_display") or previous}


# The fields that say WHAT the conversation is about, as opposed to what this turn ran.
_SUBJECT_FIELDS = ("entity", "entity_display", "route", "source_id", "entity_is_document")


def keep_entity_on_lane_change(prev_frame: Optional[QueryFrame], harvested: Dict[str, Any],
                               referential: bool) -> Dict[str, Any]:
    """Keep a table conversation on its table when a follow-up came back from documents.

    The mirror of stabilise_document_entity. Measured 2026-09-23: a conversation on
    `assets_asset` asked "only the Nagpur ones", the RAG head answered from the employee
    handbook, that turn counted as `answered`, and the frame moved to the handbook —
    anchoring every later turn to a document nobody had asked about.

    Only the subject is held. What the turn actually ran (row count, SQL, filters) is a
    fact and is still recorded. Gated on `referential` so a genuinely new question may go
    anywhere; a document conversation that finds a table is left to the other guard; and
    with no route on either side there is no lane change to judge.
    """
    if not referential or not harvested:
        return harvested
    prev = prev_frame or {}
    if not str(prev.get("entity") or "").strip():
        return harvested
    if not prev.get("route") or not harvested.get("route"):
        return harvested
    if is_document_frame(prev) or not is_document_frame(harvested):
        return harvested
    logger.info("keep_entity_on_lane_change: follow-up on %r was answered from documents "
                "(%r) — keeping the table as the frame's subject",
                prev.get("entity"), harvested.get("entity"))
    out = {k: v for k, v in harvested.items() if k not in _SUBJECT_FIELDS}
    out.update({k: prev[k] for k in _SUBJECT_FIELDS if k in prev})
    return out


def _render_document_query(frame: QueryFrame, message: str, delta_type: str,
                           shape_delta: bool, referential: bool) -> str:
    """Name the document the conversation is already in, when the turn is a follow-up.

    Anchoring here REVERSES an earlier decision of mine, and the reversal is measured.
    I first returned the message untouched, reasoning that the entity is a FILE NAME and
    that appending "Samta-Employee Handbook April 2026" would hand the pipeline "April
    2026" as a date the user never asked for. Both halves of that were wrong:

      · the date fear does not apply. That is the SQL planner's failure mode; a
        retrieval pipeline reads extra words as retrieval signal. Measured 2026-09-22
        with the dated name appended: no date filter, citations unchanged, 67s -> 26s,
        and the answer went from "the context does not specify" to "Yes, it is paid".

      · "the RAG path resolves follow-ups fine on its own" held only INSIDE one
        document. This corpus is 92% one file (the handbook holds 165 of 179 chunks),
        so an unconstrained follow-up drifts there on volume alone. Measured: after an
        answer from maintenance_policy.docx, "and what about response times" came back
        from the HANDBOOK, about absence time tracking — a different document, silently,
        and the frame followed it there.

    WHY `referential` AND NOT `delta_type`. Every other route in this function keys off
    delta_type; on a document frame it cannot, because it is not informative there.
    Measured over 6 live document turns: delta_type was "new_topic" or "ambiguous" on
    ALL of them — continuations and genuinely new questions alike. That is not a model
    defect so much as a category error: the delta block reasons about filters and slots,
    and a document frame has neither, so "no filter changed" reads as "new topic".

    classify_node's own action label is the usable signal, and it is already computed —
    no extra model call. Measured on the same turns: 2/2 genuinely new questions were
    action="answer" (so they are NOT anchored, which is the outcome that matters), and
    2/3 continuations were action="followup". The third was missed, and a miss is
    simply today's unanchored behaviour — this errs toward under-anchoring, never
    toward trapping a new question in the previous document.
    """
    doc = str(frame.get("entity") or "").strip()
    if not doc:
        return message
    if shape_delta or delta_type in ("drill_up", "remove"):
        # Contentless navigation triggers, and shape ops, name no subject — "go back
        # (in maintenance_policy.docx)" is not a question. A document frame has no
        # drill stack and no shape slots anyway; classify_node settles these from
        # memory without the engine. Left here in case one still reaches this path.
        return message
    if not referential and delta_type in ("new_topic", "ambiguous"):
        return message
    if _already_names(message, doc):
        # "what does the MAINTENANCE POLICY say about response times" anchors itself;
        # appending "(in maintenance policy)" only says it twice.
        return message
    return f"{message} (in {doc})".strip()


def render_frame_as_query(frame: Optional[QueryFrame], message: str, delta_type: str,
                          shape_delta: bool = False, referential: bool = False) -> str:
    """NO LONGER THE MAIN EXECUTION CONTRACT. Two narrow callers remain.

    This built the resolved_query by combining the frame's facts with the user's message.
    That is what injected the engine's own display label for the table into the user's
    words — "only the debit ones (for Single Financial Transactions
    (accounts_generalledger))" — and `veda/validation.py::qualifier_completeness`, whose
    contract is "every content token THE USER NAMED", then refused on `financial`. The
    ordinary follow-up path now sends the user's words unchanged and carries the state
    structurally instead (chatbot/memory/context.py::ConversationContext).

    The two callers that remain, both in context_resolve_node, are deliberate:

      · DOCUMENT frames — a RAG answer has no table, no qualifier gate and no structured
        hint channel, so _render_document_query's measured anchoring still applies.
      · COMPARISON turns — two sides have to be expressed somehow, and the engine has no
        two-sided context to receive them in.

    Do not reintroduce this for relational follow-ups. If the two cases above gain
    structured support, this function and its helpers should go with them. The contract is
    pinned by tests/test_resolved_query_contract.py.

    PRE-execution: combines the frame's own previously-proven facts with the user's new
    message VERBATIM. The engine still independently re-validates everything from scratch
    (L6a-L6c) — this only gives it a better-grounded input, never a shortcut around that
    validation.

    For "drill_up", the caller MUST pass an already-rebuilt frame (see
    chatbot/nodes.py::context_resolve_node — pop_drill() +
    rebuild_frame_from_stack() BEFORE calling this) — this function no
    longer drops a filter of its own (audit C2 fix: that was a double-pop).

    "drill_up" is the one delta_type whose `message` ("go back", "go back
    again", ...) carries NO data content of its own — it's a pure navigation
    trigger, not a fragment to combine with the frame. Gluing it onto `ctx`
    (e.g. "go back (for accounts_paymenttransaction, ...)")  sent the literal
    words "back"/"again" to the engine as if they were part of the question,
    which then tried (and failed) to match them against columns/values. The
    resolved query for drill_up is just the popped frame's own restatement —
    the user's exact words never carried the intent, the stack pop already
    did.
    """
    if not frame or not frame.get("entity"):
        return message
    if is_document_frame(frame):
        return _render_document_query(frame, message, delta_type, shape_delta, referential)
    if delta_type in ("new_topic", "ambiguous"):
        return message
    ctx = _describe_frame(frame)
    if not ctx:
        return message
    if delta_type == "remove" and not shape_delta and not (frame.get("filters") or []):
        # Removing the LAST filter leaves only a table name — "Assets (assets_asset)" as
        # the entire question, with the measure, the verb and the user's intent gone.
        # Keep their words alongside it: the removal already happened in the frame, and
        # the engine still needs to be told what to do with what is left.
        return f"{message} (for {ctx})".strip()
    if delta_type == "drill_up" and not shape_delta and not (frame.get("filters") or []):
        # Popped the LAST level: the conversation is back at the question it started
        # from, and the frame's bare restatement ("Assets (assets_asset)") is a table
        # name, not a question. Measured 2026-09-22 across four scenarios, the engine
        # refused it every time — "I couldn't apply the condition you asked for" — so a
        # user who drilled once and said "go back" always got a refusal.
        #
        # Candidate renderings were measured rather than chosen: "all <entity>" and
        # "show <entity>" were refused too, and "list <entity>" ANSWERED but ran a
        # different question (DISTINCT project_name), which is the dangerous kind of
        # wrong. The user's own pre-drill question is the only one that returned the
        # pre-drill answer, so that is what is replayed — their words, recorded when
        # they asked, not a sentence this layer invented.
        base = (frame.get("base_query") or "").strip()
        if base:
            return base
    if delta_type in ("drill_up", "remove") and not shape_delta:
        # These two carry NO data content of their own — they are pure navigation. Their
        # words name what to STOP doing, and the engine parses every word of the resolved
        # query as data: "Remove India (for Revenue, CustomerType equals Enterprise)"
        # hands it the word "India" it was just told to drop, which is exactly backwards.
        # The frame has already been mutated by apply_context_delta, so the restated
        # context IS the new question. Same reasoning the drill_up case has carried since
        # 2026-07, now that "remove" joins it as a second contentless trigger.
        return ctx
    return f"{message} (for {ctx})".strip()


def rebuild_frame_from_stack(frame: QueryFrame, stack: List[DrillLevel]) -> QueryFrame:
    """After a drill_up pop, re-derive filters from the (now shorter) stack so
    frame.filters stays consistent with drill_path for the NEXT turn's
    render/prompt — the actual authoritative filters still get overwritten by
    harvest_frame() once the engine re-executes and returns fresh evidence;
    this only keeps the pre-call view honest in the interim.

    Each surviving level keeps the frame's OWN filter record when it has one. A DrillLevel
    stores only dimension + value, so rebuilding from it alone dropped the raw `column`
    (and the operator actually executed) — and the inference boundary discards any filter
    without a column. Measured 2026-09-24: "go back" from depth 2 reached the engine with
    `filter_values` but no `filters`, so the remaining level was lost and the base
    question came back ungrouped."""
    existing = list(frame.get("filters") or [])
    filters: List[FilterFact] = []
    for lvl in stack:
        match = next((f for f in existing if _same_field(f.get("field"), lvl["dimension"])
                      and str(f.get("value")).lower() == str(lvl.get("value")).lower()), None)
        filters.append(dict(match) if match else
                       {"field": lvl["dimension"], "operator": "equals",
                        "value": lvl.get("value"), "source": "executed_sql"})
    return {**frame, "filters": filters, "drill_path": stack}


def hold_subject_on_unplaced_turn(prev_frame: Optional[QueryFrame], harvested: Dict[str, Any],
                                  delta_type: str, prev_stack: Optional[List[DrillLevel]]) -> Dict[str, Any]:
    """Do not let a turn the classifier could not PLACE rewrite the conversation's subject.

    Same principle as keep_entity_on_lane_change, applied to `ambiguous`.

    `ambiguous` is what parse_delta_response returns for a genuine judgment call AND for a
    model call that failed or timed out — chatbot/llm.py returns None uniformly for both.
    It is therefore not evidence about the topic either way. But it has a second-order
    effect that IS destructive: an ambiguous turn carries no context (carry_state excludes
    it), so the engine answers the bare message, lands on whatever table that fragment
    routes to, and `is_topic_switch` then reads that as "the user changed subject" — at
    BOTH call sites, the drill-stack reset in memory_write_node and the frame reset in
    merge_frame_post_execution. Measured 2026-09-24, 3 runs of 3: a refused turn mid-drill
    made the NEXT follow-up classify ambiguous, and a live drill path was erased by a turn
    nobody intended as a new question.

    Holding the entity fixes both sites at once, because is_topic_switch compares entities.

    Deliberately narrow — an ambiguous turn on its own is NOT enough:
      · mid-drill only. A live drill stack is positive evidence that a continuation is the
        better reading. With no stack there is nothing to protect and a genuinely new
        subject, phrased oddly, is free to move the frame as it always could.
      · a DIFFERENT entity only; same entity is a no-op.
      · nothing else about the turn is touched: sql, row count and status are facts about
        what just ran and are written unchanged. Only the claim "this is what the
        conversation is about now" is refused.
    """
    if delta_type != "ambiguous" or not prev_frame or not harvested or not prev_stack:
        return harvested
    if not prev_frame.get("entity") or not harvested.get("entity"):
        return harvested
    if harvested["entity"] == prev_frame["entity"]:
        return harvested
    out = dict(harvested)
    out["entity"] = prev_frame.get("entity")
    out["entity_display"] = prev_frame.get("entity_display")
    if prev_frame.get("route"):
        out["route"] = prev_frame.get("route")
    if prev_frame.get("source_id") is not None:
        out["source_id"] = prev_frame.get("source_id")
    logger.info("hold_subject_on_unplaced_turn: turn came back 'ambiguous' and routed to %r, "
                "but a %d-level drill on %r is live — keeping the subject",
                harvested.get("entity"), len(prev_stack), prev_frame.get("entity"))
    return out
