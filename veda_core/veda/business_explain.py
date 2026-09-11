# =============================================================================
# veda/business_explain.py
# Deterministic, code-only, end-user-facing explainability.
#
# Design principle: explain = f(final validated SQL, semantic model, validation
# checks) — NEVER f(retrieval/routing/ranking internals). Everything here is
# derived by parsing the SQL text that already passed validation, plus static
# business metadata from the semantic model. No LLM. No dependence on which
# retrieval/routing strategy produced the SQL — so this stays stable even if
# that internal machinery changes completely.
#
# Contrast with veda/explain.py's ExplainTrace: that is an engineering trace
# (candidate tables, retrieval scores, router tie-breaks) for US to debug the
# pipeline. This module is the business-facing "what query did it run" view,
# for END USERS — never expose the former through the latter's shape.
# =============================================================================

import re
from typing import Any, Dict, List, Optional, Tuple
import sqlglot
from sqlglot import exp

_PRIMARY_ENTITY_RE = re.compile(r"^(?:a|an)\s+(.+?)\.?$", re.IGNORECASE)

_AGG_WORD = {"SUM": "total", "AVG": "average", "MIN": "minimum", "MAX": "maximum"}

_OP_WORD = {
    "EQ": "equals", "NEQ": "not equals", "GT": "greater than", "GTE": "greater than or equal to",
    "LT": "less than", "LTE": "less than or equal to", "Like": "contains",
    "In": "is one of", "Is": "is",
}

# One underlying check can be worth more than one plain-language guarantee to a
# reader (e.g. "read-only" and "duplicate-safe" are both real properties that a
# single AST check verifies at once) — so this maps to a LIST of labels.
_CHECK_LABELS = {
    "ast_readonly_parameterized_fanout": ["Read-only query", "Duplicate-safe (no double-counting)"],
    "qualifier_completeness": ["No requested filters were ignored"],
    "ir_equivalence": ["No extra filters, joins, or grouping were added"],
    "value_grounding": ["All filter values exist in the data"],
}

# Stripped when humanizing a raw table name into a business-facing dataset name,
# used only as a fallback when the semantic model has no better label.
_TABLE_PREFIXES = ("assets_", "accounts_", "worklists_", "organization_", "attachments_",
                   "evaluation_", "ingestion_", "query_", "chat_")

# Deterministic visualization-reasoning phrasing, keyed by chart type — the
# ONLY thing build_explain's visualization block adds beyond what already
# existed. Standardized rather than the SLM's own free-text "reason", so
# explainability stays LLM-free like the rest of this module.
_CHART_REASON_TEMPLATES = {
    "bar":  "Bar chart selected because the query compares a numeric measure across discrete categories.",
    "line": "Line chart selected because the query tracks a numeric measure over time.",
    "pie":  "Pie chart selected because the query breaks a numeric measure down by category.",
    "line_histogram": "Combo chart selected because the query compares two numeric measures across the same dimension.",
}


def _humanize(name: str) -> str:
    return " ".join(w for w in name.replace("_", " ").split() if w).title()


def _pluralize(word: str) -> str:
    lower = word.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if lower.endswith("y") and len(word) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _pluralize_phrase(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    words[-1] = _pluralize(words[-1])
    return " ".join(words)


def _business_table_name(table: str, sm: Optional[dict]) -> str:
    """Dataset display name. No short display-name field exists on tables in the
    semantic model today — prefer extracting the noun phrase from the table's
    "primary_entity" sentence (e.g. "An annotation record." -> "annotation record"),
    which is already business-authored and doesn't need a table name to survive
    concatenation (e.g. "leaselisting" can't be re-split into "Lease Listing"
    without a dictionary). Falls back to humanizing the raw table name."""
    if not table:
        return ""
    meta = (sm or {}).get("tables", {}).get(table) or {}
    entity = meta.get("primary_entity")
    if entity:
        m = _PRIMARY_ENTITY_RE.match(entity.strip())
        if m and m.group(1).split():
            return _pluralize_phrase(" ".join(w.capitalize() for w in m.group(1).split()))

    name = table
    for p in _TABLE_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    humanized = _humanize(name)
    return _pluralize_phrase(humanized) if humanized else ""


def _business_field_name(table: str, col: str, sm: Optional[dict]) -> str:
    cols_meta = (sm or {}).get("columns", {}) or {}
    meta = cols_meta.get(f"{table}.{col}") if table else None
    if meta is None:
        # filters/aggregations/orderings from the SQL AST carry bare column names
        # (no table qualifier) — fall back to a suffix match across the model.
        meta = next((v for k, v in cols_meta.items() if k.endswith(f".{col}")), None)
    if meta and meta.get("business_role"):
        return meta["business_role"]
    return _humanize(col)


def _extract(sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
    """One self-contained sqlglot pass over the final SQL. Deliberately NOT a
    reuse of veda/ir_equivalence.py's extract_sql_ir — that module's shape is
    owned by SQL-safety validation and free to change for validation reasons;
    this module's contract must stay independently stable for explainability.

    `sql` is the EXECUTED sql — veda/validation.py's validate_and_parameterize()
    rewrites every filter literal into a %s placeholder (bound separately in
    `params`, in the same left-to-right order they appear in the rendered SQL)
    for safe execution. Without `params`, every filter's value would come back
    None (a placeholder has no exp.Literal to find) — `params` lets filter
    values be resolved back by position for explainability/memory purposes."""

    out = {"entities": [], "filters": [], "aggregations": [], "groupings": [],
           "orderings": [], "distinct": False, "limit": None, "aliases": {}}
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return out
    if tree is None:
        return out

    # Map each Placeholder node (by identity) to its bound value, in the SAME
    # left-to-right document order validate_and_parameterize() used to build
    # `params` — find_all() walks the tree in source order, matching that.
    placeholder_values = {}
    if params:
        for i, ph in enumerate(tree.find_all(exp.Placeholder)):
            if i < len(params):
                placeholder_values[id(ph)] = params[i]

    out["entities"] = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    out["distinct"] = tree.find(exp.Distinct) is not None

    # Aggregations, DEDUPED by (function, column). A ranked aggregate repeats its
    # measure in both the SELECT list and the ORDER BY ("SELECT dim, AVG(m) ... ORDER
    # BY AVG(m) DESC" — exactly what the deterministic grouped/superlative planner
    # emits), so a raw find_all counts the SAME measure twice. That inflated the
    # measure count and made detect_result_shape misread a single-measure RANKING/
    # GROUPED result as a multi-measure PIVOT, suppressing its findings + chart. The
    # aggregations list represents DISTINCT computed measures, so an aggregate is
    # counted once regardless of how many clauses reference it; genuinely different
    # aggregates (SUM(a), COUNT(b)) are still distinct entries → real PIVOTs survive.
    _seen_aggs = set()
    for a in tree.find_all(exp.AggFunc):
        col = a.find(exp.Column)
        key = (a.key.upper(), col.name if col is not None else None)
        if key in _seen_aggs:
            continue
        _seen_aggs.add(key)
        out["aggregations"].append(key)

    # SELECT-list aliases (e.g. `SUM(lease_amount) AS total`) — GROUP BY/ORDER BY
    # elsewhere in the query reference the alias, not the real column, so without
    # this map a field like "total" could never resolve back to "Lease Amount".
    select_node = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select_node is not None:
        for proj in select_node.expressions:
            if isinstance(proj, exp.Alias):
                col = proj.this.find(exp.Column)
                if col is not None:
                    out["aliases"][proj.alias] = col.name

    grp = tree.find(exp.Group)
    if grp is not None:
        for e in grp.expressions:
            c = e.find(exp.Column)
            if c is not None:
                out["groupings"].append(c.name)

    order = tree.find(exp.Order)
    if order is not None:
        for e in order.expressions:
            c = e.find(exp.Column)
            if c is not None:
                out["orderings"].append((c.name, bool(e.args.get("desc"))))

    limit_node = tree.find(exp.Limit)
    if limit_node is not None:
        try:
            out["limit"] = int(limit_node.expression.name)
        except Exception:
            out["limit"] = None

    def _in_subquery(node) -> bool:
        p = node.parent
        while p is not None:
            if isinstance(p, (exp.Exists, exp.Subquery)):
                return True
            p = p.parent
        return False

    where = tree.find(exp.Where)
    if where is not None:
        ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Is)
        for pred in where.find_all(ops):
            if _in_subquery(pred):
                continue
            col = pred.find(exp.Column)
            if col is None:
                continue
            lit = pred.find(exp.Literal)
            if lit is not None:
                val = lit.name
            else:
                b = pred.find(exp.Boolean)
                if b is not None:
                    val = str(b.this)
                else:
                    ph = pred.find(exp.Placeholder)
                    val = placeholder_values.get(id(ph)) if ph is not None else None
            out["filters"].append((col.name, type(pred).__name__, val))
    return out


def extract_sql_facts(sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Public entry point onto `_extract()` — the same zero-LLM sqlglot pass
    business_explain already runs, exposed for callers outside this module
    (e.g. veda/result_analyzer.py's InsightContext) that need the same
    entities/filters/aggregations/groupings/orderings/limit facts without a
    second SQL parse."""
    return _extract(sql, params=params)


def _filter_phrase(field: str, op_class: str, val: Optional[str]) -> str:
    if op_class == "Is" and val is None:
        return f"{field} is empty"
    word = _OP_WORD.get(op_class, op_class.lower())
    return f"{field} {word} {val}" if val is not None else f"{field} {word}"


def _build_understanding(*, dataset: str, aggregations: List[Tuple[str, Optional[str]]],
                          groupings: List[str], orderings: List[Tuple[str, bool]],
                          limit: Optional[int], filter_phrases: List[str],
                          field_of: Any) -> str:
    order_field = field_of(orderings[0][0]) if orderings else None

    if limit is not None and order_field:
        # "top N X by Y" — when grouped, N counts groups (e.g. projects), not raw
        # dataset rows, so the group field names X; otherwise fall back to dataset.
        subject = _pluralize_phrase(field_of(groupings[0])) if groupings else dataset
        head = f"Find the top {limit} {subject} by {order_field}"
    elif aggregations:
        func, col = aggregations[0]
        if func == "COUNT":
            head = f"Count all {dataset}"
        else:
            head = f"Calculate {_AGG_WORD.get(func, func.lower())} {field_of(col) if col else dataset}"
        if groupings:
            head += f", grouped by {', '.join(field_of(g) for g in groupings)}"
    else:
        head = f"List {dataset}"
        if groupings:
            head += f", grouped by {', '.join(field_of(g) for g in groupings)}"
        if limit is not None:
            head += f" (top {limit})"

    if filter_phrases:
        head += " where " + ", ".join(filter_phrases)
    return head.strip() + "."


def _expose_sql() -> bool:
    """Whether the generated SQL may appear in the end-user explain payload."""
    try:
        import config
        return bool(getattr(config, "EXPLAIN_EXPOSE_SQL", True))
    except Exception:
        return True


#: The federated executor attaches every source under a catalog named `src_<id>`
#: (query/federated_executor.py::catalog_name), so a cross-source statement reads
#: `src_2.homzhub."assets_asset"`. That is the SOURCE ID, in the one string this
#: layer publishes verbatim — it would walk straight past `demote_source_ids`,
#: which exists precisely to keep ids out of the user-facing blocks.
_CATALOG_REF = re.compile(r"\bsrc_([A-Za-z0-9_]+)\b")


def _name_catalogs(sql):
    """Replace `src_<id>` catalog qualifiers with the source's display NAME.

    The reader gets the same statement, still says which source each table came
    from, and no longer carries an internal identifier. An unknown or unauthorised
    source resolves to the generic label rather than its id — display_name never
    falls back to the raw id, which is what makes this safe rather than cosmetic.

    Always quoted, because a display name may contain spaces or dots. A statement
    with no `src_` in it — every single-source query — comes back untouched.
    """
    if not sql:
        return sql or None
    try:
        from veda import source_names as _sn
        return _CATALOG_REF.sub(
            lambda m: '"%s"' % _sn.display_name(m.group(1)).replace('"', ""), sql)
    except Exception:
        # Never publish the raw catalog because the lookup failed.
        return _CATALOG_REF.sub('"a data source"', sql)


def _apply_v2(out: Dict[str, Any], *, trace: Any = None, trace_id: str = "") -> None:
    """Merge the v2 blocks (sources / routing / execution / warnings / result /
    cross_source / support) into an already-built v1 payload, in place.

    ADDITIVE BY CONTRACT: no v1 key is removed, renamed or reshaped, so a client
    reading `data_used.datasets` or `validation.checks` is unaffected. Only the
    `version` string changes, which is exactly how a consumer detects the richer
    shape. A no-op when EXPLAIN_V2_ENABLED is off or no trace is available.

    Everything merged here comes from veda/safe_projection.py — the single place
    allowed to read the internal trace — so this function never touches a trace
    section itself.
    """
    try:
        import config
        if not bool(getattr(config, "EXPLAIN_V2_ENABLED", False)):
            return
    except Exception:
        return
    try:
        tr = trace
        if tr is None:
            from veda.explain import current_trace
            tr = current_trace()
        if tr is None or not getattr(tr, "enabled", False):
            return
        from veda import safe_projection as sp
        ext = sp.build_explain_extension(
            tr, trace_id=trace_id or getattr(tr, "trace_id", "") or "",
            operations=out.get("operations"),
            validation=out.get("validation"))
        if not ext:
            return
        out.update(ext)
        out["version"] = "2.0"
    except Exception:
        # An explainability failure must never cost the caller its v1 payload.
        pass


def build_explain(*, sql: str, table: str, sm: Optional[dict],
                   checks: Optional[List[dict]] = None,
                   visualization: Optional[dict] = None,
                   params: Optional[List[Any]] = None,
                   timeline: Optional[List[Tuple[str, str]]] = None,
                   confidence: Optional[float] = None,
                   trace: Any = None,
                   trace_id: str = "") -> Dict[str, Any]:
    """Deterministic, LLM-free explainability for the end-user chat UI.
    Returns a plain dict matching the documented explainability schema.

    `visualization`: the Insight Engine's already-validated chart spec
    (query/result_explainer.py's validate_visualization — never the raw,
    unvalidated SLM suggestion), when one was produced. Optional and additive:
    omitted entirely from the returned dict when None, so every existing
    caller/consumer of build_explain() is unaffected.

    `params`: the bound values validate_and_parameterize() rewrote `sql`'s
    filter literals into %s placeholders for (veda/pipeline.py's `params`,
    same order) — without these every filter's value comes back None (see
    _extract()'s docstring).

    `timeline`: the run's own `_tick()` (phase, message) checkpoints
    (veda/pipeline.py's `_ticks`), passively collected — NOT recomputed or
    re-derived here, just relayed. Always present in the returned dict as a
    list (possibly empty), same "always-present, empty/None default" schema
    convention as `confidence` below — unlike `visualization`, which is
    omitted entirely when not applicable rather than genuinely unknown."""
    ir = _extract(sql or "", params=params)
    entities = ir["entities"] or ([table] if table else [])
    primary = entities[0] if entities else table

    aliases = ir["aliases"]
    field_of = lambda col: _business_field_name(primary, aliases.get(col, col), sm)   # noqa: E731

    datasets = [_business_table_name(t, sm) for t in entities] or (
        [_business_table_name(table, sm)] if table else [])

    fields: List[str] = []
    for col in [col for _, col in ir["aggregations"] if col] + ir["groupings"] + \
                [col for col, _ in ir["orderings"]] + [col for col, _, _ in ir["filters"]]:
        name = field_of(col)
        if name and name not in fields:
            fields.append(name)

    filter_phrases = [_filter_phrase(field_of(c), op, v) for c, op, v in ir["filters"]]

    operations: List[Dict[str, str]] = []
    if ir["aggregations"]:
        for func, col in ir["aggregations"]:
            if func == "COUNT":
                summary = "Count distinct records" if ir["distinct"] else "Count records"
                operations.append({"type": "count", "summary": summary})
            else:
                word = _AGG_WORD.get(func, func.lower())
                operations.append({"type": word, "summary": f"Calculate {word} {field_of(col) if col else ''}".strip()})
    for g in ir["groupings"]:
        operations.append({"type": "group", "summary": f"Group by {field_of(g)}"})
    for col, desc in ir["orderings"]:
        operations.append({"type": "sort", "summary": f"Sort by {field_of(col)} ({'highest' if desc else 'lowest'} first)"})
    if ir["limit"] is not None:
        operations.append({"type": "limit", "summary": f"Return top {ir['limit']}"})
    if not operations:
        operations.append({"type": "list", "summary": "List records"})

    understanding = _build_understanding(
        dataset=(datasets[0] if datasets else "records").lower(),
        aggregations=ir["aggregations"], groupings=ir["groupings"], orderings=ir["orderings"],
        limit=ir["limit"], filter_phrases=filter_phrases, field_of=field_of,
    )

    check_items = []
    all_passed = True
    for c in (checks or []):
        passed = c.get("status") == "pass"
        all_passed = all_passed and passed
        for label in _CHECK_LABELS.get(c.get("name"), [c.get("name")]):
            check_items.append({"label": label, "passed": passed})

    # One short phrase per operation/filter, for callers that want a
    # breakdown instead of parsing the single run-on `summary` sentence —
    # pure assembly of `operations`/`filter_phrases`, both already computed
    # above; no new derivation. Additive alongside `summary`, which stays
    # unchanged for any existing consumer relying on it as one string.
    breakdown = [op["summary"] for op in operations] + filter_phrases

    out = {
        "version": "1.0",
        "understanding": {"summary": understanding, "breakdown": breakdown},
        "data_used": {"datasets": datasets, "fields": fields},
        "operations": operations,
        "filters": {
            "applied": [
                {"field": field_of(c), "operator": _OP_WORD.get(op, op.lower()), "value": v}
                for c, op, v in ir["filters"]
            ],
            "summary": ", ".join(filter_phrases) if filter_phrases else "No filters applied.",
        },
        # `passed` is None when NOTHING was checked. It used to be True, because
        # all_passed starts True and an empty check list never falsifies it — so a
        # payload with `checks: []` claimed `passed: true`. That is a false
        # assurance: the reader is told the result cleared checks that never ran.
        # Observed live on a document answer.
        "validation": {"passed": (all_passed if check_items else None),
                       "checks": check_items},
        # SQL visibility is now a decision, not a constant. This used to be a
        # hardcoded True, so the generated SQL reached EVERY end user with no way
        # to turn it off. EXPLAIN_EXPOSE_SQL defaults True again (2026-09-11, after
        # a brief spell defaulting off under D2) — and the api tier can gate it by
        # setting the flag or stripping the block for non-technical users. When
        # off, the key stays present with query=None so no consumer has to
        # null-check the block itself.
        # `enabled` means "there IS SQL and you may see it", not merely "you may
        # see SQL". Both halves matter now that EXPLAIN_EXPOSE_SQL defaults ON
        # again (2026-09-11): a head that ran NO SQL — a document answer, a
        # refusal — would otherwise advertise `enabled: true, query: null`, which
        # reads as "SQL exists and we are withholding it" rather than "this
        # question was not answered with SQL at all". That exact contradiction was
        # observed live on a document answer and is what apps/chat/services.py's
        # `_NO_EXPLAIN` fallback already guards against on its own path.
        "sql": {"enabled": bool(sql) and _expose_sql(),
                "query": _name_catalogs(sql) if _expose_sql() else None},
        # Weakest-link confidence from the run's own anchor-selection + join-plan
        # gating signals (veda/pipeline.py's _done(), query/result_explainer.py's
        # synthesize_confidence) — never an LLM self-report. None only when the
        # caller couldn't compute one (e.g. synthesize_confidence itself raised).
        # Always present as a key, matching _NO_EXPLAIN's own "key present, value
        # None" convention for not-yet-available fields (apps/chat/services.py).
        "confidence": confidence,
        "timeline": [{"phase": p, "message": m} for p, m in (timeline or [])],
    }
    _apply_v2(out, trace=trace, trace_id=trace_id)
    if visualization:
        vtype = visualization.get("type")
        out["visualization"] = {
            "type": vtype,
            # Deterministic, standardized phrasing — not the SLM's raw "reason"
            # text (which can be vague/generic) — same principle as the rest of
            # this module: explain = f(final SQL/shape), never f(an LLM's prose).
            "reason": _CHART_REASON_TEMPLATES.get(vtype, visualization.get("reason")),
            "fields": [f for f in (field_of(visualization.get("x_axis")) if visualization.get("x_axis") else None,
                                   field_of(visualization.get("y_axis")) if visualization.get("y_axis") else None)
                      if f],
        }
    return out


def build_refusal_explain(status: str, feedback: Optional[dict],
                          *, trace: Any = None, trace_id: str = "") -> Optional[Dict[str, Any]]:
    """The refusal-path counterpart to build_explain() — same explainability
    CONTRACT (a structured object the chat UI can render), but for a turn
    that never produced SQL. Deliberately thin: reuses veda/feedback.py's
    explain_failure() output verbatim (why/what_needed/suggestions are
    already deterministic, human-authored-template sentences, per status —
    see that module) rather than re-deriving anything from `sql`/`sm`, which
    don't exist for a refusal. Returns None when there's no feedback to show
    (FEEDBACK_ENABLED=False, or explain_failure() itself failed) — the same
    "no explain object" signal build_explain()'s own caller already handles
    via the existing `explain = None` init in pipeline.py::_done()."""
    if not feedback:
        return None
    out = {
        "version": "1.0",
        "status": status,
        "understanding": {"summary": feedback.get("why")},
        "why": feedback.get("why"),
        "what_would_help": feedback.get("what_needed"),
        "suggestions": feedback.get("suggestions") or [],
    }
    # A REFUSED turn is exactly where a user most needs the caveats and a support
    # reference, and it previously got neither (the v2 extension only reached the
    # success path). Only the blocks that MEAN something without a result are
    # merged — never `routing`/`execution`/`result`, which would describe work
    # that did not produce an answer.
    _apply_v2_refusal(out, trace=trace, trace_id=trace_id)
    return out


def _apply_v2_refusal(out: Dict[str, Any], *, trace: Any = None, trace_id: str = "") -> None:
    """The refusal-path counterpart to _apply_v2: warnings + limitations + timeline
    + provenance + support only. No-op when EXPLAIN_V2_ENABLED is off. Never raises.

    Provenance is included here deliberately. A replayed verified query is MORE
    relevant on a refusal than on a success — measured live, 2 of 3 verified-cache
    replays were stopped by the alignment gate, so the reuse is often the very
    reason the question could not be answered. The blocks that are still omitted
    (routing / execution / execution_plan) are omitted because they genuinely do
    not apply: a refusal executed nothing.
    """
    try:
        import config
        if not bool(getattr(config, "EXPLAIN_V2_ENABLED", False)):
            return
    except Exception:
        return
    try:
        tr = trace
        if tr is None:
            from veda.explain import current_trace
            tr = current_trace()
        if tr is None or not getattr(tr, "enabled", False):
            return
        from veda import safe_projection as sp
        out["warnings"] = sp.build_warnings(tr)
        out["limitations"] = sp.build_limitations(tr)
        # WHERE WE LOOKED. A refusal executed nothing, so no source "participated"
        # — but the reader still needs to know which data was searched, and the
        # thinking model was already telling them one source was found while this
        # payload named none. Two true facts that read as a contradiction.
        #
        # Safe against overclaiming: build_data_sources only ever names a source
        # with proof it was in play, and the per-source `rows` it can attach is
        # absent here because nothing was retrieved — so the block says "this is
        # where we looked", never "this is what answered".
        _srcs = sp.build_data_sources(tr)
        if _srcs:
            out["sources"] = _srcs
            sp.demote_source_ids(out)      # ids belong in `audit`, on BOTH paths
        # Under `audit` — level 3 (§9), the same as the answered path. Keeping a
        # top-level copy here is how `source_selection` was still reaching the
        # normal UX on refusals after the answered path had been moved: the SAME
        # "wired to one path" mistake as EXP-B1/B4/B5 and the terminal step frame.
        out.setdefault("audit", {})["timeline_summary"] = sp.build_timeline_summary(tr)
        _prov = sp.build_provenance(tr)
        if _prov:
            out["provenance"] = _prov
        _tid = trace_id or getattr(tr, "trace_id", "") or ""
        if _tid:
            out["support"] = {"trace_id": _tid}
        out["version"] = "2.0"
    except Exception:
        pass
