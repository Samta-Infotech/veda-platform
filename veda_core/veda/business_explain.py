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
    import sqlglot
    from sqlglot import exp

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

    for a in tree.find_all(exp.AggFunc):
        col = a.find(exp.Column)
        out["aggregations"].append((a.key.upper(), col.name if col is not None else None))

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


def build_explain(*, sql: str, table: str, sm: Optional[dict],
                   checks: Optional[List[dict]] = None,
                   visualization: Optional[dict] = None,
                   params: Optional[List[Any]] = None,
                   timeline: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
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
        "validation": {"passed": all_passed, "checks": check_items},
        "sql": {"enabled": True, "query": sql or None},
        # Placeholder key for the separately-scoped "universal routing confidence"
        # work — schema-only, deliberately not computed here (no fake number).
        # Always present (None until populated), matching _NO_EXPLAIN's own
        # "key present, value None" convention for not-yet-available fields
        # (apps/chat/services.py) rather than visualization's omit-when-N/A one.
        "confidence": None,
        "timeline": [{"phase": p, "message": m} for p, m in (timeline or [])],
    }
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


def build_refusal_explain(status: str, feedback: Optional[dict]) -> Optional[Dict[str, Any]]:
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
    return {
        "version": "1.0",
        "status": status,
        "understanding": {"summary": feedback.get("why")},
        "why": feedback.get("why"),
        "what_would_help": feedback.get("what_needed"),
        "suggestions": feedback.get("suggestions") or [],
    }
