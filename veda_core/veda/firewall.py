"""veda.firewall — the ONE firewall over (IR, SQL) (M3 checkpoint 1, 2026-09-16).

    firewall.check(ir, sql, sm, *, query, allowed_tables, allowed_columns, ...) -> FirewallVerdict

Every head calls this before executing SQL: the single-source deterministic branches and
the LLM branch (veda/pipeline.py), the fast path, the verified-cache replay, Tier-2's
envelope / shared-planner / IR paths (veda_hybrid.py), the understanding re-entry, and the
federated head (query/federated_route.py → compose_federated). It is a pure MOVE of the
gates that already existed, in the order they already ran — zero behaviour change for
checkpoint 1 — with one addition: where the IR is COMPLETE, the qualifier / alignment
checks are IR-vs-SQL (does the SQL carry every filter, group key, measure, limit, distinct
the IR has?) and the text heuristics are only consulted for slots the IR does not know.

Gate order (unchanged from the pipeline):
    1. value grounding          (validation.value_grounding)        → ungrounded
    2. qualifier completeness   (validation.qualifier_completeness) → qualifier_dropped
    3. alignment guards         (intent_sql_alignment.*)            → shape_mismatch
    4. IR equivalence           (ir_equivalence.validate_ir_equivalence) → ir_mismatch
    5. RBAC allow-list          (rbac_filter.narrow_allowed)        → rbac
    6. AST validation + params  (validation.validate_and_parameterize) → invalid | ok
Callers keep their own reaction to a verdict (refuse / re-enter / fall back / repair) —
the firewall decides, the head reacts. Every verdict is written to the explain trace as
`firewall = {verdict, ir_partial, checks_run, head, slot, reason}`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp

from veda.ir import QueryIR, partial as _partial_ir

# Verdict kinds — the typed refusal vocabulary every head maps onto its own status.
OK, UNGROUNDED, QUALIFIER_DROPPED, SHAPE_MISMATCH, IR_MISMATCH, RBAC, INVALID = (
    "ok", "ungrounded", "qualifier_dropped", "shape_mismatch", "ir_mismatch", "rbac", "invalid")


@dataclass
class FirewallVerdict:
    verdict: str
    sql: Optional[str] = None            # parameterised SQL when ok
    params: List[Any] = field(default_factory=list)
    reason: str = ""
    slot: Optional[str] = None           # the offending IR slot / column, when known
    detail: Any = None                   # gate-specific payload (e.g. (column, value))
    checks_run: List[str] = field(default_factory=list)
    ir_partial: bool = False
    head: str = "unknown"
    allowed_tables: Set[str] = field(default_factory=set)
    allowed_columns: List[str] = field(default_factory=list)
    rbac_removed: List[str] = field(default_factory=list)
    shape_kind: Optional[str] = None     # which alignment guard fired (shape_mismatch only)
    dim_out: Optional[str] = None        # DIM_REFUSE / DIM_CLARIFY for the dimension guard

    @property
    def ok(self) -> bool:
        return self.verdict == OK

    def trace_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "ir_partial": self.ir_partial,
                "checks_run": list(self.checks_run), "head": self.head,
                "slot": self.slot, "reason": (self.reason or "")[:200]}


# ── IR-vs-SQL structural checks (used when the IR knows the slot) ──────────────
def _sql_facts(sql: str) -> Dict[str, Any]:
    """Columns / literals / aggregates / group keys / limit / distinct in the SQL, parsed."""
    out = {"columns": set(), "literals": set(), "aggs": set(), "group": set(),
           "limit": None, "distinct": False, "has_where": False, "ops": set(), "having": False}
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return out
    for c in tree.find_all(exp.Column):
        out["columns"].add(c.name.lower())
    for lit in tree.find_all(exp.Literal):
        if lit.is_string:
            out["literals"].add(str(lit.this).lower())
        else:
            out["literals"].add(str(lit.this))
    for a in tree.find_all(exp.AggFunc):
        out["aggs"].add(type(a).__name__.lower())
    for g in tree.find_all(exp.Group):
        for e in g.expressions:
            for c in e.find_all(exp.Column):
                out["group"].add(c.name.lower())
    lim = tree.find(exp.Limit)
    if lim is not None:
        try:
            out["limit"] = int(lim.expression.this)
        except Exception:
            pass
    out["distinct"] = tree.find(exp.Distinct) is not None
    out["has_where"] = tree.find(exp.Where) is not None
    out["having"] = tree.find(exp.Having) is not None
    for k, name in ((exp.GT, ">"), (exp.GTE, ">="), (exp.LT, "<"), (exp.LTE, "<="),
                    (exp.EQ, "="), (exp.NEQ, "!="), (exp.In, "IN"), (exp.Between, "BETWEEN"),
                    (exp.Is, "IS")):
        if tree.find(k) is not None:
            out["ops"].add(name)
    return out


_AGG_NAMES = {"count": {"count"}, "sum": {"sum"}, "avg": {"avg"}, "min": {"min"}, "max": {"max"}}


def _ir_vs_sql(ir: QueryIR, sql: str) -> Optional[FirewallVerdict]:
    """Structural completeness of the SQL against a COMPLETE IR. Returns a refusal or None.
    Only slots the IR knows are checked; a partial IR skips the unknown ones."""
    f = _sql_facts(sql)
    if ir.knows("filters"):
        for flt in ir.filters:
            col = (flt.column or "").lower()
            if col and col not in f["columns"]:
                return FirewallVerdict(QUALIFIER_DROPPED, reason=f"IR filter on {flt.column} absent from SQL",
                                       slot=f"filter:{flt.column}")
            if flt.op in (">", ">=", "<", "<=") and not (f["ops"] & {">", ">=", "<", "<="}) and not f["having"]:
                return FirewallVerdict(QUALIFIER_DROPPED, slot=f"filter:{flt.column}",
                                       reason=f"IR comparator {flt.op} on {flt.column} not in SQL")
            if flt.op in ("IS NULL", "IS NOT NULL") and "IS" not in f["ops"]:
                return FirewallVerdict(QUALIFIER_DROPPED, slot=f"filter:{flt.column}",
                                       reason=f"IR existence predicate on {flt.column} not in SQL")
    if ir.knows("group_keys"):
        for g in ir.group_keys:
            gcol = g.split(".")[-1].lower()
            gcol = gcol.split("(")[-1].rstrip(")") if "(" in gcol else gcol
            if gcol and gcol not in f["group"] and gcol not in f["columns"]:
                return FirewallVerdict(SHAPE_MISMATCH, slot=f"group:{g}", shape_kind="group_by",
                                       reason=f"IR group key {g} absent from SQL GROUP BY")
        if ir.group_keys and not f["group"]:
            return FirewallVerdict(SHAPE_MISMATCH, slot="group_keys", shape_kind="group_by",
                                   reason="IR is grouped, SQL has no GROUP BY")
    if ir.knows("measure") and ir.measure is not None and ir.measure.aggregation not in (None, "none", "ratio"):
        want = _AGG_NAMES.get(ir.measure.aggregation, set())
        if want and not (f["aggs"] & want):
            return FirewallVerdict(SHAPE_MISMATCH, slot="measure", shape_kind="aggregate",
                                   reason=f"IR aggregation {ir.measure.aggregation} absent from SQL")
    if ir.knows("limit") and ir.limit is not None and f["limit"] not in (None, ir.limit):
        return FirewallVerdict(SHAPE_MISMATCH, slot="limit", shape_kind="limit",
                               reason=f"IR limit {ir.limit}, SQL LIMIT {f['limit']}")
    if ir.knows("distinct") and ir.distinct and not f["distinct"]:
        return FirewallVerdict(SHAPE_MISMATCH, slot="distinct", shape_kind="distinct",
                               reason="IR asks DISTINCT, SQL has none")
    if ir.knows("time_window") and ir.time_window and ir.time_window.get("column"):
        tc = ir.time_window["column"].lower()
        if tc not in f["columns"] or not ({"BETWEEN", ">=", "<=", ">", "<"} & f["ops"]):
            return FirewallVerdict(QUALIFIER_DROPPED, slot="time_window",
                                   reason=f"IR time window on {ir.time_window['column']} not in SQL")
    return None


def qualifier_only(ir: Optional[QueryIR], sql: str, sm: dict, *, query: str,
                   user_message: Optional[str] = None) -> bool:
    """The qualifier gate alone (fast path decline check): IR-vs-SQL filters for a complete
    IR, then validation.qualifier_completeness. True = passes.

    `user_message` — see check() below."""
    from veda.validation import qualifier_completeness
    if ir is not None and not ir.ir_partial:
        v = _ir_vs_sql(ir, sql)
        if v is not None and v.verdict == QUALIFIER_DROPPED:
            return False
    ok, _missing = qualifier_completeness(query, sql, sm, user_message=user_message)
    return bool(ok)


# ── the entry point ───────────────────────────────────────────────────────────
def check(ir: Optional[QueryIR], sql: str, sm: dict, *, query: str,
          allowed_tables, allowed_columns, ctx=None,
          resolve_table: Optional[Callable] = None, skip_values=(),
          strict_qualifier: bool = False, llm_generated: bool = False,
          tf=None, join_constraints=None, fanout_guard=None,
          skip_predicate_cols=None, run_alignment: bool = True,
          run_ir_equivalence: bool = True, run_rbac: bool = True,
          head: str = "unknown", trace=None, _semantic_only: bool = False,
          _skip_value_and_qualifier: bool = False,
          user_message: Optional[str] = None) -> FirewallVerdict:
    """Run every gate, in the pipeline's order, and return ONE verdict. `ir=None` is
    treated as a fully partial IR (text heuristics for every slot). `_semantic_only`:
    stop after the semantic gates (value / qualifier / alignment / IR-equivalence) and
    return ok without parameterising — for a caller that already holds parameterised SQL
    (Tier-2's post-firewall validate).

    `user_message` is WHOSE WORDS the qualifier gate is about — the user's own message when
    the caller has one distinct from `query`. A chat follow-up reaches the engine as the
    user's bare words with the remembered state alongside, but the deterministic heads can
    still rebuild a fuller `query`, and the gate's contract is "every content token THE USER
    NAMED must appear in the SQL". None = use `query`, which is every non-chat caller.
    See veda/validation.py::qualifier_completeness."""
    if ir is None:
        ir = _partial_ir(head)
    checks: List[str] = []
    allowed_tables = set(allowed_tables or ())
    allowed_columns = list(allowed_columns or ())

    def _verdict(v: FirewallVerdict) -> FirewallVerdict:
        v.checks_run = list(checks)
        v.ir_partial = bool(ir.ir_partial)
        v.head = head
        v.allowed_tables, v.allowed_columns = allowed_tables, allowed_columns
        if trace is not None:
            try:
                # the pipeline calls check() in stages: ACCUMULATE checks_run across them
                # so the trace shows every gate the executed SQL passed, not the last stage
                prev = (getattr(trace, "sections", {}) or {}).get("firewall", {}) or {}
                seen = list(prev.get("checks_run") or [])
                v.checks_run = seen + [c for c in checks if c not in seen]
                trace.set("firewall", **v.trace_dict())
                trace.check("firewall", v.ok, v.reason)
            except Exception:
                pass
        return v

    # 1. value grounding — every string literal must exist in its column's sampled values
    from veda.validation import value_grounding, qualifier_completeness, validate_and_parameterize
    cols_meta = (sm or {}).get("columns", {}) or {}
    if resolve_table is None:
        _default = next(iter(allowed_tables)) if len(allowed_tables) == 1 else (ir.anchor or None)

        def resolve_table(colexp):  # noqa: E306
            if getattr(colexp, "table", None):
                return colexp.table
            owners = [t for t in allowed_tables if f"{t}.{colexp.name}" in cols_meta]
            return owners[0] if len(owners) == 1 else _default
    # The pipeline calls check() in STAGES (it reacts differently to each stage's refusal
    # — salvage after a dropped qualifier, re-entry after an alignment refusal); a later
    # stage skips the ones already passed via _skip_value_and_qualifier. Other heads call
    # it once for everything.
    if not _skip_value_and_qualifier:
        checks.append("value_grounding")
        ok_val, bad = value_grounding(sql, resolve_table, cols_meta, skip_values)
        if not ok_val:
            return _verdict(FirewallVerdict(UNGROUNDED, reason=f"value {bad[1]!r} not present in {bad[0]}",
                                            slot=f"filter:{bad[0]}", detail=bad))

        # 2. qualifier completeness — IR-vs-SQL when the IR is complete, else the text heuristic
        checks.append("qualifier")
        if not ir.ir_partial:
            v = _ir_vs_sql(ir, sql)
            if v is not None and v.verdict == QUALIFIER_DROPPED:
                v.detail = [v.slot]
                return _verdict(v)
        ok_q, missing = qualifier_completeness(query, sql, sm, strict=strict_qualifier,
                                               user_message=user_message)
        if not ok_q:
            return _verdict(FirewallVerdict(QUALIFIER_DROPPED, reason=f"dropped qualifier {missing!r}",
                                            slot="qualifier", detail=missing))

    # 3. alignment guards (temporal, entity-anchor, aggregate presence, filter presence,
    #    dimension) — IR-vs-SQL structural check first for a complete IR, then the
    #    existing text guards (they stay, unchanged, for checkpoint 1's zero-change rule).
    if run_alignment:
        checks.append("alignment")
        if not ir.ir_partial:
            v = _ir_vs_sql(ir, sql)
            if v is not None:
                return _verdict(v)
        from veda.intent_sql_alignment import (alignment_ok, aggregate_presence_ok,
                                               filter_presence_ok, dimension_alignment,
                                               DIM_REFUSE, DIM_CLARIFY)
        ok_a, why = alignment_ok(query, sql, sm)
        if not ok_a:
            return _verdict(FirewallVerdict(SHAPE_MISMATCH, reason=why, shape_kind="alignment"))
        ok_a, why = aggregate_presence_ok(query, sql, sm)
        if not ok_a:
            return _verdict(FirewallVerdict(SHAPE_MISMATCH, reason=why, shape_kind="aggregate_omission",
                                            slot="measure"))
        ok_a, why = filter_presence_ok(query, sql, sm)
        if not ok_a:
            return _verdict(FirewallVerdict(SHAPE_MISMATCH, reason=why, shape_kind="filter_omission",
                                            slot="filters"))
        dim_out, why = dimension_alignment(query, sql, sm)
        if dim_out in (DIM_REFUSE, DIM_CLARIFY):
            # a grounded GROUP BY (complete IR, type-checked) is not an ambiguity — M2 rule
            if dim_out == DIM_CLARIFY and not ir.ir_partial and ir.group_keys:
                if trace is not None:
                    try:
                        trace.set("dimension_alignment", grounded_group_by=ir.group_keys, guard_note=why)
                    except Exception:
                        pass
            else:
                return _verdict(FirewallVerdict(SHAPE_MISMATCH, reason=why, shape_kind="dimension",
                                                slot="group_keys", dim_out=dim_out))

    # 4. IR equivalence — LLM SQL must not add semantics the question never asked for
    if run_ir_equivalence:
        checks.append("ir_equivalence")
        from veda.ir_equivalence import validate_ir_equivalence
        _tcols = ({k.split(".", 1)[1] for k, m in cols_meta.items()
                   if k.split(".", 1)[0] in allowed_tables and (m or {}).get("semantic_type") == "TEMPORAL"}
                  if (tf is not None and (getattr(tf, "start", None) or getattr(tf, "end", None))) else set())
        ok_ir, viol = validate_ir_equivalence(query, sql, sm, allowed_tables=allowed_tables,
                                              skip_predicate_cols=skip_predicate_cols or set(),
                                              temporal_cols=_tcols, llm_generated=llm_generated)
        if not ok_ir:
            return _verdict(FirewallVerdict(IR_MISMATCH, reason="; ".join(viol), detail=list(viol)))

    if _semantic_only:
        return _verdict(FirewallVerdict(OK, sql=sql, params=[]))

    # 5. RBAC allow-list narrowing (the centralized final gate)
    removed: List[str] = []
    if run_rbac:
        checks.append("rbac")
        from veda.rbac_filter import narrow_allowed
        before = set(allowed_tables)
        allowed_tables, allowed_columns = narrow_allowed(allowed_tables, allowed_columns, sm, ctx)
        allowed_tables = set(allowed_tables or ())
        allowed_columns = list(allowed_columns or ())
        removed = sorted(before - allowed_tables)

    # 6. AST validation + parameterisation (the hard allow-list)
    checks.append("ast_parameterize")
    psql, params, err = validate_and_parameterize(sql, allowed_tables, allowed_columns,
                                                  join_constraints=join_constraints,
                                                  fanout_guard=fanout_guard)
    if err:
        v = FirewallVerdict(RBAC if removed and any(r in str(err) for r in removed) else INVALID,
                            reason=str(err), rbac_removed=removed)
        return _verdict(v)
    v = FirewallVerdict(OK, sql=psql, params=list(params or []), rbac_removed=removed)
    return _verdict(v)


def check_federated(ir: Optional[QueryIR], sql: str, *, query: str, scope_sources: List[str],
                    value_exists: Callable[[str, str, str], bool], head: str = "federated",
                    trace=None) -> FirewallVerdict:
    """The federated variant: no single semantic model / allow-list (each source has its
    own), so the gates are (1) every string literal in the SQL must exist in the OWNING
    source's sampled values (`value_exists(source_id, column, literal)`), (2) IR-vs-SQL
    completeness when the IR is known, (3) every table the SQL names must be in scope.
    Parameterisation is the federated executor's own (DuckDB); we return the SQL as-is."""
    if ir is None:
        ir = _partial_ir(head, source_scope=scope_sources)
    checks: List[str] = ["scope"]

    def _v(v: FirewallVerdict) -> FirewallVerdict:
        v.checks_run, v.ir_partial, v.head = list(checks), bool(ir.ir_partial), head
        if trace is not None:
            try:
                trace.set("firewall", **v.trace_dict())
            except Exception:
                pass
        return v

    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except Exception as e:
        return _v(FirewallVerdict(INVALID, reason=f"unparseable federated SQL: {e}"))
    # (3) scope: src_<id>.<schema>.<table> — every src id must be in scope
    import re as _re
    for t in tree.find_all(exp.Table):
        full = t.sql(dialect="duckdb")
        m = _re.search(r"src_(\d+)", full)
        if m and m.group(1) not in set(map(str, scope_sources)):
            return _v(FirewallVerdict(RBAC, reason=f"table {full} outside scope {scope_sources}",
                                      slot="source_scope"))
    # (1) literal value grounding against the owning source
    checks.append("value_grounding")
    for cmp in list(tree.find_all(exp.EQ)) + list(tree.find_all(exp.NEQ)) + list(tree.find_all(exp.In)):
        col = cmp.find(exp.Column)
        if col is None:
            continue
        lits = [l for l in cmp.find_all(exp.Literal) if l.is_string]
        if not lits:
            continue
        # owning source from the column's table qualifier, else from the FROM tables
        tbl = col.table or ""
        src = None
        for t in tree.find_all(exp.Table):
            full = t.sql(dialect="duckdb")
            if (t.alias and t.alias == tbl) or (tbl and tbl in full) or not tbl:
                m = _re.search(r"src_(\d+)", full)
                if m:
                    src = m.group(1)
                    break
        for l in lits:
            if src and not value_exists(src, col.name, str(l.this)):
                return _v(FirewallVerdict(UNGROUNDED, slot=f"filter:{col.name}", detail=(col.name, str(l.this)),
                                          reason=f"value {l.this!r} not present in source {src} column {col.name}"))
    # (2) IR completeness
    checks.append("qualifier")
    if not ir.ir_partial:
        v = _ir_vs_sql(ir, sql)
        if v is not None:
            return _v(v)
    # (2b) TEXT qualifier completeness. The IR check above only runs for a COMPLETE IR,
    # and a federated plan's IR is almost always partial — so a plan that simply omits a
    # filter had nothing checking it: no literal to validate (the filter is absent), no
    # IR to compare against. That is how a "properties in Mumbai" follow-up came back as
    # a GROUP BY over every city (cross-source battery §9.2). The single-source gate has
    # always had this check; the federated one never did.
    try:
        from veda.validation import federated_qualifier_completeness
        ok_q, missing = federated_qualifier_completeness(query, sql)
        if not ok_q:
            return _v(FirewallVerdict(QUALIFIER_DROPPED,
                                      reason=f"dropped qualifier {missing!r}",
                                      slot="qualifier", detail=missing))
    except Exception:
        pass                                  # never fail a query on the gate's own error
    return _v(FirewallVerdict(OK, sql=sql, params=[]))
