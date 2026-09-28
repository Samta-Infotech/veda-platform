"""veda.agent.federated — a planner-agent plan across two sources (planner agent, part D).

Where the federated route (query/federated_route.py::run_federated) has decided a
question names ≥ 2 in-scope sources, the agent plans over the WHOLE scope: its tools see
every source's tables (the relational model + each datalake's column model, merged the
way veda/runtime merges a multi-source scope), and join_path returns the tenant's
`cross_source_fk` edges between tables of two sources. A plan whose tables span two
sources over such an edge is compiled by the same builders (plan.compile_plan) with every
table written catalog-qualified for DuckDB (src_<id>."table" / src_<id>.<schema>."table"),
then handed to the federated route's own `_fed_compose` — firewall.check_federated, then
the FederatedExecutor. A plan inside ONE source is not this route's job: None, and the
federated route continues unchanged.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


def scope_sm(source_ids: List[str], tenant: str) -> Dict[str, Any]:
    """Every source's model merged under scope keys (colliding bare names → src{N}.t)."""
    from veda.runtime import _load_one_sm, _merge_scoped_sms
    parts = []
    for sid in source_ids:
        sm_i = None
        try:
            import veda_hybrid as VH
            prof = {}
            try:
                from veda_core.context import current_source_profiles
                prof = (current_source_profiles() or {}).get(str(sid)) or {}
            except Exception:
                prof = {}
            if str(prof.get("source_type") or "").lower() == "datalake":
                iso = VH._datalake_isolated_sm(sid)
                sm_i = iso[0] if iso else None
        except Exception:
            sm_i = None
        if sm_i is None:
            try:
                sm_i = _load_one_sm(str(sid), tenant)
            except Exception:
                sm_i = None
        if sm_i and sm_i.get("tables"):
            parts.append((str(sid), sm_i))
    return _merge_scoped_sms(parts) if parts else {"tables": {}, "columns": {}}


def _catalog(sid: str, bare: str, kinds: Dict[str, str]) -> str:
    from query.federated_route import _catalog_table
    return _catalog_table(str(sid), bare, kinds.get(str(sid), "parquet"))


def compile_federated(plan, sm, kinds: Dict[str, str], source_of) -> Optional[Tuple[str, Any]]:
    """Plan → (DuckDB SQL with catalog-qualified tables, IR) or None."""
    from veda.agent.plan import compile_plan, _bare
    from veda.understanding.frame_compiler import Declined
    bares = [_bare(t) for t in plan.tables]
    if len(set(bares)) != len(bares):
        return None                      # the same bare name in two sources: not expressible
    comp = compile_plan(plan, sm, source_scope=sorted({str(source_of(t)) for t in plan.tables}))
    if isinstance(comp, Declined):
        return None
    sql = comp.sql
    for t in plan.tables:
        b = _bare(t)
        cat = _catalog(source_of(t), b, kinds)
        sql = re.sub(rf'"{re.escape(b)}"(\s+t\d+\b)', lambda m: cat + m.group(1), sql)
    comp.ir.head = "federated.agent"
    return sql, comp.ir


def plan_federated(query: str, tenant: str, source_ids: List[str], kinds: Dict[str, str],
                   trace_sink: Optional[Dict[str, Any]] = None):
    """(sql, ir, cols) for a cross-source agent plan, or None."""
    from veda.understanding.vocabulary import scope_vocab
    from veda.agent.planner import run_planner
    from veda.agent.tools import ToolBox
    sids = [str(s) for s in source_ids]
    sm = scope_sm(sids, tenant)
    if not sm.get("tables"):
        return None
    vocab = scope_vocab(sm, source_ids=sids, tenant=tenant)
    tb = ToolBox(sm, vocab, query)
    ar = run_planner(query, sm, vocab, toolbox=tb, source_scope=sids)
    sec = {"trigger": "federated", **ar.trace()}
    if trace_sink is not None:
        trace_sink.update(sec)
    try:
        from veda.explain import current_trace
        current_trace().set("agent", **sec)
    except Exception:
        pass
    if ar.kind != "sql" or ar.plan is None:
        return None
    srcs = sorted({str(tb.source_of(t)) for t in ar.plan.tables})
    if len(srcs) < 2:
        return None                      # a single-source plan: the normal path's job
    if not any(j.get("kind") == "cross_source_fk" for j in ar.plan.joins):
        return None
    out = compile_federated(ar.plan, sm, kinds, tb.source_of)
    if out is None:
        return None
    sql, ir = out
    cols = [SimpleNamespace(source_id=s) for s in srcs]
    return sql, ir, cols
