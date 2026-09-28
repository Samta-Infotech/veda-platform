"""veda.understanding.frame_path — the meaning-first query path (behind FRAME_PATH_ENABLED).

    frame → ground → probes (one revision round) → compile → [pipeline: firewall →
    execute → narrate]

run_frame_path() returns one of:
  kind="sql"      a compiled, verified statement + its COMPLETE IR (head "frame.*"); the
                  pipeline runs it through the same firewall / RBAC / parameterisation /
                  execution / narration as every head, via the fast-path lane.
  kind="clarify"  a typed clarify naming the unresolved slot and real alternatives.
  kind="degrade"  the frame could not be extracted, or its entity is only a guess the
                  router does not share → the existing chain answers, unchanged.

Authority: a NAME-grounded entity (or NAME+COVERAGE) is authoritative and may override
the router; a MODEL / SYNTHETIC / RETRIEVAL entity is used only when it equals the router's
primary (agreement), otherwise the path degrades. A clarify is only issued on an
authoritative entity (or for an entity ambiguity the question itself names) — the frame
path never replaces an answer the old chain could give with a question on a guess.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veda.understanding.frame import Frame
from veda.understanding.frame_grounding import (GroundedFrame, FrameClarify, ground_frame,
                                                AUTHORITATIVE)


@dataclass
class FramePathResult:
    kind: str                                   # sql | clarify | degrade
    reason: str = ""
    sql: Optional[str] = None
    ir: Any = None
    anchor: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    message: Optional[str] = None
    frame: Optional[Frame] = None
    grounded: Optional[GroundedFrame] = None
    notes: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    # the ONE source the statement's tables belong to (multi-source scope: the pipeline
    # executes there); None = the scope's primary, as before
    source_id: Optional[str] = None
    parts: List[str] = field(default_factory=list)          # kind="split": the sub-questions
    rag_sources: List[str] = field(default_factory=list)    # kind="rag": the document sources


class FrameLane:
    """Duck-types the fast-path result the pipeline's lane selection consumes."""
    route = "frame"

    def __init__(self, res: FramePathResult):
        self.sql = res.sql
        self.primary = res.anchor
        self.tables = list(res.tables)
        # every column the statement references, bare (the RBAC narrowing and the
        # allowed-column checks downstream read bare names, as for the fast path)
        cols = list(dict.fromkeys(c.split(".")[-1] for c in res.columns))
        try:
            import sqlglot
            from sqlglot import exp
            for c in sqlglot.parse_one(res.sql, read="postgres").find_all(exp.Column):
                if c.name and c.name not in cols:
                    cols.append(c.name)
        except Exception:
            pass
        self.columns = cols
        self.why = [f"frame {res.grounded.anchor_method if res.grounded else ''}".strip()]
        self.result = res


#: A frame handed in by the front door (one intent of a compound message): the part runs
#: through this path with THAT frame — no re-extraction. Request-scoped (ContextVar), set
#: and reset by veda_hybrid around exactly one part.
_INJECTED: "ContextVar[Optional[Frame]]" = ContextVar("veda_injected_frame", default=None)


def inject_frame(fr: Optional[Frame]):
    """Bind `fr` as the frame for the next run_frame_path in this context. Returns the
    token for reset_injected()."""
    return _INJECTED.set(fr)


def reset_injected(token) -> None:
    try:
        _INJECTED.reset(token)
    except Exception:
        _INJECTED.set(None)


def injected_frame() -> Optional[Frame]:
    return _INJECTED.get()


def _prepare_injected(inj: Frame, query: str, vocab, st: Dict[str, Any]) -> Frame:
    """The front door's frame, re-keyed onto THIS source's vocabulary: its entity table
    (a front-door key, possibly `src{N}.t`) becomes the per-source key, its entity concept
    name is mapped to a card when it names one, and the deterministic producers run on the
    part's own words exactly as they do for an extracted frame."""
    import copy as _copy
    from veda.understanding import producers as P
    from veda.understanding.vocabulary import norm_phrase
    fr = _copy.deepcopy(inj)
    et = fr.provenance.get("entity_table")
    if isinstance(et, str):
        bare = et.split(".", 1)[1] if et.startswith("src") and "." in et else et
        hit = next((t for t in vocab.cards if t == bare or vocab.cards[t].get("_bare_table") == bare), None)
        if hit:
            fr.provenance["entity_table"] = hit
        else:
            fr.provenance.pop("entity_table", None)
    if "entity_table" not in fr.provenance and fr.entity:
        lst = vocab.name_index().get(norm_phrase(fr.entity)) or []
        if lst:
            fr.provenance["entity_table"] = lst[0][0]
    for f in fr.filters:
        t = f.__dict__.get("_table")
        if isinstance(t, str) and t.startswith("src") and "." in t:
            f.__dict__["_table"] = t.split(".", 1)[1]
    frags = P.run_producers(query, vocab)
    st["producers"] = sorted(frags)
    st["injected"] = True
    return P.merge(fr, frags)


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "FRAME_PATH_ENABLED", False))
    except Exception:
        return False


def _summary(fr: Optional[Frame]) -> Dict[str, Any]:
    if fr is None:
        return {}
    d = fr.to_dict()
    d.pop("version", None)
    prov = d.get("provenance") or {}
    d["provenance"] = {k: v for k, v in prov.items() if k != "_alternatives"}
    return d


def _g_summary(g: Optional[GroundedFrame]) -> Dict[str, Any]:
    if g is None:
        return {}
    return {"anchor": g.anchor, "anchor_method": g.anchor_method,
            "measure": g.measure, "filters": [(f.table, f.column, f.op, f.value, f.grounding) for f in g.filters],
            "group_by": g.group_by, "order": g.order, "limit": g.limit, "time": g.time,
            "joins": [(j.table, j.fk_column) for j in g.joins], "notes": g.notes,
            "evidence": {k: v for k, v in (g.evidence or {}).items() if k in
                         ("name_hits", "model_pick", "coverage", "reanchored", "decision_shadow")}}


def run_frame_path(query: str, sm, *, router_primary: Optional[str] = None,
                   router_hint=None, force: bool = False) -> FramePathResult:
    """`router_primary`: the router's primary table, when the caller already has it.
    `router_hint`: a callable → (primary, [(table, score)…]) the pipeline passes instead
    (its memoised retrieval + rerank + select_primary_table) — called only when a decision
    needs the router's opinion, so a NAME-grounded turn never pays for retrieval here."""
    inj = _INJECTED.get()
    if not (force or _enabled() or inj is not None):
        return FramePathResult("degrade", "disabled")
    t0 = time.time()
    tr: Dict[str, Any] = {}
    fu = _agent_follow_up(query, sm, tr)
    if fu is not None:
        return fu
    _narrow_saved = _wide_tok = None
    try:
        from veda.understanding.vocabulary import scope_vocab
        from veda.understanding.frame_extractor import extract_frame
        vocab = scope_vocab(sm)
        tr["vocabulary"] = {"cards": len(vocab.cards), "built": vocab.built,
                            "examples": len(vocab.examples)}
        if not vocab.cards:
            return FramePathResult("degrade", "no_vocabulary", trace=tr)
        tr["scope_ids"] = _scope_ids()
        router_primary = _RouterPrimary(router_primary, router_hint, vocab, tr)
        stats: Dict[str, Any] = {}
        fr = (_prepare_injected(inj, query, vocab, stats) if inj is not None
              else extract_frame(query, vocab, stats=stats))
        tr["extract"] = stats
        if fr is None:
            return FramePathResult("degrade", "extract_failed", trace=tr)
        tr["frame"] = _summary(fr)
        # who produced this frame: the compound front door (injected) or this path's own
        # extraction — the "one extraction per first turn" check reads it
        tr["frame"]["source"] = "front_door" if inj is not None else "extract"
        if fr.provenance.get("existence") and not fr.filters:
            return FramePathResult("degrade", f"existence:{fr.provenance['existence']}", frame=fr, trace=tr)

        g = ground_frame(fr, vocab, sm, query, router_primary)
        res = _decide(g, fr, query, vocab, sm, router_primary, tr, stats)
        if res.kind != "sql" or res.grounded is None:
            res.trace = tr
            if res.kind == "clarify":
                return _agent_or(res, query, sm, vocab, fr, f"clarify:{res.reason}", tr)
            return res
        g = res.grounded
        try:
            from veda_core.context import try_current
            _ctx = try_current()
            _scope = [str(x) for x in (_ctx.source_ids or (_ctx.source_id,))] if _ctx else []
            if len(_scope) > 1 and g.source_id and str(g.source_id) != str(_ctx.source_id):
                # a multi-source scope executes against its primary source, so an anchor owned
                # by another source ran on the wrong DB (measured: exec_error on every source-4
                # question unpinned). When every table the frame touches lives in the anchor's
                # source, the probes, the compile and the execution NARROW to that source — the
                # entity is settled, only the DB was wrong. A frame spanning sources is
                # federation's job: degrade.
                srcs = _sources_of([g.anchor] + [j.table for j in g.joins], vocab, sm)
                tr["scope"] = {"anchor_source": g.source_id, "primary": str(_ctx.source_id),
                               "frame_sources": sorted(s for s in srcs if s)}
                if srcs != {str(g.source_id)}:
                    return FramePathResult("degrade", "anchor_in_other_source", frame=fr, grounded=g, trace=tr)
                _narrow_saved = _narrow(g.source_id)
                if _narrow_saved is not None:
                    _wide_tok = _WIDE.set(_narrow_saved)
                tr["scope"]["narrowed_to"] = str(g.source_id)
        except Exception:
            pass

        # ── Stage 4: verification probes + one revision round ──
        from veda.understanding.frame_probes import probe
        pr = probe(g)
        tr["probes"] = {"ran": pr.ran, "total": pr.total, "filtered": pr.filtered,
                        "per_filter": pr.per_filter, "order_distinct": pr.order_distinct,
                        "problems": [p["detail"] for p in pr.problems], "error": pr.error, "ms": pr.ms}
        if pr.ran and pr.problems:
            # an injected compound PART is not re-extracted (its budget is 30 s): its
            # problems go straight to the typed clarify below. A front-door frame for a
            # single-intent message keeps the revision round the extracted frame had.
            fr2 = (extract_frame(query, vocab, stats=tr.setdefault("revision", {}),
                                 observations=pr.observations)
                   if inj is None or inj.provenance.get("_front_door_single") else None)
            tr.setdefault("revision", {})
            g2 = ground_frame(fr2, vocab, sm, query, router_primary) if fr2 is not None else None
            res2 = _decide(g2, fr2, query, vocab, sm, router_primary, tr, stats) if fr2 is not None else None
            pr2 = probe(res2.grounded) if (res2 is not None and res2.kind == "sql" and res2.grounded) else None
            tr["revision"]["result"] = (res2.kind if res2 else "none")
            if pr2 is not None:
                tr["revision"]["problems"] = [p["detail"] for p in pr2.problems]
            if pr2 is not None and pr2.ran and not pr2.problems:
                fr, g, pr = fr2, res2.grounded, pr2
                tr["frame"] = _summary(fr)
                tr["frame"]["source"] = "revision"
            else:
                first = pr.problems[0]
                card = vocab.cards.get(g.anchor) or {}
                what = card.get("plural") or g.anchor
                msg = {
                    "empty_filter": lambda p: (f"No {what} match {p['filter'].column.replace('_', ' ')} "
                                               f"{p['filter'].op} {p['filter'].value} (0 of {pr.total})."),
                    "vacuous_range": lambda p: (f"Every one of the {pr.total} {what} satisfies "
                                                f"{p['filter'].column.replace('_', ' ')} {p['filter'].op} "
                                                f"{p['filter'].value} — did you mean a different amount?"),
                    "constant_order": lambda p: (f"Sorting {what} by {g.order[1].replace('_', ' ')} "
                                                 f"would not change anything — which field should I sort by?"),
                    "empty_conjunction": lambda p: (f"No {what} match all of those conditions together "
                                                    f"(0 of {pr.total}). Which condition should I relax?"),
                }[first["kind"]](first)
                if first["kind"] == "empty_filter" and first["filter"].grounding != "temporal":
                    from veda.understanding.frame_grounding import column_domain
                    dom = column_domain(vocab, sm, g.anchor, first["filter"].column)
                    if dom:
                        msg += f" Its {first['filter'].column.replace('_', ' ')} values are {', '.join(dom[:12])}."
                return _agent_or(FramePathResult("clarify", f"probe:{first['kind']}", message=msg, frame=fr,
                                                 grounded=g, anchor=g.anchor, trace=tr),
                                 query, sm, vocab, fr, f"probe:{first['kind']}", tr)

        # ── Stage 5: compile ──
        from veda.understanding.frame_compiler import compile_frame, Declined
        try:
            from veda_core.context import try_current
            ctx = try_current()
            scope = [str(s) for s in (ctx.source_ids or (ctx.source_id,))] if ctx else []
        except Exception:
            scope = []
        comp = compile_frame(g, sm, source_scope=scope)
        if isinstance(comp, Declined):
            tr["compile"] = {"declined": comp.slot, "reason": comp.reason}
            return _agent_or(FramePathResult("clarify", f"declined:{comp.slot}",
                                             message=(f"I understood the question but can't express part of it "
                                                      f"({comp.reason}). Could you rephrase that part?"),
                                             frame=fr, grounded=g, anchor=g.anchor, trace=tr),
                             query, sm, vocab, fr, f"declined:{comp.slot}", tr)
        # entity coverage: a NAME-grounded entity must be in the statement
        if g.anchor not in comp.tables:
            tr["compile"] = {"refused": "entity_coverage"}
            return _agent_or(FramePathResult("clarify", "entity_coverage",
                                             message="I could not build a query over the records you named.",
                                             frame=fr, grounded=g, anchor=g.anchor, trace=tr),
                             query, sm, vocab, fr, "entity_coverage", tr)
        tr["compile"] = {"sql": comp.sql, "head": comp.ir.head}
        tr["total_ms"] = round((time.time() - t0) * 1000.0, 1)
        compiled = FramePathResult("sql", "compiled", sql=comp.sql, ir=comp.ir, anchor=g.anchor,
                                   tables=comp.tables, columns=comp.columns, frame=fr, grounded=g,
                                   notes=list(g.notes), trace=tr,
                                   source_id=(str(g.source_id) if _narrow_saved is not None else None))
        snap = _snapped(fr, g, comp, vocab, query) if _agent_enabled() else None
        if snap:
            # the compiled shape is narrower than the question: an entity the question
            # names is not in the statement, or a grouping key was dropped. The agent
            # plans it; when it cannot, the compiled answer stands (the fast path stays).
            return _agent_or(compiled, query, sm, vocab, fr, f"snap:{snap}", tr, keep_on_fail=True)
        return compiled
    except Exception as e:
        tr["exception"] = f"{type(e).__name__}: {str(e)[:200]}"
        return FramePathResult("degrade", "exception", trace=tr)
    finally:
        if _wide_tok is not None:
            _WIDE.reset(_wide_tok)
        _restore(_narrow_saved)


# ── router primary, scope narrowing ──────────────────────────────────────────────────
class _RouterPrimary:
    """The router's primary table, resolved at most once and only when asked: a MODEL /
    SYNTHETIC / RETRIEVAL anchor needs it (agreement), a NAME-grounded one never does.
    Callable; the grounding's RETRIEVAL fallback calls it too. Mapped onto this scope's
    card keys (a multi-source scope keys colliding tables `src{N}.t`)."""

    def __init__(self, given, hint, vocab, tr):
        self._given, self._hint, self._vocab, self._tr = given, hint, vocab, tr
        self._done, self._value = False, None

    def __call__(self) -> Optional[str]:
        if self._done:
            return self._value
        self._done = True
        p, scores, err, t0 = self._given, None, None, time.time()
        if p is None and self._hint is not None:
            try:
                p, scores = self._hint()
            except Exception as e:
                p, err = None, f"{type(e).__name__}: {str(e)[:160]}"
        cards = getattr(self._vocab, "cards", None) or {}
        if p and p not in cards:
            same = [k for k, c in cards.items() if c.get("_bare_table") == p]
            if len(same) == 1:
                p = same[0]
        self._value = p
        self._tr["router"] = {"primary": p, "scores": scores,
                              "ms": round((time.time() - t0) * 1000.0, 1),
                              **({"error": err} if err else {})}
        return p


def _question_evidence(anchor: str, query: str, sm) -> Optional[str]:
    """Why the question points at `anchor` on its own: typed anchor evidence (entity /
    value) at the fast path's floor, or a column of the anchor named in full. None → none."""
    bare = anchor.split(".", 1)[1] if anchor.startswith("src") and "." in anchor else anchor
    try:
        from config import QSR_FP_EVIDENCE_FLOOR
        from query.resolution import typed_anchor_evidence
        ev, _ = typed_anchor_evidence(query, sm)
        if max(float(ev.get(anchor, 0.0) or 0.0), float(ev.get(bare, 0.0) or 0.0)) >= QSR_FP_EVIDENCE_FLOOR:
            return "typed"
    except Exception:
        pass
    import re as _re
    qt = set(_re.findall(r"[a-z0-9]+", query.lower()))
    for k in ((sm or {}).get("columns") or {}):
        t, _, c = k.rpartition(".")
        if t not in (anchor, bare):
            continue
        toks = [x for x in c.lower().split("_") if len(x) > 2 and x != "id"]
        if toks and all(x in qt or x + "s" in qt for x in toks):
            return f"column:{c}"
    return None


def _scope_ids() -> List[str]:
    try:
        from veda_core.context import try_current
        c = try_current()
        return [str(s) for s in (c.source_ids or (c.source_id,))] if c else []
    except Exception:
        return []


def _sources_of(tables, vocab, sm) -> set:
    """The sources owning `tables` (a card's source, else the scope model's `_source_id`;
    None for a table neither knows)."""
    tabs = (sm or {}).get("tables") or {}
    out = set()
    for t in tables:
        s = (getattr(vocab, "source_of", None) or {}).get(t) or (tabs.get(t) or {}).get("_source_id")
        out.add(str(s) if s is not None else None)
    return out


def _narrow(source_id):
    """Narrow the ambient request scope to one source (RequestContext.narrowed carries
    RBAC allowed_resources and cache_back). Returns the saved context for _restore, or
    None when nothing changed."""
    try:
        from veda_core.context import try_current, set_context
        c = try_current()
        if c is None or source_id is None:
            return None
        if str(c.source_id) == str(source_id) and len(c.source_ids or ()) <= 1:
            return None
        set_context(c.narrowed(int(source_id)))
        return c
    except Exception:
        return None


#: the caller's full scope while run_frame_path has narrowed the ambient one to its anchor's
#: source — the planner agent plans over the full scope (_restore_wide)
_WIDE: "ContextVar[Any]" = ContextVar("veda_frame_wide_ctx", default=None)


def _restore_wide():
    """Re-bind the full scope for the duration of an agent run; returns the narrowed
    context to _restore afterwards (None when nothing was narrowed)."""
    wide = _WIDE.get()
    if wide is None:
        return None
    try:
        from veda_core.context import try_current, set_context
        cur = try_current()
        set_context(wide)
        return cur
    except Exception:
        return None


def _restore(saved) -> None:
    if saved is None:
        return
    try:
        from veda_core.context import set_context
        set_context(saved)
    except Exception:
        pass


@contextmanager
def narrowed_scope(source_id):
    """`with narrowed_scope(sid):` — the block runs on that one source; the caller's scope
    is restored on exit, whatever happens."""
    saved = _narrow(source_id)
    try:
        yield
    finally:
        _restore(saved)


#: > 0 while the sub-questions of an agent `split` run: a sub-question the agent would split
#: again keeps its frame-path result instead (no recursive decomposition).
_SPLIT_DEPTH: "ContextVar[int]" = ContextVar("veda_agent_split_depth", default=0)


def enter_split():
    return _SPLIT_DEPTH.set(_SPLIT_DEPTH.get() + 1)


def exit_split(token) -> None:
    try:
        _SPLIT_DEPTH.reset(token)
    except Exception:
        _SPLIT_DEPTH.set(max(0, _SPLIT_DEPTH.get() - 1))


def _doc_sources(scope: List[str]) -> List[str]:
    """The sources in `scope` that hold document chunks — where a `rag` plan is answered."""
    if not scope:
        return []
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT source_id::text FROM graph_nodes WHERE node_type='chunk' "
                            "AND source_id::text = ANY(%s)", [list(scope)])
                have = {r[0] for r in cur.fetchall()}
        finally:
            release_internal_connection(conn)
        return [s for s in scope if s in have]
    except Exception:
        return []


def _agent_wall() -> Optional[float]:
    """The agent's wall budget: AGENT_PART_BUDGET_S, capped by what is left of this part's
    SLM deadline (a compound part's budget, so the frame time already spent is counted)
    minus AGENT_TAIL_RESERVE_S for execution + the summary. None → too little left to plan."""
    try:
        import config
        wall = float(getattr(config, "AGENT_PART_BUDGET_S", 90.0))
        reserve = float(getattr(config, "AGENT_TAIL_RESERVE_S", 10.0))
        floor = float(getattr(config, "AGENT_MIN_WALL_S", 8.0))
    except Exception:
        wall, reserve, floor = 90.0, 10.0, 8.0
    try:
        from slm._call_slm import slm_deadline
        dl = slm_deadline.get()
    except Exception:
        dl = None
    if dl is not None:
        wall = min(wall, dl - time.time() - reserve)
    return wall if wall >= floor else None


# ── the planner agent (veda/agent/, AGENT_PLANNER_ENABLED) ────────────────────────────
def _agent_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "AGENT_PLANNER_ENABLED", False))
    except Exception:
        return False


def _snapped(fr: Frame, g: GroundedFrame, comp, vocab, question: str = "") -> Optional[str]:
    """Why the compiled statement is narrower than the question, or None:
      * a secondary entity the frame names maps to a card that is not in the statement;
      * the question NAMES two or more entities (strong card-name hits, the same
        name_hits the grounding uses) and one of them is not in the statement;
      * the frame asks for ≥ 2 grouping keys and fewer were grounded."""
    try:
        from veda.understanding.vocabulary import norm_phrase
        from veda.understanding.frame_grounding import name_hits
        idx = vocab.name_index()
        for sec in fr.secondaries or []:
            hits = [t for t, _k in (idx.get(norm_phrase(sec)) or [])]
            if hits and not any(t in comp.tables for t in hits):
                return f"entity:{sec}"
        if question:
            named = {h["table"] for h in name_hits(vocab, question)
                     if h.get("strong") and not h.get("modifier")}
            if len(named) >= 2 and any(t not in comp.tables for t in named):
                return "entities:" + ",".join(sorted(t for t in named if t not in comp.tables))
        if len(fr.group_by or []) >= 2 and len(g.group_by or []) < len(fr.group_by):
            return f"group_by:{len(g.group_by or [])}/{len(fr.group_by)}"
    except Exception:
        return None
    return None


def _agent_grounded(plan, fr: Optional[Frame]) -> GroundedFrame:
    """The plan as a GroundedFrame — the narration context and interpretation notes the
    pipeline reads off a frame answer (filters, order, measure, notes)."""
    from veda.understanding.frame_grounding import GFilter
    from veda.agent.plan import split_ref
    order = None
    if plan.order:
        t, c = split_ref(plan.order[0]["expr"])
        order = (t or plan.anchor, c, plan.order[0]["dir"])
    agg = plan.aggregates[0] if plan.aggregates else None
    return GroundedFrame(
        frame=fr or Frame(), anchor=plan.anchor, anchor_method="AGENT", source_id=plan.source_id,
        measure=((("count" if agg["fn"] == "count_distinct" else agg["fn"]), agg.get("column")) if agg else None),
        filters=[GFilter(table=f["table"], column=f["column"], op=f["op"], value=f.get("value"),
                         grounding="agent") for f in plan.filters],
        group_by=[(x["table"], x["column"]) for x in plan.group_by], order=order, limit=plan.limit,
        distinct=plan.distinct, projection=[(x["table"], x["column"]) for x in plan.projection],
        notes=[], evidence={"agent": plan.evidence}, confidence=plan.confidence)


def _agent_or(res: FramePathResult, query: str, sm, vocab, fr: Optional[Frame], why: str,
              tr: Dict[str, Any], keep_on_fail: bool = False) -> FramePathResult:
    """Try the planner agent where the frame path would clarify / decline / snap. An agent
    plan that validates and compiles replaces `res`; an agent clarify replaces a frame
    clarify only when it carries data (probe counts / budget); anything else keeps `res`."""
    if not _agent_enabled():
        return res
    wall = _agent_wall()
    if wall is None:
        tr["agent"] = {"trigger": why, "skipped": "no_time_left_in_part_budget"}
        print(f"  [Agent] {why} → skipped (part budget spent)")
        return res
    try:
        from veda.agent.planner import run_planner
        # the request's FULL scope (the frame path may have narrowed the ambient one to the
        # anchor's source): in a multi-source scope the agent may plan on any source in it
        scope = list(tr.get("scope_ids") or _scope_ids())
        narrowed = _restore_wide()
        try:
            ar = run_planner(query, sm, vocab, source_scope=scope, wall_s=wall)
        finally:
            _restore(narrowed)
        tr["agent"] = {"trigger": why, **ar.trace()}
        _publish_agent(tr["agent"])
        print(f"  [Agent] {why} → {ar.kind} ({ar.reason}) steps={ar.budget.get('steps')} "
              f"calls={ar.budget.get('tool_calls')} {ar.budget.get('wall_s')}s")
        if ar.kind == "sql" and ar.compiled is not None:
            comp, plan = ar.compiled, ar.plan
            sid, refused = _plan_source(comp, plan, scope, vocab, sm)
            if refused:
                tr["agent"]["refused"] = refused
                return res
            if sid:
                tr["agent"]["source_id"] = sid
            g = _agent_grounded(plan, fr)
            return FramePathResult("sql", f"agent:{why}", sql=comp.sql, ir=comp.ir, anchor=plan.anchor,
                                   tables=comp.tables, columns=comp.columns, frame=fr, grounded=g,
                                   notes=[], trace=tr, source_id=sid)
        if ar.kind == "split" and len(ar.parts) >= 2:
            # the question is several questions: each sub-question runs as its own part
            # (veda_hybrid expands it into the MultiResult) — never one inside another
            if _SPLIT_DEPTH.get() > 0:
                tr["agent"]["refused"] = "split_inside_split"
                return res
            tr["agent"]["consumed"] = "split"
            return FramePathResult("split", f"agent_split:{why}", parts=list(ar.parts), frame=fr,
                                   message=("This asks " + str(len(ar.parts)) + " things: "
                                            + "; ".join(ar.parts)),
                                   trace=tr)
        if ar.kind == "rag" and res.kind != "sql":
            # a document question: answered by the RAG layer over the scope's document
            # sources (a compiled frame answer is never replaced by one)
            docs = _doc_sources(scope)
            if not docs:
                tr["agent"]["refused"] = "rag_without_document_source"
                return res
            tr["agent"]["consumed"] = "rag"
            return FramePathResult("rag", f"agent_rag:{why}", rag_sources=docs, frame=fr, trace=tr)
        if ar.kind == "clarify" and ar.message and res.kind == "clarify" and ar.reason in ("probe",):
            return FramePathResult("clarify", f"agent:{ar.reason}", message=ar.message, frame=fr,
                                   anchor=res.anchor, trace=tr)
    except Exception as e:
        tr["agent"] = {"trigger": why, "exception": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"  [Agent] {why} failed: {type(e).__name__}: {str(e)[:200]}")
    return res


def _plan_source(comp, plan, scope: List[str], vocab, sm):
    """(source the plan's statement runs on, refusal reason). A single-source scope → (None,
    None): the primary, as always. In a multi-source scope a plan on ANY in-scope source runs
    there (the pipeline narrows the request scope to it); one whose tables span sources
    cannot run on one DB — federation's job."""
    if len(scope) <= 1:
        return None, None
    srcs = _sources_of(comp.tables, vocab, sm) | ({str(plan.source_id)} if plan.source_id else set())
    if len(srcs) != 1 or None in srcs or next(iter(srcs)) not in scope:
        return None, f"plan_spans_sources:{sorted(str(x) for x in srcs)}"
    return next(iter(srcs)), None


def _agent_follow_up(query: str, sm, tr: Dict[str, Any]) -> Optional[FramePathResult]:
    """A chat follow-up to an AGENT-planned turn: the remembered plan is the draft and its
    tool log is attached, so "of those, by city" is a one- or two-step plan edit, not a
    re-plan. None → not such a turn (the caller continues as usual); a degrade → the
    existing chain answers."""
    if not _agent_enabled():
        return None
    try:
        from veda_core.context import current_conversation_context
        conv = current_conversation_context() or {}
    except Exception:
        conv = {}
    draft = conv.get("agent_plan") if isinstance(conv, dict) else None
    if not isinstance(draft, dict) or not draft.get("tables"):
        return None
    try:
        from veda.understanding.vocabulary import scope_vocab
        from veda.agent.planner import run_planner
        vocab = scope_vocab(sm)
        q = conv.get("user_message") or query
        try:
            from veda_core.context import try_current
            _c = try_current()
            scope = [str(x) for x in (_c.source_ids or (_c.source_id,))] if _c else []
        except Exception:
            scope = []
        wall = _agent_wall()
        if wall is None:
            tr["agent"] = {"trigger": "follow_up", "skipped": "no_time_left_in_part_budget"}
            return FramePathResult("degrade", "agent_follow_up:no_budget", trace=tr)
        ar = run_planner(q, sm, vocab, draft=draft, prior_log=list(conv.get("agent_log") or []),
                         draft_question=conv.get("agent_question"), source_scope=scope, wall_s=wall)
        tr["agent"] = {"trigger": "follow_up", **ar.trace()}
        _publish_agent(tr["agent"])
        print(f"  [Agent] follow_up → {ar.kind} ({ar.reason}) steps={ar.budget.get('steps')}")
        if ar.kind == "sql" and ar.compiled is not None:
            comp, plan = ar.compiled, ar.plan
            sid, refused = _plan_source(comp, plan, scope, vocab, sm)
            if refused:
                tr["agent"]["refused"] = refused
                return FramePathResult("degrade", "agent_follow_up:" + refused.split(":")[0], trace=tr)
            return FramePathResult("sql", "agent:follow_up", sql=comp.sql, ir=comp.ir, anchor=plan.anchor,
                                   tables=comp.tables, columns=comp.columns, frame=None,
                                   grounded=_agent_grounded(plan, None), trace=tr, source_id=sid)
        if ar.kind == "clarify" and ar.message:
            return FramePathResult("clarify", f"agent:{ar.reason}", message=ar.message, trace=tr)
        return FramePathResult("degrade", f"agent_follow_up:{ar.kind}", trace=tr)
    except Exception as e:
        tr["agent"] = {"trigger": "follow_up", "exception": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"  [Agent] follow_up failed: {type(e).__name__}: {str(e)[:200]}")
        return FramePathResult("degrade", "agent_follow_up:exception", trace=tr)


def _draft_of(plan) -> Dict[str, Any]:
    """A Plan back in the model's compact form — the next turn's DRAFT."""
    from veda.agent.plan import split_ref  # noqa: F401
    refs = {a.get("alias"): a.get("ref") for a in plan.aggregates}
    d: Dict[str, Any] = {"tables": list(plan.tables)}
    if plan.joins:
        d["joins"] = list(dict.fromkeys(j["via"] for j in plan.joins))
    if plan.projection:
        d["select"] = [f"{p['table']}.{p['column']}" for p in plan.projection]
    if plan.filters:
        d["filters"] = [{"col": f"{f['table']}.{f['column']}", "op": f["op"], "value": f.get("value")}
                        for f in plan.filters]
    if plan.group_by:
        d["group_by"] = [f"{g['table']}.{g['column']}" for g in plan.group_by]
    if plan.aggregates:
        d["aggregates"] = [{"fn": a["fn"], "col": (f"{a['table']}.{a['column']}" if a.get("column") else "*")}
                           for a in plan.aggregates]
    if plan.order:
        d["order"] = [{"by": refs.get(o["expr"], o["expr"]), "dir": o["dir"]} for o in plan.order]
    if plan.limit:
        d["limit"] = plan.limit
    if plan.distinct:
        d["distinct"] = True
    if plan.time:
        w = plan.time.get("window") or {}
        d["time"] = {"col": f"{plan.time['table']}.{plan.time['column']}", "from": w.get("from"), "to": w.get("to")}
    return d


def _publish_agent(section: Dict[str, Any]) -> None:
    """The `agent` explain section (explain._SECTIONS) — steps, tool calls, final plan,
    validation, budget."""
    try:
        from veda.explain import current_trace
        current_trace().set("agent", **section)
    except Exception:
        pass


def _decide(g, fr, query, vocab, sm, router_primary, tr, stats) -> FramePathResult:
    """Authority rules over a grounding result."""
    if g is None:
        return FramePathResult("degrade", "no_entity", frame=fr)
    if isinstance(g, FrameClarify):
        ev = g.evidence or {}
        method = ev.get("anchor_method")
        named = bool(ev.get("name_hits"))
        tr["grounding"] = {"clarify": g.slot, "message": g.message, "candidates": g.candidates,
                           "anchor": ev.get("anchor"), "anchor_method": method,
                           "all": ev.get("all"), "name_hits": ev.get("name_hits")}
        if (g.slot == "entity" and named) or method in AUTHORITATIVE:
            return FramePathResult("clarify", f"slot:{g.slot}", message=g.message, frame=fr,
                                   anchor=ev.get("anchor"))
        return FramePathResult("degrade", f"unauthoritative_clarify:{g.slot}", frame=fr)
    tr["grounding"] = _g_summary(g)
    if g.anchor_method not in AUTHORITATIVE:
        # an advisory anchor stands only when the router's primary agrees with it
        rp = router_primary() if callable(router_primary) else router_primary
        tr["grounding"]["router_agrees"] = (g.anchor == rp)
        if g.anchor != rp:
            return FramePathResult("degrade", f"advisory_anchor:{g.anchor_method}", frame=fr, grounded=g)
        # agreement is only a second opinion when the router had something to go on: on a
        # one-table source (or any question that names nothing) the router's primary is
        # simply the only / likeliest table, and "how many gadgets are there" would count
        # amenities. The question must give the anchor typed evidence or name one of its
        # columns — the fast path's own evidence guard.
        ev = _question_evidence(g.anchor, query, sm)
        tr["grounding"]["router_evidence"] = ev
        if not ev:
            return FramePathResult("degrade", f"advisory_anchor_unevidenced:{g.anchor_method}",
                                   frame=fr, grounded=g)
    return FramePathResult("sql", "grounded", frame=fr, grounded=g, anchor=g.anchor)
