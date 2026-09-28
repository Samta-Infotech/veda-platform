"""veda.agent.plan — the plan object and its contract (planner agent, part B).

    Plan {kind, source_id, tables, joins, projection, filters, group_by, aggregates,
          order, limit, distinct, time, evidence, confidence, open_questions}

`validate(plan, log)` rejects every table / column / join edge / filter value that is not
present in a tool result of THIS run (`ToolBox.log`) — the model can only plan over what
it retrieved. `compile_plan` then goes through the EXISTING builders:

    joins        → query.join_planner.build_skeleton         (FROM / JOIN / ON)
    filters      → veda.understanding.frame_compiler.predicate (the frame path's literals,
                   parameterised later by validation.validate_and_parameterize)
    scalar agg   → veda.planning.build_aggregate_sql         (single table, no joins)
    grouped agg / projection → one SELECT over the skeleton  (as frame_compiler does)

and returns the frame compiler's `Compiled` (a COMPLETE QueryIR, head "frame.agent") or
its `Declined`. No new SQL builder, no SQL from the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from veda.ir import QueryIR, IRFilter, IRMeasure

AGG_FNS = ("count", "count_distinct", "sum", "avg", "min", "max")
OPS = ("=", "!=", ">", ">=", "<", "<=", "between", "in", "is_null", "is_not_null")
_NUMERIC_KINDS = {"MONETARY", "METRIC"}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def _bare(t: str) -> str:
    return t.split(".", 1)[1] if t.startswith("src") and "." in t else t


def split_ref(ref: str, tables: Optional[List[str]] = None) -> Tuple[Optional[str], str]:
    """'table.column' → (table, column). A scope key may itself contain a dot
    (`src2.assets_asset`), so the COLUMN is the last dotted part."""
    s = str(ref or "").strip()
    if "." not in s:
        return None, s
    t, _, c = s.rpartition(".")
    return t, c


@dataclass
class Plan:
    kind: str = "sql"                                     # sql | rag | tabular | federated
    source_id: Optional[str] = None
    tables: List[str] = field(default_factory=list)       # tables[0] is the grain (FROM)
    joins: List[Dict[str, Any]] = field(default_factory=list)       # {from, to, on:[(c1,c2)], via}
    projection: List[Dict[str, Any]] = field(default_factory=list)  # {table, column, alias?}
    filters: List[Dict[str, Any]] = field(default_factory=list)     # {table, column, op, value}
    group_by: List[Dict[str, Any]] = field(default_factory=list)    # {table, column}
    aggregates: List[Dict[str, Any]] = field(default_factory=list)  # {fn, table, column, alias}
    order: List[Dict[str, Any]] = field(default_factory=list)       # {expr, dir}
    limit: Optional[int] = None
    distinct: bool = False
    time: Optional[Dict[str, Any]] = None                 # {table, column, window:{from,to}}
    evidence: Dict[str, Any] = field(default_factory=dict)   # {tool_call_ids, slots}
    confidence: float = 0.0
    open_questions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def anchor(self) -> Optional[str]:
        return self.tables[0] if self.tables else None


# ── what a run's tool log licenses ────────────────────────────────────────────────────
@dataclass
class Seen:
    tables: Set[str] = field(default_factory=set)
    columns: Set[Tuple[str, str]] = field(default_factory=set)
    kinds: Dict[Tuple[str, str], str] = field(default_factory=dict)
    routes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)       # route id → its edges
    values: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    probed: Dict[Tuple[str, str, str, str], int] = field(default_factory=dict)   # (t,c,op,val) → rows
    phrases: Dict[Tuple[str, str, str], List[str]] = field(default_factory=dict)  # (t,c,val) → phrases
    call_of: Dict[Any, str] = field(default_factory=dict)                        # identifier → call id


def route_edges(route: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A join_path route {id, path: ["child.fk=parent.pk", …], cards?, basis?} → its edges."""
    cards = list(route.get("cards") or [])
    out = []
    for i, hop in enumerate(route.get("path") or []):
        left, _, right = str(hop).partition("=")
        ft, fc = split_ref(left)
        tt, tc = split_ref(right)
        out.append({"source_table": ft, "source_column": fc, "target_table": tt, "target_column": tc,
                    "cardinality": cards[i] if i < len(cards) else "N:1",
                    "kind": route.get("basis") or "declared", "route": route.get("id")})
    return out


def _vkey(v) -> str:
    if isinstance(v, (list, tuple)):
        return "|".join(_vkey(x) for x in v)
    return str(v).strip().lower()


def seen_identifiers(log: List[Dict[str, Any]]) -> Seen:
    """Every identifier a tool RESULT of this run contains (arguments alone never count)."""
    s = Seen()

    def tab(t, cid):
        if t:
            s.tables.add(t)
            s.call_of.setdefault(("t", t), cid)

    def col(t, c, cid, kind=None):
        if t and c:
            s.columns.add((t, c))
            s.call_of.setdefault(("c", t, c), cid)
            if kind:
                s.kinds.setdefault((t, c), kind)

    for e in log or []:
        r, cid, tool = e.get("result") or {}, e.get("id"), e.get("tool")
        if not isinstance(r, dict) or r.get("error"):
            continue
        if tool == "find_entities":
            for x in r.get("entities") or []:
                tab(x.get("table"), cid)
        elif tool == "describe":
            t = r.get("table")
            tab(t, cid)
            if r.get("business_date"):
                col(t, r["business_date"], cid, "TEMPORAL")
            if r.get("lifecycle"):
                col(t, r["lifecycle"].get("column"), cid, "CATEGORY")
                for v in r["lifecycle"].get("values") or []:
                    s.values.setdefault((t, r["lifecycle"].get("column")), set()).add(_vkey(v))
            for m in r.get("measures") or []:
                col(t, m.get("column"), cid, "MONETARY")
            for d in r.get("dimensions") or []:
                col(t, d, cid, "CATEGORY")
            if r.get("display"):
                col(t, r["display"], cid)
            if r.get("key"):
                col(t, r["key"], cid, "IDENTIFIER")
            for p in r.get("parents") or []:
                tab(p.get("table"), cid)
                col(t, p.get("via"), cid, "IDENTIFIER")
            for ch in r.get("children") or []:
                tab(ch.get("table"), cid)
                col(ch.get("table"), ch.get("via"), cid, "IDENTIFIER")
        elif tool == "columns":
            t = r.get("table")
            for x in r.get("columns") or []:
                col(t, x.get("column"), cid, x.get("kind"))
        elif tool == "values":
            t, c = r.get("table"), r.get("column")
            col(t, c, cid, r.get("kind"))
            vs = s.values.setdefault((t, c), set())
            for x in r.get("values") or []:
                vs.add(_vkey(x.get("value")))
                if x.get("phrases"):
                    s.phrases.setdefault((t, c, _vkey(x.get("value"))), []).extend(x["phrases"])
            if r.get("match"):
                vs.add(_vkey(r["match"].get("value")))
        elif tool == "retrieve":
            for x in r.get("tables") or []:
                tab(x.get("table"), cid)
            for x in r.get("columns") or []:
                tab(x.get("table"), cid)
                col(x.get("table"), x.get("column"), cid, x.get("kind"))
        elif tool == "join_path":
            for o in r.get("routes") or []:
                es = route_edges(o)
                s.routes[o["id"]] = es
                s.call_of.setdefault(("e", o["id"]), cid)
                for ed in es:
                    tab(ed["source_table"], cid)
                    tab(ed["target_table"], cid)
                    col(ed["source_table"], ed["source_column"], cid, "IDENTIFIER")
                    col(ed["target_table"], ed["target_column"], cid, "IDENTIFIER")
        elif tool == "probe":
            for f in r.get("filters") or []:
                t, c = split_ref(f.get("col"))
                s.probed[(t, c, str(f.get("op") or "="), _vkey(f.get("value")))] = int(f.get("rows") or 0)
                s.call_of.setdefault(("p", t, c), cid)
    return s


# ── validation ────────────────────────────────────────────────────────────────────────
def validate(plan: Plan, log: List[Dict[str, Any]], sm=None, *, question: Optional[str] = None,
             vocab=None) -> List[str]:
    """[] = valid. Every error names the offending slot and identifier. With `question`,
    every filter must also be LICENSED by the user's words (see `licensed`)."""
    errs: List[str] = []
    s = seen_identifiers(log)
    if plan.kind == "rag":
        return errs
    if not plan.tables:
        return ["tables: empty"]
    for t in plan.tables:
        if t not in s.tables:
            errs.append(f"table {t} was not returned by any tool")
    tabs = set(plan.tables)
    # joins: every edge must be an edge of a route a join_path call returned, verbatim
    for j in plan.joins:
        route = s.routes.get(j.get("via"))
        if route is None:
            errs.append(f"join {j.get('via')} was not returned by join_path")
            continue
        on = [tuple(p) for p in (j.get("on") or [])]
        match = [e for e in route if {j.get("from"), j.get("to")} == {e["source_table"], e["target_table"]}]
        if not match:
            errs.append(f"join {j.get('via')} tables differ from join_path's")
        elif on and not any(on == [(e["source_column"], e["target_column"])] for e in match):
            errs.append(f"join {j.get('via')} keys {on} differ from join_path's")

    def colok(t, c, slot):
        if t not in tabs:
            errs.append(f"{slot}: {t}.{c} is on a table not in the plan")
        elif (t, c) not in s.columns:
            errs.append(f"{slot}: column {t}.{c} was not returned by any tool")

    for p in plan.projection:
        colok(p.get("table"), p.get("column"), "projection")
    for g in plan.group_by:
        colok(g.get("table"), g.get("column"), "group_by")
    for a in plan.aggregates:
        if a.get("fn") not in AGG_FNS:
            errs.append(f"aggregate: unknown fn {a.get('fn')}")
        if a.get("column") not in (None, "*"):
            colok(a.get("table"), a.get("column"), "aggregate")
            if a.get("fn") in ("sum", "avg") and not _numeric(s, sm, a.get("table"), a.get("column")):
                errs.append(f"aggregate: {a['fn']} over non-numeric {a.get('table')}.{a.get('column')}")
        elif a.get("fn") in ("sum", "avg", "min", "max"):
            errs.append(f"aggregate: {a.get('fn')} needs a column")
    if plan.time:
        colok(plan.time.get("table"), plan.time.get("column"), "time")
        for k, v in (plan.time.get("window") or {}).items():
            if v is not None and not _ISO_DATE.match(str(v)):
                errs.append(f"time: {k} {v!r} is not a YYYY-MM-DD date")
    for o in plan.order:
        t, c = split_ref(o.get("expr"))
        if t is not None:
            colok(t, c, "order")
        elif c not in {a.get("alias") for a in plan.aggregates}:
            errs.append(f"order: {o.get('expr')} is neither a column nor an aggregate")
    for f in plan.filters:
        t, c, op, v = f.get("table"), f.get("column"), f.get("op"), f.get("value")
        colok(t, c, "filter")
        if op not in OPS:
            errs.append(f"filter: unknown op {op}")
            continue
        if op in ("is_null", "is_not_null"):
            continue
        kind = s.kinds.get((t, c)) or _kind_sm(sm, t, c)
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if kind in _NUMERIC_KINDS or kind == "TEMPORAL" or op in (">", ">=", "<", "<=", "between"):
            if not any(k[:3] == (t, c, op) for k in s.probed):
                errs.append(f"filter: {t}.{c} {op} {v} was never probed")
            continue
        known = s.values.get((t, c), set())
        for x in vals:
            if _vkey(x) in known:
                continue
            rows = s.probed.get((t, c, op, _vkey(v)))
            if rows is None and op == "in":
                rows = s.probed.get((t, c, "=", _vkey(x)))
            if not rows:
                errs.append(f"filter: value {x!r} for {t}.{c} came from no values() result "
                            f"and no probe confirmed it")
    # every table reachable from the grain through the plan's joins
    if len(tabs) > 1:
        adj: Dict[str, Set[str]] = {}
        for j in plan.joins:
            adj.setdefault(j.get("from"), set()).add(j.get("to"))
            adj.setdefault(j.get("to"), set()).add(j.get("from"))
        reach, stack = {plan.anchor}, [plan.anchor]
        while stack:
            n = stack.pop()
            for m in adj.get(n, ()):
                if m not in reach:
                    reach.add(m)
                    stack.append(m)
        for t in tabs - reach:
            errs.append(f"table {t} is not joined to {plan.anchor}")
    fo = fanout(plan, s)
    if fo:
        errs.append(fo)
    if question:
        nums = sorted(question_numbers(question))
        for why in unlicensed(plan, question, s, vocab, sm):
            if why.startswith("#"):
                why = why[1:]
                errs.append(f"filter: {why} — that number is not in the question"
                            + (f" (it states {', '.join(_fmt(n) for n in nums)}); use the question's number"
                               if nums else "; remove it"))
            else:
                errs.append(f"filter: {why} — the question does not ask for it; remove it")
    # a grouped / aggregated answer can only be ordered by a group key or an aggregate
    if plan.aggregates:
        gk = {(g.get("table"), g.get("column")) for g in plan.group_by}
        for o in plan.order:
            t, c = split_ref(o.get("expr"))
            if t is not None and (t, c) not in gk:
                errs.append(f"order: {t}.{c} is neither a group_by column nor an aggregate — order by agg1")
    return errs


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


# ── licensing: a filter must come from the user's words ─────────────────────────────
def _has_window(q: str) -> bool:
    """The question names a time window (the frame path's temporal producer: an explicit
    period from query.temporal_parser; 'latest/recent' is an order, not a window)."""
    try:
        from veda.understanding.producers import p_temporal
        return p_temporal(q or "") is not None
    except Exception:
        return False


def asks_grouping(q: str) -> bool:
    """The question asks for a per-X breakdown (veda.planning's grouped-mode detectors)."""
    try:
        from veda.planning import grouped_mode, grouped_count_mode, superlative_mode
        return bool((grouped_mode(q) or grouped_count_mode(q)) and not superlative_mode(q))
    except Exception:
        return False
_NUM = re.compile(r"(?<![A-Za-z0-9_\-.])(\d[\d,]*(?:\.\d+)?)(?![\-_/][A-Za-z0-9])\s*(k|thousand|lakh|lakhs|lac|m|mn|million|cr|crore)?\b", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "m": 1e6, "mn": 1e6,
         "million": 1e6, "cr": 1e7, "crore": 1e7}
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "twenty": 20, "fifty": 50, "hundred": 100}


def question_numbers(q: str) -> Set[float]:
    out: Set[float] = set()
    for m in _NUM.finditer(q or ""):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.add(v)
        if m.group(2):
            out.add(v * _MULT.get(m.group(2).lower(), 1))
    for w, v in _WORDNUM.items():
        if re.search(rf"\b{w}\b", q or "", re.I):
            out.add(float(v))
    return out


def stated_numbers(q: str) -> Set[float]:
    """The numbers the question WRITES in digits, one per mention (with its multiplier
    applied: '5 lakh' is 500000) — what a plan must account for."""
    out: Set[float] = set()
    for m in _NUM.finditer(q or ""):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.add(v * _MULT.get((m.group(2) or "").lower(), 1) if m.group(2) else v)
    return out


def _norm(s: str) -> str:
    try:
        from veda.understanding.vocabulary import norm_phrase
        return norm_phrase(s)
    except Exception:
        return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


def unlicensed(plan: Plan, question: str, s: Optional[Seen] = None, vocab=None, sm=None) -> List[str]:
    """Filters the question's words do not license. A categorical value is licensed when
    the value itself, a glossary phrase for it (vocabulary or a values() result), or —
    for a yes/no column — the column's own words appear in the question; a number when
    the question writes it; a date when the question has a time expression."""
    s = s or Seen()
    qn = " " + _norm(question) + " "
    nums = question_numbers(question)
    phrases: Dict[Tuple[str, str, str], List[str]] = {}
    gl = (getattr(vocab, "value_glossary", None) or {}) if vocab is not None else {}
    out = []
    for f in plan.filters:
        t, c, op, v = f.get("table"), f.get("column"), f.get("op"), f.get("value")
        kind = s.kinds.get((t, c)) or _kind_sm(sm, t, c)
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if op in ("is_null", "is_not_null"):
            toks = [x for x in str(c).lower().split("_") if len(x) > 2 and x not in ("id", "is", "has")]
            if not any(f" {_norm(x)} " in qn for x in toks):
                out.append(f"{t}.{c} {op}")
            continue
        if kind == "TEMPORAL" or (isinstance(v, str) and _ISO_DATE.match(v)):
            if not _has_window(question):
                out.append(f"{t}.{c} {op} {v}")
            continue
        if kind in _NUMERIC_KINDS or _isnum(v):
            try:
                ok = all(float(x) in nums for x in vals)
            except Exception:
                ok = False
            if not ok:
                out.append(f"#{t}.{c} {op} {v}")
            continue
        g = gl.get(f"{t}.{c}") or {}
        for x in vals:
            words = [str(x)] + list(g.get(x) or g.get(str(x)) or [])
            words += _result_phrases(s, t, c, x)
            if any(w and f" {_norm(w)} " in qn for w in words):
                continue
            if str(x).lower() in ("true", "false", "t", "f", "yes", "no", "1", "0") or kind == "FLAG":
                ctoks = [y for y in str(c).lower().split("_") if y not in ("is", "has", "can", "was", "flag") and len(y) > 2]
                if ctoks and all(re.search(rf"\b{re.escape(y)}", (question or "").lower()) for y in ctoks):
                    continue
            out.append(f"{t}.{c} {op} {x}")
            break
    if plan.time and not _has_window(question):
        w = plan.time.get("window") or {}
        out.append(f"time window {plan.time.get('column')} {w.get('from')}..{w.get('to')}")
    return out


def licensing_phrases(plan: Plan, question: str, s: Optional[Seen] = None, vocab=None) -> List[str]:
    """The question's own words that licensed a categorical filter through a glossary
    phrase ('on the market' → status APPROVED). The lexical qualifier gate looks for the
    user's words in the SQL; these words are ACCOUNTED FOR by the filter value instead."""
    s = s or Seen()
    qn = " " + _norm(question) + " "
    gl = (getattr(vocab, "value_glossary", None) or {}) if vocab is not None else {}
    out = []
    for f in plan.filters:
        t, c, v = f.get("table"), f.get("column"), f.get("value")
        g = gl.get(f"{t}.{c}") or {}
        for x in (list(v) if isinstance(v, (list, tuple)) else [v]):
            for w in list(g.get(x) or g.get(str(x)) or []) + _result_phrases(s, t, c, x):
                if w and f" {_norm(w)} " in qn:
                    out.append(str(w))
    return out


def accounted_phrases(plan: Plan, question: str, log, vocab=None, sm=None) -> List[str]:
    """Every phrase of the question that an identifier the plan USES accounts for: the
    business name / plural / aliases of each table; the glossary / alias phrases of each
    column it selects, filters, groups, aggregates or orders by; the glossary phrases that
    licensed a filter value; and the time expression behind a window or date filter.
    The lexical qualifier gate then only asks about the words that remain."""
    s = seen_identifiers(log)
    out: List[str] = list(licensing_phrases(plan, question, s, vocab))
    cards = (getattr(vocab, "cards", None) or {}) if vocab is not None else {}
    mg = (getattr(vocab, "measure_glossary", None) or {}) if vocab is not None else {}
    for t in plan.tables:
        c = cards.get(t) or {}
        out += [c.get("business_name") or "", c.get("plural") or ""] + list(c.get("aliases") or [])
        for e in log or []:
            for x in ((e.get("result") or {}).get("entities") or []):
                if x.get("table") == t and x.get("business_name"):
                    out.append(x["business_name"])
    used = set()
    for x in plan.projection + plan.group_by + plan.filters:
        used.add((x.get("table"), x.get("column")))
    for a in plan.aggregates:
        if a.get("column"):
            used.add((a.get("table"), a.get("column")))
    for o in plan.order:
        used.add(split_ref(o["expr"]))
    if plan.time:
        used.add((plan.time.get("table"), plan.time.get("column")))
    cols = (sm or {}).get("columns") or {}
    for (t, c) in used:
        if not t or not c:
            continue
        out.append(c.replace("_", " "))
        out += list((mg.get(f"{t}.{c}") or {}).get("phrases") or [])
        out += list((cols.get(f"{t}.{c}") or {}).get("aliases") or [])
        for e in log or []:
            r = e.get("result") or {}
            if e.get("tool") == "columns" and r.get("table") == t:
                for x in r.get("columns") or []:
                    if x.get("column") == c:
                        out += list(x.get("phrases") or [])
            if e.get("tool") == "describe" and r.get("table") == t:
                for m in r.get("measures") or []:
                    if m.get("column") == c:
                        out += list(m.get("phrases") or [])
    if plan.time or any((s.kinds.get((f.get("table"), f.get("column"))) or _kind_sm(sm, f.get("table"), f.get("column")))
                        == "TEMPORAL" for f in plan.filters):
        try:
            from query.temporal_parser import run_temporal_parser
            out += list(run_temporal_parser(question).raw_expressions or [])
        except Exception:
            pass
    # a numeric comparison is accounted for with its comparator: the (up to two) words
    # right before a number the plan compares on ("above 4", "more than 10,000")
    nums = set()
    for f in plan.filters:
        if f.get("op") in (">", ">=", "<", "<=", "between", "=", "!=") and _isnum(f.get("value")):
            for x in (f["value"] if isinstance(f["value"], (list, tuple)) else [f["value"]]):
                nums.add(float(x))
    if nums:
        toks = re.findall(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?", question or "")
        for i, tk in enumerate(toks):
            if tk[0].isdigit():
                try:
                    v = float(tk.replace(",", ""))
                except ValueError:
                    continue
                if v in nums:
                    out += [" ".join(toks[max(0, i - 2):i]), toks[i - 1] if i else ""]
                    if i + 2 < len(toks) and toks[i + 2][:1].isdigit():
                        out.append(toks[i + 1])                  # "between 100 AND 500"
    qn = " " + _norm(question) + " "
    return sorted({w for w in out if w and f" {_norm(w)} " in qn}, key=len, reverse=True)


def _result_phrases(s: Seen, t, c, x) -> List[str]:
    return list((getattr(s, "phrases", {}) or {}).get((t, c, _vkey(x))) or [])


def license_limit(plan: Plan, question: str) -> None:
    """A row count the question does not state is not the question's (the page size is
    ours): keep a limit the question writes, or 1 for a singular superlative."""
    if plan.limit is None:
        return
    if float(plan.limit) in question_numbers(question):
        return
    if plan.limit == 1 and re.search(r"\b(which|what|who)\b[^?]*\b(most|highest|lowest|largest|smallest|"
                                     r"cheapest|biggest|best|worst|top|latest|oldest|newest|max|maximum|"
                                     r"min|minimum|least|fewest)\b", question or "", re.I) \
            and not re.search(r"\b(top|most|highest|lowest|latest|oldest|newest)\s+\d", question or "", re.I):
        return
    plan.notes.append(f"row limit {plan.limit} not in the question — dropped")
    plan.limit = None


def _kind_sm(sm, t, c) -> Optional[str]:
    meta = ((sm or {}).get("columns") or {}).get(f"{t}.{c}") or ((sm or {}).get("columns") or {}).get(f"{_bare(t)}.{c}") or {}
    st = str(meta.get("semantic_type") or "").upper()
    return {"FLAG": "CATEGORY", "BOOLEAN": "CATEGORY"}.get(st, st) or None


def _numeric(s: Seen, sm, t, c) -> bool:
    k = _kind_sm(sm, t, c) or s.kinds.get((t, c))
    if k in _NUMERIC_KINDS:
        return True
    meta = ((sm or {}).get("columns") or {}).get(f"{t}.{c}") or {}
    dt = str(meta.get("data_type") or meta.get("dtype") or "").lower()
    return any(x in dt for x in ("int", "numeric", "decimal", "float", "double", "real"))


def fanout(plan: Plan, s: Optional[Seen] = None) -> Optional[str]:
    """A SUM/AVG/COUNT(*) over a join that multiplies the measured rows. Rooted at the
    measured table, crossing an N:1 edge from its '1' side to its 'N' side fans out."""
    # COUNT(*) counts the GRAIN (the first table): joined to its children it would count
    # the children instead; SUM/AVG/COUNT of a column counts its own table's rows
    risky = [a for a in plan.aggregates if a.get("fn") in ("sum", "avg", "count")]
    if not risky or not plan.joins:
        return None
    edges = []
    for j in plan.joins:
        src, tgt = j.get("from"), j.get("to")
        card = str(j.get("cardinality") or "N:1")
        if card == "1:N":
            src, tgt = tgt, src
        edges.append((src, tgt, card))
    for a in risky:
        root = a.get("table") or plan.anchor
        seen, stack = {root}, [root]
        while stack:
            n = stack.pop()
            for (src, tgt, card) in edges:
                if card == "1:1":
                    nxt = tgt if src == n else src if tgt == n else None
                    fan = False
                elif tgt == n:
                    nxt, fan = src, True               # from the '1' side to the 'N' side
                elif src == n:
                    nxt, fan = tgt, False
                else:
                    continue
                if nxt is None or nxt in seen:
                    continue
                if fan:
                    return (f"aggregate {a.get('fn')}({a.get('column') or '*'}) would be multiplied by "
                            f"the join to {nxt} (one {n} has many {nxt}): if you count or measure {nxt} rows, "
                            f"make {nxt} the first table; to count {n} rows use count_distinct of {n}.id")
                seen.add(nxt)
                stack.append(nxt)
    return None


# ── model output → Plan ──────────────────────────────────────────────────────────────
def from_model(obj: Dict[str, Any], log: List[Dict[str, Any]], *, source_of=None) -> Plan:
    """The model's compact `final` object → Plan. Route ids become every edge of the
    join_path route they name (with the intermediate tables the route passes through);
    column refs 'table.col' are split; aggregates get stable aliases. Evidence: the call
    that returned each identifier."""
    s = seen_identifiers(log)
    obj = dict(obj or {})
    p = Plan(kind=str(obj.get("kind") or "sql"))
    p.tables = [str(t) for t in (obj.get("tables") or []) if t]
    have = set()
    for jid in dict.fromkeys(obj.get("joins") or []):
        route = s.routes.get(jid)
        if route is None:
            p.joins.append({"from": None, "to": None, "on": [], "via": jid})
            continue
        for e in route:
            k = (e["source_table"], e["source_column"], e["target_table"], e["target_column"])
            if k in have:
                continue
            have.add(k)
            p.joins.append({"from": e["source_table"], "to": e["target_table"],
                            "on": [(e["source_column"], e["target_column"])], "via": jid,
                            "kind": e.get("kind"), "cardinality": e.get("cardinality")})
            for t in (e["source_table"], e["target_table"]):
                if t and t not in p.tables:
                    p.tables.append(t)
                    p.notes.append(f"joined through {t}")
    for ref in obj.get("select") or []:
        t, c = split_ref(ref)
        p.projection.append({"table": t, "column": c})
    for ref in obj.get("group_by") or []:
        t, c = split_ref(ref)
        p.group_by.append({"table": t, "column": c})
    for i, a in enumerate(obj.get("aggregates") or []):
        fn = str(a.get("fn") or "count").lower()
        ref = a.get("col") or "*"
        t, c = split_ref(ref) if ref != "*" else (p.anchor, "*")
        if fn == "count_distinct" and c == "*":
            c = "id"                                  # COUNT(DISTINCT *) = distinct grain rows
        alias = "count" if (fn == "count" and c == "*") else f"{fn}_{c}"
        p.aggregates.append({"fn": fn, "table": t, "column": (None if c == "*" else c), "alias": alias,
                             "ref": f"agg{i + 1}"})
    aliases = {a["ref"]: a["alias"] for a in p.aggregates}
    agg_of = {f"{a['table']}.{a['column']}": a["alias"] for a in p.aggregates if a.get("column")}
    gkeys = {f"{g['table']}.{g['column']}" for g in p.group_by}
    for o in obj.get("order") or []:
        by = str(o.get("by") or "")
        if p.aggregates and by in agg_of and by not in gkeys:
            by = agg_of[by]            # "order by amount" in a SUM(amount) answer = by that sum
        p.order.append({"expr": aliases.get(by, by), "dir": ("asc" if str(o.get("dir")).lower() == "asc" else "desc")})
    for f in obj.get("filters") or []:
        t, c = split_ref(f.get("col"))
        p.filters.append({"table": t, "column": c, "op": str(f.get("op") or "="), "value": f.get("value")})
    tm = obj.get("time")
    if isinstance(tm, dict) and tm.get("col") and (tm.get("from") or tm.get("to")):
        t, c = split_ref(tm["col"])
        p.time = {"table": t, "column": c, "window": {"from": tm.get("from"), "to": tm.get("to")}}
    lim = obj.get("limit")
    try:
        p.limit = int(lim) if lim not in (None, "", 0) else None
    except Exception:
        p.limit = None
    p.distinct = bool(obj.get("distinct"))
    p.open_questions = [str(q) for q in (obj.get("open_questions") or []) if q]
    if source_of and p.tables:
        sids = {source_of(t) for t in p.tables}
        p.source_id = next(iter(sids)) if len(sids) == 1 else None
        if len(sids) > 1:
            p.kind = "federated"
    # evidence: which call justified each identifier
    ev: Dict[str, str] = {}
    for t in p.tables:
        ev[f"table:{t}"] = s.call_of.get(("t", t), "?")
    for j in p.joins:
        ev[f"join:{j['via']}"] = s.call_of.get(("e", j["via"]), "?")
    for slot, items in (("select", p.projection), ("group_by", p.group_by), ("filter", p.filters)):
        for x in items:
            ev[f"{slot}:{x.get('table')}.{x.get('column')}"] = s.call_of.get(("c", x.get("table"), x.get("column")), "?")
    p.evidence = {"tool_call_ids": sorted({v for v in ev.values() if v != "?"}, key=lambda x: int(x[1:])),
                  "slots": ev}
    return p


# ── compile (existing builders) ───────────────────────────────────────────────────────
def _q(ident: str) -> str:
    return '"' + str(ident).replace('"', '""') + '"'


def _skeleton(plan: Plan) -> Tuple[str, Dict[str, str], List[str]]:
    """FROM/JOIN via join_planner.build_skeleton. Returns (from_sql, table→first alias,
    tables that received more than one alias)."""
    from query.join_planner import build_skeleton
    anchor = _bare(plan.anchor)
    path = []
    for j in plan.joins:
        c1, c2 = (list(j.get("on") or []) or [(None, None)])[0]
        path.append({"source_table": _bare(j["from"]), "source_column": c1,
                     "target_table": _bare(j["to"]), "target_column": c2,
                     "requires_predicate": j.get("requires_predicate")})
    frm, amap = build_skeleton({"anchor": anchor, "join_path": path})
    t2a: Dict[str, str] = {}
    dupes: List[str] = []
    for a, t in sorted(amap.items(), key=lambda kv: int(kv[0][1:])):
        if t in t2a:
            dupes.append(t)
        t2a.setdefault(t, a)
    out = {}
    for t in plan.tables:
        if _bare(t) in t2a:
            out[t] = t2a[_bare(t)]
    return frm.replace("\n", " "), out, dupes


def _gfilter(f: Dict[str, Any], sm, kinds=None):
    from veda.understanding.frame_grounding import GFilter
    t, c, op, v = f["table"], f["column"], f["op"], f.get("value")
    kind = (kinds or {}).get((t, c)) or _kind_sm(sm, t, c)
    op2 = {"in": "IN", "is_null": "IS NULL", "is_not_null": "IS NOT NULL"}.get(op, op)
    if kind == "TEMPORAL":
        g = "temporal"
    elif kind in _NUMERIC_KINDS or (op in (">", ">=", "<", "<=", "between") and _isnum(v)):
        g = "numeric"
    else:
        g = "domain"
    if op2 == "IN" and not isinstance(v, (list, tuple)):
        v = [v]
    return GFilter(table=t, column=c, op=op2, value=v, grounding=g)


def _isnum(v) -> bool:
    try:
        if isinstance(v, (list, tuple)):
            return all(_isnum(x) for x in v)
        float(v)
        return True
    except Exception:
        return False


def where_parts(plan: Plan, t2a: Dict[str, str], sm, kinds=None):
    from veda.understanding.frame_compiler import predicate
    parts = []
    for f in plan.filters:
        g = _gfilter(f, sm, kinds)
        p = predicate(g, t2a.get(f["table"], "t0"))
        if p is None:
            return None, f"filter {f['column']} {f['op']} is not expressible"
        parts.append(p)
    if plan.time:
        a = t2a.get(plan.time["table"], "t0")
        col = f"{a}.{_q(plan.time['column'])}"
        w = plan.time.get("window") or {}
        from veda.understanding.frame_compiler import _lit
        if w.get("from"):
            parts.append(f"{col} >= {_lit(w['from'])}")
        if w.get("to"):
            parts.append(f"{col} <= {_lit(w['to'])}")
    return parts, None


def _agg_expr(a: Dict[str, Any], t2a) -> str:
    fn, c = a["fn"], a.get("column")
    al = t2a.get(a.get("table"), "t0")
    if fn == "count":
        return "COUNT(*)" if not c else f"COUNT({al}.{_q(c)})"
    if fn == "count_distinct":
        return f"COUNT(DISTINCT {al}.{_q(c or 'id')})"
    return f"{fn.upper()}({al}.{_q(c)})"


def to_ir(plan: Plan, source_scope=None, kinds=None, sm=None) -> QueryIR:
    filters = []
    for f in plan.filters:
        g = _gfilter(f, sm, kinds)
        filters.append(IRFilter(table=f["table"], column=f["column"],
                                op=("BETWEEN" if g.op == "between" else g.op), value=g.value,
                                grounding=g.grounding, concept="agent"))
    measure = None
    if plan.aggregates:
        a = plan.aggregates[0]
        measure = IRMeasure(aggregation=("count" if a["fn"] == "count_distinct" else a["fn"]),
                            column=a.get("column"), table=a.get("table"),
                            distinct=(a["fn"] == "count_distinct"))
    order = None
    if plan.order:
        o = plan.order[0]
        t, c = split_ref(o["expr"])
        order = {"column": c, "table": t, "direction": o["dir"],
                 **({"alias": c} if t is None else {})}
    tw = None
    if plan.time:
        w = plan.time.get("window") or {}
        tw = {"column": plan.time["column"], "start": w.get("from"), "end": w.get("to")}
    gm = {"anchor": "agent"}
    for f in filters:
        gm[f"filter:{f.column}"] = f"agent.{f.grounding}"
    return QueryIR(anchor=plan.anchor, secondaries=[t for t in plan.tables[1:]], measure=measure,
                   filters=filters, group_keys=[g["column"] for g in plan.group_by], time_window=tw,
                   order=order, limit=plan.limit, distinct=bool(plan.distinct and not plan.aggregates),
                   source_scope=list(source_scope or []), grounding_method=gm,
                   confidence=float(plan.confidence or 0.0), head="frame.agent")


def compile_plan(plan: Plan, sm, source_scope=None, kinds=None):
    """Plan → frame_compiler.Compiled | frame_compiler.Declined."""
    from veda.understanding.frame_compiler import Compiled, Declined
    if not plan.tables:
        return Declined("tables", "the plan names no table")
    try:
        frm, t2a, dupes = _skeleton(plan)
    except Exception as e:
        return Declined("joins", f"join skeleton failed: {type(e).__name__}")
    missing = [t for t in plan.tables if t not in t2a]
    if missing:
        return Declined("joins", f"{', '.join(missing)} not connected by the plan's joins")
    # a table reached by two different joins (staff as assignee AND as creator) gets two
    # aliases, but a plan names columns by TABLE — which occurrence a column means is not
    # expressible, so say so rather than silently binding every column to the first
    if dupes:
        return Declined("joins", f"{dupes[0]} is joined twice (two relationships); use one route to it")
    parts, why = where_parts(plan, t2a, sm, kinds)
    if parts is None:
        return Declined("filters", why)
    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    ir = to_ir(plan, source_scope, kinds, sm)
    tables = list(plan.tables)
    anchor = plan.anchor

    def ref_sql(t, c):
        return f"{t2a.get(t, 't0')}.{_q(c)}"

    def order_sql(default=None):
        items = []
        for o in plan.order:
            t, c = split_ref(o["expr"])
            expr = _q(c) if t is None else ref_sql(t, c)
            items.append(f"{expr} {o['dir'].upper()} NULLS LAST")
        if not items and default:
            items = [default]
        return (" ORDER BY " + ", ".join(items)) if items else ""

    # ── scalar aggregate(s) ──
    if plan.aggregates and not plan.group_by:
        if len(plan.aggregates) == 1 and not plan.joins:
            a = plan.aggregates[0]
            try:
                from veda.planning import build_aggregate_sql
                agg = "count" if a["fn"] == "count_distinct" else a["fn"]
                sql, _t = build_aggregate_sql(_bare(anchor), [], sm, measure_agg=agg,
                                              measure_column=a.get("column"),
                                              distinct=(a["fn"] == "count_distinct"),
                                              where_sql=(" AND ".join(parts) or None))
            except Exception as e:
                return Declined("aggregates", f"aggregate builder failed: {type(e).__name__}")
            if not sql:
                return Declined("aggregates", "aggregate builder returned nothing")
            ir.order = None
            ir.limit = None
            return Compiled(sql=sql, ir=ir, tables=tables, columns=[c for c in [a.get("column")] if c],
                            head="frame.agent")
        sel = ", ".join(f"{_agg_expr(a, t2a)} AS {_q(a['alias'])}" for a in plan.aggregates)
        ir.order = None
        ir.limit = None
        return Compiled(sql=f"SELECT {sel} {frm}{where}", ir=ir, tables=tables,
                        columns=[a["column"] for a in plan.aggregates if a.get("column")], head="frame.agent")

    # ── grouped aggregate ──
    if plan.group_by:
        aggs = plan.aggregates or [{"fn": "count", "table": anchor, "column": None, "alias": "count"}]
        if not plan.aggregates:
            plan.aggregates = aggs
            ir.measure = IRMeasure(aggregation="count", column=None, table=anchor)
        gcols = [ref_sql(g["table"], g["column"]) for g in plan.group_by]
        labels, sel = set(), []
        for g, gc in zip(plan.group_by, gcols):
            lab = g["column"] if g["column"] not in labels else f"{_bare(g['table']).split('_')[-1]}_{g['column']}"
            labels.add(lab)
            sel.append(gc + (f" AS {_q(lab)}" if lab != g["column"] else ""))
        sel += [f"{_agg_expr(a, t2a)} AS {_q(a['alias'])}" for a in aggs]
        dflt = f"{_q(aggs[0]['alias'])} DESC NULLS LAST"
        if not plan.order:
            ir.order = {"column": aggs[0]["alias"], "alias": aggs[0]["alias"], "table": None, "direction": "desc"}
        # No LIMIT unless the plan named one explicitly — the executor's
        # EXECUTION_RESULT_LIMIT caps rows for every head (§10.7).
        limit_sql = f" LIMIT {int(plan.limit)}" if plan.limit else ""
        sql = (f"SELECT {', '.join(sel)} {frm}{where} GROUP BY {', '.join(gcols)}"
               f"{order_sql(dflt)}{limit_sql}")
        if plan.limit is None:
            ir.limit = None
        return Compiled(sql=sql, ir=ir, tables=tables,
                        columns=[g["column"] for g in plan.group_by] + [a["column"] for a in aggs if a.get("column")],
                        head="frame.agent")

    # ── list ──
    proj = [(p["table"], p["column"]) for p in plan.projection] or [(anchor, "id")]
    seen, sel = set(), []
    for t, c in proj:
        lab = c if c not in seen else f"{_bare(t).split('_')[-1]}_{c}"
        seen.add(lab)
        sel.append(ref_sql(t, c) + (f" AS {_q(lab)}" if lab != c else ""))
    distinct = "DISTINCT " if plan.distinct else ""
    osql = order_sql()
    if osql and not plan.distinct and (anchor, "id") in proj and not any(
            split_ref(o["expr"]) == (anchor, "id") for o in plan.order):
        osql += f", t0.{_q('id')} {plan.order[0]['dir'].upper()}"
    # No LIMIT unless the plan named one explicitly (see the grouped branch above).
    limit_sql = f" LIMIT {int(plan.limit)}" if plan.limit else ""
    sql = f"SELECT {distinct}{', '.join(sel)} {frm}{where}{osql}{limit_sql}"
    if plan.limit is None:
        ir.limit = None
    return Compiled(sql=sql, ir=ir, tables=tables, columns=[c for _t, c in proj], head="frame.agent")


# ── probe SQL (the probe tool; same predicates the compiler writes) ──────────────────
def probe_sql(tables: List[str], edges: List[Dict[str, Any]], filters: List[Dict[str, Any]],
              order: Optional[str], sm):
    """One COUNT round trip: total rows of the (joined) grain, rows per filter, rows after
    all filters, distinct values of the order column. Returns (sql, keys) where keys are
    (kind, result-column, filter-dict)."""
    p = Plan(tables=list(tables))
    for e in edges:
        p.joins.append({"from": e["source_table"], "to": e["target_table"],
                        "on": [(e["source_column"], e["target_column"])], "via": e.get("id"),
                        "requires_predicate": e.get("requires_predicate")})
        for t in (e["source_table"], e["target_table"]):
            if t not in p.tables:
                p.tables.append(t)
    frm, t2a, _dupes = _skeleton(p)
    from veda.understanding.frame_compiler import predicate
    sel, keys, preds = ["COUNT(*) AS n_all"], [], []
    for i, f in enumerate(filters):
        t, c = split_ref(f.get("col"))
        if t is None:
            t = p.anchor
        op = str(f.get("op") or "=")
        g = _gfilter({"table": t, "column": c, "op": op, "value": f.get("value")}, sm)
        pr = predicate(g, t2a.get(t, "t0"))
        if not pr:
            continue
        preds.append(pr)
        sel.append(f"COUNT(*) FILTER (WHERE {pr}) AS n_f{i}")
        keys.append(("f", f"n_f{i}", {"col": f"{t}.{c}", "op": op, "value": f.get("value")}))
    if preds:
        sel.append(f"COUNT(*) FILTER (WHERE {' AND '.join(preds)}) AS n_where")
    if order:
        t, c = split_ref(order)
        sel.append(f"COUNT(DISTINCT {t2a.get(t or p.anchor, 't0')}.{_q(c)}) AS n_order_distinct")
        keys.append(("o", "n_order_distinct", None))
    return f"SELECT {', '.join(sel)} {frm}", keys


def plan_fragment(plan: Plan) -> Dict[str, Any]:
    """The probe tool's arguments for a full plan (the harness's automatic final probe)."""
    frag = {"tables": [plan.anchor], "joins": list(dict.fromkeys(j["via"] for j in plan.joins if j.get("via"))),
            "filters": [{"col": f"{f['table']}.{f['column']}", "op": f["op"], "value": f.get("value")}
                        for f in plan.filters if f.get("op") not in ("is_null", "is_not_null")]}
    if plan.time:
        w = plan.time.get("window") or {}
        col = f"{plan.time['table']}.{plan.time['column']}"
        if w.get("from") and w.get("to"):
            frag["filters"].append({"col": col, "op": "between", "value": [w["from"], w["to"]]})
        elif w.get("from"):
            frag["filters"].append({"col": col, "op": ">=", "value": w["from"]})
        elif w.get("to"):
            frag["filters"].append({"col": col, "op": "<=", "value": w["to"]})
    o = next((o for o in plan.order if split_ref(o["expr"])[0] is not None), None)
    if o and not plan.group_by:
        frag["order"] = o["expr"]
    return frag
