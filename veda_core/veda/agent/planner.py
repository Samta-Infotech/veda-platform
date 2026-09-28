"""veda.agent.planner — the planner loop (planner agent, part C).

ReAct over the tools in tools.py, one constrained JSON object per step:

    {"thought": str, "action": {"tool": <name>, "args": {...}}}      a tool call
    {"thought": str, "action": {"final": <plan>}}                    the plan
    {"thought": str, "action": {"tool": "clarify", "args": {"question": str}}}
    {"thought": str, "action": {"tool": "split", "args": {"parts": [str]}}}

The JSON schema handed to Ollama's `format` is rebuilt every step with ENUMS over the
identifiers the run's tools have returned so far (tables, table.column refs, join edge
ids) — the decoder cannot emit an identifier it was not shown; `Plan.validate` is the
backstop over the tool log. Temperature 0, num_predict ≤ AGENT_NUM_PREDICT, num_ctx =
SLM_NUM_CTX, and the system block (tools, plan schema, six worked traces, three rules) is
static so the host's prefix cache keeps it; the per-question message only ever APPENDS.

Automatic, deterministic first calls: find_entities(<question>) and, when its top entity
is confident, describe(<top>). They are logged like any other call (`auto: true`) and
count against the tool budget.

Budgets (config): ≤ AGENT_MAX_TOOL_CALLS tool calls, ≤ AGENT_MAX_STEPS SLM steps, wall ≤
AGENT_PART_BUDGET_S. Over budget → a typed clarify carrying the open questions.
Reflection: a probe anomaly (0 rows after the filters, a vacuous range, a constant order
column) buys ONE revision; a second → clarify with the counts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from veda.agent.tools import ToolBox, TOOL_NAMES, KINDS, render as render_call, compact
from veda.agent.plan import (Plan, from_model, validate, compile_plan, plan_fragment,
                             seen_identifiers, split_ref, AGG_FNS, OPS)

_CHARS_PER_TOKEN = 3.6          # measured: a 7,786-char system block = 2,086 tokens (3.73)
_OBS_CHARS = 520                # one tool result in the working memory (≈ 145 tokens)
_PRIOR_OBS_CHARS = 320          # one result from the PREVIOUS turn's log (a follow-up's context)


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


@dataclass
class AgentResult:
    kind: str                                   # sql | clarify | rag | split | fail
    plan: Optional[Plan] = None
    compiled: Any = None
    message: Optional[str] = None
    parts: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    validation: List[Dict[str, Any]] = field(default_factory=list)   # [{step, errors}]
    budget: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    question: str = ""
    judge: List[Dict[str, Any]] = field(default_factory=list)

    def trace(self) -> Dict[str, Any]:
        """The `agent` explain section."""
        return {"kind": self.kind, "reason": self.reason,
                "steps": [{k: v for k, v in s.items() if k != "raw"} for s in self.steps],
                "tool_calls": [{"id": c.get("id"), "tool": c.get("tool"), "args": c.get("args") or {},
                                "ms": c.get("ms", 0.0),
                                "auto": c.get("auto", False),
                                "result": compact(c.get("result"))} for c in self.tool_calls],
                "question": self.question,
                "final_plan": (self.plan.to_dict() if self.plan else None),
                "draft": (_draft(self.plan) if self.plan is not None and self.kind == "sql" else None),
                "sql": getattr(self.compiled, "sql", None),
                "validation": self.validation, "budget": self.budget, "judge": self.judge,
                "message": self.message, "parts": self.parts}


def _draft(plan):
    from veda.understanding.frame_path import _draft_of
    return _draft_of(plan)


# ── the static system block ───────────────────────────────────────────────────────────
_SYSTEM_HEAD = """You plan answers over a business database: call TOOLS, read their results, then emit a PLAN. Never SQL. One compact JSON per turn: {"thought":"<=12 words","action":{"tool":..,"args":{..}}} or {"thought":..,"action":{"final":<plan>}}.
TOOLS
retrieve: (done for you first) the columns most related to the question, best first.
find_entities(phrase): tables whose meaning matches a phrase (by embedding), best first; via_column = the column that matched.
describe(table): business_date, lifecycle column+values, measures, dimensions, display (name) column, parents/children via FK.
columns(table,kind,phrase): the table's columns closest in meaning to the phrase (kind MONETARY TEMPORAL CATEGORY IDENTIFIER METRIC any).
join_path(a,b): join routes r1,r2… ranked by how well their relationship ("the user this ticket is assigned to") matches the question; "says" = the relationship.
values(table,column,phrase): stored values, the closest to the phrase first.
probe(tables,joins,filters,order): row counts per filter / after filters / distinct order values.
doc_sections(query): handbook passages. clarify(question). split(parts).
PLAN {"tables":[grain table (the rows listed/counted) first, …],"joins":[route ids],"select":["t.col"],"filters":[{"col":"t.col","op":"= != > >= < <= between in is_null is_not_null","value":v}],"group_by":["t.col"],"aggregates":[{"fn":"count count_distinct sum avg min max","col":"t.col" or "*"}],"order":[{"by":"t.col" or "agg1","dir":"asc|desc"}],"limit":n|null} — every key in this order, [] when empty; optional "distinct":true, "time":{"col":"t.col","from":"YYYY-MM-DD","to":"YYYY-MM-DD"}. {"kind":"rag"} for a document question.
RULES 1 only identifiers a tool returned. 2 join_path before joining; put a route id in joins; prefer the best-ranked route — if you pick a lower one, cite the observation [cN] in thought. 3 probe before final when there is a filter or an order column. 4 only the filters and limit the question states. 5 every thing the question names must map to a table/column you looked up.
MEANING latest = order by business date desc (not a filter) | top N = order + limit | above N = > N | how many = count | total = sum | per X = group_by X | X with the most Y = count Y per X, order agg1 desc | status words are lifecycle values | name rows with the display column
EXAMPLES (illustrative schema — never use its names)
Q: tickets per category and assigned staff member
[c2] find_entities → tickets, ticket_staff (links a ticket to the staff member it is assigned to), staff
you: {"thought":"assigned = the link table","action":{"tool":"join_path","args":{"a":"tickets","b":"staff"}}}
[c3] → {"routes":[{"id":"r1","path":["ticket_staff.ticket_id=tickets.id","ticket_staff.staff_id=staff.id"],"says":"the staff member this ticket is assigned to"},{"id":"r2","path":["tickets.created_by_id=staff.id"],"says":"the staff member who created this ticket"}]}
you: {"thought":"r1 is assignment","action":{"final":{"tables":["tickets","staff"],"joins":["r1"],"select":[],"filters":[],"group_by":["tickets.category","staff.full_name"],"aggregates":[{"fn":"count","col":"*"}],"order":[{"by":"agg1","dir":"desc"}],"limit":null}}}
"""


def system_block() -> str:
    """Static across parts (the host keeps it in the agent's prompt-cache slot); the
    worked examples are per part — the nearest verified plans, in the user message."""
    return _SYSTEM_HEAD


# ── the per-step response schema (enums over what the tools returned) ────────────────
def _enum_or_str(vals: List[str]) -> Dict[str, Any]:
    vals = sorted(set(v for v in vals if v))
    return {"type": "string", "enum": vals} if vals else {"type": "string"}


def step_schema(tb: ToolBox, final_only: bool = False, allow_clarify: bool = True) -> Dict[str, Any]:
    s = seen_identifiers(tb.log)
    tabs = sorted(s.tables)
    cols = sorted(f"{t}.{c}" for t, c in s.columns)
    edges = sorted(s.routes)
    T, C, E = _enum_or_str(tabs), _enum_or_str(cols), _enum_or_str(edges)
    n_edges = 4 if edges else 0          # no join_path result yet → the plan cannot join
    scalar = {"type": ["string", "number", "boolean", "null"]}
    value = {"anyOf": [scalar, {"type": "array", "items": scalar, "maxItems": 12}]}
    filt = {"type": "object", "properties": {"col": C, "op": {"type": "string", "enum": list(OPS)},
                                             "value": value}, "required": ["col", "op"]}
    order_by = _enum_or_str(cols + [f"agg{i}" for i in (1, 2, 3)])
    # EVERY slot is required, in the order the examples write them: a constrained decoder
    # emits properties in schema order, so an optional slot the model skipped (group_by)
    # could never be written after a later one (aggregates) — measured on the dev set.
    plan = {"type": "object", "properties": {
        "tables": {"type": "array", "items": T, "maxItems": 5},
        "joins": {"type": "array", "items": E, "maxItems": n_edges},
        "select": {"type": "array", "items": C, "maxItems": 8},
        "filters": {"type": "array", "items": filt, "maxItems": 5},
        "group_by": {"type": "array", "items": C, "maxItems": 3},
        "aggregates": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "properties": {"fn": {"type": "string", "enum": list(AGG_FNS)},
                                             "col": _enum_or_str(cols + ["*"])}, "required": ["fn", "col"]}},
        "order": {"type": "array", "maxItems": 2, "items": {
            "type": "object", "properties": {"by": order_by, "dir": {"type": "string", "enum": ["asc", "desc"]}},
            "required": ["by", "dir"]}},
        "limit": {"type": ["integer", "null"]},
        "distinct": {"type": "boolean"},
        "time": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
            "col": C, "from": {"type": ["string", "null"]}, "to": {"type": ["string", "null"]}},
            "required": ["col", "from", "to"]}]},
    }, "required": ["tables", "joins", "select", "filters", "group_by", "aggregates", "order", "limit"]}
    rag = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["rag"]}}, "required": ["kind"]}
    tools = [
        ("find_entities", {"phrase": {"type": "string"}}, ["phrase"]),
        ("describe", {"table": T}, ["table"]),
        ("columns", {"table": T, "kind": {"type": "string", "enum": list(KINDS)}, "phrase": {"type": "string"}},
         ["table", "kind"]),
        ("join_path", {"a": T, "b": T}, ["a", "b"]),
        ("values", {"table": T, "column": {"type": "string"}, "phrase": {"type": "string"}}, ["table", "column"]),
        ("similar_questions", {"text": {"type": "string"}}, ["text"]),
        ("probe", {"tables": {"type": "array", "items": T, "minItems": 1, "maxItems": 1},
                   "joins": {"type": "array", "items": E, "maxItems": n_edges},
                   "filters": {"type": "array", "items": filt, "maxItems": 5},
                   "order": C}, ["tables"]),
        ("doc_sections", {"query": {"type": "string"}}, ["query"]),
        ("clarify", {"question": {"type": "string"}}, ["question"]),
        ("split", {"parts": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4}}, ["parts"]),
    ]
    alts = [{"type": "object", "properties": {
        "tool": {"type": "string", "enum": [name]},
        "args": {"type": "object", "properties": props, "required": req}}, "required": ["tool", "args"]}
        for name, props, req in tools
        if (not final_only or name == "clarify") and (allow_clarify or name != "clarify")]
    alts.append({"type": "object", "properties": {"final": {"anyOf": [plan, rag]}}, "required": ["final"]})
    return {"type": "object", "properties": {"thought": {"type": "string", "maxLength": 120},
                                             "action": {"anyOf": alts}},
            "required": ["thought", "action"]}


# ── working memory ───────────────────────────────────────────────────────────────────
def _memory(question: str, draft: Optional[Dict[str, Any]], tb: ToolBox, events: List[Dict[str, Any]],
            fewshot: Optional[List[str]] = None) -> str:
    """QUESTION, optional DRAFT, then the run's events in order: each tool call (with the
    action that asked for it) and its result, each rejected final and harness note.

    PREFIX-STABLE: every event renders the same way on every step (a fixed size cap per
    result, decided by the event alone), so each step's prompt extends the previous one
    and the host's prompt cache re-evaluates only the new tail. The context budget is
    enforced by the loop (it forces the final plan), never by re-shrinking the past."""
    head = []
    for ex in (fewshot or []):
        head.append(f"SOLVED: {ex}")
    head.append(f"QUESTION: {question}")
    if draft:
        head.append("DRAFT (the previous turn's plan — edit it for this follow-up): "
                    + json.dumps(draft, separators=(",", ":"), ensure_ascii=False))
    by_id = {e["id"]: e for e in tb.log}
    lines = []
    for ev in events:
        if ev["type"] == "call":
            e = by_id.get(ev["id"])
            if e is None:
                continue
            prior = str(e["id"]).startswith("p")
            if prior and e.get("tool") in ("find_entities", "similar_questions", "doc_sections"):
                continue                    # the previous turn's search results: the draft has them
            line = render_call(e, max_chars=(_PRIOR_OBS_CHARS if prior else _OBS_CHARS))
            if ev.get("act"):
                lines.append(f"you: {ev['act']}")
                line = f"[{e['id']}] → " + line.split(" → ", 1)[1]
            lines.append(line)
        else:
            lines.append(ev["text"])
    return "\n".join(head + lines)


def _act_text(obj: Dict[str, Any]) -> str:
    a = obj.get("action") or {}
    th = str(obj.get("thought") or "")[:120]
    if "final" in a:
        return f"({th}) final " + json.dumps(a["final"], separators=(",", ":"), ensure_ascii=False)
    args = json.dumps(a.get("args") or {}, separators=(",", ":"), ensure_ascii=False)[1:-1]
    return f"({th}) {a.get('tool')}({args})"


# ── the SLM step ─────────────────────────────────────────────────────────────────────
def _slm_step(system: str, user: str, schema: Dict[str, Any], timeout: float):
    if _cfg("AGENT_SLM_URL", ""):
        return _slm_step_server(system, user, schema, timeout)
    from slm import call_slm
    from slm._call_slm import collect_usage
    with collect_usage() as u:
        raw = call_slm(user, system=system, purpose="agent_plan", temperature=0.0, seed=7,
                       num_predict=int(_cfg("AGENT_NUM_PREDICT", 160)),
                       num_ctx=int(_cfg("SLM_NUM_CTX", 4096)), timeout=max(1, int(timeout)),
                       json_schema=schema)
        calls = u.calls()
    usage = calls[-1] if calls else {}
    return raw, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def _slm_step_server(system: str, user: str, schema: Dict[str, Any], timeout: float):
    """The agent's own OpenAI-compatible server (llama-server on the same GGUF), pinned to
    one slot so its prompt cache holds the agent's prefix between steps. Honors the part's
    SLM deadline and records the call in the query trace like call_slm does."""
    import urllib.request
    try:
        from slm._call_slm import slm_deadline
        dl = slm_deadline.get()
        if dl is not None:
            left = dl - time.time()
            if left <= 0.5:
                raise TimeoutError("SLM deadline passed before 'agent_plan'")
            timeout = min(timeout, left)
    except TimeoutError:
        raise
    except Exception:
        pass
    body = {"model": _cfg("SLM_MODEL_NAME", "qwen2.5:7b-instruct"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0, "seed": 7, "max_tokens": int(_cfg("AGENT_NUM_PREDICT", 160)),
            "response_format": {"type": "json_schema", "json_schema": {"name": "step", "schema": schema}},
            "id_slot": int(_cfg("AGENT_SLM_SLOT", 1)), "cache_prompt": True}
    url = str(_cfg("AGENT_SLM_URL", "")).rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    ok, err = True, None
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout))) as r:
            d = json.loads(r.read())
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
        if "timed out" in str(e).lower():
            raise TimeoutError(str(e))
        raise
    finally:
        try:
            from veda.explain import current_trace
            current_trace().slm_call("agent_plan", body["model"], (time.time() - t0) * 1000.0, ok, err)
        except Exception:
            pass
    u = d.get("usage") or {}
    return (d["choices"][0]["message"]["content"],
            {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
             "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")})


def _parse(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(raw)
    except Exception:
        s = (raw or "").strip()
        i, j = s.find("{"), s.rfind("}")
        try:
            obj = json.loads(s[i:j + 1]) if i >= 0 < j else None
        except Exception:
            obj = None
    return obj if isinstance(obj, dict) else None


# ── probe anomalies (reflection) ────────────────────────────────────────────────────
def anomalies(res: Dict[str, Any], plan: Optional[Plan] = None, sm=None) -> List[str]:
    out: List[str] = []
    if not isinstance(res, dict) or res.get("error"):
        return out
    tot = res.get("rows_total")
    if not tot:
        return out
    fl = res.get("filters") or []
    temporal_only = bool(fl) and all(_is_temporal(f, sm) for f in fl)
    if "rows_after_filters" in res and res["rows_after_filters"] == 0 and not temporal_only:
        zero = [f for f in fl if not f.get("rows")]
        what = "; ".join(f"{f['col']} {f['op']} {f.get('value')} keeps 0" for f in zero) or "the filters together keep 0"
        out.append(f"0 of {tot} rows after the filters ({what})")
    for f in fl:
        if f.get("op") in (">", ">=", "<", "<=") and tot > 20 and f.get("rows") == tot:
            out.append(f"{f['col']} {f['op']} {f.get('value')} is true for all {tot} rows (vacuous)")
    if "distinct_order_col" in res and tot > 1 and (res.get("distinct_order_col") or 0) <= 1:
        out.append(f"the order column has {res.get('distinct_order_col')} distinct value(s) — sorting by it changes nothing")
    return out


def _is_temporal(f, sm) -> bool:
    t, c = split_ref(f.get("col"))
    meta = ((sm or {}).get("columns") or {}).get(f"{t}.{c}") or {}
    return str(meta.get("semantic_type") or "").upper() == "TEMPORAL"


def _post_compile_checks(plan: Plan, comp, question: str, draft_question: Optional[str], sm, vocab,
                         log) -> List[str]:
    """Two existing gates, asked BEFORE the plan is accepted so the model can still fix it:
    the lexical qualifier gate (veda.validation.qualifier_completeness — every content
    word of the user's own message must be accounted for by the SQL; words that licensed
    a filter through a glossary phrase count as accounted for), and a per-X breakdown the
    question asks for (veda.planning's grouped-mode detectors) must be a GROUP BY."""
    errs: List[str] = []
    from veda.agent.plan import asks_grouping, accounted_phrases, question_numbers, stated_numbers, _fmt, _isnum
    if plan.aggregates and not plan.group_by and asks_grouping(question):
        errs.append("the question asks for a per-item breakdown: add group_by")
    # every number the question states is a limit, part of a time window, or a filter value
    used = set()
    for f in plan.filters:
        for x in (f["value"] if isinstance(f.get("value"), (list, tuple)) else [f.get("value")]):
            if _isnum(x):
                used.add(float(x))
    if plan.limit:
        used.add(float(plan.limit))
    try:
        from query.temporal_parser import run_temporal_parser
        texpr = " ".join(run_temporal_parser(question).raw_expressions or [])
    except Exception:
        texpr = ""
    in_time = question_numbers(texpr) if texpr else set()
    in_draft = question_numbers(draft_question or "")
    for n in sorted(stated_numbers(question) - used - in_time - in_draft):
        if not any(str(_fmt(n)) in str(f.get("value")) for f in plan.filters):
            errs.append(f"the question's number {_fmt(n)} is not used by the plan (a filter or the limit)")
    try:
        from veda.validation import qualifier_completeness
        q = _strip_phrases(question, accounted_phrases(plan, question, log, vocab, sm))
        ok, missing = qualifier_completeness(q, comp.sql, sm, user_message=q)
        if not ok:
            miss = missing if isinstance(missing, str) else ", ".join(map(str, missing or []))
            errs.append(f"the plan does not account for '{miss}' from the question")
    except Exception:
        pass
    return errs


def _strip_phrases(question: str, phrases: List[str]) -> str:
    """Remove each phrase (case-, plural- and punctuation-insensitive, whole words)."""
    from veda.agent.plan import _norm
    words = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", question)
    toks = [(i, w) for i, w in enumerate(words) if re.match(r"[A-Za-z0-9]", w)]
    drop = set()
    for ph in phrases:
        pt = _norm(ph).split()
        if not pt:
            continue
        for k in range(len(toks) - len(pt) + 1):
            if [_norm(toks[k + j][1]) for j in range(len(pt))] == pt:
                drop.update(toks[k + j][0] for j in range(len(pt)))
    return "".join(" " if i in drop else w for i, w in enumerate(words))


def _join_hint(errs: List[str], plan: Plan, log) -> str:
    """For 'not joined' / 'table not in the plan' rejections: the route id a join_path call
    already returned for that table, or the join_path call to make."""
    s = seen_identifiers(log)
    for e in errs:
        m = (re.match(r"table (\S+) is not joined to (\S+)", e)
             or re.search(r": (\S+)\.[^.\s]+ is on a table not in the plan", e))
        if not m:
            continue
        t = m.group(1)
        for rid, edges in s.routes.items():
            ends = {x for ed in edges for x in (ed["source_table"], ed["target_table"])}
            if t in ends and plan.anchor in ends:
                return f" Add route {rid} (from join_path) to joins — it reaches {t}."
        return f" Call join_path(a={plan.anchor}, b={t}) and put its route id in joins."
    return ""


def _license_aggregates(plan: Plan, question: str) -> List[str]:
    """An ungrouped aggregate must be LICENSED by the question's wording — the frame path's
    rule (frame_grounding.license_aggregation, its word patterns): 'the cheapest listings'
    is a ranking of rows, not MIN(); so an unlicensed MIN/MAX becomes an ORDER on its
    column (a note records it) and an unlicensed COUNT/SUM/AVG is sent back."""
    if plan.group_by or not plan.aggregates:
        return []
    try:
        from veda.understanding.frame_grounding import _MINMAX_W, _COUNT_W, _SUMAVG_W
    except Exception:
        return []
    ql = (question or "").lower()
    errs: List[str] = []
    for a in list(plan.aggregates):
        fn = a.get("fn")
        if fn in ("min", "max") and a.get("column") and not _MINMAX_W.search(ql):
            plan.aggregates.remove(a)
            ref = f"{a['table']}.{a['column']}"
            # an order on that aggregate is now an order on its column
            plan.order = [dict(o, expr=ref) if o.get("expr") == a.get("alias") else o for o in plan.order]
            if not plan.order:
                plan.order = [{"expr": ref, "dir": "asc" if fn == "min" else "desc"}]
            plan.notes.append(f"{fn.upper()} read as a ranking by {a['column']}")
        elif fn in ("count", "count_distinct") and not _COUNT_W.search(ql):
            errs.append(f"aggregate: the question does not ask for a count — list the rows instead")
        elif fn in ("sum", "avg") and fn in _SUMAVG_W and not _SUMAVG_W[fn].search(ql):
            errs.append(f"aggregate: the question does not ask for {'a total' if fn == 'sum' else 'an average'}"
                        f" — list the rows instead")
    return errs


def _uncited_lower_routes(plan: Plan, log, thought: str, already: set) -> List[Tuple[str, str, str]]:
    """Routes the plan uses that are not rank 1 of their join_path call, when the thought
    cites no observation ([cN] / the route id)."""
    out = []
    used = list(dict.fromkeys(j.get("via") for j in plan.joins if j.get("via")))
    for rid in used:
        if rid in already or re.search(r"\[c\d+\]|\bc\d+\b|\b" + re.escape(rid) + r"\b", thought or ""):
            continue
        for e in log:
            rs = ((e.get("result") or {}).get("routes") or []) if e.get("tool") == "join_path" else []
            ids = [r.get("id") for r in rs]
            if rid in ids and ids.index(rid) > 0:
                best = rs[0]
                out.append((rid, best.get("id"), str(best.get("says") or best.get("why") or "")[:80]))
                break
    return out


def _phrases_of(tb) -> Dict[str, str]:
    ph = {}
    for d in (getattr(tb, "_fkph", None) or {}).values():
        ph.update(d or {})
    if not ph:
        try:
            from ingestion.link_text import for_source
            sids = {tb.source_of(t) for t in tb.tables()[:1]}
            for sid in sids:
                ph.update(for_source(sid, getattr(tb.ctx, "tenant", None) or "default")[1])
            tb._fkph = {s_: ph for s_ in sids}
        except Exception:
            pass
    return ph


def _default_projection(plan: Plan, log) -> None:
    """A list plan names its rows: with no select, the grain's id + display column (from
    describe); and the column the rows are sorted by is shown, so the order is visible."""
    if plan.aggregates or plan.group_by or not plan.anchor:
        return
    seen = seen_identifiers(log).columns
    if not plan.projection:
        if (plan.anchor, "id") in seen:
            plan.projection = [{"table": plan.anchor, "column": "id"}]
        for e in log:
            r = e.get("result") or {}
            if e.get("tool") == "describe" and r.get("table") == plan.anchor and r.get("display"):
                plan.projection.append({"table": plan.anchor, "column": r["display"]})
                break
    have = {(p["table"], p["column"]) for p in plan.projection}
    for o in plan.order:
        t, c = split_ref(o["expr"])
        if t is not None and (t, c) not in have and (t, c) in seen and plan.projection:
            plan.projection.append({"table": t, "column": c})
            have.add((t, c))


def _covering_probe(plan: Plan, log) -> Optional[Dict[str, Any]]:
    """The latest successful probe whose filters/joins/order cover this plan's, if any."""
    want = plan_fragment(plan)
    need = {(f.get("col"), f.get("op"), json.dumps(f.get("value"), default=str)) for f in want["filters"]}
    for e in reversed(log):
        if e["tool"] != "probe" or (e.get("result") or {}).get("error"):
            continue
        a = e.get("args") or {}
        got = {(f.get("col"), f.get("op"), json.dumps(f.get("value"), default=str)) for f in a.get("filters") or []}
        if need <= got and set(want.get("joins") or []) <= set(a.get("joins") or []) \
                and (not want.get("order") or a.get("order") == want.get("order")):
            return e
    return None


def _needs_probe(plan: Plan) -> bool:
    return bool(plan.filters or plan.time or (any(split_ref(o["expr"])[0] for o in plan.order)
                                               and not plan.group_by))


# ── the loop ─────────────────────────────────────────────────────────────────────────
def run_planner(question: str, sm, vocab, *, draft: Optional[Dict[str, Any]] = None,
                prior_log: Optional[List[Dict[str, Any]]] = None,
                draft_question: Optional[str] = None,
                toolbox: Optional[ToolBox] = None, slm: Optional[Callable] = None,
                source_scope=None, max_steps: Optional[int] = None,
                max_tool_calls: Optional[int] = None, wall_s: Optional[float] = None) -> AgentResult:
    """Plan one part. `slm(system, user, schema, timeout) -> (raw, usage)` is injectable
    (the tests replay the worked traces through a stub)."""
    t0 = time.time()
    max_steps = int(max_steps or _cfg("AGENT_MAX_STEPS", 10))
    max_calls = int(max_tool_calls or _cfg("AGENT_MAX_TOOL_CALLS", 8))
    wall = float(wall_s or _cfg("AGENT_PART_BUDGET_S", 40.0))
    ctx_tokens = int(_cfg("AGENT_NUM_CTX", 8192) if _cfg("AGENT_SLM_URL", "") else _cfg("SLM_NUM_CTX", 4096))
    slm = slm or _slm_step
    tb = toolbox or ToolBox(sm, vocab, question)
    if prior_log:
        tb.log = [{"tool": e.get("tool"), "args": e.get("args") or {}, "result": e.get("result") or {},
                   "ms": 0.0, "id": f"p{i + 1}", "auto": True} for i, e in enumerate(prior_log)] + tb.log
        from veda.agent.plan import route_edges
        for e in prior_log:
            for o in ((e.get("result") or {}).get("routes") or []):
                tb.routes.setdefault(o["id"], route_edges(o))
    res = AgentResult(kind="fail")
    system = system_block()
    events: List[Dict[str, Any]] = []
    revisions = 0
    rejections = 0
    judge_revisions = 0
    forced: set = set()
    cited: set = set()
    force_final = False

    def n_calls():
        return sum(1 for e in tb.log if not str(e["id"]).startswith("p"))

    def finish(r: AgentResult) -> AgentResult:
        r.question = question
        r.steps, r.tool_calls = res.steps, list(tb.log)
        r.validation = res.validation
        r.judge = res.judge
        r.budget = {"steps": len(res.steps), "tool_calls": n_calls(), "wall_s": round(time.time() - t0, 2),
                    "limits": {"steps": max_steps, "tool_calls": max_calls, "wall_s": wall},
                    "prompt_tokens": sum(int(s.get("prompt_tokens") or 0) for s in res.steps),
                    "completion_tokens": sum(int(s.get("completion_tokens") or 0) for s in res.steps)}
        return r

    def over_budget_clarify(why: str) -> AgentResult:
        s = seen_identifiers(tb.log)
        names = []
        for e in tb.log:
            for x in ((e.get("result") or {}).get("entities") or [])[:3]:
                names.append(x.get("business_name") or x.get("table"))
        oq = [f"I could not finish planning this ({why})."]
        if names:
            oq.append("Which of these records do you mean: " + ", ".join(dict.fromkeys(names)) + "?")
        return finish(AgentResult(kind="clarify", message=" ".join(oq), reason=f"budget:{why}"))

    # automatic, deterministic first calls
    for e in tb.log:
        events.append({"type": "call", "id": e["id"]})
    if _cfg("AGENT_SEED_RETRIEVAL", True) and not draft:
        # observation 0: the retrieval spine (bi-encoder + cross-encoder) on the part's text
        cid, _sr = tb.call("retrieve", {"text": question, "k": int(_cfg("AGENT_SEED_K", 15))}, auto=True)
        events.append({"type": "call", "id": cid})
    cid, fe = tb.call("find_entities", {"phrase": question, "k": 5}, auto=True)
    events.append({"type": "call", "id": cid})
    try:
        from veda.agent.judge import fewshot as _fewshot
        fewshot = _fewshot(question, vocab) if vocab is not None else []
    except Exception:
        fewshot = []
    ents = (fe or {}).get("entities") or []
    if ents and not draft:
        # describe the confident top entity; when the question NAMES two entities
        # (two strong name hits), describe both and fetch their join edges — the moves
        # every multi-entity plan starts with, done without spending SLM steps
        strong = [e["table"] for e in ents[:3]
                  if float(e.get("score") or 0) >= 0.95 and str(e.get("why", "")).startswith("name:")]
        first = strong[:2] if len(strong) >= 2 else ([ents[0]["table"]] if float(ents[0].get("score") or 0) >= 0.6 else [])
        for t in first:
            cid, _r = tb.call("describe", {"table": t}, auto=True)
            events.append({"type": "call", "id": cid})
        if len(first) == 2:
            cid, _r = tb.call("join_path", {"a": first[0], "b": first[1]}, auto=True)
            events.append({"type": "call", "id": cid})

    while True:
        if len(res.steps) >= max_steps:
            return over_budget_clarify("step budget")
        if n_calls() >= max_calls + 1:        # the auto probe may be the 1 over
            return over_budget_clarify("tool budget")
        left = wall - (time.time() - t0)
        if left <= 1.0:
            return over_budget_clarify("time budget")
        budget_chars = int((ctx_tokens - int(_cfg("AGENT_NUM_PREDICT", 160)) - 60) * _CHARS_PER_TOKEN) - len(system)
        user = _memory(question, draft, tb, events, fewshot)
        calls_left = max_calls - n_calls()
        exhausted = calls_left <= 0 or len(res.steps) >= max_steps - 1 or len(user) > budget_chars - 2 * _OBS_CHARS
        if exhausted:
            force_final = True
        if len(user) > budget_chars:
            # the working memory no longer fits the context: never let the host truncate
            # the system block — stop here with what the tools established
            return over_budget_clarify("context budget")
        user += (f"\nSTEP {len(res.steps) + 1}/{max_steps}; tool calls left {max(0, calls_left)}."
                 + (" Emit the final plan now." if force_final else ""))
        model_calls = sum(1 for e in tb.log if not e.get("auto"))
        schema = step_schema(tb, final_only=force_final, allow_clarify=(exhausted or (model_calls >= 3 and not force_final)))
        st = {"i": len(res.steps) + 1}
        ts = time.time()
        try:
            raw, usage = slm(system, user, schema, left)
        except TimeoutError as e:
            st.update(error=f"timeout: {str(e)[:80]}", ms=round((time.time() - ts) * 1000.0, 1))
            res.steps.append(st)
            return over_budget_clarify("time budget")
        except Exception as e:
            st.update(error=f"{type(e).__name__}: {str(e)[:120]}", ms=round((time.time() - ts) * 1000.0, 1))
            res.steps.append(st)
            return finish(AgentResult(kind="fail", reason=f"slm_error:{type(e).__name__}"))
        st.update(ms=round((time.time() - ts) * 1000.0, 1), **(usage or {}))
        obj = _parse(raw)
        res.steps.append(st)
        if obj is None or not isinstance(obj.get("action"), dict):
            st["error"] = "unparseable"
            st["raw"] = str(raw)[:300]
            events.append({"type": "text", "text": "HARNESS: the last reply was not valid JSON; reply with one JSON object."})
            continue
        st["thought"] = str(obj.get("thought") or "")[:160]
        act = obj["action"]
        if "final" in act:
            fin = act.get("final") or {}
            st["action"] = {"final": fin}
            plan = from_model(fin, tb.log, source_of=tb.source_of)
            if plan.kind == "rag":
                res.plan = plan
                return finish(AgentResult(kind="rag", plan=plan, reason="document question"))
            events.append({"type": "text", "text": "you: " + _act_text(obj)})
            cover = _covering_probe(plan, tb.log) if _needs_probe(plan) else None
            if _needs_probe(plan) and cover is None and n_calls() <= max_calls + 1:
                pc, _pres = tb.call("probe", plan_fragment(plan), auto=True)
                events.append({"type": "call", "id": pc})
                cover = _covering_probe(plan, tb.log)
            from veda.agent.plan import license_limit
            agg_errs = _license_aggregates(plan, question + (" " + str(draft_question) if draft_question else ""))
            _default_projection(plan, tb.log)
            lic_q = question + (" " + str(draft_question) if draft_question else "")
            license_limit(plan, lic_q)
            errs = agg_errs + validate(plan, tb.log, sm, question=lic_q, vocab=vocab)
            anom = anomalies(cover["result"], plan, sm) if cover is not None else []
            force_final = False
            if errs:
                rejections += 1
                res.validation.append({"step": st["i"], "errors": errs[:6]})
                hint = _join_hint(errs, plan, tb.log)
                events.append({"type": "text",
                               "text": "HARNESS: plan rejected — " + "; ".join(errs[:4])
                                       + "." + (hint or " Fix it using identifiers from the tool results.")})
                if rejections >= 3:
                    return finish(AgentResult(kind="clarify", plan=plan, reason="validation",
                                              message="I could not build a plan over the records you named — "
                                                      + errs[0] + ". Could you rephrase?"))
                continue
            if anom:
                if revisions >= 1:
                    res.plan = plan
                    return finish(AgentResult(kind="clarify", plan=plan, reason="probe",
                                              message=f"Checking the data: {anom[0]}. Could you say which "
                                                      f"condition or field you meant?"))
                revisions += 1
                events.append({"type": "text",
                               "text": "HARNESS: the probe shows a problem — " + "; ".join(anom)
                                       + ". Revise the plan once (another value, column or join), or clarify."})
                continue
            # B5: every noun phrase of the question maps to something a tool returned
            if vocab is not None:
                try:
                    from veda.agent.judge import unmapped_phrases
                    um = [x for x in unmapped_phrases(question, plan, tb.log, vocab, sm) if x not in forced]
                except Exception:
                    um = []
                if um and n_calls() <= max_calls + 2:
                    for np_ in um[:2]:
                        forced.add(np_)
                        fc, _r = tb.call("find_entities", {"phrase": np_, "k": 4}, auto=True)
                        events.append({"type": "call", "id": fc})
                    res.validation.append({"step": st["i"], "errors": [f"unmapped: {', '.join(um[:2])}"]})
                    events.append({"type": "text", "text": "HARNESS: the plan does not account for "
                                   + ", ".join(f"'{x}'" for x in um[:2])
                                   + " from the question — see what find_entities returned and revise."})
                    continue
            # C: a lower-ranked route needs a reason
            low = _uncited_lower_routes(plan, tb.log, st.get("thought", ""), cited)
            if low:
                cited |= {r for r, _b, _s in low}
                msg = "; ".join(f"you joined with {r} but {b} ranks higher ({s})" for r, b, s in low)
                res.validation.append({"step": st["i"], "errors": [f"route: {msg}"]})
                events.append({"type": "text", "text": f"HARNESS: {msg}. Use the higher-ranked route, or "
                               "cite the observation [cN] that justifies yours in thought."})
                continue
            # D: the meaning judge
            mode = str(_cfg("AGENT_JUDGE_MODE", "enforce"))
            if mode in ("enforce", "shadow") and vocab is not None:
                try:
                    from veda.agent.judge import judge as _judge
                    v = _judge(question + (" " + str(draft_question) if draft_question else ""), plan, vocab,
                               phrases=_phrases_of(tb))
                    res.judge.append({"step": st["i"], **v.to_dict()})
                except Exception as _je:
                    v = None
                    res.judge.append({"step": st["i"], "error": f"{type(_je).__name__}: {str(_je)[:120]}"})
                if v is not None and not v.ok and mode == "enforce":
                    if judge_revisions >= 1:
                        alt = (v.nearest[0] if v.nearest else {})
                        return finish(AgentResult(
                            kind="clarify", plan=plan, reason="judge",
                            message=(f"I'm not sure my reading answers this: it reads as \"{v.plan_text}\"."
                                     + (f" Did you mean something like \"{alt.get('question')}\"?" if alt.get("question") else
                                        " Could you rephrase?"))))
                    judge_revisions += 1
                    events.append({"type": "text", "text": f"JUDGE: {v.reason}. Revise the plan once "
                                   f"(another table, route or column) or clarify."})
                    continue
            scope = source_scope
            comp = compile_plan(plan, sm, source_scope=scope,
                                kinds=seen_identifiers(tb.log).kinds)
            from veda.understanding.frame_compiler import Declined
            if isinstance(comp, Declined):
                rejections += 1
                res.validation.append({"step": st["i"], "errors": [f"compile:{comp.slot}: {comp.reason}"]})
                events.append({"type": "text",
                               "text": f"HARNESS: the plan cannot be compiled ({comp.reason}). Fix it."})
                if rejections >= 3:
                    return finish(AgentResult(kind="clarify", plan=plan, reason="compile",
                                              message=f"I understood the question but can't express part of it "
                                                      f"({comp.reason}). Could you rephrase that part?"))
                continue
            post = _post_compile_checks(plan, comp, question, draft_question, sm, vocab, tb.log)
            if post:
                rejections += 1
                res.validation.append({"step": st["i"], "errors": post})
                events.append({"type": "text", "text": "HARNESS: plan rejected — " + "; ".join(post) + "."})
                if rejections >= 3:
                    return finish(AgentResult(kind="clarify", plan=plan, reason="qualifier",
                                              message="I could not account for part of the question — "
                                                      + post[0] + ". Could you rephrase?"))
                continue
            plan.confidence = 0.9 if not res.validation else 0.75
            comp.ir.confidence = plan.confidence
            res.plan, res.compiled = plan, comp
            res.validation.append({"step": st["i"], "errors": []})
            return finish(AgentResult(kind="sql", plan=plan, compiled=comp, reason="planned"))
        tool = str(act.get("tool") or "")
        args = act.get("args") if isinstance(act.get("args"), dict) else {}
        st["action"] = {"tool": tool, "args": args}
        if tool == "clarify":
            return finish(AgentResult(kind="clarify", message=str(args.get("question") or "Could you rephrase?"),
                                      reason="model_clarify"))
        if tool == "split":
            parts = [str(p) for p in (args.get("parts") or []) if p]
            if len(parts) >= 2:
                return finish(AgentResult(kind="split", parts=parts, reason="model_split"))
            events.append({"type": "text", "text": "HARNESS: split needs at least two parts."})
            continue
        if n_calls() >= max_calls:
            events.append({"type": "text", "text": "HARNESS: no tool calls left — emit the final plan."})
            continue
        def _same(e):
            if e["tool"] != tool:
                return False
            if tool == "join_path":                 # join_path(a,b) == join_path(b,a)
                ea = e.get("args") or {}
                return {ea.get("a"), ea.get("b")} == {args.get("a"), args.get("b")}
            return e.get("args") == args
        dup = next((e for e in tb.log if _same(e)), None)
        if dup is not None:
            events.append({"type": "text", "text": f"you: {_act_text(obj)}"})
            events.append({"type": "text", "text": f"HARNESS: already called as [{dup['id']}] — use that result "
                                                   f"and emit the final plan."})
            force_final = True
            continue
        cid, r = tb.call(tool, args)
        events.append({"type": "call", "id": cid, "act": _act_text(obj)})
        if tool == "probe":
            an = anomalies(r, None, sm)
            if an:
                events.append({"type": "text", "text": "HARNESS: probe anomaly — " + "; ".join(an)})
