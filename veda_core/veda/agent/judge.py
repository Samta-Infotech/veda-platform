"""veda.agent.judge — the embedding does the checking (planner agent, pass 2).

The validator proves every identifier came from a tool; it cannot tell `created_by` from
"assigned to". Three deterministic checks here can, all on the SAME BGE-M3 encoder and
cross-encoder the retrieval spine uses:

  plan_to_text(plan)   the plan read back as business English ("count of tickets per
                       status, joined to the user who created this ticket")
  judge(...)           sim_q = cosine(question, plan text) (+ the cross-encoder score);
                       the nearest verified / synthetic questions by embedding — when the
                       nearest is close (≥ τ_near) and its entity is not in the plan, that
                       is a DISAGREEMENT. Accept iff sim_q ≥ τ_plan and no disagreement.
  unmapped_phrases(…)  the question's noun phrases no identifier of the plan accounts for
                       (the harness forces a find_entities call on each)
  fewshot(…)           the 2 nearest verified plans, as the model's worked examples

Thresholds come from config (set on evaluation/agent_dev_questions.txt, never the
held-out sets).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veda.agent.plan import Plan, split_ref, accounted_phrases, _norm


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _words(s: str) -> str:
    return re.sub(r"_+", " ", str(s or "")).replace(" id", "").strip()


def _tname(t: str, cards: Dict[str, dict]) -> str:
    c = cards.get(t) or {}
    return c.get("plural") or c.get("business_name") or _words(t.split(".")[-1].split("_", 1)[-1])


def plan_to_text(plan: Plan, cards: Optional[Dict[str, dict]] = None, phrases: Optional[Dict[str, str]] = None) -> str:
    """The plan in business words — deterministic, from the plan and the cards / link
    sentences only."""
    cards = cards or {}
    phrases = phrases or {}
    anchor = plan.anchor or ""
    bits: List[str] = []
    if plan.aggregates:
        aggs = []
        for a in plan.aggregates:
            fn = {"count_distinct": "number of distinct", "count": "number of", "sum": "total",
                  "avg": "average", "min": "minimum", "max": "maximum"}.get(a["fn"], a["fn"])
            what = _words(a["column"]) if a.get("column") else _tname(anchor, cards)
            if a.get("column") and a.get("table") and a["table"] != anchor:
                what += f" of {_tname(a['table'], cards)}"
            aggs.append(f"{fn} {what}")
        bits.append(" and ".join(aggs))
        if a.get("column") or plan.group_by:
            bits.append(f"of {_tname(anchor, cards)}")
    else:
        cols = [f"{_words(p['column'])}" + (f" of the {(cards.get(p['table']) or {}).get('business_name') or _words(p['table'])}"
                                             if p["table"] != anchor else "") for p in plan.projection[:6]]
        bits.append(_tname(anchor, cards) + (f" with {', '.join(cols)}" if cols else ""))
    if plan.group_by:
        bits.append("per " + " and ".join(
            _words(g["column"]) + (f" of the {(cards.get(g['table']) or {}).get('business_name') or _words(g['table'])}"
                                   if g["table"] != anchor else "") for g in plan.group_by))
    rel = []
    for j in plan.joins:
        ph = phrases.get(f"{j.get('from')}.{(j.get('on') or [(None, None)])[0][0]}")
        if ph:
            rel.append(ph)
    if rel:
        bits.append("joined to " + "; ".join(dict.fromkeys(rel)))
    for f in plan.filters:
        v = f.get("value")
        v = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
        op = {"=": "is", "!=": "is not", ">": "above", ">=": "at least", "<": "below", "<=": "at most",
              "in": "in", "between": "between", "is_null": "missing", "is_not_null": "present"}.get(f["op"], f["op"])
        bits.append(f"where {_words(f['column'])} {op} {v if v is not None else ''}".strip())
    if plan.time:
        w = plan.time.get("window") or {}
        bits.append(f"{_words(plan.time['column'])} from {w.get('from') or '…'} to {w.get('to') or '…'}")
    for o in plan.order:
        t, c = split_ref(o["expr"])
        bits.append(f"ordered by {_words(c)} {'descending' if o['dir'] == 'desc' else 'ascending'}")
    if plan.limit:
        bits.append(f"top {plan.limit}")
    return re.sub(r"\s+", " ", " ".join(bits)).strip()


@dataclass
class Verdict:
    ok: bool
    sim_q: float
    ce: Optional[float]
    plan_text: str
    nearest: List[Dict[str, Any]] = field(default_factory=list)
    disagreement: Optional[str] = None
    reason: str = ""

    def to_dict(self):
        return {"ok": self.ok, "sim_q": round(self.sim_q, 3), "ce": (round(self.ce, 3) if self.ce is not None else None),
                "plan_text": self.plan_text, "nearest": self.nearest, "disagreement": self.disagreement,
                "reason": self.reason}


def _encode(texts: List[str]):
    import numpy as np
    from ingestion.m3_encoder import encode_dense
    m = np.asarray(encode_dense(texts), dtype="float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def _ce(q: str, t: str) -> Optional[float]:
    try:
        from query.reranker import _get_reranker
        rr = _get_reranker()
        return float(rr.predict([[q, t[:512]]], batch_size=1)[0]) if rr is not None else None
    except Exception:
        return None


def _ex_table(ex: Dict[str, Any]) -> Optional[str]:
    if ex.get("table"):
        return ex["table"]
    fr = ex.get("frame") or {}
    return ((fr.get("provenance") or {}).get("entity_table")) if isinstance(fr, dict) else None


def judge(question: str, plan: Plan, vocab, *, phrases: Optional[Dict[str, str]] = None,
          tau_plan: Optional[float] = None, tau_near: Optional[float] = None,
          tau_ce: Optional[float] = None) -> Verdict:
    cards = dict(getattr(vocab, "cards", None) or {})
    tau_plan = float(_cfg("AGENT_JUDGE_TAU_PLAN", 0.55) if tau_plan is None else tau_plan)
    tau_near = float(_cfg("AGENT_JUDGE_TAU_NEAR", 0.80) if tau_near is None else tau_near)
    tau_ce = float(_cfg("AGENT_JUDGE_TAU_CE", 0.0) if tau_ce is None else tau_ce)
    text = plan_to_text(plan, cards, phrases)
    m = _encode([question, text])
    sim_q = float(m[0] @ m[1])
    ce = _ce(question, text)
    near: List[Dict[str, Any]] = []
    dis = None
    try:
        from veda.understanding.vocabulary import nearest_examples
        for ex in nearest_examples(vocab, question, k=3):
            t = _ex_table(ex)
            near.append({"question": str(ex.get("question"))[:90], "table": t,
                         "sim": round(float(ex.get("_sim") or 0.0), 3)})
        close = [n for n in near if n["sim"] >= tau_near and n["table"]]
        # a disagreement only when NO close known question uses a table of the plan (the
        # dev set had a tie at 0.807 between a listing-visit and a sale-listing question)
        if close and not any(n["table"] in plan.tables for n in close):
            dis = (f"the nearest known question (\"{close[0]['question']}\", similarity {close[0]['sim']}) "
                   f"is answered from {close[0]['table']}, which your plan does not use")
    except Exception:
        pass
    ce_ok = ce is None or ce >= tau_ce
    ok = sim_q >= tau_plan and ce_ok and dis is None
    low = sim_q < tau_plan or not ce_ok
    why = "" if ok else ("; ".join(x for x in [
        (f"your plan reads as: \"{text}\" — that does not match what the question asks "
         f"(similarity {sim_q:.2f}, cross-encoder {ce if ce is not None else 'n/a'})" if low else ""),
        dis or ""] if x))
    return Verdict(ok, sim_q, ce, text, near, dis, why)


# ── noun phrases the plan does not account for ───────────────────────────────────────
_FUNCTION = None


def _function_words() -> set:
    global _FUNCTION
    if _FUNCTION is None:
        try:
            from veda.validation import _gate_strip
            _FUNCTION = set(_gate_strip())
        except Exception:
            _FUNCTION = set()
        _FUNCTION |= {"a", "an", "the", "of", "in", "on", "to", "for", "is", "are", "do", "does", "did",
                      "it", "its", "by", "with", "and", "or", "all", "any", "what", "which", "who", "how"}
    return _FUNCTION


def data_words(vocab, sm) -> set:
    """Every word the scope's data vocabulary uses (card names / aliases, column names)."""
    key = id(vocab)
    cache = getattr(data_words, "_c", {})
    if key in cache:
        return cache[key]
    out = set()
    for ph in (vocab.name_index() if vocab is not None else {}):
        out |= set(ph.split())
    for k in ((sm or {}).get("columns") or {}):
        out |= set(_norm(k.rsplit(".", 1)[-1].replace("_", " ")).split())
    cache[key] = out
    data_words._c = cache
    return out


def noun_phrases(question: str, vocab=None, sm=None) -> List[str]:
    """Runs of content words (≤ 3). A word the grammar stoplist drops is kept when the
    scope's data vocabulary uses it ('assigned', 'listing', 'amount')."""
    fw = _function_words()
    dw = data_words(vocab, sm) if vocab is not None else set()
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*|\d[\d,.]*", question or "")
    out, cur = [], []
    for t in toks:
        w = t.lower()
        n = _norm(w)
        content = w[0].isalpha() and len(w) > 2 and (w not in fw or n in dw)
        if content:
            cur.append(w)
            if len(cur) == 3:
                out.append(" ".join(cur))
                cur = []
        elif cur:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return list(dict.fromkeys(out))


def unmapped_phrases(question: str, plan: Plan, log, vocab, sm) -> List[str]:
    """Noun phrases none of whose content words an identifier of the plan accounts for
    (accounted_phrases + the plan's own column / table words), and that no find_entities /
    columns / values call has been asked about."""
    acc = set()
    for ph in accounted_phrases(plan, question, log, vocab, sm):
        acc |= set(_norm(ph).split())
    cards = dict(getattr(vocab, "cards", None) or {})
    for t in plan.tables:
        c = cards.get(t) or {}
        for ph in [c.get("business_name") or "", c.get("plural") or ""] + list(c.get("aliases") or []):
            acc |= set(_norm(ph).split())
        acc |= set(_norm(t.split(".")[-1].replace("_", " ")).split())
    for x in plan.projection + plan.group_by + plan.filters:
        acc |= set(_norm(str(x.get("column") or "").replace("_", " ")).split())
        if isinstance(x.get("value"), str):
            acc |= set(_norm(x["value"]).split())
    for a in plan.aggregates:
        acc |= set(_norm(str(a.get("column") or "").replace("_", " ")).split())
    asked = []
    qn = _norm(question)
    for e in log or []:
        ph = (e.get("args") or {}).get("phrase")
        # a lookup OF THIS PHRASE — the automatic call on the whole question does not count
        if e.get("tool") in ("find_entities", "columns", "values") and ph and _norm(ph) != qn \
                and len(_norm(ph).split()) <= 5:
            asked.append(_norm(ph))
    out = []
    for np_ in noun_phrases(question, vocab, sm):
        words = [w for w in _norm(np_).split() if w not in _function_words() or w in data_words(vocab, sm)]
        if not words or all(w in acc for w in words):
            continue
        if any(_norm(np_) in a for a in asked):
            continue
        out.append(np_)
    return out


# ── dynamic few-shot ─────────────────────────────────────────────────────────────────
def fewshot(question: str, vocab, k: int = 2) -> List[str]:
    """The k nearest verified / synthetic plans to the question, one line each."""
    try:
        from veda.understanding.vocabulary import nearest_examples
        exs = nearest_examples(vocab, question, k=k * 3)
    except Exception:
        return []
    out = []
    for ex in exs:
        t = _ex_table(ex)
        fr = ex.get("frame") or {}
        if not t or not isinstance(fr, dict):
            continue
        plan = {"table": t, "agg": fr.get("aggregation") or "none"}
        for key in ("measure", "group_by", "order", "limit"):
            if fr.get(key):
                plan[key] = fr[key]
        if fr.get("filters"):
            plan["filters"] = [[f.get("concept"), f.get("op"), f.get("value")] for f in fr["filters"]][:3]
        import json
        out.append(f"\"{str(ex.get('question'))[:90]}\" → {json.dumps(plan, separators=(',', ':'), ensure_ascii=False)}")
        if len(out) >= k:
            break
    return out
