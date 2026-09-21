"""query/routing_slm.py — bounded SLM ambiguity resolver + output validator (routing Phase 3.4/3.5).

Called ONLY when the deterministic policy returns AMBIGUOUS. The SLM is never the primary router:
it receives the query plus the CANDIDATE sources only (their descriptions + grounded evidence) and
the allowed decision modes, and returns a structured choice. Its output is then deterministically
validated (Phase 3.5) before it can route — an invalid/unsafe output never executes; it degrades to
a clarification.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from query.routing_contracts import (
    CandidateSource, RoutingDecision, ClarificationRequired,
    STATUS_ROUTED, STATUS_NO_MATCH, MODE_SINGLE, MODE_MULTI, MODE_NONE,
    METHOD_SLM, RC_SLM_RESOLVED, RC_INVALID_SLM, RC_AMBIGUOUS, RC_NO_EVIDENCE,
)

_SYSTEM = (
    "You are a bounded source DECISION component for an enterprise data assistant. Given a question "
    "and a FIXED list of candidate data sources (each with a description and the evidence it surfaced), "
    "decide how the question should be answered. Choose only from the given source_ids; never invent "
    "one. Reply with STRICT JSON: {\"decision\": \"SINGLE\"|\"MULTI\"|\"NONE\", "
    "\"selected_source_ids\": [..], \"reason\": \"<short, grounded in the evidence>\"}.\n"
    "Decide as follows:\n"
    "- SINGLE: one source's data can answer the whole question. This is the DEFAULT and most common "
    "case — if a single source is clearly the best fit, pick it, even if other sources are loosely "
    "related.\n"
    "- MULTI: the question genuinely requires COMBINING data from two or more of these sources — it "
    "explicitly compares, relates, joins, or cross-references information that lives in different "
    "sources. Only pick MULTI when a single source truly cannot answer it.\n"
    "- NONE: choose this ONLY when the question is clearly UNRELATED to every source's data (e.g. "
    "weather, jokes, booking travel, general chit-chat). If ANY source could plausibly answer it, do "
    "NOT choose NONE — pick SINGLE (or MULTI). When in doubt between NONE and SINGLE, choose SINGLE.\n"
    "\nEXAMPLES (illustrative — apply the same reasoning to the real candidates):\n"
    "Q: 'how many customers do we have' | sources: [A: customer database (tables: customers, orders)], "
    "[B: product catalog] → {\"decision\":\"SINGLE\",\"selected_source_ids\":[\"A\"],\"reason\":\"customer "
    "count lives entirely in the customer database\"}  (a plain count/list of one entity is SINGLE, even "
    "if phrased generically.)\n"
    "Q: 'total refund amount per region where we have stores' | sources: [A: refunds ledger (refund "
    "amount, store_id)], [B: store directory (store_id, region)] → {\"decision\":\"MULTI\","
    "\"selected_source_ids\":[\"A\",\"B\"],\"reason\":\"refund amounts are in A but the region-per-store "
    "mapping is in B; the answer must JOIN them\"}  (when the question relates a fact in one source to a "
    "grouping/filter that only exists in ANOTHER source, it is MULTI.)\n"
    "Q: 'what is the capital of France' | sources: [A: sales DB], [B: HR docs] → "
    "{\"decision\":\"NONE\",\"selected_source_ids\":[],\"reason\":\"unrelated to any source\"}."
)


# V2 few-shot (flag ROUTING_SLM_MULTI_FEWSHOT_ENABLED). MINIMAL: the ORIGINAL prompt verbatim, plus
# just TWO extra MULTI examples (set-intersection, existence-filter). The first V2 attempt rewrote the
# whole prompt with a long "requiredness test" paragraph + 4 examples; the A/B showed the 7B could not
# absorb the longer/denser prompt (clarification 2→13, NONE 3→15 — format breakage + over-caution). So
# this keeps the prompt SHORT and only adds MULTI pattern coverage. No keywords/thresholds.
_SYSTEM_V2 = _SYSTEM + (
    "\nQ: 'which of our products are also on the recall list' | [A: product catalog (product_id)], "
    "[B: recall notices (product_id)] → {\"decision\":\"MULTI\",\"selected_source_ids\":[\"A\",\"B\"],"
    "\"reason\":\"the intersection matches product ids in A against B\"}  (a MATCH/intersection across "
    "two sources is MULTI.)\n"
    "Q: 'list suppliers located in cities where we operate warehouses' | [A: suppliers (city)], "
    "[B: warehouses (city)] → {\"decision\":\"MULTI\",\"selected_source_ids\":[\"A\",\"B\"],\"reason\":"
    "\"suppliers in A filtered by warehouse cities that only B knows\"}  (an EXISTENCE filter by an "
    "attribute only in another source is MULTI.)"
)


def _system_prompt():
    """Select the routing-decision system prompt. Default (flag OFF) is the original _SYSTEM →
    byte-identical. ROUTING_SLM_MULTI_FEWSHOT_ENABLED swaps in the sharper V2 few-shot."""
    try:
        from config import ROUTING_SLM_MULTI_FEWSHOT_ENABLED
        return _SYSTEM_V2 if ROUTING_SLM_MULTI_FEWSHOT_ENABLED else _SYSTEM
    except Exception:
        return _SYSTEM


def _card_block(c: "CandidateSource") -> str:
    """This candidate's ROUTING CARD as prompt text, or "" when it has none.

    The card (ingestion/routing_card.py, built by L5) is what the source IS: its main
    entities, how they are named in business terms, what they can be grouped and
    measured by, real sample values, and which other sources it can be JOINED to. The
    evidence block below it is what THIS question matched. The model needs both — the
    evidence alone tells it which columns lit up but never what the source is for, and
    "can these two be combined at all" is a structural fact no per-question evidence
    carries.

    Best-effort: a source ingested before cards existed has none, and this returns ""
    so the prompt is exactly what it was before.
    """
    try:
        from config import ROUTING_CARDS_ENABLED
        if not ROUTING_CARDS_ENABLED:
            return ""       # A/B switch for scripts/eval_routing_cards.py
    except Exception:
        pass
    try:
        from ingestion.routing_card import load_routing_card, render_card
    except Exception:
        return ""
    try:
        from veda_core.context import current_source_profiles as _csp
    except Exception:
        try:
            from context import current_source_profiles as _csp
        except Exception:
            _csp = lambda: {}
    card = load_routing_card(c.source_id)
    if not card:
        return ""
    profile = (_csp() or {}).get(str(c.source_id)) or {}
    # the candidate's own already-resolved profile fields win when the ambient
    # profiles map is empty (a CLI/eval call with no request context)
    profile = {**{"source_type": c.source_type, "domain_tags": list(c.domain_tags or [])},
               **{k: v for k, v in profile.items() if v}}
    try:
        return render_card(card, profile=profile)
    except Exception:
        return ""


def _build_user_message(query: str, candidates: List[CandidateSource],
                        prior: Optional[dict] = None) -> str:
    lines = [f"QUESTION: {query}", ""]
    if prior:
        # A stated PRIOR, not an instruction. In a multi-turn session the previous turn
        # already established which source the conversation is on, and a follow-up
        # ("of those, only the Mumbai ones") carries almost no routing signal of its own
        # — scored cold it looks like a new, vague question. Saying what the last turn
        # answered from lets the model keep the thread when the new message is a
        # continuation, while leaving it free to move when the message is genuinely
        # about something else. It is deliberately phrased as context the model may
        # override, never as "use this source".
        _ps = ", ".join(str(x) for x in (prior.get("source_ids") or []))
        if _ps:
            _anchor = prior.get("anchor")
            lines.append(f"CONTEXT: the previous turn in this conversation was answered "
                         f"from source {_ps}"
                         + (f" (about {_anchor})" if _anchor else "")
                         + ". If this question continues that thread, prefer that source; "
                           "if it is about something else, ignore this.")
            lines.append("")
    lines.append("CANDIDATE SOURCES (choose only from these source_ids):")
    for c in candidates:
        ev = c.evidence_summary or {}
        cols = ", ".join(ev.get("columns", [])[:8])
        docs = ", ".join(ev.get("documents", [])[:5])
        card = _card_block(c)
        if card:
            lines.append(card)
        else:
            lines.append(f"- source_id={c.source_id} type={c.source_type} domains={c.domain_tags}")
            if ev.get("description"):
                lines.append(f"    description: {ev['description']}")
            for it in (ev.get("items") or [])[:3]:
                lines.append(f"    item: {it.get('name')} — {it.get('summary')}")
        # evidence is per-QUESTION and always shown: it is what this question matched
        if cols:
            lines.append(f"    matched columns: {cols}")
        if docs:
            lines.append(f"    matched documents: {docs}")
    lines.append("")
    lines.append('Allowed decisions: SINGLE, MULTI, NONE. Reply with STRICT JSON only.')
    return "\n".join(lines)


def _default_slm_call(system: str, user: str) -> str:
    """Real SLM call through the shared choke-point. Lazily imported so this module stays import-cheap
    and the call is monkeypatchable in tests.

    Bounded on all three axes, because this call sits in front of EVERY ambiguous
    question and a hung one stalls the whole turn:
      * num_ctx  — the configured window, never the model default (a request that
                   omits it makes Ollama re-load the model at a different context
                   length than the sized calls around it).
      * num_predict — the reply is one small JSON object ({decision, selected_source_ids,
                   reason}); without a cap the model may keep generating prose past it.
      * timeout  — ROUTING_SLM_TIMEOUT_SECS (default 20s), not the shared
                   SLM_TIMEOUT_SECS of 240s. Routing is a pre-answer decision with a
                   deterministic fallback (validate_slm_decision degrades to
                   clarification), so waiting four minutes on it is strictly worse
                   than degrading: the fallback is instant and safe.
    Temperature is left at call_slm's 0.0 default — a routing decision must not vary run to run."""
    from slm._call_slm import call_slm
    try:
        from config import SLM_NUM_CTX, ROUTING_SLM_TIMEOUT_SECS
    except Exception:  # importable outside the engine cwd (unit tests)
        SLM_NUM_CTX, ROUTING_SLM_TIMEOUT_SECS = 4096, 20
    return call_slm(user, system=system, purpose="source_routing",
                    num_ctx=SLM_NUM_CTX, num_predict=192,
                    timeout=ROUTING_SLM_TIMEOUT_SECS)


def _decision_field(parsed):
    """Read the decision, tolerating the old `mode` key or the new `decision` key."""
    return parsed.get("decision", parsed.get("mode")) if isinstance(parsed, dict) else None


def _normalize_sid(v):
    """One selected id as the bare source_id string the candidate set is keyed by.

    _build_user_message lists candidates as "- source_id=4 type=... domains=...", so a small
    model very reasonably answers "source_id=4" instead of "4". validate_slm_decision then found
    it in no candidate set and rejected the whole decision as invalid, and the caller degraded to
    "Which data source should answer? 2, 3, 4, 5. (selected non-candidate source(s):
    ['source_id=4'])" — a clarification asking the user to pick the source the model had ALREADY
    picked correctly. Stripping the prompt's own label is not guessing: the id still has to exist
    in candidate_ids afterwards, so a genuinely invented source is rejected exactly as before.
    """
    t = str(v).strip().strip('"\'')
    if "=" in t:                       # "source_id=4" / "source_id = 4"
        t = t.rsplit("=", 1)[1].strip()
    for pre in ("source_id:", "source_id", "source:", "source", "src_", "src"):
        if t.lower().startswith(pre):  # "source_id 4", "src_4", "source 4"
            t = t[len(pre):].strip(" :_-")
            break
    return t or str(v).strip()


def _selected_field(parsed):
    """Read the selection, tolerating `selected_source_ids` or the old `source_ids`."""
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("selected_source_ids", parsed.get("source_ids"))
    if isinstance(raw, list):
        return [_normalize_sid(s) for s in raw]
    return raw


def validate_slm_decision(parsed, candidate_ids: set) -> Tuple[bool, str]:
    """Strict deterministic validation of the bounded SLM's output. Returns (ok, reason).

    Contract (SINGLE | MULTI | NONE):
      - decision must be one of the three;
      - NONE must select NO source;
      - SINGLE must select exactly ONE candidate; MULTI at least TWO;
      - every selected id must be within the supplied candidate set (never a new/invented source).
    Invalid output is rejected (ok=False) so the caller degrades to a controlled clarification/refusal —
    it never falls back to an arbitrary source.
    """
    if not isinstance(parsed, dict):
        return False, "output is not an object"
    decision = _decision_field(parsed)
    if decision not in (MODE_SINGLE, MODE_MULTI, MODE_NONE):
        return False, f"invalid decision {decision!r}"
    sids = _selected_field(parsed)
    if decision == MODE_NONE:
        if sids:
            return False, "NONE must not select any source"
        return True, "none"
    if not isinstance(sids, list) or not sids:
        return False, "selected_source_ids must be a non-empty list"
    sids = [str(s) for s in sids]
    unknown = [s for s in sids if s not in candidate_ids]
    if unknown:
        return False, f"selected non-candidate source(s): {unknown}"
    if decision == MODE_SINGLE and len(sids) != 1:
        return False, "SINGLE must select exactly one source"
    if decision == MODE_MULTI and len(sids) < 2:
        return False, "MULTI must select at least two sources"
    return True, "ok"


def _parse(raw: str):
    """Best-effort JSON parse of the SLM reply (tolerates code fences / surrounding prose)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if "```" in s:
        s = s.split("```")[1].lstrip("json").strip() if s.count("```") >= 2 else s
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a:b + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def resolve_boundary(query: str, candidates: List[CandidateSource], *,
                     slm_call=None, query_id: str = "", trace_id: str = "",
                     prior: Optional[dict] = None) -> RoutingDecision:
    """The bounded SEMANTIC decision boundary (P1). The coordinator calls this ONLY for
    evidence-grounded boundary cases — never for high-confidence deterministic SINGLE/structural MULTI.
    The SLM sees only the query + candidate sources (descriptions + evidence summaries) and returns
    SINGLE | MULTI | NONE; the output is strictly validated before it can route.

    Controlled outcomes:
      - SINGLE / MULTI (valid)  → a ROUTED decision (decision_method=slm).
      - NONE (valid)            → a NO_MATCH decision (out of scope) — no source is executed.
      - invalid / SLM error / unparseable → CLARIFICATION_REQUIRED — never a silent fallback source.
    """
    slm_call = slm_call or _default_slm_call
    cand_by_id = {c.source_id: c for c in candidates}
    cand_ids = set(cand_by_id)

    def _clarify(reason_code: str, question: str) -> RoutingDecision:
        return ClarificationRequired(
            reason_code=reason_code, candidate_sources=list(candidates), question=question,
            query_id=query_id, trace_id=trace_id).to_decision()

    try:
        raw = slm_call(_system_prompt(), _build_user_message(query, candidates, prior=prior))
    except Exception as e:  # noqa: BLE001
        return _clarify(RC_INVALID_SLM,
                        f"Could not decide the source automatically ({type(e).__name__}).")

    parsed = _parse(raw)
    ok, reason = validate_slm_decision(parsed, cand_ids)
    if not ok:
        return _clarify(RC_INVALID_SLM,
                        "Which data source should answer? " +
                        ", ".join(sorted(cand_ids)) + f". ({reason})")

    decision = _decision_field(parsed)
    if decision == MODE_NONE:
        # 2026-09-18: the SLM says "no source" at a boundary the EVIDENCE put it at — i.e.
        # at least one candidate is STRONG. A NONE against a tabular source with a real
        # table/column hit ("maintenance records per vendor" → maintenance 0.59) is the
        # weaker signal (measured: 2 of the 12 probe questions were lost exactly here).
        # Fall back to the deterministic leader when it is TABULAR and leads every other
        # candidate by the compete window; otherwise honour the NONE (document-led ties,
        # or no clear leader, still refuse — never a silent fallback source).
        try:
            from config import ROUTING_COMPETE_WINDOW as _win, ROUTING_TIER_TABULAR_STRONG as _abs
        except Exception:
            _win, _abs = 0.08, 0.55
        _tab = [c for c in candidates if (c.source_type or "").lower() not in ("document", "doc", "filesystem")]
        _lead = max(candidates, key=lambda c: getattr(c, "top_score", 0.0)) if candidates else None
        # the leader must be strong on its OWN evidence (absolute floor), not merely the
        # least-weak of a weak field: a 1900-column source always has SOME 0.5 hit, and a
        # nonsense concept ("gizmos") must stay NO_MATCH (probe, 2026-09-18)
        if (_lead is not None and _lead in _tab and getattr(_lead, "presence_tier", "") == "STRONG"
                and getattr(_lead, "top_score", 0.0) >= float(_abs)):
            _others = [getattr(c, "top_score", 0.0) for c in candidates if c is not _lead]
            if not _others or getattr(_lead, "top_score", 0.0) >= max(_others) + float(_win) / 2:
                return RoutingDecision(
                    status=STATUS_ROUTED, mode=MODE_SINGLE, source_ids=[_lead.source_id],
                    candidate_sources=list(candidates),
                    evidence_summary=[_lead.evidence_summary] if _lead.evidence_summary else [],
                    decision_method=METHOD_SLM, reason_code=RC_SLM_RESOLVED,
                    reason=("SLM answered NONE at an evidence boundary; the tabular leader "
                            f"(source {_lead.source_id}) holds the strongest signal — routed to it."),
                    validation_status="passed", query_id=query_id, trace_id=trace_id)
        return RoutingDecision(
            status=STATUS_NO_MATCH, mode=MODE_NONE, candidate_sources=list(candidates),
            decision_method=METHOD_SLM, reason_code=RC_NO_EVIDENCE,
            reason=parsed.get("reason") or "No source is relevant to this query.",
            validation_status="passed", query_id=query_id, trace_id=trace_id)

    chosen = [cand_by_id[str(s)] for s in _selected_field(parsed)]
    return RoutingDecision(
        status=STATUS_ROUTED, mode=decision, source_ids=[c.source_id for c in chosen],
        candidate_sources=list(candidates),
        evidence_summary=[c.evidence_summary for c in chosen if c.evidence_summary],
        decision_method=METHOD_SLM, reason_code=RC_SLM_RESOLVED,
        reason=parsed.get("reason") or "Resolved by the bounded SLM from the candidate set.",
        validation_status="passed", query_id=query_id, trace_id=trace_id)


# Backwards-compatible alias (older call sites / tests used the ambiguity name).
resolve_ambiguity = resolve_boundary
