"""VEDA · Failure feedback — turn a refusal/error into actionable guidance.

Refuse-over-guess stays the policy; this only makes a refusal USEFUL: a plain-language
WHY + WHAT's-needed + concrete SUGGESTIONS pulled from the real schema/data (valid
column values, closest tables). Deterministic by default and the source of truth. An
optional, gated LLM pass (FEEDBACK_LLM_POLISH) only REPHRASES these facts — it never
invents values/tables — and the deterministic text is the guaranteed fallback.
"""
import re
from veda.runtime import get_db_config


def _distinct_values(table, column, limit=8):
    """Read-only sample of a column's ACTUAL values, for 'did you mean' suggestions."""
    if not table or not column:
        return []
    try:
        import psycopg2
        cfg = get_db_config()
        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"], dbname=cfg["database"],
            user=cfg["user"], password=cfg["password"])
        try:
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 5000")
                cur.execute(f'SELECT DISTINCT "{column}"::text FROM "{table}" '
                            f'WHERE "{column}" IS NOT NULL LIMIT %s', (limit,))
                return [r[0] for r in cur.fetchall() if r[0] is not None]
        finally:
            conn.close()
    except Exception:
        return []


def _closest(name, options, n=3):
    """Cheap dependency-free ranking: shared word tokens + substring overlap."""
    name = (name or "").lower()
    ntoks = set(re.findall(r"[a-z]+", name))
    scored = []
    for o in options:
        ol = str(o).lower()
        s = len(ntoks & set(re.findall(r"[a-z]+", ol)))
        if name and (name in ol or ol in name):
            s += 1
        if s:
            scored.append((s, o))
    scored.sort(key=lambda x: -x[0])
    return [o for _, o in scored[:n]]


def explain_failure(status, sm, *, column=None, value=None, missing=None,
                    candidates=None, msg=None, error=None):
    """Return {why, what_needed, suggestions, text} for a non-answered status.

    Fully deterministic. `text` is the user-facing block. Never raises."""
    sm = sm or {}
    why = what = ""
    sugg = []

    if status == "ungrounded":
        tbl, _, col = (column or "").partition(".")
        vals = _distinct_values(tbl, col)
        why = f"'{value}' is not a value that exists in {column}."
        what = "Use a value that's actually present in that column."
        sugg = _closest(value, vals) or vals[:6]
    elif status == "no_table":
        why = "I couldn't confidently match your question to a table in the schema."
        what = "Name the entity you're asking about, or pick a closest match."
        sugg = (candidates or [])[:5]
    elif status == "qualifier_dropped":
        cols = [k.split(".", 1)[1] for k in sm.get("columns", {})]
        why = f"I couldn't map '{missing}' to any column or value in the data."
        what = (f"Tell me what '{missing}' refers to — a column, or a value to filter on?")
        sugg = _closest(missing, cols)
    elif status == "clarify":
        why = msg or "Your question is ambiguous about which entity it's about."
        what = "Re-ask naming the subject explicitly (e.g. 'for each <entity> …')."
    elif status == "refuse":
        why = msg or "I can't answer this correctly with the available schema."
        what = "Rephrase, or ask about entities that are related in the schema."
    elif status == "ir_mismatch":
        _m = (msg or "").lower()
        if "filter" in _m:
            why = (f"The generated SQL added a filter the question didn't ask for ({msg}); "
                   f"I refused rather than invent a condition you didn't state.")
            what = ("If you DO want that condition, state it explicitly — otherwise the "
                    "answer is the unfiltered total.")
        elif "group by" in _m:
            why = (f"The generated SQL grouped the results when the question didn't ask "
                   f"for a breakdown ({msg}).")
            what = "Say 'per/by <X>' for a grouped count, or ask for the plain total."
        elif "order by" in _m:
            why = (f"The generated SQL sorted the results when the question didn't ask "
                   f"for ranking ({msg}).")
            what = "Add 'top N' / 'highest' / 'sorted by <X>' if you want ordering."
        else:
            why = f"The generated SQL added semantics the question didn't ask for ({msg})."
            what = "Rephrase to state exactly the condition/shape you want."
    elif status in ("invalid", "exec_error", "tier2_rejected", "tier2_exec_error"):
        why = "I built a query but it didn't pass the safety/validity checks."
        what = "Try rephrasing more simply, or split a complex request into parts."
    else:
        why = msg or "I couldn't answer this query."
        what = "Try rephrasing it."

    lines = [f"❌ {why}"]
    if what:
        lines.append(f"   → {what}")
    if sugg:
        lines.append(f"   Did you mean: {', '.join(str(s) for s in sugg)}")
    text = "\n".join(lines)

    out = {"why": why, "what_needed": what, "suggestions": sugg, "text": text}

    try:
        from config import FEEDBACK_LLM_POLISH
    except Exception:
        FEEDBACK_LLM_POLISH = False
    if FEEDBACK_LLM_POLISH:
        polished = _polish(out)
        if polished:
            out["text"] = polished
    return out


def _polish(facts):
    """Rephrase the structured facts into one friendly line — add NOTHING. None on any
    failure (caller keeps the deterministic text)."""
    try:
        import json
        from slm import call_slm
        system = ("You rephrase database-query failure facts into ONE short, friendly "
                  "sentence for the user. Use ONLY the given facts — never invent column "
                  "names, values, or tables. If suggestions are given, keep them verbatim. "
                  "Output one sentence, no markdown.")
        user = json.dumps({k: facts[k] for k in ("why", "what_needed", "suggestions")})
        txt = call_slm(user, system=system, purpose="refusal_polish",
                       temperature=0.1, num_predict=96, timeout=30).strip()
        return txt or None
    except Exception:
        return None
