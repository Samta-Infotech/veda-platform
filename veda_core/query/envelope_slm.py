"""VEDA · One-call intent-envelope SLM emission (warm-env).

A SINGLE JSON-constrained Ollama call that emits the frozen intent envelope
(INTENT_ENVELOPE_CONTRACT.md) — replacing the 4 sequential LangGraph nodes. The model sees
opaque handles (t1/c3), classifies the intent by MEANING, and never writes SQL. The
deterministic `map_envelope_to_intent` mapper turns the envelope into a QueryIntent.

Self-contained + graceful: any failure (Ollama down, bad JSON) → (None, handle_map) so the
caller falls back to the existing IR→sql_builder path. Offline this returns None (no Ollama),
which is exactly the fallback contract.
"""
import os
import re
import json
import urllib.request

from config import (SLM_MODEL_NAME, SLM_OLLAMA_BASE_URL, SLM_TIMEOUT_SECS,
                    SLM_IR_MAX_TOKENS, SLM_NUM_CTX, SEMANTIC_MODEL_FILE)

_SYSTEM = """You convert a database question into ONE intent envelope (JSON). You do NOT write SQL.
Choose exactly ONE intent by MEANING (not keywords), then fill only the fields that intent needs, using ONLY the given handles.

intents:
  count          - how many rows / how many <entity>
  measure        - a sum or average of a numeric column ("total X", "average X")
  ratio          - what share / percentage / rate of rows meet a condition
  trend          - a count over time buckets ("X per month", "X over time")
  compare        - this period vs last period ("X this month vs last month")
  group          - a count (or measure) broken down BY a category ("X by status")
  dimension_list - list the distinct values of a category ("what statuses exist")

rules:
  - entity: the handle of the ONE table the question is about.
  - Refer to tables/columns ONLY by their given handles (t1, c3). Never invent a handle or name. If a needed column is absent, omit that field.
  - filters: categorical conditions named. value = copy the user's literal word exactly. op = "eq" or "ne".
  - measure: {"col":"c?","agg":"sum|avg"} (for measure; optional for group). col must be a numeric column.
  - ratio: {"col":"c?","value":"..."} - the condition whose share you want.
  - group_col: category handle to group/list by.
  - time_bucket: day|week|month|quarter|year (trend). compare_unit: week|month|year (compare). Do NOT compute dates. Do NOT add date filters.
  - If the question needs TWO different tables joined, return {"intent":"count","entity":"<best single table>"} only.
  - Output ONLY the JSON object. No explanation, no SQL, no markdown.

examples:
Q: "how many level 1 incidents" -> {"intent":"count","entity":"t1","filters":[{"col":"c1","value":"Level 1","op":"eq"}]}
Q: "what percent of incidents are escalated" -> {"intent":"ratio","entity":"t1","ratio":{"col":"c2","value":"Escalated"}}
Q: "incidents by status" -> {"intent":"group","entity":"t1","group_col":"c2"}
Q: "incident count per month" -> {"intent":"trend","entity":"t1","time_bucket":"month"}
Q: "escalation rate" -> {"intent":"ratio","entity":"t1","ratio":{"col":"c2","value":"Escalated"}}
"""

_TYPE_TAG = {"CATEGORY": "category", "TEMPORAL": "time", "IDENTIFIER": "id",
             "METRIC": "numeric", "MONETARY": "numeric", "FREE_TEXT": "text"}

_SM_CACHE = {"v": None}


def _sample_values(col_id):
    if _SM_CACHE["v"] is None:
        try:
            path = SEMANTIC_MODEL_FILE if os.path.isabs(SEMANTIC_MODEL_FILE) else \
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), SEMANTIC_MODEL_FILE)
            _SM_CACHE["v"] = json.load(open(path)).get("columns", {})
        except Exception:
            _SM_CACHE["v"] = {}
    return (_SM_CACHE["v"].get(col_id, {}) or {}).get("sample_values") or []


def _entity_candidates(query, sel_columns):
    """The ENTITY (subject) candidates offered to the LLM = the schema CONCEPTS the query
    names (registry.match_concepts on the query tokens), intersected with what retrieval
    surfaced. This removes the reranker's column-relevance bias from subject selection (a
    column like organizations.is_active can no longer become the subject just because it
    reranked #1) WITHOUT computing the subject ourselves — the LLM still picks, but from a
    clean concept set. Falls back to retrieved tables when no concept is grounded.

    Returned ALPHABETICALLY by table name: a deterministic, reproducible, content-neutral
    order (alpha has no correlation with subject-likelihood, so it injects no relevance
    prior — unlike reranker order, and unlike a shuffle it stays reproducible)."""
    retrieved_tables = list(dict.fromkeys(r.table_name for r in sel_columns if r.table_name))
    try:
        from semantic import registry as reg
        reg.load()
        hits = reg.match_concepts(reg.query_tokens(query))     # [(concept, score), ...]
        concept_tables = [c["resolves_to"]["table"] for c, _ in hits
                          if c.get("resolves_to", {}).get("table")]
        # keep only concepts retrieval also surfaced (answerable), preserve nothing-order
        grounded = [t for t in concept_tables if t in retrieved_tables]
        cands = grounded or concept_tables or retrieved_tables
    except Exception:
        cands = retrieved_tables
    return sorted(set(cands))                                  # alphabetical = neutral


def build_handles(query, sel_columns):
    """Assign opaque handles → (handle_map, display_text). Tables (t1..) = query-grounded
    concept candidates in NEUTRAL (alphabetical) order; columns (c1..) = retrieved columns
    (kept for filters/measure/group). The LLM picks the subject from the concept set; the
    handle order carries no relevance signal."""
    handle_map, tlines, clines = {}, [], []
    for i, t in enumerate(_entity_candidates(query, sel_columns), 1):
        h = f"t{i}"
        handle_map[h] = {"table": t, "col": None}
        tlines.append(f"  {h} = {t}")
    for i, r in enumerate(sel_columns, 1):
        h = f"c{i}"
        handle_map[h] = {"table": r.table_name, "col": r.col_name}
        tag = _TYPE_TAG.get((r.semantic_type or "").upper(), "")
        vals = _sample_values(f"{r.table_name}.{r.col_name}")[:3]
        ex = f"  e.g. {', '.join(map(str, vals))}" if vals else ""
        clines.append(f"  {h} = {r.table_name}.{r.col_name}   [{tag}]{ex}")
    display = ("Tables (candidates for the subject — no particular order):\n"
               + "\n".join(tlines) + "\nColumns:\n" + "\n".join(clines))
    return handle_map, display


def emit_envelope(query, sel_columns, verbose=False):
    """One JSON-constrained Ollama call → (envelope_dict | None, handle_map)."""
    handle_map, display = build_handles(query, sel_columns)
    user = f'Question: "{query}"\n\n{display}\nReturn the envelope JSON.'
    payload = json.dumps({
        "model": SLM_MODEL_NAME,
        "stream": False,
        "format": "json",
        "keep_alive": "24h",
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.0, "num_predict": SLM_IR_MAX_TOKENS, "num_ctx": SLM_NUM_CTX},
    }).encode("utf-8")
    url = f"{SLM_OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=SLM_TIMEOUT_SECS) as resp:
            content = json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")
        content = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
        s, e = content.find("{"), content.rfind("}") + 1
        env = json.loads(content[s:e]) if s >= 0 and e > s else None
        if verbose:
            print(f"  [Envelope] {env}")
        return (env if isinstance(env, dict) else None), handle_map
    except Exception as ex:
        if verbose:
            print(f"  [Envelope] emission failed ({type(ex).__name__}) — fallback to IR path")
        return None, handle_map
