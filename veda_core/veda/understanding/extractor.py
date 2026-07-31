"""veda.understanding.extractor — LLM concept extraction → RawIntent.

The ONLY LLM step in the understanding layer. It extracts the query's structure as
CONCEPTS (never columns). Enterprise properties:
  - temperature 0 + fixed seed  → reproducible decoding
  - strict JSON, with a repair pass (strip fences, slice to the object)
  - shape validation against the typed contract (INTENTS); bad shape → None
  - graceful degradation: ANY failure (SLM down, timeout, unparseable) → None, so the
    caller falls back to the existing pipeline instead of crashing
  - never emits a table/column — output is untrusted concepts, grounded separately
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from veda.understanding.schema import RawIntent, INTENTS

_SYS = (
    "You are a query-understanding module for an analytics engine. Extract the STRUCTURED "
    "INTENT of the user's question as JSON. You describe WHAT the user wants — you do NOT "
    "write SQL and you do NOT invent column names. Use only business concepts from the "
    "question itself.\n"
    "Fields:\n"
    "- intent: one of list, count, sum, avg, max, min, rank, compare, refuse, clarify.\n"
    "- grain: the MAIN entity the answer is about / reported PER / the ranking SUBJECT. "
    "'top 5 PROPERTIES by number of payments' → grain='property'; 'average rent of LEASE "
    "TRANSACTIONS' → grain='lease transaction'; 'payments per user' → grain='user'. Never "
    "the measured thing.\n"
    "- measure: what is aggregated or ranked-by ('number of payment transactions', 'total "
    "paid amount'), or null for a plain list.\n"
    "- dimensions: group-by concepts (a category after 'by/per'); [] if none.\n"
    "- filters: [{concept, value}] the user constrained on; []. NEVER treat English "
    "connectives (with, their, relate, across, of, and) as filters.\n"
    "- entities: every business entity the question mentions.\n"
    "- confidence: 0.0-1.0.\n"
    "\nWHEN TO REFUSE vs CLARIFY vs ANSWER (important):\n"
    "- refuse: the question asks for an attribute/thing that cannot exist in business data "
    "(color, weather, favourite food, shoe size, spaceship). Example: 'color of a payment' "
    "→ refuse. 'sale listings with their spaceship' → refuse.\n"
    "- clarify: the question names NO concrete entity — just a bare measure/attribute with "
    "no subject. 'show the amount' → clarify. 'show rent' → clarify. 'top transactions' → "
    "clarify (which transactions?). 'recent activity' → clarify.\n"
    "- ANSWER (do NOT refuse/clarify) when a clear business entity is named: 'show tickets' "
    "→ list; 'list properties' → list; 'how many users' → count. A plain 'show/list <entity>' "
    "is always answerable — never refuse it.\n"
    "Output ONLY the JSON object."
)


def _repair_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


def _norm_list(v) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if v:
        return [str(v)]
    return []


def extract(query: str, entity_catalog: List[str], *, timeout: int = 60) -> Optional[RawIntent]:
    """Query → RawIntent, or None on ANY failure (graceful degrade).
    `entity_catalog` = business entity names (concepts) the grounding layer knows about —
    the only schema hint given to the LLM, so it grounds toward real concepts."""
    try:
        from slm import call_slm
    except Exception:
        return None
    user = (f"Business entities available (concepts): {', '.join(entity_catalog[:60])}\n\n"
            f"Question: {query}\n\nJSON:")
    try:
        raw_text = call_slm(user, system=_SYS, purpose="query_understanding",
                            temperature=0, seed=0, num_predict=320, num_ctx=2048,
                            timeout=timeout)
    except Exception:
        return None                          # SLM unreachable / timeout → degrade
    obj = _repair_json(raw_text or "")
    if not isinstance(obj, dict):
        return None
    intent = str(obj.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        return None                          # shape invalid → degrade (never guess)
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    ri = RawIntent(
        intent=intent,
        grain=(str(obj["grain"]).strip() if obj.get("grain") else None),
        measure=(str(obj["measure"]).strip() if obj.get("measure") else None),
        dimensions=_norm_list(obj.get("dimensions")),
        filters=[f for f in (obj.get("filters") or []) if isinstance(f, dict)],
        entities=_norm_list(obj.get("entities")),
        confidence=max(0.0, min(1.0, conf)),
        raw=obj,
    )
    return ri if ri.is_valid_shape() else None
