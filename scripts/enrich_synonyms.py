#!/usr/bin/env python3
"""
enrich_synonyms.py — TARGETED business-vocabulary synonym generation.

Closes the vocabulary gap that makes analytical queries refuse: the semantic model's
generated `domain_synonyms` covers low-level column paraphrases ("txn category", "payer
name") but NOT the high-level business terms users actually type ("property", "listing",
"financial value", "payment amount", "revenue"). So "maximum financial value per property"
never resolves to a metric/entity and refuses.

This does an SLM pass over ONLY the ENTITY tables and MEASURE columns (a few hundred items,
not all ~1900 columns), asking for the everyday business words a user would use, then MERGES
the result into veda_domain_synonyms.json. It reuses the already-persisted semantic model
(table/column names + business_purpose/business_definition + sample values) — it does NOT
re-run the Stage 3/4 table/column understanding. After it finishes, recompile the registries
(compile_semantic_layer) and republish to Redis.

Resumable: every generated item is cached to <data>/synonym_enrich_cache.json, so a killed
run resumes instead of restarting.

Run (inside the inference container, which reaches the host-Metal Ollama via OLLAMA_URL):
    python3 scripts/enrich_synonyms.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

# engine imports resolve from cwd=veda_core (bare) or PYTHONPATH=/app (package)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "veda_core"))

import config  # noqa: E402
from slm import call_slm  # noqa: E402
from semantic.compile_semantic_layer import build_concepts  # noqa: E402

CACHE_PATH = os.path.join(os.path.dirname(config.DOMAIN_SYNONYMS_FILE) or ".",
                          "synonym_enrich_cache.json")

# Domain context so generic descriptions ("asset") map to the platform's real vocabulary
# ("property"). Override via SYNONYM_DOMAIN env.
DOMAIN = os.environ.get("SYNONYM_DOMAIN",
                        "a real-estate / property management platform (properties, sale "
                        "listings, leases, tenants, payments, accounting)")

_ENTITY_SYS = (
    f"You work on {DOMAIN}. You map a database table to the everyday BUSINESS words end-users "
    "type when they ask about those records. Output ONLY a compact JSON array of 3-6 short "
    "lowercase noun phrases — no explanation, no keys, no aggregation words.")
_MEASURE_SYS = (
    f"You work on {DOMAIN}. You map a numeric column to the business NOUN NAMES end-users use "
    "for that value. Output ONLY a compact JSON array of 3-6 short lowercase NOUN phrases. "
    "Do NOT include aggregation words (total/sum/average/max/min/highest/largest/smallest).")


def _clean(items) -> list[str]:
    out = []
    for t in items:
        t = str(t).lower().strip()
        if 2 < len(t) <= 40 and re.fullmatch(r"[a-z0-9 /&-]+", t):
            out.append(t)
    return list(dict.fromkeys(out))


def _parse_arr(text: str) -> list[str]:
    text = text or ""
    # 1) a JSON array anywhere
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            return _clean(json.loads(m.group()))
        except Exception:
            pass
    # 2) a JSON object → take list values (model sometimes wraps despite instructions)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            vals = []
            for v in obj.values():
                vals += v if isinstance(v, list) else [v]
            if vals:
                return _clean(vals)
        except Exception:
            pass
    # 3) fallback: extract quoted strings
    q = re.findall(r'["\']([a-z0-9 /&-]{3,40})["\']', text.lower())
    return _clean(q)


def gen_entity(table: str, purpose: str) -> list[str]:
    prompt = (f"Table: {table}\nWhat it stores: {purpose or '(no description)'}\n"
              f"Business words a user would use to refer to ONE of these records "
              f"(e.g. a sale-listing table → [\"property\",\"listing\",\"property for sale\"]).")
    try:
        return _parse_arr(call_slm(prompt, system=_ENTITY_SYS, purpose="synonym_gen",
                                   temperature=0.2, num_predict=120, json_format=True, timeout=30))
    except Exception as e:
        print(f"    entity slm err {table}: {e}", flush=True)
        return []


def gen_measure(col_id: str, cname: str, defn: str, tpurpose: str) -> list[str]:
    prompt = (f"Column: {col_id}\nColumn means: {defn or cname.replace('_',' ')}\n"
              f"Table context: {tpurpose or ''}\n"
              f"The business NOUN NAME(S) a user calls this value "
              f"(e.g. a transaction amount column → [\"amount\",\"financial value\","
              f"\"transaction amount\",\"payment amount\",\"money\"]). Nouns only.")
    try:
        return _parse_arr(call_slm(prompt, system=_MEASURE_SYS, purpose="synonym_gen",
                                   temperature=0.2, num_predict=120, json_format=True, timeout=30))
    except Exception as e:
        print(f"    measure slm err {col_id}: {e}", flush=True)
        return []


def main():
    with open(config.SEMANTIC_MODEL_FILE) as f:
        sm = json.load(f)
    tabs = sm.get("tables", {})
    cols = sm.get("columns", {})
    concepts = build_concepts(sm)                      # entity tables (same set the engine uses)

    # cache {key -> [phrases]}  (key = "E:<table>" or "M:<col_id>")
    cache: dict = {}
    if os.path.exists(CACHE_PATH):
        try:
            cache = json.load(open(CACHE_PATH))
        except Exception:
            cache = {}

    entity_tables = list(concepts.keys())
    measure_cols = [(cid, c) for cid, c in cols.items() if c.get("analytics_role") == "MEASURE"
                    and c.get("table_name") in concepts]
    total = len(entity_tables) + len(measure_cols)
    print(f"targets: {len(entity_tables)} entities + {len(measure_cols)} measure cols = {total}",
          flush=True)

    done = 0
    t0 = time.time()

    def _flush():
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

    for t in entity_tables:
        done += 1
        k = f"E:{t}"
        if k in cache:
            continue
        cache[k] = gen_entity(t, (tabs.get(t) or {}).get("business_purpose", ""))
        if done % 10 == 0:
            _flush(); print(f"[{done}/{total}] {t} -> {cache[k]}", flush=True)
    _flush()

    for cid, c in measure_cols:
        done += 1
        k = f"M:{cid}"
        if k in cache:
            continue
        t = c.get("table_name", "")
        cache[k] = gen_measure(cid, c.get("col_name", ""),
                               c.get("business_definition", ""),
                               (tabs.get(t) or {}).get("business_purpose", ""))
        if done % 10 == 0:
            _flush(); print(f"[{done}/{total}] {cid} -> {cache[k]}", flush=True)
    _flush()

    # ── merge into domain_synonyms (phrase -> [col_id]) ────────────────────────
    ds = {}
    if os.path.exists(config.DOMAIN_SYNONYMS_FILE):
        try:
            ds = json.load(open(config.DOMAIN_SYNONYMS_FILE))
        except Exception:
            ds = {}

    def _add(phrase: str, col_id: str):
        phrase = phrase.lower().strip()
        if not phrase:
            return
        ds.setdefault(phrase, [])
        if col_id not in ds[phrase]:
            ds[phrase].append(col_id)

    added = 0
    for k, phrases in cache.items():
        kind, ref = k.split(":", 1)
        # entity synonyms attach to the table's pk column (a real col_id) so both the
        # retrieval enrichment AND build_concepts (reverse-indexed by table) pick them up.
        target = f"{ref}.id" if kind == "E" else ref
        for p in phrases:
            before = len(ds.get(p, []))
            _add(p, target)
            added += len(ds.get(p, [])) - before

    with open(config.DOMAIN_SYNONYMS_FILE, "w") as f:
        json.dump(ds, f)
    print(f"\nDONE in {round(time.time()-t0)}s. domain_synonyms now {len(ds)} phrases "
          f"(+{added} mappings). Wrote {config.DOMAIN_SYNONYMS_FILE}", flush=True)
    print("Next: recompile (compile_semantic_layer) + publish_registry + restart inference.", flush=True)


if __name__ == "__main__":
    main()
