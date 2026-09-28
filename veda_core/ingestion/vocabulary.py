"""ingestion/vocabulary.py — the per-source BUSINESS VOCABULARY (meaning-first pass, Stage 1).

Four artifacts per (tenant, source), written through `config.source_artifact_path` and
published by an L5 stage (`publish_vocabulary`), each with a TRACKED, human-editable seed
at `data/seeds/<source>/…` layered over the SLM/deterministic draft (the seed wins):

  veda_entity_cards.json     one card per table — what a row IS in business words, the
                             business date, the lifecycle column, key dimensions/measures,
                             display column, parents, importance.
  veda_value_glossary.json   {"table.col": {VALUE: [phrases]}} for lifecycle / category
                             columns ("on the market" → assets_salelisting.status=APPROVED).
                             Seeded by the existing veda_value_aliases.seed.json too.
  veda_measure_glossary.json {"table.col": {"type": MONETARY|METRIC, "phrases": [...]}}.
  veda_questions.jsonl       synthetic questions per card, each with its EXPECTED FRAME
                             (+ veda_questions.emb.npy, BGE-M3 dense vectors, row-aligned).

Every card field the SLM drafts is VALIDATED deterministically: named columns must exist;
the business date must be TEMPORAL; the lifecycle column CATEGORY; measures METRIC/MONETARY.
An invalid field falls back to the deterministic draft and is recorded in
`card["validation"]` — the SLM can make a card better, never make it lie about the schema.

The routing card (ingestion/routing_card.py) is then REBUILT FROM THE CARDS: business names,
plurals, aliases and example questions come from here, not from table prose.

This module is import-light: no Django, and the SLM / encoder / DB are imported lazily so
`--no-slm` builds work anywhere the semantic model can be loaded.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

CARDS_ARTIFACT = "veda_entity_cards.json"
VALUE_GLOSSARY_ARTIFACT = "veda_value_glossary.json"
MEASURE_GLOSSARY_ARTIFACT = "veda_measure_glossary.json"
QUESTIONS_ARTIFACT = "veda_questions.jsonl"
QUESTIONS_EMB_ARTIFACT = "veda_questions.emb.npy"

CARDS_SEED = "veda_entity_cards.seed.json"
VALUE_SEED = "veda_value_glossary.seed.json"
LEGACY_VALUE_SEED = "veda_value_aliases.seed.json"
MEASURE_SEED = "veda_measure_glossary.seed.json"

VOCAB_VERSION = 1
_MAX_DOMAIN = 40            # a CATEGORY column with more distinct values is not a lifecycle
_TEMPORAL = "TEMPORAL"
_CATEGORY = {"CATEGORY", "FLAG"}
_MEASURE = {"METRIC", "MONETARY"}
_AUDIT_DATE = re.compile(r"^(created|updated|modified|deleted|approved|last_login|last_modified)(_at|_on|_date|_time)?$")
_DISPLAY_TOK = ("name", "title", "label", "code", "number", "email", "username")


# ── helpers ──────────────────────────────────────────────────────────────────────────
def seed_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "seeds")


def _read_json(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _strip_doc(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if not str(k).startswith("_")}
    return d


def load_seed(source_id, name: str) -> Dict[str, Any]:
    return _strip_doc(_read_json(os.path.join(seed_dir(), str(source_id), name)) or {}) or {}


def _write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
    os.replace(tmp, path)
    return path


def humanize(ident: str) -> str:
    return re.sub(r"\s+", " ", str(ident or "").replace("_", " ")).strip().lower()


def table_phrase(table: str) -> str:
    """accounts_generalledger → 'generalledger'; assets_salelisting → 'salelisting';
    maintenance → 'maintenance'. The app prefix of a Django table is not business
    vocabulary. (The SLM / seed supply the real business name.)"""
    parts = [p for p in str(table).lower().split("_") if p]
    if len(parts) >= 2:
        parts = parts[1:]
    return " ".join(parts)


def pluralize(noun: str) -> str:
    n = (noun or "").strip()
    if not n:
        return n
    head, _, last = n.rpartition(" ")
    low = last.lower()
    if low.endswith(("s", "x", "z", "ch", "sh")):
        p = last + "es" if not low.endswith("s") or low.endswith("ss") else last
    elif low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        p = last[:-1] + "ies"
    else:
        p = last + "s"
    return (head + " " + p).strip()


def _col_meta(sm, table) -> Dict[str, dict]:
    cols = (sm or {}).get("columns", {}) or {}
    return {k.split(".", 1)[1]: (v or {}) for k, v in cols.items() if k.split(".", 1)[0] == table}


def _stype(meta: dict) -> str:
    return str((meta or {}).get("semantic_type") or "").upper()


def _importance(meta: dict) -> str:
    return str((meta or {}).get("importance_class") or "").upper()


# ── measure validation: an amount must be an amount ──────────────────────────────────
_ID_SHAPED = re.compile(
    r"(^|_)(id|ids|uuid|code|codes|ifsc|swift|iban|pin|pincode|zip|zipcode|postal|phone|mobile|"
    r"title|name|lat|lng|lon|latitude|longitude|year|version|seq|sequence|index|hash|token|ref)$")
_MONEYISH = re.compile(r"(amount|price|fee|cost|rent|paid|balance|value|salary|total|charge|tax|"
                       r"deposit|revenue|income|expense|budget|rate|views|count|area|size|floors|rating|score|qty|quantity)")
_MEASURE_CACHE: Dict[Tuple[str, str], bool] = {}


def measure_ok(table: str, col: str, meta: dict, live: bool = True) -> bool:
    """False for identifier-shaped columns the model typed METRIC/MONETARY — ids, codes,
    IFSC, titles, review ids, lat/long, years (the 'total ifsc code of all user bank infos'
    synthetic questions came from these). Name pattern first; then, when a source is
    reachable, the data: a non-numeric sample, or near-unique integers with no money word
    in the name, is an identifier, not an amount."""
    key = (table, col)
    if key in _MEASURE_CACHE:
        return _MEASURE_CACHE[key]
    ok = True
    if _ID_SHAPED.search(col.lower()) and not _MONEYISH.search(col.lower()):
        ok = False
    elif live:
        try:
            from veda.execution import execute_sql
            _c, rows, err = execute_sql(
                f'SELECT CAST("{col}" AS TEXT), COUNT(*) OVER (), COUNT(DISTINCT "{col}") OVER () '
                f'FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 50', None, timeout_ms=3000)
            if not err and rows:
                vals = [r[0] for r in rows]

                def _num(v):
                    try:
                        float(v)
                        return True
                    except (TypeError, ValueError):
                        return False
                if not all(_num(v) for v in vals):
                    ok = False
                else:
                    ints = all(float(v).is_integer() for v in vals)
                    total, distinct = rows[0][1] or 0, rows[0][2] or 0
                    if ints and total >= 20 and distinct / max(1, total) >= 0.95 and not _MONEYISH.search(col.lower()):
                        ok = False
        except Exception:
            pass
    _MEASURE_CACHE[key] = ok
    return ok


# ── value domains (scope-safe, offline) ──────────────────────────────────────────────
def value_domains(source_id, tenant, sm) -> Dict[str, List[str]]:
    """{"table.col": [raw values]} for low-cardinality columns, from (in order) the model's
    own sample_values, the per-source value-referents artifact, and — for tabular/datalake
    sources — a DuckDB DISTINCT sample. Never the unscoped column_values table."""
    out: Dict[str, List[str]] = {}

    def _add(key, vals):
        cur = out.setdefault(key, [])
        for v in vals:
            s = str(v) if v is not None else ""
            if s and s not in cur:
                cur.append(s)

    for k, meta in ((sm or {}).get("columns", {}) or {}).items():
        sv = (meta or {}).get("sample_values") or []
        if sv:
            _add(k, sv)
    try:
        from config import source_artifact_path
        ref = _read_json(source_artifact_path("veda_value_referents.json", source_id, tenant)) or {}
        for _vn, lst in (ref.get("referents") or {}).items():
            for r in lst or []:
                if r.get("kind") == "direct" and r.get("table") and r.get("column"):
                    _add(f"{r['table']}.{r['column']}", [r.get("value_raw")])
    except Exception:
        pass
    try:
        from query.datalake_values import _sample_source
        idx = _sample_source(str(source_id), tenant, 200) or {}
        for _tok, hits in idx.items():
            for (t, c, _st, raw) in hits:
                _add(f"{t}.{c}", [raw])
    except Exception:
        pass
    return {k: v for k, v in out.items() if 0 < len(v) <= 400}


def live_domain(table: str, column: str, limit: int = _MAX_DOMAIN + 1) -> Optional[List[str]]:
    """DISTINCT values straight from the source (read-only, bounded). Used only when the
    offline domain is missing for a CATEGORY column. Requires an ambient RequestContext."""
    try:
        from veda.execution import execute_sql
        cols, rows, err = execute_sql(
            f'SELECT DISTINCT CAST("{column}" AS TEXT) AS v FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {int(limit)}', None)
        if err or rows is None:
            return None
        return [str(r[0]) for r in rows if r and r[0] is not None]
    except Exception:
        return None


# ── the relationship graph ───────────────────────────────────────────────────────────
def _graph(source_id, tenant) -> Dict[str, Any]:
    try:
        from config import source_artifact_path
        return _read_json(source_artifact_path("veda_relationship_graph.json", source_id, tenant)) or {}
    except Exception:
        return {}


def parents_of(graph, table) -> List[Tuple[str, str]]:
    """[(fk_column, parent_table)] — declared N:1 edges out of `table`."""
    out = []
    for e in (graph or {}).get("edges", []) or []:
        if (e.get("source_table") == table and e.get("cardinality") in ("N:1", "1:1")
                and not e.get("polymorphic") and e.get("target_column") == "id"
                and e.get("discovery") in ("declared_fk", None)):
            p = (e.get("source_column"), e.get("target_table"))
            if p not in out:
                out.append(p)
    return out


def _in_degree(graph) -> Dict[str, int]:
    deg: Dict[str, int] = {}
    for e in (graph or {}).get("edges", []) or []:
        if e.get("discovery") == "declared_fk":
            deg[e.get("target_table")] = deg.get(e.get("target_table"), 0) + 1
    return deg


# ── entity cards ─────────────────────────────────────────────────────────────────────
def _pick_business_date(cols: Dict[str, dict], candidates: List[str]) -> Optional[str]:
    temporal = [c for c, m in cols.items() if _stype(m) == _TEMPORAL]
    ordered = [c for c in candidates if c in temporal] + [c for c in temporal if c not in candidates]
    domain = [c for c in ordered if not _AUDIT_DATE.match(c)]
    if domain:
        # a *_date column named for the business event beats a validity/end date
        domain.sort(key=lambda c: (("end" in c or "expiry" in c or "valid" in c), ordered.index(c)))
        return domain[0]
    for pref in ("created_at", "created_on", "created_date"):
        if pref in temporal:
            return pref
    return ordered[0] if ordered else None


def _pick_lifecycle(cols: Dict[str, dict], domains: Dict[str, List[str]], table: str) -> Optional[str]:
    cands = []
    for c, m in cols.items():
        if _stype(m) not in _CATEGORY or _stype(m) == "FLAG":
            continue
        dom = domains.get(f"{table}.{c}") or []
        score = 0
        if re.search(r"(^|_)(status|state|stage|lifecycle)$", c):
            score += 3
        if re.search(r"(^|_)(type|kind)($|_)", c):
            score += 1
        if 1 < len(dom) <= _MAX_DOMAIN:
            score += 1
        if _importance(m) == "HIGH":
            score += 1
        if score >= 3 or (score >= 2 and re.search(r"(status|state|stage|type)", c)):
            cands.append((score, c))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return cands[0][1] if cands else None


def _pick_display(cols: Dict[str, dict]) -> Optional[str]:
    best = None
    for c, m in cols.items():
        st = _stype(m)
        if st not in ("FREE_TEXT", "CATEGORY", "IDENTIFIER") or c == "id" or c.endswith("_id"):
            continue
        toks = c.split("_")
        for rank, t in enumerate(_DISPLAY_TOK):
            if t in toks:
                cand = (rank, len(c), c)
                if best is None or cand < best:
                    best = cand
                break
    return best[2] if best else None


def deterministic_card(table: str, sm, graph, domains, in_deg) -> Dict[str, Any]:
    meta = ((sm or {}).get("tables", {}) or {}).get(table) or {}
    cols = _col_meta(sm, table)
    name = table_phrase(table)
    measures = [c for c in (meta.get("candidate_measure_columns") or [])
                if c in cols and _stype(cols[c]) in _MEASURE and not c.endswith("_id") and c != "id"]
    measures += [c for c, m in cols.items()
                 if _stype(m) in _MEASURE and c not in measures and _importance(m) != "LOW"
                 and not c.endswith("_id") and c != "id"]
    measures = [c for c in measures if measure_ok(table, c, cols.get(c) or {})]
    dims = [c for c, m in cols.items()
            if _stype(m) in _CATEGORY and _importance(m) != "LOW"]
    life = _pick_lifecycle(cols, domains, table)
    if life and life in dims:
        dims.remove(life)
        dims.insert(0, life)
    parents = list(dict.fromkeys(p for _fk, p in parents_of(graph, table)))
    tt = str(meta.get("table_type") or "").upper()
    imp = in_deg.get(table, 0) + (3 if tt in ("TRANSACTION", "EVENT") else 1 if tt == "MASTER" else 0) \
        + 2 * len(measures[:3])
    if tt == "BRIDGE":
        imp = 0
    return {
        "table": table,
        "business_name": name,
        "plural": pluralize(name),
        "aliases": [],
        "primary_entity": str(meta.get("primary_entity") or ""),
        "one_row_is": str(meta.get("primary_entity") or f"one {name}"),
        "business_date_column": _pick_business_date(cols, list(meta.get("candidate_temporal_columns") or [])),
        "lifecycle_column": life,
        "key_dimensions": dims[:6],
        "key_measures": measures[:6],
        "display_column": _pick_display(cols),
        "parent_entities": parents[:8],
        "importance": int(imp),
        "table_type": tt,
        "drafted_by": "deterministic",
    }


_CARD_SYS = (
    "You write the business vocabulary for ONE database table, as JSON. You never invent "
    "columns: every column you name must be one of the listed columns. Business names are "
    "what a non-technical user of this business would call ONE row (singular noun phrase, "
    "2-4 words, lower case): e.g. 'ledger entry', 'sale listing', 'property', 'payment'.\n"
    "Fields: business_name (singular), plural, aliases (3-8 other nouns users say for these "
    "rows, singular or plural), one_row_is (one sentence), business_date_column (the date "
    "that says WHEN the business event happened — not an audit timestamp unless nothing "
    "else exists; null if none), lifecycle_column (the status/state column, or null), "
    "key_dimensions (columns users group or filter by), key_measures (numeric columns users "
    "sum/average/rank by), display_column (the column that names a row, or null)."
)


def _card_schema(cols: Dict[str, dict]) -> Dict[str, Any]:
    temporal = [c for c, m in cols.items() if _stype(m) == _TEMPORAL]
    cats = [c for c, m in cols.items() if _stype(m) in _CATEGORY]
    meas = [c for c, m in cols.items() if _stype(m) in _MEASURE]
    allc = list(cols)

    def _enum_or_null(vals):
        return {"anyOf": [{"type": "null"}, {"type": "string", "enum": vals}]} if vals else {"type": "null"}

    return {
        "type": "object",
        "properties": {
            "business_name": {"type": "string"},
            "plural": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "one_row_is": {"type": "string"},
            "business_date_column": _enum_or_null(temporal),
            "lifecycle_column": _enum_or_null(cats),
            "key_dimensions": {"type": "array", "items": ({"type": "string", "enum": cats} if cats else {"type": "string"}), "maxItems": 6},
            "key_measures": {"type": "array", "items": ({"type": "string", "enum": meas} if meas else {"type": "string"}), "maxItems": 6},
            "display_column": _enum_or_null(allc),
        },
        "required": ["business_name", "plural", "aliases", "one_row_is", "business_date_column",
                     "lifecycle_column", "key_dimensions", "key_measures", "display_column"],
    }


def _card_prompt(table, sm, graph, domains) -> str:
    meta = ((sm or {}).get("tables", {}) or {}).get(table) or {}
    cols = _col_meta(sm, table)
    lines = [f"Table: {table}",
             f"Purpose: {meta.get('business_purpose') or ''}",
             f"Each row: {meta.get('primary_entity') or ''}",
             f"Table type: {meta.get('table_type') or ''}",
             "Links to: " + (", ".join(f"{p} (via {fk})" for fk, p in parents_of(graph, table)[:8]) or "none"),
             "Columns (name: type [sample values]):"]
    for c, m in cols.items():
        if _importance(m) == "LOW" and _stype(m) not in (_TEMPORAL,) and c != "id":
            continue
        dom = domains.get(f"{table}.{c}") or []
        sv = f" [{', '.join(dom[:5])}]" if dom and len(dom) <= _MAX_DOMAIN else ""
        lines.append(f"  - {c}: {_stype(m) or '?'}{sv}")
    return "\n".join(lines)


def slm_card(table, sm, graph, domains, *, timeout: int = 60) -> Optional[Dict[str, Any]]:
    try:
        from slm import call_slm
        import config
    except Exception:
        return None
    cols = _col_meta(sm, table)
    try:
        raw = call_slm(_card_prompt(table, sm, graph, domains) + "\n\nJSON:", system=_CARD_SYS,
                       purpose="vocabulary_card", temperature=0, seed=0, num_predict=320,
                       num_ctx=getattr(config, "SLM_NUM_CTX", 4096), timeout=timeout,
                       json_schema=_card_schema(cols))
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_card(card: Dict[str, Any], draft: Dict[str, Any], sm) -> Dict[str, Any]:
    """Merge an SLM draft over the deterministic card, keeping ONLY schema-true fields."""
    table = card["table"]
    cols = _col_meta(sm, table)
    issues: List[str] = []
    out = dict(card)

    def _txt(v, maxlen=60):
        s = re.sub(r"\s+", " ", str(v or "")).strip().lower()
        return s if 0 < len(s) <= maxlen and not re.search(r"[_{}\[\]<>]", s) else None

    bn = _txt(draft.get("business_name"), 40)
    if bn:
        out["business_name"] = bn
        out["plural"] = _txt(draft.get("plural"), 44) or pluralize(bn)
    al = []
    for a in (draft.get("aliases") or [])[:8]:
        a = _txt(a, 40)
        if a and a not in al and a not in (out["business_name"], out["plural"]):
            al.append(a)
    out["aliases"] = al
    ori = _txt(draft.get("one_row_is"), 200)
    if ori:
        out["one_row_is"] = ori

    def _col(field, types, allow_none=True):
        v = draft.get(field, "__absent__")
        if v == "__absent__":
            return
        if v is None:
            if allow_none and card.get(field) is None:
                out[field] = None
            return
        if v not in cols:
            issues.append(f"{field}:{v} not a column")
            return
        if types and _stype(cols[v]) not in types:
            issues.append(f"{field}:{v} is {_stype(cols[v])}, needs {sorted(types)}")
            return
        out[field] = v

    _col("business_date_column", {_TEMPORAL})
    # a validity-window END is when something stops, not when the business event happened
    bd = out.get("business_date_column")
    if (bd and bd != card.get("business_date_column") and card.get("business_date_column")
            and re.search(r"(^|_)(end|expiry|expires|valid_to|valid_until|till|to)(_|$)", bd)):
        issues.append(f"business_date_column:{bd} is a window end — kept {card['business_date_column']}")
        out["business_date_column"] = card["business_date_column"]
    _col("lifecycle_column", {"CATEGORY"})
    _col("display_column", None)
    for field, types in (("key_dimensions", _CATEGORY | {"IDENTIFIER", "FREE_TEXT"}),
                         ("key_measures", _MEASURE)):
        vals = draft.get(field)
        if isinstance(vals, list):
            good = []
            for v in vals:
                if field == "key_measures" and v in cols and not measure_ok(table, v, cols[v]):
                    issues.append(f"key_measures:{v} is identifier-shaped, not an amount")
                    continue
                if v in cols and _stype(cols[v]) in types:
                    if v not in good:
                        good.append(v)
                else:
                    issues.append(f"{field}:{v} rejected")
            if good:
                # keep deterministic extras after the SLM's picks
                out[field] = (good + [c for c in card.get(field, []) if c not in good])[:6]
    out["drafted_by"] = "slm+validated"
    if issues:
        out["validation"] = issues
    return out


def apply_card_seed(cards: Dict[str, Dict[str, Any]], seed: Dict[str, Any], sm) -> None:
    """Seed wins per field; aliases are UNIONED (seed first). A seed entry for a table this
    source doesn't have is ignored; a seed column that doesn't exist is ignored (reported)."""
    for table, patch in (seed or {}).items():
        if table not in cards or not isinstance(patch, dict):
            continue
        cols = _col_meta(sm, table)
        c = cards[table]
        for k, v in patch.items():
            if str(k).startswith("_"):
                continue
            if k in ("business_date_column", "lifecycle_column", "display_column") and v is not None and v not in cols:
                c.setdefault("validation", []).append(f"seed {k}:{v} not a column — ignored")
                continue
            if k == "aliases":
                c["aliases"] = list(dict.fromkeys([str(a).lower() for a in (v or [])] + list(c.get("aliases") or [])))
            elif k == "distinguishes_from" and isinstance(v, dict):
                c["distinguishes_from"] = {**(c.get("distinguishes_from") or {}), **v}
            else:
                c[k] = v
        if "business_name" in patch and "plural" not in patch:
            c["plural"] = pluralize(c["business_name"])
        c["seeded"] = True


CARDS_CHECKPOINT = "veda_entity_cards.draft.json"


def build_entity_cards(source_id, tenant, sm, *, use_slm: bool = True, tables: Optional[Iterable[str]] = None,
                       domains: Optional[Dict[str, List[str]]] = None, verbose: bool = False,
                       slm_budget: Optional[int] = None, reuse_draft: bool = False) -> Dict[str, Dict[str, Any]]:
    """Cards for every table. The SLM-drafted (pre-seed) cards are checkpointed to
    veda_entity_cards.draft.json after each drafted table, so a later failure (or a
    `reuse_draft` rebuild) never repeats the slow drafting."""
    try:
        from config import source_artifact_path
        ckpt = source_artifact_path(CARDS_CHECKPOINT, source_id, tenant)
    except Exception:
        ckpt = None
    drafted = (_read_json(ckpt) or {}) if (ckpt and reuse_draft) else {}
    graph = _graph(source_id, tenant)
    domains = domains if domains is not None else value_domains(source_id, tenant, sm)
    in_deg = _in_degree(graph)
    names = list(tables) if tables else list(((sm or {}).get("tables", {}) or {}).keys())
    cards: Dict[str, Dict[str, Any]] = {}
    for t in names:
        cards[t] = deterministic_card(t, sm, graph, domains, in_deg)
    if use_slm:
        order = sorted(cards, key=lambda t: -cards[t]["importance"])
        if slm_budget is not None:
            order = order[:slm_budget]
        for i, t in enumerate(order):
            if cards[t]["table_type"] == "BRIDGE":
                continue
            d = drafted.get(t) if t in drafted else slm_card(t, sm, graph, domains)
            if d:
                drafted[t] = d
                cards[t] = validate_card(cards[t], d, sm)
                if ckpt:
                    try:
                        _write_json(ckpt, drafted)
                    except Exception:
                        pass
            if verbose:
                print(f"  [vocab] card {i + 1}/{len(order)} {t} → {cards[t]['business_name']!r}"
                      f" date={cards[t]['business_date_column']} life={cards[t]['lifecycle_column']}",
                      flush=True)
    apply_link_semantics(cards, sm, graph)
    apply_card_seed(cards, load_seed(source_id, CARDS_SEED), sm)
    return cards


def apply_link_semantics(cards: Dict[str, Dict[str, Any]], sm, graph) -> List[str]:
    """Link tables get a link card (ingestion/link_text.py): one_row_is "links a ticket to
    the user it is assigned to", aliases from the FK verbs ("assigned", "assignee",
    "ticket assignment", "subscription"…); a run-together deterministic name is split on
    the other tables' names ('ticketuser' → 'ticket user'). An SLM-drafted or seeded name
    / sentence is kept — only aliases are added to it. Returns the tables changed."""
    try:
        from ingestion.link_text import link_tables, compound_name, known_names
    except Exception:
        return []
    links = link_tables(sm, graph, cards)
    known = known_names(cards, sm)
    changed = []
    for t, c in cards.items():
        det = c.get("drafted_by") == "deterministic" and not c.get("seeded")
        lk = links.get(t)
        before = (c.get("business_name"), c.get("one_row_is"), tuple(c.get("aliases") or []))
        if lk:
            c["link"] = {"a": lk["a"], "b": lk["b"], "verbs": lk["verbs"]}
            if det:
                c["business_name"] = lk["business_name"]
                c["plural"] = pluralize(lk["business_name"])
                c["one_row_is"] = lk["one_row_is"]
            c["aliases"] = list(dict.fromkeys(list(c.get("aliases") or []) + lk["aliases"]))
        elif det:
            cn = compound_name(t, known)
            if cn and cn != c.get("business_name"):
                c["business_name"], c["plural"] = cn, pluralize(cn)
        if (c.get("business_name"), c.get("one_row_is"), tuple(c.get("aliases") or [])) != before:
            changed.append(t)
    return changed


# ── sibling disambiguation ───────────────────────────────────────────────────────────
_GENERIC = {"record", "entry", "item", "detail", "info", "data", "log", "type", "status", "user",
            "asset", "list", "the", "and", "of"}


def _name_words(card: Dict[str, Any]) -> set:
    words = set()
    for ph in [card.get("business_name"), card.get("plural"), card.get("table", "").replace("_", " ")]:
        for w in re.findall(r"[a-z]+", str(ph or "").lower()):
            if len(w) > 2:
                words.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return words - _GENERIC


def siblings(cards: Dict[str, Dict[str, Any]], t: str, k: int = 4) -> List[str]:
    """Tables a user could confuse with `t`: their business names share a content word
    ('sale listing' / 'sale transaction', 'payment transaction' / 'payment settlement')."""
    me = _name_words(cards[t])
    out = []
    for o, c in cards.items():
        if o == t or c.get("table_type") == "BRIDGE":
            continue
        shared = me & _name_words(c)
        if shared:
            out.append((-len(shared), -c.get("importance", 0), o))
    return [o for *_x, o in sorted(out)[:k]]


_DIST_SYS = (
    "For a business table and a few similarly named ones, write for EACH of the others one short "
    "phrase (4-15 words) that tells a user which one they mean — e.g. 'a sale listing is the "
    "advert; a sale transaction is the completed deal'. Plain business words, no column names. "
    "Output JSON mapping each other table's name to its phrase."
)


def slm_distinguish(t: str, card: Dict[str, Any], sibs: List[str], cards, *, timeout=60) -> Dict[str, str]:
    try:
        from slm import call_slm
        import config
    except Exception:
        return {}
    schema = {"type": "object", "properties": {o: {"type": "string"} for o in sibs}, "required": list(sibs)}
    lines = [f"THIS: {t} — {card.get('business_name')}: {card.get('one_row_is') or ''}"]
    for o in sibs:
        c = cards[o]
        lines.append(f"OTHER: {o} — {c.get('business_name')}: {c.get('one_row_is') or ''}")
    try:
        raw = call_slm("\n".join(lines) + "\n\nJSON:", system=_DIST_SYS, purpose="vocabulary_distinguish",
                       temperature=0, seed=0, num_predict=260, num_ctx=getattr(config, "SLM_NUM_CTX", 4096),
                       timeout=timeout, json_schema=schema)
        obj = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for o, ph in obj.items():
        ph = re.sub(r"\s+", " ", str(ph or "")).strip()
        if o in sibs and 3 <= len(ph.split()) <= 20 and not re.search(r"[_{}]", ph):
            out[o] = ph
    return out


def add_distinctions(cards, *, use_slm=True, budget: Optional[int] = 80, verbose=False) -> int:
    """card['distinguishes_from'] = {sibling_table: phrase}; seeded values are kept."""
    order = sorted(cards, key=lambda t: -cards[t].get("importance", 0))
    n = 0
    for t in order[:budget] if budget else order:
        c = cards[t]
        if c.get("table_type") == "BRIDGE":
            continue
        sibs = siblings(cards, t)
        if not sibs:
            continue
        have = dict(c.get("distinguishes_from") or {})
        todo = [o for o in sibs if o not in have]
        if todo and use_slm:
            have.update(slm_distinguish(t, c, todo, cards))
        if have:
            c["distinguishes_from"] = {o: ph for o, ph in have.items() if o in cards}
            n += 1
        if verbose:
            print(f"  [vocab] distinguish {t}: {list(have)}", flush=True)
    return n


# ── value glossary ───────────────────────────────────────────────────────────────────
_VALUE_SYS = (
    "For ONE status/category column of a business table you list the everyday PHRASES a "
    "user would say to mean each stored value. Phrases are short (1-5 words), lower case, "
    "and must describe the VALUE, not the table itself. Give 2-6 phrases per value; give an "
    "empty list when a value has no everyday phrase. Output JSON mapping each value to its "
    "list of phrases."
)


def slm_value_phrases(table, column, values, card, *, timeout=60) -> Dict[str, List[str]]:
    try:
        from slm import call_slm
        import config
    except Exception:
        return {}
    schema = {"type": "object",
              "properties": {v: {"type": "array", "items": {"type": "string"}, "maxItems": 6} for v in values},
              "required": list(values)}
    prompt = (f"Table: {table} — each row is {card.get('one_row_is') or card.get('business_name')}\n"
              f"Column: {column}\nStored values: {', '.join(values)}\n\nJSON:")
    try:
        raw = call_slm(prompt, system=_VALUE_SYS, purpose="vocabulary_values", temperature=0, seed=0,
                       num_predict=320, num_ctx=getattr(config, "SLM_NUM_CTX", 4096), timeout=timeout,
                       json_schema=schema)
        obj = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if k in values and isinstance(v, list)}


_GRAMMAR_WORDS = {"count", "counts", "counter", "of", "number", "numbers", "total", "totals", "sum", "average",
                  "avg", "mean", "how", "many", "much", "list", "show", "all", "the", "a", "an", "per", "each",
                  "every", "by", "top", "most", "least", "highest", "lowest", "max", "min", "maximum", "minimum",
                  "latest", "oldest", "recent", "first", "last", "and", "or", "in", "on", "for", "with", "is", "are"}


def phrase_is_grammar(ph: str) -> bool:
    """'count of', 'number of', 'total' — query grammar, never the name of a stored value
    (a spacetype value COUNTER got the phrase 'count of' and turned every count question
    into a filter)."""
    toks = re.findall(r"[a-z]+", str(ph or "").lower())
    return not toks or all(t in _GRAMMAR_WORDS for t in toks)


def clean_value_glossary(vg: Dict[str, Dict[str, List[str]]]) -> int:
    n = 0
    for key, vmap in vg.items():
        if not isinstance(vmap, dict):
            continue
        for v, ph in list(vmap.items()):
            if isinstance(ph, list):
                keep = [p for p in ph if not phrase_is_grammar(p)]
                n += len(ph) - len(keep)
                vmap[v] = keep
    return n


def build_value_glossary(source_id, tenant, sm, cards, domains, *, use_slm=True,
                         live=False, verbose=False) -> Dict[str, Dict[str, List[str]]]:
    """Per lifecycle / key CATEGORY column with a small real domain: VALUE → phrases.
    Phrases that name the table itself, collide across values, or ARE another stored value
    are dropped (the 'for sale' lesson in veda_value_aliases.seed.json)."""
    out: Dict[str, Dict[str, List[str]]] = {}
    targets: List[Tuple[str, str]] = []
    for t, c in cards.items():
        cols = [c.get("lifecycle_column")] + [d for d in c.get("key_dimensions", []) if d != c.get("lifecycle_column")][:2]
        for col in cols:
            if col:
                targets.append((t, col))
    for t, col in targets:
        key = f"{t}.{col}"
        dom = domains.get(key)
        if not dom and live:
            dom = live_domain(t, col)
            if dom:
                domains[key] = dom
        if not dom or len(dom) > _MAX_DOMAIN:
            continue
        entry: Dict[str, List[str]] = {v: [] for v in dom}
        # deterministic phrase: the value itself, humanized ("IN_PROGRESS" → "in progress")
        for v in dom:
            hv = humanize(v)
            if hv and hv != v.lower():
                entry[v].append(hv)
        if use_slm and cards[t].get("lifecycle_column") == col:
            for v, ph in slm_value_phrases(t, col, dom, cards[t]).items():
                entry[v].extend(ph)
        out[key] = entry
        if verbose:
            print(f"  [vocab] values {key}: {len(dom)} values", flush=True)
    # seeds (new name wins; the legacy value-alias seed is the same shape)
    for seed in (load_seed(source_id, LEGACY_VALUE_SEED), load_seed(source_id, VALUE_SEED)):
        for key, vmap in (seed or {}).items():
            if not isinstance(vmap, dict) or key.split(".", 1)[0] not in cards:
                continue
            cur = out.setdefault(key, {})
            drop = {str(p).lower() for p in (vmap.get("_drop") or [])}
            for v, ph in vmap.items():
                if str(v).startswith("_"):
                    continue
                cur[v] = list(dict.fromkeys([str(p).lower() for p in (ph or [])] + list(cur.get(v) or [])))
            if drop:
                # a curator's "never this phrase" beats the SLM's proposal ('listed' names the
                # sale-listing entity, not the APPROVED value — see the seed's _pruned note)
                for v in list(cur):
                    if isinstance(cur[v], list):
                        cur[v] = [p for p in cur[v] if p not in drop]
            out[key]["_seeded"] = True        # type: ignore[assignment]
    clean_value_glossary(out)
    # clean: lower, dedupe, drop table names / cross-value collisions / other values
    for key, vmap in out.items():
        t = key.split(".", 1)[0]
        card = cards.get(t) or {}
        entity_words = {card.get("business_name"), card.get("plural"), *(card.get("aliases") or [])}
        seeded = bool(vmap.pop("_seeded", False)) if isinstance(vmap, dict) else False
        values_low = {str(v).lower() for v in vmap}
        seen: Dict[str, str] = {}
        clash = set()
        for v, ph in vmap.items():
            clean = []
            for p in ph or []:
                p = re.sub(r"\s+", " ", str(p).lower()).strip()
                if not p or len(p) > 40 or p in entity_words:
                    continue
                if p in values_low and p != str(v).lower():
                    continue
                if p in seen and seen[p] != v:
                    clash.add(p)
                seen.setdefault(p, v)
                if p not in clean:
                    clean.append(p)
            vmap[v] = clean
        for v in vmap:
            vmap[v] = [p for p in vmap[v] if p not in clash]
        if seeded:
            vmap["_seeded"] = True            # type: ignore[assignment]
    return out


# ── measure glossary ─────────────────────────────────────────────────────────────────
_MONEY_WORDS = ("price", "amount", "cost", "fee", "rent", "value", "paid", "total")


def build_measure_glossary(source_id, sm, cards) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t, card in cards.items():
        cols = _col_meta(sm, t)
        for col in card.get("key_measures") or []:
            m = cols.get(col) or {}
            st = _stype(m)
            if st not in _MEASURE or not measure_ok(t, col, m):
                continue
            phrases = [humanize(col)]
            toks = [p for p in col.split("_") if len(p) > 2]
            if len(toks) > 1:
                phrases.append(toks[-1])          # "expected_price" → "price"
            for a in (m.get("aliases") or [])[:8]:
                a = re.sub(r"\s+", " ", str(a).lower()).strip()
                if a and len(a) <= 30 and not a.endswith(" id") and a not in phrases:
                    phrases.append(a)
            if st == "MONETARY":
                for w in ("price", "amount", "value", "cost"):
                    if w in col and w not in phrases:
                        phrases.append(w)
            out[f"{t}.{col}"] = {"type": st, "phrases": phrases[:10]}
    seed = load_seed(source_id, MEASURE_SEED)
    for key, ent in (seed or {}).items():
        if key.split(".", 1)[0] not in cards or not isinstance(ent, dict):
            continue
        cur = out.setdefault(key, {"type": ent.get("type") or "METRIC", "phrases": []})
        cur["phrases"] = list(dict.fromkeys([str(p).lower() for p in ent.get("phrases") or []] + cur["phrases"]))
        if ent.get("type"):
            cur["type"] = ent["type"]
        cur["seeded"] = True
    return out


# ── synthetic questions (each with its EXPECTED FRAME) ───────────────────────────────
def _measure_phrase(mg, table, col) -> str:
    ph = (mg.get(f"{table}.{col}") or {}).get("phrases") or [humanize(col)]
    return ph[0]


def template_questions(cards, vg, mg, *, max_cards: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    order = sorted(cards, key=lambda t: -cards[t].get("importance", 0))
    if max_cards:
        order = order[:max_cards]
    for t in order:
        c = cards[t]
        if c.get("table_type") == "BRIDGE":
            continue
        ent, pl = c["business_name"], c["plural"]

        def q(text, frame, kind):
            fr = {"entity": ent, "secondaries": [], "measure": None, "aggregation": "none",
                  "filters": [], "group_by": [], "order": None, "limit": None, "time": None,
                  "distinct": False, "confidence": 1.0}
            fr.update(frame)
            out.append({"table": t, "question": text, "frame": fr, "kind": kind, "origin": "template"})

        q(f"how many {pl} are there", {"aggregation": "count"}, "count")
        q(f"list all {pl}", {}, "list")
        bd = c.get("business_date_column")
        if bd:
            q(f"show the most recent {pl}", {"order": {"concept": humanize(bd), "dir": "desc"}}, "recent")
            q(f"what are the oldest {pl}", {"order": {"concept": humanize(bd), "dir": "asc"}}, "oldest")
        for col in (c.get("key_measures") or [])[:2]:
            mp = _measure_phrase(mg, t, col)
            q(f"top 5 {pl} by {mp}", {"order": {"concept": mp, "dir": "desc"}, "limit": 5}, "top_n")
            q(f"total {mp} of all {pl}", {"measure": mp, "aggregation": "sum"}, "sum")
            q(f"average {mp} of {pl}", {"measure": mp, "aggregation": "avg"}, "avg")
            q(f"{pl} with {mp} above 1000",
              {"filters": [{"concept": mp, "op": ">", "value": 1000}]}, "numeric_filter")
        life = c.get("lifecycle_column")
        vmap = vg.get(f"{t}.{life}") if life else None
        if vmap:
            for v, ph in list(vmap.items())[:3]:
                if str(v).startswith("_"):
                    continue
                phrase = (ph or [humanize(v)])[0]
                q(f"{pl} that are {phrase}",
                  {"filters": [{"concept": humanize(life), "op": "=", "value": v}]}, "value_filter")
        for dcol in [d for d in (c.get("key_dimensions") or []) if d != life][:1] + ([life] if life else []):
            q(f"how many {pl} per {humanize(dcol)}",
              {"aggregation": "count", "group_by": [humanize(dcol)]}, "grouped")
            break
    return out


def compound_examples(cards: Dict[str, Dict[str, Any]], doc_cards: Optional[List[Dict[str, Any]]] = None,
                      source_of: Optional[Dict[str, str]] = None,
                      vg: Optional[Dict[str, Dict[str, List[str]]]] = None,
                      mg: Optional[Dict[str, Any]] = None, k: int = 8) -> List[Dict[str, Any]]:
    """COMPOUND synthetic examples — one message, several independent (or one dependent)
    questions, across sources — each with its EXPECTED INTENT LIST. Built from the scope's
    own cards and document sections (never from an eval set), so the intent extractor has
    seen the shape {intents: [...], relation} on this deployment's vocabulary.

    Deterministic: the highest-importance card of each source, the first sections of the
    first documents. 2- and 3-part, mixed sources, one dependent example.

    `vg`/`mg` (value / measure glossary, same shape `template_questions` uses) are
    optional: when given, the FIRST two examples demonstrate the `filters` slot (a value
    filter, then a numeric one) — the single-frame extractor already sees these via
    `template_questions`' `value_filter`/`numeric_filter` examples, but nothing taught
    the compound path the same shape, which is why it dropped filters (§10.3)."""
    source_of = source_of or {}
    by_src: Dict[str, List[str]] = {}
    for t in sorted(cards, key=lambda t: (-cards[t].get("importance", 0), t)):
        c = cards[t]
        if c.get("table_type") == "BRIDGE" or not c.get("business_name"):
            continue
        by_src.setdefault(str(source_of.get(t) or c.get("_source_id") or "?"), []).append(t)
    picks: List[str] = []
    for i in range(3):                           # round-robin across sources
        for s in sorted(by_src):
            if i < len(by_src[s]) and by_src[s][i] not in picks:
                picks.append(by_src[s][i])
    if not picks:
        return []
    secs: List[Tuple[str, str]] = []            # (document title, section phrase)
    for d in (doc_cards or []):
        for sname in (d.get("sections") or [])[1:]:
            ph = re.sub(r"\s*\([^)]*\)", "", sname).strip().lower()
            if 3 <= len(ph) <= 40:
                secs.append((d.get("title") or d.get("doc_name") or "document", ph))
        if len(secs) >= 6:
            break

    def card(i):
        return cards[picks[i % len(picks)]]

    def measure(c):
        km = [m for m in (c.get("key_measures") or [])]
        return humanize(km[0]) if km else None

    def fr(part, c=None, kind="sql", **slots):
        f = {"part": part, "kind": kind, "entity": (c or {}).get("business_name") if c else None,
             "aggregation": "none"}
        f.update(slots)
        return f

    ex: List[Dict[str, Any]] = []
    a, b, c3 = card(0), card(1), card(2)
    ma, mb = measure(a), measure(b)
    sec = secs[0] if secs else None
    sec2 = secs[1] if len(secs) > 1 else sec

    # 0a. 2-part: VALUE FILTER (op '=') + count, two sources — the shape most often
    #     dropped: a lifecycle-column phrase ("in Kochi", "that are gated") turned into
    #     a bare group/count instead of a WHERE.
    vg = vg or {}
    life = a.get("lifecycle_column")
    vmap = vg.get(f"{picks[0]}.{life}") if life else None
    fval, fphrase = None, None
    for v, ph in (vmap or {}).items():
        if str(v).startswith("_"):
            continue
        fval, fphrase = v, (ph or [humanize(v)])[0]
        break
    if fval is not None:
        p1 = f"how many {a['plural']} are {fphrase}"
        p2 = f"list all {b['plural']}"
        ex.append({"question": f"{p1.capitalize()}, and {p2}?", "relation": "independent",
                   "intents": [fr(p1, a, aggregation="count",
                                 filters=[{"concept": humanize(life), "op": "=", "value": fval}]),
                               fr(p2, b)]})
    # 0b. 2-part: NUMERIC FILTER (op '>') + document topic — the other filter shape.
    mg = mg or {}
    if ma and sec:
        p1 = f"{a['plural']} with {ma} above 1000"
        p2 = f"what does the {sec[0]} say about {sec[1]}"
        ex.append({"question": f"Show {p1}, and {p2}?", "relation": "independent",
                   "intents": [fr(p1, a, filters=[{"concept": ma, "op": ">", "value": 1000}]),
                               fr(p2, None, "rag", entity=sec[0], topics=[sec[1]])]})
    # 1. 2-part: count + document topic
    if sec:
        p1, p2 = f"how many {a['plural']} are there", f"what does the {sec[0]} say about {sec[1]}"
        ex.append({"question": f"{p1.capitalize()}, and {p2}?", "relation": "independent",
                   "intents": [fr(p1, a, aggregation="count"),
                               fr(p2, None, "rag", entity=sec[0], topics=[sec[1]])]})
    # 2. 2-part: ranking + average, two sources
    if mb:
        p1 = f"show the most recent {a['plural']}"
        p2 = f"what is the average {mb} of {b['plural']}"
        ex.append({"question": f"{p1.capitalize()} and {p2}?", "relation": "independent",
                   "intents": [fr(p1, a, order={"concept": "date", "dir": "desc"}),
                               fr(p2, b, measure=mb, aggregation="avg")]})
    # 3. 3-part: list + document + superlative
    if sec2 and mb:
        p1, p2 = f"list the {a['plural']}", f"how does the {sec2[1]} policy work"
        p3 = f"which {b['business_name']} has the highest {mb}"
        ex.append({"question": f"Can you {p1}, {p2}, and {p3}?", "relation": "independent",
                   "intents": [fr(p1, a), fr(p2, None, "rag", entity=sec2[0], topics=[sec2[1]]),
                               fr(p3, b, order={"concept": mb, "dir": "desc"}, limit=1)]})
    # 4. 3-part: three data questions, three entities
    if ma:
        dims = [d for d in (b.get("key_dimensions") or [])]
        p1 = f"what is the total {ma} of all {a['plural']}"
        p2 = (f"how many {b['plural']} are there per {humanize(dims[0])}" if dims
              else f"how many {b['plural']} are there")
        p3 = f"how many {c3['plural']} are there"
        ex.append({"question": f"{p1.capitalize()}; {p2}; and {p3}?", "relation": "independent",
                   "intents": [fr(p1, a, measure=ma, aggregation="sum"),
                               fr(p2, b, aggregation="count", group_by=[humanize(dims[0])] if dims else []),
                               fr(p3, c3, aggregation="count")]})
    # 5. 2-part DEPENDENT: the second part uses the first part's rows
    if ma:
        p1 = f"show the top 5 {a['plural']} by {ma}"
        p2 = f"for those, what is the total {ma}"
        ex.append({"question": f"{p1.capitalize()}, and {p2}?", "relation": "dependent",
                   "intents": [fr(p1, a, order={"concept": ma, "dir": "desc"}, limit=5),
                               fr(p2, a, measure=ma, aggregation="sum", depends_on=0)]})
    # 6. 2-part: document first, data second
    if sec:
        p1, p2 = f"what is the rule for {sec[1]}", f"list all {c3['plural']}"
        ex.append({"question": f"{p1.capitalize()}, and also {p2}.", "relation": "independent",
                   "intents": [fr(p1, None, "rag", entity=sec[0], topics=[sec[1]]), fr(p2, c3)]})
    return ex[:k]


_PARA_SYS = (
    "Rewrite each numbered question the way a busy business user would actually ask it — "
    "3 different paraphrases each, keeping EXACTLY the same meaning (same entity, same "
    "numbers, same ordering direction, same filter). Output JSON: {\"1\": [..3..], \"2\": [...]}."
)


def slm_paraphrases(questions: List[Dict[str, Any]], *, timeout=90) -> Dict[int, List[str]]:
    try:
        from slm import call_slm
        import config
    except Exception:
        return {}
    keys = [str(i + 1) for i in range(len(questions))]
    schema = {"type": "object",
              "properties": {k: {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}
                             for k in keys},
              "required": keys}
    prompt = "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(questions)) + "\n\nJSON:"
    try:
        raw = call_slm(prompt, system=_PARA_SYS, purpose="vocabulary_paraphrase", temperature=0,
                       seed=0, num_predict=900, num_ctx=getattr(config, "SLM_NUM_CTX", 4096),
                       timeout=timeout, json_schema=schema)
        obj = json.loads(raw)
    except Exception:
        return {}
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        try:
            i = int(k) - 1
        except ValueError:
            continue
        if 0 <= i < len(questions) and isinstance(v, list):
            out[i] = [str(x).strip() for x in v if str(x).strip()][:3]
    return out


def build_questions(cards, vg, mg, *, use_slm=True, paraphrase_cards: int = 25,
                    verbose=False) -> List[Dict[str, Any]]:
    qs = template_questions(cards, vg, mg)
    if use_slm and paraphrase_cards:
        top = sorted(cards, key=lambda t: -cards[t].get("importance", 0))[:paraphrase_cards]
        extra: List[Dict[str, Any]] = []
        for i, t in enumerate(top):
            mine = [q for q in qs if q["table"] == t]
            for chunk in (mine[:7], mine[7:14]):
                if not chunk:
                    continue
                try:
                    got = slm_paraphrases(chunk)
                except Exception as e:           # one bad reply must not lose the source
                    print(f"  [vocab] paraphrases skipped for {t}: {type(e).__name__}: {e}", flush=True)
                    continue
                for j, paras in got.items():
                    for p in paras:
                        extra.append({**chunk[j], "question": p, "origin": "paraphrase",
                                      "paraphrase_of": chunk[j]["question"]})
            if verbose:
                print(f"  [vocab] paraphrases {i + 1}/{len(top)} {t}", flush=True)
        qs += extra
    return qs


def embed_questions(qs: List[Dict[str, Any]]):
    try:
        from ingestion.m3_encoder import encode_dense
        import numpy as np
        if not qs:
            return None
        return np.asarray(encode_dense([q["question"] for q in qs]), dtype="float32")
    except Exception as e:
        print(f"  [vocab] question embedding skipped: {type(e).__name__}: {e}")
        return None


# ── routing card rebuilt from the cards ──────────────────────────────────────────────
def routing_card_from_cards(source_id, tenant, cards, qs, *, max_entities: int = 20) -> Optional[str]:
    """Rewrite the source's routing card ENTITIES + EXAMPLE QUESTIONS from the entity
    cards (business names, plurals, aliases, business-named dims/measures, parents). The
    card's joins_to (cross-source edges) and header are kept as the L5 stage wrote them."""
    try:
        from ingestion.routing_card import load_routing_card, CARD_ARTIFACT
        from config import source_artifact_path
    except Exception:
        return None
    card = load_routing_card(source_id, tenant) or {"version": 1, "source_id": str(source_id),
                                                    "tenant": tenant, "joins_to": []}
    old = {e.get("table"): e for e in (card.get("entities") or [])}
    ents = []
    for t in sorted(cards, key=lambda t: -cards[t].get("importance", 0)):
        c = cards[t]
        if c.get("table_type") == "BRIDGE":
            continue
        prev = old.get(t) or {}
        ents.append({
            "table": t,
            "business_name": c["business_name"],
            "plural": c["plural"],
            "aliases": c.get("aliases") or [],
            "purpose": c.get("one_row_is") or prev.get("purpose"),
            "row_count": prev.get("row_count"),
            "key_dims": [humanize(d) for d in c.get("key_dimensions") or []],
            "key_measures": [humanize(m) for m in c.get("key_measures") or []],
            "sample_values": prev.get("sample_values") or {},
            "links_to": [cards[p]["business_name"] for p in c.get("parent_entities") or [] if p in cards][:5],
        })
        if len(ents) >= max_entities:
            break
    card["entities"] = ents
    card["example_questions"] = [q["question"] for q in qs if q.get("origin") == "template"
                                 and q["table"] in {e["table"] for e in ents[:6]}][:8]
    card["built_from"] = "entity_cards"
    path = source_artifact_path(CARD_ARTIFACT, source_id, tenant)
    # keep the prose-built card the first time it is replaced, so the switch is reversible
    _bak = path + ".pre_vocab"
    if os.path.exists(path) and not os.path.exists(_bak):
        prev = _read_json(path) or {}
        if prev.get("built_from") != "entity_cards":
            import shutil
            shutil.copy2(path, _bak)
    return _write_json(path, card)


# ── publish (L5 stage + CLI) ─────────────────────────────────────────────────────────
def publish_vocabulary(source_id, tenant, sm, *, use_slm: bool = True, live_values: bool = False,
                       embed: bool = True, paraphrase_cards: int = 25, slm_budget: Optional[int] = None,
                       verbose: bool = False, reuse_draft: bool = False) -> Dict[str, Any]:
    from config import source_artifact_path
    if not (sm or {}).get("tables"):
        return {"skipped": "no tables in the semantic model (document source?)"}
    domains = value_domains(source_id, tenant, sm)
    cards = build_entity_cards(source_id, tenant, sm, use_slm=use_slm, domains=domains,
                               verbose=verbose, slm_budget=slm_budget, reuse_draft=reuse_draft)
    add_distinctions(cards, use_slm=use_slm, budget=slm_budget or 80, verbose=verbose)
    apply_card_seed(cards, {t: {"distinguishes_from": v["distinguishes_from"]} for t, v in
                            load_seed(source_id, CARDS_SEED).items()
                            if isinstance(v, dict) and v.get("distinguishes_from")}, sm)
    vg = build_value_glossary(source_id, tenant, sm, cards, domains, use_slm=use_slm,
                              live=live_values, verbose=verbose)
    mg = build_measure_glossary(source_id, sm, cards)
    qs = build_questions(cards, vg, mg, use_slm=use_slm, paraphrase_cards=paraphrase_cards,
                         verbose=verbose)
    about = {"_about": f"Business vocabulary for source {source_id} (meaning-first pass, "
                       f"ingestion/vocabulary.py). Derived — edit the seeds under "
                       f"data/seeds/{source_id}/, not this file.",
             "_version": VOCAB_VERSION}
    paths = {
        "cards": _write_json(source_artifact_path(CARDS_ARTIFACT, source_id, tenant), {**about, **cards}),
        "value_glossary": _write_json(source_artifact_path(VALUE_GLOSSARY_ARTIFACT, source_id, tenant), {**about, **vg}),
        "measure_glossary": _write_json(source_artifact_path(MEASURE_GLOSSARY_ARTIFACT, source_id, tenant), {**about, **mg}),
    }
    qpath = source_artifact_path(QUESTIONS_ARTIFACT, source_id, tenant)
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    with open(qpath + ".tmp", "w") as f:
        for q in qs:
            f.write(json.dumps(q, default=str) + "\n")
    os.replace(qpath + ".tmp", qpath)
    paths["questions"] = qpath
    if embed:
        emb = embed_questions(qs)
        if emb is not None:
            import numpy as np
            epath = source_artifact_path(QUESTIONS_EMB_ARTIFACT, source_id, tenant)
            with open(epath + ".tmp", "wb") as f:
                np.save(f, emb)
            os.replace(epath + ".tmp", epath)
            paths["questions_emb"] = epath
    rc = routing_card_from_cards(source_id, tenant, cards, qs)
    if rc:
        paths["routing_card"] = rc
    return {"paths": paths, "cards": len(cards), "value_columns": len(vg),
            "measures": len(mg), "questions": len(qs),
            "card_issues": sum(len(c.get("validation") or []) for c in cards.values())}


def publish_from_state(ctx, state, verbose: bool = False) -> Dict[str, Any]:
    sm = (state or {}).get("semantic_model") or {}
    return publish_vocabulary(ctx.source_id, getattr(ctx, "tenant", "default"), sm,
                              use_slm=not getattr(ctx, "skip_llm", False), verbose=verbose)
