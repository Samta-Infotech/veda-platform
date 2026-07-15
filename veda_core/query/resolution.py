# =============================================================================
# query/resolution.py
# VEDA — QSR (Query Semantic Resolution): ONE place where query tokens resolve
# to typed schema referents. Phase A of ARCHITECTURE_ROOT_CAUSE_PLAN.md.
#
# Before this module, "what does this query word refer to?" was answered by six
# duplicate singularizers, 38 ad-hoc tokenization sites, and 19 files privately
# consulting the value store — several through a connection pointed at the WRONG
# database (the source DB has no `column_values`; the arbiter's typed lookup was
# returning [] in production behind its fail-open try/except). Every resolution
# bug of the anchor/routing stream is one symptom of that fragmentation.
#
# Contract:
#   resolve(query, sm=None)  -> [TokenResolution]  (unigram + bigram value spans)
#   typed_value_lookup()     -> drop-in for value_arbiter.column_values_typed_lookup
#                               (artifact-first, INTERNAL-store fallback — the fix
#                               for the dead channel). DIRECT referents only.
#   closed_value_tables(tok) -> tables credited via FK closure — ANCHOR/ROUTING
#                               evidence ONLY, never SQL predicate material (the
#                               label lives one join away).
#   domain_for(table, col)   -> the column's actual value domain (grounded
#                               clarifies: "statuses here are captured/…").
#
# Genericity contract: no table/column/value names; typing comes from the
# semantic model's own metadata (analytics_role, semantic_type,
# candidate_measure_columns, aliases, domain_synonyms), the derived
# value-referent artifact, and the language layer (config.QUERY_GRAMMAR /
# QUERY_LANGUAGE via validation._gate_strip). Scope-cached like name_tokens.
# =============================================================================
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from retrieval.query_enrichment import _singularize
except Exception:  # pragma: no cover
    def _singularize(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


# ---------------------------------------------------------------------------
# result shape
# ---------------------------------------------------------------------------
@dataclass
class TokenResolution:
    span: str                                   # query word or bigram
    grammar: bool = False                       # language layer (operators/function words)
    values: Dict[str, list] = field(default_factory=lambda: {"direct": [], "closed": []})
    entities: List[tuple] = field(default_factory=list)    # [(table, idf_weight)]
    measures: List[str] = field(default_factory=list)      # [col_id]
    dimensions: List[str] = field(default_factory=list)    # [col_id]

    @property
    def unresolved(self) -> bool:
        return not (self.grammar or self.values["direct"] or self.values["closed"]
                    or self.entities or self.measures or self.dimensions)

    @property
    def content(self) -> bool:
        """A non-grammar span the user is asking ABOUT (must be accounted for)."""
        return not self.grammar


# ---------------------------------------------------------------------------
# scope-cached indexes (same scoping discipline as semantic.name_tokens)
# ---------------------------------------------------------------------------
_SCOPES: dict = {}
_SCOPES_MAX = 8


def _scope_key():
    from semantic.name_tokens import _scope_key as k
    return k()


def _load_sm():
    from veda.runtime import _load_scoped_sm
    return _load_scoped_sm()


def _split_col_id(cid: str):
    t, _, c = cid.rpartition(".")
    return (t, c) if t else (cid, "")


def _words(text: str):
    return [_singularize(w) for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 2]


def _build_col_index(sm: dict):
    """word → measure col_ids / dimension col_ids, from the semantic model's OWN
    typing: analytics_role, semantic_type, candidate_measure_columns, aliases,
    base_aliases, domain_synonyms. No name heuristics beyond tokenization."""
    measures: Dict[str, set] = defaultdict(set)
    dimensions: Dict[str, set] = defaultdict(set)

    cand_measures = set()
    for t, tm in (sm.get("tables") or {}).items():
        for c in (tm or {}).get("candidate_measure_columns") or []:
            cand_measures.add(f"{t}.{c}")

    for cid, cm in (sm.get("columns") or {}).items():
        cm = cm or {}
        _, col = _split_col_id(cid)
        words = set(_words(col))
        for al in (cm.get("aliases") or []) + (cm.get("base_aliases") or []):
            words.update(_words(al))
        role = (cm.get("analytics_role") or "").upper()
        st = (cm.get("semantic_type") or "").upper()
        is_measure = role == "MEASURE" or cid in cand_measures
        is_dim = st in ("CATEGORY", "CATEGORICAL", "FLAG") or role == "DIMENSION"
        for w in words:
            if is_measure:
                measures[w].add(cid)
            elif is_dim:
                dimensions[w].add(cid)

    for phrase, cids in (sm.get("domain_synonyms") or {}).items():
        cl = cids if isinstance(cids, list) else [cids]
        for w in _words(phrase):
            for cid in cl:
                cid = str(cid)
                if cid in cand_measures:
                    measures[w].add(cid)
                else:
                    dimensions[w].add(cid)

    return {w: sorted(s) for w, s in measures.items()}, \
           {w: sorted(s) for w, s in dimensions.items()}


def _scope(sm=None):
    key = _scope_key()
    ent = _SCOPES.get(key)
    ncols = len((sm or {}).get("columns", {}) or {}) if sm is not None else None
    if ent is not None and (ncols is None or ent["ncols"] == ncols):
        return ent
    if sm is None:
        sm = _load_sm()
    measures, dimensions = _build_col_index(sm)
    from semantic.name_tokens import table_tokens, token_table_idf
    # column-NAME token vocabulary (no aliases): engineer-written words only —
    # the high-precision referent set the strict qualifier gate keys on.
    # Owners/counts alongside: referent_tables needs WHICH tables own a column
    # word and HOW MANY columns carry it (generic-word guard) — same loop.
    colname_owners: Dict[str, set] = defaultdict(set)
    colname_counts: Dict[str, int] = defaultdict(int)
    for cid in (sm.get("columns") or {}):
        t, col = _split_col_id(cid)
        for w in set(_words(col)):
            colname_owners[w].add(t)
            colname_counts[w] += 1
    ent = {"ncols": len(sm.get("columns", {}) or {}),
           "sm": sm, "measures": measures, "dimensions": dimensions,
           "entity_index": _build_entity_index(sm, table_tokens),
           "colname_tokens": frozenset(colname_owners),
           "colname_owners": {w: sorted(s) for w, s in colname_owners.items()},
           "colname_counts": dict(colname_counts),
           "idf": token_table_idf(sm)}
    _SCOPES[key] = ent
    while len(_SCOPES) > _SCOPES_MAX:
        _SCOPES.pop(next(iter(_SCOPES)))
    return ent


def _build_entity_index(sm: dict, table_tokens) -> Dict[str, list]:
    idx: Dict[str, set] = defaultdict(set)
    for t in (sm.get("tables") or {}):
        for w in table_tokens(t, sm):
            idx[w].add(t)
    return {w: sorted(s) for w, s in idx.items()}


# ---------------------------------------------------------------------------
# value referents: artifact-first, INTERNAL-store fallback
# ---------------------------------------------------------------------------
def _norm(v) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(v or "").lower()))


def _internal_conn():
    import psycopg2
    from veda.runtime import _internal_db_config
    db = _internal_db_config()
    return psycopg2.connect(host=db["host"], port=db["port"], dbname=db["database"],
                            user=db["user"], password=db["password"])


def value_referents(token: str) -> Dict[str, list]:
    """{'direct': [...], 'closed': [...]} for a normalized token/bigram. Artifact
    first (precise FK closure, sub-ms); live internal store as fallback so a stale
    or missing artifact degrades to direct-only, never to a dead channel."""
    token = _norm(token)
    if len(token) < 3:
        return {"direct": [], "closed": []}
    try:
        from ingestion.value_referents import load_value_referents
        refs = load_value_referents().get("referents", {}).get(token)
        if refs is not None:
            return {"direct": [r for r in refs if r.get("kind") == "direct"],
                    "closed": [r for r in refs if r.get("kind") == "closed"]}
    except Exception:
        pass
    # fallback: live internal store, direct only
    out = {"direct": [], "closed": []}
    try:
        from config import COLUMN_VALUES_TABLE_NAME as TBL
        conn = _internal_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT table_name, col_name, semantic_type, value_raw "
                        f"FROM {TBL} WHERE value_norm = %s LIMIT 16", (token,))
            out["direct"] = [{"kind": "direct", "table": r[0], "column": r[1],
                              "type": r[2], "value_raw": r[3]} for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        pass
    return out


def typed_value_lookup():
    """Drop-in replacement for value_arbiter.column_values_typed_lookup(conn_fn):
    token -> [(table, col, semantic_type, value_raw)]. DIRECT referents only —
    exactly the tuples the arbiter/filter layer may turn into SQL predicates.
    Fixes the dead production channel (the old callers passed the SOURCE-DB
    connection, where `column_values` does not exist)."""
    def _lookup(token: str):
        return [(r["table"], r["column"], r.get("type"), r.get("value_raw"))
                for r in value_referents(token)["direct"]]
    return _lookup


def closed_value_tables(token: str) -> Dict[str, list]:
    """{table: [closed referent dicts]} — FK-closure evidence that `token`'s value
    lives one join away. ROUTING/ANCHOR evidence only; never predicate material."""
    out: Dict[str, list] = defaultdict(list)
    for r in value_referents(token)["closed"]:
        out[r["table"]].append(r)
    return dict(out)


def typed_anchor_evidence(query: str, sm=None, exclude_spans=()):
    """Typed anchor evidence per table (the role-discipline scorer shared by the
    superlative builder and vet_primary's rerank). Rules, all schema-generic:
      · per (span, table) the MAX weight counts ONCE (a span matching three columns
        of one table is one piece of evidence — else shared lookups outvote
        everything through their own rows)
      · a span whose value FK-closes elsewhere marks its direct table as a LABEL
        STORE (demoted to 0.3): the value's home serves other tables
      · grammar-flagged spans contribute NOTHING — not an entity vote ('value'
        must not anchor the valuebundle family) and not a value vote (a stopword
        coinciding with sampled data — "are" = country code 'ARE' — is
        coincidence, not evidence; it manufactured anchor ties for tables the
        user never referenced); excluded spans (e.g. a group-by dimension word)
        likewise contribute nothing
    Returns (evidence: {table: score}, why: {table: [reasons]})."""
    from collections import defaultdict
    DIRECT_W, CLOSED_W, LABEL_W = 1.0, 0.6, 0.3
    evidence: Dict[str, float] = defaultdict(float)
    why: Dict[str, list] = defaultdict(list)
    for tr in resolve(query, sm):
        if tr.span in exclude_spans or tr.grammar:
            continue
        per_table: Dict[str, float] = {}
        closed_vias = {r["via_table"] for r in tr.values["closed"]}
        for r in tr.values["direct"]:
            w = LABEL_W if r["table"] in closed_vias else DIRECT_W
            if w > per_table.get(r["table"], 0.0):
                per_table[r["table"]] = w
                why[r["table"]].append(f"value '{tr.span}'={r['column']}")
        for r in tr.values["closed"]:
            if CLOSED_W > per_table.get(r["table"], 0.0):
                per_table[r["table"]] = CLOSED_W
                why[r["table"]].append(f"value '{tr.span}' via {r['column']}")
        for t, w in tr.entities:
            if w > per_table.get(t, 0.0):
                per_table[t] = w
                why[t].append(f"entity '{tr.span}' (idf {w})")
        for t, w in per_table.items():
            evidence[t] += w
    return dict(evidence), dict(why)


def has_schema_referent(token: str, sm=None) -> bool:
    """True when `token` STRONGLY refers to something in this scope's schema — a
    sampled value, a distinctive entity table, or a specific column. The qualifier
    gate's wrong-table blind spot: an unaccounted token was treated as filler
    unless it named a column of the QUERIED table — so the more wrong the table,
    the more everything looked ignorable ('financial records' vs SELECT * FROM
    assets_asset). A token with strong referents anywhere is evidence of dropped
    semantics, not filler.

    Strength rules (schema-derived, no word lists): sampled VALUES are always
    strong; ENTITY references need idf ≥ QSR_REFERENT_MIN_IDF (an app prefix like
    'asset' is not a subject claim); COLUMN references count only when the token
    appears in a column's OWN NAME tokens ('financial' ∈ financial_year_id) —
    engineer-written vocabulary. Alias word-soup is deliberately excluded here:
    measured on the golden baseline, prepositions ('across', 'toward', 'along')
    match stray alias words and would refuse nearly every valid answer. Aliases
    still power RESOLUTION (routing evidence); they just don't gate refusals."""
    token = (token or "").lower()
    if len(token) < 3:
        return False
    try:
        from config import QSR_REFERENT_MIN_IDF
    except Exception:
        QSR_REFERENT_MIN_IDF = 0.35
    try:
        ent = _scope(sm)
        sw = _singularize(token)
        vr = value_referents(token)
        if vr["direct"] or vr["closed"]:
            return True
        idf = ent["idf"]
        if ent["entity_index"].get(sw) and idf.get(sw, idf.get(token, 0.0)) >= QSR_REFERENT_MIN_IDF:
            return True
        return sw in ent["colname_tokens"] or token in ent["colname_tokens"]
    except Exception:
        return False


def referent_tables(token: str, sm=None) -> List[dict]:
    """Ranked tables `token` refers to in this scope — the qualifier-salvage
    resolver. When the qualifier gate is about to refuse a dropped token, the
    honest next question is "what IS this token here?": a sampled value (its
    owner/FK-referring tables), an entity word, or a word engineers put in a
    column name. All referents come from the scope's own artifacts — no word
    lists, no schema names in code — so the behavior is identical on any source.

    Ranking mirrors typed_anchor_evidence's role discipline: per table the MAX
    class weight counts (direct value 1.0, label store 0.3, FK-closed 0.6,
    entity 0.4+0.4·idf gated by QSR_REFERENT_MIN_IDF, column-name word 0.5
    guarded by QSR_REFERENT_MAX_COLS so request verbs matching every boolean
    column stay out). Deterministically sorted. Never raises; [] on any failure.

    Returns [{"table": str, "score": float, "why": [str]}], best first."""
    token = (token or "").lower().strip()
    if len(token) < 3:
        return []
    try:
        from config import QSR_REFERENT_MIN_IDF, QSR_REFERENT_MAX_COLS
    except Exception:
        QSR_REFERENT_MIN_IDF, QSR_REFERENT_MAX_COLS = 0.35, 20
    try:
        ent = _scope(sm)
    except Exception:
        return []
    sw = _singularize(token)
    scores: Dict[str, float] = {}
    why: Dict[str, list] = defaultdict(list)

    def _credit(t, w, reason):
        if w > scores.get(t, 0.0):
            scores[t] = w
        if reason not in why[t]:
            why[t].append(reason)

    vr = value_referents(token)
    closed_vias = {r.get("via_table") for r in vr["closed"] if r.get("via_table")}
    for r in vr["direct"]:
        w = 0.3 if r["table"] in closed_vias else 1.0
        _credit(r["table"], w, f"value of {r['column']}")
    for r in vr["closed"]:
        _credit(r["table"], 0.6, f"value via {r['column']}")

    idf = ent["idf"]
    _w_idf = idf.get(sw, idf.get(token, 0.0))
    if _w_idf >= QSR_REFERENT_MIN_IDF:
        for t in ent["entity_index"].get(sw, ()):
            _credit(t, 0.4 + 0.4 * min(_w_idf, 1.0),
                    f"entity word (idf {round(_w_idf, 2)})")

    owners = ent.get("colname_owners", {})
    counts = ent.get("colname_counts", {})
    for key in ((sw,) if sw == token else (sw, token)):
        cnt = counts.get(key, 0)
        if 0 < cnt <= QSR_REFERENT_MAX_COLS:
            for t in owners.get(key, ()):
                _credit(t, 0.5, f"column-name word ({cnt} column{'s' if cnt > 1 else ''})")

    return [{"table": t, "score": round(s, 3), "why": why[t]}
            for t, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def domain_via(table: str, fk_column: str, via_column: Optional[str] = None,
               limit: int = 12) -> List[str]:
    """The FK-SCOPED value domain of `table.fk_column` — the labels actually
    reachable through that foreign key (precomputed per edge at derive time).
    THIS is the grounded-clarify payload for closed referents: a shared lookup's
    full domain would mix in every other list it serves. `via_column` scopes to
    one label column (pass the via_column of a sibling resolved value); omitted →
    the smallest available label domain (label columns are tighter than meta
    columns of shared lookups, so smallest ≈ the human label set)."""
    try:
        from ingestion.value_referents import load_value_referents
        doms = load_value_referents().get("edge_domains", {})
        prefix = f"{table}.{fk_column}|"
        cands = {k[len(prefix):]: v for k, v in doms.items() if k.startswith(prefix)}
        if not cands:
            return []
        if via_column and via_column in cands:
            return cands[via_column][:limit]
        # default: smallest INFORMATIVE domain. Size-1 "domains" are meta columns
        # (a lookup's table_name/column_name) — no clarify value, drop them first.
        informative = [v for v in cands.values() if len(v) > 1] or list(cands.values())
        return min(informative, key=len)[:limit]
    except Exception:
        return []


def domain_for(table: str, column: str, limit: int = 12) -> List[str]:
    """The column's actual sampled value domain — the payload of a grounded
    clarify ('payment statuses here are captured / authorized / cancelled')."""
    try:
        from config import COLUMN_VALUES_TABLE_NAME as TBL
        conn = _internal_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT DISTINCT value_raw FROM {TBL} "
                        f"WHERE table_name = %s AND col_name = %s LIMIT %s",
                        (table, column, limit))
            return sorted({str(r[0]) for r in cur.fetchall() if r[0] is not None})
        finally:
            conn.close()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------
_GATE_STRIP = None


def _language_layer() -> set:
    global _GATE_STRIP
    if _GATE_STRIP is None:
        try:
            from veda.validation import _gate_strip
            _GATE_STRIP = _gate_strip()
        except Exception:
            _GATE_STRIP = set()
    return _GATE_STRIP


def resolve(query: str, sm=None) -> List[TokenResolution]:
    """Resolve every content span of `query` to typed referents. Bigrams are tried
    first for VALUE spans ('high priority'); consumed words don't re-resolve as
    unigrams. Grammar/language words are typed, not dropped — consumers decide."""
    ent = _scope(sm)
    lang = _language_layer()
    words = re.findall(r"[a-z0-9]+", query.lower())
    out: List[TokenResolution] = []
    consumed = [False] * len(words)

    # pass 1: bigram VALUE spans
    for i in range(len(words) - 1):
        if consumed[i] or consumed[i + 1]:
            continue
        bigram = f"{words[i]} {words[i + 1]}"
        vr = value_referents(bigram)
        if vr["direct"] or vr["closed"]:
            out.append(TokenResolution(span=bigram, values=vr))
            consumed[i] = consumed[i + 1] = True

    # pass 2: unigrams
    for i, w in enumerate(words):
        if consumed[i] or len(w) < 3:
            continue
        sw = _singularize(w)
        tr = TokenResolution(span=w)
        # grammar is a FLAG, not a filter: 'value' is language-layer (ignored by the
        # qualifier gate) yet also a measure alias — consumers need both readings.
        tr.grammar = w in lang or sw in lang
        tr.values = value_referents(w)
        idf = ent["idf"]
        tr.entities = sorted(((t, round(idf.get(sw, idf.get(w, 1.0)), 3))
                              for t in ent["entity_index"].get(sw, ())),
                             key=lambda x: (-x[1], x[0]))
        tr.measures = ent["measures"].get(sw, [])
        tr.dimensions = ent["dimensions"].get(sw, [])
        out.append(tr)
    return out
