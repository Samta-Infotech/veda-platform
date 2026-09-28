"""veda.agent.tools — the substrate as tools (planner agent, part A).

Thin, deterministic, READ-ONLY wrappers over what ingestion already publishes. Each tool
returns a compact JSON-able dict (≈ ≤ 300 tokens), scoped to the request's sources and
RBAC, and every call is appended to `ToolBox.log` — the ONLY record `Plan.validate`
accepts identifiers from. Nothing here calls the SLM.

    find_entities(phrase, k)        entity cards (name/plural/alias), synthetic-question
                                    embeddings (BGE-M3), card-word overlap
    describe(table)                 entity card + semantic types + relationship graph
    columns(table, kind, phrase)    semantic types + measure glossary + column aliases
    join_path(a, b)                 relationship graph (declared first, then inferred) via
                                    query.join_planner; cross_source_fk edges across sources
    values(table, column, phrase)   value glossary + sampled values + live DISTINCT (bounded)
    similar_questions(text, k)      synthetic questions + verified trace frames
    probe(fragment)                 one read-only COUNT round trip (2 s)
    doc_sections(query, source_id)  chunk retrieval (no synthesis)

Table keys are the SCOPE's keys (a colliding bare name is `src{N}.table`, exactly as the
vocabulary and the frame path key them); SQL is always written against the bare name.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

TOOL_NAMES = ("find_entities", "describe", "columns", "join_path", "values",
              "similar_questions", "probe", "doc_sections", "retrieve")
KINDS = ("MONETARY", "TEMPORAL", "CATEGORY", "IDENTIFIER", "METRIC", "any")

_MAX_STR = 90


def _bare(t: str) -> str:
    return t.split(".", 1)[1] if t.startswith("src") and "." in t else t


def _short(s: Any, n: int = _MAX_STR) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _toks(s: str) -> set:
    try:
        from veda.understanding.vocabulary import norm_phrase
        return set(norm_phrase(s).split())
    except Exception:
        return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))


def _stype(meta: dict) -> str:
    return str((meta or {}).get("semantic_type") or "").upper()


def _kind_of(meta: dict) -> str:
    st = _stype(meta)
    if st in ("MONETARY", "TEMPORAL", "IDENTIFIER", "METRIC"):
        return st
    if st in ("CATEGORY", "FLAG", "BOOLEAN"):
        return "CATEGORY"
    return st or "OTHER"


class ToolError(Exception):
    pass


class ToolBox:
    """One planner run's tools over (sm, vocab) with a call log.

    `log` entries: {id, tool, args, result, ms, auto}. `routes` holds every join route a
    join_path call returned, by id (its edges) — the plan may only join along these."""

    def __init__(self, sm, vocab, question: str = "", *, ctx=None,
                 executor: Optional[Callable] = None):
        self.sm = sm or {}
        self.vocab = vocab
        self.question = question or ""
        self.ctx = ctx if ctx is not None else _ctx()
        self._exec = executor
        self.log: List[Dict[str, Any]] = []
        self.routes: Dict[str, List[Dict[str, Any]]] = {}
        self._allowed = None

    # ── dispatch ────────────────────────────────────────────────────────────────────
    def call(self, tool: str, args: Optional[Dict[str, Any]] = None, *, auto: bool = False
             ) -> Tuple[str, Dict[str, Any]]:
        args = dict(args or {})
        cid = f"c{len(self.log) + 1}"
        t0 = time.time()
        fn = getattr(self, f"t_{tool}", None) if tool in TOOL_NAMES else None
        try:
            if fn is None:
                raise ToolError(f"unknown tool {tool!r}")
            res = fn(**args)
        except ToolError as e:
            res = {"error": str(e)}
        except TypeError as e:
            res = {"error": f"bad arguments for {tool}: {str(e)[:120]}"}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        ent = {"id": cid, "tool": tool, "args": args, "result": res,
               "ms": round((time.time() - t0) * 1000.0, 1), "auto": bool(auto)}
        self.log.append(ent)
        return cid, res

    # ── scope / RBAC ────────────────────────────────────────────────────────────────
    def tables(self) -> List[str]:
        if self._allowed is None:
            # the scope's own tables: the vocabulary's cards (built for exactly the request's
            # sources); the semantic model only when there is no vocabulary. A tabular part's
            # model is the relational one with the datalake merged in — its homzhub tables
            # are not this part's to plan over
            cards = list((self.vocab.cards if self.vocab else {}) or {})
            tabs = set(cards) if cards else set((self.sm.get("tables") or {}).keys())
            try:
                from veda.rbac_filter import narrow_allowed
                t2, _ = narrow_allowed(tabs, [], self.sm, self.ctx)
                tabs = set(t2)
            except Exception:
                pass
            self._allowed = tabs
        return sorted(self._allowed)

    def _table(self, table: str) -> str:
        t = str(table or "").strip()
        allowed = set(self.tables())
        if t in allowed:
            return t
        hit = [x for x in allowed if _bare(x) == t]
        if len(hit) == 1:
            return hit[0]
        raise ToolError(f"unknown table {t!r} — use a table returned by find_entities/describe")

    def _cols(self, table: str) -> Dict[str, dict]:
        cols = (self.sm.get("columns") or {})
        out = {k.rsplit(".", 1)[1]: (v or {}) for k, v in cols.items() if k.rsplit(".", 1)[0] == table}
        if not out and table != _bare(table):
            out = {k.rsplit(".", 1)[1]: (v or {}) for k, v in cols.items()
                   if k.rsplit(".", 1)[0] == _bare(table)}
        try:
            from veda.rbac_filter import narrow_allowed
            _t, keep = narrow_allowed({table}, list(out), self.sm, self.ctx)
            if keep is not None and self.ctx is not None and getattr(self.ctx, "allowed_resources", None) is not None:
                out = {c: m for c, m in out.items() if c in set(keep)}
        except Exception:
            pass
        return out

    def _card(self, table: str) -> Dict[str, Any]:
        return dict(((self.vocab.cards if self.vocab else {}) or {}).get(table) or {})

    def source_of(self, table: str) -> Optional[str]:
        sid = ((self.vocab.source_of if self.vocab else {}) or {}).get(table)
        if sid is None:
            sid = str(((self.sm.get("tables") or {}).get(table) or {}).get("_source_id") or "") or None
        if sid is None and self.ctx is not None:
            sid = str(getattr(self.ctx, "source_id", "") or "") or None
        return sid

    def _name(self, table: str) -> str:
        c = self._card(table)
        return c.get("business_name") or _bare(table).replace("_", " ")

    # ── embedding lookups (BGE-M3 dense stores + the cross-encoder) ────────────────────
    def _scope_ids(self) -> List[str]:
        sids = sorted({str(v) for v in ((self.vocab.source_of if self.vocab else {}) or {}).values()})
        if not sids and self.ctx is not None:
            sids = [str(x) for x in (getattr(self.ctx, "source_ids", None) or (getattr(self.ctx, "source_id", None),)) if x is not None]
        return sids

    def _key_of(self, sid: str, bare: str) -> Optional[str]:
        allowed = set(self.tables())
        if bare in allowed and (self.source_of(bare) in (None, sid)):
            return bare
        q = f"src{sid}.{bare}"
        return q if q in allowed else (bare if bare in allowed else None)

    def _qvec(self, text: str):
        from veda.runtime import _encode_query
        return _encode_query(text)

    def _ann(self, phrase: str, *, tables: bool, k: int, table_filter: Optional[str] = None):
        """[(scope_table, column|None, cosine, passage_text)] nearest to `phrase`."""
        # the embedding stores live in the ENGINE database (veda_engine), never the
        # source's: veda.runtime._pg() would query the source DB and find no such table
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        from config import BIENCODER_COL_TABLE, BIENCODER_TABLE_TABLE
        v = self._qvec(phrase)
        vec = "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"
        sids = self._scope_ids()
        tbl = BIENCODER_TABLE_TABLE if tables else BIENCODER_COL_TABLE
        sql = (f"SELECT source_id, table_name, col_name, text, 1 - (embedding <=> %s::vector) AS cos "
               f"FROM {tbl} WHERE source_id = ANY(%s)")
        args: List[Any] = [vec, sids]
        if table_filter:
            sql += " AND table_name = %s"
            args.append(table_filter)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        args += [vec, int(k)]
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
        out = []
        for sid, t, c, txt, cos in rows:
            key = self._key_of(str(sid), t)
            if key is None:
                continue
            out.append((key, None if tables else c, float(cos), str(txt or "")))
        return out

    def _cross(self, query: str, texts: List[str]) -> Optional[List[float]]:
        if not texts:
            return []
        try:
            from query.reranker import _get_reranker
            rr = _get_reranker()
            if rr is None:
                return None
            sc = rr.predict([[query, t[:512]] for t in texts], batch_size=32)
            return [float(x) for x in sc]
        except Exception:
            return None

    # ── retrieve (the spine: the agent's first observation) ─────────────────────────
    def t_retrieve(self, text: str, k: int = 15) -> Dict[str, Any]:
        """The existing retrieval spine (query.retrieval_select.select_retrieval: schema
        link / bi-encoder + cross-encoder rerank / graph) on the part's text — its top
        columns with their tables, business names, kinds and scores."""
        from query.retrieval_select import select_retrieval
        sel = select_retrieval(query=str(text or self.question), source_ids=self._scope_ids() or None,
                               intent="sql", verbose=False)
        allowed = set(self.tables())
        cols = []
        for c in list(getattr(sel, "columns", None) or []):
            key = self._key_of(str(getattr(c, "source_id", "") or ""), getattr(c, "table_name", ""))
            if key is None or key not in allowed:
                continue
            cols.append((float(getattr(c, "similarity", 0.0) or 0.0), key, getattr(c, "col_name", "")))
        cols.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for sc, t, c in cols:
            if (t, c) in seen:
                continue
            seen.add((t, c))
            meta = self._cols(t).get(c) or {}
            out.append({"table": t, "column": c, "kind": _kind_of(meta), "score": round(sc, 3)})
            if len(out) >= int(k or 15):
                break
        tabs = list(dict.fromkeys(x["table"] for x in out))
        return {"columns": out,
                "tables": [{"table": t, "business_name": self._name(t)} for t in tabs[:8]]}

    # ── find_entities ───────────────────────────────────────────────────────────────
    def t_find_entities(self, phrase: str, k: int = 5) -> Dict[str, Any]:
        """Embed the phrase → cosine over the table and column embedding stores (scoped) →
        cross-encoder rerank → top k. Card-name / alias hits are merged in as a bonus."""
        phrase = str(phrase or "").strip()
        if not phrase:
            raise ToolError("find_entities needs a phrase")
        k = max(1, min(int(k or 5), 8))
        allowed = set(self.tables())
        cand: Dict[str, Dict[str, Any]] = {}

        def add(t, cos, via=None, why=""):
            if t not in allowed:
                return
            c = cand.setdefault(t, {"cos": 0.0, "via": None, "why": why})
            if cos > c["cos"]:
                c["cos"] = cos
                if via:
                    c["via"] = via
            if why and not c["why"]:
                c["why"] = why
        try:
            for t, _c, cos, _txt in self._ann(phrase, tables=True, k=10):
                add(t, cos, why="table embedding")
            for t, c, cos, _txt in self._ann(phrase, tables=False, k=20):
                add(t, cos, via=c, why="column embedding")
        except Exception:
            pass
        lex: Dict[str, float] = {}
        try:
            from veda.understanding.frame_grounding import name_hits
            for h in name_hits(self.vocab, phrase):
                if h["table"] in allowed:
                    s = 1.0 if (h.get("strong") and not h.get("modifier")) else 0.6
                    lex[h["table"]] = max(lex.get(h["table"], 0.0), s)
                    add(h["table"], 0.0, why=f"name:{h['phrase']}")
        except Exception:
            pass
        if not cand:
            return {"phrase": phrase, "entities": []}
        # cross-encoder over the card sentence (+ the column that brought the table in)
        order = sorted(cand, key=lambda t: -(cand[t]["cos"] + 0.2 * lex.get(t, 0.0)))[:14]
        texts = []
        for t in order:
            c = self._card(t)
            txt = f"{self._name(t)} ({', '.join((c.get('aliases') or [])[:6])}): {c.get('one_row_is') or ''}"
            if cand[t]["via"]:
                txt += f"; column {cand[t]['via'].replace('_', ' ')}"
            texts.append(txt)
        ce = self._cross(phrase, texts)
        scored = []
        for i, t in enumerate(order):
            x = cand[t]
            base = x["cos"] if ce is None else 0.6 * ce[i] + 0.4 * x["cos"]
            scored.append((base + 0.15 * lex.get(t, 0.0), t))
        scored.sort(key=lambda z: (-z[0], t))
        ranked = scored[:k]
        top = {t for _s, t in ranked}
        out = []
        for sc, t in ranked:
            c = self._card(t)
            dist = {o: _short(v, 70) for o, v in (c.get("distinguishes_from") or {}).items() if o in top}
            why = cand[t]["why"]
            if lex.get(t, 0) >= 1.0:
                why = next((w for w in [cand[t]["why"]] if w.startswith("name:")), "name:" + self._name(t))
            out.append({"table": t, "source_id": self.source_of(t), "business_name": self._name(t),
                        "one_row_is": _short(c.get("one_row_is") or c.get("primary_entity"), 80),
                        "score": round(min(sc, 1.5), 2), "why": why,
                        **({"via_column": cand[t]["via"]} if cand[t]["via"] else {}),
                        **({"distinguishes_from": dist} if dist else {})})
        return {"phrase": phrase, "entities": out}

    # ── describe ────────────────────────────────────────────────────────────────────
    def _graph_edges(self, table: str) -> List[dict]:
        sid = self.source_of(table)
        try:
            from ingestion.vocabulary import _graph
            g = _graph(sid, getattr(self.ctx, "tenant", None) or "default")
            return list(g.get("edges") or [])
        except Exception:
            return []

    def _scope_key(self, bare_table: str, like: str) -> Optional[str]:
        """The scope key of a bare graph table in the same source as `like`."""
        allowed = set(self.tables())
        if bare_table in allowed and self.source_of(bare_table) == self.source_of(like):
            return bare_table
        q = f"src{self.source_of(like)}.{bare_table}"
        if q in allowed:
            return q
        return bare_table if bare_table in allowed else None

    def t_describe(self, table: str) -> Dict[str, Any]:
        t = self._table(table)
        c = self._card(t)
        cols = self._cols(t)
        mg = (self.vocab.measure_glossary if self.vocab else {}) or {}
        vg = (self.vocab.value_glossary if self.vocab else {}) or {}
        life = c.get("lifecycle_column")
        life_vals = list((vg.get(f"{t}.{life}") or {}).keys())[:8] if life else []
        if life and not life_vals:
            life_vals = [str(v) for v in ((cols.get(life) or {}).get("sample_values") or [])[:8]]
        measures = []
        for m in (c.get("key_measures") or [])[:5]:
            if m in cols:
                ph = (mg.get(f"{t}.{m}") or {}).get("phrases") or []
                measures.append({"column": m, "phrases": [p for p in ph if p != m.replace("_", " ")][:3]})
        dims = [d for d in (c.get("key_dimensions") or []) if d in cols and d != life][:5]
        parents, children = [], []
        b = _bare(t)
        for e in self._graph_edges(t):
            if e.get("discovery") != "declared_fk" or e.get("polymorphic"):
                continue
            if e.get("source_table") == b and e.get("cardinality") in ("N:1", "1:1"):
                pk = self._scope_key(e.get("target_table"), t)
                if pk and pk != t and all(p["table"] != pk or p["via"] != e.get("source_column") for p in parents):
                    parents.append({"table": pk, "via": e.get("source_column"), "name": self._name(pk)})
            elif e.get("target_table") == b and e.get("cardinality") in ("N:1", "1:1"):
                ck = self._scope_key(e.get("source_table"), t)
                if ck and ck != t and all(x["table"] != ck for x in children):
                    children.append({"table": ck, "via": e.get("source_column"), "name": self._name(ck),
                                     "_imp": int(self._card(ck).get("importance") or 0)})
        parents = parents[:6]
        children = sorted(children, key=lambda x: -x.pop("_imp"))[:3] if children else []
        bdate = c.get("business_date_column")
        return {"table": t, "business_name": self._name(t),
                "one_row_is": _short(c.get("one_row_is") or c.get("primary_entity"), 90),
                "business_date": bdate if bdate in cols else None,
                "lifecycle": ({"column": life, "values": life_vals} if life in cols else None),
                "measures": measures, "dimensions": dims,
                "display": c.get("display_column") if c.get("display_column") in cols else None,
                "key": "id" if "id" in cols else None,
                "parents": parents, "children": children, "n_columns": len(cols)}

    # ── columns ─────────────────────────────────────────────────────────────────────
    def t_columns(self, table: str, kind: str = "any", phrase: Optional[str] = None) -> Dict[str, Any]:
        t = self._table(table)
        kind = str(kind or "any").upper()
        kind = "any" if kind in ("ANY", "", "NONE") else kind
        if kind != "any" and kind not in KINDS:
            raise ToolError(f"kind must be one of {', '.join(KINDS)}")
        mg = (self.vocab.measure_glossary if self.vocab else {}) or {}
        pt = _toks(phrase or "")
        rows = []
        # MONETARY and METRIC are both measures — a model asking for one gets both
        want = {"MONETARY", "METRIC"} if kind in ("MONETARY", "METRIC") else {kind}
        for col, meta in self._cols(t).items():
            k = _kind_of(meta)
            if kind != "any" and k not in want:
                continue
            phr = [p for p in ((mg.get(f"{t}.{col}") or {}).get("phrases") or [])]
            phr += [a for a in (meta.get("aliases") or []) if a not in phr]
            sc = 0.0
            if pt:
                words = _toks(col.replace("_", " ") + " " + " ".join(phr))
                sc = len(pt & words) / max(1, len(pt))
            imp = {"HIGH": 2, "MEDIUM": 1}.get(str(meta.get("importance_class") or "").upper(), 0)
            rows.append((sc, imp, col, k, phr, meta))
        if phrase:
            # rank by meaning: the phrase against this table's column embeddings
            try:
                emb = {c: cos for _t, c, cos, _x in self._ann(phrase, tables=False, k=60,
                                                            table_filter=_bare(t))}
            except Exception:
                emb = {}
            if emb:
                rows = [(0.7 * emb.get(r[2], 0.0) + 0.3 * r[0],) + r[1:] for r in rows]
        rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
        if pt and not phrase and any(r[0] > 0 for r in rows):
            rows = [r for r in rows if r[0] > 0] + [r for r in rows if r[0] == 0][:3]
        out = []
        for sc, _imp, col, k, phr, meta in rows[:10]:
            ent = {"column": col, "kind": k}
            if phr:
                ent["phrases"] = [_short(p, 30) for p in phr[:3]]
            sv = [str(v)[:24] for v in (meta.get("sample_values") or [])[:3]]
            if sv:
                ent["sample_values"] = sv
            out.append(ent)
        return {"table": t, "kind": kind, "columns": out,
                **({"more": max(0, len(rows) - 10)} if len(rows) > 10 else {})}

    # ── join_path ───────────────────────────────────────────────────────────────────
    # A join is chosen as a whole ROUTE (id r1, r2…): every edge of a multi-hop route is
    # part of it, so the model never has to assemble a path edge by edge. Each route's
    # path reads child.fk=parent.pk per hop (the FK side first).
    def _edge(self, e: dict, basis: str, a_key: str, b_key: str) -> Dict[str, Any]:
        st, tt = e["source_table"], e["target_table"]

        def key(bare):
            for k in (a_key, b_key):
                if _bare(k) == bare:
                    return k
            return self._scope_key(bare, a_key) or bare
        ent = {"source_table": key(st), "source_column": e["source_column"],
               "target_table": key(tt), "target_column": e["target_column"],
               "cardinality": e.get("cardinality") or "N:1", "kind": basis,
               "requires_predicate": e.get("requires_predicate"),
               "relationship_type": e.get("relationship_type")}
        if e.get("_sources"):
            ent["sources"] = e["_sources"]
        return ent

    def _route(self, edges: List[dict], basis: str, why: str, ta: str, tb: str) -> Dict[str, Any]:
        # after the highest id in use: a follow-up's prior routes keep their ids, and its log
        # may have been trimmed, so len() could collide with a surviving prior id
        rid = f"r{max((int(k[1:]) for k in self.routes if k[1:].isdigit()), default=0) + 1}"
        es = [self._edge(e, basis, ta, tb) for e in edges]
        self.routes[rid] = es
        via = [t for ed in es for t in (ed["source_table"], ed["target_table"]) if t not in (ta, tb)]
        out = {"id": rid, "path": [f"{ed['source_table']}.{ed['source_column']}={ed['target_table']}.{ed['target_column']}"
                                   for ed in es], "why": why, "basis": basis}
        if via:
            out["via"] = list(dict.fromkeys(via))
        cards = [ed["cardinality"] for ed in es]
        if any(c != "N:1" for c in cards):
            out["cards"] = cards
        return out

    def t_join_path(self, a: str, b: str) -> Dict[str, Any]:
        ta, tb = self._table(a), self._table(b)
        if ta == tb:
            raise ToolError("join_path needs two different tables")
        sa, sb = self.source_of(ta), self.source_of(tb)
        if sa and sb and sa != sb:
            opts = self._cross_source(ta, tb, sa, sb)
            if not opts:
                return {"a": ta, "b": tb, "routes": [],
                        "note": f"no join between source {sa} and source {sb} — answer them as separate parts (split)"}
            return {"a": ta, "b": tb, "routes": opts}
        try:
            from query.join_planner import _join_key_candidates
        except Exception as e:
            raise ToolError(f"join planner unavailable: {type(e).__name__}")
        edges = self._graph_edges(ta)
        declared = {"edges": [e for e in edges if e.get("discovery") == "declared_fk"]}
        ba, bb = _bare(ta), _bare(tb)
        seen_paths = set()
        cands: List[Tuple[List[dict], str, str]] = []

        def add(path, basis, why):
            k = tuple((e["source_table"], e["source_column"], e["target_table"], e["target_column"]) for e in path)
            if not path or k in seen_paths:
                return
            seen_paths.add(k)
            cands.append((path, basis, why))

        # several DIRECT declared keys (assigned_to_id vs created_by_id) are alternatives
        direct = list(_join_key_candidates(declared, ba, bb))
        if len(direct) > 1:
            for e in direct[:4]:
                add([e], "declared", f"direct via {e['source_column']}")
        # multi-hop routes over DECLARED keys: a chain in one direction, a link table, a
        # shared parent — each labelled with the tables it passes through
        for sc, path, why in _routes(ba, bb, declared["edges"])[:12]:
            add(path, "declared", why)
        if not cands:
            # value-overlap (inferred) edges: a DIRECT edge only, never a multi-hop chain of guesses
            inf = [e for e in edges if {e.get("source_table"), e.get("target_table")} == {ba, bb}
                   and e.get("discovery") != "declared_fk"]
            for e in inf[:2]:
                add([e], "inferred", f"inferred value overlap on {e['source_column']}")
        if not cands:
            return {"a": ta, "b": tb, "routes": [], "note": "no join path within 3 hops"}
        # rank by MEANING: the question against each route's verbalisation (the relationship
        # sentences of ingestion/link_text: "the user this ticket is assigned to"), minus a
        # penalty for passing through audit / history / attachment tables
        scores = self._route_scores(ta, [c[0] for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i][0])[:3]
        opts = []
        for i in order:
            path, basis, why = cands[i]
            r = self._route(path, basis, why, ta, tb)
            r["route_score"] = round(scores[i][0], 3)
            r["says"] = _short(scores[i][1], 120)
            opts.append(r)
        return {"a": ta, "b": tb, "routes": opts}

    def _route_scores(self, ta: str, paths: List[List[dict]]) -> List[Tuple[float, str]]:
        """(score, verbalisation) per path; score = cosine(question, verbalisation) − penalty."""
        sid = self.source_of(ta)
        if not hasattr(self, "_fkph"):
            self._fkph = {}
        if sid not in self._fkph:
            try:
                from ingestion.link_text import for_source
                self._fkph[sid] = for_source(sid, getattr(self.ctx, "tenant", None) or "default")[1]
            except Exception:
                self._fkph[sid] = {}
        ph = self._fkph[sid]
        texts, pens = [], []
        for path in paths:
            bits, pen = [], 0.0
            for e in path:
                bits.append(ph.get(f"{e['source_table']}.{e['source_column']}")
                            or f"{e['source_table']} {e['source_column']} to {e['target_table']}")
                if e.get("relationship_type") == "audit":
                    pen += 0.04
            for t in {x for e in path for x in (e["source_table"], e["target_table"])}:
                if re.search(r"(history|_log|log$|audit|attachment|document|comment|notification)", t):
                    pen += 0.03
            texts.append("; ".join(bits))
            pens.append(pen + 0.01 * (len(path) - 1))
        try:
            import numpy as np
            from ingestion.m3_encoder import encode_dense
            m = np.asarray(encode_dense(texts), dtype="float32")
            m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
            q = np.asarray(self._qvec(self.question), dtype="float32").reshape(-1)
            cos = [float(x) for x in m @ q]
        except Exception:
            cos = [0.0] * len(texts)
        return [(c - p, t) for c, p, t in zip(cos, pens, texts)]

    def _cross_source(self, ta, tb, sa, sb) -> List[Dict[str, Any]]:
        try:
            from query.federated_route import _join_hints
            hints = _join_hints({sa: {_bare(ta): []}, sb: {_bare(tb): []}})
        except Exception:
            hints = []
        out = []
        for h in hints:
            ends = {(h["a_src"], h["a_tbl"]), (h["b_src"], h["b_tbl"])}
            if ends != {(sa, _bare(ta)), (sb, _bare(tb))}:
                continue
            a_first = (h["a_src"], h["a_tbl"]) == (sa, _bare(ta))
            e = {"source_table": _bare(ta), "target_table": _bare(tb),
                 "source_column": h["a_col"] if a_first else h["b_col"],
                 "target_column": h["b_col"] if a_first else h["a_col"],
                 "cardinality": "N:1", "_sources": [sa, sb]}
            out.append(self._route([e], "cross_source_fk", f"cross-source value overlap ({h.get('tier')})", ta, tb))
        return out[:2]

    # ── values ──────────────────────────────────────────────────────────────────────
    def t_values(self, table: str, column: str, phrase: Optional[str] = None, k: int = 8) -> Dict[str, Any]:
        t = self._table(table)
        cols = self._cols(t)
        if column not in cols:
            raise ToolError(f"{t} has no column {column!r}")
        k = max(1, min(int(k or 8), 12))
        vg = ((self.vocab.value_glossary if self.vocab else {}) or {}).get(f"{t}.{column}") or {}
        from veda.understanding.frame_grounding import column_domain, ground_value
        dom = column_domain(self.vocab, self.sm, t, column, live=True) if self.vocab else []
        if not dom:
            try:
                from veda.understanding.frame_probes import distinct_values
                dom = distinct_values(_bare(t), column, limit=41) or []
            except Exception:
                dom = []
        matched = None
        order = list(dom)
        if phrase:
            v, note, method = ground_value(self.vocab, self.sm, t, column, phrase, self.question)
            if v is not None:
                matched = {"value": v, "method": method, **({"note": note} if note else {})}
                order = [v] + [x for x in order if str(x) != str(v)]
            else:
                # not an exact / glossary value: rank the stored values by meaning
                pt = _toks(phrase)
                sims = {}
                try:
                    import numpy as np
                    from ingestion.m3_encoder import encode_dense
                    vals = [str(x) for x in order[:60]]
                    if vals:
                        m = np.asarray(encode_dense(vals), dtype="float32")
                        q = np.asarray(self._qvec(phrase), dtype="float32").reshape(-1)
                        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
                        sims = {v: float(x) for v, x in zip(vals, m @ q)}
                except Exception:
                    sims = {}
                order = sorted(order, key=lambda x: (-len(pt & _toks(str(x))), -sims.get(str(x), 0.0)))
        out = [{"value": str(v)[:40], **({"phrases": [p for p in vg.get(v, [])][:3]} if vg.get(v) else {})}
               for v in order[:k]]
        return {"table": t, "column": column, "values": out,
                **({"match": matched} if matched else {}),
                **({"more": len(order) - k} if len(order) > k else {}),
                **({"kind": _kind_of(cols[column])})}

    # ── similar_questions ───────────────────────────────────────────────────────────
    def t_similar_questions(self, text: str, k: int = 3) -> Dict[str, Any]:
        from veda.understanding.vocabulary import nearest_examples
        k = max(1, min(int(k or 3), 5))
        out = []
        allowed = set(self.tables())
        for ex in nearest_examples(self.vocab, str(text or self.question), k=k * 2):
            fr = ex.get("frame") or {}
            t = ex.get("table")
            if t and t not in allowed:
                continue
            plan = {"table": t, "agg": fr.get("aggregation") or "none"}
            for key in ("measure", "group_by", "order", "limit"):
                if fr.get(key):
                    plan[key] = fr[key]
            if fr.get("filters"):
                plan["filters"] = [[f.get("concept"), f.get("op"), f.get("value")] for f in fr["filters"]][:3]
            out.append({"question": _short(ex.get("question"), 80), "plan": plan})
            if len(out) >= k:
                break
        return {"examples": out}

    # ── probe ───────────────────────────────────────────────────────────────────────
    def t_probe(self, tables: List[str] = None, joins: List[str] = None, filters: List[dict] = None,
                order: Optional[str] = None, **_ignored) -> Dict[str, Any]:
        from veda.agent.plan import split_ref
        tabs = [self._table(t) for t in (tables or [])]
        if not tabs:
            raise ToolError("probe needs tables")
        edges = []
        for j in joins or []:
            if j not in self.routes:
                raise ToolError(f"unknown join {j!r} — use a route id returned by join_path")
            edges.extend(self.routes[j])
        filters = list(filters or [])
        src0 = self.source_of(tabs[0])
        # a probe runs on ONE source: filters on another source's table are probed on
        # that table alone, and a cross-source join is not probed (the executor that can
        # run it is the federated one) — the per-filter counts are what the plan needs
        local_edges = [e for e in edges if self.source_of(e["source_table"]) == src0
                       and self.source_of(e["target_table"]) == src0]
        groups: Dict[str, List[dict]] = {}
        for f in filters:
            t, _c = split_ref(f.get("col"))
            groups.setdefault(self.source_of(t) if t else src0, []).append(f)
        res = self._probe_one(tabs[0], local_edges, groups.pop(src0, []), order, src0)
        if res.get("error"):
            return res
        for sid, fl in groups.items():
            ft = split_ref(fl[0].get("col"))[0]
            r2 = self._probe_one(self._table(ft), [], fl, None, sid)
            if r2.get("error"):
                return r2
            res.setdefault("filters", []).extend(r2.get("filters") or [])
            res.pop("rows_after_filters", None)          # not one number across two sources
            res["note"] = "filters on another source were counted on their own table"
        return res

    def _probe_one(self, table: str, edges, filters, order, sid) -> Dict[str, Any]:
        from veda.agent.plan import probe_sql
        sql, keys = probe_sql([table], edges, filters, order, self.sm)
        t0 = time.time()
        cols, rows, err = self._run(sql, sid)
        ms = round((time.time() - t0) * 1000.0, 1)
        if err or not rows:
            return {"error": _short(err or "no rows", 160), "sql_ms": ms}
        rec = dict(zip([c.lower() for c in cols], rows[0]))
        res = {"rows_total": int(rec.get("n_all") or 0)}
        if any(k[0] == "f" for k in keys):
            res["rows_after_filters"] = int(rec.get("n_where") or 0)
            res["filters"] = [{"col": f.get("col"), "op": f.get("op", "="), "value": f.get("value"),
                               "rows": int(rec.get(name) or 0)}
                              for (kind, name, f) in keys if kind == "f"]
        if any(k[0] == "o" for k in keys):
            res["distinct_order_col"] = int(rec.get("n_order_distinct") or 0)
        res["sql_ms"] = ms
        return res

    def _run(self, sql: str, sid: Optional[str] = None):
        if self._exec is not None:
            return self._exec(sql)
        from veda.execution import execute_sql
        try:
            import config
            tmo = int(getattr(config, "FRAME_PROBE_TIMEOUT_MS", 2000))
        except Exception:
            tmo = 2000
        ctx = self.ctx
        narrowed = None
        # execute against the table's OWN source (a multi-source scope runs on its primary)
        if ctx is not None and sid and (str(getattr(ctx, "source_id", "")) != str(sid)
                                        or len(getattr(ctx, "source_ids", None) or ()) > 1):
            try:
                from veda_core.context import set_context
                narrowed = ctx
                set_context(ctx.narrowed(int(sid)))
            except Exception:
                narrowed = None
        try:
            return execute_sql(sql, None, timeout_ms=tmo)
        finally:
            if narrowed is not None:
                try:
                    from veda_core.context import set_context
                    set_context(narrowed)
                except Exception:
                    pass

    # ── doc_sections ────────────────────────────────────────────────────────────────
    def t_doc_sections(self, query: str, source_id: Optional[str] = None, k: int = 5) -> Dict[str, Any]:
        from query.rag_layer import _encode_rag_query
        from ingestion.chunk_embedder import retrieve_top_k_chunks
        k = max(1, min(int(k or 5), 6))
        sids = [str(source_id)] if source_id else [str(s) for s in (getattr(self.ctx, "source_ids", None) or ())]
        vec = _encode_rag_query(str(query or self.question))
        if vec is None:
            raise ToolError("embedding unavailable")
        chunks = retrieve_top_k_chunks(query_vector=vec, source_ids=sids or None, top_k=k)
        out = []
        for c in chunks or []:
            text = str(getattr(c, "text", "") or "")
            title = text.split("\n", 1)[0][:60]
            out.append({"title": _short(getattr(c, "doc_name", ""), 50), "section": _short(title, 60),
                        "page": getattr(c, "page_num", None), "snippet": _short(text, 160)})
        return {"sections": out}


def _routes(a: str, b: str, edges: List[dict], max_hops: int = 3):
    """Every simple route a→b of ≤ max_hops declared, non-polymorphic edges, scored
    (lower = better). Stepping child→parent is UP, parent→child is DOWN; an audit edge
    (created_by…) may only be a direct hop."""
    adj: Dict[str, List[Tuple[str, dict, str]]] = {}
    for e in edges:
        if e.get("polymorphic") or e.get("source_table") == e.get("target_table"):
            continue
        adj.setdefault(e["source_table"], []).append((e["target_table"], e, "up"))
        adj.setdefault(e["target_table"], []).append((e["source_table"], e, "down"))
    out = []

    def walk(node, path, dirs, seen):
        if node == b and path:
            ups, downs = dirs.count("up"), dirs.count("down")
            turns = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
            shared_parent = any(dirs[i] == "up" and dirs[i + 1] == "down" for i in range(len(dirs) - 1))
            score = len(path) + (2.5 if shared_parent else 0) + (1.0 * turns if not shared_parent else 0)
            via = [x for e in path for x in (e["source_table"], e["target_table"]) if x not in (a, b)]
            kind = ("chain" if turns == 0 else "shared parent" if shared_parent else "link table")
            why = f"{kind}" + (f" via {', '.join(dict.fromkeys(via))}" if via else "")
            out.append((score, list(path), why))
            return
        if len(path) >= max_hops:
            return
        for nxt, e, d in adj.get(node, []):
            if nxt in seen:
                continue
            if e.get("relationship_type") == "audit" and not (not path and nxt == b):
                continue
            walk(nxt, path + [e], dirs + [d], seen | {nxt})

    walk(a, [], [], {a})
    out.sort(key=lambda x: (x[0], [(e["source_table"], e["source_column"]) for e in x[1]]))
    return out


def _ctx():
    try:
        from veda_core.context import try_current
    except Exception:
        try:
            from context import try_current          # type: ignore
        except Exception:
            return None
    try:
        return try_current()
    except Exception:
        return None


_DROP_KEYS = {"n_columns", "sql_ms", "one_row_is", "source_id", "kind", "basis"}


def compact(obj: Any, _top: bool = True) -> Any:
    """Drop empty / null values and bookkeeping keys — what the model reads."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _DROP_KEYS and not (k == "basis" and v in ("inferred", "cross_source_fk")) \
                    and not (k == "kind" and (not _top or v != "any")):
                continue
            if k == "why" and _top is False and isinstance(v, str) and v.startswith(("name:", "similar", "word")):
                continue
            v2 = compact(v, False)
            if v2 in (None, [], {}, ""):
                continue
            out[k] = v2
        return out
    if isinstance(obj, list):
        return [compact(x, False) for x in obj]
    return obj


def render(entry: Dict[str, Any], max_chars: int = 900) -> str:
    """One log entry as a compact observation line for the model's working memory."""
    args = ",".join(f"{k}={json.dumps(v, ensure_ascii=False, separators=(',', ':'))}"
                    for k, v in (entry.get("args") or {}).items())
    body = json.dumps(compact(entry.get("result")), ensure_ascii=False, separators=(",", ":"), default=str)
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return f"[{entry['id']}] {entry['tool']}({args}) → {body}"
