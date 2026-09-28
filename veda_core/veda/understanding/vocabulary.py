"""veda.understanding.vocabulary — the QUERY-TIME view of the Stage 1 business vocabulary.

Loads the per-source artifacts written by ingestion/vocabulary.py (entity cards, value and
measure glossaries, synthetic questions + embeddings) for every source in the ambient
request scope, and presents them keyed by the SCOPE's table names (a table whose bare
name collides across sources is `src{ID}.{table}` in the merged semantic model — the
cards follow the same key).

A source with no published cards gets DETERMINISTIC cards built on the fly from the
scope's semantic model (plus its tracked seeds), so the frame path never depends on an
ingest having run the vocabulary stage — it just grounds with less vocabulary.

Pure reads, cached per (source, tenant, artifact mtime). No SLM, no DB.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LOCK = threading.Lock()
_CACHE: Dict[Tuple[str, str], Tuple[float, "SourceVocab"]] = {}
_SCOPE_CACHE: Dict[Any, "ScopeVocab"] = {}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())).strip()


def _sing(w: str) -> str:
    try:
        from retrieval.query_enrichment import _singularize
        return _singularize(w)
    except Exception:
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w


def norm_phrase(s: str) -> str:
    """Case/space/plural-insensitive phrase key: 'Ledger Entries' → 'ledger entry'."""
    return " ".join(_sing(w) for w in _norm(s).split())


@dataclass
class SourceVocab:
    source_id: str
    cards: Dict[str, Dict[str, Any]] = field(default_factory=dict)          # bare table → card
    value_glossary: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    measure_glossary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Any = None
    built: str = "artifact"                                                  # artifact | on_the_fly


@dataclass
class ScopeVocab:
    """All sources in scope, keyed by SCOPE table names."""
    cards: Dict[str, Dict[str, Any]] = field(default_factory=dict)          # scope table → card
    value_glossary: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)   # "scope_t.col"
    measure_glossary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)            # questions + trace frames
    embeddings: Any = None                                                   # rows aligned to `examples`
    source_of: Dict[str, str] = field(default_factory=dict)                 # scope table → source id
    built: Dict[str, str] = field(default_factory=dict)

    # ── name index ────────────────────────────────────────────────────────────────
    def name_index(self) -> Dict[str, List[Tuple[str, str]]]:
        """normalised phrase → [(scope_table, kind)] where kind ∈ business_name | plural |
        alias | primary_entity. Built lazily, once."""
        idx = getattr(self, "_name_idx", None)
        if idx is not None:
            return idx
        idx = {}
        for t, c in self.cards.items():
            for kind, phrases in (("business_name", [c.get("business_name")]),
                                  ("plural", [c.get("plural")]),
                                  ("alias", c.get("aliases") or [])):
                for p in phrases:
                    k = norm_phrase(p or "")
                    if k and (t, kind) not in idx.get(k, []):
                        idx.setdefault(k, []).append((t, kind))
        self._name_idx = idx                                   # type: ignore[attr-defined]
        return idx


def _artifact(name, sid, tenant) -> Optional[str]:
    try:
        from config import source_artifact_path
        p = source_artifact_path(name, sid, tenant)
        return p if os.path.exists(p) else None
    except Exception:
        return None


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _strip(d):
    return {k: v for k, v in (d or {}).items() if not str(k).startswith("_")}


def load_source(sid, tenant: str = "default", sm_for_fallback=None) -> SourceVocab:
    from ingestion.vocabulary import (CARDS_ARTIFACT, VALUE_GLOSSARY_ARTIFACT,
                                      MEASURE_GLOSSARY_ARTIFACT, QUESTIONS_ARTIFACT,
                                      QUESTIONS_EMB_ARTIFACT)
    sid = str(sid)
    cpath = _artifact(CARDS_ARTIFACT, sid, tenant)
    mt = os.path.getmtime(cpath) if cpath else -1.0
    key = (sid, tenant)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == mt and mt >= 0:
            return hit[1]
    sv = SourceVocab(source_id=sid)
    if cpath:
        sv.cards = _strip(_read(cpath))
        sv.value_glossary = _strip(_read(_artifact(VALUE_GLOSSARY_ARTIFACT, sid, tenant) or "") or {})
        sv.measure_glossary = _strip(_read(_artifact(MEASURE_GLOSSARY_ARTIFACT, sid, tenant) or "") or {})
        qp = _artifact(QUESTIONS_ARTIFACT, sid, tenant)
        if qp:
            try:
                with open(qp) as f:
                    sv.questions = [json.loads(l) for l in f if l.strip()]
            except Exception:
                sv.questions = []
        ep = _artifact(QUESTIONS_EMB_ARTIFACT, sid, tenant)
        if ep:
            try:
                import numpy as np
                emb = np.load(ep)
                if emb.shape[0] == len(sv.questions):
                    sv.embeddings = emb
            except Exception:
                sv.embeddings = None
    elif sm_for_fallback is not None:
        sv = _on_the_fly(sid, tenant, sm_for_fallback)
    if mt >= 0:
        with _LOCK:
            _CACHE[key] = (mt, sv)
    return sv


def _on_the_fly(sid, tenant, sm) -> SourceVocab:
    """Deterministic cards + glossaries from the (source-filtered) semantic model + seeds."""
    from ingestion import vocabulary as V
    sv = SourceVocab(source_id=str(sid), built="on_the_fly")
    try:
        domains = V.value_domains(sid, tenant, sm)
        sv.cards = V.build_entity_cards(sid, tenant, sm, use_slm=False, domains=domains)
        sv.value_glossary = V.build_value_glossary(sid, tenant, sm, sv.cards, domains, use_slm=False)
        sv.value_glossary = {k: {vv: ph for vv, ph in m.items() if not str(vv).startswith("_")}
                             for k, m in sv.value_glossary.items()}
        sv.measure_glossary = V.build_measure_glossary(sid, sm, sv.cards)
        sv.questions = V.template_questions(sv.cards, sv.value_glossary, sv.measure_glossary)
    except Exception:
        pass
    return sv


def _trace_frames(sid) -> List[Dict[str, Any]]:
    """Stage 0.2 verified (question, frame) pairs — few-shot pool, no embeddings."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "evaluation", "frames", f"{sid}.jsonl")
    out = []
    try:
        with open(root) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("question") and r.get("frame"):
                        out.append({**r, "origin": "trace"})
    except Exception:
        pass
    return out


def _source_sm(sm, sid) -> Dict[str, Any]:
    """The part of a (possibly merged) scope model owned by source `sid`, bare-keyed."""
    tabs = (sm or {}).get("tables", {}) or {}
    if not any("_source_id" in (m or {}) for m in tabs.values()):
        return sm
    keep = {t for t, m in tabs.items() if str((m or {}).get("_source_id")) == str(sid)}

    def bare(t):
        return t.split(".", 1)[1] if t.startswith("src") and "." in t else t
    return {"tables": {bare(t): tabs[t] for t in keep},
            "columns": {f"{bare(k.rsplit('.', 1)[0])}.{k.rsplit('.', 1)[1]}": v
                        for k, v in ((sm or {}).get("columns", {}) or {}).items()
                        if k.rsplit(".", 1)[0] in keep}}


def scope_vocab(sm, source_ids: Optional[List[str]] = None, tenant: Optional[str] = None) -> ScopeVocab:
    """Merge every in-scope source's vocabulary under the scope's table keys."""
    if source_ids is None or tenant is None:
        try:
            from veda_core.context import try_current
        except Exception:
            from context import try_current                 # type: ignore
        ctx = try_current()
        source_ids = source_ids or ([str(s) for s in (ctx.source_ids or (ctx.source_id,))] if ctx else [])
        tenant = tenant or (ctx.tenant if ctx else "default")
    tabs = (sm or {}).get("tables", {}) or {}
    from ingestion.vocabulary import CARDS_ARTIFACT
    ck = (tuple(str(x) for x in source_ids), tenant, id(sm), len(tabs),
          tuple((lambda p: os.path.getmtime(p) if p else -1.0)(_artifact(CARDS_ARTIFACT, str(x), tenant))
                for x in source_ids))
    with _LOCK:
        hit = _SCOPE_CACHE.get(ck)
    if hit is not None:
        return hit
    out = ScopeVocab()
    rowvecs: List[Any] = []
    for sid in [str(s) for s in source_ids]:
        sv = load_source(sid, tenant, sm_for_fallback=_source_sm(sm, sid))
        if not sv.cards and sv.built == "artifact":
            sv = _on_the_fly(sid, tenant, _source_sm(sm, sid))
        out.built[sid] = sv.built

        def skey(t):
            if t in tabs and (len(source_ids) == 1 or str((tabs[t] or {}).get("_source_id", sid)) == sid):
                return t
            q = f"src{sid}.{t}"
            return q if q in tabs else None
        remap = {}
        for t, c in sv.cards.items():
            k = skey(t)
            if k:
                remap[t] = k
                out.cards[k] = {**c, "table": k, "_bare_table": t, "_source_id": sid}
                out.source_of[k] = sid
        for key, v in sv.value_glossary.items():
            t, _, col = key.partition(".")
            if t in remap:
                out.value_glossary[f"{remap[t]}.{col}"] = {vv: ph for vv, ph in v.items()
                                                          if not str(vv).startswith("_")}
        for key, v in sv.measure_glossary.items():
            t, _, col = key.partition(".")
            if t in remap:
                out.measure_glossary[f"{remap[t]}.{col}"] = v
        qs = [dict(q, table=remap.get(q.get("table"), q.get("table")), _source_id=sid)
              for q in sv.questions if q.get("table") in remap]
        have = sv.embeddings is not None and sv.embeddings.shape[0] == len(sv.questions)
        rows = [i for i, q in enumerate(sv.questions) if q.get("table") in remap]
        for q, i in zip(qs, rows):
            out.examples.append(q)
            rowvecs.append(sv.embeddings[i] if have else None)
        for fr in _trace_frames(sid):
            out.examples.append({**fr, "_source_id": sid})
            rowvecs.append(None)
    out.__dict__["_rowvecs"] = rowvecs
    if rowvecs and all(v is not None for v in rowvecs):
        import numpy as np
        out.embeddings = np.vstack([np.asarray(v, dtype="float32").reshape(1, -1) for v in rowvecs])
    with _LOCK:
        if len(_SCOPE_CACHE) > 16:
            _SCOPE_CACHE.clear()
        _SCOPE_CACHE[ck] = out
    return out


def _ensure_embeddings(vocab: ScopeVocab):
    """Vectors for every example row: the synthetic questions arrive pre-embedded from
    ingest; rows without one (trace frames, on-the-fly sources) are embedded once here."""
    if vocab.embeddings is not None:
        return vocab.embeddings
    rowvecs = vocab.__dict__.get("_rowvecs")
    if not rowvecs or len(rowvecs) != len(vocab.examples):
        return None
    import numpy as np
    missing = [i for i, v in enumerate(rowvecs) if v is None]
    if missing:
        from ingestion.m3_encoder import encode_dense
        new = np.asarray(encode_dense([vocab.examples[i]["question"] for i in missing]), dtype="float32")
        for i, v in zip(missing, new):
            rowvecs[i] = v
    vocab.embeddings = np.vstack([np.asarray(v, dtype="float32").reshape(1, -1) for v in rowvecs])
    return vocab.embeddings


def nearest_examples(vocab: ScopeVocab, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """k nearest synthetic/trace examples by BGE-M3 cosine; lexical fallback without vectors."""
    if not vocab.examples or k <= 0:
        return []
    try:
        import numpy as np
        emb = _ensure_embeddings(vocab)
        if emb is not None and emb.shape[0] == len(vocab.examples):
            from veda.runtime import _encode_query
            qv = np.asarray(_encode_query(query), dtype="float32").reshape(-1)
            sims = emb @ qv
            order = np.argsort(-sims)[:k]
            return [dict(vocab.examples[i], _sim=float(sims[i])) for i in order]
    except Exception:
        pass
    qt = set(norm_phrase(query).split())
    scored = sorted(vocab.examples,
                    key=lambda ex: -len(qt & set(norm_phrase(ex.get("question", "")).split())))
    return scored[:k]


def reset_cache():
    with _LOCK:
        _CACHE.clear()
        _SCOPE_CACHE.clear()
        _DOC_CACHE.clear()
        _FRONT_CACHE.clear()


# ── the WHOLE scope, for the front door (compound messages) ──────────────────────────
# The per-source vocabulary above is keyed by the scope's semantic model — which holds
# the RELATIONAL model only, so a datalake's cards and a document source have no key in
# it. The front door decides which SOURCE each part of a message goes to before any
# model is loaded, so it sees every source's cards (tables of 2/4/5) plus one card per
# DOCUMENT, built from the chunk headings.
_DOC_CACHE: Dict[Tuple[str, str], Tuple[float, List[Dict[str, Any]]]] = {}
_FRONT_CACHE: Dict[Any, ScopeVocab] = {}
_DOC_TTL_S = 600.0

#: source kinds, from the source profile's source_type / connector
_KIND = {"relational": "sql", "postgres": "sql", "mysql": "sql",
         "datalake": "tabular", "csv_lake": "tabular", "parquet": "tabular",
         "document": "rag", "filesystem": "rag"}

_HEAD_NOISE = re.compile(r"\b(table of contents|contents|index|important notes?|option \d+|"
                         r"you must( not)?|eligibility|exceptions|process|entitlements|"
                         r"using personal social media at work)\b", re.I)


def _clean_heading(h: str) -> str:
    h = re.sub(r"[*_#`]+", "", str(h or "")).strip()
    h = re.sub(r"\s*:+\s*$", "", h).strip()
    h = re.sub(r"\.{3,}\s*\d*\s*$", "", h).strip()
    return re.sub(r"\s+", " ", h)


def headings_of(texts: List[str]) -> List[str]:
    """Section headings from chunk texts. A chunk's first line is its heading breadcrumb
    ("**HANDBOOK** > **LEAVE POLICY** > **SICK LEAVES (SLS):**:") when the chunker kept
    one; a short first line ("Fee Schedule:") is a heading on its own. Order-preserving,
    de-duplicated, table-of-contents lines dropped."""
    out: List[str] = []
    for t in texts:
        first = str(t or "").split("\n", 1)[0]
        if " > " in first or first.rstrip().endswith(":"):
            parts = [_clean_heading(p) for p in first.split(" > ")]
        else:
            continue
        for p in parts:
            if not p or len(p) > 70 or _HEAD_NOISE.search(p) or not re.search(r"[A-Za-z]{3}", p):
                continue
            if p not in out:
                out.append(p)
    return out


def _topic_phrases(sections: List[str]) -> List[str]:
    """Lower-cased phrases a user would type for each section: the heading itself, the
    heading without its parenthesised acronym, and the acronym ("WORK FROM HOME (WFH)" →
    'work from home', 'wfh')."""
    out: List[str] = []
    for s in sections:
        low = s.lower()
        bits = [low, re.sub(r"\s*\([^)]*\)", "", low).strip()]
        bits += [a.strip().lower() for a in re.findall(r"\(([^)]{2,12})\)", s)]
        for sep in (" / ", " & ", " and "):
            if sep in low:
                bits += [x.strip() for x in re.sub(r"\s*\([^)]*\)", "", low).split(sep)]
        for b in bits:
            b = re.sub(r"[^a-z0-9 ]+", " ", b).strip()
            b = re.sub(r"\s+", " ", b)
            if len(b) >= 3 and b not in out:
                out.append(b)
    return out


def document_card(doc_name: str, texts: List[str], source_id: str) -> Dict[str, Any]:
    """One card per document: title, sections, topic phrases."""
    secs = headings_of(texts)
    title = re.sub(r"\.[a-z0-9]{2,5}$", "", str(doc_name), flags=re.I)
    title = re.sub(r"[_]+", " ", title).strip()
    # the top breadcrumb names the document better than its file name
    top = secs[0] if secs and texts and " > " in str(texts[0]).split("\n", 1)[0] else None
    business = (top.title() if top and len(top.split()) <= 5 else title)
    body = " ".join(str(t or "") for t in texts[:3])
    if not secs:
        # a one-chunk note/readme: its first sentence is the only topic signal it has
        secs = [s for s in [_clean_heading(re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0])[:70]] if s]
    return {"doc_name": doc_name, "title": business, "file_title": title,
            "sections": secs[:80], "topics": _topic_phrases(secs[:80])[:160],
            "source_id": str(source_id), "chunks": len(texts)}


def load_document_cards(source_id, tenant: str = "default") -> List[Dict[str, Any]]:
    """Document cards for one document source, from its chunk headings (veda_engine
    doc_chunks, reached through the internal connection — never Django's). Cached for
    _DOC_TTL_S; any failure → []."""
    import time as _time
    key = (str(source_id), str(tenant))
    with _LOCK:
        hit = _DOC_CACHE.get(key)
    if hit and (_time.time() - hit[0]) < _DOC_TTL_S:
        return hit[1]
    cards: List[Dict[str, Any]] = []
    try:
        from ingestion.db_abstraction import get_internal_connection, release_internal_connection
        conn = get_internal_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_name, text FROM doc_chunks WHERE source_id = %s "
                            "ORDER BY doc_name, chunk_index", [str(source_id)])
                rows = cur.fetchall()
        finally:
            release_internal_connection(conn)
        by_doc: Dict[str, List[str]] = {}
        for dn, tx in rows:
            by_doc.setdefault(str(dn), []).append(str(tx or ""))
        cards = [document_card(dn, txs, str(source_id)) for dn, txs in by_doc.items()]
    except Exception:
        cards = []
    with _LOCK:
        _DOC_CACHE[key] = (_time.time(), cards)
    return cards


def source_kind(profile: Optional[Dict[str, Any]]) -> str:
    p = profile or {}
    for k in ("source_type", "connector_type", "dialect"):
        v = str(p.get(k) or "").strip().lower()
        if v in _KIND:
            return _KIND[v]
    return "sql"


def front_door_vocab(source_ids: List[str], tenant: str = "default",
                     profiles: Optional[Dict[str, Any]] = None) -> ScopeVocab:
    """Every in-scope source's entity cards (from their published artifacts) + one card
    per document of every document source. Keys: the bare table name, or
    `src{ID}.{table}` when two sources share it. `doc_cards` / `source_kind` ride on the
    ScopeVocab. No semantic model needed."""
    profiles = profiles or {}
    sids = [str(s) for s in source_ids]
    ck = (tuple(sids), str(tenant),
          tuple(source_kind(profiles.get(s)) for s in sids),
          tuple((lambda p: os.path.getmtime(p) if p else -1.0)(
              _artifact(__import__("ingestion.vocabulary", fromlist=["x"]).CARDS_ARTIFACT, s, tenant))
                for s in sids))
    with _LOCK:
        hit = _FRONT_CACHE.get(ck)
    if hit is not None:
        return hit
    out = ScopeVocab()
    kinds: Dict[str, str] = {}
    per: Dict[str, SourceVocab] = {}
    for sid in sids:
        prof = profiles.get(sid) or {}
        if not prof.get("source_type"):
            # no profile (in-process callers, CLI): the routing card states the kind
            card = _read(_artifact("veda_routing_card.json", sid, tenant) or "") or {}
            prof = {**prof, "source_type": {"document": "document", "datalake": "datalake"}.get(
                str(card.get("kind") or "").lower(), "relational")}
        kinds[sid] = source_kind(prof)
        if kinds[sid] == "rag":
            continue
        per[sid] = load_source(sid, tenant)
        out.built[sid] = per[sid].built
    seen: Dict[str, int] = {}
    for sv in per.values():
        for t in sv.cards:
            seen[t] = seen.get(t, 0) + 1
    for sid, sv in per.items():
        remap = {}
        for t, c in sv.cards.items():
            k = t if seen.get(t, 0) == 1 else f"src{sid}.{t}"
            remap[t] = k
            out.cards[k] = {**c, "table": k, "_bare_table": t, "_source_id": sid}
            out.source_of[k] = sid
        for key, v in sv.value_glossary.items():
            t, _, col = key.partition(".")
            if t in remap:
                out.value_glossary[f"{remap[t]}.{col}"] = v
        for key, v in sv.measure_glossary.items():
            t, _, col = key.partition(".")
            if t in remap:
                out.measure_glossary[f"{remap[t]}.{col}"] = v
        for q in sv.questions:
            if q.get("table") in remap:
                out.examples.append(dict(q, table=remap[q["table"]], _source_id=sid))
    docs: List[Dict[str, Any]] = []
    for sid in sids:
        if kinds[sid] == "rag":
            docs.extend(load_document_cards(sid, tenant))
    out.__dict__["doc_cards"] = docs
    out.__dict__["source_kind"] = kinds
    with _LOCK:
        if len(_FRONT_CACHE) > 16:
            _FRONT_CACHE.clear()
        _FRONT_CACHE[ck] = out
    return out


def doc_cards_of(vocab: ScopeVocab) -> List[Dict[str, Any]]:
    return list(vocab.__dict__.get("doc_cards") or [])


def source_kinds_of(vocab: ScopeVocab) -> Dict[str, str]:
    return dict(vocab.__dict__.get("source_kind") or {})
