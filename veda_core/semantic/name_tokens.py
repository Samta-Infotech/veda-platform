# =============================================================================
# semantic/name_tokens.py
# VEDA — schema-vocabulary table-name tokenizer.
#
# Problem: anchor/routing lexical signals tokenize table names by splitting on
# "_" with whole-token matching. Schemas whose table names concatenate entity
# words WITHOUT separators (Django's app_modelname convention:
# accounts_paymenttransaction, assets_assetverificationdocumenttype) produce
# single opaque tokens, so a query word ("payments") can never match the table
# that answers it — the lexical channel is structurally dead for the whole
# schema and anchoring degrades to normalized retrieval noise.
#
# Fix: segment opaque name tokens using the SCHEMA'S OWN VOCABULARY — the
# underscore-separated words the same source already exposes in its column
# names (payment_id, transaction_type), table names, ingestion-generated
# column aliases, and domain-synonym phrases. "paymenttransaction" splits into
# [payment, transaction] because BOTH words exist in this source's schema —
# not because of any built-in word list. A token the vocabulary cannot fully
# cover is returned unchanged (failure-safe: never worse than today).
#
# Genericity contract: NO hardcoded vocabulary, NO language assumptions beyond
# the [a-z]+ word convention the rest of the pipeline already uses. Everything
# is derived from the ACTIVE scope's semantic model, cached per (tenant,
# source-set) scope exactly like the retrieval engines, so multi-source
# deployments segment each source with its own vocabulary.
# =============================================================================

import re

try:
    from retrieval.query_enrichment import _singularize
except Exception:  # pragma: no cover
    def _singularize(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _flags():
    try:
        from config import NAME_SUBWORD_SPLIT_ENABLED, NAME_SUBWORD_MIN_PIECE
        return NAME_SUBWORD_SPLIT_ENABLED, NAME_SUBWORD_MIN_PIECE
    except Exception:
        return True, 3


def _scope_key():
    """(tenant, frozen source-set) — same shape as veda.runtime._engine_scope, computed
    locally so semantic/ does not import veda/ (layering). No request context (dev CLI)
    → a single global scope backed by SEMANTIC_MODEL_FILE."""
    try:
        from veda_core import context
        ctx = context.try_current()
        if ctx is not None:
            return (str(ctx.tenant), frozenset(int(s) for s in ctx.source_ids))
    except Exception:
        pass
    return ("_global", frozenset())


def _load_sm():
    """Scoped semantic model when no caller-supplied model is available."""
    from veda.runtime import _load_scoped_sm
    return _load_scoped_sm()


def _build_vocab(sm) -> frozenset:
    """Word vocabulary of one scope's schema: every [a-z]+ run (len ≥ 3) found in
    column names, ingestion-generated column aliases and domain-synonym phrases —
    plus singularized forms. Purely data-derived.

    Deliberately EXCLUDES raw table-name tokens: in those, entity words may be fused
    ("paymenttransaction"), so admitting them would (a) make every opaque token its
    own vocabulary word — stopping the very segmentation this module exists for —
    and (b) offer fused junk as split pieces. Column names, aliases and synonym
    phrases are separated by construction, so each of their runs is a GENUINE word;
    a token found here is protected from false splitting ("transaction" never
    becomes trans+action), and Django FK columns (`<model>_id`) put nearly every
    entity word into this vocabulary anyway."""
    words = set()

    def _add(text):
        for w in re.findall(r"[a-z]+", (text or "").lower()):
            if len(w) >= 3:
                words.add(w)
                words.add(_singularize(w))

    cols = sm.get("columns", {}) or {}
    for ck, cm in cols.items():
        _add(ck.rsplit(".", 1)[1] if "." in ck else ck)
        for al in ((cm or {}).get("aliases") or []) + ((cm or {}).get("base_aliases") or []):
            _add(al)
    for phrase in (sm.get("domain_synonyms", {}) or {}):
        _add(phrase)
    return frozenset(w for w in words if w)


# scope key -> {"vocab": frozenset, "ncols": int, "seg": {token: tuple}, "tab": {name: frozenset}}
_SCOPES = {}
_SCOPES_MAX = 8


def _scope_entry(sm=None):
    key = _scope_key()
    ent = _SCOPES.get(key)
    ncols = len((sm or {}).get("columns", {}) or {}) if sm is not None else None
    # weak version check: same scope but a different/refreshed model → rebuild
    if ent is not None and (ncols is None or ent["ncols"] == ncols):
        return ent
    if sm is None:
        sm = _load_sm()
    vocab = _build_vocab(sm)
    ent = {"vocab": vocab, "ncols": len(sm.get("columns", {}) or {}), "seg": {}, "tab": {}}
    _SCOPES[key] = ent
    while len(_SCOPES) > _SCOPES_MAX:
        _SCOPES.pop(next(iter(_SCOPES)))
    return ent


def _segment(token, vocab, min_piece):
    """Full-coverage segmentation of `token` into vocabulary words, or None.

    Longest-piece-first recursion with memo, preferring the FEWEST pieces (so a word
    that is itself in the vocabulary is never split further, and 'paymenttransaction'
    prefers [payment, transaction] over any 3-way cover). Every piece must be a
    vocabulary word (raw or singularized) of length ≥ min_piece; anything less than a
    full cover fails → caller keeps the original token."""
    n = len(token)
    memo = {}

    def solve(i):
        if i == n:
            return ()
        if i in memo:
            return memo[i]
        best = None
        for j in range(n, i + min_piece - 1, -1):
            piece = token[i:j]
            if piece in vocab or _singularize(piece) in vocab:
                rest = solve(j)
                if rest is not None:
                    cand = (piece,) + rest
                    if best is None or len(cand) < len(best):
                        best = cand
        memo[i] = best
        return best

    return solve(0)


def segment_token(token, sm=None):
    """One lowercase token → tuple of vocabulary words covering it, else (token,)."""
    enabled, min_piece = _flags()
    token = (token or "").lower()
    if not token:
        return ()
    if not enabled:
        return (token,)
    try:
        ent = _scope_entry(sm)
    except Exception:
        return (token,)
    seg = ent["seg"].get(token)
    if seg is None:
        vocab = ent["vocab"]
        if token in vocab or _singularize(token) in vocab or len(token) < 2 * min_piece:
            seg = (token,)
        else:
            seg = _segment(token, vocab, min_piece) or (token,)
        ent["seg"][token] = seg
    return seg


def token_table_idf(sm=None):
    """token → normalized IDF in (0, 1] over the scope's TABLE token sets
    (segmentation-aware). A token carried by few tables ("payment") is strong
    lexical evidence for its table; one shared by many ("value", "type", an app
    prefix like "assets") is weak. Data-derived from the schema itself — the
    generic-word discount needs no stopword list. Cached per scope."""
    try:
        ent = _scope_entry(sm)
    except Exception:
        return {}
    idf = ent.get("idf")
    if idf is None:
        import math
        if sm is None:
            try:
                sm = _load_sm()
            except Exception:
                ent["idf"] = {}
                return {}
        tabs = list(sm.get("tables", {}) or {})
        df = {}
        for t in tabs:
            for w in table_tokens(t, sm):
                df[w] = df.get(w, 0) + 1
        n = max(len(tabs), 1)
        denom = math.log(1.0 + n) or 1.0
        idf = {w: math.log(1.0 + n / d) / denom for w, d in df.items()}
        ent["idf"] = idf
    return idf


def table_tokens(table_name, sm=None):
    """Singularized word tokens of a table name, with opaque concatenations segmented
    by the scope's schema vocabulary. Drop-in replacement for the
    `{_singularize(w) for w in t.split("_") if len(w) > 2}` idiom (same length/case
    conventions); flag off or any failure → exactly that legacy behavior."""
    name = (table_name or "").lower()
    enabled, _ = _flags()
    if not enabled:
        return {_singularize(w) for w in re.findall(r"[a-z]+", name) if len(w) > 2}
    try:
        ent = _scope_entry(sm)
        cached = ent["tab"].get(name)
        if cached is not None:
            return set(cached)
        out = set()
        for w in re.findall(r"[a-z]+", name):
            if len(w) > 2:
                for piece in segment_token(w, sm):
                    if len(piece) > 2:
                        out.add(_singularize(piece))
        ent["tab"][name] = frozenset(out)
        return out
    except Exception:
        return {_singularize(w) for w in re.findall(r"[a-z]+", name) if len(w) > 2}
