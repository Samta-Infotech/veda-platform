"""VEDA · Value-vs-Column Arbitration Layer.

Runs BEFORE retrieval expansion and SQL generation. Classifies each query span as:

    SCHEMA_REF     a table / column identifier        ("incident", "assigned_to")
    VALUE          a categorical filter value         ("critical", "open")
    NEGATED_VALUE  a negated categorical value        ("unresolved" -> status != resolved)
    ENTITY         a free-text / identifier value      ("Raj", "ServiceNow")
    UNKNOWN        none of the above (filler, verbs)

Why this exists
---------------
Retrieval and SQL generation often treat business adjectives ("critical", "open",
"active") as *columns* (e.g. open -> incident.open_date) instead of *values*. This
layer flips that: if a token matches a sampled categorical value, it is grounded as a
VALUE and must NOT drive column retrieval.

Grounded by construction — DATA decides
---------------------------------------
A token is a VALUE only if it appears in the `column_values` store (the value_sampler's
sampled distinct values). There is NO hand-maintained value list. The store's
`semantic_type` separates VALUE (CATEGORY) from ENTITY (FREE_TEXT / IDENTIFIER).
Negation is data-grounded too: a negated token is recognised only when stripping its
negator yields a base form that EXISTS as a sampled value (`unresolved` -> `resolved`).

This is a classification step, not a subsystem. The production lookup reuses the
existing `column_values` table; the resolver/pipeline consume the result. The `lookup`
is injected, so the layer is unit-testable with no DB and no Ollama.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

try:
    from config import VALUE_EXPANSION_MIN_TOKEN_LEN as _MIN_LEN
except Exception:
    _MIN_LEN = 3

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:                                    # pragma: no cover - logging optional
    import logging
    logger = logging.getLogger(__name__)

# Closed-class grammatical negators — function words, NOT a domain value list.
_NEGATORS: Set[str] = {"not", "no", "without", "never", "non"}
# Morphological negation prefixes. Only ACCEPTED when stripping yields a base form that
# is a real sampled value (the data gate makes this prefix set safe — "incident" does
# not become a negation because "cident" is not a value).
_NEG_PREFIXES: Tuple[str, ...] = ("un", "in", "im", "ir", "il", "non", "dis")

_CATEGORY_TYPES = {"CATEGORY"}
_ENTITY_TYPES = {"FREE_TEXT", "IDENTIFIER"}

# A typed value lookup: token -> [(table_name, col_name, semantic_type, value_raw), ...]
TypedLookup = Callable[[str], List[Tuple[str, str, str, str]]]


@dataclass
class TokenClass:
    """Classification of a single query span."""
    span:          str
    kind:          str                      # SCHEMA_REF|VALUE|NEGATED_VALUE|ENTITY|UNKNOWN
    table:         Optional[str] = None
    column:        Optional[str] = None     # col_name on `table`
    value:         Optional[str] = None     # value_raw — original casing (display only)
    value_norm:    Optional[str] = None     # lowercase normalised form — used for the
                                            # case-insensitive comparison (High/high/HIGH)
    op:            str = "="                # "=" for VALUE/ENTITY, "!=" for NEGATED_VALUE
    semantic_type: Optional[str] = None
    candidates:    List[Tuple[str, str, str, str]] = field(default_factory=list)
    reason:        str = ""

    @property
    def qualified_column(self) -> Optional[str]:
        return f"{self.table}.{self.column}" if self.table and self.column else None


@dataclass
class ArbitrationResult:
    query:  str
    tokens: List[TokenClass]

    def _by(self, kind: str) -> List[TokenClass]:
        return [t for t in self.tokens if t.kind == kind]

    @property
    def value_filters(self) -> List[TokenClass]:
        """VALUE + NEGATED_VALUE — the grounded categorical filters."""
        return [t for t in self.tokens if t.kind in ("VALUE", "NEGATED_VALUE")]

    @property
    def entity_mentions(self) -> List[TokenClass]:
        return self._by("ENTITY")

    @property
    def schema_refs(self) -> List[TokenClass]:
        return self._by("SCHEMA_REF")

    @property
    def value_tokens(self) -> Set[str]:
        """Surface tokens classified as VALUE/NEGATED_VALUE — retrieval must NOT treat
        these as column-name candidates."""
        out: Set[str] = set()
        for t in self.value_filters:
            out.update(t.span.split())
        return out

    def explain(self) -> str:
        """Human-readable debug trace, one block per classified (non-UNKNOWN) span."""
        blocks: List[str] = []
        for t in self.tokens:
            if t.kind == "UNKNOWN":
                continue
            lines = [f"Token: {t.span}", f"Classification: {t.kind}"]
            if t.kind == "NEGATED_VALUE":
                lines.append(f"Target: {t.qualified_column}")
            elif t.qualified_column:
                lines.append(f"Column: {t.qualified_column}")
            lines.append(f"Reason: {t.reason}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Core classifier — pure, DB-agnostic (lookup injected)
# ---------------------------------------------------------------------------
def _norm(span: str) -> str:
    return re.sub(r"\s+", " ", span.lower().strip())


def _pick_value_column(
    cands: List[Tuple[str, str, str, str]],
) -> Optional[Tuple[str, str, str, str]]:
    """Prefer a CATEGORY column (a controlled vocabulary) for a VALUE classification;
    fall back to the first candidate. Deterministic ordering."""
    cats = sorted([c for c in cands if c[2] in _CATEGORY_TYPES])
    if cats:
        return cats[0]
    return sorted(cands)[0] if cands else None


def _classify_match(
    span: str,
    cands: List[Tuple[str, str, str, str]],
) -> TokenClass:
    """A span that exact-matches sampled value(s): VALUE (categorical) or ENTITY (name)."""
    has_category = any(c[2] in _CATEGORY_TYPES for c in cands)
    if has_category:
        tbl, col, st, raw = _pick_value_column(cands)
        return TokenClass(span=span, kind="VALUE", table=tbl, column=col, value=raw,
                          value_norm=_norm(span), op="=", semantic_type=st,
                          candidates=cands, reason="matched sampled value set")
    # Only free-text / identifier matches -> an entity mention (name), not a category value
    tbl, col, st, raw = sorted(cands)[0]
    return TokenClass(span=span, kind="ENTITY", table=tbl, column=col, value=raw,
                      value_norm=_norm(span), op="=", semantic_type=st,
                      candidates=cands, reason="matched sampled name/identifier value")


def _try_negation(
    base: str,
    lookup: TypedLookup,
    span: str,
    via: str,
) -> Optional[TokenClass]:
    """If `base` is a real sampled value, emit a NEGATED_VALUE on its (preferred) column."""
    if len(base) < _MIN_LEN:
        return None
    cands = lookup(base) or []
    if not cands:
        return None
    pick = _pick_value_column(cands)
    if not pick:
        return None
    tbl, col, st, raw = pick
    return TokenClass(span=span, kind="NEGATED_VALUE", table=tbl, column=col, value=raw,
                      value_norm=_norm(base), op="!=", semantic_type=st, candidates=cands,
                      reason=f"{base} exists in sampled values ({via})")


def arbitrate(
    query: str,
    value_lookup: TypedLookup,
    schema_terms: Optional[Set[str]] = None,
    min_len: int = _MIN_LEN,
) -> ArbitrationResult:
    """Classify every span of `query`.

    Parameters
    ----------
    query        : raw user query.
    value_lookup : token -> [(table, col, semantic_type, value_raw)] over column_values.
    schema_terms : optional set of literal schema identifiers (table/column name parts)
                   used only for the lowest-priority SCHEMA_REF tag. Values always win.
    """
    words = re.findall(r"[a-z0-9]+", query.lower())
    n = len(words)
    consumed = [False] * n
    out: List[TokenClass] = []
    schema_terms = schema_terms or set()

    # SUBJECT-entity skip: a token naming the query's SUBJECT concept ("user" in "show user
    # email") is an entity reference, NOT a categorical filter value — even when it
    # coincidentally exists as a sampled value (change_request.object_type='User'). Reuses
    # value_filter's SUBJECT-scoped detection (the SAME policy, one source), so a NON-subject
    # concept word stays a value ("role" in "role-type change requests" → object_type='Role').
    # All-concepts skip would wrongly drop that — subject-scope is what keeps it correct.
    try:
        from query.value_filter import _subject_entity_tokens, _names_subject
        _subj_toks = _subject_entity_tokens(query)
    except Exception:
        _subj_toks, _names_subject = set(), (lambda *_a, **_k: False)

    def _is_subject_ref(span: str) -> bool:
        return any(_names_subject(w, _subj_toks) for w in span.split())

    # ---- Pass 1: bigrams (explicit negation "not closed"; multiword values "high priority")
    i = 0
    while i < n - 1:
        if consumed[i] or consumed[i + 1]:
            i += 1
            continue
        w0, w1 = words[i], words[i + 1]
        # explicit negation: NEGATOR + base
        if w0 in _NEGATORS and len(w1) >= min_len:
            tc = _try_negation(w1, value_lookup, span=f"{w0} {w1}", via="explicit negator")
            if tc:
                out.append(tc)
                consumed[i] = consumed[i + 1] = True
                i += 2
                continue
        # multiword categorical value / entity ("high priority")
        bigram = f"{w0} {w1}"
        cands = value_lookup(bigram) or []
        if cands:
            if _is_subject_ref(bigram):
                out.append(TokenClass(span=bigram, kind="SCHEMA_REF",
                                      reason="names the query subject entity (not a filter value)"))
            else:
                out.append(_classify_match(bigram, cands))
            consumed[i] = consumed[i + 1] = True
            i += 2
            continue
        i += 1

    # ---- Pass 2: unigrams
    for j, w in enumerate(words):
        if consumed[j]:
            continue
        if len(w) < min_len:
            out.append(TokenClass(span=w, kind="UNKNOWN", reason="below min token length"))
            continue
        # (a) direct value — checked BEFORE schema so a token that is both a value and a
        #     column name is grounded as a VALUE ("open" -> status value, not open_date).
        cands = value_lookup(w) or []
        if cands:
            if _is_subject_ref(w):
                out.append(TokenClass(span=w, kind="SCHEMA_REF",
                                      reason="names the query subject entity (not a filter value)"))
            else:
                out.append(_classify_match(w, cands))
            continue
        # (b) morphological negation ("unresolved" -> resolved, "inactive" -> active)
        neg = None
        for pre in _NEG_PREFIXES:
            if w.startswith(pre) and len(w) - len(pre) >= min_len:
                neg = _try_negation(w[len(pre):], value_lookup, span=w,
                                    via=f"prefix '{pre}-'")
                if neg:
                    break
        if neg:
            out.append(neg)
            continue
        # (c) schema reference (lowest priority)
        if w in schema_terms:
            out.append(TokenClass(span=w, kind="SCHEMA_REF", reason="matches schema identifier"))
            continue
        out.append(TokenClass(span=w, kind="UNKNOWN", reason="no value/schema match"))

    result = ArbitrationResult(query=query, tokens=out)
    if logger.isEnabledFor(10):                       # DEBUG
        for blk in result.explain().split("\n\n"):
            if blk:
                logger.debug("[value_arbiter]\n%s", blk)
    return result


# ---------------------------------------------------------------------------
# Production helpers — reuse existing structures (column_values, semantic model)
# ---------------------------------------------------------------------------
def column_values_typed_lookup(conn_fn) -> TypedLookup:
    """Typed lookup backed by the existing `column_values` store. Reuses the same table
    as value_resolver.column_values_lookup, additionally returning `semantic_type` so the
    arbiter can separate VALUE (CATEGORY) from ENTITY (FREE_TEXT/IDENTIFIER).

    EXACT match on `value_norm` only — never LIKE / fuzzy / embeddings."""
    try:
        from config import COLUMN_VALUES_TABLE_NAME as _TBL
    except Exception:
        _TBL = "column_values"

    def _lookup(token: str) -> List[Tuple[str, str, str, str]]:
        token = _norm(token)
        if len(token) < _MIN_LEN:
            return []
        conn = None
        try:
            conn = conn_fn()
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT table_name, col_name, semantic_type, value_raw "
                    f"FROM {_TBL} WHERE value_norm = %s LIMIT 16",
                    (token,),
                )
                return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    return _lookup


def build_schema_terms(semantic_model: dict) -> Set[str]:
    """Literal schema identifiers (table names + column-name word parts) for the
    lowest-priority SCHEMA_REF tag. Aliases are intentionally excluded — they are handled
    by retrieval, and including them would make almost every token a SCHEMA_REF and
    weaken the value-over-column flip."""
    terms: Set[str] = set()
    for tname in (semantic_model.get("tables") or {}):
        for part in str(tname).split("_"):
            if len(part) >= _MIN_LEN:
                terms.add(part)
    for key, col in (semantic_model.get("columns") or {}).items():
        cname = col.get("col_name", "")
        for part in str(cname).split("_"):
            if len(part) >= _MIN_LEN:
                terms.add(part)
    return terms


def _anchor_column(t: TokenClass, anchor_table: str) -> Optional[str]:
    """Column on `anchor_table` for this value token. The same value can be sampled on
    several tables (e.g. 'high' on incident.priority AND signal_rules.priority); the
    globally-preferred pick may be off-anchor, so re-scan candidates and prefer a
    CATEGORY column on the anchor. Returns None if the value lives on no anchor column."""
    if t.table == anchor_table and t.column:
        return t.column
    cats = sorted(c for c in t.candidates if c[0] == anchor_table and c[2] in _CATEGORY_TYPES)
    anyc = sorted(c for c in t.candidates if c[0] == anchor_table)
    pick = cats or anyc
    return pick[0][1] if pick else None


def anchor_filters(result: ArbitrationResult, anchor_table: str) -> List[dict]:
    """Direct (same-table) filters for `anchor_table`, including negation.

    Includes VALUE/NEGATED_VALUE *and* on-anchor ENTITY matches: an exact value match on
    a column OF THE ANCHOR TABLE is a direct filter regardless of the CATEGORY-vs-FREE_TEXT
    label — which is unreliable (e.g. incident.priority is mistyped FREE_TEXT though it's a
    3-value enum, so "high" classifies as ENTITY). On-anchor *placement* is the real
    signal. Cross-table values/entities ("assigned to Raj" → users) have no anchor column
    here, so _anchor_column drops them and value_resolver handles the FK path.
        {"column", "op", "value_norm", "value", "kind": "value"|"negated_value", "reason"}
    """
    out: List[dict] = []
    _anchor_l = (anchor_table or "").lower()
    for t in result.value_filters + result.entity_mentions:
        # Entity-name skip: a grounded value equal to the anchor's own entity/table name
        # is the ENTITY NOUN ("user" in "user created last month"), not a filter — even
        # though it happens to exist in a column (user.last_name='User'). Drop it. Mirrors
        # the same skip in value_resolver.resolve_value_filter.
        if str(t.value_norm).lower() == _anchor_l:
            continue
        col = _anchor_column(t, anchor_table)
        if not col:
            continue
        out.append({"column": col, "op": t.op,
                    "value_norm": t.value_norm, "value": t.value,
                    "kind": "negated_value" if t.kind == "NEGATED_VALUE" else "value",
                    "reason": t.reason})
    return out


def _sql_lit(s: str) -> str:
    """Single-quote-escape a grounded literal (value_norm is already lowercase a-z0-9/space
    from the sampler, so this is belt-and-braces, not the primary safety boundary)."""
    return str(s).replace("'", "''")


def where_clause(filters: List[dict], alias: Optional[str] = None) -> str:
    """Build a case-INSENSITIVE grounded predicate from anchor_filters output, with
    correct boolean composition:

      • multiple values on the SAME column  → OR  → ``lower(col) IN ('a','b')``
        (negated values on the same column  → AND → ``lower(col) NOT IN ('x','y')``,
         which is the De Morgan form of ``!= 'x' AND != 'y'``)
      • DIFFERENT columns                    → AND

    So "high or critical incidents" becomes ``lower(priority) IN ('critical','high')``
    instead of the impossible ``='high' AND ='critical'``. The column side is lowered and
    compared to `value_norm` (the sampler's lowercase form), so High/high/HIGH all match —
    categorical case normalisation, NOT fuzzy matching. Centralised here so every caller
    (pipeline, tests) composes identically. Deterministic: column order follows first
    appearance; values within a column are sorted."""
    by_col: "OrderedDict[str, dict]" = OrderedDict()
    for f in filters:
        col = f["column"]
        if col not in by_col:
            by_col[col] = {"pos": [], "neg": []}
        bucket = "neg" if f["op"] == "!=" else "pos"
        v = f["value_norm"]
        if v not in by_col[col][bucket]:
            by_col[col][bucket].append(v)

    def _eq(lc, vals, in_op, eq_op):
        if len(vals) == 1:
            return f"{lc} {eq_op} '{_sql_lit(vals[0])}'"
        joined = ", ".join(f"'{_sql_lit(v)}'" for v in vals)
        return f"{lc} {in_op} ({joined})"

    _pfx = f'{alias}.' if alias else ''      # table alias for JOIN contexts (answer-entity)
    col_clauses: List[str] = []
    for col, buckets in by_col.items():
        lc = f'lower({_pfx}"{col}"::text)'
        sub: List[str] = []
        if buckets["pos"]:
            sub.append(_eq(lc, sorted(buckets["pos"]), "IN", "="))
        if buckets["neg"]:
            sub.append(_eq(lc, sorted(buckets["neg"]), "NOT IN", "!="))
        col_clauses.append(" AND ".join(sub))
    return " AND ".join(col_clauses)
