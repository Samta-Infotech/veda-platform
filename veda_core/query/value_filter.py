# query/value_filter.py
# VEDA — Value-aware filter column retrieval (Gap 4)
#
# Finds columns whose SAMPLED VALUES contain a query token and returns them
# as candidates to force-include AFTER the reranker cutoff.
#
# Why this is needed:
#   Relevance rerankers find OUTPUT columns ("decision reason").
#   Filter columns ("escalated" → incident_status) score low on relevance
#   because the query term is a VALUE of that column, not its name.
#   A relevance-only cutoff drops them. This step adds them back deterministically.

import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional

from typing import Dict

from config import (
    VALUE_FILTER_MIN_TOKEN_LEN,
    VALUE_FILTER_SCOPE_TO_CANDIDATES,
    VALUE_FILTER_ALLOW_CROSS_TABLE,
    VALUE_FILTER_DETERMINISTIC,
    VALUE_FILTER_VALUE_ONLY,
    VALUE_FILTER_SKIP_BOOLEAN,
)
from ingestion.vector_store import RetrievalResult


def _build_cand_entity_toks(candidate_table_ids: set, value_store: dict) -> set:
    """Return word tokens from the names of candidate tables.

    Used to detect entity-reference tokens: if query token "roles" maps to
    table "role" in the candidate set, it is an entity reference — not a
    filter value — and should be excluded from value-filter injection.
    """
    if not candidate_table_ids:
        return set()
    toks: set = set()
    seen: set = set()
    for sc in value_store.values():
        if sc.table_id in candidate_table_ids and sc.table_name not in seen:
            seen.add(sc.table_name)
            for w in (sc.table_name or "").lower().replace("_", " ").split():
                toks.add(w)
    return toks


def _rank_filter_cols(results: List[RetrievalResult], tokens: List[str]) -> List[RetrievalResult]:
    """Rank in-scope value-filter columns: CATEGORY/low-cardinality first, then name overlap."""
    _type_priority = {"CATEGORY": 0, "IDENTIFIER": 1, "METRIC": 2, "MONETARY": 3, "FREE_TEXT": 4, "TEMPORAL": 5}
    tok_set = set(tokens)

    def _key(r: RetrievalResult):
        type_rank = _type_priority.get(getattr(r, "semantic_type", "FREE_TEXT") or "FREE_TEXT", 99)
        name_parts = set((r.col_name or "").lower().split("_"))
        name_overlap = -len(name_parts & tok_set)
        return (type_rank, name_overlap)

    return sorted(results, key=_key)


def _query_value_tokens(query: str, max_ngram: int = 3) -> List[str]:
    """Tokens to look up against sampled values.

    Single words (len >= VALUE_FILTER_MIN_TOKEN_LEN) PLUS consecutive word n-grams
    (2..max_ngram) joined by a space. Multi-word values like "IT Admin" are stored
    in the value index as "it admin"; single-word tokenisation can never match them
    ("admin" isn't a stored value, "it" is below the length floor), so n-grams are
    required to recover them. The index holds only real sampled values, so an n-gram
    matches only when that exact phrase is a stored value — inherently selective."""
    words = [w.lower() for w in re.findall(r"\w+", query)]
    out: List[str] = []
    seen: set = set()

    def _add(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    for w in words:
        if len(w) >= VALUE_FILTER_MIN_TOKEN_LEN:
            _add(w)
    for n in range(2, max_ngram + 1):
        for i in range(len(words) - n + 1):
            _add(" ".join(words[i:i + n]))
    return out


def _subject_entity_tokens(query: str) -> set:
    """Tokens that name the query's DOMINANT entity (the subject noun).

    A token that names the subject entity is an entity reference, never a filter value —
    even when the same word also happens to be a categorical VALUE elsewhere (the schema
    has both a `role` table and object_type='Role'). Scoped to the SUBJECT only, not every
    entity: in "role-type change requests" the subject is change_request, so "role" stays a
    legitimate value. Subject = the top match_concepts result(s); ties at the top included."""
    try:
        from semantic import registry as reg
        hits = reg.match_concepts(reg.query_tokens(query))
        if not hits:
            return set()
        top = hits[0][1]
        toks: set = set()
        for c, score in hits:
            if score == top:
                toks |= set(c.get("match_tokens", []))
        return toks
    except Exception:
        return set()


def _names_subject(token: str, subject_toks: set) -> bool:
    if not subject_toks:
        return False
    sing = token[:-1] if token.endswith("s") and len(token) > 3 else token
    return token in subject_toks or sing in subject_toks


def find_value_filter_columns(
    query:               str,
    source_ids:          Optional[List[str]] = None,
    candidate_table_ids: Optional[set]       = None,
) -> List[RetrievalResult]:
    """
    Return columns whose sampled values contain a query token.
    Tries in-memory _VALUE_INDEX first (O(1), populated during ingestion).
    Falls back to a pgvector LIKE query if the in-memory store is empty.
    Results are returned with similarity=1.0 to signal exact value-match origin.

    When candidate_table_ids is provided and VALUE_FILTER_SCOPE_TO_CANDIDATES is True,
    only returns columns from tables already in the candidate set — prevents the filter
    from pointing at an unjoinable table that L4 will skip.
    """
    tokens = _query_value_tokens(query)
    # Drop tokens that name the subject entity: "role" in "roles assigned to X" is the
    # entity, not a value — without this it collides with object_type='Role' and pulls
    # retrieval to change_request/workflow (the singular-vs-plural wrong-table bug).
    _subj = _subject_entity_tokens(query)
    tokens = [t for t in tokens if not _names_subject(t, _subj)]
    if not tokens:
        return []

    # Fast path: in-memory inverted index
    results: List[RetrievalResult] = []
    try:
        from ingestion.value_sampler import _VALUE_INDEX, _VALUE_STORE
        if _VALUE_INDEX:
            _cet = _build_cand_entity_toks(candidate_table_ids or set(), _VALUE_STORE)
            results = _lookup_in_memory(tokens, source_ids, _VALUE_INDEX, _VALUE_STORE, _cet)
    except Exception:
        pass

    # Slow fallback: pgvector LIKE query
    if not results:
        results = _lookup_pgvector(tokens, source_ids)

    if not results:
        return []

    # Scope to candidate tables so the filter lands on a table L4 can join
    if candidate_table_ids and VALUE_FILTER_SCOPE_TO_CANDIDATES:
        in_scope = [r for r in results if r.table_id in candidate_table_ids]
        if in_scope:
            return _rank_filter_cols(in_scope, tokens)
        if not VALUE_FILTER_ALLOW_CROSS_TABLE:
            return []

    return _rank_filter_cols(results, tokens)


def _lookup_in_memory(
    tokens:           List[str],
    source_ids:       Optional[List[str]],
    index:            dict,
    store:            dict,
    cand_entity_toks: set = None,
) -> List[RetrievalResult]:
    seen: set = set()
    results: List[RetrievalResult] = []
    for token in tokens:
        _tok_sing = token[:-1] if token.endswith("s") and len(token) > 3 else token
        for col_id in index.get(token, []):
            sc = store.get(col_id)
            if sc is None:
                continue
            # Entity-name skip: token "incident" matching object_type.value="Incident"
            # is an entity-reference, not a filter-value. Without this, find_value_filter_columns
            # returns object_type for every "show incidents" query, and Rule 14 in the SLM
            # prompt then fires: "query word 'incident' matches value 'Incident' → add WHERE".
            # Extended: also skip when the token (or its singular) matches ANY candidate
            # table name — e.g. "roles" names the "role" table, "permission" names the
            # "permission" table, so they must not be used as filter values.
            _table_toks = set((sc.table_name or "").lower().replace("_", " ").split())
            if (token in _table_toks
                    or (cand_entity_toks and (
                        token in cand_entity_toks
                        or _tok_sing in cand_entity_toks))):
                continue
            # VALUE_FILTER_SKIP_BOOLEAN: skip is_*/has_* cols for non-boolean tokens
            if VALUE_FILTER_SKIP_BOOLEAN and _is_boolean_col_name(sc.col_name):
                _BOOL_QUERY_TOKENS = {"true", "false", "yes", "no"}
                if token not in _BOOL_QUERY_TOKENS:
                    continue
            # VALUE_FILTER_VALUE_ONLY: skip when token matches col-name token, not a value
            if VALUE_FILTER_VALUE_ONLY:
                _col_toks = set((sc.col_name or "").lower().replace("_", " ").split())
                if token in _col_toks:
                    continue
            if col_id in seen:
                continue
            seen.add(col_id)
            results.append(RetrievalResult(
                col_id        = sc.col_id,
                col_name      = sc.col_name,
                table_id      = sc.table_id,
                table_name    = sc.table_name,
                semantic_type = sc.semantic_type,
                similarity    = 1.0,
                source_id     = getattr(sc, "source_id", ""),
            ))
    return results


def _is_boolean_column(sc) -> bool:
    """True when all sampled values for the column are boolean-like literals."""
    _BOOL_NORMS = {"true", "false", "t", "f", "1", "0", "yes", "no"}
    return bool(sc.values) and all(v in _BOOL_NORMS for v in sc.values)


def _is_boolean_col_name(col_name: str) -> bool:
    """True when the column name has an is_/has_/can_/should_ prefix — semantically boolean."""
    _BOOL_PREFIXES = ("is_", "has_", "can_", "should_", "was_", "will_")
    lower = (col_name or "").lower()
    return any(lower.startswith(p) for p in _BOOL_PREFIXES)


def build_value_filters(
    query:               str,
    source_ids:          Optional[List[str]] = None,
    candidate_table_ids: Optional[set]       = None,
) -> List[Dict]:
    """
    Return [{col_id, operator:'EQ', value:<exact DB value>}] for in-scope
    CATEGORY/low-card columns whose values match a query token (case-insensitive).

    Emits the EXACT stored casing, not the query token, so 'escalated' → 'Escalated'.
    Skips boolean columns unless the query token is literally true/false/yes/no.
    Only called when VALUE_FILTER_DETERMINISTIC=True.

    Multiple values on the same column are collected into a single IN condition
    when more than one token matches.
    """
    if not VALUE_FILTER_DETERMINISTIC:
        return []

    tokens = _query_value_tokens(query)
    _subj = _subject_entity_tokens(query)
    tokens = [t for t in tokens if not _names_subject(t, _subj)]
    if not tokens:
        return []

    _BOOL_QUERY_TOKENS = {"true", "false", "yes", "no"}

    try:
        from ingestion.value_sampler import _VALUE_INDEX, _VALUE_STORE
        if not _VALUE_INDEX:
            return []
    except Exception:
        return []

    # Entity tokens from candidate table names — used to skip tokens that name
    # a candidate table (e.g. "roles" → "role" table, "permission" → "permission"
    # table).  These are entity-reference tokens, not filter values.
    _cet = _build_cand_entity_toks(candidate_table_ids or set(), _VALUE_STORE)

    # col_id → list of exact-cased matched values (for IN support, Latent E)
    matched: Dict[str, List] = {}

    for token in tokens:
        _tok_sing = token[:-1] if token.endswith("s") and len(token) > 3 else token
        for col_id in _VALUE_INDEX.get(token, []):
            sc = _VALUE_STORE.get(col_id)
            if sc is None:
                continue

            # Scope: only include columns from candidate tables (or FK-adjacent tables
            # that were added to the expanded candidate set in retrieval_select.py).
            if candidate_table_ids and VALUE_FILTER_SCOPE_TO_CANDIDATES:
                if sc.table_id not in candidate_table_ids:
                    if not VALUE_FILTER_ALLOW_CROSS_TABLE:
                        continue

            # Skip if the query token is the column's own table name — that makes it
            # an entity-reference token ("incidents" → table "incident"), not a filter value.
            # Extended: also skip when token (or singular) names ANY candidate table,
            # e.g. "roles" → "role" table in candidates, "permission" → "permission" table.
            _table_toks = set((sc.table_name or "").lower().replace("_", " ").split())
            if (token in _table_toks
                    or token in _cet
                    or _tok_sing in _cet):
                continue

            # VALUE_FILTER_SKIP_BOOLEAN: skip is_*/has_* cols for non-boolean tokens
            if VALUE_FILTER_SKIP_BOOLEAN and _is_boolean_col_name(sc.col_name):
                if token not in _BOOL_QUERY_TOKENS:
                    continue
            # VALUE_FILTER_VALUE_ONLY: skip when token appears in col-name tokens
            if VALUE_FILTER_VALUE_ONLY:
                _col_toks = set((sc.col_name or "").lower().replace("_", " ").split())
                if token in _col_toks:
                    continue

            is_bool = _is_boolean_column(sc)

            if is_bool:
                if token not in _BOOL_QUERY_TOKENS:
                    continue
                # Boolean: emit typed bool, not a string
                exact_value = token in {"true", "yes"}
            else:
                # Find exact DB casing: sc.values[i] (normalised) → sc.raw_values[i]
                exact_value = None
                for rv, nv in zip(sc.raw_values, sc.values):
                    if nv == token:
                        exact_value = rv
                        break
                if exact_value is None:
                    if VALUE_FILTER_VALUE_ONLY:
                        continue   # no exact value match found — skip this column
                    exact_value = token   # fallback: use token as-is

            if col_id not in matched:
                matched[col_id] = []
            if exact_value not in matched[col_id]:
                matched[col_id].append(exact_value)

    if not matched:
        return []

    conditions: List[Dict] = []
    for col_id, values in matched.items():
        if len(values) == 1:
            conditions.append({"col_id": col_id, "operator": "EQ", "value": values[0]})
        else:
            # Multiple distinct values matched → use IN (Latent E)
            conditions.append({"col_id": col_id, "operator": "IN", "value": values})

    return conditions


def _lookup_pgvector(
    tokens:     List[str],
    source_ids: Optional[List[str]],
) -> List[RetrievalResult]:
    from config import COLUMN_VALUES_TABLE_NAME, VEDA_INTERNAL_DB
    try:
        import psycopg2
    except ImportError:
        return []

    cfg = VEDA_INTERNAL_DB
    try:
        conn = psycopg2.connect(
            host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
            user=cfg["user"], password=cfg["password"],
        )
    except Exception:
        return []

    results: List[RetrievalResult] = []
    seen: set = set()
    try:
        with conn.cursor() as cur:
            for token in tokens:
                params: list = [f"%{token}%"]
                src_clause = ""
                if source_ids:
                    placeholders = ",".join(["%s"] * len(source_ids))
                    src_clause = f" AND source_id IN ({placeholders})"
                    params.extend(source_ids)
                cur.execute(
                    f"SELECT DISTINCT col_id, col_name, table_id, table_name, source_id "
                    f"FROM {COLUMN_VALUES_TABLE_NAME} "
                    f"WHERE lower(value_raw) LIKE %s{src_clause} "
                    f"LIMIT 20",
                    params,
                )
                for row in cur.fetchall():
                    col_id = row[0]
                    if col_id in seen:
                        continue
                    seen.add(col_id)
                    results.append(RetrievalResult(
                        col_id        = col_id,
                        col_name      = row[1],
                        table_id      = row[2] or "",
                        table_name    = row[3],
                        semantic_type = "CATEGORY",
                        similarity    = 1.0,
                        source_id     = row[4] or "",
                    ))
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return results
