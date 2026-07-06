"""VEDA · generic value resolver.

ONE mechanism for value filters, no hardcoded column-name vocabulary:

    value token  →  value index (DATA)  →  (table, column) that CONTAINS it
                 →  FK graph             →  place the filter relative to the anchor

The column is chosen by the DATA (a value index — which column's sampled values
contain the token), never by guessing column names. Placement is by the FK graph:
  • value lives on the anchor table        → direct  : col = value
  • value lives in an FK-reachable table    → subquery: anchor_col IN (SELECT … WHERE col = value)
  • value lives nowhere reachable / ambiguous → None  (caller falls through; never guess)

`lookup(token) -> [(table, column, exact_value), …]` is INJECTED, so the data source
is pluggable: the value sampler's `column_values` index (default), a live probe, etc.
Grounded by construction — the value is one the data actually holds.
"""


def resolve_value_filter(anchor, qtoks, graph, lookup, anchor_cols=None):
    """Return a filter descriptor, or None.

    direct   : {"kind":"direct",   "column", "value"}
    subquery : {"kind":"subquery", "anchor_col", "target", "target_col", "filter_col", "value"}
               → WHERE anchor_col IN (SELECT target_col FROM target WHERE filter_col = value)

    anchor_cols (optional, injected): the anchor table's COLUMN NAMES. A query token that
    names an anchor column ("email" → user.email) is a COLUMN REFERENCE to project/show, NOT
    a filter value — even when it coincidentally exists as a value in some FK-reachable table
    (notification.mode='email'). Such tokens are dropped before grounding so they never build
    a spurious cross-table value-filter. EXACT column-name match (or singular) only — so a
    value word like "active" is not skipped just because a column is named "is_active". Same
    family as the entity-name skip below; the schema (anchor's columns) decides, no hardcode."""
    # 0. ANCHOR-COLUMN skip — a token naming an anchor column is a column-ref, not a value.
    if anchor_cols:
        _ac = {str(c).lower() for c in anchor_cols}
        def _names_col(t):
            tl = t.lower()
            sing = tl[:-1] if tl.endswith("s") and len(tl) > 3 else tl
            return tl in _ac or sing in _ac
        qtoks = [t for t in qtoks if not _names_col(t)]

    # 1. DATA decides which (table, column) holds the value — for every query token.
    hits = []
    seen = set()
    # Batch all token lookups into ONE query when the lookup supports it (avoids N
    # sequential DB round-trips); fall back to per-token for any other lookup.
    _batch = getattr(lookup, "batch", None)
    if _batch is not None:
        try:
            found_map = _batch(qtoks) or {}
        except Exception:
            found_map = {}
        per_tok = [(t, found_map.get(t.lower(), [])) for t in qtoks]
    else:
        per_tok = []
        for tok in qtoks:
            try:
                per_tok.append((tok, lookup(tok) or []))
            except Exception:
                per_tok.append((tok, []))
    for _tok, found in per_tok:
        for (tbl, col, val) in found:
            key = (tbl, col)
            if key not in seen:
                seen.add(key)
                hits.append((tbl, col, val))

    # 1a. ENTITY-NAME skip. A grounded value equal to the ANCHOR's own entity/table name
    #     is a polymorphic TYPE DISCRIMINATOR (e.g. change_request.object_type='Role' when
    #     the anchor IS 'role'), not a filter value — the entity noun leaked into value
    #     grounding. Drop such hits. Targeted (anchor-scoped); mirrors the entity-name skip
    #     already in query/value_filter.py.
    _anchor_l = (anchor or "").lower()
    hits = [h for h in hits if str(h[2]).lower() != _anchor_l]

    # 1b. REACHABILITY scope. A value sitting in a table with NO FK path to the anchor
    #     cannot be filtered for THIS query (e.g. a denormalised name copy in an
    #     unrelated table) — it is not a real alternative interpretation, so it must not
    #     count toward ambiguity. Keep only hits on the anchor or its FK neighbours.
    reachable = {anchor}
    for e in graph.get("edges", []):
        if e["source_table"] == anchor:
            reachable.add(e["target_table"])
        if e["target_table"] == anchor:
            reachable.add(e["source_table"])
    hits = [h for h in hits if h[0] in reachable]

    # 2. ENTITY-LEVEL grounding (task #6). Hits come ONLY from EXACT value grounding
    #    (lookup = column_values WHERE lower(value)=token) — never similarity, aliases,
    #    LIKE, fuzzy, or embeddings. A value may exact-match SEVERAL columns of the SAME
    #    entity (user.username AND user.first_name): that is ONE entity, not ambiguity →
    #    resolve, and the caller OR-matches across exactly those columns (no
    #    column-preference guess). The value spanning DIFFERENT tables (customer.name vs
    #    vendor.name) is genuine ambiguity → refuse.
    # ANCHOR-DIRECT wins. When the value sits directly on the anchor table, the same
    # value also living in FK-neighbour tables (object_type='Level 1' on incident AND
    # rfi_questions) is NOT ambiguity for THIS query — the anchor-direct hit is the
    # intended one. Without this, a value common to several reachable tables refuses.
    if anchor in {h[0] for h in hits} and len({h[0] for h in hits}) > 1:
        hits = [h for h in hits if h[0] == anchor]
    if len({h[0] for h in hits}) != 1:
        return None                             # 0 hits, or value spans >1 entity → ambiguous
    tbl = hits[0][0]
    pairs, _seen = [], set()                    # [(col, exact_value)] — exact-grounded only
    for (_t, c, v) in hits:
        if c not in _seen:
            _seen.add(c)
            pairs.append((c, v))
    pairs.sort()                                # deterministic OR ordering (cosmetic only)
    col, val = pairs[0]                         # legacy single-column fields (back-compat)

    # 3. Place it relative to the anchor using the FK graph (no name heuristics).
    if tbl == anchor:
        return {"kind": "direct", "table": tbl, "pairs": pairs,
                "column": col, "value": val}

    # candidate FK edges connecting anchor ↔ tbl. (anchor_col, target_col, fk_col)
    cands = []
    for e in graph.get("edges", []):
        s, t = e["source_table"], e["target_table"]
        sc, tc = e["source_column"], e["target_column"]
        if s == anchor and t == tbl:            # anchor.fk → tbl.pk
            cands.append((sc, tc, sc))
        elif t == anchor and s == tbl:          # tbl.fk → anchor.pk (reverse)
            cands.append((tc, sc, sc))
    if not cands:
        return None                             # value in an unrelated table → can't place

    # When several FKs connect the pair (assigned_to_id AND updated_by_id → user),
    # disambiguate by the relation the QUERY names — match the FK column's tokens to
    # the query ("assigned" → assigned_to_id). Generic (same idea as the join planner);
    # 0 or >1 surviving → ambiguous → None.
    if len(cands) > 1:
        named = [c for c in cands
                 if {p for p in c[2].split("_") if len(p) > 2} & set(qtoks)]
        if len(named) != 1:
            return None
        cands = named
    anchor_col, target_col, _fk = cands[0]
    return {"kind": "subquery", "anchor_col": anchor_col, "target": tbl,
            "target_col": target_col, "pairs": pairs,
            "filter_col": col, "value": val}


# ---------------------------------------------------------------------------
# Default data-driven lookup: the value sampler's column_values index.
# Pluggable — pass any lookup(token)->[(table,column,value)] to resolve_value_filter.
# ---------------------------------------------------------------------------
def _mirror_scope():
    """(source_id, tenant) for the Redis value-mirror key, from the ambient context."""
    try:
        from veda_core.context import try_current
        ctx = try_current()
        if ctx is not None:
            return str(getattr(ctx, "source_id", "")), str(getattr(ctx, "tenant", "default"))
    except Exception:
        pass
    return "", "default"


def _mirror_lookup(token):
    """Q-5: Redis-first value resolution → [(table, col, raw)] or None (Postgres fallback)."""
    try:
        from config import VALUE_MIRROR_ENABLED
        if not VALUE_MIRROR_ENABLED:
            return None
        from ingestion.value_mirror import lookup_value
        sid, tenant = _mirror_scope()
        entries = lookup_value(token.lower(), source_id=sid, tenant=tenant)
        if entries is None:
            return None
        return [(e["table"], e["col"], e["raw"]) for e in entries]
    except Exception:
        return None


def column_values_lookup(conn_fn):
    """Build a lookup backed by the `column_values` store (data-driven; the sampler
    already chose which columns to index, so there's no name/role heuristic here).
    Returns a function lookup(token) -> [(table, column, exact_value)]."""
    def _lookup(token):
        if len(token) < 3:
            return []
        mirror = _mirror_lookup(token)     # Q-5: sub-ms Redis hit before the PG round trip
        if mirror is not None:
            return mirror
        conn = None
        try:
            conn = conn_fn()
            with conn.cursor() as cur:
                # column_values stores value_norm (normalised, lowercase) + value_raw
                # (original casing). Match on value_norm (EXACT — no LIKE/fuzzy); return
                # value_raw as the real filter literal. (Prior code referenced a
                # non-existent `value` column and silently returned [] on every call.)
                cur.execute(
                    "SELECT table_name, col_name, value_raw FROM column_values "
                    "WHERE value_norm = %s LIMIT 8", (token.lower(),))
                return [(r[0], r[1], r[2]) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _batch(tokens):
        """All tokens in ONE query → {value_norm: [(table, column, value_raw), …]}.
        Same exact value_norm match as _lookup; just avoids N sequential round-trips."""
        toks = sorted({t.lower() for t in tokens if t and len(t) >= 3})
        if not toks:
            return {}
        # Q-5: resolve what the Redis mirror covers first; only miss to Postgres.
        out = {}
        remaining = []
        for tok in toks:
            mirror = _mirror_lookup(tok)
            if mirror is not None:
                if mirror:
                    out[tok] = mirror
            else:
                remaining.append(tok)
        if not remaining:
            return out
        toks = remaining
        conn = None
        try:
            conn = conn_fn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value_norm, table_name, col_name, value_raw FROM column_values "
                    "WHERE value_norm = ANY(%s)", (toks,))
                for vnorm, table_name, col_name, value_raw in cur.fetchall():
                    out.setdefault(vnorm, []).append((table_name, col_name, value_raw))
                return out
        except Exception:
            return out
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    _lookup.batch = _batch
    return _lookup
