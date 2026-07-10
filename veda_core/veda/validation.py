"""VEDA · L6 — value grounding, qualifier-completeness gate, AST validate + parameterize."""
import os, re, sys, time, json, logging, threading
from veda.runtime import _pg


def validate_and_parameterize(sql, allowed_tables, allowed_columns,
                              join_constraints=None, fanout_guard=None):
    """AST-validate the generated SQL and rewrite literals into bound parameters.

    Enforces (enterprise/security):
      • single, read-only SELECT (reject any DML/DDL at the AST level)
      • every TABLE referenced actually exists  → blocks invented tables
      • every COLUMN referenced actually exists  → blocks invented columns
      • all filter literals become %s parameters → no value interpolation
      • ON-integrity (multi-table): the LLM didn't alter the deterministic join —
        every planned join-key pair + polymorphic predicate column must be present.
      • fan-out safety: reject COUNT/SUM/AVG over a column on the PARENT (1) side of
        a 1:N/N:M join — it double-counts (the "valid SQL, wrong number" failure).

    join_constraints (optional): {"key_pairs": [frozenset({colA,colB}), ...],
                                  "predicate_cols": {colname, ...}}
    fanout_guard   (optional): {"parent_aliases": {alias, ...}}  (aliases of parent-side tables)
    Returns (param_sql, params, error). On any violation returns (None, None, reason).
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:
        return None, None, f"unparseable SQL: {e}"
    if tree is None or not isinstance(tree, exp.Select):
        return None, None, "not a single SELECT statement"

    # read-only: reject any write/DDL node anywhere in the tree
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
                 exp.Create, exp.Command, exp.TruncateTable, exp.Merge)
    for node in tree.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, forbidden):
            return None, None, f"forbidden operation: {type(n).__name__}"

    # hallucination guard — tables. CTE names defined in the statement itself
    # (WITH agg_0 AS …) are legitimate table references, not hallucinations.
    allowed_t = {t.lower() for t in allowed_tables}
    for cte in (getattr(tree, "ctes", None) or []):
        if cte.alias:
            allowed_t.add(cte.alias.lower())
    used_t = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    bad_t = used_t - allowed_t
    if bad_t:
        return None, None, f"references unknown table(s): {sorted(bad_t)}"

    # hallucination guard — columns (allow * and aggregates over real columns).
    # SELECT aliases (e.g. COUNT(*) AS annotation_count) are valid references in
    # ORDER BY / HAVING, so add them to the allowed set — they're not hallucinations.
    allowed_c = {c.lower() for c in allowed_columns}
    allowed_c |= {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    used_c = {c.name.lower() for c in tree.find_all(exp.Column) if c.name and c.name != "*"}
    bad_c = used_c - allowed_c
    if bad_c:
        return None, None, f"references unknown column(s): {sorted(bad_c)}"

    # ON-integrity: verify the LLM kept the deterministic joins exactly.
    if join_constraints:
        eq_pairs, eq_qualified = set(), set()
        for eq in tree.find_all(exp.EQ):
            l, r = eq.this, eq.expression
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                eq_pairs.add(frozenset({l.name.lower(), r.name.lower()}))
                if l.table and r.table:
                    eq_qualified.add(frozenset({f"{l.table.lower()}.{l.name.lower()}",
                                                f"{r.table.lower()}.{r.name.lower()}"}))
        for pair in join_constraints.get("key_pairs", []):
            if frozenset({c.lower() for c in pair}) not in eq_pairs:
                return None, None, f"join altered: missing join key {sorted(pair)}"
        # Alias-qualified pass: name pairs collide when two joins share column
        # names (both ON ..."id"); qualified pairs pin each join to its exact
        # occurrence aliases, so swapping t1/t2 in a same-table-twice join or
        # re-pointing an ON clause is caught even when the names still match.
        for qpair in join_constraints.get("qualified_pairs", []):
            if frozenset({c.lower() for c in qpair}) not in eq_qualified:
                return None, None, f"join altered: missing aliased join key {sorted(qpair)}"
        for pcol in join_constraints.get("predicate_cols", set()):
            if pcol.lower() not in used_c:
                return None, None, f"join altered: missing polymorphic predicate on {pcol}"

    # Author-agnostic graph guards: every JOIN key must be a REAL FK edge, and the
    # query must be connected (no cartesian). Runs for ALL SQL — the deterministic
    # head's own shapes (joins / EXISTS / pre-agg CTE / same-table-twice) all pass
    # (verified); this closes the wrong-join + cartesian classes for plan-less SQL
    # (fast-path today, LLM-IR fallback later). Also supplies parent-side aliases from
    # graph cardinality when there's no plan-based fan-out guard.
    try:
        from config import GRAPH_GUARD_ENABLED
    except Exception:
        GRAPH_GUARD_ENABLED = False
    if GRAPH_GUARD_ENABLED:
        try:
            from veda.graph_guard import (verify_joins_against_graph, check_connectivity,
                                          fanout_parent_aliases)
            okj, rj = verify_joins_against_graph(sql)
            if not okj:
                return None, None, rj
            okc, rc = check_connectivity(sql)
            if not okc:
                return None, None, rc
            if not (fanout_guard and fanout_guard.get("parent_aliases")):
                _pa = fanout_parent_aliases(sql)
                if _pa:
                    fanout_guard = dict(fanout_guard or {})
                    fanout_guard["parent_aliases"] = _pa
        except Exception:
            pass

    # Fan-out safety: COUNT/SUM/AVG over a PARENT-side column across a 1:N/N:M join
    # double-counts (e.g. COUNT(counterparty.alias_name) after joining annotations
    # counts annotations, not aliases). MIN/MAX are duplication-safe; COUNT(*) and
    # DISTINCT are fine; child-side columns are the correct grain. All structural —
    # no schema specifics.
    if fanout_guard and (fanout_guard.get("parent_aliases") or fanout_guard.get("parent_only_cols")):
        parent_aliases   = {a.lower() for a in fanout_guard.get("parent_aliases", set())}
        parent_only_cols = {c.lower() for c in fanout_guard.get("parent_only_cols", set())}
        affected = (exp.Count, exp.Sum, exp.Avg)
        for agg in tree.find_all(*affected):
            if isinstance(agg.args.get("this"), exp.Distinct):
                continue  # COUNT(DISTINCT …) removes the fan-out
            for col in agg.find_all(exp.Column):
                ct = (col.table or "").lower()
                # alias-qualified parent column, OR an unqualified column whose name is
                # owned exclusively by a parent table (LLM omitted the alias prefix).
                if ct in parent_aliases or (not ct and col.name.lower() in parent_only_cols):
                    label = f"{col.table}.{col.name}" if col.table else col.name
                    return None, None, (
                        f"fan-out risk: {type(agg).__name__.upper()}({label}) "
                        f"aggregates a one-side column across a one-to-many join "
                        f"(double-counts) — use DISTINCT or pre-aggregate")

    # Parameterize every literal EXCEPT the LIMIT count. Use ordered sentinels so
    # the params list matches the %s order in the *rendered* SQL (a plain append
    # during transform would mis-order them).
    marks = {}

    def _to_param(node):
        # Keep LIMIT counts and INTERVAL constants inline — they're structural,
        # not user-injected values, and INTERVAL literals can't be cleanly bound.
        if isinstance(node, exp.Literal) and not node.find_ancestor(exp.Limit, exp.Interval):
            key = f"__P{len(marks)}__"
            marks[key] = (node.name if node.is_string
                          else (int(node.name) if node.name.isdigit() else float(node.name)))
            return exp.Literal.string(key)   # sentinel string literal
        return node

    # identify=True double-quotes every identifier so reserved-word tables/columns
    # (group, user, order, …) are emitted safely instead of being parsed by Postgres
    # as keywords (the "syntax error at or near group/order" + bogus "column does not
    # exist" failures all trace to unquoted reserved words).
    rendered = tree.transform(_to_param).sql(dialect="postgres", identify=True)
    params = []

    def _repl(m):
        params.append(marks[m.group(1)])
        return "%s"

    param_sql = re.sub(r"'(__P\d+__)'", _repl, rendered)
    # Safety: a leftover sentinel means a literal couldn't be cleanly parameterized
    # (e.g. embedded in a composite literal) — never execute that.
    if "__P" in param_sql:
        return None, None, "could not safely parameterize a literal"
    if "limit" not in param_sql.lower():
        param_sql += " LIMIT 100"
    return param_sql, params, None


def _gate_strip():
    """The query LANGUAGE layer (closed linguistic classes) from config — NOT schema
    vocabulary. Tokens here are 'how the question is phrased' and are ignored by the
    gate; everything else is CONTENT the user is asking for and must appear in the SQL.
    Schema descriptors (mapping/config/assignment) are deliberately NOT here — they're
    handled data-derived (SQL-accounting + routing), never by a stoplist."""
    from config import QUERY_GRAMMAR, QUERY_LANGUAGE
    s = set()
    for ops in QUERY_GRAMMAR.values():
        for w in ops:
            s.update(w.split())
    for cls in QUERY_LANGUAGE.values():
        s.update(cls)
    # Conversational function words (pronouns / interrogatives / polite requests) — query
    # LANGUAGE, never schema vocabulary. Without these, filler in "can YOU find X" false-
    # refuses when the word coincidentally exists as a data VALUE ("you" as a payer name).
    s.update({"you", "your", "yours", "we", "us", "our", "ours", "i", "me", "my", "mine",
              "he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs",
              "who", "whom", "whose", "can", "could", "would", "should", "please", "find",
              "show", "give", "tell", "get", "see", "want", "need", "pull", "provide"})
    return s


_GATE_STRIP = None


def _is_grounded_filter_value(token, tables_in_sql, sm):
    """Decide whether an unaccounted query token is a REAL dropped filter or just filler.

    A token is a dropped FILTER if it's an actual value in a DIMENSION/IDENTIFIER column
    of ANY table in the schema — NOT only the tables this SQL happens to query. Scoping the
    check to the queried tables was a FALSE-NEGATIVE that passed silent wrong answers: when
    the head anchors on the WRONG table (e.g. role_permissions for "roles assigned to
    ekaansh and permissions added last month"), a dropped cross-entity value — 'ekaansh'
    lives in user.username, a table the SQL never touched — looked like filler and the gate
    approved a mangled answer. Completeness is measured against the WHOLE utterance, so the
    value is checked against the WHOLE schema. IDENTIFIER columns are checked first (proper
    nouns like usernames live there) so the dangerous dropped-value case short-circuits fast.
    The DATA decides (live value lookup); there is NO word list.

    Returns True  → real filter value (keep refusing — dangerous silent drop)
            False → confirmed not a value in any categorical/identifier column anywhere
    On any error / inability to assess → True (safe: preserve the existing refusal)."""
    if not sm:
        return True
    import time as _time
    cols_meta = sm.get("columns", {})
    in_sql = tables_in_sql or set()
    # Only DIMENSION / IDENTIFIER columns can hold a filterable value. Order them so the
    # ones that hold HUMAN-READABLE values (usernames, names, codes — where dropped proper
    # nouns like 'ekaansh' live) are checked FIRST: queried-table label columns, then any
    # label column, then the rest. Numeric IDENTIFIER ids rarely match a word token, so they
    # come last. This makes the dangerous dropped-value case short-circuit fast.
    def _labelish(key):
        cn = key.split(".", 1)[1].lower()
        return any(t in cn for t in ("name", "user", "email", "title", "label", "code", "type"))
    cand = [k for k, m in cols_meta.items()
            if (m or {}).get("analytics_role") in ("DIMENSION", "IDENTIFIER")]
    targets = sorted(cand, key=lambda k: (
        k.split(".", 1)[0] not in in_sql,             # queried tables first
        not _labelish(k),                             # label-bearing columns first
    ))
    if not targets:
        return True                                   # nothing to check against → safe refuse
    conn = None
    try:
        conn = _pg()
        conn.set_session(readonly=True, autocommit=True)
        deadline = _time.monotonic() + 6.0            # wall-clock budget across all lookups
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 3000")
            for key in targets:
                if _time.monotonic() > deadline:
                    return True                       # ran out of budget → can't confirm filler → safe refuse
                tbl, col = key.split(".", 1)
                cur.execute(f'SELECT 1 FROM "{tbl}" WHERE lower("{col}"::text) = lower(%s) LIMIT 1',
                            (token,))
                if cur.fetchone() is not None:
                    return True                       # token IS a real value somewhere → dropped filter
        return False                                  # confirmed: not a value in any categorical/id col
    except Exception:
        return True                                   # safe: preserve refusal on any error
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _names_entity_column(token, tables_in_sql, sm):
    """True if `token` names a COLUMN (or alias) of a queried table. An unaccounted token
    that names a real column the SQL didn't project is a DROPPED ATTRIBUTE, not filler —
    "role codes" → role.role_code, "email" → user.email. (Pure in-memory; no DB.) This is
    what keeps the value-grounding filler relaxation from silently dropping requested
    columns; filler like 'database'/'system' matches no column and stays ignored."""
    if not tables_in_sql or not sm:
        return False
    cols_meta = sm.get("columns", {})
    for key, m in cols_meta.items():
        tbl, _, cn = key.partition(".")
        if tbl not in tables_in_sql:
            continue
        ctoks = [w for w in re.findall(r"[a-z]+", cn.lower()) if len(w) > 2]
        if token in ctoks or (len(token) >= 4 and any(len(w) >= 4 and (token in w or w in token)
                                                      for w in ctoks)):
            return True
        for al in (m or {}).get("aliases", []) or []:
            if token in [w for w in re.findall(r"[a-z]+", al.lower()) if len(w) > 2]:
                return True
    return False


_DS_SYN_CACHE = {"v": None}


def _domain_synonyms() -> dict:
    """Authoritative domain_synonyms (business phrase → [col_id]) from DOMAIN_SYNONYMS_FILE
    — the file retrieval + scripts/enrich_synonyms.py maintain — independent of sm's
    possibly-stale embedded copy. Cached per process; empty dict when absent."""
    if _DS_SYN_CACHE["v"] is None:
        try:
            import json as _json
            import os as _os
            from config import DOMAIN_SYNONYMS_FILE as _p
            _DS_SYN_CACHE["v"] = _json.load(open(_p)) if _os.path.exists(_p) else {}
        except Exception:
            _DS_SYN_CACHE["v"] = {}
    return _DS_SYN_CACHE["v"]


def qualifier_completeness(query, sql, sm=None):
    """Unified correctness gate (all paths): every CONTENT token the user named must
    appear somewhere in the generated SQL — as a table, column, string literal, or
    SELECT alias (or as a descriptor in a referenced table's business purpose). A named
    qualifier that's nowhere in the SQL (e.g. 'itadmin' in "permissions for itadmin
    role", 'abhijit' in "incidents assigned to abhijit") is a DROPPED FILTER: the SQL
    would answer a broader question. We refuse rather than silently mislead. Substring
    matching (≥4 chars) absorbs morphology (flagged↔flag, active↔is_active); the table
    business-purpose absorbs NL descriptors (workflow↔state) — neither admits a dropped
    proper-noun value, which is the dangerous case. Returns (ok, missing|None)."""
    global _GATE_STRIP
    if _GATE_STRIP is None:
        _GATE_STRIP = _gate_strip()
    from retrieval.query_enrichment import _singularize
    content = {_singularize(w) for w in re.findall(r"[a-z]+", query.lower())
               if len(w) > 2 and w not in _GATE_STRIP and _singularize(w) not in _GATE_STRIP}
    if not content:
        return True, None
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return True, None                  # don't block on parse issues; the AST validator handles those

    def _idtoks(name):
        return {_singularize(w) for w in re.findall(r"[a-z]+", (name or "").lower()) if len(w) > 2}

    sqltoks = set()
    tables_in_sql = set()
    for t in tree.find_all(exp.Table):
        sqltoks |= _idtoks(t.name)
        tables_in_sql.add(t.name)
    for c in tree.find_all(exp.Column):
        sqltoks |= _idtoks(c.name)
    for a in tree.find_all(exp.Alias):
        if a.alias:
            sqltoks |= _idtoks(a.alias)
    for lit in tree.find_all(exp.Literal):
        if lit.is_string:
            sqltoks |= _idtoks(lit.name)
    # Data-derived synonyms: the NL vocabulary of the entities actually queried — table
    # business-purpose + the aliases/business_role of the columns the SQL references.
    # This admits descriptors/synonyms ("workflow"↔state, "mapping"↔a junction) from the
    # semantic model itself, NOT from a hardcoded list, while still never admitting a
    # dropped proper-noun value (those appear in no metadata).
    if sm:
        cols_meta = sm.get("columns", {})
        for tname in tables_in_sql:
            tmeta = sm.get("tables", {}).get(tname, {})
            sqltoks |= _idtoks(tmeta.get("business_purpose", ""))
        # Domain-synonym vocabulary for the QUERIED tables: a business term the ingestion
        # mapped to a column/entity of a table in this SQL ("property"→assets_asset) is
        # ACCOUNTED — otherwise the gate false-refuses the synonym as a "dropped" column
        # (e.g. "property" substring-matches assets_asset.corner_property, so a join to the
        # asset entity is wrongly seen as dropping "property").
        for _phrase, _cids in _domain_synonyms().items():
            _cl = _cids if isinstance(_cids, list) else [_cids]
            if any(str(_c).split(".")[0] in tables_in_sql for _c in _cl):
                sqltoks |= _idtoks(_phrase)
        referenced_cols = {c.name for c in tree.find_all(exp.Column) if c.name}
        for tname in tables_in_sql:
            for cname in referenced_cols:
                cm = cols_meta.get(f"{tname}.{cname}")
                if not cm:
                    continue
                sqltoks |= _idtoks(cm.get("business_role", ""))
                for al in (cm.get("aliases") or []):
                    sqltoks |= _idtoks(al)

    def _accounted(ct):
        if ct in sqltoks:
            return True
        return len(ct) >= 4 and any(len(s) >= 4 and (ct in s or s in ct) for s in sqltoks)

    unaccounted = sorted(ct for ct in content if not _accounted(ct))
    # An unaccounted token is a real gap only if the user named either (a) a COLUMN/alias
    # of a queried table the SQL didn't project (a DROPPED ATTRIBUTE — "role codes" →
    # role.role_code, "email" → user.email), or (b) a real value in a DIMENSION/IDENTIFIER
    # column (a DROPPED FILTER). Tokens that name NEITHER ("in the database/system/
    # platform", or any noun a user invents) are filler — ignoring them avoids a false
    # refuse. Schema + data decide, no word lists.
    missing = [ct for ct in unaccounted
               if _names_entity_column(ct, tables_in_sql, sm)
               or _is_grounded_filter_value(ct, tables_in_sql, sm)]
    return (not missing), (missing[0] if missing else None)


def value_grounding(sql, resolve_table, cols_meta, skip_values=()):
    """Value validation: reject when a filter compares a CATEGORICAL or IDENTIFIER column
    to a literal that doesn't exist in that column's data — whether the operator is
    `=` (returns nothing), `<>`/`!=`, or `IN`/`NOT IN`. A fabricated value is fabricated
    regardless of operator: `incident_status = 'failed_review'` returns nothing, while
    `incident_status <> 'Closed'` (when 'Closed' is not a value of the column) is VACUOUS —
    it silently returns every row, a wrong answer disguised as success.

    Only checks low-cardinality / identifier columns (where an absent value means a
    fabricated mapping); skips free-text/measure columns (an absent value there is just an
    empty result) and our own deterministic predicate values. Predicates wrapped in
    `lower(...)` are skipped (the arbiter's grounded, case-insensitive SQL — already
    value-checked at grounding time; re-checking its lowercased literal vs raw-cased data
    would false-reject). No LLM, no lists.

    resolve_table(column_exp) -> table name (or None).  Returns (ok, (col, val) | None).
    """
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return True, None

    # Collect (column_exp, literal) pairs from =, <>/!=, and IN/NOT IN lists. Only plain
    # column references are taken — a `lower(col)`/cast wrapper yields no Column here, so
    # the arbiter's own `lower(col) = 'x'` / `IN (...)` predicates are intentionally skipped.
    def _col_lit(node):
        col = node.this if isinstance(node.this, exp.Column) else (
            node.expression if isinstance(node.expression, exp.Column) else None)
        lit = node.expression if isinstance(node.expression, exp.Literal) else (
            node.this if isinstance(node.this, exp.Literal) else None)
        return col, lit

    pairs = []
    for node in tree.find_all(exp.EQ, exp.NEQ):
        col, lit = _col_lit(node)
        if col is not None and lit is not None and lit.is_string:
            pairs.append((col, lit.name))
    for inn in tree.find_all(exp.In):                 # catches IN and the In inside NOT IN
        col = inn.this if isinstance(inn.this, exp.Column) else None
        if col is None:
            continue
        for e in (inn.expressions or []):
            if isinstance(e, exp.Literal) and e.is_string:
                pairs.append((col, e.name))

    conn = None
    try:
        for col, val in pairs:
            if val in skip_values:
                continue                      # our deterministic polymorphic predicate
            tbl = resolve_table(col)
            if not tbl:
                continue
            meta = cols_meta.get(f"{tbl}.{col.name}", {}) or {}
            role = meta.get("analytics_role", "")
            # only validate where an absent value implies a fabricated mapping
            if role not in ("DIMENSION", "IDENTIFIER"):
                continue
            if conn is None:
                conn = _pg()
            with conn.cursor() as cur:
                cur.execute(f'SELECT 1 FROM "{tbl}" WHERE "{col.name}"::text = %s LIMIT 1', (val,))
                if cur.fetchone() is None:
                    return False, (f"{tbl}.{col.name}", val)
        return True, None
    finally:
        if conn:
            conn.close()
