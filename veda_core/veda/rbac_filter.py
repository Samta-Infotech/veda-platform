"""veda.rbac_filter — Gate 1 (User Story 3, Task 16) engine-side data filtering.

Applies the RBAC data-scope payload (``RequestContext.allowed_resources``, built
Django-side by ``apps.access_management.services.compute_data_scope`` and carried
across the HTTP boundary — see ``veda_core/context.py``) to PER-REQUEST data only.

WHY NOT FILTER THE SHARED RETRIEVAL ENGINE ITSELF
    ``veda.runtime.get_engine()`` builds one ``RetrievalEnginePhase3`` (BM25/signal
    index) PER SCOPE (tenant + source_ids) and caches it in ``_ENGINES`` for reuse
    by every request in that scope — deliberately NOT per user, because building it
    is expensive and the scope's data doesn't change between users. If this module
    filtered the semantic model BEFORE it reached ``get_engine()``, whichever
    request happened to build the engine first would permanently bake that user's
    permissions into the shared index — every other user sharing the scope
    afterward would silently inherit it (or its absence). That is a cross-user
    leak, not a narrowing.

    So the retrieval engine is left untouched: still built once per scope, still
    unfiltered internally. RBAC instead narrows the per-request CANDIDATE LIST
    ``engine.retrieve(...)`` returns (``filter_retrieval_results``), before it
    reaches reranking/SQL generation. The SQL head's semantic model
    (``veda_hybrid.py::_load_semantic_model()``) has no such shared-cache hazard —
    it is read fresh from a small per-scope dict cache on every call, so filtering
    its RETURN VALUE per request (``filter_sm``) is safe.

WHY ``narrow_allowed`` EXISTS TOO — ``filter_retrieval_results`` IS NOT ENOUGH
    A production audit (2026-08-08) found that narrowing the retrieval CANDIDATE
    LIST does not stop a forbidden table/column from reaching generated SQL: the
    deterministic single/multi-table planners, FK/entity-neighbour expansion, and
    anchor-hint salvage all independently consult the RAW, unfiltered ``sm`` (or a
    process-global, RBAC-oblivious FK graph) rather than the filtered candidate
    list — and a FastPath answer or a verified-query CACHE HIT bypasses retrieval
    (and therefore ``filter_retrieval_results``) entirely, since it never calls
    ``get_engine().retrieve()`` at all.

    Rather than auditing and patching every one of those discovery sites (a large,
    open-ended surface across ``veda/pipeline.py``, ``veda/planning.py`` and
    ``veda/routing.py``, plus ``veda_hybrid.py``'s separate Tier-2 paths), Gate 1
    adds ONE centralized check instead: ``narrow_allowed(allowed_tables,
    allowed_columns, sm, ctx)``, called immediately before every
    ``veda.validation.validate_and_parameterize(...)`` call — the ONE place, across
    every SQL-generation path (FastPath, cache replay, single-table, multi-table,
    Tier-2 shared-planner, Tier-2 sql_builder), that already enforces
    ``allowed_tables``/``allowed_columns`` as a hard allowlist against the
    generated SQL's AST ("references unknown table(s)/column(s)" → refused). No
    matter which upstream path proposed a table/column, this narrows the allowlist
    to what RBAC actually permits before that allowlist is enforced — the SQL is
    then REJECTED (not silently rewritten) by ``validate_and_parameterize`` if it
    references anything trimmed out here, the exact same way it already rejects an
    LLM hallucination. This is the "centralized enforcement, avoid scattered
    checks" requirement satisfied literally: one gate, not N discovery-site patches.

All three functions are pure identity (same object/inputs, unchanged) when
``ctx`` is None or ``ctx.allowed_resources`` is None (RBAC off, staff bypass, or
any caller that predates this module) — every existing call site stays
byte-identical until it actually forwards a real data scope.

NO Django import, no ``apps.access_management`` import: this module runs in the
inference tier, which has neither installed (see ``inference/requirements.txt`` /
this package's own "no Django" precedent in ``veda_core/context.py``).
"""
from __future__ import annotations

import dataclasses

_MISSING = object()


def _as_source_id(value):
    """One source id, normalized to ``int`` — the type every comparison in this
    module assumes.

    Every id here is compared against ``open_sources``/``table_index`` keys built
    from ``parse_allowed_resources``, which ints them. A source id that arrives as
    a STRING (an ``sm`` entry's ``_source_id`` published by the assembler, a
    hand-built ``RequestContext``) therefore compares unequal to the very entry
    that permits it, and the table is reported as restricted for a source the
    caller fully owns — a silent false DENIAL, not a leak, but a user-visible one.
    ``filter_doc_chunks`` already coerced its own ids for exactly this reason;
    this is the same coercion, applied to the one path that lacked it.

    Non-numeric input is returned unchanged rather than raising: an unrecognisable
    id then simply matches nothing, which is the fail-closed outcome."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _index(ctx):
    """``ctx.allowed_resources`` unpacked into two cheap lookup structures, built
    once per call (not per candidate/table):

      * ``open_sources``  — source ids with no restriction at all.
      * ``table_index``   — ``{(source_id, real_table_name): frozenset(cols) | None}``
        for restricted sources; ``None`` means "every column of this table"."""
    open_sources = set()
    table_index = {}
    for source_id, (is_open, tables) in ctx.allowed_resources:
        source_id = _as_source_id(source_id)
        if is_open:
            open_sources.add(source_id)
            continue
        for name, cols in tables:
            table_index[(source_id, name)] = frozenset(cols) if cols is not None else None
    return open_sources, table_index


def _real_table_name(table_key: str, source_id) -> str:
    """Undo ``_merge_scoped_sms``'s ``src{ID}.`` qualification (applied only on a
    cross-source table-name collision, see ``veda_core/veda/runtime.py``) to
    recover the real table name — the same name ``compute_data_scope`` enumerated
    against on the Django side, which carries no such qualifier."""
    prefix = f"src{source_id}."
    return table_key[len(prefix):] if table_key.startswith(prefix) else table_key


def _table_source_id(entry: dict, ctx):
    """The owning source for one ``sm['tables']``/``sm['columns']`` entry. A
    merged (multi-source) entry carries ``_source_id`` (``_merge_scoped_sms``); a
    single-source scope's entries don't need one — there is only one member to
    attribute to."""
    sid = entry.get("_source_id")
    if sid is not None:
        return _as_source_id(sid)
    ids = tuple(ctx.source_ids)
    return _as_source_id(ids[0]) if ids else None


def _table_allowed(open_sources, table_index, sid, table_key) -> bool:
    if sid in open_sources:
        return True
    return (sid, _real_table_name(table_key, sid)) in table_index


def _column_allowed(open_sources, table_index, sid, table_key, col_name) -> bool:
    if sid in open_sources:
        return True
    cols = table_index.get((sid, _real_table_name(table_key, sid)), _MISSING)
    if cols is _MISSING:
        return False
    return cols is None or col_name in cols


def restricted_names(sm: dict, ctx) -> dict:
    """Bare table/column names RBAC would remove from ``sm`` for ``ctx`` — the
    same test ``filter_sm`` applies, without building a filtered copy of ``sm``
    itself. ``{"tables": [], "columns": []}`` when there is nothing to restrict.

    For a caller that must not change what the rest of the pipeline sees (e.g.
    ``pipeline.py``'s feedback/explanation path, which needs to know a term
    names something real-but-restricted without narrowing the live ``sm`` every
    other part of ``run_query`` still reasons over) — see ``veda.feedback``.
    """
    if ctx is None or getattr(ctx, "allowed_resources", None) is None:
        return {"tables": [], "columns": []}

    open_sources, table_index = _index(ctx)

    tables = sm.get("tables") or {}
    restricted_tables = [
        table_key.rpartition(".")[2] or table_key
        for table_key, entry in tables.items()
        if not _table_allowed(open_sources, table_index, _table_source_id(entry, ctx), table_key)
    ]

    columns = sm.get("columns") or {}
    restricted_columns = []
    for key, entry in columns.items():
        table_key, _, col_name = key.rpartition(".")
        if not table_key:
            continue
        sid = _table_source_id(entry, ctx)
        if not _column_allowed(open_sources, table_index, sid, table_key, col_name):
            restricted_columns.append(col_name)

    return {"tables": sorted(restricted_tables), "columns": sorted(restricted_columns)}


def filter_sm(sm: dict, ctx) -> dict:
    """A COPY of ``sm`` with ``tables``/``columns``/``retrieval_documents``
    narrowed to ``ctx.allowed_resources`` — or ``sm`` itself, unchanged, when
    there is nothing to restrict. Never mutates the input: ``sm`` is a shared,
    scope-cached object (``_SM`` in ``veda_hybrid.py``).

    Also attaches ``_rbac_restricted`` (bare table/column names removed by this
    call) — never consumed by retrieval/SQL-generation, which only ever see the
    narrowed ``tables``/``columns`` themselves. It exists solely so a downstream
    refusal (``veda.feedback.explain_failure``) can tell "this term matches a
    real but restricted table/column" apart from "this term matches nothing at
    all", and phrase the two differently — see that module for why."""
    if ctx is None or getattr(ctx, "allowed_resources", None) is None:
        return sm

    open_sources, table_index = _index(ctx)

    tables = sm.get("tables") or {}
    filtered_tables = {
        table_key: entry for table_key, entry in tables.items()
        if _table_allowed(open_sources, table_index, _table_source_id(entry, ctx), table_key)
    }

    columns = sm.get("columns") or {}
    filtered_columns = {}
    for key, entry in columns.items():
        # rsplit on the LAST dot, matching the engine's own convention
        # (retrieval_engine_phase3.py::_results_from_tuples) — the table part of a
        # qualified key ("src2.employee.id") itself contains a dot, so splitting
        # on the first dot would cut it in the wrong place.
        table_key, _, col_name = key.rpartition(".")
        if not table_key:  # a key with no dot names nothing addressable at column grain
            continue
        sid = _table_source_id(entry, ctx)
        if _column_allowed(open_sources, table_index, sid, table_key, col_name):
            filtered_columns[key] = entry

    docs = sm.get("retrieval_documents") or {}
    filtered_docs = {}
    for key, doc in docs.items():
        table_key, sep, _col_name = key.rpartition(".")
        if sep and key in filtered_columns:
            filtered_docs[key] = doc
        elif not sep and key in filtered_tables:
            filtered_docs[key] = doc

    return {**sm, "tables": filtered_tables, "columns": filtered_columns,
            "retrieval_documents": filtered_docs,
            "_rbac_restricted": restricted_names(sm, ctx)}


def filter_retrieval_results(results, sm: dict, ctx):
    """``results`` (a list of the engine's ``RetrievalResult``s), narrowed to what
    ``ctx.allowed_resources`` permits — applied to the per-request CANDIDATE LIST
    returned by ``get_engine(sm).retrieve(...)``, never to the shared engine
    itself (see module docstring). ``results`` unchanged when there is nothing to
    restrict, including ``None`` (a failed/empty retrieval)."""
    if ctx is None or getattr(ctx, "allowed_resources", None) is None or not results:
        return results

    open_sources, table_index = _index(ctx)
    tables = sm.get("tables") or {}

    def ok(result) -> bool:
        table_key, _, col_name = result.col_id.rpartition(".")
        if not table_key:
            return False
        entry = tables.get(table_key, {})
        sid = _table_source_id(entry, ctx)
        return _column_allowed(open_sources, table_index, sid, table_key, col_name)

    return [r for r in results if ok(r)]


def narrow_allowed(allowed_tables, allowed_columns, sm: dict, ctx):
    """The centralized final gate (see module docstring): ``allowed_tables``
    (any iterable of table names) and ``allowed_columns`` (any iterable of BARE
    column names — matching ``validate_and_parameterize``'s own AST-comparison
    convention, which does not qualify columns by table either) narrowed to what
    ``ctx.allowed_resources`` permits, however ``allowed_tables``/``allowed_columns``
    were originally built (FastPath, cache replay, retrieval-driven planning,
    FK/entity expansion, Tier-2).

    Returns ``(allowed_tables, allowed_columns)`` UNCHANGED (not copied) when
    there is nothing to restrict, so a caller that never forwards a real scope
    pays no cost and sees no behaviour change.

    A table with no entry in ``sm['tables']`` at all (can happen for a CTE alias
    or a name ``validate_and_parameterize`` itself adds — see its own docstring)
    is dropped here rather than assumed safe: this function only ever WIDENS
    nothing, and ``validate_and_parameterize`` re-adds CTE/SELECT aliases on its
    own after receiving these narrowed sets, so nothing legitimate is lost.
    """
    if ctx is None or getattr(ctx, "allowed_resources", None) is None:
        return allowed_tables, allowed_columns

    open_sources, table_index = _index(ctx)
    tables_meta = sm.get("tables") or {}

    def table_ok(name) -> bool:
        entry = tables_meta.get(name)
        if entry is None:
            return False
        return _table_allowed(open_sources, table_index, _table_source_id(entry, ctx), name)

    narrowed_tables = {t for t in allowed_tables if table_ok(t)}
    if not narrowed_tables:
        return narrowed_tables, []

    proposed_columns = set(allowed_columns or [])
    permitted_columns = set()
    for key, entry in (sm.get("columns") or {}).items():
        table_key, _, col_name = key.rpartition(".")
        if not table_key or table_key not in narrowed_tables or col_name not in proposed_columns:
            continue
        sid = _table_source_id(entry, ctx)
        if _column_allowed(open_sources, table_index, sid, table_key, col_name):
            permitted_columns.add(col_name)

    return narrowed_tables, sorted(permitted_columns)


def filter_nosql_collections(collections, source_id, ctx):
    """The NoSQL mirror of ``filter_sm`` (2026-08-08 audit finding: ``_run_nosql``
    in ``veda_hybrid.py`` had no RBAC filtering at all — the relational path's
    ``sm['tables']``/``sm['columns']`` shape doesn't apply here, since NoSQL
    schema comes from ``connectors.base.NoSQLCollection`` objects, not the
    semantic model).

    ``collections`` (one source's ``get_nosql_schema()`` result) is narrowed to
    what ``ctx.allowed_resources`` permits for ``source_id`` — a collection
    ~= a table, each ``inferred_fields`` entry (``{"name": ..., ...}``) ~= a
    column. Returns ``collections`` UNCHANGED when there is nothing to
    restrict. Never mutates a kept ``NoSQLCollection`` in place — a partially
    restricted collection is returned as a new instance via
    ``dataclasses.replace``.
    """
    if ctx is None or getattr(ctx, "allowed_resources", None) is None:
        return collections

    open_sources, table_index = _index(ctx)
    if source_id in open_sources:
        return collections

    kept = []
    for coll in collections:
        cols = table_index.get((source_id, coll.collection_name), _MISSING)
        if cols is _MISSING:
            continue  # this collection is not permitted at all
        if cols is None:
            kept.append(coll)  # whole collection open — nothing to narrow
            continue
        fields = [f for f in coll.inferred_fields if f.get("name") in cols]
        if fields:  # a collection with zero reachable fields is not addressable
            kept.append(dataclasses.replace(coll, inferred_fields=fields))
    return kept


def filter_doc_chunks(chunks, ctx):
    """The document-retrieval mirror of ``filter_nosql_collections``:
    ``chunks`` (``ingestion.chunk_embedder.ChunkRetrievalResult``s, each already
    carrying its own ``source_id``/``doc_name``) narrowed to what
    ``ctx.allowed_resources`` permits — a document ~= a NoSQL collection (a leaf;
    documents have no column-level grain beneath them, unlike a db table). The
    producer of this same per-document allow-list is
    ``apps.access_management.services.data_scope``'s ``files``-kind branch of
    ``_source_scope`` (``TableScope(name=doc_name, columns=None)`` per allowed
    document).

    Returns ``chunks`` UNCHANGED when there is nothing to restrict (including an
    empty list) — same contract as every other function in this module. Apply
    this to the OVER-FETCHED candidate list ``retrieve_top_k_chunks`` returns,
    before truncating to the caller's requested ``top_k``: same reasoning as
    ``filter_retrieval_results`` — the shared retrieval query stays RBAC-
    oblivious, only the per-request result list is narrowed."""
    if ctx is None or getattr(ctx, "allowed_resources", None) is None or not chunks:
        return chunks

    open_sources, table_index = _index(ctx)

    def sid_of(chunk):
        try:
            return int(chunk.source_id)
        except (TypeError, ValueError):
            return chunk.source_id

    if all(sid_of(c) in open_sources for c in chunks):
        return chunks  # nothing restricted among these chunks' sources — zero-cost pass-through

    def ok(chunk) -> bool:
        sid = sid_of(chunk)
        if sid in open_sources:
            return True
        return (sid, chunk.doc_name) in table_index

    return [c for c in chunks if ok(c)]
