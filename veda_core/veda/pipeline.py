"""VEDA · The L1→L7 orchestrator (run_query)."""
import os, re, sys, time, json, logging, threading
from query.ranking_parser import parse_ranking
from veda.cache import save_verified_query, verified_cache_lookup
from veda.execution import execute_sql
from veda.generation import generate_sql
from veda.planning import existence_mode, try_multitable
from veda.routing import recommended_projection, select_primary_table, vet_primary
from veda.rbac_filter import filter_retrieval_results, narrow_allowed, restricted_names
from veda.runtime import get_engine
from veda.validation import (qualifier_completeness, validate_and_parameterize, value_grounding,
                             grouped_shape_ok, distinct_shape_ok, ranked_shape_ok)
from utils.logger import get_logger
import importlib
from veda.explain import new_trace
from veda.execution_state import ExecutionState
from slm._call_slm import collect_usage, usage_totals
from query.temporal_parser import run_temporal_parser
from veda.planning import aggregate_mode as _agg_mode, grouped_mode as _grp_mode, ratio_mode as _rat_mode, superlative_mode as _sup_mode
from veda.planning import grouped_count_mode as _grpc_mode   # grouped COUNT per dimension (2026-09-15)
from query.fast_path import try_fast_path, log_route
import sqlglot as _sg
from sqlglot import exp as _exp
from veda.ir_equivalence import validate_ir_equivalence

logger = get_logger(__name__)


def _ambient_ctx():
    """The ambient RequestContext, or None — both context module names (see
    veda_hybrid._current_ctx / veda.execution._scope_source_ids for why: the engine
    is imported both as bare `context` and as `veda_core.context`, which Python
    loads as two separate module objects with separate thread-locals)."""
    for modname in ("veda_core.context", "context"):
        try:
            ctx = importlib.import_module(modname).try_current()
            if ctx is not None:
                return ctx
        except Exception:
            continue
    return None


def _resolve_temporal_column(table, sm, query=None):
    """Canonical temporal column of `table` — schema-metadata driven (semantic_type=
    TEMPORAL). When several exist, REUSE the existing canonical-temporal chooser
    (query.sql_builder._pick_best_temporal) so there's ONE source of truth for the
    event-time preference, not a second hardcoded name list. None if no temporal column.

    `query` lets that chooser bind the column to the verb the question used
    ("most recently UPDATED" -> updated_at, "most recently DATED" -> transaction_date)
    and prefer the business event date over created_at."""
    temporal = [k.split(".", 1)[1] for k, m in (sm.get("columns", {}) or {}).items()
                if k.startswith(table + ".") and (m or {}).get("semantic_type") == "TEMPORAL"]
    if not temporal:
        return None
    if len(temporal) == 1:
        return temporal[0]
    try:
        from query.sql_builder import _pick_best_temporal
        return _pick_best_temporal(temporal, {c: {"col_name": c} for c in temporal},
                                   query=query)
    except Exception:
        return sorted(temporal)[0]


# Bare "recent" joins the list (2026-09-23): "any RECENT payments between 100 and
# 50,000" is the same vague wording as "recently" and must not pin a synthetic 30-day
# window onto a question whose real constraint is the amount range.
_VAGUE_RECENCY_RE = re.compile(
    r'\b(?:recently|recent|lately|latest|newest|most\s+recent(?:ly)?)\b', re.IGNORECASE)


def _sql_references(sql: str, name: str) -> bool:
    """Whether ``sql`` names ``name`` as an identifier — quoted (``"x"``) OR bare (``x``).

    Bare-name matching is REQUIRED, not a convenience. This predicate decides
    whether a validation rejection was RBAC-caused, and the earlier version tested
    only ``f'"{name}"' in sql`` on the stated assumption that "SQL here always
    double-quotes identifiers". That assumption does not hold on the verified-cache
    REPLAY path, which hands back the stored SQL unquoted
    (``FROM accounts_generalledger``). So every RBAC-caused rejection on that path
    was misclassified as a generic ``invalid`` with NO feedback attached, which
    reached the user as ``ask_clarification_node``'s "Could you clarify what you're
    asking about?" — an ambiguity prompt for what was actually a permission denial,
    and nothing printed to the engine log either (only ``_feedback`` prints).

    Word-boundary anchored, so it still cannot fire on an unrelated identifier that
    merely CONTAINS a restricted name — the false-positive the quoted-only test was
    reaching for. ``"`` is not a word character, so one pattern covers both forms.
    """
    return re.search(rf'(?<!\w){re.escape(name)}(?!\w)', sql) is not None


def _is_vague_recency_only(raw_expressions):
    """True when every temporal span the L1 parser matched is a bare vague-recency
    word (latest/newest/most recent/recently/lately) — i.e. the derived 30-day
    BETWEEN window is a heuristic guess, not something the user explicitly asked
    for. Used to prefer ORDER BY + LIMIT (an explicit top-N ranking request like
    'latest 10 ledger entries') over a hard date-range filter that would silently
    exclude rows outside an arbitrary 30-day window and ignore the requested N."""
    return bool(raw_expressions) and all(_VAGUE_RECENCY_RE.search(e) for e in raw_expressions)


def _resolve_rank_metric_column(table, sm):
    """The anchor's single unambiguous measure column, if the schema names exactly
    one (veda_semantic_model.json tables[...].candidate_measure_columns) — lets a
    'top N'/'highest N' style ranking pick an ORDER BY column without guessing
    among several measures."""
    candidates = (sm.get("tables", {}).get(table, {}) or {}).get("candidate_measure_columns") or []
    return candidates[0] if len(candidates) == 1 else None


def _rank_sort_column(rank, table, sm, tcol):
    """The single column (if any) `_rank_order_limit_sql` below will actually
    ORDER BY for this ranking request — extracted as its own function so
    callers that need to know (recommended_projection's must_include, so a
    "latest 10 X" result is never sorted by a column it doesn't also show)
    don't duplicate this basis/temporal/metric branching. None means no
    ranking was requested (plain LIMIT, no ORDER BY)."""
    if rank.basis == "temporal" and tcol:
        return tcol
    if rank.basis == "metric":
        return _resolve_rank_metric_column(table, sm)
    return None


def _rank_order_limit_sql(rank, table, sm, tcol, alias=None):
    """Shared ' ORDER BY ... LIMIT ...' tail for every hand-built single-table SQL
    branch (FK / multi-hop / value-filter / temporal-only / plain listing) — so
    'latest 10 X' / 'top 5 Y' / 'bottom 3 Z' are honored everywhere a single-table
    SELECT is constructed deterministically, not only on the LLM-generated path.
    Emits NO LIMIT when the query named no row count. It used to fall back to a
    historical ' LIMIT 100', which was not a display cap: every figure the user is
    given is derived from the rows that come back, so the page silently became the
    population — measured 2026-09-22 on assets_asset (7,814 rows), answers asserted
    "There are 1000 assets listed", "The average carpet area is 1628.46 square meters"
    and "86% of the assets listed are in Pune" (on a query whose WHERE already
    restricted every row to Pune, so the true share is 100%).

    This was the FOURTH place imposing that cap, and the last to be found, because each
    one hid the ones below it: veda/generation.py's own default, veda/validation.py
    appending one to any SQL that carried none, and the model writing "LIMIT 100"
    itself. Removing the first three left this one still capping every hand-built
    single-table branch, which is the "deterministic" route most listing questions take.

    A ranking the USER asked for ("top 5", "latest 10") is unchanged.

    `alias`: some branches FROM the anchor under an alias (answer-entity's `a`/`t`
    join) — the ORDER BY column must be qualified there to stay unambiguous."""
    limit = rank.top_n
    prefix = f"{alias}." if alias else ""
    sort_col = _rank_sort_column(rank, table, sm, tcol)
    _lim = f" LIMIT {limit}" if limit is not None else ""
    if sort_col:
        direction = "ASC" if rank.direction == "asc" else "DESC"
        return f' ORDER BY {prefix}"{sort_col}" {direction}{_lim}'
    if rank.ranked:
        # A RANKED request we could not resolve an ORDER BY for (no canonical temporal
        # column for a recency ask; several equally-plausible measures for a magnitude
        # one). Emitting ' LIMIT N' here is the worst available answer: it returns N
        # ARBITRARY rows under the user's own words "the top 5" / "the last 5", which is
        # indistinguishable from a real ranking (2026-09-23 question.txt, Q11 — the
        # ORDER BY was dropped and a LIMIT 5 survived, so five unordered rows were
        # rendered as "the top 5 most recently dated"). Drop the COUNT instead — and
        # emit no LIMIT at all, not the historical ' LIMIT 100' this used to return,
        # which was itself a silent population cap (see the docstring). The unranked
        # page that comes back no longer impersonates a ranking, and the intent/SQL
        # alignment guard downstream refuses it as a typed clarify rather than letting
        # it render.
        return ''
    return _lim


def _temporal_predicate(table, sm, tf):
    """Grounded BETWEEN/>=/<= predicate on the anchor's canonical temporal column, or ''
    when there's no window or no temporal column. Literals are parameterised downstream by
    validate_and_parameterize (same as every other deterministic-path literal)."""
    if not tf or not (tf.start or tf.end):
        return ""
    col = _resolve_temporal_column(table, sm)
    if not col:
        return ""
    q = f'"{col}"'
    if tf.start and tf.end:
        return f"{q} BETWEEN '{tf.start}' AND '{tf.end}'"
    if tf.start:
        return f"{q} >= '{tf.start}'"
    return f"{q} <= '{tf.end}'"


def _anchor_from_sql(sql: str) -> str:
    """The table a statement actually selects FROM.

    The executed SQL is the ground truth for what ran. Anchor SELECTION (the router,
    entity resolution, vet_primary) decides what to build; this reports what was built,
    and the two were measurably able to disagree.
    """
    if not sql:
        return ""
    try:
        tree = _sg.parse_one(sql, read="postgres")
        frm = tree.find(_exp.From)
        tbl = frm.find(_exp.Table) if frm is not None else None
        return (tbl.name or "") if tbl is not None else ""
    except Exception:
        return ""


def run_query(query, sm, all_cols, return_result=False, anchor_hint=None, on_event=None,
              _reentry=False, summarise=True):
    """Run one NL→SQL→result. Reuses the shared engine; never closes it.

    Returns an int status code (0 ok / 1 error) by default — backward-compatible.
    With return_result=True, returns a dict {status, ok, cols, rows, answer, sql, …}
    so callers (the hybrid fusion, the Tier-2 fallback) can use the executed rows and
    distinguish 'answered' from 'refused'/'clarify'/error (the int code can't).

    anchor_hint (internal, qualifier salvage): force this table as the primary anchor
    — set only by the salvage retry after a first pass refused with a dropped
    qualifier whose QSR referent lives in a table retrieval never surfaced. Every
    downstream correctness gate still judges the plan; the hint also marks the run as
    a retry so salvage can never recurse.

    summarise (default True): when False, skip the SLM PROSE call (insight engine /
    run_nl_answer) and leave the deterministic fallback answer in place. Everything
    else — SQL, execution, the analytics pass, `explain` — is unchanged. Set by
    callers that discard this answer and write their own: the hybrid path runs
    run_query for its correct-by-construction rows and then has run_hybrid_layer
    synthesise the prose over those same rows, so the summary paid for here was
    computed and thrown away — a second summary-class SLM call on every hybrid turn."""
    start = time.time()
    join_constraints = None
    fanout_guard = None
    _llm_sql = False          # True only when the SQL's SELECT/WHERE was LLM-written
    # M3 checkpoint 1: branch state the firewall's IR is built from (bound later by the
    # branches that have it; these defaults mean "unknown" → partial IR slots).
    _u = None; _arb_filters = []; _rank = None; _tpred = None; _tcol = None; _rank_sort_col = None
    # Same rule, same reason, for the two filter classes added 2026-09-23: they are bound
    # inside the single-table planning block, but READ unconditionally by the branch
    # selection (`elif _arb_filters or _num_filters or _vg_filters`) and by the IR
    # construction far below. Any path that reaches those without entering that block hit
    # `UnboundLocalError: cannot access local variable '_num_filters'`, which the API tier
    # surfaced as a 502 LLM_UNAVAILABLE — five of the twenty question.txt questions failed
    # this way the moment the scope was pinned to one source. `_arb_filters` above already
    # carried this default for exactly this reason; these two were missing it.
    _num_filters = []; _num_clarify = None; _vg_filters = []; _vg_matched = []
    # M4: the QueryIR the firewall checked, surfaced on the RESULT so the api tier's
    # session memory can stack it (chatbot/memory/frame.py). A mutable holder, not a
    # plain local, because _done() is a closure that also runs on refusal paths that
    # return BEFORE the IR is built — reading an unbound local there would NameError.
    _ir_holder = {"ir": None}
    # the understanding candidate's state — bound in the planning block, read by the firewall
    # IR on EVERY path (the fast path returns before the planning block runs)
    _analytical_sql = None; _analytical_primary = None; _analytical_spec = None
    _analytical_used = False; _analytical_reentry_ok = False

    def _guard_refusal(status, msg, fb, tail, always_done=False, **done_kw):
        """M2 (2026-09-16): a deterministic answer (fast path, planner, branch) that a shape
        guard / planner REFUSED is not terminal while the understanding layer is on — the
        grounded candidate never got its turn ("no deterministic branch matched" must
        include "the one that matched was refused"). Re-enter ONCE: fast path skipped,
        the candidate first in the planning chain. Flags off, already re-entered, or a
        cache replay → the refusal stands exactly as before."""
        try:
            from config import QUERY_UNDERSTANDING_ENABLED as _qu
        except Exception:
            _qu = False
        if _qu and not _reentry and not from_cache:
            print(f"  [QU] {tail}: deterministic answer refused → re-entering once with the "
                  f"grounded candidate (fast path off)")
            log_route(f"{tail}.reentry", query, (time.time() - start) * 1000)
            return run_query(query, sm, all_cols, return_result=return_result,
                             anchor_hint=anchor_hint, on_event=on_event, _reentry=True,
                             summarise=summarise)
        log_route(tail, query, (time.time() - start) * 1000)
        _kw = dict(done_kw)
        if msg is not None:
            _kw["msg"] = msg
        return _done(0, status, feedback=fb, **_kw) if (return_result or always_done) else 0
    tr = new_trace(query)
    es = ExecutionState()
    _usage = collect_usage()
    _usage.__enter__()  # closed in _done() — the single funnel for every exit below

    _ticks: list = []   # passive (phase, message) record of every _tick() below,
                        # for build_explain()'s "timeline" — see _done()'s return.
    # In-flight narration handles. Cancelled in _done() so a narration that loses
    # the race to the answer is DISCARDED rather than waited for (veda/narrator.py).
    _narration_handles: list = []

    def _tick(phase, message):
        """Fire a live, user-facing thinking event. Static string only — no LLM/SLM
        call, no extra DB round-trip. A no-op when on_event is None or itself raises,
        so progress reporting can never fail a query."""
        _ticks.append((phase, message))
        if on_event is None:
            return
        try:
            on_event(phase, message, {})
        except Exception:
            logger.exception("_tick: on_event callback raised for phase=%s", phase)

    # The conversation layer's structured state for this turn, when this is a chat
    # follow-up (veda_core/context.py::set_conversation_context, bound in
    # run_hybrid_query). {} for every other caller — /api/v1/query, the CLI, ingestion,
    # evaluation — so everything below is a no-op for them.
    #
    # It is consumed in exactly three places, all of them narrow:
    #   · entity_table  -> the EXISTING anchor_hint, so the remembered table no longer
    #                      has to be smuggled in as prose
    #   · user_message  -> the qualifier gate's basis, so it judges the user's words
    # `filter_values` reaches here in the payload but is deliberately NOT consumed — see
    # the note at the value arbiter below for the measurement that decided that.
    # Nothing here plans, joins or generates SQL; the semantic layer keeps that job.
    try:
        from veda_core.context import current_conversation_context as _cur_conv
        _conv = _cur_conv() or {}
    except Exception:
        _conv = {}
    _conv_user_message = _conv.get("user_message") if isinstance(_conv, dict) else None
    _conv_filters = [f for f in (_conv.get("filters") or [])
                     if isinstance(f, dict) and f.get("column") and f.get("value") is not None]
    try:
        _conv_limit = int(_conv["limit"]) if _conv.get("limit") is not None else None
    except (TypeError, ValueError):
        _conv_limit = None
    _conv_group_by = [str(g) for g in (_conv.get("group_by") or [])]
    _conv_measures = [str(m) for m in (_conv.get("measures") or [])]
    _conv_order_by = [str(o) for o in (_conv.get("order_by") or [])]

    def _anchor_columns_for(_sm, _table):
        """The columns the semantic model says `_table` has, or None when it says nothing.
        sm["columns"] is keyed "table.column" — the only statement of that fact."""
        cols = {k.split(".", 1)[1] for k in (_sm.get("columns") or {})
                if k.startswith(_table + ".")}
        return cols or None

    def _grouped_this_turn(_q):
        """Did the USER ask for a grouping in THIS message? Their words win over a
        remembered shape — reusing the same grammar signal veda/validation.py's
        grouped-shape guard already uses, rather than inventing a second one."""
        ql = " " + (_q or "").lower().strip() + " "
        return any(w in ql for w in (" by ", " per ", " each ", "distribution",
                                     "breakdown", "broken down", "grouped"))
    # A remembered table is an anchor HINT, never an override of one the caller passed:
    # anchor_hint is also the qualifier-salvage retry marker (see below), and that retry
    # must keep deciding its own anchor.
    # SOURCE CHECK, before anything in the context is used. The conversation remembers
    # which source its entity came from; the CURRENT turn's scope is resolved fresh from
    # RBAC every request (apps/chat/views.py) and is the only authority. If the remembered
    # source is not in it — a revoked grant, a narrowed pin, a different scope — the whole
    # context is dropped rather than partially applied. Memory is context, not authority.
    if _conv.get("source_id") is not None:
        try:
            from veda_core.context import try_current as _try_ctx
            _rc = _try_ctx()
            _scope = {str(x) for x in (getattr(_rc, "source_ids", ()) or ())} if _rc else set()
        except Exception:
            _scope = set()
        if _scope and str(_conv["source_id"]) not in _scope:
            print(f"  [conversation] remembered source {_conv['source_id']} is outside this "
                  f"turn's scope {sorted(_scope)} — context dropped")
            _conv, _conv_user_message, _conv_filters = {}, None, []
            _conv_limit, _conv_group_by, _conv_measures = None, [], []

    if anchor_hint is None and _conv.get("entity_table"):
        _hinted = _conv.get("entity_table")
        if _hinted in (sm.get("tables") or {}):
            anchor_hint = _hinted
            print(f"  [conversation] anchor from remembered context: {_hinted}")
        else:
            print(f"  [conversation] remembered table {_hinted!r} is not in this scope's "
                  f"semantic model — ignored")

    def _feedback(status, **ctx):
        """Build + print actionable failure guidance (why / what's needed / suggestions).
        Returns the feedback dict (or None). Never raises — falls back to silence."""
        try:
            from config import FEEDBACK_ENABLED
        except Exception:
            FEEDBACK_ENABLED = True
        if not FEEDBACK_ENABLED:
            return None
        try:
            from veda.feedback import explain_failure
            # Attached to a COPY passed only to explain_failure, never to the
            # `sm` the rest of run_query reasons over — a refusal must be able
            # to say "you don't have permission" instead of "that doesn't
            # exist" for a restricted table/column, without narrowing what
            # retrieval/planning/SQL-generation see (that stays exactly the
            # existing filter_retrieval_results/narrow_allowed's job).
            _restricted = restricted_names(sm, _ambient_ctx())
            _sm_for_feedback = (
                {**sm, "_rbac_restricted": _restricted}
                if (_restricted["tables"] or _restricted["columns"]) else sm)
            fb = explain_failure(status, _sm_for_feedback, **ctx)
            print("\n" + fb["text"] + "\n")
            # NOTE: the access_check downgrade used to happen HERE, the moment this
            # feedback object was built. That was wrong: building feedback is not the
            # same as ANSWERING with it. Measured live on the CSV-lake path, feedback
            # was built for an intermediate failure, the pipeline then RECOVERED via
            # Tier-2 and ended in a clarify about an ambiguous column — yet the user
            # was left with a red cross on "Checking access permissions" and a reply
            # that had nothing to do with permissions. The downgrade now happens in
            # `_done`, the one exit, where the TERMINAL feedback is known.
            return fb
        except Exception:
            return None

    # Entity-coverage verdict (veda/intent_sql_alignment.entity_coverage), filled in after the
    # qualifier gate below and read by _done(): the entities the question named that this SQL does
    # NOT cover. A dict, not a name, so the check can be recorded where the SQL is final while the
    # confidence/explainability that report it stay in the one place that builds them.
    _coverage = {}
    # Which answer-producing LANE served this query. A plain local wouldn't do:
    # `_done` is a closure and `from_cache` is assigned far below it, so an early
    # refusal would read it UNBOUND. A holder set at the lane itself is always safe.
    #
    # WHY THIS EXISTS AT ALL: the audit's `cache_hit` was detected by comparing
    # `engine_result["table"]` to the sentinel "(cached)". That sentinel was removed
    # from this file (it was poisoning topic-switch detection), so the comparison has
    # been silently False ever since — measured: the last `cache_hit=True` audit row
    # is 2026-07-09, while the engine serves verified-cache hits every day, and
    # `veda_cache_hits_total` has been stuck at 0 with it. Detect the lane at the
    # lane, not by guessing from a display field.
    _lane = {"cache": False}

    def _emit_validation_phase(status: str = "answered"):
        """EXP-B5: report the validation ledger on EVERY terminal path.

        The previous emit sat after the AST check, which a refusal never reaches — so
        refused answers showed no validation line at all (measured: 4 of 10 benchmark
        queries). Called from _done, the one exit every status funnels through.

        Emits NOTHING when the ledger is empty. A query refused before any SQL existed
        genuinely ran no checks, and printing "checks passed" there would be a lie —
        absent is the honest representation, not a green tick."""
        try:
            from veda import lifecycle as _lcd
            _tld = _lcd.current_timeline()
            if not getattr(_tld, "enabled", False):
                return
            if any(e.phase == _lcd.PHASE_VALIDATION for e in _tld.events):
                return                      # already reported on this query
            _ck = tr.sections.get("validation", {}).get("checks", []) or []
            if not _ck:
                return
            _bad = [c for c in _ck if c.get("status") != "pass"]
            if _bad:
                _tld.failed(_lcd.PHASE_VALIDATION)
            elif status == "answered":
                # No COUNT here. This line is emitted before build_explain runs,
                # so it can only count the trace's checks — while the panel the
                # reader opens lists the EXPANDED labels, a different (larger)
                # number. A live line that says "4" above a list of 5 is worse than
                # one that says neither; the number belongs where the list is.
                _tld.completed(_lcd.PHASE_VALIDATION, "Safety checks passed")
            else:
                # The ledger holds the AST/read-only/fan-out checks, and those DID
                # pass. But this turn produced no answer — it was stopped by a LATER
                # correctness gate (intent/SQL alignment, aggregate omission,
                # dimension alignment) that writes no ledger entry. Reporting a green
                # "N safety checks passed" under a step titled "checking the result is
                # complete and safe", on a turn whose reply says it could not answer,
                # is precisely the contradiction this layer exists to remove.
                # Measured live on a verified-cache replay: 2 of 3 cache hits were
                # stopped by the alignment gate and still rendered four green ticks.
                _tld.warning(_lcd.PHASE_VALIDATION,
                             "This result did not pass a completeness check")
        except Exception:
            pass

    def _done(rc, status, **kw):
        # Report the anchor the SQL actually ran against, decided ONCE at Tier-1's
        # single exit so every branch gets it — the discipline _mark_empty_results
        # already uses for the empty-result outcome.
        #
        # Measured 2026-09-22: "what about Mumbai", third turn of a drill-down, built
        #   FROM "assets_asset" t0 JOIN generics_country JOIN users_useraddress
        #                          JOIN accounts_generalledger
        # and reported table="accounts_generalledger". harvest_frame takes the
        # QueryFrame's entity from that field, so the conversation silently relocated
        # to a financial-ledger topic and every later turn — "only the residential
        # ones", "go back", "remove the city filter" — asked about the wrong table and
        # came back asking for clarification. The SQL's own FROM said assets_asset the
        # whole time.
        _sql_anchor = _anchor_from_sql(kw.get("sql") or "")
        if _sql_anchor and kw.get("table") and kw["table"] != _sql_anchor:
            print(f"  [L7] reported table {kw['table']!r} disagrees with the executed "
                  f"SQL's FROM {_sql_anchor!r} — reporting the executed anchor")
            kw["table"] = _sql_anchor
        for _h in _narration_handles:          # answer wins the race, always
            try:
                _h.cancel()
            except Exception:
                pass
        # NOTE: the access_check downgrade is deliberately NOT made here either.
        # This is TIER-1's exit, and Tier-1 refusing is not the TURN refusing — the
        # deterministic head hands off to Tier-2, which can answer or clarify on a
        # completely unrelated ground. Measured live on a data-lake question: Tier-1
        # refused, feedback classified it as an access problem, Tier-2 then produced
        # a clarify about an ambiguous column, and the user was left with a red cross
        # on "Checking access permissions" above a reply about column names.
        # The downgrade is made once, at the FRONT DOOR, from the turn's terminal
        # feedback (veda_hybrid._reconcile_access_check).
        _emit_validation_phase(status)
        try:
            # Resolve any still-open phase BEFORE build_explain reads the timeline.
            # `access_check` completes late (after validation), so without this the
            # PERSISTED payload recorded it as "started" forever — an unresolved
            # step inside a finished, stored answer.
            from veda import lifecycle as _lcc
            _lcc.current_timeline().close_open_phases(failed=(status not in
                ("answered", "clarify")))
        except Exception:
            pass
        if status != "answered":
            _refusal = kw.get("msg") or kw.get("error") or kw.get("missing")
            tr.set("output", refusal=_refusal)
            es.refusal_reason = _refusal
        else:
            _tick("output", "Done — here's your answer")
        _calls = _usage.calls()
        _totals = usage_totals(_calls)
        tr.total_prompt_tokens = _totals["prompt_tokens"]
        tr.total_completion_tokens = _totals["completion_tokens"]
        tr.total_tokens = _totals["total_tokens"]
        _sql_calls = [c for c in _calls if c["purpose"] in ("sql_single_table", "sql_join")]
        if _sql_calls:
            tr.set("output", prompt_tokens=_sql_calls[-1]["prompt_tokens"],
                   completion_tokens=_sql_calls[-1]["completion_tokens"],
                   sql_model=_sql_calls[-1]["model"])
        _nl_calls = [c for c in _calls if c["purpose"] in ("nl_answer", "insight_engine")]
        if _nl_calls:
            tr.set("nl_summary",
                   summary_tokens=_nl_calls[-1]["prompt_tokens"] + _nl_calls[-1]["completion_tokens"],
                   summary_model=_nl_calls[-1]["model"])
        _usage.__exit__(None, None, None)
        tr.finish(status)
        # Tier2 continuation context (Tier1→Tier2 propagation) — deliberately NOT the
        # full trace (that stays below, for debugging); just what Tier2 needs to avoid
        # recomputing temporal parsing / query understanding / retrieval / primary table.
        es.sql_planning = dict(tr.sections.get("sql_planning", {}))
        if return_result:
            explain = None
            if status == "answered":
                _confidence = None
                try:
                    from query.result_explainer import synthesize_confidence
                    _anchor_conf = tr.sections.get("anchor_selection", {}).get("confidence")
                    _join_conf = tr.sections.get("join_planning", {}).get("confidence")
                    _conf_inputs = {k: v for k, v in
                                   (("anchor", _anchor_conf), ("join", _join_conf)) if v is not None}
                    # A partial answer is a good answer, not a certain one — it must never
                    # reach the 1.0 an uncaveated complete answer gets.
                    if _coverage.get("missing"):
                        from config import ENTITY_COVERAGE_CONFIDENCE
                        _conf_inputs["entity_coverage"] = ENTITY_COVERAGE_CONFIDENCE
                    _confidence = synthesize_confidence(_conf_inputs)
                except Exception:
                    logger.exception("synthesize_confidence failed — result confidence omitted")
                try:
                    # EXP-B3: a low-confidence answer must SAY so. `confidence` was
                    # already computed and already shipped inside the payload, but
                    # nothing SURFACED it — measured on a real query, a 0.143-confidence
                    # answer carried exactly the same green ticks as a 1.0 one.
                    #
                    # Raised here because this is the one place confidence exists; the
                    # threshold is configurable because the right value is a product
                    # call, not an engineering one (see LOW_CONFIDENCE_WARNING_BELOW).
                    # Only a COMPUTED score triggers the caveat. Treating "no
                    # score" as low was tried and reverted: on this path an empty
                    # input set is normal for fast-path and cached answers, so it
                    # put a "limited matching data" caveat on perfectly good
                    # answers — measured on "top 5 credit transactions", which is
                    # correct and was flagged. A caveat on everything is a caveat
                    # on nothing.
                    if _confidence is not None:
                        import config as _cfgc
                        _floor = float(getattr(_cfgc, "LOW_CONFIDENCE_WARNING_BELOW", 0.0) or 0.0)
                        if _floor > 0 and float(_confidence) < _floor:
                            from veda import warnings as _vwc
                            _vwc.add(_vwc.LOW_EVIDENCE)
                            # E.1 (2026-09-23): ZERO ROWS at low confidence is not an
                            # answer. "No results found." is a factual claim about the
                            # data — it tells the user their portfolio contains nothing
                            # matching — and when the engine is itself unsure it picked
                            # the right table, that claim is unfounded. Q11 rendered
                            # "No results found." at confidence 0.045 against the WRONG
                            # table; the right table had 754 rows. Say which anchor was
                            # searched and let the user redirect, rather than reporting
                            # an emptiness that may be an artefact of the routing.
                            # Narrow on purpose. An empty result is usually a genuine,
                            # useful fact ("no users were created last month"), so this
                            # must fire only when the engine is BADLY unsure it even
                            # looked in the right place — Q11 scored 0.045 against the
                            # wrong table. At the plain warning threshold (0.5) it
                            # converted correct empty answers into clarifies, which is a
                            # worse failure than the one it fixes.
                            _rows_out = kw.get("rows")
                            _very_low = float(_confidence) < min(_floor / 3.0, 0.15)
                            if _very_low and rc == 0 and isinstance(_rows_out, (list, tuple)) \
                                    and len(_rows_out) == 0:
                                # kw["table"] is the anchor this answer ran against;
                                # `primary` is a closure local that is not guaranteed to
                                # be bound at every _done() exit, so it is not used here.
                                _anchor_name = kw.get("table") or "that data"
                                _lcmsg = (
                                    f"I searched {_anchor_name} and found no matching "
                                    f"rows, but I'm not confident that's the right place "
                                    f"to look for this — so I'd rather not report it as "
                                    f"an empty result. Which entity did you mean?")
                                status = "clarify"
                                kw["msg"] = _lcmsg
                                kw["feedback"] = _feedback("clarify", msg=_lcmsg)
                                tr.finish(status)
                except Exception:
                    pass
                try:
                    from veda.business_explain import build_explain
                    explain = build_explain(
                        sql=kw.get("sql") or "", table=kw.get("table") or "", sm=sm,
                        checks=tr.sections.get("validation", {}).get("checks", []),
                        visualization=kw.get("visualization"),
                        params=params,
                        timeline=_ticks,
                        confidence=_confidence,
                        not_included=_coverage.get("missing"),
                        # The v2 blocks (sources / routing / execution / warnings /
                        # result / cross_source / support) are projected FROM the
                        # trace by veda/safe_projection.py. No-op while
                        # EXPLAIN_V2_ENABLED is off — payload stays v1.
                        trace=tr,
                        trace_id=getattr(tr, "trace_id", "") or "",
                    )
                except Exception:
                    logger.exception("business_explain failed — end-user explainability omitted")
                # EXPLAINABILITY (compact) — never the full payload; just the shape.
                try:
                    if isinstance(explain, dict):
                        # Reads build_explain()'s REAL nested shape (data_used.datasets,
                        # validation.checks) — see explain.summarize_explain_payload for
                        # why this is shared rather than re-derived here.
                        from veda.explain import summarize_explain_payload
                        tr.set("explainability", **summarize_explain_payload(explain))
                except Exception:
                    pass
            elif kw.get("feedback"):
                # Refusal path: same structured-explainability CONTRACT as a
                # success, built from the feedback _feedback() already computed
                # above (why/what_needed/suggestions) — not from SQL, which
                # doesn't exist for a refusal. None when no feedback dict is
                # available (invalid/exec_error's _done() calls don't build
                # one — see those call sites), same as before this change.
                try:
                    from veda.business_explain import build_refusal_explain
                    explain = build_refusal_explain(
                        status, kw.get("feedback"),
                        trace=tr, trace_id=getattr(tr, "trace_id", "") or "")
                except Exception:
                    logger.exception("build_refusal_explain failed — refusal explainability omitted")
            # business_intent (advisory, Phase-1 business-aware output): the
            # deterministic one-sentence business reading of the EXECUTED SQL —
            # build_explain's understanding.summary, surfaced as a convenience
            # top-level key. Derived from the SQL that actually ran (source of
            # truth), never from an LLM's own claim about what it meant to do.
            if status == "answered" and explain:
                kw.setdefault("business_intent",
                              (explain.get("understanding") or {}).get("summary"))
            _ir_obj = _ir_holder.get("ir")
            return {"status": status, "ok": (status == "answered"),
                    # the source this answer ran against (P4, 2026-09-18): the chat frame
                    # records it so a drill-down stays on the same source
                    "source_id": getattr(_ambient_ctx(), "source_id", None),
                    # M4: the structured question this answer actually answered. The chat
                    # tier stacks it and applies the next turn's delta to it, instead of
                    # re-deriving intent from a restated English sentence. None on a
                    # refusal that never reached the firewall.
                    "ir": (_ir_obj.to_dict() if _ir_obj is not None else None),
                    # Private (underscore) by convention: audit-only. The user-facing
                    # payload is built solely by veda/safe_projection.py, which has no
                    # reader for this, so it cannot leak into an explanation.
                    "_from_cache": bool(_lane["cache"]),
                    "trace": tr.to_dict(), "explain": explain,
                    "usage": {"prompt_tokens": tr.total_prompt_tokens,
                              "completion_tokens": tr.total_completion_tokens,
                              "total_tokens": tr.total_tokens},
                    "latency_ms": tr.total_ms,
                    "context": es, **kw}
        return rc

    def _rec_plan(p):
        tr.set("join_planning", confidence=p.get("confidence"),
               max_fanout=p.get("max_fanout"),
               join_path=[f"{e['source_table']}.{e['source_column']}→"
                          f"{e['target_table']}.{e['target_column']}"
                          for e in p.get("join_path", [])],
               unreachable=p.get("unreachable") or [],
               ambiguous=[a.get("target") for a in p.get("ambiguous", [])])
        for w in p.get("why", []):
            tr.note("join_planning", w)

    # Which column a "top N"/"latest N" ranking request was actually ordered by —
    # set only on the single-table path below; stays None (and the NL-answer SLM
    # gets no ranking hint) for every other route, unchanged from before.
    _rank_column_for_nl = None

    _tp_result = run_temporal_parser(query)
    es.temporal_result = _tp_result
    tf = _tp_result.temporal_filter
    if tf and (tf.start or tf.end):
        print(f"  [L1] Temporal     {tf.start}  →  {tf.end}")
    else:
        print("  [L1] Temporal     (no date range)")

    # Intent comes from the grammar signals that actually exist (existence_mode /
    # aggregate_mode / superlative_mode below). The rule-based
    # query_engine.IntentDetector present in this tree is deliberately NOT wired in:
    # its keyword classes overlap the grammar planners, and flipping intent to
    # MULTI_TABLE/AGGREGATE here re-opens the multi-table planning latency that
    # SUPERLATIVE_JOIN_ROUTING deliberately gates off (see config.py).
    intent = "SIMPLE"
    print(f"  [L4] Intent       {intent} (grammar-derived below)")

    # Existence queries (with/without/how-many-have) are deterministic + fast, and the
    # embedding cache CAN'T tell "with" from "without" (near-identical vectors, opposite
    # SQL) — so never cache or serve them from the verified-query cache.
    is_existence = existence_mode(query) is not None
    if is_existence:
        print(f"  [L4a] Existence    semi/anti-join operator detected → {existence_mode(query)}")

    _agg, _sup, _grp, _rat = (_agg_mode(query), _sup_mode(query), _grp_mode(query),
                              _rat_mode(query))
    if _sup:
        # Routing a superlative into join planning is gated: until the grain planner
        # can actually CONSUME a superlative (group-by dim + measure), the extra
        # multi-table planning is pure latency on wide schemas (measured: 4 suite
        # queries pushed past the 120s budget). The trace still records the
        # superlative either way.
        try:
            from config import SUPERLATIVE_JOIN_ROUTING
        except Exception:
            SUPERLATIVE_JOIN_ROUTING = False
        if SUPERLATIVE_JOIN_ROUTING:
            intent = "AGGREGATE"
        print(f"  [L4] Intent       {intent} (superlative: {_sup['term']} → {_sup['superlative']})")

    # Retrieval-only intent (P1-2, 2026-09-10): `intent` above also gates multi-table
    # JOIN PLANNING (SUPERLATIVE_JOIN_ROUTING) and stays "SIMPLE" there deliberately —
    # this is a SEPARATE signal, fed only to retrieve()'s intent-aware column boosting
    # (retrieval/intent_boosting.py), which never affects planning/routing. Derived from
    # the same grammar classifiers already computed above (_agg_mode/_grp_mode/temporal
    # tf), so it costs nothing extra to compute. TEMPORAL takes priority over AGGREGATE
    # when a query names both (e.g. "revenue by month" — both a date range AND a
    # grouped measure are present; boosting the date column first is the safer default,
    # since a wrong measure choice is more visibly wrong than a wrong time bucket).
    if tf and (tf.start or tf.end):
        _retrieval_intent = "TEMPORAL"
    elif _agg or _grp:
        _retrieval_intent = "AGGREGATE"
    else:
        _retrieval_intent = intent

    _qu = dict(query=query, intent=intent,
               temporal=({"start": tf.start, "end": tf.end}
                         if tf and (tf.start or tf.end) else None),
               existence=existence_mode(query), aggregation=_agg, superlative=_sup,
               grouped=_grp, ratio=_rat)
    tr.set("query_understanding", **_qu)
    es.query_understanding = _qu
    try:  # user-safe: says WHAT was recognised, never the internal grammar dicts
        from veda import lifecycle as _lc
        _tl = _lc.current_timeline()
        if getattr(_tl, "enabled", False):
            _bits = []
            if _qu.get("temporal"):
                _bits.append("the requested time range")
            if _qu.get("aggregation", {}) and _qu["aggregation"].get("op"):
                _bits.append("the requested metric")
            if _qu.get("grouped"):
                _bits.append("how to group the results")
            # SAFE STRUCTURED FACTS for the progress UI (4-step thinking UX).
            # Only the SHAPE of the question and the period the user themselves
            # asked for — never a table, column, source or score. The api tier
            # turns these into the contextual sentence under each step, and the
            # narrator is allowed to see exactly this and nothing more.
            _facts = {}
            _agg = (_qu.get("aggregation") or {})
            if _qu.get("existence"):
                _facts["intent"] = "existence"
            elif _qu.get("superlative") or _agg.get("ranked"):
                _facts["intent"] = "ranking"
            elif _agg.get("op"):
                _facts["intent"] = str(_agg["op"]).lower()
            if _qu.get("grouped"):
                _facts["grouped"] = True
            _tf = _qu.get("temporal") or {}
            if _tf.get("start") or _tf.get("end"):
                _facts["period"] = f"{_tf.get('start') or '?'} to {_tf.get('end') or '?'}"
            _tl.completed(
                _lc.PHASE_UNDERSTANDING,
                ("Identified " + " and ".join(_bits)) if _bits else None,
                **_facts)
            # Fire-and-forget narration. Never awaited, cancelled at the terminal —
            # see veda/narrator.py on why this cannot delay the answer.
            try:
                from veda import narrator as _nar
                if _facts and on_event is not None:
                    def _deliver(_step, _sentence):
                        try:
                            on_event("narration", _sentence, {"step": _step})
                        except Exception:
                            pass
                    # `trace=tr` so the narrator thread's SLM call lands in THIS
                    # query's ledger — a fresh thread has no trace bound, so without
                    # it the call would burn latency and tokens invisibly.
                    _narration_handles.append(
                        _nar.start(_facts, "analyzing", _deliver, trace=tr))
            except Exception:
                pass
    except Exception:
        pass

    # Deterministic fast path: count / aggregate / dimension-list questions resolve
    # straight from the compiled registries — no retrieval, no planner, no LLM (and
    # they never touch get_engine(), so they're fast even on a cold process). Existence
    # already has its own deterministic path. Conservative match → falls through on miss.
    from config import FAST_PATH_ENABLED
    fp = None
    # A drill-UP replays the user's original question with the levels that REMAIN carried
    # structurally. Both deterministic short-circuits below — this fast path and the
    # grouped planner — build their SQL straight from the registries and never look at the
    # conversation, so they answer the replayed question as if nothing had been narrowed.
    # Measured 2026-09-24: one "go back" from depth 2 returned the unfiltered base answer,
    # and memory_write_node's prune (which keeps only levels still present in the frame's
    # filters) then dropped EVERY level — a single "go back" erased the whole path.
    #
    # With remembered filters in hand the turn goes the deterministic-branch route
    # instead, which applies them and rebuilds the remembered GROUP BY around them.
    # `_conv_filters` is empty for every non-chat caller and every first turn, so this is
    # byte-identical outside a live drill.
    if FAST_PATH_ENABLED and not is_existence and not _reentry and _conv_filters:
        print(f"  [conversation] {len(_conv_filters)} remembered filter(s) still apply — "
              f"skipping the fast path, which cannot carry them")
    elif FAST_PATH_ENABLED and not is_existence and not _reentry:   # re-entry: fast path off (M2)
        try:
            fp = try_fast_path(query, tf)
        except Exception as _fpe:
            print(f"  [FastPath] warning: {_fpe} — falling through")
            fp = None

    # Deterministic superlative-by-dimension planner (QSR-backed, Phase B): grouped
    # ranked aggregation straight from resolution artifacts — no retrieval, no LLM.
    # May also return a grounded clarify (ambiguous dimension/measure listed).
    if fp is None and not is_existence and _sup:
        try:
            from config import SUPERLATIVE_PLAN_ENABLED
        except Exception:
            SUPERLATIVE_PLAN_ENABLED = False
        if SUPERLATIVE_PLAN_ENABLED:
            try:
                from query.superlative_plan import try_superlative_plan
                _sp = try_superlative_plan(query, sm)
            except Exception as _spe:
                print(f"  [SupPlan] warning: {_spe} — falling through")
                _sp = None
            if isinstance(_sp, tuple) and _sp and _sp[0] == "clarify":
                fb = _feedback("clarify", msg=_sp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_sp[1], feedback=fb)
            if _sp is not None:
                fp = _sp

    # Deterministic grouped-breakdown planner (same QSR machinery, non-ranked
    # sibling): "how much does each <dim> contribute" → GROUP BY dim, SUM(measure).
    # Same clarify/fall-through contract as the superlative planner above.
    # A drill-UP replays the user's ORIGINAL question ("what is the distribution of
    # properties by furnishing?") with the levels that remain carried structurally. That
    # text is a grouped question, so the grouped planner below would answer it — and it
    # builds its SQL directly, short-circuiting the deterministic section that is the only
    # place remembered filters are applied. Measured 2026-09-24: one "go back" from depth 2
    # therefore returned the UNFILTERED base answer, and memory_write_node's stack prune
    # (which keeps only levels still present in the frame's filters) then dropped every
    # level — so a single "go back" erased the whole path instead of one step of it.
    #
    # When the conversation carries filters, the turn is routed to the deterministic branch
    # instead, which applies them AND rebuilds the remembered GROUP BY around them. With no
    # conversation context this is byte-identical: `_conv_filters` is empty for every
    # non-chat caller and for every first turn.
    if _conv_filters and fp is None and not is_existence and (_grp or _grpc_mode(query)):
        print(f"  [conversation] {len(_conv_filters)} remembered filter(s) still apply — "
              f"not taking the grouped fast path, which cannot carry them")
    elif fp is None and not is_existence and (_grp or _grpc_mode(query)):
        try:
            from config import GROUPED_PLAN_ENABLED
        except Exception:
            GROUPED_PLAN_ENABLED = False
        if GROUPED_PLAN_ENABLED:
            try:
                from query.superlative_plan import try_grouped_plan
                _gp = try_grouped_plan(query, sm)
            except Exception as _gpe:
                print(f"  [GrpPlan] warning: {_gpe} — falling through")
                _gp = None
            if isinstance(_gp, tuple) and _gp and _gp[0] == "clarify":
                fb = _feedback("clarify", msg=_gp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_gp[1], feedback=fb)
            if _gp is not None:
                fp = _gp

    # Deterministic ratio planner: "ratio of X to Y" → single-scan divided sums
    # on the measure-owning anchor; ungroundable side → grounded clarify with the
    # anchor's real value domain (terminal — never retried by Tier-2).
    if fp is None and not is_existence and _rat:
        try:
            from config import RATIO_PLAN_ENABLED
        except Exception:
            RATIO_PLAN_ENABLED = False
        if RATIO_PLAN_ENABLED:
            try:
                from query.ratio_plan import try_ratio_plan
                _rp = try_ratio_plan(query, sm)
            except Exception as _rpe:
                print(f"  [RatioPlan] warning: {_rpe} — falling through")
                _rp = None
            if isinstance(_rp, tuple) and _rp and _rp[0] == "clarify":
                fb = _feedback("clarify", msg=_rp[1])
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_rp[1], feedback=fb)
            if _rp is not None:
                fp = _rp

    # FAST-PATH EVIDENCE GUARD: the fast path bypasses anchor vetting, so an answer
    # built entirely on tables the query gives NO typed evidence for (no entity
    # word, no value, no closure) is the wrong-pick signature — "annual sum of
    # financial records…" answered from users_userpreference via the
    # financial_year_id accident. DEMOTE, don't refuse: fall through to the full
    # pipeline, which has anchor vetting and its own gates. (Measured on the golden
    # baseline: refusing here flipped good answers on descriptor words; demotion
    # only costs latency on the rare zero-evidence picks.)
    if fp is not None and not isinstance(fp, dict):
        try:
            from config import FASTPATH_EVIDENCE_GUARD, QSR_FP_EVIDENCE_FLOOR
            if FASTPATH_EVIDENCE_GUARD and fp.tables:
                from query.resolution import typed_anchor_evidence
                _ev, _ = typed_anchor_evidence(query, sm)
                if not any(_ev.get(t, 0.0) >= QSR_FP_EVIDENCE_FLOOR for t in fp.tables):
                    # A pick deterministically GROUNDED from a business noun the retrieval-
                    # based evidence probe can't see ("property"→assets_asset via the alias
                    # glossary / collapsed-token concatenation) carries its OWN typed
                    # evidence — the grounding itself. Don't demote it for the very blindness
                    # it was built to fix. Flag-gated: when OFF the fallback returns None, so
                    # this is a no-op and the guard is byte-identical.
                    _grounded_ok = False
                    try:
                        from config import FASTPATH_ENTITY_GLOSSARY
                        if FASTPATH_ENTITY_GLOSSARY:
                            from query.fast_path import _grounded_entity_fallback
                            from semantic import registry as _greg
                            _ge = _grounded_entity_fallback(query, _greg.query_tokens(query))
                            if _ge and _ge.get("resolves_to", {}).get("table") in fp.tables:
                                _grounded_ok = True
                    except Exception:
                        _grounded_ok = False
                    # A metric-derived pick (route "metric.*") is grounded by the METRIC-LABEL match
                    # itself: match_metric_labels only returns a metric whose labels (built from that
                    # column/table) the query named, so the pick carries its own typed evidence — the
                    # metric grounding. The retrieval evidence probe can't see it because the user
                    # named the MEASURE ("average rent"), not the table ENTITY, so it wrongly reads as
                    # zero-evidence. Don't demote a metric pick for the blindness it was built to fix.
                    # Flag-gated: OFF -> this exemption is skipped, byte-identical to the prior guard.
                    if not _grounded_ok:
                        try:
                            from config import AGG_VERB_COMPLETENESS_ENABLED as _agg_ex
                            if _agg_ex and str(getattr(fp, "route", "")).startswith("metric."):
                                _grounded_ok = True
                        except Exception:
                            pass
                    if not _grounded_ok:
                        # Column-name evidence (M1 close-out, 2026-09-15): typed_anchor_
                        # evidence counts VALUE and ENTITY(table-name) evidence only — a
                        # question that names the plan's own COLUMNS verbatim ("average
                        # MONTHLY FEE per CATEGORY" → amenities_catalog.monthly_fee /
                        # .category) scored 0 and a correct deterministic grouped plan was
                        # demoted into a row list (then refused). Schema-driven: the plan's
                        # own columns' name tokens, no vocabulary. One fully-named column
                        # on the picked table is anchoring evidence.
                        try:
                            _qtoks = set(re.findall(r"[a-z0-9]+", query.lower()))
                            for _pc in (getattr(fp, "columns", None) or []):
                                _ctoks = [t for t in str(_pc).lower().split(".")[-1].split("_") if len(t) > 2]
                                if _ctoks and all(t in _qtoks for t in _ctoks):
                                    _grounded_ok = True
                                    tr.note("schema_linking",
                                            f"fast-path pick kept: column '{_pc}' named in the query")
                                    break
                        except Exception:
                            pass
                    if not _grounded_ok:
                        print(f"  [FastPath] demoted: no typed evidence for "
                              f"{sorted(fp.tables)[:3]} — full pipeline")
                        tr.note("schema_linking",
                                f"fast-path pick {sorted(fp.tables)[:3]} demoted (zero typed evidence)")
                        fp = None
        except Exception:
            pass

    # cache_back=False (RequestContext, 2026-09-16): the request opted out of the
    # verified-query cache — no replay here, no write at the end (eval/battery traffic).
    _cache_back = getattr(_ambient_ctx(), "cache_back", True)
    # A context-dependent turn is not looked up either, for the same reason it is not
    # saved (see the save site below): its words do not identify its question.
    _context_dependent_turn = bool(_conv.get("entity_table"))
    cached_sql, sim = ((None, 0.0)
                       if (is_existence or fp or not _cache_back or _context_dependent_turn)
                       else verified_cache_lookup(query))
    # Same evidence guard for the CACHED lane — the fourth answer-producing lane,
    # which replays SQL verified under OLDER code: a cached answer whose tables get
    # zero typed evidence from the query is a stale wrong pick → recompute.
    if cached_sql:
        try:
            from config import FASTPATH_EVIDENCE_GUARD, QSR_FP_EVIDENCE_FLOOR
            if FASTPATH_EVIDENCE_GUARD:
                from query.resolution import typed_anchor_evidence
                _ct = set(re.findall(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)',
                                     cached_sql))
                _ev, _ = typed_anchor_evidence(query, sm)
                if _ct and not any(_ev.get(t, 0.0) >= QSR_FP_EVIDENCE_FLOOR for t in _ct):
                    print(f"  [cache] demoted: no typed evidence for cached tables "
                          f"{sorted(_ct)[:3]} — recompute")
                    tr.note("schema_linking", "verified-cache hit demoted (zero typed evidence)")
                    cached_sql = None
        except Exception:
            pass
    if cached_sql:
        # QUALIFIER re-check (distinct from the table-level evidence guard above):
        # the evidence guard only asks "is THIS query plausibly about the cached
        # SQL's table(s)" — it can't catch a same-table cache entry whose WHERE
        # clause answers a DIFFERENT question (found in production: a cached
        # "properties in the UAE" answer replayed verbatim for "properties priced
        # above 10,000" — same table, unrelated filter, similarity ≥0.85 anyway).
        # Reuses the SAME gate the main pipeline already applies to freshly-built
        # SQL (below, ~line 1090) — a cache hit must clear the identical bar a
        # fresh answer would, not a lesser one just because it was pre-verified
        # once under possibly-older code.
        try:
            from veda.firewall import qualifier_only as _fw_qual
            from veda.ir import partial as _ir_partial_c
            ok_cache_q = _fw_qual(_ir_partial_c("cache"), cached_sql, sm, query=query,
                                  user_message=_conv_user_message)
            missing_cache_q = None if ok_cache_q else qualifier_completeness(
                query, cached_sql, sm, user_message=_conv_user_message)[1]
            if not ok_cache_q:
                print(f"  [cache] demoted: cached SQL drops qualifier {missing_cache_q!r} "
                      f"for THIS query — recompute")
                tr.note("schema_linking",
                        f"verified-cache hit demoted (dropped qualifier {missing_cache_q!r})")
                cached_sql = None
        except Exception:
            pass
    if cached_sql:
        # SHAPE re-check (2026-09-15) — the third demotion, distinct from both above: the
        # evidence guard asks "right table?", the qualifier guard asks "every named
        # qualifier present?"; neither asks "same ANSWER SHAPE?". Found live on source 4
        # right after its re-ingest: "how many maintenance records PER VENDOR" hit the
        # cached "how many maintenance records are there" (sim=0.88) and replayed its
        # scalar COUNT(*) — one number for a question that asked for a breakdown, and
        # the qualifier gate let it through because 'vendor' loosely matches the
        # vendors table. A grouping phrase in the question with no GROUP BY in the cached
        # SQL (or a GROUP BY the question never asked for), or scalar-aggregate wording
        # with an aggregate-less cached SQL, is a different question → recompute. Same
        # grammar list every other grouping check uses; same aggregate gate the fresh
        # path runs (intent_sql_alignment.aggregate_presence_ok).
        try:
            from config import QUERY_GRAMMAR as _QG_cache
            from veda.intent_sql_alignment import aggregate_presence_ok as _agg_ok_cache
            _qlc = f" {query.lower()} "
            _q_grouped = any((" " in w and w in _qlc) or re.search(rf"\b{re.escape(w)}\b", _qlc)
                             for w in _QG_cache.get("grouping", []))
            _sql_grouped = bool(re.search(r"\bGROUP\s+BY\b", cached_sql, re.I))
            _shape_why = None
            if _q_grouped != _sql_grouped:
                _shape_why = ("question asks for a breakdown, cached SQL has no GROUP BY"
                              if _q_grouped else
                              "cached SQL groups, question asks for no breakdown")
            else:
                _ok_agg_c, _ = _agg_ok_cache(query, cached_sql, sm)
                if not _ok_agg_c:
                    _shape_why = "question asks for a figure, cached SQL has no aggregate"
            if not _shape_why:
                # LIMIT N is part of the answer SHAPE too: "top 3 vendors by rating" hit the
                # cached "which vendor has the highest rating" (sim=0.86) and replayed its
                # LIMIT 1 — a silently wrong N. Same guard, one more shape attribute.
                _mtop = re.search(r"\b(?:top|latest|last|bottom|first)\s+(\d+)\b", _qlc)
                _mlim = re.search(r"\bLIMIT\s+(\d+)\b", cached_sql, re.I)
                if _mtop and (not _mlim or _mlim.group(1) != _mtop.group(1)):
                    _shape_why = (f"question asks for top {_mtop.group(1)}, cached SQL has "
                                  f"LIMIT {_mlim.group(1) if _mlim else 'none'}")
            if not _shape_why:
                # RANKING is a shape attribute too, same principle: an entry verified for a
                # query that named no count is replayed for one that DOES ("top 5"), and a
                # similarity of 0.97 does not notice that the cached statement carries
                # `LIMIT 10` and no ORDER BY. Demote rather than refuse outright —
                # recomputing gives the deterministic path its own chance to build a
                # properly ordered statement; if it cannot, the universal shape guard
                # below refuses there.
                ok_cache_r, why_cache_r = ranked_shape_ok(query, cached_sql)
                if not ok_cache_r:
                    # why_cache_r is written for the USER (it describes the statement THIS
                    # query would get); here it only labels the demotion, so keep it
                    # schematic — the cached statement's own LIMIT may differ from the
                    # one asked for.
                    _shape_why = ("cached SQL does not honor this query's ranking "
                                  f"({why_cache_r.split(', but')[0]})")
            if _shape_why:
                print(f"  [cache] demoted: {_shape_why} — recompute")
                tr.note("schema_linking", f"verified-cache hit demoted (shape: {_shape_why})")
                cached_sql = None
        except Exception:
            pass
    if fp:
        print(f"  [FastPath] {fp.route}  ({'; '.join(fp.why)})  — no retrieval / no LLM")
        # The two SHORTEST lanes (fast path, cache replay) answer without touching the
        # planner, which is where every other _tick() lives — so their timeline arrived
        # at the UI with a single "output" entry and the progress feature looked dead
        # for exactly the queries that reach the user fastest.
        _tick("sql_planning", "Answering this one directly")
        sql, primary, from_cache = fp.sql, fp.primary, False
        allowed_tables, allowed_columns = set(fp.tables), list(fp.columns)
    elif cached_sql:
        print(f"  [cache] verified-query hit (sim={sim:.2f}) — skipping retrieval + SLM")
        _tick("sql_planning", "Reusing a query already verified for this question")
        sql, from_cache = cached_sql, True
        _lane["cache"] = True
        # Recorded in the trace as well as the private result key, because the
        # USER-facing payload is projected only from the trace. A cached answer
        # replays a query verified under possibly-OLDER code, so "where did this
        # answer come from" is a fact the person reading it is entitled to.
        try:
            tr.set("execution", from_cache=True)
        except Exception:
            pass
        import sqlglot
        from sqlglot import exp
        try:
            ct = sqlglot.parse_one(sql, read="postgres")
            allowed_tables = {t.name for t in ct.find_all(exp.Table) if t.name}
        except Exception:
            allowed_tables = set()
        # The real table name, derived from the cached SQL text itself — NOT the
        # literal string "(cached)" this used to be (a display-only leftover that
        # ended up as engine_result["table"], then as the QueryFrame's "entity"
        # via harvest_frame(), poisoning memory/topic-switch detection on every
        # cache-hit turn). Multi-table cached query: pick deterministically
        # (first alphabetically) rather than guess — never crashes downstream,
        # which only special-cases an empty/unknown primary already.
        # The anchor is the table the statement selects FROM, which the SQL states
        # outright — not a tie to be broken alphabetically. Measured 2026-09-22 on a
        # drill-down conversation: "what about Mumbai" produced
        #   FROM "assets_asset" t0 JOIN generics_country t1
        #                          JOIN users_useraddress t2
        #                          JOIN accounts_generalledger t3
        # and sorted() put accounts_generalledger first, so THAT was reported as the
        # table. harvest_frame takes the QueryFrame's entity from this field, so the
        # conversation silently relocated to a financial-ledger topic and every later
        # turn ("only the residential ones", "go back", "remove the city filter") asked
        # about the wrong table and came back asking for clarification. Alphabetical
        # order is stable, which is what the note above wanted, but stability is
        # worthless when the value is wrong.
        primary = ""
        try:
            _from = ct.find(exp.From)
            _tbl = _from.find(exp.Table) if _from is not None else None
            primary = (_tbl.name or "") if _tbl is not None else ""
        except Exception:
            primary = ""
        if not primary:
            # Unparseable FROM (or sqlglot already failed above) — keep the previous
            # deterministic choice rather than report nothing.
            primary = (next(iter(allowed_tables)) if len(allowed_tables) == 1
                       else (sorted(allowed_tables)[0] if allowed_tables else ""))
        allowed_columns = [k.split(".", 1)[1] for k in all_cols
                           if k.split(".", 1)[0] in allowed_tables]
    else:
        from config import QUERY_ENHANCEMENT_ENABLED
        enh = None
        if QUERY_ENHANCEMENT_ENABLED:
            try:
                from veda.query_enhancement import enhance_query
                enh = enhance_query(query, sm)
            except Exception:
                enh = None
        _search = enh.search_query if enh else query
        tr.set("query_understanding", enhancement=(enh.to_dict() if enh else None))
        if enh and _search != query:
            print(f"  [L2+] Enhance      +{len(enh.search_terms) + len(enh.expanded_aliases)} "
                  f"search terms  ({'; '.join(enh.enhancement_trace[:3])})")
        print("  [L2] Retrieval     6-signal (BGE-M3 dense+sparse + FK subgraph/path + "
              "value + table-prior) → weighted RRF")
        try:
            from config import RETRIEVAL_CACHE_ENABLED as _RC
        except Exception:
            _RC = False
        # Pass THIS (source, tenant)'s semantic model so the engine for this scope is built
        # from the right source's BM25/signals (P5 multi-source); Signal-1 store is source-scoped.
        # `_retrieval_intent` (P1-2), not `intent` — see its computation above for why they
        # deliberately differ.
        results = get_engine(sm).retrieve(query=_search, intent=_retrieval_intent, top_k=15, use_cache=_RC)

        # ── Unified-graph recall booster (Phase 4): ADD columns the 5-signal engine may
        # have missed, via synonym/alias resolution + FK-neighbour reach. Purely additive
        # (the cross-encoder rerank below re-scores everything), flag-guarded, and fully
        # try/except'd → on ANY failure retrieval is byte-identical to before. col_id here
        # is the "table.col" string the engine already uses, so no UUID lookup is needed.
        try:
            from config import GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX
        except Exception:
            GRAPH_EXPAND_ENABLED, GRAPH_EXPAND_MAX = False, 12
        if GRAPH_EXPAND_ENABLED and results is not None:
            try:
                from graph.query_graph import suggest_expansions
                from retrieval.retrieval_engine_phase3 import RetrievalResult as _RR
                _have_cols = {r.col_id for r in results}
                _have_tabs = {r.table_name for r in results}
                _seeds, _added, _syn = suggest_expansions(
                    query, _have_cols, _have_tabs, max_add=GRAPH_EXPAND_MAX)
                # Source isolation (marker-gated, default OFF): suggest_expansions reaches over the
                # GLOBAL unified FK/synonym graph, so on an isolated single-source sm it re-admits
                # other sources' tables (a datalake "vendors" pulls homzhub worklists_quote/reviews_*).
                # Drop additions outside this sm's tables. No marker (normal path) → untouched.
                if sm.get("__source_isolated__"):
                    _iso_tabs = set((sm.get("tables") or {}).keys())
                    _added = [n for n in _added if n.split(".", 1)[0] in _iso_tabs]
                for _name in _added:
                    _tt, _cc = _name.split(".", 1)
                    results.append(_RR(col_id=_name, column_name=_cc,
                                       table_name=_tt, final_score=0.0))
                if _added:
                    print(f"  [L2g] Graph expand  +{len(_added)} cols "
                          f"(seeds={_seeds[:4]}): {_added[:5]}")
                tr.set("graph_expansion", seeds=_seeds, synonyms=_syn, added=_added)
            except Exception as _ge:
                print(f"  [L2g] graph expand skipped: {type(_ge).__name__}: {str(_ge)[:80]}")

        # ── Gate 1 (User Story 3, Task 16) RBAC candidate filter. Applied to the
        # per-request CANDIDATE LIST only — NEVER to `sm` before get_engine(sm)
        # above, which would bake this request's permissions into the shared,
        # per-scope-cached retrieval engine (see veda.rbac_filter's module
        # docstring). A no-op when the ambient context carries no
        # allowed_resources (RBAC off, staff, or a pre-Gate-1 caller).
        _before_rbac = len(results) if results else 0
        results = filter_retrieval_results(results, sm, _ambient_ctx())
        # Fail-closed SCOPE filter (M1 close-out, 2026-09-15), independent of RBAC: a
        # candidate whose table is not in THIS scope's semantic model cannot be planned
        # against, whichever retrieval signal produced it. Found live: a source-5
        # (parquet, one table) query retrieved `maintenance` (source 4) and homzhub
        # tables, and routing anchored on a table the source doesn't have. Whatever
        # signal leaks is a bug to fix on its own; this guard makes the leak harmless
        # in the meantime — the same "never plan against another source's schema"
        # contract the artifact resolver now enforces. Multi-source scopes carry
        # `src{ID}.<table>` keys for collisions; both spellings are admitted.
        if results and _ambient_ctx() is not None:
            _scope_tables = set((sm.get("tables") or {}).keys())
            _scope_tables |= {k.split(".", 1)[1] for k in _scope_tables if k.startswith("src") and "." in k}
            _before = len(results)
            results = [r for r in results if getattr(r, "table_name", "") in _scope_tables]
            if len(results) != _before:
                print(f"  [L2s] Scope filter  dropped {_before - len(results)} candidate(s) "
                      f"from tables outside this source's model")
                tr.note("retrieval", f"scope filter dropped {_before - len(results)} foreign candidates")
        if results is not None and len(results) != _before_rbac:
            tr.set("rbac_filter", before=_before_rbac, after=len(results))
            # Part 3: RBAC narrowing was SILENT before this — a user could get a
            # quietly narrower answer with no indication anything was withheld.
            #
            # Raised HERE, and only when the count actually CHANGED, because that is
            # the one honest signal available: candidates this query's retrieval
            # considered relevant were removed for access reasons. Merely running the
            # filter is not newsworthy; removing a relevant candidate is.
            #
            # The warning states only THAT data was excluded — never which table,
            # column or source, since naming it would be exactly the disclosure the
            # restriction exists to prevent. `before`/`after` counts stay in the
            # internal trace and are NOT projected (veda/safe_projection.py has no
            # reader for the rbac_filter section) because the COUNT of hidden things
            # is itself a disclosure.
            try:
                from veda import warnings as _vw
                _vw.add(_vw.RESTRICTED_DATA)
            except Exception:
                pass

        # ── PRIMARY cross-encoder rerank (Step 2): the precision ranker now runs on the
        # PRIMARY path (not only Tier-2). Reorders candidates + updates final_score so anchor
        # selection ranks off reranked scores — directly tightening the near-tie RRF margins
        # that caused mis-anchoring. Generic: reranker no longer carries a hardcoded business
        # map (it uses the generated domain_synonyms). Graceful: any failure keeps RRF order.
        try:
            from config import (PRIMARY_RERANK_ENABLED, RERANKER_BATCH_SIZE,
                                 RERANK_SKIP_GAP, RERANK_MAX_CANDIDATES, RERANKER_MAX_TEXT_LEN)
        except Exception:
            PRIMARY_RERANK_ENABLED = False

        def _rrf_gap_unambiguous(_results) -> bool:
            """True when candidate #1 clearly leads #2 AND both are the same table —
            reranking would not change the anchor, so skip it (F4)."""
            if len(_results) < 2:
                return True
            s0, s1 = _results[0].final_score, _results[1].final_score
            same_table = _results[0].col_id.split(".")[0] == _results[1].col_id.split(".")[0]
            return same_table and (s0 - s1) >= RERANK_SKIP_GAP

        _rk_before = _rk_after = None   # top-5 col_ids around the rerank (trace only)
        if PRIMARY_RERANK_ENABLED and results and not _rrf_gap_unambiguous(results):
            try:
                from query.reranker import _get_reranker, _precomputed_rerank_text
                _rk = _get_reranker()
                if _rk is not None:
                    # F4: cap candidate width — the tail never wins anchor selection.
                    _head = results[:RERANK_MAX_CANDIDATES]
                    _tail = results[RERANK_MAX_CANDIDATES:]
                    # Same enriched cross-encoder text query/reranker.py's own _col_text()
                    # uses (business definition/aliases/role/etc., precomputed at ingestion,
                    # WP7) — not bare column_name+table_name. This is the SAME model as
                    # rerank_columns()/rerank_tables(); it was just seeing less context here
                    # than at that other call site. Falls back to the bare name pair when no
                    # precomputed doc exists for a column (identical fallback _col_text uses).
                    _pairs = [
                        [_search, (_precomputed_rerank_text(r.col_id, is_table=False)
                                   or f"{r.column_name} {r.table_name}")[:RERANKER_MAX_TEXT_LEN]]
                        for r in _head
                    ]
                    _enriched_n = sum(1 for r in _head
                                      if _precomputed_rerank_text(r.col_id, is_table=False) is not None)
                    print(f"  [L2b] Enriched rerank input: {_enriched_n}/{len(_head)} candidates "
                          f"used precomputed metadata, {len(_head) - _enriched_n} fell back to bare name")
                    _sc = _rk.predict(_pairs, batch_size=RERANKER_BATCH_SIZE)
                    # NOISE FLOOR: the cross-encoder's output is calibrated (sigmoid) — when
                    # its BEST pair is near zero it is affirmatively saying NO candidate is
                    # relevant to this query. Overwriting final_score then replaces the RRF
                    # consensus (BM25+embedding+graph) with pure noise that downstream anchor
                    # normalization stretches to 1.0 — mis-anchoring on garbage. Keep the RRF
                    # order instead; the floor is in the model's own output space, no schema
                    # or vocabulary assumption.
                    try:
                        from config import RERANK_NOISE_FLOOR
                    except Exception:
                        RERANK_NOISE_FLOOR = 0.0
                    _smax = max((float(s) for s in _sc), default=0.0)
                    if _smax < RERANK_NOISE_FLOOR:
                        print(f"  [L2b] Primary rerank UNINFORMATIVE (max {_smax:.5f} < "
                              f"{RERANK_NOISE_FLOOR}) — keeping RRF order")
                        tr.set("reranking", input_candidate_count=len(_head),
                               reranker_skipped=True, skip_reason="noise_floor",
                               score_max=round(_smax, 5), noise_floor=RERANK_NOISE_FLOOR)
                    else:
                        _rk_before = [r.col_id for r in results[:5]]   # pre-rerank order (trace)
                        _ranked = sorted(zip(_sc, _head), key=lambda x: float(x[0]), reverse=True)
                        for _s, _r in _ranked:
                            _r.cross_encoder_score = float(_s)   # keep the CE score visible (trace)
                            _r.final_score = float(_s)   # anchor reads final_score → now reranked
                        # SCALE GUARD (H-0): reranked head carries cross-encoder scores, the tail
                        # keeps RRF scores — incomparable, so floor the tail below the head to keep
                        # it from hijacking anchor selection. (Verified NOT the count-for-sale
                        # regression culprit; the anchor ambiguity there is pre-existing.)
                        if _ranked and _tail:
                            _floor = min(float(_s) for _s, _ in _ranked)
                            for _i, _r in enumerate(_tail):
                                _r.final_score = _floor - 1.0 - _i * 1e-6
                        results = [_r for _, _r in _ranked] + _tail
                        _rk_after = [r.col_id for r in results[:5]]   # post-rerank order (trace)
                        print(f"  [L2b] Primary rerank (cross-encoder, top {RERANK_MAX_CANDIDATES}) → top: {results[0].col_id}")
                        # RERANK observability — enough to diagnose noisy/compressed
                        # reranker scores (score spread + top1/top2 gap). Uses the _sc
                        # already predicted above; no re-scoring.
                        try:
                            _scores = sorted((float(s) for s in _sc), reverse=True)
                            _n = len(_scores)
                            tr.set("reranking",
                                   input_candidate_count=len(_head),
                                   output_candidate_count=len(results),
                                   reranker_skipped=False,
                                   score_min=round(_scores[-1], 5) if _n else None,
                                   score_max=round(_scores[0], 5) if _n else None,
                                   score_mean=round(sum(_scores) / _n, 5) if _n else None,
                                   top1_top2_gap=round(_scores[0] - _scores[1], 5) if _n >= 2 else None)
                            tr.cand("reranking", "top_before", _rk_before)
                            tr.cand("reranking", "top_after", _rk_after)
                        except Exception:
                            pass
            except Exception as _rr_e:
                print(f"  [L2b] primary rerank skipped: {type(_rr_e).__name__}: {str(_rr_e)[:100]}")
        elif PRIMARY_RERANK_ENABLED and results:
            print(f"  [L2b] Primary rerank SKIPPED (unambiguous RRF gap) → top: {results[0].col_id}")
            tr.set("reranking", reranker_skipped=True, skip_reason="rrf_gap_unambiguous")

        _cand_tabs = []
        for r in results:
            _t = r.col_id.split(".")[0]
            if _t not in _cand_tabs:
                _cand_tabs.append(_t)
        _router_primary = select_primary_table(results, query, sm, trace=tr)
        _er = None
        _analytical_sql = None          # Phase 1 ANALYTICAL_SQL_V2 (set in understanding block)
        _analytical_primary = None
        _analytical_spec = None         # M2: the spec itself (list specs carry only where_sql)
        _analytical_used = False        # M2: True once the grounded candidate IS the SQL being validated
        _analytical_reentry_ok = False  # pre-M3: anchor grounded by NAME (not retrieval) → may lead on re-entry
        # ── Query-Understanding layer (flag-gated, default OFF) — enterprise ─────
        # LLM extracts typed intent → deterministic grounding validates to REAL tables
        # → adapted into a ResolvedEntities so the EXISTING ER plumbing (primary-pin,
        # _er_multi gate, build_from_entities) consumes it unchanged. Refusal → grounded
        # clarify/refuse (never a guess). None → fall through to ER-V1/existing (degrade).
        try:
            from config import QUERY_UNDERSTANDING_ENABLED as _QU_ON
        except Exception:
            _QU_ON = False
        if _QU_ON:
            try:
                from veda.understanding import (understand_query,
                                                GroundedIntent as _GI, Refusal as _RF)
                _rscore = {}
                for _r in results:
                    _t = _r.col_id.split(".")[0]
                    _rscore[_t] = max(_rscore.get(_t, 0.0), getattr(_r, "final_score", 0.0))
                _u = understand_query(query, sm, retrieval_scores=_rscore, tf=tf)
                try:
                    from config import QUERY_UNDERSTANDING_MIN_CONFIDENCE as _qu_minc
                except Exception:
                    _qu_minc = 0.5
                if isinstance(_u, _RF):
                    # M2 (2026-09-16): a Refusal is ADVISORY, never terminal. The
                    # understanding layer is a strictly weaker signal than the deterministic
                    # branches below (§5 regressions: "show all vendors" refused by a layer
                    # that couldn't ground a table the router had). Record it, keep going;
                    # the existing path decides. (Rule: a new decision layer never gates a
                    # strictly more capable path on a strictly weaker signal.)
                    _st = "clarify" if _u.reason == "ambiguous" else "refuse"
                    tr.set("understanding", decision=_st, reason=_u.reason,
                           unresolved=_u.unresolved, advisory=True)
                    print(f"  [QU] advisory {_st}: {_u.message[:100]} — existing path decides")
                    _u = None
                if isinstance(_u, _GI) and _u.anchor and not (
                        _u.fully_grounded and (_u.confidence or 0.0) >= _qu_minc):
                    # M2: a PARTIALLY grounded intent (an entity that didn't ground, or
                    # low confidence) is a candidate for the trace only — it must not
                    # override the router's anchor / the ER pin.
                    tr.set("understanding", decision="advisory", anchor=_u.anchor,
                           unresolved=(_u.evidence or {}).get("unresolved_nonfatal"),
                           confidence=_u.confidence)
                    print(f"  [QU] advisory: anchor={_u.anchor} not fully grounded — not pinned")
                    _u = None
                if isinstance(_u, _GI) and _u.anchor:
                    # M2 (2026-09-16): a grounded intent is a CANDIDATE, never a pin. It
                    # used to become a RESOLVED ResolvedEntities that bypassed vet_primary
                    # and the ER path — "show all vendors" then planned on whatever table
                    # the LLM's grain grounded to (§5 regression). Now it only contributes
                    # deterministic analytical SQL, and that SQL is used only when its
                    # anchor IS the router's primary and no deterministic branch matched
                    # (checked where the candidate is consumed below).
                    tr.set("understanding", decision="candidate", anchor=_u.anchor,
                           secondaries=_u.secondaries, intent=_u.intent,
                           confidence=_u.confidence)
                    # ── Phase 1: ANALYTICAL_SQL_V2 (flag-gated) ─────────────────
                    # Single-anchor analytical query → deterministic structured SQL that
                    # CONSUMES the spec (never re-infers the aggregation from language,
                    # the cause of raw-row-list output). None → existing path unchanged.
                    try:
                        from config import ANALYTICAL_SQL_V2 as _asql_on
                    except Exception:
                        _asql_on = False
                    if _asql_on:
                        try:
                            from veda.analytical_spec import derive_spec, emit_sql
                            _aspec = derive_spec(_u, query, sm)
                            _analytical_spec = _aspec
                            _analytical_reentry_ok = bool(getattr(_u, "reentry_eligible", False))
                            tr.set("understanding", anchor_method=getattr(_u, "anchor_method", None),
                                   reentry_eligible=_analytical_reentry_ok)
                            _cand = emit_sql(_aspec, sm) if _aspec else None
                            if _aspec is not None and _aspec.aggregation == "list":
                                _analytical_primary = _u.anchor
                                tr.set("analytical_sql_v2", used=True, anchor=_u.anchor,
                                       aggregation="list", where=_aspec.where_sql)
                                print(f"  [AnalyticalV2] list on {_u.anchor} WHERE {_aspec.where_sql}"
                                      f" — grounded predicate, projection from the pipeline")
                            if _cand:
                                _analytical_sql = _cand
                                _analytical_primary = _u.anchor
                                tr.set("analytical_sql_v2", used=True, anchor=_u.anchor,
                                       aggregation=_aspec.aggregation,
                                       shape=_aspec.output_shape, group_keys=_aspec.group_keys)
                                print(f"  [AnalyticalV2] {_aspec.aggregation} "
                                      f"{_aspec.output_shape} on {_u.anchor} — deterministic SQL")
                        except Exception as _ae:
                            print(f"  [AnalyticalV2] skipped: {type(_ae).__name__}")
            except Exception as _ue:
                print(f"  [QU] understanding skipped: {type(_ue).__name__}: {str(_ue)[:120]}")
                _er = None
        # ── Entity Resolution V1 (flag-gated, default OFF) ──────────────────────
        # Fuse existing name-coverage evidence into a confidence-gated canonical anchor.
        # RESOLVED + pin-eligible → PIN the primary (bypass vet_primary, the proven
        # override point). AMBIGUOUS / UNGROUNDED / not pin-eligible → existing
        # vet_primary path UNCHANGED (zero-risk fallback). Skipped when the understanding
        # layer already produced a grounded anchor above.
        try:
            from config import ENTITY_RESOLUTION_V1 as _ER_ON
        except Exception:
            _ER_ON = False
        if _ER_ON and _er is None:
            try:
                from query.entity_resolver import resolve_entities
                _er = resolve_entities(query, results, sm, all_cols)
                tr.set("entity_resolution", status=_er.status, anchor=_er.anchor,
                       secondaries=_er.secondaries, confidence=_er.confidence,
                       distinct_tables=_er.distinct_tables,
                       anchor_coverage=_er.evidence.get("anchor_coverage"),
                       margin=_er.evidence.get("margin"),
                       pin_eligible=_er.evidence.get("pin_eligible"))
                for _c in (_er.evidence.get("candidates") or []):
                    tr.cand("entity_resolution", "candidates", _c)
                es.resolved_anchor = _er.anchor
                es.resolved_secondaries = list(_er.secondaries)
                es.entity_resolution_status = _er.status
                es.entity_resolution_confidence = _er.confidence
            except Exception as _ere:
                print(f"  [ER] entity resolution skipped: {type(_ere).__name__}: {str(_ere)[:120]}")
                _er = None
        # ── RC3: grounded clarification (flag-gated, default OFF) ───────────────
        # AMBIGUOUS = two candidates tied for the SAME entity slot. Ask the user
        # which one instead of coin-flipping into a confident wrong answer. Scoped
        # to AMBIGUOUS only (UNGROUNDED keeps the retrieval fallback) so answerable
        # single-table queries are never regressed.
        if _er is not None and _er.status == "AMBIGUOUS":
            try:
                from config import ER_GROUNDED_REFUSAL as _ER_REFUSE
            except Exception:
                _ER_REFUSE = False
            if _ER_REFUSE:
                _cands = _er.evidence.get("candidates") or []
                _opts = [c.get("master") or c.get("table") for c in _cands[:2]]
                _opts = [o for o in _opts if o]
                if _opts:
                    _cmsg = ("This question is ambiguous — it could refer to "
                             + " or ".join(repr(o) for o in _opts)
                             + ". Please specify which one you mean.")
                else:
                    _cmsg = ("This question is ambiguous between multiple entities. "
                             "Please specify which one you mean.")
                fb = _feedback("clarify", msg=_cmsg)
                tr.set("entity_resolution", grounded_clarify=True, clarify_options=_opts)
                log_route("clarify", query, (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_cmsg, feedback=fb)
        if (_er is not None and _er.status == "RESOLVED"
                and _er.evidence.get("pin_eligible") and _er.anchor):
            primary = _er.anchor
            tr.set("entity_resolution", primary_before=_router_primary,
                   primary_after=primary, vet_primary_bypassed=True)
            print(f"  [ER] pinned primary {primary!r} (conf {_er.confidence}) — bypass vet_primary")
        else:
            primary = vet_primary(query, _router_primary, results, sm, trace=tr)
        if anchor_hint and anchor_hint in (sm.get("tables") or {}):
            # Qualifier-salvage retry: the first pass refused with a dropped qualifier
            # whose QSR referent lives in anchor_hint — retrieval/vetting never
            # surfaced it (single-table planning takes its columns from all_cols, not
            # from retrieval, so the miss doesn't matter). Overrides a clarify verdict
            # too: this run exists to test the hinted anchor against the full gates.
            if primary != anchor_hint:
                print(f"  [L3] Anchor hint   "
                      f"{(_router_primary if isinstance(primary, dict) else primary)!r}"
                      f" → {anchor_hint!r} (qualifier salvage)")
                tr.note("schema_linking", f"anchor_hint override → {anchor_hint}")
            primary = anchor_hint
        if isinstance(primary, dict):
            # single-table ambiguity gate: two sub-margin, differently-named subjects —
            # ask which grain the user means instead of silently picking one.
            _cmsg = primary.get("clarify")
            fb = _feedback("clarify", msg=_cmsg)
            log_route("clarify", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=_cmsg, feedback=fb)
        if primary != _router_primary:
            print(f"  [L3] Grain vet     router primary {_router_primary!r} → {primary!r} "
                  f"(word-order / grain-hint)")
        from config import PRIMARY_TABLE_SEED_BOOST
        es.primary_table = primary
        es.candidate_tables = list(_cand_tabs)
        # Plain {table_name, col_name, score} dicts — connector-agnostic, and reused
        # as-is by select_retrieval()'s seed-candidate merge (no second DB lookup).
        # Fields from the VETTED primary table get a small score boost — this is how
        # `primary_table` actually influences Tier2 (not just an inert log flag): Tier1
        # already spent a whole retrieval+grain-vet pass deciding this table is the
        # anchor, so Tier2's reranker should start from that prior, not from zero.
        # Small and additive — never overrides the cross-encoder's own judgment.
        # Enriched with retrieval PROVENANCE (RC-5): each entry keeps the raw RRF
        # score and the cross-encoder score separately (and a `reranked` flag), plus
        # the field's semantic_type from the model — so Tier2 can tell a resolved
        # MEASURE from a DIMENSION from an IDENTIFIER and knows which score is raw vs
        # reranked, instead of receiving one flattened number. The first three keys
        # are unchanged, so every existing consumer keeps working.
        _sm_cols = (sm or {}).get("columns", {})
        es.candidate_fields = [
            {"table_name": (_t := r.col_id.split(".", 1)[0]),
             "col_name":   r.col_id.split(".", 1)[1] if "." in r.col_id else r.column_name,
             "score":      float(getattr(r, "final_score", 0.0)) + (PRIMARY_TABLE_SEED_BOOST
                                                                     if _t == primary else 0.0),
             "semantic_type": (_sm_cols.get(r.col_id, {}) or {}).get("semantic_type"),
             "rrf_score":   float(getattr(r, "rrf_score", 0.0)),
             "cross_encoder_score": (float(r.cross_encoder_score)
                                     if getattr(r, "cross_encoder_score", None) is not None
                                     else None),
             "reranked":    getattr(r, "cross_encoder_score", None) is not None}
            for r in results[:15]
        ]
        # The exact text the cross-encoder reranked against (the ENHANCED query when
        # enhancement ran, else the raw query) — recorded so Tier2, which reranks
        # against the RAW query, can tell whether its scores are comparable to these.
        es.rerank_query = _search if _rk_after is not None else None
        tr.set("retrieval", candidate_tables=_cand_tabs[:8], n_columns=len(results))
        for r in results[:15]:
            # Per-signal scores (semantic_score/sparse_score/subgraph_score/fk_path_score/
            # value_index_score) are now actually populated (retrieval_engine_phase3.py) —
            # surface them here so the trace explains WHY a candidate ranked well, not just
            # that it did. "type" used to read `semantic_type`, a field RetrievalResult
            # never had (always None) — replaced with real signal-level evidence.
            tr.cand("retrieval", "top_columns",
                    {"col": r.col_id, "score": round(getattr(r, "final_score", 0.0), 3),
                     "signals": {
                         "semantic": round(getattr(r, "semantic_score", 0.0), 3),
                         "sparse":   round(getattr(r, "sparse_score", 0.0), 3),
                         "subgraph": round(getattr(r, "subgraph_score", 0.0), 3),
                         "fk_path":  round(getattr(r, "fk_path_score", 0.0), 3),
                         "value":    round(getattr(r, "value_index_score", 0.0), 3),
                     }})
        tr.set("schema_linking", selected_table=primary,
               router_primary=_router_primary, candidate_tables=_cand_tabs[:8])
        # ENTITY SELECTION (Tier-1): the vetted primary + the candidate tables it was
        # chosen from. Secondary entities (if any) enter via the join plan below and
        # are recorded there. Reuses values already computed — no new ranking.
        tr.set("entity_selection", primary_table=primary,
               candidate_tables=_cand_tabs[:12],
               candidate_field_count=len(es.candidate_fields),
               selected_reason=("router" if primary == _router_primary else "grain_vet_override"))
        if primary:
            # PRE-EXISTING LEAK, fixed here: this tick lands in the v1
            # explainability payload's `timeline`, which is user-facing — and
            # `primary` is a RAW TABLE NAME (observed live: "Using assets_asset
            # for this"). A table name must never reach a user-facing payload.
            # The table's identity is already available safely as a business
            # label in `data_used.datasets`, so the progress line does not need
            # to carry the identifier at all.
            _tick("schema_linking", "Identified the relevant records")
        print(f"  [L3] Routing       {len(results)} cols across {len(_cand_tabs)} tables "
              f"({', '.join(_cand_tabs[:4])}…) → primary: {primary}")
        if not primary:
            fb = _feedback("no_table", candidates=_cand_tabs)
            log_route("no_table", query, (time.time() - start) * 1000)
            return _done(1, "no_table", feedback=fb,
                         msg="no single table confidently matched the question")
        from_cache = False

        # Multi-table: deterministic join plan (LLM never writes joins). Fires for
        # MULTI_TABLE / AGGREGATE, and for any existence query (with/without/how-many-have)
        # — negation like "without" isn't tagged MULTI_TABLE, so detect it directly.
        needs_join = intent in ("MULTI_TABLE", "AGGREGATE") or is_existence
        # TYPED_MULTITABLE_ROUTE (flag-gated, default OFF): the audit proved needs_join is
        # False for every grouped/aggregate/relational query because `intent` is always
        # SIMPLE, so the deterministic planner is never reached. When ON, route clearly-
        # typed analytical queries into the EXISTING try_multitable planner using signals
        # ALREADY computed above (grammar modes) + generic relational phrasing — no new
        # planner, no intent rewrite, no table-name heuristics. try_multitable falls back
        # gracefully, so a false positive degrades to prior behavior.
        try:
            from config import TYPED_MULTITABLE_ROUTE
        except Exception:
            TYPED_MULTITABLE_ROUTE = False
        _typed_join = False
        if TYPED_MULTITABLE_ROUTE and not needs_join:
            _ql = f" {query.lower()} "
            _join_phrase = any(p in _ql for p in (
                " with their ", " with the ", " and their ", " of their ",
                " for each ", " per "))
            _typed_join = bool(_agg or _grp or _rat or _join_phrase)
            if _typed_join:
                needs_join = True
        # Entity Resolution V1: ≥2 DISTINCT resolved entity tables is structural evidence
        # that multi-table planning is required (independent of the SIMPLE-intent gate).
        # NOT "concept count > 1" — two concepts on one table stay single-table.
        _er_multi = bool(_er is not None and _er.status == "RESOLVED"
                         and _er.distinct_tables >= 2 and _er.anchor)
        if _er_multi:
            needs_join = True
        # Phase 1 ANALYTICAL_SQL_V2: a single-anchor analytical query has a deterministic
        # SQL already built — force the single-table path (no join planner) so it flows
        # straight to validation + execution with our structured SQL.
        if _analytical_sql or _analytical_spec is not None:
            if _analytical_primary and _analytical_primary != primary and _reentry \
                    and not _analytical_reentry_ok:
                # pre-M3 item 1: a RETRIEVAL-grounded anchor never overrides the router,
                # even on re-entry — candidate-only (discarded below like a first pass).
                tr.set("analytical_sql_v2", used=False, discarded="retrieval-grounded anchor is not re-entry-eligible",
                       anchor=_analytical_primary, router_primary=primary)
                print(f"  [AnalyticalV2] re-entry: anchor {_analytical_primary} grounded by retrieval "
                      f"— not eligible to override router primary {primary}")
                _analytical_sql, _analytical_spec, _analytical_primary = None, None, None
            elif _analytical_primary and _analytical_primary != primary and _reentry:
                # M2: the ONE case a grounded intent takes authority — fully grounded (it
                # would not exist otherwise) AND no deterministic branch matched (the one
                # that did was refused, which is why we are re-entering). The router chose
                # the DIMENSION table as primary ("average payment amount by currency" →
                # generics_currency): the grain-inversion class this layer exists to fix.
                tr.set("analytical_sql_v2", reentry_override=True,
                       anchor=_analytical_primary, router_primary=primary)
                print(f"  [AnalyticalV2] re-entry: grounded anchor {_analytical_primary} "
                      f"overrides router primary {primary} (no deterministic branch matched)")
                primary = _analytical_primary
            elif _analytical_primary and _analytical_primary != primary:
                # first pass: candidate only — the router's primary stands; a spec built on a
                # different anchor is discarded (trace), never overrides.
                tr.set("analytical_sql_v2", used=False, discarded="anchor != router primary",
                       anchor=_analytical_primary, router_primary=primary)
                print(f"  [AnalyticalV2] candidate discarded: anchor {_analytical_primary} "
                      f"≠ router primary {primary}")
                _analytical_sql, _analytical_spec, _analytical_primary = None, None, None
            elif _analytical_sql or (_reentry and _analytical_spec is not None):
                needs_join = False               # single-table candidate (a list spec too, on re-entry)
                _er_multi = False
        # Observability: record whether the deterministic multi-table planner is even
        # REACHED — needs_join gates try_multitable (planner-reachability signal).
        tr.set("join_planning", needs_join=needs_join, intent_for_join=intent,
               typed_multitable_route=_typed_join, entity_resolution_multi=_er_multi,
               try_multitable_invoked=bool(needs_join))
        if _er_multi:
            # Entity-first path: drive the EXISTING join engine with the resolved
            # canonical entities (anchor + distinct secondaries) — same build_from_entities
            # contract the LangGraph/Tier-2 path already uses. No new planner.
            # ANCHOR-AGNOSTIC: the planner is anchor-rooted, so which entity is the anchor
            # decides whether the join builds ("landlords and their properties" refuses
            # asset-rooted but builds user-rooted). Try each resolved entity as anchor and
            # take the first that produces SQL — the same multi-anchor strategy that moved
            # the isolated planner test from 31→40/48. Deterministic order (ER anchor first).
            from veda.planning import build_from_entities
            _ents = [_er.anchor] + [t for t in _er.secondaries if t != _er.anchor]
            mt = {"action": "fallback"}
            _tried = None
            # Grain-first (M1 close-out, 2026-09-15): "how many maintenance records PER
            # VENDOR" — the loop below anchors on ER's first entity (maintenance) and
            # grouped by its own CATEGORY column, a silently wrong dimension. try_multitable()
            # already resolves the grain from the "per/by X" phrase (whole-entity match →
            # anchor vendors, COUNT maintenance per vendor); when the question carries a
            # grouping phrase, let it go first and keep its SQL. Everything else falls
            # through to the entity-first loop unchanged.
            if _grp or _grpc_mode(query) or (_agg and re.search(r"\b(?:per|by)\s+[a-z]", query.lower())):
                try:
                    _mt0 = try_multitable(query, results, sm, all_cols, tf, primary=primary)
                    if isinstance(_mt0, dict) and _mt0.get("action") == "sql" and _mt0.get("sql"):
                        mt, _tried = _mt0, ((_mt0.get("plan") or {}).get("anchor") or "grain-first")
                        tr.note("join_planning", "grain-first: try_multitable resolved the grouping "
                                                  f"anchor {_tried!r} ahead of the entity-first loop")
                except Exception:
                    pass
            for _a in ([] if mt.get("action") == "sql" else _ents):
                _tg = [t for t in _ents if t != _a]
                _cand = build_from_entities(query, sm, all_cols, tf, _a, _tg, results=results)
                if isinstance(_cand, dict) and _cand.get("action") == "sql" and _cand.get("sql"):
                    mt, _tried = _cand, _a
                    break
                if mt.get("action") == "fallback" and isinstance(_cand, dict):
                    mt, _tried = _cand, _a          # keep best non-fallback (existence/aggregate)
            tr.set("join_planning", entity_first=True, er_anchor=_er.anchor,
                   er_targets=list(_er.secondaries), er_anchor_used=_tried)
        else:
            mt = try_multitable(query, results, sm, all_cols, tf, primary=primary) if needs_join else {"action": "fallback"}
        tr.set("join_planning", multitable_action=mt.get("action"))

        if mt["action"] == "clarify":
            fb = _feedback("clarify", msg=mt.get("msg"))
            return _guard_refusal("clarify", mt.get("msg"), fb, "clarify", always_done=True)
        if mt["action"] == "refuse":
            fb = _feedback("refuse", msg=mt.get("msg"))
            return _guard_refusal("refuse", mt.get("msg"), fb, "refuse", always_done=True)
        if mt["action"] == "existence":
            # Deterministic EXISTS / NOT EXISTS — no LLM, no fan-out, no join skeleton.
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="existence", anchor=mt["anchor"],
                   mode=mt["mode"], tables=sorted(mt["tables"]))
            _tick("sql_planning", "Checking which records match")
            print(f"  [L4b] Existence    {mt['mode']}  {mt['anchor']} ⟕ "
                  f"{' '.join(t for t in mt['tables'] if t != mt['anchor'])}")
            for w in p["why"]:
                print(f"        ↳ {w}")
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            print("  [L5] SQL           deterministic (no LLM)")
        elif mt["action"] == "aggregate":
            # Deterministic pre-aggregation CTEs — no LLM, fan-out-free by construction.
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="aggregate", anchor=mt["anchor"],
                   measures=mt.get("metrics"), dimension=mt.get("group_col"),
                   threshold=mt.get("threshold"), top_n=mt.get("top_n"))
            _tick("sql_planning", "Calculating the numbers")
            thr = mt.get("threshold")
            print(f"  [L4c] Grain plan   {mt['anchor']} ⟕ {', '.join(mt['metrics'])}"
                  + (f"  (filter {thr}+)" if thr is not None else ""))
            for w in p["why"]:
                print(f"        ↳ {w}")
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            print("  [L5] SQL           deterministic pre-aggregation (no LLM)")
        elif mt["action"] == "sql":
            p = mt["plan"]
            _rec_plan(p)
            tr.set("sql_planning", action="sql", tables=sorted(mt["tables"]))
            _tick("sql_planning", "Building the query")
            _llm_sql = True
            print(f"  [L4b] Join plan    {' ⋈ '.join(sorted(mt['tables']))}  "
                  f"(conf {p['confidence']}, fan-out {p['max_fanout']})")
            for w in p["why"]:
                print(f"        ↳ {w}")
            t_sql = time.time()
            sql = mt["sql"]
            allowed_tables, allowed_columns = mt["tables"], mt["columns"]
            # ON-integrity constraints: the LLM must keep these exact join keys + predicates
            pred_cols = set()
            for e in p["join_path"]:
                if e.get("requires_predicate"):
                    m = re.search(r"\.(\w+)\s*=", e["requires_predicate"])
                    if m:
                        pred_cols.add(m.group(1))
            join_constraints = {
                "key_pairs": [frozenset({e["source_column"], e["target_column"]}) for e in p["join_path"]],
                "qualified_pairs": mt.get("qualified_key_pairs") or [],
                "predicate_cols": pred_cols}
            fanout_guard = {"parent_aliases": mt.get("parent_aliases", set()),
                            "parent_only_cols": mt.get("parent_only_cols", set())}
            print(f"  [L5] SQL gen       {time.time()-t_sql:.1f}s (join skeleton fixed)")
        else:
            # Single-table path
            allowed_tables = {primary}
            allowed_columns = [c.split(".", 1)[1] for c in all_cols if c.startswith(primary + ".")]

            # "latest 10 X" / "top 5 Y" / "bottom 3 Z" — an explicit ranking request
            # the deterministic branches below must honor (ORDER BY + LIMIT N),
            # instead of every branch hardcoding "LIMIT 100" with no ordering.
            _rank = parse_ranking(query)

            # FK→label resolution (task #6): if a query VALUE grounds (EXACT) to a related
            # table reachable by an FK, build the filter THROUGH that FK deterministically
            # (subquery) — never let the LLM match a name against a foreign-key id. The
            # resolver returns None on cross-table / ambiguous / no-FK-path values, so we
            # fall through to the normal LLM path (refuse-over-guess preserved downstream).
            _fk = None
            try:
                from config import FK_VALUE_RESOLUTION_ENABLED
            except Exception:
                FK_VALUE_RESOLUTION_ENABLED = True
            if FK_VALUE_RESOLUTION_ENABLED:
                try:
                    from query.value_resolver import resolve_value_filter, column_values_lookup
                    from veda.runtime import get_graph
                    from veda.runtime import _pg as _pgc
                    _qtoks = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
                    # anchor's own column names: a token naming one ("email"→user.email) is a
                    # column to project, not a cross-table value filter — pass so it's skipped.
                    _anchor_cols = {c.split(".", 1)[1] for c in sm.get("columns", {})
                                    if c.split(".", 1)[0] == primary}
                    _fk = resolve_value_filter(primary, _qtoks, get_graph(),
                                               column_values_lookup(_pgc), anchor_cols=_anchor_cols)
                except Exception:
                    _fk = None

            # Multi-hop FK resolution (OFF by default): only when 1-hop found nothing, try a
            # junction-membership path (e.g. tags on a document via document_tags). Fires
            # ONLY for a single unambiguous path; multiple paths (RBAC direct+role) or shared
            # dimensions → None → falls to the LLM. Never guesses/unions.
            _mh = None
            try:
                from config import MULTIHOP_FK_RESOLUTION_ENABLED
            except Exception:
                MULTIHOP_FK_RESOLUTION_ENABLED = False
            if _fk is None and MULTIHOP_FK_RESOLUTION_ENABLED:
                try:
                    from query.fk_path_resolver import resolve_fk_path
                    from veda.runtime import get_graph as _gg_mh
                    from veda.runtime import _pg as _pgc_mh
                    from query.value_resolver import column_values_lookup as _cvl
                    from retrieval.query_enrichment import _singularize as _sg_mh
                    _qtoks_mh = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
                    # Anchor attribute tokens: a query word naming one ("state"→workflow_state)
                    # is projection, not a cross-table value filter — never fabricate a join for it.
                    _anchor_col_toks_mh = {_sg_mh(tok) for c in all_cols
                                           if c.split(".", 1)[0] == primary
                                           for tok in c.split(".", 1)[1].split("_") if len(tok) > 2}
                    _mh = resolve_fk_path(primary, _qtoks_mh, _gg_mh(), _cvl(_pgc_mh),
                                          anchor_col_toks=_anchor_col_toks_mh)
                    if _mh:
                        print(f"  [L4d] multi-hop FK  {' → '.join(_mh['path'])}  — deterministic, no LLM")
                except Exception:
                    _mh = None

            # Value-vs-Column Arbitration (runs before SQL generation): classify query
            # spans against the sampled column_values store. Business adjectives that
            # match a categorical value ("critical", "open") are grounded as VALUE
            # filters on the anchor table — never as columns — and negations
            # ("unresolved" -> status != resolved) become structured filters the LLM
            # would otherwise miss. Data-driven (EXACT value match); no word lists.
            _arb_filters = []
            try:
                from config import VALUE_ARBITER_ENABLED
            except Exception:
                VALUE_ARBITER_ENABLED = False
            if VALUE_ARBITER_ENABLED:
                try:
                    from query.value_arbiter import (arbitrate, anchor_filters,
                                                     build_schema_terms)
                    # QSR typed lookup — the old column_values_typed_lookup(runtime._pg)
                    # pointed at the SOURCE DB (no column_values there) and silently
                    # returned [] for every token; the arbiter never saw a value.
                    from query.resolution import typed_value_lookup
                    # REMEMBERED FILTER VALUES ARE NOT CARRIED AS BARE VALUES. They were,
                    # briefly, and
                    # the measurement is why they are not: a frame stores the engine's
                    # humanised LABEL as a filter's field ("Location"), never the real
                    # column, so a remembered filter can only travel as a bare VALUE — and
                    # a bare value does not always name its own column. Measured
                    # 2026-09-23, carrying "true" forward from an `is_gated` filter
                    # re-grounded it onto `all_day_access`, filtering the follow-up on a
                    # column the conversation never mentioned. A grounding-multiplicity
                    # test does not catch it either: the arbiter resolves "true" to exactly
                    # one column, just not the right one.
                    #
                    # Filtering on a column the user never named is silent-wrong, so the
                    # value is dropped rather than guessed. The consequence is honest and
                    # known: a filter from an EARLIER turn is not re-applied by a later one
                    # — the user's own words for THIS turn still are. Fixing it properly
                    # means the frame harvesting the REAL column out of `last_sql` (which
                    # it already stores) instead of the display label; that is its own
                    # piece of work, tracked as a gap rather than guessed at here.
                    _arb = arbitrate(query, typed_value_lookup(),
                                     build_schema_terms(sm))
                    _arb_filters = anchor_filters(_arb, primary)
                    # REMEMBERED FILTERS, applied structurally. Each carries the RAW column
                    # the previous turn's SQL actually filtered on (business_explain now
                    # keeps it alongside the humanised label), so nothing is re-grounded
                    # from a bare value and nothing is guessed — this is the same column,
                    # on the same table, that already executed. Applied ONLY when this turn
                    # anchored on the SAME table the conversation is about, and never over
                    # a column THIS turn already filtered on: the user's current words win,
                    # which is what makes "what about Mumbai" a replacement rather than an
                    # impossible "Pune AND Mumbai".
                    if _conv_filters and primary == _conv.get("entity_table"):
                        _own_cols = {f.get("column") for f in _arb_filters}
                        # sm["columns"] is keyed "table.column" — the only place the
                        # semantic model states which columns a table really has.
                        _anchor_cols = {k.split(".", 1)[1]
                                        for k in (sm.get("columns") or {})
                                        if k.startswith(primary + ".")} or None
                        for _cf in _conv_filters:
                            _c = _cf["column"]
                            if _c in _own_cols:
                                continue
                            if _anchor_cols is not None and _c not in _anchor_cols:
                                continue
                            # Same dict shape anchor_filters produces: where_clause
                            # compares lower(col) against `value_norm`, the sampler's
                            # lowercase form, so High/high/HIGH all match. `value` is
                            # kept for the trace/explain that read it.
                            _v = str(_cf["value"])
                            _arb_filters.append({"column": _c, "op": "=", "value": _v,
                                                 "value_norm": _v.strip().lower()})
                            print(f"  [conversation] carried filter {_c} = "
                                  f"{_cf['value']!r} (from the previous turn's SQL)")
                    if _arb.value_filters:
                        for _ln in _arb.explain().splitlines():
                            print("  [L4c] " + _ln)
                        tr.set("value_arbitration", table=primary, filters=[
                            (f["column"], f["op"], f["value"]) for f in _arb_filters])
                except Exception:
                    _arb_filters = []

            # Lifecycle/status phrase grounding (D.5, 2026-09-23): "currently on the
            # market for sale" means assets_salelisting.status = 'APPROVED', but no
            # sampled VALUE contains those words, so the value arbiter (exact-match only)
            # never saw the qualifier and it was dropped in silence — "the cheapest
            # properties currently on the market" came back with 74 of 100 rows DRAFT or
            # CANCELLED. The curated phrase->value glossary grounds it; an unmapped state
            # word on an anchor that HAS a status column becomes a clarify naming that
            # column's real domain, never a silent unfiltered answer.
            _vg_filters, _vg_matched = [], []
            try:
                from query.value_glossary import phrase_filters as _vg_build
                _vg_filters, _vg_matched = _vg_build(query, primary, sm)
                if _vg_filters:
                    from query.value_glossary import explain as _vg_explain
                    print("  [L4c] lifecycle    " + _vg_explain(_vg_filters))
            except Exception:
                _vg_filters, _vg_matched = [], []
            # Runs even when a phrase DID ground: "which properties available for sale
            # currently have an ACTIVE status" grounds "available for sale" to APPROVED
            # while still naming a state ("active") that does not exist in the column at
            # all. Answering that with APPROVED silently substitutes a different question.
            if True:
                try:
                    from query.value_glossary import unmapped_state_clarify as _vg_clar
                    _sc = _vg_clar(query, primary, sm, _vg_matched)
                    if _sc:
                        fb = _feedback("clarify", msg=_sc)
                        # A literal tag, not `_route + ...`: `_route` is not bound
                        # until the branch chain has chosen a head (~500 lines below),
                        # and referencing it here raised UnboundLocalError, which the
                        # API tier reported as a 502 LLM_UNAVAILABLE.
                        log_route("deterministic.state_ungrounded", query,
                                  (time.time() - start) * 1000)
                        return _done(0, "clarify", msg=_sc, feedback=fb) if return_result else 0
                except Exception:
                    pass

            # Numeric-predicate grounding (D.1, 2026-09-23): "between 100 and 50,000",
            # "priced above 10,000". The value arbiter only ever grounded CATEGORICAL
            # spans, so a numeric range reached SQL generation carried in English alone
            # and came back as an unfiltered LIMIT 100 dump with the range silently
            # dropped — the worst failure shape in the set, because it renders as a real
            # answer. query/numeric_filter resolves the comparison onto a REAL measure
            # column of the anchor, or returns a clarify NAMING the candidates when the
            # anchor has several and the question doesn't say which. It never guesses.
            _num_filters, _num_clarify = [], None
            try:
                from query.numeric_filter import build_filters as _num_build
                _num_filters, _num_clarify, _num_cands = _num_build(query, primary, sm)
                if _num_filters:
                    from query.numeric_filter import explain as _num_explain
                    print("  [L4d] numeric      " + _num_explain(_num_filters))
                    tr.set("numeric_filters", table=primary, filters=[
                        (f["column"], f["op"], f["value"], f.get("value2"))
                        for f in _num_filters])
            except Exception:
                _num_filters, _num_clarify = [], None
            if _num_clarify:
                fb = _feedback("clarify", msg=_num_clarify)
                # literal tag — `_route` is not bound this early (see above)
                log_route("deterministic.numeric_ambiguous", query,
                          (time.time() - start) * 1000)
                return _done(0, "clarify", msg=_num_clarify, feedback=fb) if return_result else 0

            # Temporal window → grounded predicate on the anchor's canonical temporal
            # column, applied DIRECTLY in the deterministic SQL (FK / arbiter / temporal-
            # only). A date filter is never silently dropped, and we never fall back to the
            # LLM just to add a BETWEEN. Schema-metadata driven (_resolve_temporal_column).
            #
            # An explicit ranking request needing a temporal sort ("latest 10 X") also
            # needs the canonical column resolved even when L1 found no date RANGE at
            # all (e.g. "last 10 ledger entries" — "last" without a time unit sets no
            # temporal_filter), so ORDER BY has something to sort by.
            # `ranked`, not `top_n is not None`: "the LATEST payments" and "the OLDEST
            # financial records" name a sort without naming a count, and gating on an
            # explicit N meant those fell through with no ORDER BY at all — the engine
            # returned an arbitrary 100-row page and called it "the latest" (2026-09-23
            # question.txt, Q1/Q5/Q6/Q18/Q19/Q20). The count stays optional; the SORT is
            # what the question actually asked for.
            _want_rank_order = _rank.ranked and _rank.basis == "temporal"
            # Bare-count shape (2026-09-15): "how many X are there" — aggregate_mode()'s
            # "counting" branch (veda/planning.py) already flags this correctly in
            # query_understanding.aggregation (_agg here), but nothing downstream in this
            # branch chain used to CONSUME that signal for the plain, unqualified case (no
            # value filter, no per-anchor threshold, no top-N/ranking) — it fell all the way
            # to the generic `else` below, which builds a row-list SELECT with zero aggregate
            # functions. veda/intent_sql_alignment.py::aggregate_presence_ok then correctly
            # refused ("I couldn't work out a reliable total...") rather than show that row
            # list as if it were the count — a real, working safety net catching a real,
            # upstream gap (confirmed live: "how many properties are there?" against source 2,
            # anchor correctly resolved to assets_asset, SQL gen still produced a 46-column row
            # list). `_agg` is the SAME dict computed once near the top of this function
            # (aggregate_mode(query)) — deliberately excludes threshold/op/top_n/ranked so this
            # never intercepts a per-anchor child-count ("X with more than one Y") or a ranked
            # count ("top 5 X by count"), which have their own existing handling elsewhere.
            # Also excludes any GROUPING phrase ("per vendor", "by city", "for each X") —
            # grouped_mode() deliberately returns None for COUNT-shaped queries (its own
            # contract: "COUNT-per-dimension ... has its own counting machinery"), so `_grp`
            # can't be used as the guard here; check the grammar's grouping words directly.
            # Found live the same day this branch landed: "how many maintenance records
            # per vendor" (source 4) was hijacked into a scalar COUNT(*) and then correctly
            # refused by the qualifier gate ('vendor' dropped) — a regression this guard closes.
            from config import QUERY_GRAMMAR as _QG_bc
            _ql_bc = f" {query.lower()} "
            _has_grouping_bc = any(
                (" " in w and w in _ql_bc) or re.search(rf"\b{re.escape(w)}\b", _ql_bc)
                for w in _QG_bc.get("grouping", []))
            _bare_count = bool(_agg) and _agg.get("op") is None and _agg.get("threshold") is None \
                and not _agg.get("top_n") and not _agg.get("ranked") and not _has_grouping_bc \
                and not _num_filters       # "how many X above 10,000" is a FILTERED count
            _tcol = (_resolve_temporal_column(primary, sm, query)
                    if (tf and (tf.start or tf.end)) or _want_rank_order else None)
            # "latest 10 X" ALSO makes L1 match the vague-recency word and derive a
            # synthetic last-30-days BETWEEN window (query/temporal_parser.py) — but
            # "the latest 10" means ORDER BY ... DESC LIMIT 10, not "some rows from an
            # arbitrary 30-day window, silently dropping the requested count". When the
            # ranking's own count is present and the ONLY temporal signal was that bare
            # vague-recency wording (never an explicit range like "last month"/"since
            # January"), skip the synthetic window and let ORDER BY + LIMIT do the job.
            _skip_vague_window = (_want_rank_order
                                  and _is_vague_recency_only(_tp_result.raw_expressions))
            _tpred = (_temporal_predicate(primary, sm, tf)
                     if _tcol and not _skip_vague_window else "")
            _rank_tail = _rank_order_limit_sql(_rank, primary, sm, _tcol)
            _rank_tail_a = _rank_order_limit_sql(_rank, primary, sm, _tcol, alias="a")
            # Whatever column the tail above actually ORDER BY's on (if any) must
            # always be part of what's shown — a "latest 10 X" result sorted by a
            # date the recommendation happened not to pick would be confusing.
            # Computed once, reused as recommended_projection's must_include below.
            _rank_sort_col = _rank_sort_column(_rank, primary, sm, _tcol)
            # Postgres requires SELECT DISTINCT's ORDER BY expressions to appear in the
            # select list — the WHO/distinct-name branch below projects only the display
            # column, so it can honor an explicit count but not an ORDER BY on a column
            # (the temporal/metric column) that isn't part of that projection.
            _limit_only_tail = f' LIMIT {_rank.top_n if _rank.top_n is not None else 100}'
            # Which column the ranking actually sorted by — passed to the L7b NL
            # summarizer (query/result_explainer.py) so it narrates the right field
            # (e.g. "amount") instead of guessing (e.g. an id column) for "top N"/
            # "latest N" style questions.
            if _rank.basis == "temporal" and _tcol:
                _rank_column_for_nl = _tcol
            elif _rank.basis == "metric":
                _rank_column_for_nl = _resolve_rank_metric_column(primary, sm)

            # Answer-Entity Discovery (OFF by default): a WHO question projects the
            # person's display column reached over a FK, not the raw id. Reuses
            # concept_graph["PERSON"] + the FK graph + _resolve_display_column. Deterministic
            # JOIN. anchor value filters (e.g. "high priority") apply, alias-prefixed.
            _ans = None
            try:
                from config import ANSWER_ENTITY_DISCOVERY_ENABLED
            except Exception:
                ANSWER_ENTITY_DISCOVERY_ENABLED = False
            if ANSWER_ENTITY_DISCOVERY_ENABLED:
                try:
                    from query.answer_entity import find_answer_entity
                    from veda.runtime import get_graph as _gg
                    _ans = find_answer_entity(query, primary, _gg(), sm)
                    if _ans:
                        print(f"  [L4e] answer-entity  {_ans['reason']}")
                        tr.set("sql_planning", action="answer_entity_detect",
                               table=primary, fk_col=_ans["fk_col"],
                               target=_ans["target_table"], display=_ans["display_col"])
                except Exception:
                    _ans = None

            if _reentry and _analytical_reentry_ok and (
                    _analytical_sql or (_analytical_spec is not None
                                        and getattr(_analytical_spec, "aggregation", None) == "list"
                                        and _analytical_spec.anchor == primary
                                        and _analytical_spec.where_sql)):
                # M2 re-entry: the deterministic answer was refused, the grounded candidate
                # (anchor == router primary, enforced above) takes the first turn.
                _anchor_cols_re = [c.split(".", 1)[1] for c in all_cols
                                   if c.split(".", 1)[0] == primary]
                if not _analytical_sql:
                    _lp_cols = recommended_projection(primary, _anchor_cols_re, results, sm, query)
                    _lp = ", ".join(f't0."{c}"' for c in (_lp_cols or [])) or "t0.*"
                    _analytical_sql = (f'SELECT {_lp} FROM "{primary}" t0 '
                                       f'WHERE {_analytical_spec.where_sql} LIMIT 100')
                sql = _analytical_sql
                _analytical_used = True
                allowed_columns = list(dict.fromkeys(list(allowed_columns) + _anchor_cols_re))
                _llm_sql = False
                tr.set("sql_planning", action="analytical_v2_reentry", table=primary)
                _tick("sql_planning", "Using the grounded analytical plan")
                print(f"  [AnalyticalV2] re-entry: grounded candidate takes the first turn — {sql[:120]}")
            elif _ans and not _tpred:
                # Project the person over the FK (display name, not the raw id). Defer to the
                # normal path when a temporal window is present so the date filter isn't dropped.
                from query.value_arbiter import where_clause as _arb_where
                _disp, _tt = _ans["display_col"], _ans["target_table"]
                _fkc, _tpk = _ans["fk_col"], _ans["target_pk"]
                _w = _arb_where(_arb_filters, alias="a") if _arb_filters else ""
                if _ans.get("mode") == "projection":
                    # "incidents and their handler" → show the anchor rows + the handler NAME.
                    # LEFT JOIN so anchor rows with no related person still appear.
                    _rl = _ans.get("rel_label", _disp)
                    _anchor_cols = [c.split(".", 1)[1] for c in all_cols
                                    if c.split(".", 1)[0] == primary]
                    # Business-facing anchor projection instead of `a.*` (every anchor
                    # column) — same recommended_projection() every other deterministic
                    # branch uses; allowed_columns below is UNCHANGED (still every
                    # anchor column, for validation) so this only narrows what's shown.
                    _ans_proj_cols = recommended_projection(primary, _anchor_cols, results,
                                                            sm, query)
                    _ans_proj = ", ".join(f'a."{c}"' for c in _ans_proj_cols) or "a.*"
                    sql = (f'SELECT {_ans_proj}, t."{_disp}" AS "{_rl}" '
                           f'FROM "{primary}" a LEFT JOIN "{_tt}" t ON a."{_fkc}" = t."{_tpk}"'
                           + (f' WHERE {_w}' if _w else '') + _rank_tail_a)
                    allowed_columns = (_anchor_cols + [_disp, _fkc, _tpk]
                                       + [f["column"] for f in _arb_filters])
                    _proj_desc = f"{_ans_proj} + {_tt}.{_disp} AS {_rl}"
                else:
                    # WHO → just the person's distinct name.
                    sql = (f'SELECT DISTINCT t."{_disp}" '
                           f'FROM "{primary}" a JOIN "{_tt}" t ON a."{_fkc}" = t."{_tpk}"'
                           + (f' WHERE {_w}' if _w else '') + _limit_only_tail)
                    allowed_columns = ([_disp, _fkc, _tpk]
                                       + [f["column"] for f in _arb_filters])
                    _proj_desc = _disp
                allowed_tables = {primary, _tt}
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="answer_entity", table=primary,
                       target=_tt, project=_proj_desc, via=_fkc,
                       filters=[(f["column"], f["op"], f["value"]) for f in _arb_filters])
                _tick("sql_planning", "Looking up who's involved")
                print(f"  [L4e] answer-entity  {_ans.get('mode','who')}: JOIN {_tt} → {_proj_desc}"
                      f"{(' + ' + str(len(_arb_filters)) + ' filter(s)') if _arb_filters else ''}"
                      "  — deterministic, no LLM")
            elif _fk and _fk.get("kind") == "subquery":
                # OR across exactly the exact-grounded columns (no preference, no LIKE).
                _or = " OR ".join(
                    f"""lower("{c}"::text) = lower('{str(v).replace("'", "''")}')"""
                    for c, v in _fk["pairs"])
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = [f'"{_fk["anchor_col"]}" IN '
                           f'(SELECT "{_fk["target_col"]}" FROM "{_fk["target"]}" WHERE {_or})']
                if _tpred:
                    _wparts.append(_tpred)        # FK + temporal stays deterministic
                sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                       + " AND ".join(_wparts) + _rank_tail)
                allowed_tables = {primary, _fk["target"]}
                allowed_columns = (allowed_columns + [c for c, _ in _fk["pairs"]]
                                   + [_fk["target_col"], _fk["anchor_col"]]
                                   + ([_tcol] if _tcol else []))
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="fk_value_resolution", table=primary,
                       target=_fk["target"], via=_fk["anchor_col"],
                       filters=[c for c, _ in _fk["pairs"]], temporal=_tcol)
                _tick("sql_planning", "Matching that to the right record")
                print(f"  [L4d] FK value     {primary}.{_fk['anchor_col']} → "
                      f"{_fk['target']}.({', '.join(c for c, _ in _fk['pairs'])}) "
                      f"= '{_fk['pairs'][0][1]}'"
                      + (f" + {_tcol} window" if _tcol else "") + "  — deterministic, no LLM")
            elif _mh:
                # Multi-hop junction membership: WHERE anchor_pk IN (<nested IN-subquery>).
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = [f'"{_mh["anchor_col"]}" IN ({_mh["subquery"]})']
                if _tpred:
                    _wparts.append(_tpred)
                sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                       + " AND ".join(_wparts) + _rank_tail)
                allowed_tables = {primary, *_mh["path"]}
                allowed_columns = allowed_columns + [_mh["anchor_col"]] + ([_tcol] if _tcol else [])
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="multihop_fk_resolution", table=primary,
                       path=_mh["path"], anchor_col=_mh["anchor_col"], temporal=_tcol)
                _tick("sql_planning", "Tracing the connection through related records")
            elif _arb_filters or _num_filters or _vg_filters:
                # Deterministic single-table SQL with arbiter-grounded categorical
                # filters (= for VALUE, != for NEGATED_VALUE) and/or grounded NUMERIC
                # comparisons ("amount BETWEEN 100 AND 50000"). All columns belong to the
                # anchor table by construction (anchor_filters filtered on `primary`;
                # numeric_filter resolves only against the anchor's own measures).
                # where_clause compares lower(col) to value_norm, so High/high/HIGH match.
                from query.value_arbiter import where_clause as _arb_where
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                # The column a numeric comparison filters on must also be SHOWN — an
                # answer to "payments between 100 and 50,000" that never displays the
                # amount is unreadable (Q10 projected twelve columns, none of them an
                # amount, while claiming to answer a range question).
                _must = [c for c in ([_rank_sort_col] if _rank_sort_col else [])
                         + [f["column"] for f in _num_filters]
                         + [f["column"] for f in _vg_filters] if c]
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=_must or None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                _wparts = []
                if _arb_filters:
                    _wparts.append(_arb_where(_arb_filters))
                if _vg_filters:
                    from query.value_glossary import where_clause as _vg_where
                    _wparts.append(_vg_where(_vg_filters))
                if _num_filters:
                    from query.numeric_filter import where_clause as _num_where
                    _wparts.append(_num_where(_num_filters))
                if _tpred:
                    _wparts.append(_tpred)        # value filter + temporal stays deterministic

                # REMEMBERED SHAPE. A drill-down into an AGGREGATED answer used to come
                # back as a raw row list: "distribution of properties by facing" plus
                # "only the ones in Pune" produced 1000 rows and no GROUP BY, and the
                # summariser then narrated those rows as if they were the distribution
                # (measured 0 of 8 aggregated bases survived, 2026-09-23). The shape is
                # not re-derived here — group_by and the measure are the columns the
                # PREVIOUS turn's SQL actually grouped and measured, replayed onto the
                # same table with this turn's filter added. Applied only when the anchor
                # IS that table, when this turn asked for no shape of its own, and when
                # every remembered column still exists on it.
                _shape_cols = [c for c in _conv_group_by
                               if _anchor_columns_for(sm, primary) is None
                               or c in _anchor_columns_for(sm, primary)]
                _reshaped = False
                # A SCALAR aggregate — "what is the total rent", "how many sale listings" —
                # has no group_by at all, so the branch below never fired for it and the
                # follow-up came back as rows: filtering "total expected monthly rent" by
                # furnishing returned a row list instead of a total (measured 4/4,
                # 2026-09-23). The question is still an aggregate once a filter is added;
                # only the population changed. Same guards as the grouped case.
                if (not _shape_cols and _conv.get("aggregation")
                        and primary == _conv.get("entity_table")
                        and not _rank_tail and not _grouped_this_turn(query)):
                    _fn0 = {"sum": "SUM", "average": "AVG", "avg": "AVG",
                            "minimum": "MIN", "min": "MIN",
                            "maximum": "MAX", "max": "MAX"}.get(
                        str(_conv.get("aggregation")).lower())
                    _m0 = next((m for m in _conv_measures
                                if _anchor_columns_for(sm, primary) is None
                                or m in _anchor_columns_for(sm, primary)), None)
                    if _fn0 and _m0:
                        _a0 = f"{_fn0.lower()}_{_m0}"
                        _sel = f'{_fn0}("{_m0}") AS "{_a0}"'
                        allowed_columns = allowed_columns + [_m0]
                    else:
                        _sel = f'COUNT(*) AS "{primary}_count"'
                    sql = (f'SELECT {_sel} FROM "{primary}" WHERE '
                           + " AND ".join(_wparts))
                    _reshaped = True
                    print(f"  [conversation] kept the previous turn's aggregate: {_sel}")
                if (not _reshaped and _shape_cols and primary == _conv.get("entity_table")
                        and not _rank_tail and not _grouped_this_turn(query)):
                    _g = ", ".join(f'"{c}"' for c in _shape_cols)
                    _meas = next((m for m in _conv_measures
                                  if _anchor_columns_for(sm, primary) is None
                                  or m in _anchor_columns_for(sm, primary)), None)
                    # Use the aggregate the PREVIOUS turn actually computed. Assuming SUM
                    # turned a COUNT(DISTINCT id) distribution into SUM("id") — a summed
                    # primary key, which the summariser then reported as a real figure
                    # ("id_total above the average of 291,689", measured 2026-09-23). A
                    # count needs no measure column at all.
                    _fn = {"sum": "SUM", "average": "AVG", "avg": "AVG",
                           "minimum": "MIN", "min": "MIN",
                           "maximum": "MAX", "max": "MAX"}.get(
                        str(_conv.get("aggregation") or "").lower())
                    if _fn and _meas:
                        _alias = f"{_meas}_{_fn.lower()}"
                        _agg = f'{_fn}("{_meas}") AS "{_alias}"'
                    else:
                        _alias = f"{primary}_count"
                        _agg = f'COUNT(*) AS "{_alias}"'
                    # ORDER BY the ALIAS, not the ordinal: `ORDER BY 2` is a positional
                    # reference and validate_and_parameterize treats a bare integer as a
                    # literal to bind, so it came back as `ORDER BY %s` — a parameter
                    # where Postgres needs an expression.
                    sql = (f'SELECT {_g}, {_agg} FROM "{primary}" WHERE '
                           + " AND ".join(_wparts)
                           + f' GROUP BY {_g} ORDER BY "{_alias}" DESC')
                    allowed_columns = allowed_columns + _shape_cols + ([_meas] if _meas else [])
                    _reshaped = True
                    print(f"  [conversation] kept the previous turn's shape: "
                          f"GROUP BY {', '.join(_shape_cols)}"
                          + (f", SUM({_meas})" if _meas else ", COUNT(*)"))
                if not _reshaped:
                    # A remembered ORDER BY / LIMIT is the user's own earlier "top 10" or
                    # "sorted by rent", so it survives a filter unless THIS turn asked for
                    # a ranking of its own (_rank_tail already carries that).
                    _tail = _rank_tail
                    if not _rank_tail and primary == _conv.get("entity_table"):
                        _ocols = [c for c in _conv_order_by
                                  if _anchor_columns_for(sm, primary) is None
                                  or c in _anchor_columns_for(sm, primary)]
                        if _ocols:
                            _tail += " ORDER BY " + ", ".join(f'"{c}" DESC' for c in _ocols)
                            allowed_columns = allowed_columns + _ocols
                        if _conv_limit:
                            _tail += f" LIMIT {_conv_limit}"
                    sql = (f'SELECT {_proj} FROM "{primary}" WHERE '
                           + " AND ".join(_wparts) + _tail)
                allowed_columns = (allowed_columns + [f["column"] for f in _arb_filters]
                                   + [f["column"] for f in _num_filters]
                                   + [f["column"] for f in _vg_filters]
                                   + ([_tcol] if _tcol else []))
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="value_arbiter_filter", table=primary,
                       filters=[(f["column"], f["op"], f["value"]) for f in _arb_filters]
                               + [(f["column"], f["op"], f["value"]) for f in _num_filters],
                       temporal=_tcol)
                _tick("sql_planning", "Applying your filters")
                print(f"  [L4c] value filter {primary} WHERE {' AND '.join(_wparts)}"
                      "  — deterministic, no LLM")
            elif _bare_count:
                # "how many X are there" (optionally + a date window, e.g. "how many X
                # were added last month") — see `_bare_count`'s own comment above for why
                # this is needed. Checked BEFORE `_tpred` on purpose: a bare count with a
                # date window should still COUNT, not fall into the temporal_only row-list
                # branch below and get refused the same way. No value/FK filter here (those
                # already matched `_arb_filters`/`_mh`/`_fk` above, which own their own SQL
                # shape) — this is deliberately the plain "just count everything (optionally
                # in a window)" case only.
                #
                # ANCHOR EVIDENCE (2026-09-18): nothing else anchors this branch — no filter,
                # no join, no measure column — so the table must be NAMED by the question
                # (a name token or its L3 primary_entity). Otherwise this counted whatever
                # table retrieval ranked first: "how many gizmos are there" → COUNT(*) FROM
                # worklists_ticketuser (battery, flags ON and OFF). Typed clarify instead.
                from veda.understanding.grounding import anchor_named_in_query as _anchor_named
                _bc_method = _anchor_named(query, primary, sm)
                if _bc_method is None:
                    _bc_disp = (sm.get("tables", {}).get(primary) or {}).get("primary_entity") or primary
                    _msg = (f"I couldn't tell which data you mean to count — the closest match is "
                            f"'{_bc_disp}', but the question doesn't name it. Could you say which "
                            f"records you'd like counted?")
                    tr.set("sql_planning", action="bare_count_ungrounded", table=primary)
                    fb = _feedback("clarify", msg=_msg)
                    log_route("clarify.bare_count_ungrounded", query, (time.time() - start) * 1000)
                    return _done(0, "clarify", msg=_msg, feedback=fb)
                tr.set("sql_planning", anchor_evidence=_bc_method)
                sql = f'SELECT COUNT(*) AS count FROM "{primary}"' + (f' WHERE {_tpred}' if _tpred else '')
                allowed_columns = allowed_columns + ([_tcol] if _tcol else [])
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="bare_count", table=primary, temporal=_tcol)
                _tick("sql_planning", "Counting the matching rows")
                print(f"  [L4f] Bare count   COUNT(*) FROM {primary}"
                      + (f" WHERE {_tcol} in window" if _tpred else "")
                      + "  — deterministic, no LLM")
            elif _tpred:
                # Temporal-only deterministic projection ("users created last month") — no
                # value/FK filter, just the date window on the canonical temporal column.
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                sql = f'SELECT {_proj} FROM "{primary}" WHERE {_tpred}' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="temporal_only", table=primary, temporal=_tcol)
                _tick("sql_planning", "Narrowing to that time period")
                print(f"  [L4e] Temporal     {primary} WHERE {_tcol} in window"
                      "  — deterministic, no LLM")
            elif _want_rank_order and _tcol:
                # "latest 10 X" / "last 10 X" — the only temporal signal was a bare
                # recency word (or none at all), so there's no real date-RANGE filter
                # to apply (see _skip_vague_window above) — just ORDER BY the anchor's
                # canonical temporal column + LIMIT N. Deterministic, no LLM.
                # Business-facing SELECT list — NOT the validation allow-list.
                # allowed_columns (extended a few lines below with WHERE/JOIN-only
                # helper columns like _tcol) stays exactly what it was: the AST
                # firewall's allow-list. This is a SEPARATE, smaller list for what
                # actually gets projected — composed from metadata VEDA already
                # computed at ingestion + this query's own retrieval relevance
                # (veda/routing.py::recommended_projection — never re-ranked here).
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                sql = f'SELECT {_proj} FROM "{primary}"' + _rank_tail
                allowed_columns = allowed_columns + [_tcol]
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="ranked_temporal_only", table=primary,
                       temporal=_tcol, top_n=_rank.top_n, direction=_rank.direction)
                _tick("sql_planning", "Sorting and picking the top results")
                print(f"  [L4e] Ranked       {primary} ORDER BY {_tcol} "
                      f"{_rank.direction.upper()} LIMIT {_rank.top_n or 100}"
                      "  — deterministic, no LLM")
            elif _rank.ranked and _rank.basis == "metric" and _rank_sort_col \
                    and _rank_tail:
                # "top 3 vendors by rating" / "top 5 amenities by monthly fee" — a ranking on a
                # MEASURE column of the anchor (2026-09-15, M1 close-out battery). Only the
                # temporal-basis ranking had a branch; a metric-basis one fell to the generic
                # path and was refused on lite-model sources. _rank_sort_col already resolved
                # the column against the anchor (deterministic, schema-driven) and _rank_tail
                # is the ORDER BY … LIMIT N it built — nothing new is inferred here.
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col])
                _proj = ", ".join(f'"{c}"' for c in _proj_cols) or "*"
                sql = f'SELECT {_proj} FROM "{primary}"' + (f' WHERE {_tpred}' if _tpred else '') + _rank_tail
                allowed_columns = allowed_columns + [_rank_sort_col] + ([_tcol] if _tcol else [])
                _llm_sql = False                 # deterministic — skip IR-equivalence
                tr.set("sql_planning", action="ranked_metric_only", table=primary,
                       sort_col=_rank_sort_col, top_n=_rank.top_n, direction=_rank.direction)
                _tick("sql_planning", "Sorting and picking the top results")
                print(f"  [L4e] Ranked       {primary} ORDER BY {_rank_sort_col} "
                      f"{_rank.direction.upper()} LIMIT {_rank.top_n}  — deterministic, no LLM")
            else:
                # ENFORCEMENT: a temporal question on an anchor with NO date column cannot
                # be answered — refuse, rather than hand the LLM an impossible "date-filter
                # a dateless table" task (it invents a column like created_at, which only
                # gets caught downstream as a confusing 'unknown column' error). The
                # deterministic temporal path (elif _tpred) already serves anchors that DO
                # have one; this guards the LLM fallback. Refuse-over-guess.
                if tf and (tf.start or tf.end) and _tcol is None:
                    _msg = (f"'{primary}' has no date/time column, so the requested time "
                            f"filter ('{tf.start or ''}'..'{tf.end or ''}') can't be applied")
                    print(f"  [L5] Temporal refuse  {_msg}")
                    fb = _feedback("refuse", msg=_msg)
                    log_route("refuse", query, (time.time() - start) * 1000)
                    return _done(0, "refuse", msg=_msg, feedback=fb)
                tr.set("sql_planning", action="single_table", table=primary)
                _tick("sql_planning", "Building the query")
                _llm_sql = True
                # in-scope column glossary (business_definition + aliases) → SQL prompt hint
                _gloss = {}
                for _c in allowed_columns:
                    _m = sm.get("columns", {}).get(f"{primary}.{_c}", {}) or {}
                    if _m.get("aliases") or _m.get("business_definition"):
                        _gloss[_c] = {"aliases": _m.get("aliases") or [],
                                      "def": _m.get("business_definition") or ""}
                # domain_synonyms-driven phrase→column directives: a query phrase that the
                # model maps to a SPECIFIC in-scope column ("last logged in" → last_logged_in)
                # tells the LLM the exact column, so it can't pick a sibling (last_login).
                # Word-boundary, len≥4 phrases only (avoid 'in'/'log' over-matching). This is
                # what enforces correct column choice WITHOUT weakening qualifier_completeness.
                _term_map, _allowed_set = [], set(allowed_columns)
                for _phrase, _cks in (sm.get("domain_synonyms", {}) or {}).items():
                    if len(_phrase) < 4 or not re.search(rf"\b{re.escape(_phrase.lower())}\b", query.lower()):
                        continue
                    for _ck in (_cks or []):
                        _pt, _, _pc = _ck.partition(".")
                        if _pt == primary and _pc in _allowed_set:
                            _term_map.append((_phrase, _pc))
                t_sql = time.time()
                _proj_cols = recommended_projection(primary, allowed_columns, results, sm, query,
                                                    must_include=[_rank_sort_col] if _rank_sort_col else None)
                # Phase 1 ANALYTICAL_SQL_V2: use the deterministic structured analytical
                # SQL (built from the grounded spec) instead of the LLM — it flows through
                # the SAME validation + execution below. Falls back to the LLM otherwise.
                # M2: a grounded LIST spec (typed predicate on the anchor, no aggregate) —
                # the pipeline's own recommended projection + the grounded WHERE. Only when
                # the spec's anchor IS this branch's primary (no anchor override here).
                if (_analytical_sql is None and _analytical_spec is not None
                        and getattr(_analytical_spec, "aggregation", None) == "list"
                        and _analytical_spec.anchor == primary and _analytical_spec.where_sql):
                    _lp = ", ".join(f't0."{c}"' for c in (_proj_cols or [])) or "t0.*"
                    _analytical_sql = (f'SELECT {_lp} FROM "{primary}" t0 '
                                       f'WHERE {_analytical_spec.where_sql} LIMIT 100')
                    _wcols = [c for c in allowed_columns] + [f.column for f in []]
                sql = _analytical_sql or generate_sql(query, primary, allowed_columns, tf,
                                   col_glossary=_gloss, term_map=_term_map, time_col=_tcol,
                                   recommended_projection=_proj_cols,
                                   rank_sort_col=_rank_sort_col)
                if not _analytical_sql:
                    # generate_sql may have answered from its OWN deterministic builder
                    # (veda/generation._deterministic_single_table_sql) rather than the
                    # SLM. That output is deterministic and must be treated like every
                    # other deterministic branch — skip IR-equivalence, which otherwise
                    # audits it as if a model had written it.
                    try:
                        from veda.generation import last_was_deterministic as _lwd
                        if _lwd():
                            _llm_sql = False
                            print("  [L5] SQL gen       deterministic builder (no SLM)")
                    except Exception:
                        pass
                if _analytical_sql:
                    # deterministic build from a grounded spec — skip IR-equivalence like every
                    # other deterministic branch (it rejected the spec's own grounded filter
                    # as "not requested by the query", 2026-09-16) and allow the anchor's
                    # grounded columns through the firewall.
                    _analytical_used = True
                    _llm_sql = False
                    allowed_columns = list(dict.fromkeys(list(allowed_columns) + [
                        c.split(".", 1)[1] for c in all_cols if c.split(".", 1)[0] == primary]))
                print(f"  [L5] SQL gen       {time.time()-t_sql:.1f}s"
                      + ("  [AnalyticalV2 deterministic]" if _analytical_sql else "")
                      + (f"  (+{len(_gloss)} col defs)" if _gloss else "")
                      + (f"  (+{len(_term_map)} term→col)" if _term_map else ""))

    # Value validation: reject fabricated filter values (e.g. 'failed_review' on a
    # column that has no such value). Resolve each column to its table via the SQL's
    # own aliases; skip our deterministic polymorphic-predicate values.
    skip_values = set()
    if join_constraints:
        # Only a literal sitting ON a polymorphic-predicate column (e.g.
        # object_type = 'counterparty') is OUR deterministic value — skip just that
        # one from value-grounding. A literal on ANY other column is a user filter and
        # must still be grounded. (Previously this skipped every string literal whenever
        # a predicate existed, silently disabling value-grounding on all polymorphic
        # joins.)
        _pred_cols = {pc.lower() for pc in join_constraints.get("predicate_cols", set())}
        if _pred_cols:
            try:
                for eqp in _sg.parse_one(sql, read="postgres").find_all(_exp.EQ):
                    _c = eqp.this if isinstance(eqp.this, _exp.Column) else (
                        eqp.expression if isinstance(eqp.expression, _exp.Column) else None)
                    _l = eqp.expression if isinstance(eqp.expression, _exp.Literal) else (
                        eqp.this if isinstance(eqp.this, _exp.Literal) else None)
                    if (_c is not None and _l is not None and _l.is_string
                            and _c.name.lower() in _pred_cols):
                        skip_values.add(_l.name)
            except Exception:
                pass
    alias_to_table = {}
    try:
        for tnode in _sg.parse_one(sql, read="postgres").find_all(_exp.Table):
            if tnode.alias:
                alias_to_table[tnode.alias.lower()] = tnode.name
            alias_to_table[tnode.name.lower()] = tnode.name
    except Exception:
        pass
    _default_tbl = primary if (not from_cache and len(allowed_tables) == 1) else (
        next(iter(allowed_tables)) if len(allowed_tables) == 1 else None)
    _cols_meta_map = sm.get("columns", {})

    def _owning_table(col_name):
        # Unqualified column in a multi-table query: resolve only when exactly one
        # in-scope table owns a column of this name (unambiguous). Otherwise return
        # None so value-grounding safely skips it — never guesses a table.
        owners = [t for t in allowed_tables if f"{t}.{col_name}" in _cols_meta_map]
        return owners[0] if len(owners) == 1 else None

    def _resolve(colexp):
        if colexp.table:
            return alias_to_table.get(colexp.table.lower())
        return _default_tbl or _owning_table(colexp.name)

    _route = (fp.route if fp else "cache" if from_cache else
              "existence" if is_existence else f"full:{intent}")

    tr.set("schema_linking", selected_table=primary)

    # ── M3 checkpoint 1: ONE firewall over (IR, SQL) ──────────────────────────────
    # The IR is built from whatever structured state THIS head has (a complete IR from
    # the understanding candidate / fast path, else the branch's own filters/time/rank
    # with the aggregate & grouping slots partial → the text heuristics keep them). Gates
    # run in the same order they always did; the head reacts to the verdict exactly as
    # before (typed refusal, salvage, re-entry). `ir_partial` is traced so the count can
    # fall through checkpoint 2.
    from veda import firewall as _fw
    from veda.ir import from_grounded_intent as _ir_from_gi, from_query_intent as _ir_from_qi, \
        from_branch_state as _ir_from_branch, partial as _ir_partial
    if _analytical_used and _analytical_spec is not None and getattr(_u, 'anchor', None):
        _ir = _ir_from_gi(_u, _analytical_spec, head=("understanding.reentry" if _reentry else "understanding"))
    elif fp is not None:
        # the fast path stashes its QueryIntent on a request-scoped ContextVar
        # (query.fast_path._capture_intent / get_preserved_intent) — the IR reads it there
        from query.fast_path import get_preserved_intent as _gpi
        _pi = (_gpi() or {}).get("intent") if callable(_gpi) else None
        _ir = (_ir_from_qi(_pi, head=f"fast_path.{fp.route}") if _pi is not None
               else _ir_partial(f"fast_path.{fp.route}", primary))
    elif from_cache:
        _ir = _ir_partial("cache", primary)
    elif _llm_sql:
        _ir = _ir_partial("llm_sql", primary, known={"filters": [
            __import__("veda.ir", fromlist=["IRFilter"]).IRFilter(table=primary, column=f["column"], op=f.get("op", "="),
                                                                  value=f.get("value_norm", f.get("value")), grounding="value_arbiter")
            for f in (_arb_filters or [])]})
    else:
        # M3 checkpoint 2 (2026-09-23): the deterministic branches KNOW their own shape
        # — this file built the WHERE, the ORDER BY and the LIMIT a few hundred lines
        # above — so their IR is complete and firewall._ir_vs_sql can check it
        # structurally instead of falling back to the text heuristic. `complete=False`
        # meant every branch.* head was reported ir_partial and the structural check was
        # skipped on Tier-1 entirely. _ir_vs_sql only asserts that slots the IR KNOWS are
        # present in the SQL (it never rejects the SQL for carrying more), so completing
        # the IR can add refusals only where the SQL genuinely lost something the branch
        # asked for.
        #
        # The numeric predicates go in as extra_filters: without them the IR would not
        # know about the comparison this branch just built, and the structural check
        # would silently skip exactly the filter class that was being dropped.
        _ir_extra = ([{"column": f["column"], "op": f["op"], "value": f.get("value"),
                       "grounding": "numeric_filter"} for f in (_num_filters or [])]
                     + [{"column": f["column"], "op": "=", "value": f.get("value"),
                         "grounding": "value_glossary"} for f in (_vg_filters or [])])
        _ir = _ir_from_branch(f"branch.{_route}", primary, arb_filters=_arb_filters,
                              tpred_col=(_tcol if _tpred else None), tf=tf,
                              rank=_rank, rank_col=_rank_sort_col,
                              extra_filters=_ir_extra, complete=True)
    _ir_holder["ir"] = _ir
    _fv = _fw.check(_ir, sql, sm, query=query, allowed_tables=allowed_tables, allowed_columns=allowed_columns,
                    ctx=_ambient_ctx(), resolve_table=_resolve, skip_values=skip_values,
                    llm_generated=_llm_sql, tf=tf, join_constraints=join_constraints, fanout_guard=fanout_guard,
                    skip_predicate_cols=(join_constraints or {}).get("predicate_cols", set()),
                    run_alignment=False, run_ir_equivalence=False, run_rbac=False,
                    head=_ir.head, trace=tr, _semantic_only=True,
                    # WHOSE words the qualifier gate is about — the user's own message, not a
                    # head-rebuilt `query`. See veda/validation.py::qualifier_completeness.
                    user_message=_conv_user_message)
    tr.check("value_grounding", _fv.verdict != _fw.UNGROUNDED, "" if _fv.verdict != _fw.UNGROUNDED else str(_fv.detail))
    if _fv.verdict == _fw.UNGROUNDED:
        colname, val = _fv.detail
        fb = _feedback("ungrounded", column=colname, value=val)
        log_route(_route + ".ungrounded", query, (time.time() - start) * 1000)
        return _done(0, "ungrounded", detail=_fv.detail,
                     msg=f"value '{val}' not present in {colname}", feedback=fb)
    print("  [L6a] Value check  ✓  filter literals exist in the data")

    # HARD GUARD: the correctness gates MUST validate against the ORIGINAL query,
    # never the enhanced retrieval search string. Enhancement is a recall-only sidecar.
    assert query == getattr(tr, "sections", {}).get("query_understanding", {}).get("query", query), \
        "validation must run on the original query, not the enhanced search string"

    # Unified qualifier-completeness gate (all paths): refuse if the user named a
    # qualifier the SQL doesn't account for (a dropped filter → broader answer).
    ok_q = _fv.verdict != _fw.QUALIFIER_DROPPED
    missing = (_fv.detail if isinstance(_fv.detail, list) else [_fv.detail]) if not ok_q else []
    if not ok_q and _fv.slot and _fv.slot.startswith("filter:"):
        missing = [_fv.slot.split(":", 1)[1]]            # IR-vs-SQL: the dropped IR slot
    # `missing` is a LIST (it can name several dropped qualifiers), but every user-facing
    # site below interpolates ONE token into a sentence, and veda/feedback.py's helpers
    # (`_restricted_match`'s term.strip(), `_closest`'s name.lower()) are str-only. Passing
    # the list rendered "'['property']' doesn't match any value in this data" to the user
    # and raised AttributeError inside _feedback — swallowed at the `except Exception:
    # return None` above, so `feedback=None` reached the chat and it printed the generic
    # "Could you clarify what you're asking about?" instead. The list stays for the trace;
    # the scalar is what human-readable text and feedback get.
    missing_tok = missing[0] if missing else ""
    # ENTITY-NOUN RULE (2026-09-23): a token that grounds to a TABLE is ACCOUNTED, and is
    # never offered to the user as a missing VALUE. "property", "payments", "accounting
    # entries" name entities, not cell contents — asking "did you mean one of <values of
    # some column>?" about them is a category error, and it is what produced the
    # "'property' doesn't match any value in this data — did you mean one of , true
    # (asset_id)?" clarify on 5 of the 20 question.txt questions. "Grounds to a table"
    # means the CURATED alias glossary maps it — see the note below for why the fuzzier
    # referent-table evidence is not accepted here. (The FK-parent and glossary
    # vocabulary that stop most of these from reaching the gate at all live in
    # veda/validation.qualifier_completeness; this is the backstop for what slips past.)
    _entity_noun_ok = False
    if not ok_q and missing_tok:
        try:
            _tok = str(missing_tok).strip().lower()
            _grounds_to_table = False
            try:
                from query.entity_resolver import _entity_glossary
                _g = _entity_glossary() or {}
                _grounds_to_table = _tok in _g or _tok.rstrip("s") in _g or (_tok + "s") in _g
            except Exception:
                pass
            # DELIBERATELY glossary-only. The first version also accepted
            # `resolution.referent_tables(tok)` as proof that a token "names a table",
            # but that function credits a table for VALUE evidence too — so "high" in
            # "show tickets with high priority" resolved to referent tables, was
            # reclassified as an entity noun, and the dropped `priority = 'high'` filter
            # sailed through the gate into an unfiltered 100-row answer (caught by the
            # per-source battery, which asserts that question must REFUSE because no
            # such priority value exists). The curated glossary contains entity nouns
            # only, so it cannot make that mistake; a genuine value keeps going down the
            # value_referents path below, which is where it belongs.
            if _grounds_to_table:
                print(f"  [L6b] Qualifier    ✓  '{_tok}' names an ENTITY (table), not a "
                      f"value — accounted")
                tr.check("qualifier_completeness", True,
                         f"'{_tok}' grounds to a table (entity noun)")
                ok_q, missing, missing_tok, _entity_noun_ok = True, [], "", True
        except Exception:
            pass
    if not _entity_noun_ok:
        tr.check("qualifier_completeness", ok_q, "" if ok_q else str(missing))
    if not ok_q:
        _sql_tabs = set(re.findall(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)', sql))
        # QUALIFIER SALVAGE (generic, schema-agnostic): before refusing, ask QSR what
        # `missing` IS in this scope. Referent tables entirely OUTSIDE the SQL mean
        # the ANCHOR was wrong, not the query ("payment" refused against a
        # document-type table while a payment table exists) — retry ONCE with that
        # table forced as primary; the retried plan faces every gate above, including
        # this one (anchor_hint marks the retry, so salvage can't recurse). Runs only
        # on would-be refusals — an answered query can never regress through here.
        _refs = []
        try:
            from config import (QUALIFIER_SALVAGE_ENABLED, QUALIFIER_REANCHOR_RETRY,
                                QUALIFIER_REANCHOR_MAX_HEAD_S)
        except Exception:
            QUALIFIER_SALVAGE_ENABLED, QUALIFIER_REANCHOR_RETRY = True, True
            QUALIFIER_REANCHOR_MAX_HEAD_S = 45.0
        if QUALIFIER_SALVAGE_ENABLED and anchor_hint is None:
            try:
                from query.resolution import referent_tables
                _refs = [r for r in referent_tables(missing_tok, sm)
                         if r["table"] not in _sql_tabs]
                # Anchor preference: a table backed by ENTITY/COLUMN-NAME evidence
                # beats a value-only home — the latter is usually a shared label
                # store ('payment' exists as a ROW in list_of_values), which is
                # filter evidence, not an anchor. Measured on the trigger query:
                # retrying against the label store refuses; retrying against the
                # entity table lands the grounded domain clarify.
                _refs.sort(key=lambda r: (
                    not any("entity" in w or "column-name" in w for w in r["why"]),
                    -r["score"], r["table"]))
            except Exception:
                _refs = []
            if (_refs and QUALIFIER_REANCHOR_RETRY
                    and (time.time() - start) <= QUALIFIER_REANCHOR_MAX_HEAD_S):
                print(f"  [L6b] Qualifier salvage  '{missing_tok}' → {_refs[0]['table']} "
                      f"({'; '.join(_refs[0]['why'][:2])}) — re-anchored retry")
                tr.note("validation", f"qualifier salvage retry → {_refs[0]['table']}")
                try:
                    _retry = run_query(query, sm, all_cols, return_result=True,
                                       anchor_hint=_refs[0]["table"])
                except Exception:
                    _retry = None
                # Surface the retry when it ANSWERED, or when it refused with
                # something MORE grounded than the original qualifier_dropped:
                # a clarify (grounded question) or an ungrounded value on the
                # re-anchored table ("'completed' is not a value of
                # payment_status — did you mean captured/authorized/…") — the
                # value-level diagnosis on the RIGHT anchor is strictly more
                # actionable than an entity-level clarify. Any other refusal
                # falls through to the referent clarify below.
                if isinstance(_retry, dict) and (_retry.get("ok")
                        or _retry.get("status") in ("clarify", "ungrounded")):
                    try:
                        _retry.setdefault("salvage", {
                            "reanchored_to": _refs[0]["table"],
                            "dropped_qualifier": missing})
                    except Exception:
                        pass
                    log_route(_route + ".salvage_reanchor", query,
                              (time.time() - start) * 1000)
                    return _retry if return_result else 0
        # Grounded clarify upgrade: when the dropped token is NOT a real data value
        # anywhere (no direct/closed referent) and the queried table exposes FK label
        # domains, the honest answer is the domain, not a generic refusal —
        # "'completed' doesn't match any payment status; statuses here are captured /
        # authorized / cancelled." Clarify is terminal (never retried by Tier-2).
        try:
            from query.resolution import value_referents, domain_via
            from query.join_planner import load_graph
            _vr = value_referents(missing_tok)
            if not _vr["direct"] and not _vr["closed"]:
                _qw = set(re.findall(r"[a-z]+", query.lower()))
                _doms = []
                for _e in load_graph().get("edges", []):
                    if _e.get("source_table") in _sql_tabs and _e.get("cardinality") == "N:1":
                        _d = domain_via(_e["source_table"], _e["source_column"], limit=6)
                        if len(_d) > 1:
                            # rank by how much the FK column's own words overlap the
                            # query ("payment_status_id" for "…completed payments")
                            _ov = len(_qw & {w for w in _e["source_column"].split("_")
                                             if len(w) > 2})
                            _doms.append((-_ov, _e["source_column"], _d))
                if _doms:
                    _doms.sort()
                    _, _col, _d = _doms[0]
                    _msg = (f"'{missing_tok}' doesn't match any value in this data — "
                            f"did you mean one of {', '.join(_d[:5])} ({_col})?")
                    fb = _feedback("clarify", msg=_msg)
                    log_route(_route + ".grounded_clarify", query, (time.time() - start) * 1000)
                    return _done(0, "clarify", msg=_msg, feedback=fb)
        except Exception:
            pass
        if _refs:
            # Referent clarify: the retry didn't rescue it (or is off/over budget),
            # but QSR knows what the token IS here — tell the user, grounded in the
            # schema's own vocabulary, instead of "couldn't map 'X'".
            try:
                from query.superlative_plan import _human
                _names = list(dict.fromkeys(_human(r["table"], sm) for r in _refs[:2]))
            except Exception:
                _names = list(dict.fromkeys(r["table"] for r in _refs[:2]))
            _msg = (f"'{missing_tok}' here refers to {' / '.join(_names)}, which this "
                    f"answer never touched — ask about {_names[0]} directly, or say "
                    f"how '{missing_tok}' relates to your question.")
            fb = _feedback("clarify", msg=_msg)
            log_route(_route + ".salvage_clarify", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=_msg, feedback=fb)
        # The dropped qualifier itself may be an innocent word ("exist") that has
        # nothing to do with RBAC — but if the SQL this query would have run was
        # already reaching into a table/column RBAC restricts, that's the real
        # reason to refuse, and the user should hear "access denied", not a
        # generic clarify prompt that reads as if the table doesn't exist.
        _restricted_here = restricted_names(sm, _ambient_ctx())
        if any(t in _sql_tabs for t in _restricted_here["tables"]):
            fb = _feedback("access_denied")
            log_route(_route + ".access_denied", query, (time.time() - start) * 1000)
            return _done(0, "access_denied", missing=missing_tok, feedback=fb)
        fb = _feedback("qualifier_dropped", missing=missing_tok)
        # M2: a dropped qualifier on a deterministic answer ("how many amenity CATEGORIES"
        # → bare COUNT(*)) is a refused branch — the grounded candidate (COUNT DISTINCT)
        # gets its turn via re-entry when the layer is on; refusal stands otherwise.
        return _guard_refusal("qualifier_dropped", None, fb, _route + ".qualifier_dropped",
                              always_done=True, missing=missing_tok)  # this site always returned _done()
    print("  [L6b] Qualifier    ✓  every named qualifier is represented in the SQL")

    # Entity COVERAGE (flag-gated, never refuses): the companion to the qualifier gate above for the
    # other half of "nothing the user asked for was silently dropped". That gate owns filters and the
    # attributes of the QUERIED tables; this one owns whole ENTITIES the question named that the SQL
    # never reached ("updates, assignees and attachments" answered with updates alone). The answer is
    # still correct for what it covers, so it ships — but as a failed check naming what was left out,
    # and at a capped confidence, instead of "no requested filters were ignored" at 1.0.
    if sql:
        try:
            from veda.intent_sql_alignment import entity_coverage
            _cov_ok, _cov_missing, _cov_terms = entity_coverage(query, sql, sm)
        except Exception:
            logger.exception("entity_coverage failed — coverage check omitted")
            _cov_ok, _cov_missing, _cov_terms = True, [], []
        if not _cov_ok:
            _coverage["missing"] = _cov_missing
            # The user's own words for what was dropped — the summariser needs these, it
            # never sees a table name (see run_nl_answer's `not_covered`).
            _coverage["terms"] = _cov_terms
            tr.check("entity_coverage", False, "not covered: " + ", ".join(_cov_missing))
            print(f"  [L6c] Coverage    ⚠  partial — not covered: {', '.join(_cov_missing)}")
        else:
            tr.check("entity_coverage", True, "")

    # Grouped-intent shape guard (flag-gated): an LLM-written PURE PROJECTION for a "how many X by Y" /
    # "distribution" query can't answer the grouping, and the NL summariser then fabricates a
    # distribution ("62% in Singapore"). Refuse instead of answering wrong. Deterministic grouped SQL
    # carries a GROUP BY and passes; a filter ("by <person>") carries a WHERE and passes.
    if _llm_sql and not grouped_shape_ok(query, sql):
        _gmsg = ("This asks for a per-group breakdown, but the query I built lists rows without "
                 "grouping them — please name the column to group by (e.g. 'by city').")
        fb = _feedback("clarify", msg=_gmsg)
        log_route(_route + ".grouped_shape_mismatch", query, (time.time() - start) * 1000)
        return _done(0, "clarify", msg=_gmsg, feedback=fb) if return_result else 0

    # Uniqueness-intent shape guard (flag-gated): an LLM query for "how many UNIQUE/DISTINCT X" that
    # drops the DISTINCT and returns a plain COUNT(*) answers total rows, not distinct values — the
    # summariser then reports that total as the unique count. Refuse. Correct SQL carries a
    # COUNT(DISTINCT …)/SELECT DISTINCT or a GROUP BY and passes.
    if _llm_sql and not distinct_shape_ok(query, sql):
        _dmsg = ("This asks for a count of distinct values, but the query I built counts rows without "
                 "de-duplicating — please confirm the column to count distinct values of.")
        fb = _feedback("clarify", msg=_dmsg)
        log_route(_route + ".distinct_shape_mismatch", query, (time.time() - start) * 1000)
        return _done(0, "clarify", msg=_dmsg, feedback=fb) if return_result else 0

    # Ranking-intent shape guard: the query named an explicit count of RANKED rows ("top 5",
    # "latest 10") and the SQL returns that many rows in no order, so they are an arbitrary N that
    # the summariser then presents as the ranked ones. Runs on EVERY produced SQL — deterministic,
    # LLM and verified-cache replay alike — because the deterministic branch is one of the two
    # producers that reached it (its ORDER BY needs a single unambiguous measure column, and an
    # anchor naming two drops the ranking silently). The measure candidates the anchor DOES name
    # are read back out of the semantic model so the question names real columns to choose between
    # rather than asking the user to guess.
    if sql:
        _ok_rank, _why_rank = ranked_shape_ok(query, sql)
        if not _ok_rank:
            _anchor_t = _anchor_from_sql(sql)
            _meas = ((sm.get("tables", {}).get(_anchor_t, {}) or {})
                     .get("candidate_measure_columns") or [])
            _choice = (" — rank by " + " or ".join(_meas) + "?") if _meas else \
                      " — please name the column to rank by."
            _rmsg = f"{_why_rank}{_choice}"
            fb = _feedback("clarify", msg=_rmsg)
            log_route(_route + ".ranked_shape_mismatch", query, (time.time() - start) * 1000)
            return _done(0, "clarify", msg=_rmsg, feedback=fb) if return_result else 0

    # CONVERSATION STATE PRESERVATION. A follow-up arrives with the narrowing the
    # conversation has already established. If the SQL built for it keeps NONE of that
    # narrowing, the turn has quietly widened the question back out — and the summariser
    # then reports figures for a population the user stopped asking about two turns ago.
    #
    # Measured 2026-09-24: "only the gated ones", after the conversation had narrowed to
    # Nagpur and EAST-facing, produced `SELECT "is_gated" FROM "assets_asset" LIMIT 1000` —
    # no GROUP BY, neither filter, and an answer delivered with full confidence. The
    # boolean could not be grounded, and instead of saying so the turn threw the
    # conversation away. Refusing names what could not be kept, which is something the
    # user can act on; answering does not.
    #
    # Only fires when the turn is ON the conversation's own table and keeps NOT ONE of the
    # remembered filters. A turn that keeps some of them is a legitimate replacement
    # ("what about Mumbai" swaps a value), and a turn on a different table is a topic
    # change, which is allowed to drop everything.
    if sql and _conv_filters and _anchor_from_sql(sql) == _conv.get("entity_table"):
        _kept = {c for c in (f.get("column") for f in _conv_filters) if c and f'"{c}"' in sql}
        if not _kept:
            _lost = ", ".join(sorted({str(f.get("column")) for f in _conv_filters
                                      if f.get("column")}))
            _cmsg = (f"I couldn't keep the narrowing we'd already applied ({_lost}), so "
                     f"I'd rather not show figures for everything. Ask this as a new "
                     f"question if you meant to start over.")
            fb = _feedback("clarify", msg=_cmsg)
            log_route(_route + ".drill_state_lost", query, (time.time() - start) * 1000)
            print(f"  [conversation] refused: the query dropped every remembered filter "
                  f"({_lost})")
            return _done(0, "clarify", msg=_cmsg, feedback=fb) if return_result else 0

    # Intent↔SQL referent alignment (flag-gated): a generalized comparator (Option B, increment 1) — SQL
    # that groups/anchors on a schema element the question does NOT refer to (a per-time breakdown grouped
    # by a non-temporal column; a superlative anchored on the wrong table) answers a different question.
    # Refuse. Runs on EVERY produced SQL — LLM, verified-cache replay, AND the DETERMINISTIC path — because
    # each can misalign: #14 (cache replay), #13 ("leads per month" grouped by lead_stage) is produced by
    # the deterministic join planner, disproving "deterministic is aligned by construction". The check is a
    # no-op when the flag is off or the referents align, so running it universally is byte-identical-safe.
    # Alignment guards (temporal / entity-anchor / aggregate presence / filter presence /
    # dimension) — the firewall's alignment stage. Reactions preserved verbatim: each kind
    # keeps its own message, `what` hint and route tail; a refused deterministic answer
    # still goes through _guard_refusal (M2 re-entry).
    if sql:
        _fa = _fw.check(_ir, sql, sm, query=query, allowed_tables=allowed_tables, allowed_columns=allowed_columns,
                        ctx=_ambient_ctx(), resolve_table=_resolve, skip_values=skip_values, llm_generated=_llm_sql,
                        tf=tf, run_alignment=True, run_ir_equivalence=False, run_rbac=False,
                        head=_ir.head, trace=tr, _semantic_only=True, _skip_value_and_qualifier=True)
        if _fa.verdict == _fw.SHAPE_MISMATCH:
            _k = _fa.shape_kind
            if _k == "alignment":
                _amsg = f"{_fa.reason}. Please rephrase or confirm the exact column to use."
                fb = _feedback("clarify", msg=_amsg)
                return _guard_refusal("clarify", _amsg, fb, _route + ".intent_sql_misalignment")
            if _k == "aggregate_omission":
                fb = _feedback("clarify", msg=_fa.reason,
                               what="Try asking for one figure at a time — a count, or a total.")
                return _guard_refusal("clarify", _fa.reason, fb, _route + ".aggregate_omission")
            if _k == "filter_omission":
                fb = _feedback("clarify", msg=_fa.reason,
                               what="Try naming the field and the value to compare it against.")
                return _guard_refusal("clarify", _fa.reason, fb, _route + ".filter_omission")
            if _k == "dimension":
                from veda.intent_sql_alignment import DIM_REFUSE
                fb = _feedback("clarify", msg=_fa.reason)
                return _guard_refusal("clarify", _fa.reason, fb,
                                      _route + (".dimension_misalignment" if _fa.dim_out == DIM_REFUSE
                                                else ".dimension_ambiguous"))
            # IR-vs-SQL structural mismatch on a complete IR (group key / aggregate / limit /
            # distinct absent from the SQL): same clarify contract, its own route tail.
            fb = _feedback("clarify", msg=_fa.reason)
            return _guard_refusal("clarify", _fa.reason, fb, _route + f".ir_shape_{_k or 'mismatch'}")

    # Canonical-QueryIntent SHADOW measurement (flag-gated, OBSERVE-ONLY, Phase 1). When the fast-path
    # DECLINED (fp is None) but preserved a QueryIntent, compare it to the final SQL's referents and LOG
    # field-level agreement. Never rejects/alters SQL or the response — pure measurement. Fully guarded.
    if fp is None and sql:
        try:
            from veda.canonical_intent_shadow import record_shadow
            record_shadow(query, sql, sm)
        except Exception:
            pass

    # IR equivalence — the firewall's stage 4 (LLM SQL must not add semantics the
    # question never asked for; deterministic builds skip it inside the gate).
    _fe = _fw.check(_ir, sql, sm, query=query, allowed_tables=allowed_tables, allowed_columns=allowed_columns,
                    ctx=_ambient_ctx(), resolve_table=_resolve, skip_values=skip_values, llm_generated=_llm_sql,
                    tf=tf, skip_predicate_cols=(join_constraints or {}).get("predicate_cols", set()),
                    run_alignment=False, run_ir_equivalence=True, run_rbac=False,
                    head=_ir.head, trace=tr, _semantic_only=True, _skip_value_and_qualifier=True)
    tr.check("ir_equivalence", _fe.verdict != _fw.IR_MISMATCH, _fe.reason if _fe.verdict == _fw.IR_MISMATCH else "")
    if _fe.verdict == _fw.IR_MISMATCH:
        fb = _feedback("ir_mismatch", msg=_fe.reason)
        log_route(_route + ".ir_mismatch", query, (time.time() - start) * 1000)
        return _done(0, "ir_mismatch", msg=_fe.reason, feedback=fb)
    if _llm_sql:
        print("  [L6b+] IR check    ✓  no unrequested filters / joins / grouping / ordering")

    # ── Shared analytical-semantics check (advisory): the SAME generic, metadata-
    # driven invariants Tier-2/LangGraph use (veda/semantic_validation.py) — requested
    # operator preserved, group-by present, user-facing dimension not an unnecessary
    # identifier. Recorded to the trace for observability + "why" provenance; it does
    # NOT block (deterministic Tier-1 SQL is already correct by construction; the LLM
    # branch gets an early signal). graph=None here — join grounding is already
    # enforced upstream by join_constraints + ir_equivalence, so we skip the graph
    # load. Fully try/except'd: a check failure can never fail a query.
    try:
        from config import SEMANTIC_VALIDATION_ENABLED as _SV_ON
    except Exception:
        _SV_ON = False
    if _SV_ON:
        try:
            from veda.semantic_validation import validate_analytical_semantics
            _sv = validate_analytical_semantics(query, sql, sm, graph=None)
            if _sv:
                tr.set("semantic_validation", findings=_sv)
                for _f in _sv:
                    tr.note("semantic_validation",
                            f"{_f['severity']}: {_f['code']} — {_f['detail']}")
                print(f"  [L6d] Semantics    {len(_sv)} advisory finding(s): "
                      f"{', '.join(sorted({f['code'] for f in _sv}))}")
        except Exception:
            logger.debug("semantic_validation (Tier-1) skipped", exc_info=True)

    # Gate 1 (User Story 3, Task 16 follow-up) — the centralized final RBAC gate.
    # Whatever path produced allowed_tables/allowed_columns (FastPath, verified-
    # cache replay, single/multi-table planning, FK/entity expansion) is narrowed
    # to what ctx.allowed_resources permits, right before validate_and_parameterize
    # enforces it as a hard allowlist — see veda.rbac_filter's module docstring for
    # why this ONE gate, not patching every discovery site. A no-op when the
    # ambient context carries no allowed_resources.
    #
    # `_restricted_for_sql` is captured BEFORE narrowing so a rejection below can
    # tell "the SQL referenced something RBAC just removed" apart from "the SQL
    # is wrong for an unrelated reason" — narrow_allowed() itself only returns
    # the narrowed sets, not what it took out.
    _restricted_for_sql = restricted_names(sm, _ambient_ctx())
    _tables_before = set(allowed_tables or ())
    _fp_v = _fw.check(_ir, sql, sm, query=query, allowed_tables=allowed_tables, allowed_columns=allowed_columns,
                      ctx=_ambient_ctx(), resolve_table=_resolve, skip_values=skip_values, llm_generated=_llm_sql,
                      tf=tf, join_constraints=join_constraints, fanout_guard=fanout_guard,
                      run_alignment=False, run_ir_equivalence=False, run_rbac=True,
                      head=_ir.head, trace=tr, _skip_value_and_qualifier=True)
    allowed_tables, allowed_columns = _fp_v.allowed_tables, _fp_v.allowed_columns
    # Say so when the gate actually took something away. Without this, an RBAC
    # narrowing and an ordinary planning miss are indistinguishable in the engine
    # log: the only visible trace was validate_and_parameterize's downstream
    # "references unknown table(s)", which reads as a hallucinated table rather
    # than a permission decision (and left an intermittent false denial
    # undiagnosable — the scope in play was recorded nowhere).
    _tables_removed = sorted(_tables_before - set(allowed_tables or ()))
    if _tables_removed:
        _ctx_dbg = _ambient_ctx()
        print(f"  [RBAC] narrowed allowlist — removed table(s) {_tables_removed} "
              f"(source_ids={list(getattr(_ctx_dbg, 'source_ids', ()) or ())}, "
              f"restricted={_restricted_for_sql['tables']})")

    param_sql, params, err = ((_fp_v.sql, _fp_v.params, None) if _fp_v.ok
                              else (None, [], _fp_v.reason))
    tr.check("ast_readonly_parameterized_fanout", not err, err or "")
    try:  # one validation event carrying the PASS/FAIL rollup, not the AST detail
        from veda import lifecycle as _lcv
        _tlv = _lcv.current_timeline()
        if getattr(_tlv, "enabled", False):
            _checks = tr.sections.get("validation", {}).get("checks", []) or []
            _failed = [c for c in _checks if c.get("status") != "pass"]
            if _failed:
                _tlv.failed(_lcv.PHASE_VALIDATION)
            elif _checks:
                _tlv.completed(_lcv.PHASE_VALIDATION, "Safety checks passed")
    except Exception:
        pass
    if not err and getattr(tr, "enabled", False):
        _jc = tr.sections.get("join_planning", {}).get("confidence")
        tr.set("output", sql=param_sql, params=[str(x) for x in (params or [])],
               confidence=_jc if _jc is not None else 1.0)
    if err:
        print(f"\n❌ [L6] Validation rejected the SQL: {err}\n      raw: {sql}\n")
        log_route(_route + ".invalid", query, (time.time() - start) * 1000, error=err)
        # Was this rejection RBAC-caused? The AST error text isn't something to
        # parse (format is validate_and_parameterize's, not ours to depend on) —
        # instead check whether the generated SQL actually references a name
        # RBAC just stripped. Identifier-anchored, quoted OR bare: the replay path
        # feeds unquoted SQL, and a quoted-only test silently turned every RBAC
        # denial there into a generic refusal (see _sql_references).
        _restricted_hit = any(
            _sql_references(sql, name)
            for name in _restricted_for_sql["tables"] + _restricted_for_sql["columns"])
        if _restricted_hit:
            fb = _feedback("access_denied")
            return _done(1, "invalid", error=err, feedback=fb)
        # Validation rejections used to exit with NO feedback object, so the chat had
        # nothing to render and printed the contentless "Could you clarify what you're
        # asking about?" (5 of the 6 generic refusals in the 2026-09-23 question.txt run
        # logged `validation: failed` right before that text). explain_failure's
        # invalid/exec_error branch already has the right words — attach them.
        return _done(1, "invalid", error=err, feedback=_feedback("invalid", error=err))
    _np = len(params) if params else 0
    print(f"  [L6c] Validate     ✓  read-only · parameterized ({_np} bound value"
          f"{'' if _np == 1 else 's'}) · AST/coverage/fan-out checked")

    print("\n  Generated SQL (parameterized):")
    print("  " + "-" * 74)
    for line in param_sql.splitlines():
        print(f"    {line}")
    if params:
        print(f"    -- params: {params}")
    print("  " + "-" * 74)

    print("  [L7] Execute       read-only connection · 30s timeout · fetch ≤20")
    # Live-verification finding: the deterministic single-source path never reached
    # source_coordinator.execute_decision, so it produced NO data_retrieval phase and
    # NO per-source record — on the most common path of all. Both are opened here, at
    # the one place this path actually touches a source.
    _dr_rec = None
    try:
        from veda import exec_records as _er2
        from veda_core.context import try_current as _tc2
        _ctx2 = _tc2()
        _sid2 = getattr(_ctx2, "source_id", None) if _ctx2 is not None else None
        if _sid2 is None:
            # A multi-source scope pins no single source on the context, so this
            # used to record NOTHING — and with no record, build_data_sources falls
            # back to the ROUTING decision, which names the source that was SELECTED,
            # not the one that produced the rows. Measured live: routing picked the
            # document store, the SQL ran against amenity data, and the payload told
            # the user the answer came from the document store.
            try:
                _routed = (tr.sections.get("routing") or {}).get("source_ids")
                if _routed:
                    _sid2 = _routed[0]
            except Exception:
                pass
        if _sid2 is not None:
            _dr_rec = _er2.current_recorder().open(
                _sid2, source_type="relational", engine="deterministic_sql", required=True)
    except Exception:
        _dr_rec = None
    cols, rows, err = execute_sql(param_sql, params)
    try:
        if _dr_rec is not None:
            from veda import exec_records as _er3
            _er3.current_recorder().close(
                _dr_rec,
                _er3.FAILED if err else _er3.COMPLETED,
                rows=(len(rows) if rows is not None else None),
                error=err or None)
    except Exception:
        pass
    if err:
        print(f"\n❌ [L7] Execution error: {err}\n")
        log_route(_route + ".exec_error", query, (time.time() - start) * 1000, error=err)
        return _done(1, "exec_error", error=err,
                     feedback=_feedback("exec_error", error=err))

    print(f"\n  Result: {len(rows)} rows (showing up to 20)\n")
    if cols:
        print("    " + " | ".join(str(c) for c in cols))
        print("    " + "-" * 74)
        for row in rows:
            cells = [("" if v is None else str(v))[:22] for v in row]
            print("    " + " | ".join(cells))

    # L7b — NL-back summarisation: turn the result rows into a one-line prose answer
    # (the SQL path otherwise returns only a table). Gated by NL_ANSWER_ENABLED; uses
    # the local SLM with a deterministic row-count fallback if Ollama is unavailable.
    # execute_sql returns tuples → zip to the dicts run_nl_answer expects.
    try:
        from config import (NL_ANSWER_ENABLED, NL_ANSWER_FAST_TIMEOUT_MS,
                            NL_SUMMARY_TIMEOUT_MS,
                            INSIGHT_ENGINE_ENABLED, RESULT_ANALYZER_MAX_ROWS)
    except Exception:
        NL_ANSWER_ENABLED = False
        NL_ANSWER_FAST_TIMEOUT_MS = 800
        NL_SUMMARY_TIMEOUT_MS = 10000
        INSIGHT_ENGINE_ENABLED = False
        RESULT_ANALYZER_MAX_ROWS = 200
    if not summarise:
        # Caller writes its own prose — don't spend a summary-class SLM call on one
        # that will be discarded. deterministic_fallback_answer() below still fills
        # `answer`, so the result dict keeps its shape for every other consumer.
        INSIGHT_ENGINE_ENABLED = False
    nl_answer_text = None
    _insight_extra = {}
    _summary_engine = None            # which summariser produced the prose (trace)
    if NL_ANSWER_ENABLED and cols is not None:
        row_dicts = [dict(zip(cols, r)) for r in rows]
        # F6: don't block on the SLM prose call. Compute the safe fallback now;
        # the caller (run_query) still returns promptly even if the SLM is slow.
        from query.nl_answer import deterministic_fallback_answer
        nl_answer_text = deterministic_fallback_answer(query, list(cols), row_dicts)

        # Deterministic analytics (ALWAYS, not flag-gated): the ONE post-execution
        # analysis pass — column stats/roles, result shape, business patterns,
        # chart candidates, grounding metadata. Pure Python over ≤RESULT_ANALYZER_
        # MAX_ROWS sampled rows, zero LLM, zero new SQL. Its JSON-safe summary
        # rides the result dict (same channel as `explain`) so every downstream
        # consumer — api-tier visualization included — reads this single
        # computation instead of re-deriving its own. Enrichment only: cols/rows
        # are never modified. Only the SLM narrative (Insight Engine) below stays
        # gated behind INSIGHT_ENGINE_ENABLED.
        _ictx = None
        try:
            from veda.result_analyzer import analyze_result, analytics_summary
            # Reuse metadata already computed upstream this same run — never a
            # second reasoning pass: intent (L4 IntentDetector) and the anchor/
            # join gating confidence already surfaced into the trace.
            _anchor_conf = tr.sections.get("anchor_selection", {}).get("confidence")
            _join_conf = tr.sections.get("join_planning", {}).get("confidence")
            _conf_inputs = {k: v for k, v in
                           (("anchor", _anchor_conf), ("join", _join_conf)) if v is not None}
            _ictx = analyze_result(query, param_sql, list(cols), row_dicts, sm=sm,
                                   table=str(primary), max_rows=RESULT_ANALYZER_MAX_ROWS,
                                   query_intent=intent, confidence_inputs=_conf_inputs,
                                   params=params)
            _insight_extra["analytics"] = analytics_summary(_ictx)
        except Exception as _ae:
            print(f"  [L7b] Analytics    (skipped: {type(_ae).__name__}: {_ae})")

        # Insight Engine (additive, flag-gated): ONE combined SLM call producing
        # summary + insights + visualization suggestion + follow-ups, REPLACING
        # (not layering on top of) the plain NL-answer SLM call below — never both,
        # so there is still only one post-query SLM call either way. Consumes the
        # SAME InsightContext computed above — never a second analysis pass.
        # The top-2 detected patterns, handed to whichever summary SLM runs so it
        # WEAVES them into the prose (natural insight, not a bolted-on suffix).
        _pattern_details = ([p.detail for p in _ictx.patterns[:2]]
                            if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        # ALL verified findings for the summarizer (it selects how many to narrate by
        # mode); the top-2 above are only the deterministic-fallback blend.
        _all_findings = ([p.detail for p in _ictx.patterns]
                         if (_ictx is not None and getattr(_ictx, "patterns", None)) else [])
        # Resolved analytical context — REUSED from this run's own understanding
        # (operation via the canonical aggregate normalizer, ranking column, temporal
        # window, explicit-id request), never re-derived in the summary layer. Lets the
        # narrator speak to the user's actual intent + preserve explicit id requests.
        _analytical_ctx = None
        try:
            from veda.planning import aggregate_operator as _agg_op
            from veda.semantic_validation import user_requested_identifier as _uri
            _analytical_ctx = {
                "intent": intent,
                "operation": _agg_op(query),
                "ranking": _rank_column_for_nl,
                "temporal": (f"{tf.start} to {tf.end}" if (tf and (tf.start or tf.end)) else None),
                "explicit_identifier": _uri(query),
            }
        except Exception:
            _analytical_ctx = None
        _slm_wove_patterns = False   # did a summary SLM already phrase the findings?
        # Did this result fill its fetch limit? If so row_count is a FLOOR, and the
        # summariser must say "at least N" instead of presenting the cap as the total.
        # Read off the executed SQL's own AST — deterministic, no LLM, no extra query.
        _truncated, _fetch_limit = False, None
        try:
            from veda.business_explain import extract_sql_facts as _esf
            _fetch_limit = (_esf(param_sql or "") or {}).get("limit")
            _truncated = bool(_fetch_limit and len(rows) >= int(_fetch_limit))
        except Exception:
            _truncated, _fetch_limit = False, None

        if INSIGHT_ENGINE_ENABLED and _ictx is not None:
            try:
                from query.result_explainer import run_insight_engine
                insight = run_insight_engine(_ictx, rank_column=_rank_column_for_nl)
                if getattr(insight, "answer", None):
                    nl_answer_text = insight.answer
                    _slm_wove_patterns = True   # its prompt already grounds on the patterns_block
                    _summary_engine = "run_insight_engine"
                    print(f"\n  [L7b] Insight      {insight.answer}")
                _insight_extra.update({          # update, not reassign — keeps "analytics"
                    "insights": insight.insights,
                    "follow_up_questions": insight.follow_up_questions,
                    "visualization": insight.visualization,
                    "confidence": insight.confidence,
                })
            except Exception as _ie:
                print(f"  [L7b] Insight Engine unavailable ({type(_ie).__name__}: {_ie}) "
                      f"— falling back to plain NL answer")
                # Record the attempt even though it failed — otherwise usage.total_tokens
                # == 0 is indistinguishable from "no LLM call was needed this turn" vs.
                # "the SLM call was attempted and timed out/errored before returning"
                # (call_slm() only records usage on a SUCCESSFUL backend.call() return —
                # an exception means nothing was recorded for this attempt at all).
                tr.set("nl_summary", insight_engine_failed=True,
                       insight_engine_error=f"{type(_ie).__name__}: {str(_ie)[:200]}")
                INSIGHT_ENGINE_ENABLED = False   # this turn only — fall through below

        if not INSIGHT_ENGINE_ENABLED and summarise:
            try:
                from query.nl_answer import run_nl_answer
                nl = run_nl_answer(query, list(cols), row_dicts,
                                   # 7B instruct summary (NL_SUMMARY_MODEL) needs the
                                   # full summary budget, not the 1.5B-era fast timeout.
                                   timeout=NL_SUMMARY_TIMEOUT_MS / 1000.0,
                                   # so the narrator can tell a LIMIT-filled page from a
                                   # complete result (config.SUMMARY_TRUNCATION_AWARE_ENABLED)
                                   sql=param_sql,
                                   table=str(primary), semantic_model=sm,
                                   rank_column=_rank_column_for_nl,
                                   patterns=_all_findings,
                                   result_shape=getattr(_ictx, "result_shape", None),
                                   analytical_context=_analytical_ctx,
                                   truncated=_truncated, fetch_limit=_fetch_limit,
                                   not_covered=_coverage.get("terms"))
                if getattr(nl, "answer", None):
                    nl_answer_text = nl.answer
                    # run_nl_answer wove the patterns only when the SLM actually ran;
                    # on its deterministic fallback it already blended them itself.
                    _slm_wove_patterns = True
                    _summary_engine = "run_nl_answer"
                    print(f"\n  [L7b] Answer       {nl.answer}")
            except Exception as _nle:
                print(f"  [L7b] Answer       (summarisation skipped: {type(_nle).__name__})")

        # Fold the deterministic analytics into the FINAL summary ONLY if no summary
        # SLM already wove them in (2026-07-17: was an unconditional "Analysis: …"
        # suffix — mechanical, and doubly-stated on SLM answers that already
        # discussed the patterns). Now a natural clause, and only as the last-resort
        # path when the summary SLM was disabled/absent. Top 2 only.
        if _pattern_details and not _slm_wove_patterns:
            from query.result_explainer import blend_patterns
            nl_answer_text = blend_patterns(nl_answer_text, _pattern_details)

        # Record the shared post-execution stages (execution / result_analysis /
        # summary / visualization) into the ONE query trace — reading only the _ictx
        # and summary this run already computed. Explainability is recorded in _done()
        # where the `explain` payload is built. Same helper Tier-2 uses → identical story.
        try:
            from veda.explain import record_result_stages
            try:
                from config import NL_SUMMARY_MODEL as _nl_model
            except Exception:
                _nl_model = None
            record_result_stages(
                engine=_summary_engine, cols=list(cols), row_count=len(rows),
                truncated=_truncated, ictx=_ictx, answer=nl_answer_text,
                summary_model=_nl_model, summary_ok=bool(_summary_engine),
                visualization=_insight_extra.get("visualization"))
            # PROJECTION funnel (Tier-1): the SQL SELECT columns actually produced,
            # against what retrieval surfaced — reveals where extra columns entered.
            tr.set("projection", sql_selected=list(cols)[:30],
                   sql_selected_count=len(cols), retrieved_count=len(results))
        except Exception:
            pass

    is_temporal = bool(tf and (tf.start or tf.end))
    # Don't cache fast-path results — they're already instant and the fast path always
    # wins ahead of the cache, so a cached copy would never be served.
    #
    # And never cache a CONTEXT-DEPENDENT turn. The verified cache is keyed on the query
    # TEXT, which used to carry the frame's description and so differed per entity. A
    # follow-up now reaches here as the user's bare words — "only the ones in Pune" — which
    # are the same three words whatever question they follow. Measured 2026-09-23: one such
    # entry, saved while following "properties where is gated is true", was then replayed
    # for sale listings, lease listings and a facing distribution, all of which came back
    # as the SAME 1000 rows of assets. The meaning of these words lives in the conversation
    # context, and the cache cannot see it, so it must not key on them.
    _context_dependent = bool(_conv.get("entity_table"))
    if (not from_cache and fp is None and rows and not is_temporal and not is_existence
            and _cache_back and not _context_dependent):
        save_verified_query(query, sql)
    elif _context_dependent and rows:
        print("  [cache] not saved — this turn's meaning comes from the conversation "
              "context, not from its own words")

    log_route(_route, query, (time.time() - start) * 1000, table=str(primary), rows=len(rows))
    tag = "cache" if from_cache else f"table={primary}"
    print("\n" + "=" * 78)
    print(f"✅ Done in {time.time()-start:.1f}s   |   intent={intent}   |   {tag}")
    print("=" * 78 + "\n")
    return _done(0, "answered", cols=list(cols) if cols else [], rows=rows,
                 answer=nl_answer_text, sql=param_sql, table=str(primary), **_insight_extra)