"""apps.chat.thinking_context — the words shown under each of the four steps.

TWO LAYERS, DELIBERATELY
    1. **Deterministic** (this module). Built from confirmed structured facts the
       engine already reported. Always available, never wrong, never late.
    2. **Narration** (optional SLM, see ``veda_core/veda/narrator.py``). Replaces the
       deterministic sentence with a more natural one *if it arrives in time*.

    The deterministic layer is the floor, not the fallback. If the SLM is off, slow,
    broken or disabled, the user still gets a real contextual sentence — so narration
    can never be load-bearing.

WHAT MAY APPEAR HERE
    Only facts the engine has CONFIRMED: the shape of the question (a count, a
    ranking, a trend), the time period the user themselves asked for, whether a
    breakdown was requested, how many sources took part, what is being produced
    (chart / table / summary).

WHAT MAY NEVER APPEAR
    Table or column names, SQL, source ids, engine names (`rag`, `tier2`,
    `deterministic_sql`), routing internals, schema-linking detail, scores,
    candidate lists, permission ids, policies, or anything a "restricted" answer
    was withholding. A sentence here is about the USER'S question, not about VEDA's
    implementation.

NOTHING IS GUESSED
    Every builder omits a clause whose fact is absent rather than filling it in.
    "Looking for the information that answers this" is the honest sentence when the
    only confirmed fact is "we are searching"; inventing a metric or a dimension to
    make it read better would be a fabrication the user cannot check.
"""
from __future__ import annotations

from . import thinking_steps as ts

# ── question shape -> the noun a user recognises ─────────────────────────────
# Keys are the engine's own confirmed grammar outcomes; values are plain English.
_INTENT_NOUN = {
    "count": "a count",
    "sum": "a total",
    "avg": "an average",
    "max": "the highest value",
    "min": "the lowest value",
    "ranking": "a ranking",
    "trend": "a trend over time",
    "list": "a list",
    "compare": "a comparison",
    "existence": "which records match",
}

_OUTPUT_NOUN = {
    "chart": "a chart",
    "table": "a table",
    "summary": "a summary",
    "chart+summary": "a chart with a short summary",
}


#: SQL-flavoured operation type -> plain phrasing for the progress UI.
#: build_explain's own `summary` strings read as SQL ("Group by Listing Status",
#: "Sort by Transaction Identifier", "Return top 100") — fine in the technical
#: explainability payload where they already ship, but the live progress UI must not
#: use that vocabulary. The BUSINESS field name is kept: it is the informative half,
#: and it is already the semantic model's own business label, never a raw column.
_OP_PHRASING = {
    "count":   "Counting the records",
    "total":   "Working out the total",
    "average": "Working out the average",
    "maximum": "Finding the highest value",
    "minimum": "Finding the lowest value",
    "group":   "Broken down by {field}",
    "sort":    "Ordered by {field}",
    "limit":   "Limited to the top results",
    "list":    "Listing the records",
}

_FIELD_FROM_SUMMARY = __import__("re").compile(
    r"^(?:Group by|Sort by|Calculate \w+)\s+(.+?)(?:\s*\(.*\))?$", __import__("re").I)

#: The document head's retrieval operation, whichever count it carries. Matched on
#: its shape rather than compared to a fixed string because the number is part of
#: the sentence ("Retrieved 5 relevant passages"). See _analyzing_details.
_RETRIEVED_PASSAGES = __import__("re").compile(
    r"^Retrieved\s+\d+\s+relevant\s+passages?$", __import__("re").I)


def _plain_operation(op: dict) -> str:
    """One operation, in plain language rather than SQL vocabulary."""
    kind = str(op.get("type") or "").lower()
    summary = str(op.get("summary") or "")
    template = _OP_PHRASING.get(kind)
    if not template:
        # An unmapped operation type: fall back to the payload's own summary rather
        # than dropping the fact, but strip the SQL verb prefixes we know about.
        return _FIELD_FROM_SUMMARY.sub(r"\1", summary)[:64]
    if "{field}" in template:
        m = _FIELD_FROM_SUMMARY.match(summary)
        field = (m.group(1).strip() if m else "").strip()
        return template.format(field=field) if field else template.replace(
            " by {field}", "")
    return template


def _humanize_period(raw: str) -> str:
    """A period a person can read.

    The engine reports the resolved window as full ISO timestamps
    ("2024-01-01T00:00:00 to 2024-12-31T23:59:59"), which is precise and unreadable.
    Collapses the common shapes — a whole year, a whole month, a plain day range —
    and otherwise just drops the time-of-day. Returns the input unchanged if it does
    not parse, rather than guessing at a prettier form.
    """
    try:
        parts = [p.strip() for p in str(raw).split(" to ")]
        if len(parts) != 2:
            return str(raw)[:64]
        a, b = (p.split("T")[0] for p in parts)
        ay, am, ad = a.split("-")
        by, bm, bd = b.split("-")
        # A WHOLE year: 1 Jan -> 31 Dec of the same year.
        if ay == by and (am, ad) == ("01", "01") and (bm, bd) == ("12", "31"):
            return ay
        # A WHOLE month: 1st -> the month's last day. The end-day check matters —
        # without it "1 Apr to 15 Apr" would be labelled "April 2026", which is a
        # different period from the one the user asked for.
        if ay == by and am == bm and ad == "01" and int(bd) >= 28:
            month = ["", "January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November",
                     "December"][int(am)]
            return f"{month} {ay}"
        if a == b:
            return a
        return f"{a} to {b}"
    except Exception:
        return str(raw)[:64]


class ThinkingContext:
    """Accumulates the confirmed, user-safe facts of one turn.

    Populated from the ``details`` the engine attaches to its own progress events —
    this class never inspects the pipeline, never reads the trace, and holds nothing
    that is not already cleared for display.
    """

    __slots__ = ("intent", "period", "grouped", "output", "source_count",
                 "multi_source", "operations", "row_count", "truncated",
                 "access_state", "found_something", "warnings",
                 "no_answer", "found_nothing", "from_cache", "failed_sources",
                 "warning_messages", "checks", "filters", "datasets",
                 "contributions", "chart_reason", "guidance", "combined_sources",
                 "match_summary", "phase_runs",
                 "source_names", "passages", "execution_type")

    def __init__(self):
        self.intent: str | None = None
        self.period: str | None = None
        self.grouped: bool = False
        self.output: str | None = None
        self.source_count: int | None = None
        self.multi_source: bool = False
        self.operations: list = []
        self.row_count: int | None = None
        self.truncated: bool = False
        self.access_state: str | None = None
        self.found_something: bool | None = None
        self.warnings: list = []
        #: True once the turn is known to have produced no answer (a refusal or a
        #: clarify). Set at the terminal frame, because until then the turn is still
        #: expected to succeed and guessing otherwise would be worse than generic
        #: copy. Only ever narrows what is claimed — never adds a claim.
        self.no_answer: bool = False
        #: The turn RAN and found nothing — distinct from a refusal, which declined
        #: to run and can say what would help instead. "See the reply for what's
        #: needed" is wrong copy for an empty result: there is nothing to supply.
        self.found_nothing: bool = False
        #: {display name: user-safe reason} for sources that did NOT deliver.
        self.failed_sources: dict = {}
        #: Authored, user-safe warning SENTENCES. `warnings` holds the codes, which
        #: were collected and then never rendered anywhere — the clearest case of
        #: "absorbed but never displayed" in this file.
        self.warning_messages: list = []
        #: The safety checks BY NAME. The payload carries five hand-written labels
        #: ("Read-only query", "Duplicate-safe (no double-counting)", …) and the
        #: panel collapsed all of them into one line saying a check happened.
        self.checks: list = []
        #: Filters the query actually applied, as plain phrases.
        self.filters: list = []
        #: Document/dataset names the answer drew on — "msa green tower" tells the
        #: reader more than the connector name "docs_contracts" ever could.
        self.datasets: list = []
        #: {source display name: rows it contributed} on a cross-source answer.
        self.contributions: dict = {}
        #: Why THIS chart was chosen, in the deterministic authored wording.
        self.chart_reason: str | None = None
        #: What would help, on a turn that could not answer.
        self.guidance: list = []
        #: Display names the FEDERATION actually combined. Distinct from
        #: `source_names`: that is who took part, this is proof they were joined.
        self.combined_sources: list = []
        #: The authored sentence about how well the records matched across sources.
        self.match_summary: str | None = None
        #: {phase: [title, start_ms, end_ms]} — the WORK THAT ACTUALLY RAN, by the
        #: authored title the engine already ships on every event, with the elapsed
        #: window it occupied. The panel was collapsing twelve real phases into four
        #: steps and showing none of them, so the longest stretch of a turn
        #: ("Checking available data", measured at 13.2s of a 26s turn) had nothing
        #: on screen at all. The phase KEY stays internal; only the title is shown.
        self.phase_runs: dict = {}
        #: True when the answer replayed SQL from the verified-query cache.
        self.from_cache: bool = False
        #: Display names of the sources that took part. A name beats a count:
        #: "Samta Employee Handbook" tells the reader something "1" cannot.
        self.source_names: list = []
        #: Retrieved passage count, for document answers. None means the backend
        #: did not report one — which is not the same as zero.
        self.passages: int | None = None
        #: sql | documents | multi_source, from the route the engine took.
        self.execution_type: str = ts.EXEC_UNKNOWN

    # -- ingest ---------------------------------------------------------------
    def absorb(self, payload: dict) -> None:
        """Take whatever confirmed facts this progress event carries. Never raises."""
        try:
            _p = payload or {}
            _chunks = _p.get("chunks")
            if _chunks is None:
                _chunks = _p.get("doc_chunks")     # hybrid_retrieve's name
            if isinstance(_chunks, int) and _chunks >= 0:
                self.passages = _chunks
            _intent = (payload or {}).get("intent")
            if _intent:
                _shape = ts._EXEC_FROM_INTENT.get(str(_intent).strip().lower())
                if _shape and (self.execution_type == ts.EXEC_UNKNOWN
                               or _shape == ts.EXEC_MULTI_SOURCE):
                    self.execution_type = _shape
        except Exception:
            pass
        try:
            d = (payload or {}).get("details") or {}
            phase = (payload or {}).get("phase")
            status = (payload or {}).get("status")
            _title = (payload or {}).get("title")
            _el = (payload or {}).get("elapsed_ms")
            if phase and _title and isinstance(_el, (int, float)):
                # [title, start_ms, end_ms, worst_status, message]. `worst_status` is
                # a RATCHET, the same pattern already used for sub-check severity: a
                # phase can report `warning` and then `completed` for the SAME piece
                # of work (measured live — `result_preparation` did exactly this,
                # warning then completed, both for one truncated result), and a later
                # `completed` must never soften an already-reported problem. Without
                # this the row rendered as a plain green tick while its OWN parent
                # step was `warning`, with nothing on screen explaining why.
                _run = self.phase_runs.setdefault(phase, [str(_title), None, None,
                                                          None, None])
                if status == "started" and _run[1] is None:
                    _run[1] = float(_el)
                elif status in ("completed", "warning", "failed"):
                    _run[2] = float(_el)
                    if _run[1] is None:
                        _run[1] = float(_el)
                    _sev = {"completed": 1, "warning": 2, "failed": 3}.get(status, 0)
                    _was = {"completed": 1, "warning": 2, "failed": 3}.get(_run[3], 0)
                    if _sev >= _was:
                        _run[3] = status
                        _msg = (payload or {}).get("message")
                        # Only keep a message that says something the row's own
                        # LABEL does not already say. The engine's `source_selection`
                        # warning ships `message: "Checking available data"` — the
                        # phase's own title, verbatim, so surfacing it would just
                        # repeat the label as if it were an explanation.
                        if _msg and str(_msg).strip().lower() != str(_title).strip().lower():
                            _run[4] = str(_msg).strip()

            if d.get("intent"):
                self.intent = str(d["intent"])[:32]
            if d.get("period"):
                self.period = _humanize_period(str(d["period"]))
            if d.get("grouped") is not None:
                self.grouped = bool(d["grouped"])
            if d.get("output"):
                self.output = str(d["output"])[:32]
            if isinstance(d.get("source_count"), int):
                self.source_count = d["source_count"]
                self.multi_source = d["source_count"] > 1
            if isinstance(d.get("row_count"), int):
                self.row_count = d["row_count"]
            if d.get("truncated") is not None:
                self.truncated = bool(d["truncated"])
            if isinstance(d.get("operations"), list):
                self.operations = [str(o)[:64] for o in d["operations"][:6]]

            if phase == "access_check" and status:
                self.access_state = status
            if phase in ("source_selection", "data_retrieval") and status:
                if status == "completed":
                    self.found_something = True
                elif status == "failed":
                    self.found_something = False
            if d.get("code"):
                code = str(d["code"])[:48]
                if code not in self.warnings:
                    self.warnings.append(code)
        except Exception:
            pass

    def absorb_explain(self, explain: dict) -> None:
        """Late facts from the final explainability payload — used for the Preparing
        step's detail, and to fill anything the live stream did not carry."""
        try:
            ex = explain if isinstance(explain, dict) else {}
            res = ex.get("result") or {}
            # Provenance: a replayed answer. Read from BOTH the dedicated block and
            # the result flag so a client of either shape is covered.
            if (ex.get("provenance") or {}).get("reused_verified_query") \
                    or res.get("reused_verified_query"):
                self.from_cache = True
            for src in (ex.get("sources") or []):
                name = src.get("name") if isinstance(src, dict) else None
                if name and name not in self.source_names:
                    self.source_names.append(name)
            # A source that FAILED must not be rendered as a completed tick. The
            # per-source execution records carry the outcome and were being ignored
            # here — measured live on a datalake turn whose payload said
            # `status: "failed", message: "This data source could not be reached"`
            # while the Finding step showed `catalog_parquet ✓` and "Relevant
            # information available ✓". The panel contradicted the payload it was
            # built from.
            for rec in ((ex.get("execution") or {}).get("sources") or []):
                if not isinstance(rec, dict):
                    continue
                nm = rec.get("name")
                if nm and rec.get("status") in ("failed", "refused", "skipped"):
                    self.failed_sources[nm] = (rec.get("message")
                                               or "This data source did not respond.")
            if isinstance(res.get("row_count"), int):
                self.row_count = res["row_count"]
            if res.get("truncated") is not None:
                self.truncated = bool(res["truncated"])
            ops = ex.get("operations") or []
            if ops:
                if not self.operations:
                    self.operations = [_plain_operation(o) for o in ops
                                       if isinstance(o, dict)][:6]
                    self.operations = [o for o in self.operations if o]
                # Recover the question SHAPE from the operations that actually ran.
                # The live stream cannot always supply it: "how many …" is detected by
                # the deterministic fast path, not by the aggregation grammar, so
                # `aggregation.op` is None and the intent was simply absent mid-flight
                # (observed live). The final frame can be specific because the
                # executed operations are now known.
                if not self.intent:
                    kinds = {str(o.get("type")) for o in ops if isinstance(o, dict)}
                    for kind, intent in (("count", "count"), ("total", "sum"),
                                         ("average", "avg"), ("maximum", "max"),
                                         ("minimum", "min")):
                        if kind in kinds:
                            self.intent = intent
                            break
                    else:
                        if "sort" in kinds and "limit" in kinds:
                            self.intent = "ranking"
                        elif isinstance(self.row_count, int) and self.row_count > 1:
                            # A plain retrieval. `_INTENT_NOUN` has carried "a list"
                            # all along and nothing ever set it, so "list 5 assets" —
                            # the commonest shape there is — fell through to the
                            # generic "Working out what you're asking for." on the
                            # step that takes the longest.
                            #
                            # Keyed on the ROWS, not on an operation type: measured,
                            # a list query emits only `limit` ("Return top 100") and
                            # no `select` at all, so matching operation names found
                            # nothing. More than one record coming back is what makes
                            # an answer a list, and it is read above this block.
                            self.intent = "list"
                if "group" in {str(o.get("type")) for o in ops if isinstance(o, dict)}:
                    self.grouped = True
            if ex.get("visualization"):
                self.output = "chart+summary" if self.output == "summary" else "chart"
            for w in (ex.get("warnings") or []):
                if isinstance(w, dict) and w.get("code") and w["code"] not in self.warnings:
                    self.warnings.append(w["code"])
                    _m = w.get("message")
                    if _m and _m not in self.warning_messages:
                        self.warning_messages.append(_m)
            # The five safety checks BY NAME. `_CHECK_LABELS` in business_explain is
            # already user-facing authored copy, and the flow block already tells the
            # reader "5 checks passed" — so the number is public and the names are
            # safe. Only the collapsed one-liner was reaching the panel.
            for c in ((ex.get("validation") or {}).get("checks") or []):
                if isinstance(c, dict) and c.get("label"):
                    self.checks.append((c["label"], bool(c.get("passed"))))
            for f in ((ex.get("filters") or {}).get("applied") or []):
                if not isinstance(f, dict):
                    continue
                _t = " ".join(str(f.get(k)) for k in ("field", "operator", "value")
                              if f.get(k) not in (None, ""))
                if _t and _t not in self.filters:
                    self.filters.append(_t)
            for d in ((ex.get("data_used") or {}).get("datasets") or []):
                if d and d not in self.datasets:
                    self.datasets.append(str(d))
            for src in (ex.get("sources") or []):
                if isinstance(src, dict) and isinstance(src.get("rows"), int) \
                        and src.get("name"):
                    self.contributions[src["name"]] = src["rows"]
            # The federation block. It has shipped on every cross-source answer and
            # nothing read it — which is why "Analyzing", the step where the
            # federation actually happened, was the ONLY empty step on exactly the
            # turns that did the most work.
            _cs = ex.get("cross_source") or {}
            if _cs.get("used"):
                self.combined_sources = [n for n in (_cs.get("sources") or []) if n]
                _j = _cs.get("join") or {}
                if _j.get("used") and _j.get("summary"):
                    self.match_summary = str(_j["summary"])
            _viz = ex.get("visualization")
            if isinstance(_viz, dict) and _viz.get("reason"):
                self.chart_reason = str(_viz["reason"])
            for g in (ex.get("suggestions") or []):
                if g and g not in self.guidance:
                    self.guidance.append(str(g))
            _wwh = ex.get("what_would_help")
            if _wwh and _wwh not in self.guidance:
                self.guidance.insert(0, str(_wwh))
        except Exception:
            pass

    # -- sentences ------------------------------------------------------------
    def sentence(self, step_key: str) -> str:
        """The collapsed one-liner for a step. Only confirmed clauses appear."""
        try:
            if step_key == ts.STEP_UNDERSTANDING:
                return self._understanding_sentence()
            if step_key == ts.STEP_FINDING:
                return self._finding_sentence()
            if step_key == ts.STEP_ANALYZING:
                return self._analyzing_sentence()
            if step_key == ts.STEP_PREPARING:
                return self._preparing_sentence()
        except Exception:
            pass
        return ts._GENERIC_CONTEXT.get(step_key, "")

    def _understanding_sentence(self) -> str:
        noun = _INTENT_NOUN.get(self.intent or "")
        if noun and self.period:
            return f"You're asking for {noun}, for {self.period}."
        if noun:
            return f"You're asking for {noun}."
        if self.period:
            return f"Reading your question, for {self.period}."
        return "Working out what you're asking for."

    def _finding_sentence(self) -> str:
        if self.multi_source and self.source_count:
            return (f"Gathering the relevant information across "
                    f"{self.source_count} sources.")
        if self.period:
            return f"Looking for the information covering {self.period}."
        return "Looking for the information that answers this."

    def _analyzing_sentence(self) -> str:
        bits = []
        if self.grouped:
            bits.append("breaking the results down")
        if self.intent in ("ranking", "max", "min"):
            bits.append("finding the top value")
        elif self.intent in ("count", "sum", "avg"):
            bits.append("working out the figure")
        elif self.intent == "trend":
            bits.append("looking at how it changes over time")
        if self.multi_source:
            bits.append("combining what each source returned")
        if self.output in ("chart", "chart+summary"):
            bits.append("preparing the data to be charted")
        if not bits:
            return "Working through the information found."
        joined = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + f" and {bits[-1]}"
        return joined[0].upper() + joined[1:] + "."

    def _preparing_sentence(self) -> str:
        # A turn that produced NO answer must not say it is assembling one. The
        # phase message is already honest ("Could not answer this from the
        # available data") but this sentence OVERWRITES it at the terminal frame,
        # so the honest text never reached the user — measured live on a clarify.
        if self.found_nothing:
            return "No matching data was found for this question."
        if self.no_answer:
            return "Couldn't answer this one — see the reply for what's needed."
        noun = _OUTPUT_NOUN.get(self.output or "")
        if noun:
            return f"Putting together {noun}."
        return "Putting your answer together."

    # -- expanded details -----------------------------------------------------
    def details(self, step_key: str) -> list:
        """Ordered evidence rows for the expanded view: `{type, label, state}`.

        A LIST, because the reader is shown a sequence of things that happened and
        the order carries meaning — access before retrieval, retrieval before
        analysis. Rows appear ONLY for facts the backend actually reported; an
        absent fact produces no row rather than a hedged one.
        """
        try:
            if step_key == ts.STEP_UNDERSTANDING:
                return self._understanding_details()
            if step_key == ts.STEP_FINDING:
                return self._finding_details()
            if step_key == ts.STEP_ANALYZING:
                return self._analyzing_details()
            if step_key == ts.STEP_PREPARING:
                return self._preparing_details()
        except Exception:
            pass
        return []

    @staticmethod
    def _row(kind: str, label: str, state: str = ts.STATE_COMPLETED,
             *, generic: bool = False) -> dict:
        """One evidence row. `generic=True` marks a placeholder that a more specific
        row of the same type should replace — "1 relevant source found" is worth
        showing while the name is unknown, and worth removing once it is."""
        row = {"type": kind, "label": label, "state": state}
        if generic:
            row["_generic"] = True
        return row

    #: Phases whose title says nothing a reader can use, or that the panel already
    #: represents better elsewhere. `access_check` is a timed sub-check with its own
    #: outcome copy; `received`/`completed` are turn boundaries, not work.
    _PHASE_ROW_SKIP = frozenset({"received", "completed", "access_check",
                                 "understanding"})

    def _phase_rows(self, step_key: str) -> list:
        """The work that actually ran inside this step, by its authored title.

        The engine ships `title` and `elapsed_ms` on every progress event and the
        panel was discarding both, folding twelve real phases into four steps. The
        result was that the longest stretch of a turn had nothing on screen: on a
        measured 26-second turn, "Checking available data" occupied 13.2s of it and
        the Finding step showed only the source name.

        A duration is attached only when the phase reported BOTH a start and an end,
        because a single timestamp is a moment, not a measurement.
        """
        rows = []
        for phase, (title, t0, t1, status, message) in self.phase_runs.items():
            if phase in self._PHASE_ROW_SKIP:
                continue
            # The engine's titles are authored per PHASE, not per route, so the
            # document head's `data_retrieval` arrives titled "Running the query" —
            # and no query ran; passages were retrieved. "Synthesized the retrieved
            # information" already says what happened on that path.
            if phase == "data_retrieval" and self.execution_type == ts.EXEC_DOCUMENTS:
                continue
            if ts.PHASE_TO_STEP.get(phase) != step_key:
                continue
            # THE ROW'S STATE FOLLOWS THE PHASE'S OWN REPORTED OUTCOME. This used to
            # be hardcoded to `completed` regardless of what the engine actually
            # said, which is how a step read `!` (warning) while every row inside it
            # showed a plain green tick — measured live on a federated fallback: the
            # reader had no way to tell WHICH part of "Analyzing" was the problem.
            _state = {"warning": ts.STATE_WARNING,
                      "failed": ts.STATE_FAILED}.get(status, ts.STATE_COMPLETED)
            # A row that just ECHOES the step's own title adds nothing while it is
            # a plain success — `result_preparation`'s authored title IS, verbatim,
            # "Preparing your answer", the exact string `Preparing` already shows as
            # its own header. It is kept when the state is NOT a plain success,
            # because then the colour on the row is itself the fact: it is how a
            # reader sees WHICH part of a `!` step is the part with a problem, even
            # before any message is attached.
            if title == ts.STEP_TITLES.get(step_key) and _state == ts.STATE_COMPLETED:
                continue
            row = self._row(ts.DETAIL_OPERATION, title, _state)
            if t0 is not None and t1 is not None and t1 >= t0:
                row["duration_ms"] = int(t1 - t0)
            # The message that came WITH a WARNING/FAILURE, and only then. A plain
            # success carries a message too — "Safety checks passed", "homzhub
            # completed" — and showing it turned a clean run noisy: "Checking the
            # query" gained "— Safety checks passed" sitting directly above a row
            # that already says "5 safety checks passed", the same fact twice on a
            # turn with nothing wrong to explain. On a real problem the message is
            # the one thing worth reading; on a success the checkmark already says
            # everything the reader needs. Still deduped against a warning already
            # surfaced elsewhere (Preparing's own warning_messages loop).
            if _state != ts.STATE_COMPLETED and message and message not in self.warning_messages:
                row["message"] = message
            rows.append(row)
        return rows

    def _understanding_details(self) -> list:
        rows = []
        if self.intent and _INTENT_NOUN.get(self.intent):
            rows.append(self._row(ts.DETAIL_OPERATION,
                                  f"Looking for {_INTENT_NOUN[self.intent]}"))
        if self.period:
            rows.append(self._row(ts.DETAIL_OPERATION, f"Period: {self.period}"))
        if self.grouped:
            rows.append(self._row(ts.DETAIL_OPERATION, "Broken down by category"))
        # What the question NARROWED to. The payload has carried this all along and
        # nothing displayed it, so a reader could not tell whether the filter they
        # asked for had actually been applied — the single most common way a wrong
        # answer looks right.
        for f in self.filters[:4]:
            rows.append(self._row(ts.DETAIL_OPERATION, f"Filtered to {f}"))
        return rows

    def _finding_details(self) -> list:
        # The search runs BEFORE it finds anything, so its row leads.
        rows = self._phase_rows(ts.STEP_FINDING)
        # Named sources beat a bare count: "Samta Employee Handbook" tells the reader
        # something a "1" cannot. The count is the fallback when no name is known.
        if self.source_names:
            for name in self.source_names[:6]:
                _why = self.failed_sources.get(name)
                rows.append(self._row(
                    ts.DETAIL_SOURCE,
                    f"{name} — {_why}" if _why else name,
                    ts.STATE_WARNING if _why else ts.STATE_COMPLETED))
        # The DOCUMENTS, by name. "docs_contracts" is the connector; "msa green
        # tower" is what the reader recognises and can go and check.
        #
        # DOCUMENT ANSWERS ONLY. On a SQL turn the same `data_used.datasets` key
        # holds the humanized TABLE name ("Assets", "Maintenances"), which is engine
        # vocabulary and exactly what this layer keeps out of the panel — it showed
        # up as a source row beside `homzhub` the moment this was added.
        if self.execution_type == ts.EXEC_DOCUMENTS:
            for d in self.datasets[:4]:
                rows.append(self._row(ts.DETAIL_SOURCE, d))
        # "1 relevant source found" IS A PLACEHOLDER FOR A NAME, so it is emitted
        # only when this build produced no named source row at all. It used to be
        # the `elif` of `source_names`, which is a narrower test than it looks: a
        # DOCUMENT answer names its sources through `datasets` (rendered just above,
        # added after this fallback was written) while `source_names` stays empty,
        # so that build emitted "1 relevant source found" and "msa green tower"
        # side by side — the same fact, once vaguely. Across the live stream the row
        # is still right and still shown, because until the terminal payload arrives
        # no name is known; the supersede rule in
        # `ThinkingStepTracker.set_details` then drops it when the name comes.
        #
        # `evidence.sources` in the same payload states the count a second time even
        # at the terminal frame, but that key is a documented part of the frontend
        # contract that other clients read, so the ROW is the half that goes.
        if not rows and self.source_count:
            plural = "s" if self.source_count > 1 else ""
            rows.append(self._row(
                ts.DETAIL_SOURCE,
                f"{self.source_count} relevant source{plural} found", generic=True))
        if isinstance(self.passages, int) and self.passages > 0:
            plural = "s" if self.passages > 1 else ""
            rows.append(self._row(ts.DETAIL_EVIDENCE,
                                  f"{self.passages} relevant passage{plural} retrieved"))
        # "Relevant information available" is ENTAILED by every named row above it.
        # If we can say the answer came from `homzhub`, or that 5 passages came
        # back, then "relevant information available" adds nothing the reader did
        # not just read — and it was measured in the FINAL frame of 12 of 14
        # database turns, sitting directly beneath the named source row `homzhub`.
        # It survived the deduplication that should have caught it because the
        # generic-supersede rule in `ThinkingStepTracker.set_details` only replaces
        # a generic row of the SAME type, and this row is `evidence` while the
        # named source is `source`.
        #
        # The condition is therefore what this build ACTUALLY produced, not the two
        # attributes the old guard happened to name (`source_names` / `passages`):
        # a document answer names its sources through `datasets`, which neither
        # attribute covers, so that path still emitted the row beside a named
        # document. The one case it was written for survives untouched — the turn
        # found something and can name nothing, where a bare generic count row is
        # not a name.
        _named_something = any(
            not r.get("_generic") and r["type"] in (ts.DETAIL_SOURCE,
                                                    ts.DETAIL_EVIDENCE)
            for r in rows)
        if self.failed_sources and len(self.failed_sources) >= len(self.source_names or [1]):
            pass                       # every source that ran failed — say nothing more
        elif self.found_something is True and not _named_something:
            rows.append(self._row(ts.DETAIL_EVIDENCE, "Relevant information available",
                                  generic=True))
        elif self.found_something is False:
            rows.append(self._row(ts.DETAIL_EVIDENCE, "No relevant information found",
                                  ts.STATE_WARNING))
        if self.period and self.found_something is True:
            rows.append(self._row(ts.DETAIL_EVIDENCE, "Requested period covered"))
        if self.from_cache:
            # Under "Finding", because reusing a verified query is precisely what
            # replaced the work of finding and building one for this question.
            rows.append(self._row(
                ts.DETAIL_EVIDENCE,
                "Reused a query already verified for a very similar question."))
        return rows

    def _analyzing_details(self) -> list:
        """The semantic operations, in the order they were applied.

        For a database answer these come from the operations the engine reported
        (filter / group / sort / limit). For a document answer the engine reports no
        operations at all, so the two things that genuinely happened are named
        instead — reading the passages and synthesising them. Nothing is listed for
        an execution shape that reported neither.
        """
        # "Limited to the top results" on a COUNT is not merely redundant, it is
        # WRONG: the deterministic head appends `LIMIT 100` to every statement, so a
        # `SELECT COUNT(...)` that can only ever return one row was telling the
        # reader their answer had been cut short (measured on 4 of 14 turns). A limit
        # is only a fact worth stating when the result was actually truncated.
        _ops = [op for op in self.operations
                if op != _OP_PHRASING["limit"] or self.truncated]
        # THE DOCUMENT TRIPLE. `_apply_document_v1` (veda_core/veda_hybrid.py) emits
        # exactly three operations for every document answer —
        #   "Retrieved 5 relevant passages" / "Read the relevant passages" /
        #   "Synthesized the retrieved information"
        # — and only the last of them describes work that is not already stated
        # elsewhere in the panel:
        #
        #  * The retrieval row is the Finding step's "5 relevant passages retrieved"
        #    with the same number and the wording reversed, so the reader is told the
        #    same event twice, one step apart. It is dropped only when that Finding
        #    row was ACTUALLY built in this turn (`passages` reported and non-zero),
        #    because on a turn where the count never reached us the retrieval row is
        #    the only place the passages are mentioned at all.
        #  * "Read the relevant passages" and "Synthesized the retrieved
        #    information" are not separable events in this pipeline — nothing
        #    retrieves without reading, and the synthesis row is the one that names
        #    work the panel states nowhere else. So reading is folded into it, and
        #    only when the synthesis row is genuinely present in the same list.
        _passages_reported = isinstance(self.passages, int) and self.passages > 0
        _synthesis_present = any(op.startswith("Synthesized") for op in _ops)
        _ops = [op for op in _ops
                if not (_passages_reported and _RETRIEVED_PASSAGES.match(op))
                and not (_synthesis_present and op == "Read the relevant passages")]
        rows = [self._row(ts.DETAIL_OPERATION, op) for op in _ops]
        if not rows and self.execution_type == ts.EXEC_DOCUMENTS and self.passages:
            # PLACEHOLDERS, marked generic. The real operations only arrive with the
            # final explainability payload, and until then these two are all we can
            # honestly say. Without the generic mark both sets survived and the step
            # listed the same work twice in two tenses — measured live on a contract
            # question: "Reading relevant passages" AND "Read the relevant passages",
            # "Synthesizing the retrieved information" AND "Synthesized the retrieved
            # information". A specific row supersedes a generic one of the same type.
            rows = [self._row(ts.DETAIL_OPERATION, "Reading relevant passages",
                              generic=True),
                    self._row(ts.DETAIL_OPERATION,
                              "Synthesizing the retrieved information",
                              generic=True)]
        # WHAT EACH SOURCE CONTRIBUTED. On a cross-source answer this step was
        # completely empty — the one step where the federation actually happened.
        for name, n in list(self.contributions.items())[:4]:
            if len(self.contributions) > 1:
                rows.append(self._row(
                    ts.DETAIL_OPERATION,
                    f"{name} contributed {n} record{'' if n == 1 else 's'}"))
        if len(self.combined_sources) > 1:
            # Name them. "Combined across sources" is true of every federation and
            # tells the reader nothing they cannot already see in the source rows.
            rows.append(self._row(
                ts.DETAIL_OPERATION,
                "Combined " + " and ".join(self.combined_sources[:4])))
            if self.match_summary:
                rows.append(self._row(ts.DETAIL_OPERATION, self.match_summary))
        elif len(self.contributions) > 1:
            rows.append(self._row(ts.DETAIL_OPERATION, "Combined across sources"))
        elif self.multi_source and self.source_count and self.source_count > 1:
            rows.append(self._row(ts.DETAIL_OPERATION,
                                  "Comparing information across sources"))
            rows.append(self._row(ts.DETAIL_OPERATION, "Reconciling results"))
        # The checks BY NAME, replacing the single line that said only that
        # checking had occurred. A failed check is named too — that is the case
        # where knowing WHICH one matters most.
        # THE CHECKS. Naming all five on every turn was measured to be 5 of the 6
        # rows in this step, identical on every SQL answer — which trains the reader
        # to skip the step that also carries the one row that varies. They collapse
        # to a single line while they all pass, and the count is the fact that
        # matters there ("we ran five, all passed").
        #
        # The moment one does NOT pass, that is no longer noise: the failing checks
        # are named individually, because WHICH one failed is the whole point. The
        # full list stays in `explainability.validation.checks` either way, so
        # nothing is lost to a reader who wants it.
        rows.extend(self._phase_rows(ts.STEP_ANALYZING))
        _failed = [l for l, ok in self.checks if not ok]
        if _failed:
            for label in _failed[:6]:
                rows.append(self._row(ts.DETAIL_VALIDATION, label, ts.STATE_WARNING))
        elif self.checks:
            _n = len(self.checks)
            rows.append(self._row(
                ts.DETAIL_VALIDATION,
                f"{_n} safety check{'' if _n == 1 else 's'} passed"))
        return rows

    def _preparing_details(self) -> list:
        # Nothing was produced, so list nothing. "Supporting summary" used to be
        # appended unconditionally, which on a clarify claimed an output that does
        # not exist.
        if self.found_nothing or self.no_answer:
            rows = [self._row(
                ts.DETAIL_OUTPUT,
                "Nothing matched — there is no result to show." if self.found_nothing
                else "No answer could be produced for this question.",
                ts.STATE_WARNING)]
            # WHAT WOULD HELP. The payload carries the guidance the refusal was
            # written to give ("tell me what 'atlanti' refers to — a column, or a
            # value to filter on?") and the panel showed only the bare statement
            # that nothing was produced, which tells the reader nothing they can act
            # on. A turn that found NOTHING has no guidance to give, so this is the
            # refusal case only.
            if not self.found_nothing:
                for g in self.guidance[:3]:
                    rows.append(self._row(ts.DETAIL_OUTPUT, g, ts.STATE_WARNING))
            return rows
        # What actually ran in THIS step, for the same reason Finding and Analyzing
        # already do this: a step reading `!` with nothing inside explaining why is
        # a legibility bug, not a subtlety — the reader cannot tell WHICH part of
        # "Preparing" had a problem. `_phase_rows` already dedupes its message
        # against `self.warning_messages`, so this never restates a caveat already
        # shown by the loop further down.
        rows = self._phase_rows(ts.STEP_PREPARING)
        if self.output in ("chart", "chart+summary"):
            # WHY THIS CHART. `_CHART_REASON_TEMPLATES` is deterministic authored
            # copy, never the model's prose — and it was being read as a boolean and
            # thrown away.
            rows.append(self._row(ts.DETAIL_OUTPUT,
                                  self.chart_reason or "Preparing visualization"))
        if isinstance(self.row_count, int) and self.row_count > 1:
            plural = "s" if self.row_count != 1 else ""
            rows.append(self._row(ts.DETAIL_OUTPUT,
                                  f"Preparing table · {self.row_count} row{plural}"))
        # "Preparing summary" USED TO BE APPENDED HERE, unconditionally. It was
        # emitted on 14 of 14 measured turns because it is true of every turn that
        # produces any answer at all, so it distinguishes nothing and carries no
        # per-turn fact — and the step's own collapsed line already says it, in the
        # same words: `_preparing_sentence` returns "Putting your answer together."
        # (or "Putting together a summary." when the output is known). The reader
        # was being shown one fact twice, once as the heading of the thing it
        # duplicates.
        #
        # CHECKED BEFORE REMOVING, because `ThinkingStepTracker.finish()` decides
        # `completed` vs `skipped` for a step that never started by whether it
        # carries details, and this was the only unconditional row here. The
        # decision is unreachable for THIS step: both that branch and the matching
        # one in `snapshot(reached_only=True)` are gated on a LATER step having
        # started, and "preparing" is the last entry in STEP_ORDER, so `later_ran`
        # is always False for it. A never-started "preparing" is withheld from the
        # snapshot entirely and never reclassified. The only visible consequence is
        # `expandable: False` on a plain-summary turn, which is correct — there is
        # now genuinely nothing inside the step that the collapsed line does not say.
        if self.truncated:
            rows.append(self._row(ts.DETAIL_OUTPUT,
                                  "Showing the first page of results only",
                                  ts.STATE_WARNING))
        # The authored warning SENTENCES. Their codes were collected and never
        # rendered — so a caveat the engine had already written for the reader
        # ("There was limited matching data for this question, so the answer may be
        # incomplete.") reached the payload and stopped there. Truncation is skipped
        # because the row above already says it, in the same step.
        for m in self.warning_messages[:3]:
            if "limited to the first" in m.lower():
                continue
            rows.append(self._row(ts.DETAIL_OUTPUT, m, ts.STATE_WARNING))
        return rows

    # -- what the narrator is allowed to see ---------------------------------
    def narrator_facts(self) -> dict:
        """The ONLY things an SLM narrator may be told. Everything here is already
        cleared for display, so a narration built strictly from it cannot leak —
        and anything the narrator says that is NOT derivable from these keys is
        rejected downstream as an invention."""
        facts = {}
        if self.intent:
            facts["intent"] = self.intent
        if self.period:
            facts["time_period"] = self.period
        if self.grouped:
            facts["breakdown"] = True
        if self.output:
            facts["requested_output"] = self.output
        if self.source_count:
            facts["source_count"] = self.source_count
        if self.operations:
            facts["completed_operations"] = self.operations[:4]
        return facts
