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
                 "no_answer", "from_cache",
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
            _chunks = (payload or {}).get("chunks")
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
                if "group" in {str(o.get("type")) for o in ops if isinstance(o, dict)}:
                    self.grouped = True
            if ex.get("visualization"):
                self.output = "chart+summary" if self.output == "summary" else "chart"
            for w in (ex.get("warnings") or []):
                if isinstance(w, dict) and w.get("code") and w["code"] not in self.warnings:
                    self.warnings.append(w["code"])
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

    def _understanding_details(self) -> list:
        rows = []
        if self.intent and _INTENT_NOUN.get(self.intent):
            rows.append(self._row(ts.DETAIL_OPERATION,
                                  f"Looking for {_INTENT_NOUN[self.intent]}"))
        if self.period:
            rows.append(self._row(ts.DETAIL_OPERATION, f"Period: {self.period}"))
        if self.grouped:
            rows.append(self._row(ts.DETAIL_OPERATION, "Broken down by category"))
        return rows

    def _finding_details(self) -> list:
        rows = []
        # Named sources beat a bare count: "Samta Employee Handbook" tells the reader
        # something a "1" cannot. The count is the fallback when no name is known.
        if self.source_names:
            for name in self.source_names[:6]:
                rows.append(self._row(ts.DETAIL_SOURCE, name))
        elif self.source_count:
            plural = "s" if self.source_count > 1 else ""
            rows.append(self._row(
                ts.DETAIL_SOURCE,
                f"{self.source_count} relevant source{plural} found", generic=True))
        if isinstance(self.passages, int) and self.passages > 0:
            plural = "s" if self.passages > 1 else ""
            rows.append(self._row(ts.DETAIL_EVIDENCE,
                                  f"{self.passages} relevant passage{plural} retrieved"))
        if self.found_something is True and not (self.source_names or self.passages):
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
        rows = [self._row(ts.DETAIL_OPERATION, op) for op in self.operations]
        if not rows and self.execution_type == ts.EXEC_DOCUMENTS and self.passages:
            rows = [self._row(ts.DETAIL_OPERATION, "Reading relevant passages"),
                    self._row(ts.DETAIL_OPERATION,
                              "Synthesizing the retrieved information")]
        if self.multi_source and self.source_count and self.source_count > 1:
            rows.append(self._row(ts.DETAIL_OPERATION,
                                  "Comparing information across sources"))
            rows.append(self._row(ts.DETAIL_OPERATION, "Reconciling results"))
        return rows

    def _preparing_details(self) -> list:
        # Nothing was produced, so list nothing. "Supporting summary" used to be
        # appended unconditionally, which on a clarify claimed an output that does
        # not exist.
        if self.no_answer:
            return [self._row(ts.DETAIL_OUTPUT,
                              "No answer could be produced for this question.",
                              ts.STATE_WARNING)]
        rows = []
        if self.output in ("chart", "chart+summary"):
            rows.append(self._row(ts.DETAIL_OUTPUT, "Preparing visualization"))
        if isinstance(self.row_count, int) and self.row_count > 1:
            plural = "s" if self.row_count != 1 else ""
            rows.append(self._row(ts.DETAIL_OUTPUT,
                                  f"Preparing table · {self.row_count} row{plural}"))
        rows.append(self._row(ts.DETAIL_OUTPUT, "Preparing summary"))
        if self.truncated:
            rows.append(self._row(ts.DETAIL_OUTPUT,
                                  "Showing the first page of results only",
                                  ts.STATE_WARNING))
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
