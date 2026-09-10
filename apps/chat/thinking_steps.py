"""apps.chat.thinking_steps — the FOUR user-facing steps of a turn.

WHAT THIS IS
    VEDA emits ~26 distinct internal progress phases (`supervisor_classify`,
    `sql_probe`, `tier2_filters`, `rag_synthesize`, …). A user should never see any
    of them. This module folds that stream into exactly FOUR fixed steps:

        1. Understanding your request
        2. Finding the right information
        3. Analyzing the information
        4. Preparing your answer

    Those four names are FIXED and deliberately generic: the same four have to read
    correctly for a SQL query, a document question, a NoSQL lookup, a hybrid answer,
    a federated multi-source join, a chart, a table and a plain summary. That is why
    there is no "Running SQL" step and no "Searching documents" step — an output-shaped
    or engine-shaped step list would leak the architecture and would not generalise.

WHY IT LIVES IN THE API TIER
    This is a display concern, and the api tier is already the boundary where internal
    phase names stop (``thinking_messages.py`` does exactly that today). It also means
    the query pipeline is untouched: this module only *reads* the event stream the
    engine already produces. No engine change, no execution risk, no reordering.

TIMING IS REAL, NEVER SIMULATED
    A step starts when the first phase mapped to it arrives, and ends when a phase
    mapped to a LATER step arrives (or when the turn terminates). Durations come from
    the engine's own `timestamp_ms` when the event carries one (lifecycle events do),
    else from this process's monotonic clock at the moment the event was consumed.
    Nothing is interpolated, and no timer runs on its own — a step's "active" duration
    is computed on demand from a real start stamp.

    The SLM narrator (see ``narration`` below) runs asynchronously in the engine and
    its latency is therefore NOT inside any step duration: narration arrives as a
    separate event and only ever replaces *text*, never a timestamp.

MONOTONIC BY CONSTRUCTION
    Steps never go backwards. VEDA's Tier-2 fallback genuinely re-does intent and
    entity work late in the turn (`tier2_intent`, `tier2_entity`), so a naive mapping
    of those to "Understanding" would make a completed step re-open — which reads as a
    bug to a user. Any event mapping to an already-finished step is recorded against
    the CURRENT step instead of reopening the old one.
"""
from __future__ import annotations

import time

# ── the four fixed steps ─────────────────────────────────────────────────────
STEP_UNDERSTANDING = "understanding"
STEP_FINDING = "finding"
STEP_ANALYZING = "analyzing"
STEP_PREPARING = "preparing"

STEP_ORDER = (STEP_UNDERSTANDING, STEP_FINDING, STEP_ANALYZING, STEP_PREPARING)

STEP_TITLES = {
    STEP_UNDERSTANDING: "Understanding your request",
    STEP_FINDING: "Finding the right information",
    STEP_ANALYZING: "Analyzing the information",
    STEP_PREPARING: "Preparing your answer",
}

STATE_PENDING = "pending"
#: The step never ran AND the turn moved past it. Distinct from `pending` (still to
#: come) and from `completed` (work happened). Needed because the four steps are a
#: FIXED narrative while the phase stream is not: the document/RAG head emits nothing
#: that maps to "Analyzing", so that step was left `pending` while step 4 showed a
#: green tick — a ○ sandwiched between two ✓, which reads as broken.
STATE_SKIPPED = "skipped"
STATE_ACTIVE = "active"
STATE_COMPLETED = "completed"
STATE_WARNING = "warning"
STATE_FAILED = "failed"

#: Internal phase -> step. Derived from the phases the repository ACTUALLY emits
#: (chatbot/nodes.py, veda_hybrid.py, veda/pipeline.py, query/rag_layer.py,
#: query/lg_nodes.py, veda/lifecycle.py, apps/chat/services.py), not from a guess.
#:
#: Two deliberate departures from the obvious reading:
#:
#:  * `tier2_*` -> ANALYZING, not UNDERSTANDING. Tier-2 is the fallback that runs
#:    AFTER the deterministic head has already failed, so its intent/entity work
#:    happens late in the turn. Filing it under Understanding would reopen a
#:    finished step. (The monotonic guard would catch it anyway; the mapping is
#:    honest about when the work happens.)
#:
#:  * `data_retrieval` + `validation` -> ANALYZING, not FINDING. In VEDA the SQL
#:    itself performs the aggregation/comparison, so executing it IS the analysis;
#:    "Finding" is about deciding WHERE the answer lives and whether the caller may
#:    see it.
PHASE_TO_STEP = {
    # ── 1. Understanding ────────────────────────────────────────────────────
    "received": STEP_UNDERSTANDING,             # lifecycle
    "supervisor_classify": STEP_UNDERSTANDING,  # chatbot/nodes.py
    "supervisor_followup": STEP_UNDERSTANDING,
    "classify": STEP_UNDERSTANDING,             # veda_hybrid
    "understanding": STEP_UNDERSTANDING,        # lifecycle

    # ── 2. Finding ──────────────────────────────────────────────────────────
    "access_check": STEP_FINDING,               # lifecycle — shown as a SUB-CHECK
    "route": STEP_FINDING,                      # veda_hybrid
    "source_selection": STEP_FINDING,           # lifecycle
    "execution_plan": STEP_FINDING,             # lifecycle
    "sql_probe": STEP_FINDING,                  # veda_hybrid
    "schema_linking": STEP_FINDING,             # pipeline _tick
    "rag": STEP_FINDING,                        # head announcements
    "hybrid": STEP_FINDING,
    "nosql": STEP_FINDING,
    "rag_retrieve": STEP_FINDING,               # rag_layer
    "hybrid_retrieve": STEP_FINDING,

    # ── 3. Analyzing ────────────────────────────────────────────────────────
    "sql_planning": STEP_ANALYZING,             # pipeline _tick
    "decompose": STEP_ANALYZING,                # veda_hybrid
    "sub_query": STEP_ANALYZING,
    "tier2": STEP_ANALYZING,
    "tier2_intent": STEP_ANALYZING,             # lg_nodes
    "tier2_entity": STEP_ANALYZING,
    "tier2_columns": STEP_ANALYZING,
    "tier2_filters": STEP_ANALYZING,
    "tier2_assemble": STEP_ANALYZING,
    "nosql_build": STEP_ANALYZING,
    "rag_synthesize": STEP_ANALYZING,           # rag_layer
    "hybrid_synthesize": STEP_ANALYZING,
    "data_retrieval": STEP_ANALYZING,           # lifecycle
    "cross_source_processing": STEP_ANALYZING,  # lifecycle
    "validation": STEP_ANALYZING,               # lifecycle

    # ── 4. Preparing ────────────────────────────────────────────────────────
    "result_preparation": STEP_PREPARING,       # lifecycle
    "answer": STEP_PREPARING,                   # veda_hybrid
    "output": STEP_PREPARING,                   # pipeline _tick
    "visualization_prep": STEP_PREPARING,       # apps/chat/services.py
    "completed": STEP_PREPARING,                # lifecycle
}

#: Phases rendered as a named SUB-CHECK inside their step rather than as the step's
#: own progress. Authorization is the one the product asked for by name: it must be
#: *visible* and *timed*, but it is not a top-level stage of answering a question.
SUB_CHECK_PHASES = {
    "access_check": ("access", "Checking access permissions"),
    "validation": ("validation", "Checking the result is complete and safe"),
}

#: Sub-check status -> the sentence shown. Never names a policy, role, permission id
#: or any other authorization internal — only the outcome, in the user's terms.
_ACCESS_COPY = {
    STATE_COMPLETED: "You have permission to access the required information",
    STATE_WARNING: ("Access is limited for some information. "
                    "I'll continue with the information available to you."),
    STATE_FAILED: "The required information isn't available with your current access.",
    STATE_ACTIVE: "Checking your access",
}

_VALIDATION_COPY = {
    STATE_COMPLETED: "Checks passed",
    STATE_WARNING: "The result has some limitations",
    STATE_FAILED: "The result did not pass a safety check",
    STATE_ACTIVE: "Checking the result",
}

_SUB_CHECK_COPY = {"access": _ACCESS_COPY, "validation": _VALIDATION_COPY}

#: The kinds of evidence a step may expose when the reader opens it. A CLOSED
#: vocabulary, like the phase list: a detail whose type is not one of these cannot
#: be rendered, so a new backend event cannot invent a new row shape in the UI.
#:
#: These describe what the system DID, in the user's terms — never how it is built.
#: `operation` is the one that carries semantics ("Grouping results", "Reading
#: relevant passages") and it is populated only from operations the backend actually
#: reported, never inferred from the question.
DETAIL_ACCESS = "access"          # authorization, timed
DETAIL_SOURCE = "source"          # a source that took part, by display name
DETAIL_EVIDENCE = "evidence"      # what was retrieved: passages, rows, records
DETAIL_OPERATION = "operation"    # a semantic step of the analysis
DETAIL_VALIDATION = "validation"  # a safety/consistency check
DETAIL_OUTPUT = "output"          # what is being produced

DETAIL_TYPES = (DETAIL_ACCESS, DETAIL_SOURCE, DETAIL_EVIDENCE,
                DETAIL_OPERATION, DETAIL_VALIDATION, DETAIL_OUTPUT)

#: Execution shape, for the "how this answer was generated" flow. Derived from the
#: route the engine actually took, never guessed from the question.
EXEC_SQL = "sql"
EXEC_DOCUMENTS = "documents"
EXEC_MULTI_SOURCE = "multi_source"
EXEC_UNKNOWN = "unknown"

#: route/intent value on the engine's `route` event -> execution shape.
_EXEC_FROM_INTENT = {
    "sql": EXEC_SQL, "deterministic": EXEC_SQL, "tier2": EXEC_SQL,
    "rag": EXEC_DOCUMENTS, "doc": EXEC_DOCUMENTS, "document": EXEC_DOCUMENTS,
    "hybrid": EXEC_MULTI_SOURCE, "federated": EXEC_MULTI_SOURCE,
    "multi": EXEC_MULTI_SOURCE, "nosql": EXEC_SQL,
}

#: Deterministic fallback sentence per step, used until (or instead of) narration.
#: Generic on purpose — it must be true for every engine and every output shape.
_GENERIC_CONTEXT = {
    STEP_UNDERSTANDING: "Working out what you're asking for.",
    STEP_FINDING: "Looking for the information that answers this.",
    STEP_ANALYZING: "Working through the information found.",
    STEP_PREPARING: "Putting your answer together.",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


class Step:
    """One of the four steps, with real timing and its own sub-checks."""

    __slots__ = ("key", "title", "index", "state", "started_ms", "ended_ms",
                 "summary", "sub_checks", "details", "_narrated")

    def __init__(self, key: str, index: int):
        self.key = key
        self.title = STEP_TITLES[key]
        self.index = index
        self.state = STATE_PENDING
        self.started_ms: int | None = None
        self.ended_ms: int | None = None
        #: The collapsed one-liner. Named `summary` to match the frontend contract.
        self.summary: str | None = None
        #: Authorization / validation checks, kept separate because they carry their
        #: own measured duration and resolve independently of the step.
        self.sub_checks: list = []
        #: Ordered evidence rows, each `{type, label, state}`. A LIST, not a dict:
        #: the reader is shown a sequence of things that happened, and order is
        #: information ("access verified" before "5 passages retrieved").
        self.details: list = []
        self._narrated = False

    # -- timing --------------------------------------------------------------
    def duration_ms(self, now_ms: int | None = None) -> int | None:
        """Real elapsed time. Frozen once the step ends; live while it is active.

        None for a step that never started — the UI shows "—" rather than 0, because
        0 s would imply the step ran instantly when in fact it has not begun.
        """
        if self.started_ms is None:
            return None
        if self.ended_ms is not None:
            return max(0, self.ended_ms - self.started_ms)
        return max(0, (now_ms if now_ms is not None else _now_ms()) - self.started_ms)

    def as_dict(self, now_ms: int | None = None) -> dict:
        d = {
            "id": self.key,
            "index": self.index,
            "title": self.title,
            "state": self.state,
            "duration_ms": self.duration_ms(now_ms),
            "summary": self.summary or _GENERIC_CONTEXT[self.key],
        }
        # A step that has not STARTED shows nothing inside it. The api tier measures
        # RBAC before the engine is even called, so the access sub-check lands on the
        # very first frame — and a resolved ✓ check displayed inside a ○ step reads as
        # broken (observed in a real stream: step 2 `pending`, its access check
        # already completed at 52 ms, for the first ~7 seconds of the turn).
        #
        # The measurement is KEPT, not discarded: it surfaces the moment the step
        # opens. Hiding it is preferable to opening the step early, which would
        # overclaim on the smalltalk path — there the api tier still measures RBAC
        # while the engine is bypassed entirely, so "Finding the right information"
        # must stay pending.
        _visible = self.started_ms is not None
        rows: list = []
        if _visible:
            # Sub-checks first: authorization and validation are the framing facts,
            # and they carry a measured duration the other rows do not.
            for c in self.sub_checks:
                row = {"type": c["kind"], "label": c.get("label") or "",
                       "state": c.get("state")}
                if c.get("duration_ms") is not None:
                    row["duration_ms"] = c["duration_ms"]
                if c.get("message"):
                    row["message"] = c["message"]
                rows.append(row)
            rows.extend({k: v for k, v in d0.items() if not k.startswith("_")}
                        for d0 in self.details)
        d["details"] = rows
        # `expandable` is the UI's cue that opening this step shows more than the
        # collapsed line already says.
        d["expandable"] = bool(rows)
        return d


class ThinkingStepTracker:
    """Folds the internal phase stream into the four steps.

    Feed it every ``thinking`` payload the turn produces; it returns whether the
    step model changed, and ``snapshot()`` renders the collapsed view. It holds no
    reference to the pipeline and cannot affect it.
    """

    def __init__(self):
        self.steps = {k: Step(k, i + 1) for i, k in enumerate(STEP_ORDER)}
        self._current: str | None = None
        self._first_ms: int | None = None
        self._last_ms: int | None = None
        self.finished = False
        #: COUNTED evidence, never a score. "5 passages" and "20 rows" are facts the
        #: reader can weigh; a confidence percentage the backend did not define is
        #: not. Only keys the backend actually reported are present.
        self.evidence: dict = {}
        #: Which shape of execution answered this — sql / documents / multi_source.
        #: Taken from the route the engine reports, never inferred from the question.
        self.execution_type: str = EXEC_UNKNOWN
        #: Terminal outcome, set once by finish().
        self.status: str = "active"
        self.error_code: str | None = None
        self.retryable: bool | None = None

    # -- ingest --------------------------------------------------------------
    def consume(self, payload: dict) -> bool:
        """Fold one ``thinking`` payload in. Returns True when the model changed.

        Never raises: a malformed or unknown event is ignored rather than allowed to
        break the turn's event stream.
        """
        try:
            self._absorb_evidence(payload)
            phase = (payload or {}).get("phase")
            if not phase:
                return False
            step_key = PHASE_TO_STEP.get(phase)
            if step_key is None:
                return False        # an unmapped internal phase is simply not shown

            # Engine timestamp when present (lifecycle events carry one) — closer to
            # when the work actually happened than our own consume time.
            ts = payload.get("timestamp_ms")
            now = int(ts) if isinstance(ts, (int, float)) and ts > 0 else _now_ms()
            if self._first_ms is None:
                self._first_ms = now
            self._last_ms = now

            status = payload.get("status")          # lifecycle events only
            sub = SUB_CHECK_PHASES.get(phase)

            # MONOTONIC: never reopen a finished step (Tier-2 re-does intent work
            # late in the turn — see the module docstring).
            target = step_key
            if self._current is not None:
                if STEP_ORDER.index(step_key) < STEP_ORDER.index(self._current):
                    target = self._current

            if sub is not None:
                # A sub-check ALWAYS attaches to its mapped step, never to whatever
                # step happens to be current. The access check's `completed` arrives
                # late (after validation), so routing it through the monotonic guard
                # filed the START under "Finding" and the COMPLETION under
                # "Preparing" — the same check appearing twice, once unresolved, and
                # authorization shown under the wrong heading. The guard exists to
                # stop STEPS moving backwards; a sub-check is not step progress.
                self._record_sub_check(step_key, sub, status, now)
                # A sub-check OPENS its own step — the access check is genuinely
                # part of "Finding", and leaving the step pending while its
                # sub-check is already running reads as inconsistent. But it must
                # never COMPLETE the step, or the access check would look like the
                # whole of "Finding".
                #
                # Safe against reopening: _ensure_started only stamps a start when
                # there isn't one, so a late `completed` on an already-finished step
                # changes nothing.
                #
                # A sub-check that opens a LATER step must also ADVANCE, closing the
                # earlier ones. Measured live: the access check opens "Finding" but
                # left `_current` on "Understanding", so Understanding kept accruing
                # until the first non-sub-check Finding event and reported 71 s for
                # work that takes ~3 s. The access check genuinely marks the start of
                # looking for the information.
                if (self._current is not None
                        and STEP_ORDER.index(step_key) > STEP_ORDER.index(self._current)):
                    self._advance_to(step_key, now)
                else:
                    self._ensure_started(step_key, now)
                return True

            self._advance_to(target, now)
            self._apply_status(target, status)
            return True
        except Exception:
            return False

    def _absorb_evidence(self, payload: dict) -> None:
        """Counted facts and the execution shape, taken from events as they pass.

        Deliberately narrow: it reads only fields the engine already emits, and it
        never derives a count from anything else. An absent count stays absent —
        "0 passages" and "we did not report passages" are different claims.
        """
        try:
            p = payload or {}
            det = p.get("details") if isinstance(p.get("details"), dict) else {}
            # `rag_retrieve` carries the passage count on the event ITSELF, while
            # `source_count` arrives nested in `details` — the engine is already
            # inconsistent about placement, so read both rather than depend on
            # which side of that inconsistency a future event lands on.
            chunks = p.get("chunks")
            if chunks is None:
                chunks = det.get("chunks")
            if isinstance(chunks, int) and chunks >= 0:
                self.evidence["passages"] = chunks
            src = det.get("source_count")
            if isinstance(src, int) and src >= 0:
                self.evidence["sources"] = src
            rows = p.get("rows") if isinstance(p.get("rows"), int) else det.get("rows")
            if isinstance(rows, int) and rows >= 0:
                self.evidence["rows"] = rows
            intent = p.get("intent") or det.get("intent")
            if intent:
                shape = _EXEC_FROM_INTENT.get(str(intent).strip().lower())
                # Only ever move toward MORE specific: a later generic event must not
                # downgrade a known shape back to "unknown".
                if shape and (self.execution_type == EXEC_UNKNOWN
                              or shape == EXEC_MULTI_SOURCE):
                    self.execution_type = shape
        except Exception:
            pass

    def set_evidence(self, **counts) -> None:
        """Counted evidence the api tier knows and the event stream does not."""
        for k, v in counts.items():
            if isinstance(v, int) and v >= 0:
                self.evidence[k] = v

    def add_detail(self, step_key: str, kind: str, label: str,
                   state: str = STATE_COMPLETED, *, terminal: bool = False) -> bool:
        """Append one evidence row to a step. Order is preserved and meaningful."""
        st = self.steps.get(step_key)
        if st is None or kind not in DETAIL_TYPES or not label:
            return False
        if not terminal and st.started_ms is None:
            return False
        row = {"type": kind, "label": str(label)[:160], "state": state}
        if row in st.details:
            return False                      # idempotent: no duplicate rows
        st.details.append(row)
        return True

    def _ensure_started(self, key: str, now: int) -> None:
        st = self.steps[key]
        if st.started_ms is None:
            st.started_ms = now
            st.state = STATE_ACTIVE
        if self._current is None:
            self._current = key

    def _advance_to(self, key: str, now: int) -> None:
        """Open `key`, closing every earlier step that is still open."""
        idx = STEP_ORDER.index(key)
        for earlier in STEP_ORDER[:idx]:
            st = self.steps[earlier]
            if st.started_ms is not None and st.ended_ms is None:
                st.ended_ms = now
                if st.state in (STATE_ACTIVE, STATE_PENDING):
                    st.state = STATE_COMPLETED
            elif st.started_ms is None:
                # A step no event ever mapped to genuinely did not run — leave it
                # pending with a null duration rather than inventing a 0 s pass.
                pass
        self._ensure_started(key, now)
        self._current = key

    def _apply_status(self, key: str, status) -> None:
        """Lifecycle statuses carry outcome; legacy phases carry none."""
        st = self.steps[key]
        if status == "completed":
            # Do NOT end the step here: several lifecycle phases map to one step
            # (data_retrieval + validation both land in Analyzing), so the step ends
            # only when a LATER step opens, or the turn terminates.
            if st.state == STATE_ACTIVE:
                st.state = STATE_ACTIVE
        elif status == "warning" and st.state != STATE_FAILED:
            st.state = STATE_WARNING
        elif status == "failed":
            st.state = STATE_FAILED

    def set_sub_check_duration(self, step_key: str, kind: str, duration_ms: float) -> None:
        """Override a sub-check's duration with a MEASURED one.

        The engine's `access_check` phase spans "scope resolved" to "validation
        passed" — that is when access could be CONFIRMED, not how long checking took.
        Inferring the duration from those two events reported 26.6 s live for work
        that is sub-millisecond, which is worse than reporting nothing. The api tier
        times the actual RBAC resolution (it performs it), so that measurement wins.
        """
        st = self.steps.get(step_key)
        if st is None:
            return
        existing = next((c for c in st.sub_checks if c["kind"] == kind), None)
        if existing is None:
            existing = {"kind": kind,
                        "label": SUB_CHECK_PHASES.get("access_check", ("", ""))[1],
                        "state": STATE_ACTIVE, "_started_ms": None,
                        "duration_ms": None, "message": None}
            st.sub_checks.append(existing)
        existing["duration_ms"] = max(0, int(duration_ms))
        existing["_measured"] = True
        # A measured duration means the api tier ALREADY completed the RBAC
        # resolution (it runs before the engine is called, and a failure there
        # never reaches this service). Leaving it "active" showed the user a
        # permanently spinning access check — observed on the smalltalk path,
        # where no engine access_check event ever arrives to resolve it.
        if existing["state"] == STATE_ACTIVE:
            existing["state"] = STATE_COMPLETED
            existing["message"] = _ACCESS_COPY[STATE_COMPLETED]


    def _record_sub_check(self, step_key: str, sub, status, now: int) -> None:
        kind, label = sub
        st = self.steps[step_key]
        existing = next((c for c in st.sub_checks if c["kind"] == kind), None)
        if existing is None:
            existing = {"kind": kind, "label": label, "state": STATE_ACTIVE,
                        "_started_ms": now, "duration_ms": None, "message": None}
            st.sub_checks.append(existing)

        if status in (STATE_COMPLETED, STATE_WARNING, STATE_FAILED):
            existing["state"] = status
            # A measured duration from the api tier is authoritative — never
            # overwrite it with the event-gap estimate (see set_sub_check_duration).
            if not existing.get("_measured") and existing.get("_started_ms"):
                existing["duration_ms"] = max(0, now - existing["_started_ms"])
        copy = _SUB_CHECK_COPY.get(kind, {})
        existing["message"] = copy.get(existing["state"]) or existing["message"]

    # -- context / details ---------------------------------------------------
    def set_context(self, step_key: str, sentence: str, *, from_narrator=False) -> bool:
        """Set a step's collapsed one-liner.

        Narration WINS over the deterministic sentence, but only once: a late second
        narration cannot churn the text a user has already read.
        """
        st = self.steps.get(step_key)
        if st is None or not sentence:
            return False
        if st._narrated and not from_narrator:
            return False
        st.summary = str(sentence)[:240]
        if from_narrator:
            st._narrated = True
        return True

    def set_details(self, step_key: str, details: list, *, terminal: bool = False) -> bool:
        """Attach expandable content to a step.

        A step that has NOT STARTED may not claim content while the turn is still
        running. The live loop refreshes all four steps on every progress frame, and
        `_preparing_details` always lists at least a "Supporting summary" — so from
        the very FIRST frame, before anything had run, step 4 sat `pending` while
        already advertising an output it might never produce. Observed in a real
        stream at `total_duration_ms: 0`.

        `terminal=True` lifts the guard, because at the terminal frame content on a
        never-started step is meaningful evidence rather than a guess: it is how the
        document/RAG head's unreported "Analyzing" work is recognised (see finish()).
        """
        st = self.steps.get(step_key)
        if st is None or not details:
            return False
        if not terminal and st.started_ms is None:
            return False
        changed = False
        for row in details:
            if not isinstance(row, dict):
                continue
            kind, label = row.get("type"), row.get("label")
            if kind not in DETAIL_TYPES or not label:
                continue
            entry = {"type": kind, "label": str(label)[:160],
                     "state": row.get("state") or STATE_COMPLETED}
            if row.get("_generic"):
                entry["_generic"] = True
            elif any(e.get("_generic") and e["type"] == kind for e in st.details):
                # A SPECIFIC row supersedes the placeholder of the same type. The
                # live loop shows "1 relevant source found" while the name is
                # unknown; once the payload names "homzhub", showing both is
                # showing the same fact twice, once vaguely.
                st.details = [e for e in st.details
                              if not (e.get("_generic") and e["type"] == kind)]
                changed = True
            if entry not in st.details:
                st.details.append(entry)
                changed = True
        return changed

    # -- terminal ------------------------------------------------------------
    def finish(self, *, failed: bool = False, error_code: str | None = None,
               retryable: bool | None = None) -> None:
        """Freeze every open step at the real terminal moment."""
        now = _now_ms()
        self._last_ms = now
        for key in STEP_ORDER:
            st = self.steps[key]
            if st.started_ms is not None and st.ended_ms is None:
                st.ended_ms = now
                if st.state == STATE_ACTIVE:
                    st.state = STATE_FAILED if failed else STATE_COMPLETED
            elif st.started_ms is None:
                # The step never ran. Two different situations, and they must not be
                # rendered the same way.
                #
                # (a) The turn moved PAST it — a later step started. Leaving it
                #     `pending` puts a ○ between two ✓ and reads as broken. Measured
                #     live on the document/RAG head, which emits no phase mapping to
                #     "Analyzing" at all: the step sat pending while "Preparing" was
                #     already green.
                #
                #     If the step carries CONTENT (details recovered from the final
                #     payload, e.g. the operations that ran), the work demonstrably
                #     happened and was simply never reported as a phase — so
                #     `completed`, but with NO duration, because none was measured and
                #     inventing one would be worse than admitting it is unknown.
                #     With no content, `skipped` is the honest answer.
                #
                # (b) Nothing after it ran either — smalltalk bypasses the engine
                #     entirely. It stays `pending`, which is correct.
                #
                # Either way nothing INSIDE a step that never ran may be displayed:
                # smalltalk still had its RBAC resolution measured by the api tier and
                # an access check attached to "Finding", under a step that never
                # happened.
                later_ran = any(self.steps[k].started_ms is not None
                                for k in STEP_ORDER[STEP_ORDER.index(key) + 1:])
                if later_ran:
                    st.state = (STATE_COMPLETED if st.details
                                else STATE_SKIPPED)
                    if st.state == STATE_SKIPPED:
                        st.details = []
                if st.sub_checks:
                    st.sub_checks = []
        self.finished = True
        self.status = "failed" if failed else "completed"
        if error_code:
            self.error_code = error_code
            self.retryable = retryable

    # -- output --------------------------------------------------------------
    def total_duration_ms(self) -> int | None:
        if self._first_ms is None:
            return None
        end = self._last_ms if self.finished else _now_ms()
        return max(0, (end or self._first_ms) - self._first_ms)

    def snapshot(self) -> list:
        """The collapsed view: all four steps, in order, with live/frozen timing."""
        now = _now_ms()
        return [self.steps[k].as_dict(now) for k in STEP_ORDER]

    def current_step(self) -> str | None:
        return self._current

    def has_progress(self) -> bool:
        """Did ANY step actually begin?

        False for a turn that bypassed the engine entirely — a canned greeting
        answers in ~100 ms and no phase is ever emitted. There is no progress to
        show, and showing the four steps anyway rendered four empty circles above a
        finished answer, under a `completed` status: nothing had completed. The
        honest representation of "no work happened" is no progress display, not a
        progress display full of blanks.
        """
        return any(st.started_ms is not None for st in self.steps.values())

    def as_payload(self) -> dict:
        """The normalized model the frontend renders. NOT the raw event stream.

        One object describes the whole turn, so a client never has to fold events
        together or decide which of several backend phases means the same thing —
        that is this module's job, and doing it here means every client agrees.

        Terminal consistency is enforced at the boundary rather than trusted: a
        finished turn cannot report an active step, and an unfinished one cannot
        report a terminal status. Both contradictions were observable in real
        streams before this.
        """
        steps = self.snapshot()
        if self.finished:
            # §12: exactly one terminal state, internally consistent.
            for st in steps:
                if st["state"] == STATE_ACTIVE:
                    st["state"] = STATE_FAILED if self.status == "failed" \
                        else STATE_COMPLETED
        out = {
            "type": "thinking",
            "status": self.status,
            "current_step": None if self.finished else self._current,
            "steps": steps,
            "evidence": dict(self.evidence),
            "execution": {"type": self.execution_type},
            "timing": {"total_duration_ms": self.total_duration_ms()},
        }
        if self.error_code:
            out["error"] = {"code": self.error_code,
                            "retryable": bool(self.retryable)}
        return out
