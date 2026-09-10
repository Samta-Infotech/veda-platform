"""veda/narrator.py — the OPTIONAL explainability narrator.

WHAT IT IS
    One short SLM call per query that turns already-confirmed structured facts into a
    more natural sentence for the progress UI. It is a NARRATOR, not a reasoning
    engine: it never decides anything, never sees the data, never sees SQL, and its
    output is text that only ever replaces other text.

WHY IT CANNOT BLOCK THE ANSWER  (the hard requirement)
    * It runs on its OWN daemon thread. Nothing in the query path awaits it.
    * It is started fire-and-forget and returns a handle immediately.
    * The handle can be cancelled. The front door cancels it the moment the query
      terminates, so a narration that loses the race is DISCARDED, not waited for.
    * Delivery is a callback into the existing progress stream. If the stream is
      already closed, the callback is a no-op.
    * Every failure path — timeout, transport error, refusal, malformed output,
      rejected validation — results in silence. The deterministic sentences in
      apps/chat/thinking_context.py are the floor, so silence costs nothing.

    There is no code path in which a narration failure can delay, reorder, prevent or
    corrupt a terminal answer event. It cannot: the answer path never reads from it.

WHY THE OUTPUT IS VALIDATED, NOT TRUSTED
    A narrator that invents a metric, names a table, or asserts a result that has not
    been computed is worse than no narrator — it manufactures confidence. So the
    output is checked against the facts it was given and rejected on any smell:
    banned terminology, a number that appears nowhere in the facts, quoting,
    multi-sentence output, or excess length.

    HONEST LIMIT: this validation is a filter, not a proof. It reliably catches
    engine/schema/SQL leakage and numeric invention, which are the failure modes that
    matter. It cannot prove a fluent sentence is semantically faithful. That is why
    the narration is confined to progress text and is never used for the answer, the
    validation ledger, or anything a user could mistake for a result.
"""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional

#: Hard ceiling on the narration. A progress line is one short sentence.
MAX_CHARS = 160

#: Terminology that must never reach a user. Covers engine/component names, storage
#: vocabulary, and the routing/planning grammar — the things the whole safe-projection
#: layer exists to keep internal.
_BANNED = (
    "sql", "select ", "from ", "where ", "join", "group by", "order by", "limit",
    "table", "column", "schema", "database", "db:", "postgres", "duckdb", "pgvector",
    "tier2", "tier-2", "tier 2", "tier1", "tier-1", "rag", "nosql", "llm", "slm",
    "embedding", "vector", "cosine", "rerank", "retrieval", "supervisor", "router",
    "routing", "schema_linking", "anchor", "pipeline", "agent", "coordinator",
    "prompt", "token", "model", "chain of thought", "chain-of-thought",
    "permission", "role", "policy", "grant", "rbac",
)

#: General-English vocabulary a narration may use freely. Everything OUTSIDE this set
#: must appear in the supplied facts.
#:
#: WHY THIS LIST HAD TO GROW. It started as a short set of progress verbs, and that
#: rejected 100% of real narrations — the SLM returned perfectly safe sentences like
#: "Analyzing data for the entire year of 2024." and the validator threw them out over
#: `entire` and `year`. Ordinary language is not a fabricated fact, and a filter that
#: rejects everything is not a safety property, it is a disabled feature.
#:
#: WHAT THE RULE STILL PROTECTS. A word that is neither general English nor present in
#: the facts is almost always an invented DOMAIN noun — "revenue", "headcount",
#: "churn", "attrition" — and that is the dangerous case, because a user can act on
#: it. Those are still rejected. The list therefore deliberately contains no business
#: metrics, no entity names and no domain nouns of any kind.
_FREE = frozenset("""
a an and the of or for to in on at by is are was were be been being with within
across over from into per each all any some this that these those it its
your you we our i'm im am not no only just also then than as if
looking checking finding working comparing reviewing identifying gathering
counting totalling totaling averaging summing preparing putting together building
calculating computing summarising summarizing breaking down ranking sorting
ordering grouping combining merging reading analysing analyzing examining
inspecting scanning searching retrieving fetching selecting filtering narrowing
matching aggregating evaluating assessing determining establishing confirming
result results answer answers output outputs information info data records record
rows entries items values value figure figures numbers number amount amounts
totals total average averages count counts sum highest lowest largest smallest
biggest greatest most least top bottom first last
period periods time times date dates day days week weeks month months
quarter quarters year years annual monthly weekly daily entire whole full complete
range window span duration
trend trends change changes changed increase increases decrease decreases
growth movement pattern patterns distribution breakdown split share
chart charts graph table tables summary summaries visualisation visualization
view report
category categories group groups groupings segment segments type types status
statuses kind kinds class classes label labels name names
source sources available covered relevant requested asked question questions
matching found present existing given specified
occurrence occurrences instance instances case cases
how many much what which where when
""".split())

_SENTENCE_END = re.compile(r"[.!?]")
_WORD = re.compile(r"[a-z][a-z'-]*")
_NUM = re.compile(r"\d+")


class NarrationHandle:
    """A cancellable, never-awaited narration in flight."""

    __slots__ = ("_cancelled", "_thread", "result")

    def __init__(self):
        self._cancelled = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.result: Optional[str] = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Discard any narration that has not been delivered yet.

        Deliberately does NOT join the thread: joining would be waiting, which is the
        one thing this must never do. The thread is a daemon and its callback checks
        `cancelled` before delivering, so an in-flight call simply drops its result.
        """
        self._cancelled.set()


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "EXPLAIN_NARRATOR_ENABLED", False))
    except Exception:
        return False


def _timeout_s() -> float:
    try:
        import config
        return float(getattr(config, "EXPLAIN_NARRATOR_TIMEOUT_S", 6.0) or 6.0)
    except Exception:
        return 6.0


def _facts_vocabulary(facts: dict) -> set:
    """Every word and number the narration is allowed to introduce."""
    vocab = set(_FREE)
    nums = set()
    for k, v in (facts or {}).items():
        for tok in _WORD.findall(str(k).lower()):
            vocab.add(tok)
        text = str(v).lower()
        for tok in _WORD.findall(text):
            vocab.add(tok)
        nums.update(_NUM.findall(text))
    return vocab, nums


def validate(text: str, facts: dict) -> Optional[str]:
    """The accepted narration, or None if it must be rejected.

    Rejects: empty/oversized output, more than one sentence, quoting or code-ish
    punctuation, any banned terminology, and any NUMBER that does not appear in the
    supplied facts (the cheapest reliable test for a fabricated result).
    """
    if not text:
        return None
    s = " ".join(str(text).split())
    if not s or len(s) > MAX_CHARS:
        return None
    # One sentence only. A narrator that starts explaining is out of scope.
    if len(_SENTENCE_END.findall(s)) > 1:
        return None
    if any(ch in s for ch in '`"\'{}[]()<>|;=*'):
        return None

    low = s.lower()
    if any(b in low for b in _BANNED):
        return None

    vocab, nums = _facts_vocabulary(facts)
    # A number the facts never mentioned is an invented quantity — the single most
    # dangerous thing a progress line can contain, because it reads like a result.
    for n in _NUM.findall(low):
        if n not in nums:
            return None
    # Content words must be grounded. Short words are treated as connective noise.
    for w in _WORD.findall(low):
        if len(w) > 3 and w not in vocab:
            return None
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


_SYSTEM = (
    "You rewrite a set of confirmed facts about a data question into ONE short, "
    "plain sentence for a progress indicator. Describe only what is being done, in "
    "the user's own terms. Never mention databases, tables, columns, SQL, models, "
    "or any system component. Never state a result or a number that is not in the "
    "facts. Never explain your reasoning. Reply with the sentence only, no quotes."
)


def _build_prompt(facts: dict, step: str) -> str:
    lines = [f"Step being described: {step}", "Confirmed facts:"]
    for k, v in (facts or {}).items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("One short sentence describing what is happening at this step:")
    return "\n".join(lines)


def start(facts: dict, step: str, on_ready: Callable[[str, str], None],
          *, slm_call=None, trace=None) -> NarrationHandle:
    """Kick off ONE narration, fire and forget. Returns immediately.

    `on_ready(step, sentence)` is invoked from the narrator thread ONLY if the call
    succeeded, validation passed, and the handle was not cancelled first. It is
    wrapped so a raising callback cannot escape into the thread and cannot affect
    anything else.

    One call per QUERY, not per phase — a call per phase would multiply latency and
    token cost for a progress line, and the spec is explicit that this starts as a
    single narration.
    """
    handle = NarrationHandle()
    if not _enabled() or not facts:
        return handle

    # Capture the caller's trace so the narrator's SLM call is VISIBLE in the
    # per-query ledger. A fresh thread starts with an empty contextvars context, so
    # without this `current_trace()` is null inside `_run` and call_slm's accounting
    # silently records nothing — the call would burn latency and tokens invisibly.
    # (Same class of bug as the source-profile ContextVar lost in the streaming
    # worker thread; captured here rather than left to be rediscovered.)
    if trace is None:
        try:
            from veda.explain import current_trace
            trace = current_trace()
        except Exception:
            trace = None

    def _run():
        try:
            if handle.cancelled:
                return
            if trace is not None and getattr(trace, "enabled", False):
                try:
                    from veda.explain import bind_trace
                    bind_trace(trace)
                except Exception:
                    pass
            call = slm_call
            if call is None:
                from slm._call_slm import call_slm

                def call(user, system):
                    return call_slm(user, system=system, purpose="explain_narrate",
                                    temperature=0.0, num_predict=48,
                                    timeout=int(_timeout_s()))
            raw = call(_build_prompt(facts, step), _SYSTEM)
            if handle.cancelled:
                return                       # answer already finished — discard
            ok = validate(raw, facts)
            if not ok:
                return
            handle.result = ok
            try:
                on_ready(step, ok)
            except Exception:
                pass
        except Exception:
            # Timeout, transport failure, circuit-open, anything: stay silent. The
            # deterministic sentence is already on screen.
            pass

    t = threading.Thread(target=_run, daemon=True, name="veda-explain-narrator")
    handle._thread = t
    t.start()
    return handle
