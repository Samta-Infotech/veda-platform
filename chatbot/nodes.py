"""chatbot.nodes — LangGraph node functions.

Each node takes a ChatState and returns a partial dict to merge into it
(standard LangGraph node signature). Nodes that need to report mid-turn
progress (classify_node, context_resolve_node, call_engine_node) also
declare a `config: RunnableConfig` parameter — LangGraph injects it
automatically for any node function whose signature names a parameter
`config` (see langgraph/_internal/_runnable.py's KWARGS_CONFIG_KEYS); nodes
that don't need it are unaffected, mixed signatures in one graph are fine.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from apps.query.inference_client import InferenceClient, InferenceUnavailable

from .llm import CHATBOT_CLASSIFY_MODEL, call_slm
from .memory import frame as memory_frame
from .memory.classify import DELTA_TYPES, classify_delta, parse_delta_response
from .memory.store import MemoryStore
from .prompts import (
    FALLBACK_REPLY,
    FOLLOWUP_SYSTEM_PROMPT,
    STANDALONE_CHECK_SYSTEM,
    build_followup_user_prompt,
    build_smalltalk_system_prompt,
    build_standalone_check_user_prompt,
    build_supervisor_system_prompt,
    build_supervisor_user_prompt,
)
from .state import ChatState

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"smalltalk", "followup", "clarify_reply", "answer"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Deterministic safety net, on top of the LLM classifier: generic (schema-
# agnostic — no table/column names) words that essentially never appear in
# pure smalltalk. If the LLM still says "smalltalk" despite one of these being
# present, override it — a real question wrongly treated as smalltalk means
# smalltalk_node (which has no data access) would have to invent an answer,
# which this system must never do (refuse-over-guess, same as the rest of
# the codebase's firewall).
_DATA_QUESTION_HINTS = re.compile(
    r"\b(count|how many|total|list|show me|average|sum|status|data|record|"
    r"table|report|number of|which|what was|remind me)\b",
    re.IGNORECASE,
)

# Cheap, deterministic PRE-FILTER for _depends_on_history: only messages that
# contain some referential/anaphoric language even PLAUSIBLY depend on earlier
# conversation to mean something concrete. A bare greeting ("hi") or a self-
# introduction ("my name is raj") contains none of these and can never depend
# on history no matter what it contains — asking a model "could this secretly
# depend on the conversation" for such messages produced real, observed false
# positives in production (a bare "hi" and "my name is raj" were each rewritten
# into bogus, unfiltered database queries — see the incident traces this fix
# responds to). This is intentionally NOT trying to detect every kind of
# follow-up (the docstring below explains why a fixed word list can't do that);
# it only needs to catch messages that couldn't possibly qualify, so the model
# call is skipped for those instead of trusted to always get them right.
_REFERENTIAL_HINTS = re.compile(
    r"\b(that|this|it|those|these|same|again|more|other|another|previous|"
    r"above|below|instead|also|too|earlier|before|last one|the one)\b",
    re.IGNORECASE,
)

def _depends_on_history(message: str, history: list) -> bool:
    """Generic (non-keyword) second opinion for a "smalltalk" verdict when prior
    turns exist AND the message contains at least some referential language
    (_REFERENTIAL_HINTS — see its docstring for why that pre-filter exists).
    Real users phrase referential follow-ups countless ways ("need more
    details", "aur bata", "what about the other one", ...) — no fixed word list
    generalizes to production traffic, so THIS part asks the LLM the underlying
    semantic question directly instead of pattern-matching specific phrasings.
    Fails closed to False (trust the original "smalltalk" verdict) on any error,
    since this is only a second-opinion check, not the primary classifier."""
    user_prompt = build_standalone_check_user_prompt(message, history)
    verdict = call_slm(STANDALONE_CHECK_SYSTEM, user_prompt, max_tokens=5,
                       model=CHATBOT_CLASSIFY_MODEL, purpose="standalone_check")
    return bool(verdict) and "dependent" in verdict.strip().lower()

# Deterministic fast path for the overwhelming majority of smalltalk: pure
# greetings/thanks/farewells with nothing else in the message. Tight, anchored
# patterns (whole-message match, not substring) so they can never misfire on a
# real question that merely starts with "hi" or ends with "thanks" — and
# _DATA_QUESTION_HINTS is still checked as a second guard before trusting this.
# Skips the classify LLM call entirely (classify_node) and lets smalltalk_node
# skip its own LLM call too — on this deployment's hardware a single such call
# alone can take ~20s, so a bare "hi" was paying 20-40+ seconds of pure LLM
# round-trip time for something that should be instant.
_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|good\s*(morning|afternoon|evening|day)|"
    r"how\s*(are\s*(you|u)|'?s\s*it\s*going)( doing)?)\s*[.,!?]*\s*$", re.IGNORECASE)
_THANKS_RE = re.compile(
    r"^\s*(thanks?( you)?( very much| so much| a lot)?|thx|ty|appreciate it|"
    r"much appreciated|cheers)\s*[.,!?]*\s*$", re.IGNORECASE)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see\s*(you|ya)( later| soon)?|take care|good\s*night)\s*[.,!?]*\s*$",
    re.IGNORECASE)

# Deterministic fast path for pure runtime-value questions ("what's the current
# date", "what time is it") — skips the classify LLM call (and its thinking event)
# the same way the smalltalk patterns above do. Deliberately a SEPARATE, minimal
# duplicate of query/runtime_context.py's patterns, not an import of it — chatbot/
# runs in the api container and must never import veda_core directly (same
# boundary chatbot/llm.py's call_slm already documents). The actual answer is
# still computed exactly once, downstream, by query/runtime_context.py in the
# inference tier — this only decides whether to skip the LLM classify round-trip.
_RUNTIME_CONTEXT_RE = re.compile(
    r"^\s*what(?:'s| is) (?:the )?(?:current )?date and time\s*\??\s*$"
    r"|^\s*current date and time\s*\??\s*$"
    r"|^\s*what(?:'s| is) (?:the )?(?:current date|today'?s? date|date(?: today)?)\s*\??\s*$"
    r"|^\s*(?:today'?s? date|current date)\s*\??\s*$"
    r"|^\s*what date is it(?: today)?\s*\??\s*$"
    r"|^\s*what day (?:is it|of the week is it)(?: today)?\s*\??\s*$"
    r"|^\s*what(?:'s| is) (?:the )?current time\s*\??\s*$"
    r"|^\s*current time\s*\??\s*$"
    r"|^\s*what time is it(?: now)?\s*\??\s*$",
    re.IGNORECASE,
)

# Deterministic fast path for an explicit hard reset of the structured
# analytical memory (audit fix H2 — MemoryStore.reset() existed but was
# never wired to anything). Whole-message match, same anchored style as the
# smalltalk patterns above, so it never misfires on a real question that
# merely contains one of these words mid-sentence.
_RESET_RE = re.compile(
    r"^\s*(start over|reset(?: everything)?|forget (everything|that|it)|"
    r"clear (the )?context|new topic|let'?s start (over|fresh|again))\s*[.,!?]*\s*$",
    re.IGNORECASE,
)

# Deterministic fast path for "pop one drill level" navigation ("go back",
# "go back again", "undo that filter", "zoom out"). These words carry NO data
# content of their own — sent through the LLM classifier, they were observed
# (2026-07 memory-layer testing) to be non-deterministically misjudged as
# smalltalk turn-to-turn (same exact message, different verdict on repeat
# runs), and even when correctly routed to "followup", the classifier's own
# delta_type still defaulted to "new_topic" (its own prompt instructs that for
# any smalltalk verdict), which then made render_frame_as_query() send the
# literal word "back"/"again" to the SQL engine as if it were part of the
# question — the engine then tried (and failed) to match it against columns/
# values. Anchored whole-message match (same style as _RESET_RE above) so it
# never misfires on a real question that merely contains "back" mid-sentence.
# Only fires when there's an actual drill level to pop (frame + non-empty
# drill_stack) — otherwise falls through to the normal LLM classification,
# since "go back" with nothing to go back FROM isn't unambiguous.
_DRILL_UP_RE = re.compile(
    r"^\s*(go\s+back(\s+(again|once\s+more|one\s+more\s+time))?|back\s+up|"
    r"go\s+up(\s+(a|one)\s+level)?|(zoom|step)\s+out|previous(\s+(level|view|step))?|"
    r"undo(\s+(that|it|the\s+(last|previous)\s+filter))?|"
    r"remove\s+(that|the\s+(last|previous))\s+filter)\s*[.,!?]*\s*$",
    re.IGNORECASE,
)


def _canned_smalltalk_reply(message: str) -> str | None:
    """Instant reply for the fast-path patterns above — None means "not a fast
    match, fall back to the LLM" (used by both classify_node and smalltalk_node
    so the two stay in lockstep on what counts as trivial smalltalk)."""
    if _DATA_QUESTION_HINTS.search(message):
        return None
    if _GREETING_RE.match(message):
        return FALLBACK_REPLY
    if _THANKS_RE.match(message):
        return "You're welcome! Let me know if you have any other data questions."
    if _BYE_RE.match(message):
        return "Goodbye! Come back anytime you have data questions."
    return None


def _emit(config: RunnableConfig | None, phase: str, message: str,
          extra: dict | None = None) -> None:
    """Best-effort progress callback — a broken/absent UI callback must never
    sink the turn it's merely reporting on. Callers stash `on_event` in
    config["configurable"] (see chatbot/run.py::run_chat_turn).

    `extra`: the inference tier's own per-phase structured fields (e.g.
    "route"'s intent=, "sub_query"'s index=/total=/sub_query=) — forwarded
    verbatim so the chat UI gets the same structured data the inference SSE
    stream carried, not just the flattened phase/message text."""
    on_event = ((config or {}).get("configurable") or {}).get("on_event")
    if on_event is None:
        return
    try:
        on_event(phase, message, extra or {})
    except Exception:
        logger.exception("_emit: on_event callback raised for phase=%s", phase)


def _turn_delta(state: ChatState, assistant_reply: str) -> list:
    """[user, assistant] pair for THIS turn — appended (not replacing) to the
    checkpointed history via the Annotated[..., operator.add] reducer on
    ChatState.history (state.py). Only terminal nodes call this, once each,
    so a turn is recorded exactly once regardless of which path it took."""
    return [
        {"role": "user", "content": state["message"]},
        {"role": "assistant", "content": assistant_reply},
    ]


def classify_node(state: ChatState, config: RunnableConfig) -> dict:
    """Decide what kind of message this is. Defaults to 'answer' (route to the
    engine) on any failure — refuse-over-guess: never silently short-circuit
    a real data question as smalltalk just because the classifier is down.

    Latency fix: when a structured QueryFrame already exists (state["frame"],
    loaded by memory_read_node before this node runs), the SAME LLM call ALSO
    asks for the memory delta classification (chatbot/memory/classify.py's
    job — new_topic|refine|drill_down|drill_up|compare|ambiguous) via
    build_supervisor_system_prompt(frame)'s addendum — see that module's
    docstring. This merges what used to be two sequential SLM round-trips
    (classify_node's own call, then context_resolve_node's separate
    classify_delta() call) into one for every follow-up turn, which was the
    original design intent and had regressed to two calls in the first cut
    of chatbot/memory/. context_resolve_node only falls back to a second,
    standalone classify_delta() call if this one didn't produce a usable
    delta_type (e.g. this call failed/timed out)."""
    message = state["message"]
    history = state.get("history", [])
    frame = state.get("frame") or {}
    deterministic_smalltalk = _canned_smalltalk_reply(message) is not None
    delta_type = None          # None = "not computed this turn", see context_resolve_node

    if deterministic_smalltalk:
        # Deterministic fast path: a bare "hi"/"thanks"/"bye" needs no LLM call
        # at all — skips both this classify round-trip AND smalltalk_node's own
        # (each ~20s on this deployment's hardware). No "thinking" event either:
        # there's nothing to think about for an instant, deterministic reply.
        action = "smalltalk"
        logger.info("classify_node: deterministic smalltalk match, message=%r", message)
    elif _RUNTIME_CONTEXT_RE.match(message):
        # Same idea, for pure system-value questions ("what's the current
        # date") — no LLM classify call, no thinking event. _route_after_classify
        # also sends this straight to call_engine_node, bypassing
        # context_resolve_node's LLM call too, since the question is always
        # self-contained regardless of history.
        action = "runtime_context"
        logger.info("classify_node: deterministic runtime-context match, message=%r", message)
    elif frame.get("entity") and state.get("drill_stack") and _DRILL_UP_RE.match(message):
        # Deterministic fast path: "go back" navigation, only when there's an
        # actual drill level to pop (see _DRILL_UP_RE's docstring for why).
        # Sets delta_type directly too — this is the ONE fast path (besides
        # the LLM branch below) that needs to, since context_resolve_node
        # reads it to trigger pop_drill()/rebuild_frame_from_stack().
        action = "followup"
        delta_type = "drill_up"
        logger.info("classify_node: deterministic drill_up match, message=%r", message)
    else:
        _emit(config, "supervisor_classify", "Understanding your message...")
        raw = call_slm(
            build_supervisor_system_prompt(frame),   # built fresh each call so "today" is always
                                                      # current; includes the delta addendum only
                                                      # when `frame` has an entity (see its docstring)
            build_supervisor_user_prompt(message, history),
            model=CHATBOT_CLASSIFY_MODEL,
            purpose="classify",
        )
        action = "answer"
        if raw:
            match = _JSON_RE.search(raw)
            if match:
                try:
                    parsed = json.loads(match.group())
                    candidate = parsed.get("action")
                    if candidate in _VALID_ACTIONS:
                        action = candidate
                except Exception:
                    logger.warning("classify_node: could not parse LLM output: %r", raw)
            if frame.get("entity"):
                # Parse delta_type/slot_candidates from the SAME raw response —
                # shares classify_delta's exact vocabulary/confidence gates via
                # parse_delta_response, so a merged response is held to the
                # identical bar as the standalone fallback call.
                dt, _slots = parse_delta_response(raw, message)
                if dt in DELTA_TYPES:
                    delta_type = dt

    # Deliberately does NOT run _depends_on_history for every "smalltalk"
    # verdict: a message with no referential language at all (_REFERENTIAL_HINTS)
    # can never depend on history to mean something concrete, regardless of
    # what that history contains — greetings and self-introductions both fall
    # in this bucket (see chatbot/nodes.py's module docstring history / the
    # incidents this responds to), and asking a model "could this secretly
    # depend on context" for them was producing real false positives.
    if (action == "smalltalk" and history and frame.get("entity")
            and _REFERENTIAL_HINTS.search(message) and _depends_on_history(message, history)):
        # frame.get("entity") gate (2026-07 fix): a message can't meaningfully
        # be a "followup" when there's nothing real to follow up ON. Without
        # this, a bare "what about the other one" right after small talk (no
        # QueryFrame ever established — no real prior data question) still
        # got forced into "followup", sent to the engine as raw unresolved
        # text, and the engine's own retrieval "successfully" matched it
        # against a totally unrelated table — a confident-looking but
        # fabricated answer, the exact thing refuse-over-guess exists to
        # prevent. Requires the SAME grounding signal already used everywhere
        # else in this file (drill_up's gate, the vague-topical override
        # below) rather than a new keyword/phrase list — the fix generalizes
        # instead of patching one more phrasing.
        logger.warning(
            "classify_node: LLM said smalltalk but message depends on the prior "
            "conversation, overriding to 'followup': %r", message,
        )
        action = "followup"
    elif action == "smalltalk" and _DATA_QUESTION_HINTS.search(message):
        logger.warning(
            "classify_node: LLM said smalltalk but message looks data-related, "
            "overriding to 'answer': %r", message,
        )
        action = "answer"
    elif (action == "smalltalk" and frame.get("entity")
            and not (_GREETING_RE.match(message) or _THANKS_RE.match(message)
                      or _BYE_RE.match(message))):
        # _DATA_QUESTION_HINTS is schema-agnostic action-words only (count/how
        # many/show me/...) — it never catches a bare entity mention like
        # "tell me something about transaction" (no table/column names
        # hardcoded there by design). But a QueryFrame from a prior successful
        # query IS a grounded, deterministic signal that the session already
        # has an active analytical topic: a message that isn't a genuine
        # greeting/thanks/bye can't really be smalltalk right after that,
        # regardless of whether the LLM recognized the entity noun. Same
        # refuse-over-guess principle as the two overrides above.
        logger.warning(
            "classify_node: LLM said smalltalk but an active QueryFrame exists "
            "(entity=%r) and message isn't a genuine greeting/thanks/bye, "
            "overriding to 'followup': %r", frame.get("entity"), message,
        )
        action = "followup"

    if (action in ("followup", "answer") and not frame.get("entity")
            and _REFERENTIAL_HINTS.search(message) and not _DATA_QUESTION_HINTS.search(message)):
        # Universal backstop, independent of HOW `action` got here (the LLM's
        # own direct verdict, OR any override above): a message that is
        # PURELY referential ("other", "that", "it", ...) with no data-
        # question content of its own, and no QueryFrame to resolve it
        # against, has nothing real to be a followup TO. Left as "followup"/
        # "answer", context_resolve_node's render_frame_as_query() returns the
        # raw ambiguous text unchanged (frame is empty) and forwards it to the
        # engine as if self-contained — the engine's own retrieval then
        # "successfully" matches it against an UNRELATED table (observed:
        # "what about the other one" right after a bare "hi" returned a
        # different table's real row, including a real name/email — a
        # confident-looking fabrication, not a refusal). smalltalk_node has
        # zero DB access, so downgrading here is a hard guarantee this can't
        # happen, not just a lower-probability one.
        logger.warning(
            "classify_node: action=%r but message is purely referential with no "
            "QueryFrame to resolve against — downgrading to 'smalltalk' rather "
            "than forward ungrounded text to the engine: %r", action, message,
        )
        action = "smalltalk"

    logger.info("classify_node: action=%s message=%r", action, message)
    # Reset per-turn output fields — the checkpointer persists the FULL state
    # across turns (that's the point, for history/context), but sql/rows/
    # status/engine_result/clarification are this-turn-only outputs. Without
    # this reset they'd leak forward from a previous turn's answer into a
    # later turn (e.g. smalltalk) that never touches these fields itself.
    # `delta_type` is included in this reset for the same reason (also fixes
    # a latent bug: a "runtime_context" turn used to leave whatever
    # delta_type a PRIOR turn's context_resolve_node had set untouched in the
    # checkpoint, since that route bypasses context_resolve_node entirely —
    # explicitly setting it here every turn means it's never stale).
    # `resolved_query` too, for the exact same route: context_resolve_node
    # normally overwrites it every turn, but a "runtime_context" turn skips
    # that node, and call_engine_node's `resolved_query or message` fallback
    # then re-sent the PREVIOUS turn's resolved text to the engine (verified:
    # "what is the current date" mid-drill re-ran the prior drill query).
    return {
        "action": action,
        "resolved_query": None,
        "sql": None,
        "rows": None,
        "status": None,
        "engine_result": {},
        "needs_clarification": False,
        "clarification_question": None,
        "engine_unavailable": False,
        "delta_type": delta_type,
    }


def smalltalk_node(state: ChatState) -> dict:
    """Direct reply for greetings/thanks/chit-chat — engine bypassed entirely."""
    message = state["message"]
    reply = _canned_smalltalk_reply(message)
    if reply is None:
        # classify_node routed here via the LLM (not the deterministic fast
        # path above) — some smalltalk beyond a bare greeting/thanks/bye, so
        # this is the one case that still needs its own LLM call.
        reply = call_slm(
            build_smalltalk_system_prompt(),   # built fresh each call so "today" is always current
            message,
            max_tokens=60,
            model=CHATBOT_CLASSIFY_MODEL,
            purpose="smalltalk",
        )
    if not reply:
        reply = FALLBACK_REPLY
    return {
        "reply_text": reply,
        "needs_clarification": False,
        "resolved_query": "",
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def memory_read_node(state: ChatState) -> dict:
    """Loads the structured analytical memory (QueryFrame + DrillStack +
    episodic buffer) from Redis for this session — see
    chatbot/memory/store.py and docs/MEMORY_ARCHITECTURE.md §5/§7. Runs
    before classify/context_resolve so both can see it; a Redis miss/error
    degrades to empty (turn treated as if no prior analytical context
    exists — never blocks the turn).

    Also the deterministic "start over" fast path (audit fix H2 —
    MemoryStore.reset() existed but nothing ever called it): a whole-message
    match against _RESET_RE wipes the session's memory keys and returns a
    guaranteed-empty frame/stack/episodic immediately, skipping the reads
    entirely (there's nothing to read after a wipe)."""
    tenant = state.get("tenant") or "default"
    session_id = state.get("session_id") or ""
    message = state.get("message", "")

    if _RESET_RE.match(message):
        MemoryStore.reset(tenant, session_id)
        logger.info("memory_read_node: deterministic reset match, message=%r", message)
        return {"frame": {}, "drill_stack": [], "episodic": []}

    frame = MemoryStore.read_frame(tenant, session_id) or {}
    stack = MemoryStore.read_stack(tenant, session_id) or []
    episodic = MemoryStore.read_episodic(tenant, session_id) or []
    return {"frame": frame, "drill_stack": stack, "episodic": episodic}


def context_resolve_node(state: ChatState, config: RunnableConfig) -> dict:
    """Resolves a context-dependent message into a self-contained
    `resolved_query`. Used for both 'followup' and 'clarify_reply' actions,
    same as before — but now tries the DETERMINISTIC structured-memory merge
    first (chatbot/memory/frame.py::render_frame_as_query, one SLM call for
    classification only, never a free rewrite) whenever a usable frame
    exists, falling back to the original free-text LLM rewrite
    (FOLLOWUP_SYSTEM_PROMPT) ONLY when there is no frame yet at all — see
    docs/MEMORY_ARCHITECTURE.md §8/§29 (staged rollout: this fallback is the
    safety net for turns the new deterministic path can't cover yet, e.g. a
    clarify_reply whose only context is raw history, before any query has
    ever succeeded in this session).

    LATENCY: classify_node (which runs right before this node — see
    chatbot/graph.py) already tries to compute delta_type in its OWN single
    LLM call whenever a frame exists (chatbot/prompts/supervisor.py's merged
    prompt). If that succeeded, `state["delta_type"]` already holds a valid
    value and this node makes ZERO additional SLM calls — it just reuses it.
    A separate classify_delta() call only happens here as a FALLBACK, for
    the (expected to be rare) case where classify_node's call failed/timed
    out/returned something unparseable. Either way, once a delta_type is in
    hand, this node NEVER makes a SECOND call on top of it — "ambiguous" (a
    genuine judgment call OR a timeout — chatbot.llm.call_slm returns None
    uniformly for both) degrades to passing the raw message through as-is,
    and the engine's own existing refuse/clarify path (unchanged) handles
    genuine ambiguity exactly as it always has, just without an extra
    multi-second round-trip."""
    _emit(config, "supervisor_followup", "Resolving your follow-up context...")
    message = state["message"]
    history = state.get("history", [])
    frame = state.get("frame") or {}
    drill_stack = state.get("drill_stack") or []

    if not frame.get("entity"):
        # No prior frame at all — classify_node's merged prompt never asked
        # for a delta_type in this case (no addendum without a frame), so
        # this is genuinely the first SLM round-trip for this node, not a
        # second one. Unchanged free-text rewrite, exactly as before.
        rewritten = call_slm(
            FOLLOWUP_SYSTEM_PROMPT,
            build_followup_user_prompt(message, history),
            max_tokens=80,
            model=CHATBOT_CLASSIFY_MODEL,
            purpose="followup",
        )
        resolved = (rewritten or message).strip().strip('"')
        logger.info("context_resolve_node: no frame yet, fallback rewrite %r -> %r",
                    message, resolved)
        return {"resolved_query": resolved, "delta_type": "new_topic"}

    delta_type = state.get("delta_type")
    if delta_type in DELTA_TYPES:
        # classify_node's merged call already produced this — ZERO extra SLM
        # call here, which is the whole point of the merge.
        logger.info("context_resolve_node: reusing delta_type=%s from classify_node's "
                    "merged call (no extra SLM round-trip)", delta_type)
    else:
        # classify_node's call didn't yield a usable delta_type this turn
        # (failed/timed out/unparseable, or this session never went through
        # a frame-aware classify at all) — fall back to one standalone call.
        episodic = state.get("episodic") or []
        delta_type, _slot_candidates = classify_delta(frame, message, episodic)
        # _slot_candidates itself is intentionally not threaded into the merge
        # (render_frame_as_query only ever uses `frame` + the verbatim
        # `message`, never a partially-extracted slot value — nothing gets
        # added that isn't either an old proven fact or a substring the user
        # actually typed). classify_delta()/parse_delta_response already
        # applied the H3 confidence gate before delta_type reaches here.

    if delta_type == "drill_up" and drill_stack:
        drill_stack = memory_frame.pop_drill(drill_stack)
        frame = memory_frame.rebuild_frame_from_stack(frame, drill_stack)

    # "new_topic"/"refine"/"drill_down"/"drill_up"/"compare" all merge
    # deterministically; "ambiguous" (judgment OR timeout) passes the message
    # through untouched — no second SLM call, see docstring above.
    resolved = (memory_frame.render_frame_as_query(frame, message, delta_type)
                if delta_type != "ambiguous" else message)

    logger.info("context_resolve_node: delta_type=%s frame-merge %r -> %r",
                delta_type, message, resolved)
    return {"resolved_query": resolved, "delta_type": delta_type,
            "frame": frame, "drill_stack": drill_stack}


def _extract_engine_result(payload: dict) -> tuple[dict, str]:
    """Walk MultiResult's wire shape ({"result": {"items": [{"result": {...}}]}}
    — same shape apps/query/inference_client.py's run_hybrid_query/
    stream_hybrid_query hand back, per inference/routes/hybrid.py's _serialize()).

    NOTE: item0 itself carries a SubResult-level status ("ok"/"refused"/"error",
    veda_core/query/multi_result.py) — that is NOT the status we want here. The
    pipeline-level status ("answered"/"refuse"/"clarify"/"no_table"/...,
    veda_core/veda/pipeline.py::_done) is one level deeper, at
    item0["result"]["status"]. Do not "simplify" this by reading item0["status"].
    """
    result = (payload or {}).get("result") or {}
    items = result.get("items") or []
    item0 = items[0] if items and isinstance(items[0], dict) else {}
    res0 = item0.get("result") or {}
    if not isinstance(res0, dict):
        res0 = {}
    # Contract normalization: the NoSQL head returns a connectors.base.QueryResult
    # (serialized via asdict), whose tabular field is `columns` — every other
    # pipeline (SQL/Tier-2/federated/hybrid) and every api-tier consumer
    # (apps/chat/services.py's viz + table builders, analytics, harvest_frame)
    # speaks `cols`. Alias it here, the ONE place res0 is assembled, so a NoSQL
    # answer charts/tables/analyzes exactly like the others instead of silently
    # having no chart/table (its rows existed under the wrong key). Only fills
    # `cols` when absent — never clobbers a pipeline that already set it.
    if "cols" not in res0 and res0.get("columns"):
        res0["cols"] = res0["columns"]
    return res0, res0.get("status", "error")


def call_engine_node(state: ChatState, config: RunnableConfig) -> dict:
    """Calls the inference tier over HTTP via apps.query.inference_client —
    same client/contract every other apps/ caller uses. The api tier never
    imports veda_core directly (see InferenceClient's own docstring); chatbot/
    now runs inside the api container's process (apps/chat/services.py), so it
    is subject to that same boundary.

    Forwards the inference tier's own stage-progress events (classify/
    decompose/route/answer/...) live via _emit as they arrive off the SSE
    stream — the loop below iterates the generator synchronously, so this
    naturally streams to the caller rather than batching.

    A transport/infra failure (InferenceUnavailable, or a mid-stream "error"
    event) is reported as status="unavailable"/engine_unavailable=True — kept
    DISTINCT from a reachable engine's own legitimate refusal, so callers
    (apps/chat/services.py) can surface a genuine outage as an error instead
    of a misleading "please clarify" chat reply.
    """
    query = state.get("resolved_query") or state["message"]
    client = InferenceClient()
    res0: dict = {}
    status = "error"

    try:
        for kind, data in client.stream_hybrid_query(
            query,
            source_id=state.get("source_id"),
            source_ids=state.get("source_ids"),
            tenant=state.get("tenant"),
            request_id=state.get("request_id"),
        ):
            if kind == "progress":
                _extra = {k: v for k, v in data.items() if k not in ("phase", "message")}
                _emit(config, data.get("phase", "progress"), data.get("message", ""), _extra)
            elif kind == "error":
                logger.warning("call_engine_node: inference stream error for query=%r: %s", query, data)
                return {"engine_result": {}, "status": "unavailable", "engine_unavailable": True}
            elif kind == "result":
                res0, status = _extract_engine_result(data)
    except InferenceUnavailable as exc:
        logger.warning("call_engine_node: inference unavailable for query=%r: %s", query, exc)
        return {"engine_result": {}, "status": "unavailable", "engine_unavailable": True}
    except Exception:
        logger.exception("call_engine_node: unexpected failure for query=%r", query)
        return {"engine_result": {}, "status": "error", "engine_unavailable": False}

    logger.info("call_engine_node: status=%s query=%r", status, query)
    return {"engine_result": res0, "status": status, "engine_unavailable": False}


def _templated_gist(engine_result: dict) -> str:
    """One-line, deterministic (NOT LLM) summary of an answered turn, for the
    episodic buffer only (chatbot/memory/store.py — capped, TTL'd, never the
    full markdown/table reply). Mirrors the "memory is evidence, not prose"
    principle: this line is stored purely to help classify_delta recognize
    "it"/"that" references, never re-parsed back into the QueryFrame."""
    answer = engine_result.get("answer")
    if answer:
        return f"answered: {answer}"[:200]
    rows = engine_result.get("rows")
    if isinstance(rows, list):
        return f"answered: {len(rows)} row(s)"
    return "answered"


def memory_write_node(state: ChatState) -> dict:
    """Writes the structured analytical memory AFTER a successful engine
    execution — evidence only, never on refuse/error/clarify/unavailable
    (hard-enforced below; see docs/MEMORY_ARCHITECTURE.md §6/§12 barrier 1).
    Every field written traces back to engine_result's own already-validated
    output (chatbot/memory/frame.py::harvest_frame) — nothing here is
    invented by this node or by any LLM."""
    if state.get("status") != "answered":
        return {}

    tenant = state.get("tenant") or "default"
    session_id = state.get("session_id") or ""
    engine_result = state.get("engine_result") or {}

    harvested = memory_frame.harvest_frame(engine_result)
    if not harvested:
        # business_explain failed server-side (already logged there) or the
        # result had no explain block — skip the write, the user's answer is
        # unaffected, memory just doesn't advance this turn.
        return {}

    prev_frame = state.get("frame") or {}
    prev_stack = state.get("drill_stack") or []
    delta_type = state.get("delta_type") or "new_topic"

    new_frame = memory_frame.merge_frame_post_execution(
        prev_frame, harvested, delta_type, tenant, session_id)

    reset = delta_type == "new_topic" or memory_frame.is_topic_switch(prev_frame, harvested)
    if reset:
        new_stack: list = []
    elif delta_type == "drill_down":
        new_stack = memory_frame.push_drill(prev_stack, harvested)
    else:
        new_stack = prev_stack

    MemoryStore.write_frame(tenant, session_id, new_frame,
                            expected_version=prev_frame.get("version") if prev_frame else None)
    MemoryStore.write_stack(tenant, session_id, new_stack)
    MemoryStore.push_episodic_turn(tenant, session_id, state.get("message", ""),
                                   _templated_gist(engine_result))

    return {"frame": new_frame, "drill_stack": new_stack}


def ask_clarification_node(state: ChatState) -> dict:
    """Turn a refusal into a conversational clarifying question — reuses the
    engine's own deterministic explanation (already computed server-side by
    veda_core/veda/pipeline.py's _feedback()/explain_failure() for every
    non-"answered" status, and embedded at res0["feedback"]["text"]), never
    invents reasons of its own (refuse-over-guess, same as the rest of the
    codebase).

    status == "unavailable" (a transport/infra failure, not a real engine
    refusal — see call_engine_node) gets its own honest reply instead of the
    generic clarification text, and is not recorded into checkpointed history
    since a transient outage isn't real conversation content.
    """
    res0 = state.get("engine_result", {})
    status = state.get("status", "refuse")

    feedback = res0.get("feedback")
    if feedback:
        question = feedback.get("text") or "Could you clarify what you're asking about?"
    else:
        # Rare fallback (e.g. FEEDBACK_ENABLED=False in the engine, so it never
        # built one). Deliberately generic, NOT a veda_core import: chatbot/
        # runs in the api container (working_dir=/app), while veda_core's own
        # internals (e.g. veda/feedback.py -> veda/runtime.py's bare
        # `from config import ...`) only resolve correctly when veda_core/
        # itself is the process root (true for the inference container's
        # working_dir=/app/veda_core, not this one) — importing
        # veda_core.veda.feedback here always raised ImportError in this
        # container, silently (caught below) but 100% of the time, not
        # "rarely." Same api/veda_core boundary chatbot/llm.py and
        # apps/query/inference_client.py already document; this path was
        # violating it. If a genuinely richer fallback message is wanted
        # later, it belongs behind an HTTP call to the inference tier (which
        # already has veda_core in scope), not a direct import here.
        question = "Could you clarify what you're asking about?"

    unavailable = status == "unavailable"
    update = {
        "reply_text": question,
        "needs_clarification": not unavailable,
        "clarification_question": None if unavailable else question,
        "engine_unavailable": unavailable,
    }
    if not unavailable:
        update["history"] = _turn_delta(state, question)
    return update


def format_reply_node(state: ChatState) -> dict:
    """Final assembly for the 'answered' path."""
    res0 = state.get("engine_result", {})
    answer = res0.get("answer") or "Here's what I found."
    return {
        "reply_text": answer,
        "needs_clarification": False,
        "sql": res0.get("sql"),
        "rows": res0.get("rows"),
        "engine_unavailable": False,
        "history": _turn_delta(state, answer),
    }
