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
import os
import re
import time
import unicodedata

from langchain_core.runnables import RunnableConfig

from apps.query.data_vocabulary import mentions_the_data
# The central function-word list (pleasantries, determiners, request verbs). Read, never
# extended here: the return-to-topic matcher below uses it to set aside the words of a
# message that name nothing, exactly as the vocabulary check does.
from apps.query.data_vocabulary import _STOPWORDS as _FUNCTION_WORDS
from apps.query.inference_client import InferenceClient, InferenceUnavailable

from .llm import CHATBOT_CLASSIFY_MODEL, call_slm
from .memory import frame as memory_frame
from .memory import reference as memory_reference
from . import telemetry as turn_telemetry
from .memory import topics as memory_topics
from .memory.classify import DELTA_TYPES, classify_delta, parse_delta_response
from .memory.context import ConversationContext
from .memory.store import MemoryStore
from .prompts import (
    CAPABILITY_REPLY,
    FALLBACK_REPLY,
    REPHRASE_REPLY,
    SMALLTALK_FALLBACK_REPLY,
    IDENTITY_REPLY,
    FOLLOWUP_SYSTEM_PROMPT,
    STANDALONE_CHECK_SYSTEM,
    CARRYOVER_CHECK_SYSTEM,
    build_carryover_check_user_prompt,
    build_followup_user_prompt,
    build_smalltalk_system_prompt,
    build_standalone_check_user_prompt,
    build_supervisor_system_prompt,
    build_supervisor_user_prompt,
)
from .state import ChatState

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"smalltalk", "followup", "clarify_reply", "answer"}
# Every call that DECIDES something about the turn (action, delta, carry-over) samples at
# 0, like the engine (veda_core SLM_TEMPERATURE). At call_slm's 0.1 default the same
# follow-up was read two ways on two runs — measured 2026-09-25, chain C1 failed on one run
# ("gated" read as a furnishing value) and passed on the next — so a rehearsed demo could
# still take a different path. Only the smalltalk REPLY keeps the default: its words may
# vary, its decision may not.
_DECISION_TEMPERATURE = 0.0
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
#
# 2026-09-24: the third-person pronouns (`they|them|their|theirs|its`) and the
# pronominal `<det> one(s)` shapes were ADDED, after measuring that the single
# commonest back-referential follow-up in a real 25-turn drill session — "What are
# their prices?", "How much are they?", "Which one is the cheapest?" — contained
# not one word on this list and so could not reach ANY of the second-opinion gates
# that read it. `it` was here; `its`, `they` and `their` were not. This is the one
# central list for anaphoric language in this package, so the words were added here
# rather than in a new list next to the new gate.
_REFERENTIAL_HINTS = re.compile(
    r"\b(that|this|it|its|they|them|their|theirs|those|these|same|again|more|"
    r"other|another|previous|above|below|instead|also|too|earlier|before|"
    # One optional word between the determiner and "one(s)": "the second one", "the 3rd
    # one", "the cheapest one". Without it an ordinal or superlative pointer at the previous
    # result contained no word on this list, so it could reach neither the back-reference
    # gate nor result-reference resolution (chatbot/memory/reference.py) and ran as a fresh
    # question. Every use of this pattern only OPENS a stricter check or RESTRICTS a
    # downgrade, so widening it cannot by itself make a turn carry context.
    r"(?:the|last|first|next|which|each|any|that|this|other)\s+(?:[a-z0-9]+\s+)?ones?)\b",
    re.IGNORECASE,
)

# Stricter sibling of _REFERENTIAL_HINTS, for the "purely referential, nothing
# to resolve it against" backstop below ONLY — anchored so the referential word
# must be the last substantive thing in the message (a dangling pronoun: "what
# about this?", "show me the other one"), not a determiner heading its own
# self-contained noun phrase ("this maintenance policy for asset 21" names its
# own referent; there is nothing to resolve). _REFERENTIAL_HINTS itself stays
# unanchored where it's used elsewhere (line ~293) — a false positive there
# only costs one extra cheap LLM gate call, never a dropped query, so it can
# afford to stay broad. This one gates a hard downgrade to smalltalk, so it
# must not fire on a question that is already complete on its own.
_BARE_REFERENTIAL_RE = re.compile(
    r"\b(that|this|it|those|these|same|again|more|other|another|previous|"
    r"above|below|instead|also|too|earlier|before|last one|the one)\b"
    r"(\s+one)?\s*[.,!?]*\s*$",
    re.IGNORECASE,
)

def _is_bare_referential(message: str) -> bool:
    """Is this message referential with nothing of its own to stand on?

    Two shapes, both requiring the caller to have already established that there is NO
    frame — which is what makes a downgrade safe here: with nothing to resolve against,
    a referential message cannot be a follow-up to anything.

    1. the referential word is the last substantive thing ("what about that", "the
       other one") — _BARE_REFERENTIAL_RE, anchored so it cannot fire on a determiner
       heading a self-contained noun phrase ("this maintenance policy for asset 21").
    2. the whole message is a handful of words built around a referential one ("more
       details", "more info"). The anchor in (1) misses these because a contentless
       noun trails the referential word, and enumerating those nouns would be a word
       list that ages badly. Brevity is the structural signal instead: a message this
       short, containing referential language and naming no data, has no self-contained
       question in it whatever the trailing word happens to be.

    Added 2026-09-22 after the supervisor's HARD RULE was rewritten. "more details" used
    to be classified smalltalk by the model and so never reached this backstop at all —
    the category was resting on the model's verdict, not on this guard. Once the rule
    correctly stopped calling information requests smalltalk, "more details" came
    through as a followup with no frame, which is exactly the ungrounded-text-to-engine
    case this backstop exists to prevent.
    """
    if _DATA_QUESTION_HINTS.search(message):
        return False
    if _BARE_REFERENTIAL_RE.search(message):
        return True
    words = [w for w in re.split(r"[^A-Za-z0-9]+", message) if w]
    return len(words) <= 3 and bool(_REFERENTIAL_HINTS.search(message))


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
                       model=CHATBOT_CLASSIFY_MODEL, purpose="standalone_check",
                       temperature=_DECISION_TEMPERATURE)
    return bool(verdict) and "dependent" in verdict.strip().lower()


def _frame_free_action(message: str, history: list) -> Optional[str]:
    """The supervisor's action for this message with NO frame in the prompt — the same
    model, prompt and history otherwise. Second opinion for a frame-bearing "smalltalk"
    verdict only (see its call site for the measurement). None on any failure, which
    callers treat as "keep the original verdict"."""
    raw = call_slm(build_supervisor_system_prompt({}),
                   build_supervisor_user_prompt(message, history),
                   model=CHATBOT_CLASSIFY_MODEL, purpose="classify_frame_free",
                   temperature=_DECISION_TEMPERATURE)
    match = _JSON_RE.search(raw or "")
    if not match:
        return None
    try:
        candidate = json.loads(match.group()).get("action")
    except Exception:
        return None
    return candidate if candidate in _VALID_ACTIONS else None


def _carries_over_subject(message: str, history: list, frame: dict) -> bool:
    """Does this message stand in for the frame's records with a pronoun/demonstrative
    instead of naming a subject of its own?

    Second opinion on the SUPERVISOR's verdict, used only where that verdict is
    `answer` + `new_topic` — the one combination that both drops the conversation's
    context and wipes the drill stack. See chatbot/prompts/carryover_check.py for the
    measurement that motivated a dedicated prompt (and for why standalone_check's was
    measured and rejected for this job: 2 of 8 self-contained CONTROL questions came
    back 'dependent' there, and a false positive here pollutes a brand-new question
    with the previous subject).

    Fails closed to False — "trust the supervisor", i.e. exactly today's behaviour —
    on an empty/failed/unrecognised reply, so an SLM outage can never make this worse
    than not having the gate. Callers apply the cheap `_REFERENTIAL_HINTS` pre-filter
    first, so no message without anaphoric language ever pays for this call."""
    verdict = call_slm(CARRYOVER_CHECK_SYSTEM,
                       build_carryover_check_user_prompt(frame, message, history),
                       max_tokens=5, model=CHATBOT_CLASSIFY_MODEL,
                       purpose="carryover_check", temperature=_DECISION_TEMPERATURE)
    return bool(verdict) and "carryover" in verdict.strip().lower()

# Deterministic fast path for the overwhelming majority of smalltalk: pure
# greetings/thanks/farewells with nothing else in the message. Tight, anchored
# patterns (whole-message match, not substring) so they can never misfire on a
# real question that merely starts with "hi" or ends with "thanks" — and
# _DATA_QUESTION_HINTS is still checked as a second guard before trusting this.
# Skips the classify LLM call entirely (classify_node) and lets smalltalk_node
# skip its own LLM call too — on this deployment's hardware a single such call
# alone can take ~20s, so a bare "hi" was paying 20-40+ seconds of pure LLM
# round-trip time for something that should be instant.
# WHY THESE ARE WIDER THAN THEY LOOK LIKE THEY NEED TO BE.
# The first cut required a greeting word and then nothing but punctuation, so it
# matched "hi" but not "hi there!" — and the natural forms people actually type
# ("hi there", "hey there, how are you?", "hello there") fell through to the full
# engine. Measured on this deployment: "hi there!" cost 16-23 s and came back with
# "It seems like you have multiple documents related to an employee handbook",
# because routing sent a greeting to document retrieval (top similarity 0.50) and
# the summariser dutifully described whatever chunks came back.
#
# Widening is safe because _canned_smalltalk_reply checks _DATA_QUESTION_HINTS
# FIRST: anything with a data verb in it ("hi, how many assets are there") never
# reaches these patterns. Guard tests cover 22 greeting forms and 11 near-misses
# that must still go to the engine.
_GREET_WORD = (r"(?:hi+|hello+|hey+|hiya|yo|greetings|"
               r"good\s*(?:morning|afternoon|evening|day))")
_HOW_ARE_YOU = (r"(?:how\s*(?:are\s*(?:you|u|ya)|'?s\s*it\s*going|'?re\s*you)"
                r"(?:\s*doing)?)")
_GREETING_RE = re.compile(
    rf"^\s*(?:{_GREET_WORD}(?:\s+there)?(?:\s*[,!.\-]+\s*|\s+)?(?:{_HOW_ARE_YOU})?"
    rf"|{_HOW_ARE_YOU})\s*[.,!?]*\s*$", re.IGNORECASE)
_THANKS_RE = re.compile(
    r"^\s*(thanks?( you)?( very much| so much| a lot| a ton)?|thx|ty|appreciate it|"
    r"much appreciated|cheers|perfect|great|awesome|nice)\s*[.,!?]*\s*$", re.IGNORECASE)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see\s*(you|ya)( later| soon)?|take care|good\s*(night|bye))"
    r"\s*[.,!?]*\s*$", re.IGNORECASE)

# "Who are you" / "what can you do" are questions about the PRODUCT, not about the
# user's data — answerable deterministically, with no model call and no engine hop.
# They matched none of the patterns above, so classify_node had to ask the
# classifier; and because classify_node defaults to "answer" whenever that call
# fails, an unreachable SLM turned every one of them into a full SQL round-trip that
# then refused. Both are WHOLE-MESSAGE anchored with no trailing wildcard — a looser
# first cut captured "who are you billing this month" and "can you help me find the
# cheapest listings", which are real data questions and must reach the engine — and
# both sit behind the same _DATA_QUESTION_HINTS guard as the greetings.
_IDENTITY_RE = re.compile(
    r"^\s*(?:so\s+|and\s+)?(?:"
    r"who\s+(?:are|r)\s+(?:you|u)"
    r"|what\s+(?:are|r)\s+(?:you|u)"
    r"|what(?:'s|\s+is)\s+(?:veda|your\s+name)"
    r"|who(?:'s|\s+is)\s+(?:this|veda)"
    r"|(?:please\s+)?introduce\s+yourself"
    r"|tell\s+me\s+about\s+yourself"
    r")\s*[.,!?]*\s*$", re.IGNORECASE)
_CAPABILITY_RE = re.compile(
    r"^\s*(?:so\s+|and\s+)?(?:"
    r"what\s+can\s+(?:you\s+do|i\s+ask(?:\s+you)?)(?:\s+for\s+me)?"
    r"|what\s+do\s+you\s+do"
    r"|what\s+are\s+you\s+(?:good\s+at|able\s+to\s+do|capable\s+of)"
    r"|what\s+(?:kind|sort|type)s?\s+of\s+(?:questions?|things?)(?:\s+can\s+i\s+ask)?"
    r"|how\s+(?:do|can)\s+(?:i|we)\s+use\s+(?:you|this)"
    r"|how\s+do\s+you\s+work"
    r"|can\s+you\s+help(?:\s+me)?"
    r"|help"
    r")\s*[.,!?]*\s*$", re.IGNORECASE)

# Skip the engine for a message that names NOTHING in the scoped data. Default OFF:
# this is the only change in this package that can withhold a message from the engine,
# so it ships dark and is turned on deliberately. With it off, every routing decision
# below is byte-identical to before it existed.
#
# WHY: classify_node defaults to "answer" whenever it is unsure (refuse-over-guess,
# deliberately — never silence a real question). That default is correct but expensive:
# the engine then searches 528 tables for something that is not there. Measured
# 2026-09-17: 32.8s, then "Could you clarify what you're asking about?". A fast honest
# answer beats a slow one.
#
# The check is deliberately weak — ONE matching content word is enough, sampled data
# VALUES count, and both number forms count (see apps/query/data_vocabulary.py). A dry
# run over 24 real questions held none of them back.
_GROUNDING_GATE_ENABLED = os.environ.get("CHATBOT_GROUNDING_GATE_ENABLED", "0") == "1"
# Ask the model for the WORDS of a smalltalk reply, on top of the classify call that
# already decided it is smalltalk. Off: see smalltalk_node.
_SMALLTALK_LLM_REPLY = os.environ.get("CHATBOT_SMALLTALK_LLM_REPLY", "0") == "1"


def _mentions_the_data(message: str, vocabulary) -> bool:
    """True (send to the engine) unless the message names nothing in the data. Fails
    OPEN on any error, and on an empty/absent vocabulary — never the other way."""
    if not vocabulary:
        return True
    try:
        return mentions_the_data(message, vocabulary)
    except Exception:
        logger.exception("_mentions_the_data: check failed — sending to the engine")
        return True


# Questions about the CONVERSATION ITSELF rather than about the data — "what was my
# last query", "what SQL did you run", "which table did you use", "how many rows did
# that return". Every one of these is already answered by the QueryFrame the memory
# layer stores after each successful turn (chatbot/memory/frame.py), so they need no
# engine call, no SQL and no model call. Routed normally they went to the engine,
# which tried to find a TABLE for "what sql did you run" — roughly 30s, then a
# refusal, or worse an answer pulled from some unrelated table.
#
# In an analytics product this class matters more than ordinary chit-chat: a user who
# asks "which table did that come from" is checking whether to trust a number. The
# whole reasoning trail is computed every turn (business_explain, zero LLM) and until
# now the user could not see any of it.
#
# The patterns are whole-message anchored and keyed on the OBJECT noun (query /
# question / sql / table you used), which is what separates them from real data
# questions that read almost identically: "what was my last QUERY" is recall,
# "what was my last PAYMENT" is a question for the engine.
_RECALL_PATTERNS = (
    ("query", re.compile(
        r"^\s*(?:so\s+|and\s+)?(?:what|which)\s+(?:was|is)?\s*(?:my|the)?\s*"
        r"(?:last|previous|prior|first)\s+(?:query|question|search)"
        r"(?:\s+again)?\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("query", re.compile(
        r"^\s*(?:so\s+)?what\s+(?:did\s+i|have\s+i)\s+ask(?:ed)?"
        r"(?:\s+(?:before|earlier|last|previously))?\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("query", re.compile(
        r"^\s*(?:what|which)\s+(?:do\s+|did\s+)?we\s+(?:have|had|got)\s+in\s+"
        r"(?:my|the|our)\s+(?:last|previous)\s+(?:query|question)"
        r"\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("sql", re.compile(
        r"^\s*(?:(?:show|give)\s+(?:me\s+)?(?:the\s+)?sql"
        r"|(?:what|which)\s+sql\s+(?:did\s+you\s+)?(?:run|use|execute|write)?"
        # "what did you just run" / "what did you run" — a near-miss of the pattern above
        # that cost ~98s and came back "Could you clarify if 'did' is a column name or a
        # value to filter on?" (live run, 2026-09-17). The whitelist is a closed list over
        # an open phrasing space, so every near-miss is a slow, confusing answer.
        r"|what\s+(?:query\s+)?did\s+you\s+(?:just\s+)?(?:run|execute)"
        r"|what\s+(?:exactly\s+)?(?:did|do)\s+you\s+run)\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("table", re.compile(
        r"^\s*(?:what|which)\s+tables?\s+(?:did\s+you\s+|do\s+you\s+|was\s+)?"
        r"(?:use|used|pick|picked|choose|chose|query|queried|from)"
        r"(?:\s+(?:that|this|it|for\s+that))?\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("filters", re.compile(
        r"^\s*(?:what|which)\s+filters?\s+(?:did\s+you\s+apply|were\s+applied"
        r"|are\s+applied|did\s+you\s+use)?\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("rows", re.compile(
        r"^\s*how\s+many\s+(?:rows|records|results)"
        r"(?:\s+(?:did\s+(?:that|it|this)\s+(?:return|come\s+back)"
        r"|were\s+(?:there|returned)|(?:did\s+)?(?:that|it|this)\s+return))?"
        r"\s*[.,!?]*\s*$", re.IGNORECASE)),
    # A question about the conversation's TOPICS ("what have we looked at?") — answered
    # from the topic index (chatbot/memory/topics.py), never the engine.
    ("topics", re.compile(
        r"^\s*(?:so\s+)?(?:what|which\s+(?:topics|things|tables))\s+(?:have|did)\s+we\s+"
        r"(?:look(?:ed)?\s+at|talk(?:ed)?\s+about|discuss(?:ed)?|cover(?:ed)?|"
        r"been\s+looking\s+at)(?:\s+(?:so\s+far|earlier|before|today))?"
        r"\s*[.,!?]*\s*$", re.IGNORECASE)),
    ("trail", re.compile(
        r"^\s*(?:(?:so\s+)?what\s+did\s+you\s+do"
        r"|how\s+did\s+you\s+(?:get|work\s+out|arrive\s+at)\s+(?:that|this|it)"
        r"|where\s+(?:did\s+)?(?:that|this|it)\s+come\s+from"
        r"|why\s+(?:did\s+you\s+(?:choose|pick|use)\s+)?(?:that|this)\s+table)"
        r"\s*[.,!?]*\s*$", re.IGNORECASE)),
)


# A clarification answer is a VALUE, not a request. Recognised by a WHITELIST of
# shapes, never by excluding known-bad words.
#
# The first cut excluded messages carrying a word from _DATA_QUESTION_HINTS and capped
# the length at 8 words. That is a blacklist over an open set, and an independent run
# walked straight through it: "top 5 cities by assets", "sale listings in mumbai",
# "revenue by month", "who owns asset 21", "kitne assets hain", "nevermind", "stop",
# "???" were all swallowed and concatenated onto the stored request, producing confident
# answers to questions nobody asked. The mirror failure was just as bad — real answers
# ("total revenue", "by status", "the count") were REJECTED because they contain an
# ordinary analytic word, and the stored request was thrown away with them.
#
# So the deterministic path now claims only the shapes it can be sure of, and everything
# else is left to the classifier, which is asked the semantic question directly ("is this
# an answer to the question you just asked?") — the one judgement a model is better at
# than a regex. Python still performs the merge.
_CLARIFICATION_VALUE_RE = re.compile(
    r"^\s*(?:(?:for|in|from|during|on|at|by|of)\s+)?"      # optional leading preposition
    r"(?:the\s+)?"
    # ONE or TWO plain words. Three or more is where genuinely ambiguous shapes start
    # — "revenue by month" and "sale listings in mumbai" read as values AND as new
    # questions, and which one they are depends on what was asked. Those are handed to
    # the classifier rather than claimed here; the deterministic path keeps only what it
    # can be certain of, which is also the common case ("2024", "by city", "Mumbai").
    r"[\w&/'’.-]+(?:\s+[\w&/'’.-]+)?"
    r"\s*[.,!?]*\s*$", re.IGNORECASE)

# Words that make a message a REQUEST rather than a value, however short it is.
_REQUEST_WORDS = frozenset("""
show give list find get make take put see look want need help please display draw plot
chart graph render export download save delete drop remove add create update run execute
who what when where why how which whose compare explain tell describe stop cancel
nevermind forget skip undo repeat again
""".split())


def _is_clarification_answer(message: str, recall_kind, presentation_kind,
                             is_smalltalk: bool, is_shape_change: bool = False) -> bool:
    """Is this message unmistakably a VALUE answering the question just asked?

    Deliberately narrow: it returns True only for the shapes it is certain about, and
    False for everything else — including real answers it cannot recognise. False does
    NOT mean "throw the pending request away"; the caller leaves those to the classifier
    (see classify_node), which can judge them semantically.

    `is_shape_change` is an EXPLICIT re-shaping of the previous question ("by city
    instead", "make it top 5" — chatbot/memory/frame.py::detect_shape_delta with
    explicit_only). Measured 2026-09-18 on the real engine: "by city instead" and a bare
    "top 5" both matched the value whitelist below, so with a clarification pending they
    were glued onto the unanswered request ("show the top 20 assets by carpet area for
    make it top 5") and the engine answered something unrelated with full confidence.
    The engine clarifies often, so a pending slot is a COMMON state, not an edge case.
    Only explicit re-shapings are excluded here: a clarifying question asks for a bare
    value, so a bare "top 5" must still be allowed to answer one.
    """
    if is_smalltalk or recall_kind or presentation_kind or is_shape_change:
        return False
    text = (message or "").strip()
    if not text:
        return False
    if _RESET_RE.match(text) or _DRILL_UP_RE.match(text) or _RUNTIME_CONTEXT_RE.match(text):
        return False
    words = [w.strip(".,!?").lower() for w in text.split()]
    if any(w in _REQUEST_WORDS for w in words):
        return False
    if not any(c.isalnum() for c in text):
        return False                       # "???", "...", punctuation only
    return bool(_CLARIFICATION_VALUE_RE.match(text))


def _recall_kind(message: str) -> str | None:
    """Which remembered fact a conversation-about-itself question is asking for —
    "query" | "sql" | "table" | "filters" | "rows" | "trail" — or None."""
    for kind, pattern in _RECALL_PATTERNS:
        if pattern.match(message):
            return kind
    return None


# A follow-up that asks to RE-PRESENT the answer already on screen — "as a pie
# chart", "show that as a table", "export it to csv". These carry no new data
# question at all: the rows are already in hand from the previous turn. Sent through
# the normal followup path they re-ran the whole SQL pipeline, which costs a full
# engine round-trip AND can come back with a DIFFERENT result set than the one the
# user is looking at (the data can change between turns; so can a non-deterministic
# plan) — so "chart that" could redraw something other than "that".
#
# Whole-message anchored behind the same _DATA_QUESTION_HINTS guard as the greetings:
# "show me sales as a pie chart" carries its own data question and must reach the
# engine, while a bare "as a pie chart" must not.
_PRESENTATION_RE = re.compile(
    r"^\s*(?:(?:and\s+|now\s+|ok(?:ay)?[,\s]+|please\s+|just\s+)*)"
    r"(?:can\s+you\s+|could\s+you\s+)?"
    r"(?:show|display|draw|plot|graph|chart|render|make|turn|put|give)?\s*"
    r"(?:me\s+|it\s+|that\s+|this\s+|them\s+|the\s+result\s+|the\s+data\s+)?"
    r"(?:in|as|into|to)?\s*(?:a|an|the)?\s*"
    r"(?P<kind>pie|bar|line|column|donut|doughnut|table|chart|graph|csv|excel|spreadsheet)"
    r"\s*(?:chart|graph|plot|format|view|instead)?\s*(?:please|thanks|pls)?"
    r"\s*[.,!?]*\s*$", re.IGNORECASE)

# "export"/"download" phrasings, where the noun may be absent entirely.
_EXPORT_RE = re.compile(
    r"^\s*(?:(?:and\s+|now\s+|please\s+)*)(?:can\s+you\s+)?"
    # The object may only be a PRONOUN. Allowing "the <noun>" made "export the invoices"
    # and "download the report" re-render the previous rows instead of reaching the
    # engine — those name data, and naming data makes it a question, not a re-render.
    r"(?:export|download|save)\s*(?:it|that|this|them|the\s+(?:result|results|table|data|rows))?\s*"
    r"(?:in|as|to)?\s*(?:a|an)?\s*(?P<kind>csv|excel|spreadsheet|file)?"
    r"\s*(?:please|thanks|pls)?\s*[.,!?]*\s*$", re.IGNORECASE)

# A bare charting verb with no noun at all ("plot it", "visualize that").
_PLOT_RE = re.compile(
    r"^\s*(?:(?:and\s+|now\s+|please\s+)*)(?:can\s+you\s+)?"
    # An OBJECT is required. A bare "chart" or "graph" is a noun as often as a verb, and
    # as an answer to "which one?" it would silently redraw instead of answering.
    r"(?P<kind>plot|graph|chart|visuali[sz]e)\s+(?:it|that|this|them)"
    r"\s*(?:please|thanks|pls)?\s*[.,!?]*\s*$", re.IGNORECASE)

# Verbs, not nouns — a one-word "export" is an instruction and nothing else, unlike a
# one-word "table" or "chart".
_UNAMBIGUOUS_RENDER_VERBS = frozenset({"export", "download"})
# The renderings represent_node can draw — the closed set both the regex path and the
# model's typed `render` output are held to.
_RENDER_KINDS = frozenset({"table", "chart", "pie", "bar", "line", "csv"})

_CHART_KIND_ALIASES = {"column": "bar", "donut": "pie", "doughnut": "pie",
                       "graph": "chart", "plot": "chart", "visualize": "chart",
                       "visualise": "chart", "excel": "csv", "spreadsheet": "csv",
                       "file": "csv"}


def _presentation_kind(message: str) -> str | None:
    """Which rendering a presentation-only follow-up asked for — "pie"/"bar"/"line"/
    "table"/"chart"/"csv" — or None when this is not one.

    NOTE there is deliberately no _DATA_QUESTION_HINTS guard here, unlike the
    smalltalk patterns: "table" and "chart" are themselves in that hint list, so the
    guard rejected the very phrasings this is for ("show that as a table"). The
    patterns are instead anchored across the WHOLE message, which is a stronger
    guarantee — a message carrying any real question alongside the rendering word
    ("show me sales as a pie chart", "give me a table of all listings") simply has
    content left over and cannot match. Guard tests cover both directions.
    """
    # A BARE rendering noun is not a command. "table", "pie", "chart" and "column" are
    # nouns as often as instructions, and because this check runs before the
    # clarification branch, answering "which one?" with "column" silently redrew the
    # previous result instead of answering the question. Something must accompany the
    # noun — a verb, a pronoun, an "as/into", or a trailing qualifier ("pie chart").
    words = [w for w in re.split(r"[^\w]+", (message or "").lower()) if w]
    if len(words) < 2 and not (words and words[0] in _UNAMBIGUOUS_RENDER_VERBS):
        return None
    for pattern in (_PRESENTATION_RE, _EXPORT_RE, _PLOT_RE):
        match = pattern.match(message)
        if match:
            kind = (match.group("kind") or "csv").lower()
            return _CHART_KIND_ALIASES.get(kind, kind)
    return None


# Hindi/Hinglish greetings — the same social intent as "hi"/"how are you", which the
# English-only patterns above silently missed in a product whose users type both.
_HINGLISH_SOCIAL_RE = re.compile(
    r"^\s*(?:namaste|namaskar|salaam|salam|shukriya|dhanyavaad|dhanyawad|"
    r"kaise\s*(?:ho|hain)|kaisa\s*hai|kya\s*haal(?:\s*(?:hai|chaal))?|"
    r"theek\s*ho|sab\s*theek|alvida|phir\s*milenge)"
    r"(?:\s*(?:ji|bhai|yaar|sir))?\s*[.,!?]*\s*$", re.IGNORECASE)


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
# A reset is composed, not enumerated: a DISCARD VERB, optionally followed by something
# that names the CONVERSATION'S OWN STATE. Two small closed sets, so combinations nobody
# wrote down still work ("wipe the context", "clear this chat", "forget the history") —
# the previous form listed whole phrases and 7 of the suite's 12 fell through it, and a
# longer list of phrases would only move that boundary.
#
# What keeps it safe is the anchored WHOLE-message match plus the object set: every
# object names the conversation itself, never data. So "clear" resets, and "clear the
# top 5 by amount" is a question that cannot match — there is no phrasing of a data
# question that is only a discard verb and a conversation noun.
#
# A reset that works only when phrased the canonical way is worse than none: the user
# believes the context is gone, and the next answer silently carries it.
_DISCARD_VERB = r"(?:clear|reset|forget|wipe|erase|drop|discard)"
# "what I said" / "what I asked" name the conversation's state as much as "history"
# does, so they belong in the object set rather than bolted on as a special case.
_CONVERSATION_STATE = (r"(?:everything|all|it|that|this|memory|context|history|chat|"
                       r"conversation|session|topic|what\s+i\s+(?:said|asked))")
_RESET_RE = re.compile(
    r"^\s*(?:"
    # a discard verb, alone or pointed at the conversation's own state
    rf"{_DISCARD_VERB}(?:\s+(?:the|this|my|our)?\s*{_CONVERSATION_STATE})*"
    # or the "begin afresh" family, which names no object at all
    r"|(?:let'?s\s+)?(?:start|begin)\s+(?:over|again|afresh|fresh)"
    rf"|(?:start|begin)?\s*(?:a\s+)?new\s+{_CONVERSATION_STATE}"
    r")\s*[.,!?]*\s*$",
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

# The SHAPE of a narrowing follow-up — "only the Nagpur ones", "just the FULL ones",
# "what about SEMI". Used only as a tie-breaker when the model could not label a turn
# (delta `ambiguous`) and a clarification from a FAILED turn is pending: nobody answers
# "what does 'pool' refer to?" with "only the Nagpur ones", so this shape continues the
# live frame instead of being glued onto the dead question. A bare answer ("Nagpur",
# "the amenities column") does not match and still completes the clarification.
_CONTINUATION_SHAPE_RE = re.compile(
    r"^\s*(and\s+)?(only|just)\s+(?!if\b|when\b)(the\s+)?\S.*$|^\s*(and\s+)?what\s+about\s+\S",
    re.IGNORECASE,
)


# ── Turn Entry Gate ──────────────────────────────────────────────────────────
#
# The gate answers ONE question — does this turn need VEDA, and does it need
# conversational context? — and it is NOT a new classifier. classify_node already
# makes exactly that decision: its seven deterministic fast paths are the L0 tier, its
# single SLM call is the model tier, and _route_after_classify already sends the result
# to the engine or away from it. A second classifier in front would have to re-derive
# the same verdict from the same message, and two classifiers that can disagree is the
# failure this package has spent its whole history removing (see the overrides in this
# file, every one of which exists because one layer overruled another).
#
# So the gate is DERIVED from the action classify_node already chose, and the only new
# behaviour is the normalisation below.
_ENTRY_DIRECT = frozenset({"smalltalk"})                      # no VEDA, no context
_ENTRY_CONTEXT_ONLY = frozenset({"recall", "represent", "reset", "no_match"})


def entry_decision(action: str, has_history: bool) -> dict:
    """{requires_veda, requires_context} for a turn classify_node has already judged.

    `requires_context` mirrors the ROUTING rule, not the action label: a turn reaches
    context_resolve_node when history exists (chatbot/graph.py::_route_after_classify),
    because whether the message needs the previous turn is decided from checkpointed
    state, not from the model calling it "followup" rather than "answer". Keeping the
    two in step here is deliberate — a gate that disagreed with the router would report
    a path the turn did not take.

    Fails toward MORE processing: an action this function does not recognise is treated
    as needing the engine, never as a direct answer.
    """
    if action in _ENTRY_DIRECT:
        return {"requires_veda": False, "requires_context": False}
    if action in _ENTRY_CONTEXT_ONLY:
        # Answered from the QueryFrame the memory layer already loaded and authorised —
        # never from the engine, and never from anything this gate invented.
        return {"requires_veda": False, "requires_context": True}
    if action == "runtime_context":
        # A current-date question: self-contained by construction, so it skips
        # context_resolve_node however much history exists.
        return {"requires_veda": True, "requires_context": False}
    return {"requires_veda": True, "requires_context": bool(has_history)}


# Runs of THREE OR MORE of the same letter, and a trailing tail of punctuation,
# whitespace and symbols. Three, not two, so a word that legitimately doubles a letter
# ("hello", "morning") is never touched.
_L0_REPEAT_RE = re.compile(r"(.)\1{2,}")
_L0_TAIL_RE = re.compile(r"[\s\W_]+$", re.UNICODE)
_L0_NORMALISE = os.environ.get("CHATBOT_ENTRY_GATE_NORMALISE", "1") == "1"


def _normalise_for_l0(message: str) -> str | None:
    """A conservative spelling of `message` to retry the EXISTING canned matcher with,
    or None when normalising changes nothing.

    This is the one piece of new behaviour in the gate, and it is deliberately NOT a
    greeting dictionary — it adds no word. It only collapses the two things people do
    to words they are not really typing: hold a key down, and end with punctuation or
    an emoji. Measured 2026-09-21 against the 346-message suite: "heeeey 😂" and
    "good morninggg" were the two of the brief's own L0 examples the canned path
    missed, and both are exactly this shape.

    A misspelling is NOT normalised ("helo", "hlo", "gud morning"): guessing at
    intended letters is where a conservative gate stops and the model starts.
    """
    if not _L0_NORMALISE or not message:
        return None
    text = _L0_TAIL_RE.sub("", message.strip())
    text = _L0_REPEAT_RE.sub(r"\1", text)
    text = " ".join(text.split())
    return text if text and text != message.strip() else None


def _is_punctuation_only(message: str) -> bool:
    """Is this message nothing but punctuation — "???", "...", "!!!" ?

    Unicode CATEGORY, not a character list: every non-space character must be in a
    punctuation class (Unicode "P*"). That is what keeps an emoji out — "👍" is category
    So (Symbol, other), not punctuation, and it is an acknowledgement rather than
    gibberish. A message with any letter or digit is never caught here, so "okay",
    "hola" and "helo" are untouched, and this can never swallow a real question.

    This is the one unintelligible shape that can be recognised with no dictionary and
    no model: a message containing no word at all is not asking anything.
    """
    text = (message or "").strip()
    if not text:
        return False
    return all(unicodedata.category(ch).startswith("P") for ch in text if not ch.isspace())


def _canned_smalltalk_reply(message: str, _depth: int = 0) -> str | None:
    """Instant reply for the fast-path patterns above — None means "not a fast
    match, fall back to the LLM" (used by both classify_node and smalltalk_node
    so the two stay in lockstep on what counts as trivial smalltalk)."""
    if _DATA_QUESTION_HINTS.search(message):
        return None
    if _is_punctuation_only(message):
        # Measured 2026-09-21: "???" was classified as a data question and spent a full
        # engine round-trip. It names nothing and asks nothing; the engine has no more
        # chance of answering it than this line does.
        return REPHRASE_REPLY
    if _GREETING_RE.match(message):
        return FALLBACK_REPLY
    if _THANKS_RE.match(message):
        return "You're welcome! Let me know if you have any other data questions."
    if _BYE_RE.match(message):
        return "Goodbye! Come back anytime you have data questions."
    if _IDENTITY_RE.match(message):
        return IDENTITY_REPLY
    if _CAPABILITY_RE.match(message):
        return CAPABILITY_REPLY
    if _HINGLISH_SOCIAL_RE.match(message):
        return FALLBACK_REPLY
    # Second and last attempt, on a conservatively normalised spelling of the SAME
    # message against the SAME patterns above — no extra words, no extra patterns.
    # `_depth` stops the retry recursing: the normalised form is tried once.
    if _depth == 0:
        normalised = _normalise_for_l0(message)
        if normalised:
            return _canned_smalltalk_reply(normalised, _depth=1)
    return None



def _mentions_frame_subject(message: str, frame: dict) -> bool:
    """Does this message actually NAME something the active frame is about?

    The positive half of the "the model said smalltalk, should we believe it?" question.
    The negative half — "is this a greeting?" — was tried first and cannot work: the set
    of things a person types after reading an answer is open ("okay", "hmm", "acha",
    "thik hai", "makes sense"), so every word missing from the list became a turn sent to
    the SQL engine as data. Measured 2026-09-19: 7 of 8 plain acknowledgements reached
    the engine, "okay" and "ok" among them — the two most common things anyone types.

    So the test is inverted. The frame is executed-SQL evidence of what the session is
    about: its entity, the fields and values it filtered on, what it grouped and
    measured. A message that shares a word with ANY of those is talking about the data;
    one that shares none cannot be, whatever it is made of. Nothing to maintain, and a
    filler word nobody has thought of yet is handled by default.
    """
    words = _content_words(message)
    if not words:
        return False
    subject: set = set()
    for value in (frame.get("entity"), frame.get("entity_display")):
        subject |= _content_words(value)
    for f in (frame.get("filters") or []):
        subject |= _content_words(f.get("field"))
        subject |= _content_words(f.get("value"))
    for group in (frame.get("group_by") or []):
        subject |= _content_words(group)
    for measure in (frame.get("measures") or []):
        subject |= _content_words(measure)
    for order in (frame.get("order_by") or []):
        subject |= _content_words((order or {}).get("field"))
    return bool(words & subject)


_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _content_words(text) -> set:
    """Words of a message or a frame fact, folded to a comparable form.

    Splits on separators AND camelCase, drops tokens under three characters (they
    collide with everything), and folds a trailing "s" so a question's "transactions"
    matches a frame's "transaction". The same crude-on-purpose folding
    apps/query/data_vocabulary.py uses — a real stemmer would also fold unrelated words
    together, which here would mean overriding the model on a message that shares
    nothing with the data.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    out = set()
    for token in _WORD_SPLIT_RE.split(spaced.lower()):
        if len(token) < 3:
            continue
        out.add(token)
        if token.endswith("s") and len(token) > 3:
            out.add(token[:-1])
        else:
            out.add(token + "s")
    return out


# ── Return to an earlier topic ───────────────────────────────────────────────────────────
#
# "go back to the properties" after the conversation moved on to payments. The topic index
# (chatbot/memory/topics.py) holds a snapshot of each earlier topic; this decides whether a
# message asks to RETURN to one of them. Deterministic, from structured evidence — the
# stored topics' own entity, display name, root question and filter values — plus two
# closed, entity-free word sets:
#
#   _RETURN_WORDS        the grammar of returning. Without one of these a message naming
#                        an earlier topic is a NEW question about it ("show payment
#                        transactions by month"), and replaying the old snapshot would
#                        answer something the user did not ask.
#   _CONVERSATION_WORDS  words that name a piece of the conversation rather than data
#                        ("the property ANALYSIS", "that TOPIC"), and the conversational
#                        verbs the central function-word list lacks ("LET's", "GOING").
#                        Set aside, like the function words: they cannot identify a topic.
#
# Every other word must be accounted for by ONE topic's own words. A message with anything
# left over ("go back to the payments in 2024") is not a pure return — it carries a request
# of its own — and falls through to the normal path unchanged. Fails closed at every step:
# no return word, nothing left to match on (plain "go back" — the drill-up path's), no
# topic matched, or the CURRENT topic among the matches → None, and the turn is handled
# exactly as it was before this existed.
_RETURN_WORDS = frozenset({"back", "return", "revisit", "resume", "earlier", "previous",
                           "again", "continue"})
_CONVERSATION_WORDS = frozenset({"analysis", "topic", "topics", "question", "questions",
                                 "query", "queries", "discussion", "conversation", "view",
                                 "report", "results", "result", "numbers", "data", "ones",
                                 "let", "lets", "going"})

# The operation a restore sends to the engine and records as this turn's delta. From the
# closed set (chatbot/prompts/delta_types.py). WHY "drill_up", recorded because it was an
# explicit design question:
#   · At the engine boundary `operation` has exactly ONE meaning (memory/context.py,
#     veda/pipeline.py:270): "the remembered shape is being REPLAYED, not replaced". A
#     restore is precisely that — the snapshot's own root question re-asked with the
#     snapshot's own filters and grouping. Every other member of the set is inert there,
#     and the root question restates its grouping in words ("distribution ... by facing"),
#     so the engine's shape rule (pipeline.py:1980-1984) then reads it as a NEW grouping
#     request and drops a remembered group column the user never said (corner_property):
#     a filtered restore would come back as raw rows — B2, measured 2026-09-25 on this
#     exact base question.
#   · The conversation-layer side effects of drill_up do NOT happen: the pop in
#     context_resolve_node is never reached (the restore branch returns first), and
#     memory_write_node keys the restore's own handling (base_query, stack) on
#     `topic_restore`, not on this label. So the label is ONLY what the engine sees; if the
#     engine ever grows a distinct "replay" operation, change it here and nothing else.
_RESTORE_OPERATION = "drill_up"


def _topic_forms(token: str) -> set:
    """_content_words' number folding, plus the -y/-ies pair it does not fold
    ("property" / "properties") — a topic is named in either number."""
    forms = _content_words(token)
    if token.endswith("ies") and len(token) > 4:
        forms |= _content_words(token[:-3] + "y")
    elif token.endswith("y") and len(token) > 3:
        forms |= _content_words(token[:-1] + "ies")
    return forms


def _topic_words(entry: dict) -> set:
    """Every word a remembered topic can be named by: its table, its business name, the
    user's own root question, and the values it was narrowed to."""
    texts = [entry.get("entity"), entry.get("entity_display"), entry.get("base_query")]
    texts += [f.get("value") for f in (entry.get("filters") or []) if isinstance(f, dict)]
    words: set = set()
    for text in texts:
        for w in _content_words(text):
            words |= _topic_forms(w)
    return words


def _match_remembered_topic(message: str, index, frame: dict) -> Optional[dict]:
    """{"kind": "restore", "topic": entry} | {"kind": "ambiguous", "candidates": [...]} |
    None. See the block comment above for the rule; `index` is memory_read_node's
    already-authorised topic index, so a revoked topic can never be matched."""
    if not index:
        return None
    tokens = [t for t in _WORD_SPLIT_RE.split((message or "").lower()) if len(t) >= 3]
    if not any(t in _RETURN_WORDS for t in tokens):
        return None
    residual = [t for t in tokens if t not in _RETURN_WORDS and t not in _FUNCTION_WORDS
                and t not in _CONVERSATION_WORDS]
    if not residual:
        return None                 # plain "go back": a drill-up of the CURRENT topic
    matched = [e for e in index if isinstance(e, dict) and e.get("entity")
               and all(_topic_forms(t) & _topic_words(e) for t in residual)]
    if not matched:
        return None
    current = memory_topics.topic_key(frame) if (frame or {}).get("entity") else None
    if current is not None and any(memory_topics.topic_key(e) == current for e in matched):
        # The message names where the conversation already IS. Not a return — whatever it
        # is, the existing path decides it.
        return None
    if len(matched) == 1:
        return {"kind": "restore", "topic": matched[0]}
    return {"kind": "ambiguous", "candidates": matched}


def _match_named_document(message: str, index, frame: dict) -> Optional[dict]:
    """{"kind": "document", "topic": entry} when the message NAMES a document this
    conversation already discussed and the conversation is not in it now; else None.

    The return path above needs a return word because a table topic named in a new
    question ("show payment transactions by month") is a new question about that table.
    A document is different: naming it IS the anchor — "what is the fee for repair in the
    maintenance policy?" asks the policy, whatever the conversation did in between.
    Measured 2026-09-25 (demo X1): after one database question that turn went to the
    federated route and answered from a structured fee table, because nothing told the
    engine the conversation had a maintenance-policy thread to go back to.

    "Named" is frame.py's own test (_already_names): every distinctive word of the
    document's name appears in the message — the stored name, not a word list. Exactly one
    document must match; two is left to the existing path rather than guessed."""
    if not index:
        return None
    docs = [e for e in index if isinstance(e, dict) and e.get("entity")
            and e.get("entity_is_document")
            and memory_frame._already_names(message or "", str(e["entity"]))]
    if len(docs) != 1:
        return None
    if (memory_frame.is_document_frame(frame)
            and memory_topics.topic_key(frame) == memory_topics.topic_key(docs[0])):
        return None                 # already in that document — the normal path anchors it
    return {"kind": "document", "topic": docs[0]}


def _returns_to_current_document(message: str, frame: dict) -> bool:
    """Is this a pure "go back to <the document we are already in>"?

    Pure means the same thing as in _match_remembered_topic: a return word, and every
    other word accounted for by the document's name or set aside as grammar. "what does
    the maintenance policy say again about fees" carries a question of its own and is not
    one. Measured 2026-09-25: this reached the engine as a question and came back as a
    bare "Sources: (maintenance_policy.docx)" with no answer in it."""
    if not memory_frame.is_document_frame(frame) or not frame.get("entity"):
        return False
    tokens = [t for t in _WORD_SPLIT_RE.split((message or "").lower()) if len(t) >= 3]
    if not any(t in _RETURN_WORDS for t in tokens):
        return False
    if not memory_frame._already_names(message, str(frame["entity"])):
        return False
    doc_words = set(_WORD_SPLIT_RE.split(str(frame["entity"]).lower()))
    return not [t for t in tokens if t not in _RETURN_WORDS and t not in _FUNCTION_WORDS
                and t not in _CONVERSATION_WORDS and t not in doc_words]


def _is_social(message: str) -> bool:
    """Is this conversational rather than analytical? Wider than the canned test.

    The canned regexes answer a STRICTER question — "can I reply to this with a
    fixed string, no model call at all" — and using them to decide "is this a
    genuine greeting" made the override below fire on anything they could not
    themselves answer. Measured: the LLM classified "hi there!" as smalltalk and
    was OVERRULED into a followup purely because the canned pattern missed it,
    sending a greeting through the full engine.

    A message that OPENS with a social token, carries no data verb, and contains
    no referential language has nothing to follow up ON, whatever the frame says.
    The referential guard is what keeps "hi, what about the other one" a followup.
    """
    if _DATA_QUESTION_HINTS.search(message):
        return False
    # A canned pattern matching the WHOLE message is unambiguous — check it before
    # the referential guard, or "how's it going" is rejected over the "it" in it.
    if (_GREETING_RE.match(message) or _THANKS_RE.match(message)
            or _BYE_RE.match(message) or _IDENTITY_RE.match(message)
            or _CAPABILITY_RE.match(message) or _HINGLISH_SOCIAL_RE.match(message)):
        return True
    # Only the LOOSER opener match needs the referential guard: "hi" at the front
    # does not make "hi, what about the other one" social.
    return bool(_SOCIAL_OPENER_RE.match(message)
                and not _REFERENTIAL_HINTS.search(message))


_SOCIAL_OPENER_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hiya|yo|greetings|thanks?|thank\s*you|thx|ty|cheers|"
    r"bye|goodbye|see\s*(?:you|ya)|take\s*care|"
    r"good\s*(?:morning|afternoon|evening|day|night))\b", re.IGNORECASE)

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
    if state.get("memory_reset"):
        logger.info("classify_node: memory was reset this turn — acknowledging without "
                    "sending %r to the engine", message)
        return {"action": "reset", "resolved_query": None, "sql": None, "rows": None,
                "status": None, "engine_result": {}, "last_result": {},
                "needs_clarification": False, "clarification_question": None,
                "engine_unavailable": False, "delta_type": None, "context_used": None}

    # Re-present the answer already on screen — no engine call, no SQL, no model.
    # Requires a PREVIOUS answered result to re-present: without one there is
    # nothing to chart, so "as a pie chart" falls through to the normal path and the
    # engine's own refuse/clarify handles it, rather than replying about nothing.
    # A question about the conversation itself, answerable from the QueryFrame the
    # memory layer already stored. Requires a frame with an entity — without one there
    # is no "last query" to describe, so it falls through to the normal path.
    recall_kind = _recall_kind(message)
    if recall_kind == "topics":
        # Answered from the topic index, which can hold topics even when THIS source has no
        # frame (they may belong to another source the caller can still see). With an empty
        # index recall_node says there is nothing to list.
        logger.info("classify_node: recall question about the conversation's topics — "
                    "answering from the topic index, engine not called: %r", message)
        return {"action": "recall", "resolved_query": None, "recall_kind": "topics",
                "needs_clarification": False, "clarification_question": None,
                "engine_unavailable": False, "pending_clarification": {},
                "engine_result": {}, "sql": None, "rows": None, "status": None,
                "context_used": None}
    if recall_kind and not frame.get("entity"):
        # Nothing to recall — but "what sql did you run" still has an honest, instant
        # answer, and the engine has none. Measured twice: 62s after a "start over", and
        # 101s as the FIRST message of a session, both returning "Could you clarify if
        # 'did' is a column name". The first-message case used to be excluded on the
        # theory that it might be a real question; it is not — there is no reading of
        # "what SQL did you run" that the SQL engine can answer.
        logger.info("classify_node: recall question with no frame to recall from — "
                    "answering without the engine: %r", message)
        return {"action": "recall", "resolved_query": None, "recall_kind": "nothing",
                "needs_clarification": False, "clarification_question": None,
                "engine_unavailable": False, "pending_clarification": {},
                "engine_result": {}, "sql": None, "rows": None, "status": None,
                "context_used": None}
    if recall_kind and frame.get("entity"):
        logger.info("classify_node: recall question (%s) — answering from the frame, "
                    "engine not called: %r", recall_kind, message)
        # Partial update, same reason as the presentation path below: the reset at the
        # end of this function would clear engine_result, and the "trail" answer reads
        # the previous turn's explain block out of it.
        # engine_result CLEARED, unlike the presentation path below. recall_node answers
        # from the FRAME alone and never reads engine_result — but apps/chat/services.py
        # renders a markdown table and charts from whatever `res0` carries, so letting the
        # previous turn's result survive made "what SQL did you run" re-emit that entire
        # table and its charts alongside the one-line answer.
        return {"action": "recall", "resolved_query": None,
                "needs_clarification": False, "clarification_question": None,
                "engine_unavailable": False, "recall_kind": recall_kind,
                "pending_clarification": {},
                "engine_result": {}, "sql": None, "rows": None, "status": None,
                "context_used": None}
    presentation_kind = _presentation_kind(message)
    # `last_result` (written by memory_write_node, and never cleared by the reset at the
    # end of this function) rather than `engine_result`: any turn in between wipes
    # engine_result, so "show me X" → "thanks" → "as a pie chart" fell through to the
    # engine and re-ran the SQL. engine_result stays as the fallback for the case where
    # the previous turn IS the answered one.
    previous_result = state.get("last_result") or state.get("engine_result") or {}
    if presentation_kind and previous_result.get("rows"):
        logger.info("classify_node: presentation-only follow-up (%s) — reusing the "
                    "previous result, engine not called: %r", presentation_kind, message)
        # Deliberately a PARTIAL update: every key the normal reset below clears
        # (engine_result/sql/rows/status) is OMITTED here, so LangGraph leaves the
        # checkpointed values from the answered turn in place for represent_node.
        return {"action": "represent", "resolved_query": None,
                "needs_clarification": False, "clarification_question": None,
                "engine_unavailable": False, "viz_override": presentation_kind,
                "pending_clarification": {}, "context_used": None}
    deterministic_smalltalk = _canned_smalltalk_reply(message) is not None
    delta_type = None          # None = "not computed this turn", see context_resolve_node
    delta_field = ""           # replace/remove only — which remembered filter it acts on
    delta_value = ""           # the grounded word from the user's own message

    # A pending clarification is consumed ONLY by a message that actually reads as an
    # ANSWER to it — checked AFTER every deterministic fast path above, and never for a
    # message carrying a data question of its own.
    #
    # The first cut ran this branch second, on nothing but "a slot exists", and a live
    # 33-turn run (2026-09-17) showed what that costs. 12 of 21 engine turns came back
    # something other than "answered", so the slot was armed most of the time, and every
    # following message — greetings included — was concatenated onto the stored request
    # and re-armed, compounding without bound:
    #   "show me the top 5 cities by number of assets for hi for as a pie chart for
    #    what sql did you run for only the top 3"
    #   -> 'Could you clarify if "chart" is a column name or a value to filter on?'
    # The engine read `hi`, `back`, `chart`, `did` and `pie` as data — every one a
    # conversation-layer token the user never meant as one. ~1,100s of engine time in
    # that sample alone.
    #
    # A message that does NOT qualify does not merely skip this branch: it CLEARS the
    # slot. Leaving it armed is what turned one unanswered turn into a poisoned session.
    # An EXPLICIT re-shaping of the previous question ("by city instead", "make it top
    # 5"). Computed here, before the pending-clarification branch, because it outranks
    # one: see _is_clarification_answer. Requires the frame to actually hold the slot, so
    # it can never fire on a session with nothing to re-shape.
    shape_change = memory_frame.detect_shape_delta(frame, message, explicit_only=True)
    # A request to RETURN to an earlier topic ("go back to the properties"). Deterministic,
    # from the authorised topic index; None on every message that is not one.
    topic_hit = _match_remembered_topic(message, state.get("topic_index") or [], frame)
    if topic_hit is None:
        # Not a topic of THIS conversation — perhaps of an earlier one ("continue the
        # property analysis" in a new chat). Same matcher, same fail-closed ambiguity.
        topic_hit = _match_remembered_topic(message, state.get("previous_topics") or [],
                                            frame)
    if topic_hit is None:
        # A question that names a document this conversation already read (no return word
        # needed — see _match_named_document).
        topic_hit = _match_named_document(message, state.get("topic_index") or [], frame)
    if (topic_hit is None and memory_frame.is_document_frame(frame)
            and state.get("message_names_only_values") is True):
        # Only data VALUES ("only the Nagpur ones", "Pune") while the conversation is on a
        # document: a value narrows a table, never a PDF. The table topic the conversation
        # was on most recently is what it narrows. Measured 2026-09-26: properties → Nagpur
        # → a handbook question → "only the FULL ones" was asked of the handbook.
        _table = next((e for e in (state.get("topic_index") or [])
                       if isinstance(e, dict) and e.get("entity")
                       and not e.get("entity_is_document")), None)
        if _table is not None:
            topic_hit = {"kind": "continue", "topic": _table}

    _pending_raw = state.get("pending_clarification")
    pending = (_pending_raw.get("original_query")
               if isinstance(_pending_raw, dict) else None)
    # A message that picks rows of the answer on screen ("the 3rd one", "the first three",
    # a shown record's name) is never the answer to a pending clarification. Measured
    # 2026-09-26: after a refused "only the ones on the moon" armed one, "the 3rd one" was
    # glued onto it and sent as "only the ones on the moon for the 3rd one".
    _picks_shown_rows = bool(memory_reference.result_pointer(message)
                             or memory_reference.selects_rows(
                                 state.get("result_reference"), message, frame))
    is_smalltalk_or_fast_path = bool(
        deterministic_smalltalk or recall_kind or presentation_kind or shape_change
        or _RESET_RE.match(message) or _DRILL_UP_RE.match(message)
        or _RUNTIME_CONTEXT_RE.match(message) or topic_hit or _picks_shown_rows)
    if pending:
        if not topic_hit and not _picks_shown_rows and _is_clarification_answer(message, recall_kind, presentation_kind,
                                                      deterministic_smalltalk,
                                                      bool(shape_change)):
            logger.info("classify_node: a clarification is pending and this message "
                        "answers it: %r", message)
            return {"action": "clarify_reply", "resolved_query": None, "sql": None,
                    "rows": None, "status": None, "engine_result": {},
                    "needs_clarification": False, "clarification_question": None,
                    "engine_unavailable": False, "delta_type": None,
                    "delta_field": "", "delta_value": "", "context_used": None}
        if is_smalltalk_or_fast_path:
            # Unambiguously not an answer — the user moved on. Drop the request rather
            # than leave it armed to swallow the turn after this one.
            logger.info("classify_node: a clarification was pending but %r took a "
                        "deterministic path — dropping the pending request", message)
            state = {**state, "pending_clarification": {}}
            pending = None
        else:
            # Genuinely ambiguous. Leave the slot ARMED and let the classifier below
            # decide: if it returns "clarify_reply" the graph completes the pending
            # request, otherwise the turn is handled on its own and the normal return
            # clears the slot. Neither guessing nor discarding.
            logger.info("classify_node: a clarification is pending and %r is not an "
                        "obvious value — deferring to the classifier", message)

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
    elif (presentation_kind and memory_frame.is_document_frame(frame)
            and not previous_result.get("rows")):
        # "as a pie chart" asked of a DOCUMENT answer. The represent fast path above
        # requires the previous result to carry rows, and a retrieval answer never
        # does, so this used to fall through to the model and then to the engine —
        # measured 2026-09-22 on the real docs_contracts source, it came back with the
        # generic "I'm here for questions about your data", which tells the user
        # nothing about why their chart did not appear.
        #
        # Same reasoning as the shape and drill_up branches below: prose from a PDF has
        # nothing to plot, permanently, so it is settled here from memory at no cost
        # rather than spending an engine round-trip to be told so.
        logger.info("classify_node: a presentation change asked of a document answer — "
                    "answering from memory, engine not called: %r", message)
        return {"action": "recall", "resolved_query": None,
                "recall_kind": "presentation_not_applicable", "needs_clarification": False,
                "clarification_question": None, "engine_unavailable": False,
                "pending_clarification": {}, "engine_result": {}, "sql": None,
                "rows": None, "status": None, "context_used": None}
    elif (memory_frame.is_document_frame(frame)
            and memory_frame.matches_shape_phrase(message)):
        # "make it top 10" / "by month instead" / "don't sort by amount" asked of a
        # DOCUMENT. There is nothing to reshape: a retrieval answer has no row limit,
        # no grouping and no ordering, and never will. Measured 2026-09-21 on the real
        # docs_contracts source: this reached the engine, cost 19 seconds, and came
        # back "The provided context does not contain information about a 'top 10'
        # list" — the same failure shape as "go back" on the same path.
        logger.info("classify_node: a shape change asked of a document answer — "
                    "answering from memory, engine not called: %r", message)
        return {"action": "recall", "resolved_query": None,
                "recall_kind": "shape_not_applicable", "needs_clarification": False,
                "clarification_question": None, "engine_unavailable": False,
                "pending_clarification": {}, "engine_result": {}, "sql": None,
                "rows": None, "status": None, "context_used": None}
    elif (state.get("memory_revoked_source") is not None and not topic_hit
            and (_REFERENTIAL_HINTS.search(message) or _picks_shown_rows)):
        # The message points back at an earlier answer ("show me those results", "the 2nd
        # one") whose source this caller can no longer see — memory_read_node just dropped
        # it. Measured 2026-09-26 (audit P1): "Show me those results" was answered from the
        # employee handbook, and "the 2nd one" as smalltalk. Neither said what happened.
        logger.info("classify_node: follow-up points at memory from a source no longer in "
                    "scope (%r) — answering from memory, engine not called: %r",
                    state.get("memory_revoked_source"), message)
        return {"action": "recall", "resolved_query": None,
                "recall_kind": "access_revoked", "needs_clarification": False,
                "clarification_question": None, "engine_unavailable": False,
                "pending_clarification": {}, "engine_result": {}, "sql": None,
                "rows": None, "status": None, "context_used": None}
    elif not topic_hit and _returns_to_current_document(message, frame):
        logger.info("classify_node: a return to the document the conversation is already "
                    "in — answering from memory, engine not called: %r", message)
        return {"action": "recall", "resolved_query": None,
                "recall_kind": "already_on_document", "needs_clarification": False,
                "clarification_question": None, "engine_unavailable": False,
                "pending_clarification": {}, "engine_result": {}, "sql": None,
                "rows": None, "status": None, "context_used": None}
    elif topic_hit:
        # RETURN TO AN EARLIER TOPIC — deterministic, no model call. context_resolve_node
        # restores the snapshot (or, when several topics match, asks which). Decided before
        # the drill-up branches below, which it can never collide with: a message this
        # matches has named a remembered NON-current topic, which a bare "go back" (the
        # drill-up path) by construction does not.
        action = "followup"
        delta_type = _RESTORE_OPERATION if topic_hit["kind"] == "restore" else None
        logger.info("classify_node: return to an earlier topic (%s) — %s: %r",
                    topic_hit["kind"],
                    [memory_topics.topic_key(e) for e in
                     ([topic_hit["topic"]] if topic_hit.get("topic")
                      else topic_hit["candidates"])], message)
    elif _DRILL_UP_RE.match(message) and not state.get("drill_stack"):
        # "go back" with nothing to go back FROM. Previously this fell through to the
        # model and then to the ENGINE: measured 2026-09-21 on the document source,
        # "go back" cost 27 seconds and came back "The provided context does not
        # contain the answer to the question 'go back'" — the retrieval pipeline was
        # asked to find a navigation word in a PDF.
        #
        # It is especially wrong on a document conversation, where a drill stack is
        # never built at all, so this is the PERMANENT state there rather than an edge
        # case. The honest answer costs nothing and needs no model: there is nothing to
        # go back to.
        logger.info("classify_node: drill-up with an empty stack — answering from "
                    "memory, engine not called: %r", message)
        return {"action": "recall", "resolved_query": None,
                "recall_kind": "drill_up_empty", "needs_clarification": False,
                "clarification_question": None, "engine_unavailable": False,
                "pending_clarification": {}, "engine_result": {}, "sql": None,
                "rows": None, "status": None, "context_used": None}
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
        raw = call_slm(
            build_supervisor_system_prompt(frame),   # built fresh each call so "today" is always
                                                      # current; includes the delta addendum only
                                                      # when `frame` has an entity (see its docstring)
            build_supervisor_user_prompt(message, history),
            model=CHATBOT_CLASSIFY_MODEL,
            purpose="classify",
            temperature=_DECISION_TEMPERATURE,
        )
        action = "answer"
        model_render = None
        if raw:
            match = _JSON_RE.search(raw)
            if match:
                try:
                    parsed = json.loads(match.group())
                    candidate = parsed.get("action")
                    if candidate in _VALID_ACTIONS:
                        action = candidate
                    model_render = parsed.get("render")
                except Exception:
                    logger.warning("classify_node: could not parse LLM output: %r", raw)
            if frame.get("entity"):
                # Parse delta_type/slot_candidates from the SAME raw response —
                # shares classify_delta's exact vocabulary/confidence gates via
                # parse_delta_response, so a merged response is held to the
                # identical bar as the standalone fallback call.
                dt, _slots, _dfield = parse_delta_response(raw, message)
                if dt in DELTA_TYPES:
                    delta_field = _dfield
                    delta_value = _slots[0] if _slots else ""
                    delta_type = dt
        # THE MODEL'S TYPED RENDER REQUEST (plan step 7). The whole-message regexes above
        # stay the fast path; this catches the phrasings they miss — the harness's own
        # recorded gaps: "can i see that as a chart", "draw it", "show me a graph of
        # that", "as a bar graph instead". The model only NAMES a rendering; code decides
        # whether to honour it, and fails closed:
        #   · the kind must be one of the closed set the regex path produces;
        #   · there must be a previous result with rows to redraw, not a document;
        #   · the message must name NOTHING in the data (message_mentions_data is False,
        #     the api tier's vocabulary check). A message that names data ("the sales as
        #     a bar chart") is a data question and goes to the engine as before; an
        #     undecidable check (None) never honours the model.
        # Presentation only ever re-renders — it cannot change the analysis or memory.
        _render = _CHART_KIND_ALIASES.get(str(model_render or "").lower(),
                                          str(model_render or "").lower())
        if (_render in _RENDER_KINDS and previous_result.get("rows")
                and not memory_frame.is_document_frame(frame)
                and state.get("message_mentions_data") is False
                # "show the first three" / "the last two" pick rows OUT of the result —
                # a question, not a redraw of all of it (measured 2026-09-26).
                and not memory_reference.selects_rows(state.get("result_reference"),
                                                      message, frame)):
            logger.info("classify_node: model asked to re-render the previous result as "
                        "%s — engine not called: %r", _render, message)
            return {"action": "represent", "resolved_query": None,
                    "needs_clarification": False, "clarification_question": None,
                    "engine_unavailable": False, "viz_override": _render,
                    "pending_clarification": {}, "context_used": None}

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
            and not _is_social(message)
            and _mentions_frame_subject(message, frame)):
        # _DATA_QUESTION_HINTS is schema-agnostic action-words only (count/how many/
        # show me/...) — it never catches a bare entity mention like "tell me something
        # about transaction" (no table/column names hardcoded there by design), which is
        # the case this override exists for.
        #
        # It used to fire on "an active frame exists AND this isn't a greeting", and that
        # second half is an open set: measured 2026-09-19, "okay", "ok", "hmm", "got it",
        # "cool", "i see" and "makes sense" were all overridden into follow-ups and sent
        # to the SQL engine as data — 7 of 8 tried. The model had said smalltalk for every
        # one of them, correctly, and was overruled by a word list.
        #
        # Now it takes POSITIVE evidence instead: the message must actually name something
        # the frame is about (_mentions_frame_subject). "tell me something about
        # transaction" still fires; "okay" cannot, and neither can any filler nobody has
        # thought of yet. When no signal fires the model's own verdict stands — its prompt
        # already carries the HARD RULE that anything naming a data entity is never
        # smalltalk, so trusting it here is not the same as having no guard.
        logger.warning(
            "classify_node: LLM said smalltalk but the message names the active frame's "
            "subject (entity=%r), overriding to 'followup': %r",
            frame.get("entity"), message,
        )
        action = "followup"
    elif (action == "smalltalk" and frame.get("entity") and not _is_social(message)
            and _frame_free_action(message, history) == "answer"):
        # A real question the FRAME talked the model out of. Measured 2026-09-25 with the
        # frame on properties: "What is the probation period for new recruits?", "what is
        # the dress code?", "how long is the probation period" came back smalltalk 9/9
        # ("asks for a fact not in the frame") and were answered "I'm here for questions
        # about your data" — while the same prompt WITHOUT the frame, same history, said
        # answer 12/12 for them and smalltalk 27/27 for acknowledgements ("okay", "got
        # it", "makes sense", "great work", "hmm interesting", ...). So the second opinion
        # is the same model and prompt minus the frame, asked only when the frame-bearing
        # call alone said smalltalk. It can only promote to a NEW question (no remembered
        # state carried); anything else, or a failed call, leaves the verdict as it was.
        logger.warning("classify_node: LLM said smalltalk with a frame present, but the "
                       "frame-free reading is a new question — overriding to 'answer': %r",
                       message)
        action = "answer"
        delta_type = "new_topic"
        delta_field = ""
        delta_value = ""

    # Computed at most ONCE per turn and only when something below actually asks for
    # it — two different branches need the same verdict and neither should pay for a
    # second round-trip (nor should any turn that reaches neither pay for one at all).
    _carryover_cache: list = []

    def _carryover() -> bool:
        if not _carryover_cache:
            _carryover_cache.append(
                bool(history) and bool(frame.get("entity"))
                and bool(_REFERENTIAL_HINTS.search(message))
                and _carries_over_subject(message, history, frame))
        return _carryover_cache[0]

    # THE BACK-REFERENCE BACKSTOP. `answer` + `new_topic` is the single verdict that
    # discards the conversation: context_resolve_node's `referential` is False for it,
    # so ConversationContext.from_frame gets carry_state=False and the frame's entity
    # and filters never leave this process — AND memory_write_node's
    # `reset = delta_type == "new_topic"` empties the drill stack on the way out. The
    # turn therefore runs as a brand-new standalone question AND destroys the path the
    # user had drilled, both silently.
    #
    # Measured 2026-09-24 (evaluation/drilldown_l7/): over a real 25-turn drill session
    # the drill stack never left 0 on ANY turn, and a 16-message / 4-family probe of the
    # SUPERVISOR itself (the merged call that actually produces these labels — NOT the
    # classify_delta fallback) reproduced why, 2/2 runs per message: "What are their
    # prices?", "How much are they?", "What do these cost?", "Which one is the
    # cheapest?", "Which one has the largest area?" and "Which of these has the lowest
    # price per square foot?" ALL came back `answer` + `new_topic`. The supervisor is
    # not unsure about these — it is confidently wrong, which is why the existing
    # `ambiguous` handling never caught them.
    #
    # The outcome is DOWNGRADED, never upgraded: `ambiguous` is the honest label for
    # "this continues the frame but which operation is unknown". It is what makes the
    # rest of the machinery behave correctly on its own terms — the frame's context
    # travels (referential is now True), `hold_subject_on_unplaced_turn` protects the
    # subject, the drill stack is neither reset nor pruned, and memory_write_node's
    # deterministic `newly_added_filter` still pushes a level when the SQL that actually
    # ran added a filter. No delta this layer did not observe is ever invented.
    #
    # Four conditions, each one there so this can never withhold or pollute a real
    # question: a frame must exist (nothing to carry otherwise), there must be history,
    # the message must contain anaphoric language at all (_REFERENTIAL_HINTS — the cheap
    # deterministic pre-filter, so a self-contained question never pays for the call),
    # and the dedicated second opinion must AGREE. That last gate was measured on the
    # same 24 back-references and 8 self-contained controls: 20/24 caught, 0/8 controls
    # misfired (see chatbot/prompts/carryover_check.py). It fails closed to today's
    # behaviour on any SLM failure.
    if (action == "answer" and delta_type == "new_topic" and _carryover()):
        logger.warning(
            "classify_node: supervisor said answer/new_topic but %r carries the frame's "
            "subject over (entity=%r) — handling it as an unplaced follow-up so the "
            "conversation's context is not silently discarded", message, frame.get("entity"))
        action = "followup"
        delta_type = "ambiguous"

    # A BARE VALUE WITH A CONVERSATION IN PROGRESS. A message that names only data values
    # ("Nagpur", "EAST", "DEBIT") and no table or column has no subject of its own, so it
    # can only narrow the conversation it arrives in — the same reasoning as a pointer.
    # The evidence is the tenant's own vocabulary (apps/query/data_vocabulary.py::
    # names_only_values: every content word a sampled VALUE, none a table/column word),
    # not a phrase list and not a second model call. Measured 2026-09-25: after
    # "distribution of properties by facing", a bare "Nagpur" was labelled answer/ambiguous
    # on one run and followup on the next; the first carried no context and was grounded
    # on a city/phone-code lookup table instead of the properties being discussed.
    # Downgraded to the honest label, as above; None (undecidable) changes nothing, and a
    # word naming a table ("vendors") is a subject, so it is never caught here.
    # Both labellings that throw the conversation away are covered: answer/new_topic and
    # FOLLOWUP/new_topic — measured 2026-09-25 (edge E7): "what about Pune?" after a Nagpur
    # drill came back followup/new_topic, and memory_write_node's `reset` on new_topic wiped
    # the drill path, so the next "go back" had nothing to return to. A message naming only a
    # value is never a new topic while a conversation is in progress.
    if (action in ("answer", "followup") and delta_type in ("new_topic", "ambiguous")
            and not (action == "followup" and delta_type == "ambiguous")
            and frame.get("entity") and history
            and state.get("message_names_only_values") is True):
        logger.info("classify_node: %r names only data values while the conversation is "
                    "on %r — continuing it rather than starting over",
                    message, frame.get("entity"))
        action = "followup"
        delta_type = "ambiguous"

    # REMOVED 2026-09-24 — a second, broader frame-preservation guard sat here
    # (`_has_frame_continuation_evidence`). Its anaphora and continuation-shape arms
    # duplicated the measured backstop above; its third arm — "the message shares a
    # content word with the frame, so it continues the frame" — is the positive
    # direction of `_mentions_frame_subject`, which is NOT what that helper establishes.
    # Its own docstring defines the INVERSE test: a message sharing no word with the
    # frame cannot be about the data, used to confirm a smalltalk verdict. Read
    # forwards it makes any new question that happens to name the same table a
    # follow-up, which then carries the previous turn's filters into it.
    # Measured: it regressed the pre-existing
    # tests/test_clarification_flow.py::test_a_message_that_is_not_an_answer_is_never_swallowed
    # ("show me the top 5 cities by number of assets" -> followup, frame entity
    # assets_asset) and two of the carry-over guard's own controls. Same-entity
    # refinements without anaphora ("how many active users?") remain unhandled, which
    # is the pre-existing behaviour and wants its own measurement, not this.

    if action == "clarify_reply" and not pending:
        # The classifier can emit this label on any turn. With nothing pending there is
        # no request to complete, and clarify_reply_node would pass the raw message to
        # the engine ungrounded — so treat it as the ordinary follow-up it is.
        #
        # Decided HERE, before the two ungrounded-text backstops below, not after them.
        # They only look at followup/answer, so a relabel that came later walked straight
        # past both: measured 2026-09-26, "Help me understand this" in a chat that had only
        # said "Hi" came back clarify_reply, was relabelled followup after the backstops,
        # and reached the engine with nothing to resolve "this" against (107s, "couldn't
        # map 'understand'").
        logger.info("classify_node: classifier said clarify_reply but nothing is "
                    "pending — handling %r as a followup", message)
        action = "followup"

    if (action in ("followup", "answer") and not frame.get("entity")
            and not topic_hit and _is_bare_referential(message)):
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

    # LAST gate before the engine, after every override above has had its say. Four
    # conditions, all required — each one exists to stop this from ever withholding a
    # real question:
    #   · the flag is on (default OFF);
    #   · the turn was heading to the engine at all;
    #   · there is NO active frame — a follow-up like "only the ones in Nagpur" names
    #     no schema word of its own and is grounded by the frame, not by its text.
    #     Gating those would break every refinement in the product;
    #   · the message names nothing in the data (vocabulary missing → never gates).
    # `message_mentions_data` is the api tier's precomputed answer (one bool); the
    # vocabulary list is the older, more expensive way of asking the same question and
    # is still honoured for callers that pass it. None from BOTH means the check could
    # not be made, and an unanswerable check must never gate.
    _mentions = state.get("message_mentions_data")
    if _mentions is None:
        _vocab = state.get("data_vocabulary")
        _mentions = _mentions_the_data(message, _vocab) if _vocab else None
    # TWO cases, with different costs, so they are gated differently.
    #
    # (1) A FOLLOW-UP WITH NO FRAME. The model called this a continuation, yet there is
    #     nothing to continue — incoherent by construction, whatever the wording — AND
    #     the message names nothing in the scoped data, so the engine has nothing to
    #     work from either. Measured 2026-09-21 over the 282 real data questions in
    #     corpus_real.jsonl: ZERO of them are classified "followup", so this withholds
    #     nothing real. Always on. It closes the case this exists for: ungrounded
    #     referential text ("the other ones") once reached the engine, matched an
    #     unrelated table, and returned a real person's name and email address.
    #
    # (2) A NEW QUESTION that names nothing in the data ("what is the meaning of life").
    #     Same evidence, higher cost: the same measurement withholds 6 of 282 real
    #     questions (2.1% — all "rows in the catalog", where "catalog" lives in a source
    #     outside the tested scope). Stays behind the flag, default OFF, unchanged.
    # `last_result`, NOT just the frame. Measured 2026-09-21 against the real document
    # source (docs_contracts, 179 chunks): a RAG answer produces no frame at all —
    # harvest_frame needs a table and an explain.data_used block, and a retrieval
    # result has neither — so "no frame" is the PERMANENT state there, and this gate
    # withheld "what about for a full time employee" right after it had correctly
    # answered "what is the notice period" from the employee handbook. A legitimate
    # follow-up, refused.
    #
    # "Nothing has been answered yet" is the condition actually wanted, and
    # `last_result` is exactly that: memory_write_node sets it on an answered turn even
    # when the harvest produces no frame. So a document conversation keeps its
    # follow-ups, and "the other ones" typed into a session that has answered nothing
    # is still caught.
    _answered_before = bool(state.get("last_result") or state.get("engine_result"))
    if (action == "followup" and not frame.get("entity") and not topic_hit
            and not _answered_before and _mentions is False):
        logger.info("classify_node: a follow-up in a session that has answered nothing, "
                    "naming nothing in the scoped data — answering without the engine: "
                    "%r", message)
        action = "no_match"
    elif (_GROUNDING_GATE_ENABLED and action == "answer"
            and not frame.get("entity")
            and _mentions is False):
        logger.info("classify_node: message names nothing in the scoped data — "
                    "answering without the engine: %r", message)
        action = "no_match"


    # A PENDING CLARIFICATION COMES FROM A TURN THAT FAILED. Completing it means gluing
    # that failed question onto this message, and the engine parses the result as one
    # question. Measured 2026-09-24: after "distribution of properties by furnishing"
    # (answered), "list 5 ledger with highest amount" (refused, which armed the slot),
    # the next message "only the Nagpur ones" reached the engine as
    #   "list 5 ledger with highest amount for only the Nagpur ones"
    # and failed — the user had moved on, and the dead question took their new one with it.
    #
    # The model's own delta says which reading it is. When it calls this turn a
    # CONTINUATION of the frame — refine/replace/remove/drill — there is a live, ANSWERED
    # query to continue, and continuing it is a coherent reading while gluing it to a
    # failed one is not. So the frame wins and the dead slot is dropped.
    #
    # Both conditions are required. Without a frame there is nothing to continue, so the
    # pending request is still the better reading; and a delta of new_topic is not
    # evidence of a continuation, so it is left alone.
    #
    # `ambiguous` is the model saying it could not place the turn — no evidence either
    # way — and leaving it to the pending slot made the outcome depend on the model's
    # label: measured 2026-09-24, the SAME "only the Nagpur ones" after a refused
    # "only the ones with a swimming pool" came back `replace` in one run (drill kept)
    # and `ambiguous` in another (glued to the dead question, sent with no context, and
    # refused). For `ambiguous` only, the message's own shape decides: a narrowing
    # follow-up continues the frame; anything else still answers the clarification.
    elif (action == "clarify_reply" and pending and frame.get("entity")
            and (delta_type in ("refine", "replace", "remove", "drill_down", "drill_up")
                 or (delta_type in ("ambiguous", None)
                     and _CONTINUATION_SHAPE_RE.match(message or "")))):
        logger.info("classify_node: a clarification was pending from a FAILED turn, but "
                    "%r continues the live frame (%s on %r) — dropping the pending "
                    "request and handling this as a followup",
                    message, delta_type, frame.get("entity"))
        action = "followup"
        pending = None

    # The same reading, reached by evidence this layer gathers rather than by the
    # supervisor's own delta. `new_topic`/`ambiguous` really are not evidence of
    # anything on their own (which is why the branch above leaves them alone) — but a
    # CONFIRMED back-reference to the live frame is: a message that stands in for the
    # frame's records with a pronoun is continuing THAT question, not supplying a bare
    # value to complete a dead one.
    #
    # Measured in the same 25-turn run: 5 turns (5, 6, 7, 9, 13 — "What is its current
    # market status?", "Where is it located?", "How many bedrooms does it have?",
    # "Which of these has the lowest price per square foot?", "Which properties are no
    # longer available?") arrived here as `clarify_reply` with a slot armed by an
    # EARLIER refused turn, and were glued onto that dead question.
    #
    # `_is_clarification_answer` has already run, far above, and returns immediately for
    # every shape it is certain IS a bare value answering the question — so nothing that
    # reaches this line was recognisable as one, and the risk of stealing a genuine
    # clarification answer is bounded by that.
    elif (action == "clarify_reply" and pending and frame.get("entity")
            and delta_type in ("new_topic", "ambiguous") and _carryover()):
        logger.info("classify_node: a clarification was pending from a FAILED turn, but "
                    "%r carries the live frame's subject over (%r) — dropping the "
                    "pending request and handling this as a followup",
                    message, frame.get("entity"))
        action = "followup"
        delta_type = "ambiguous"
        pending = None

    # Do not expose classification as user-facing "thinking" for small talk.
    # The classifier must run for uncanned small talk (for safety: a real data
    # question must not be swallowed), but once the final action is known there
    # is no useful work for the user to watch. Emitting before the SLM call made
    # messages such as "okay"/"hola" show a thinking step even though the graph
    # immediately took the smalltalk -> END branch. Emit only after all
    # smalltalk/data overrides have settled the final action.
    if action != "smalltalk":
        _emit(config, "supervisor_classify", "Understanding your message...")

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
        "delta_field": delta_field,
        "delta_value": delta_value,
        # Same per-turn reset reasoning as `resolved_query` above: only
        # context_resolve_node/clarify_reply_node set this, and every route that skips
        # them would otherwise report the PREVIOUS turn's understanding as this one's.
        "context_used": None,
        # Same per-turn reset, and for a failure with the same shape: context_resolve_node
        # is the only node that sets this, so on a route that skips it (clarify_reply,
        # runtime_context) the PREVIOUS turn's context stayed in the checkpoint and was
        # sent to the engine as if it described this turn — observed carrying a failed
        # question's `user_message` into the turn after it.
        "conversation_context": None,
        # Cleared unless the branch above handed the turn to clarify_reply. A slot that
        # survives a turn it did not answer is what compounded.
        "pending_clarification": (state.get("pending_clarification") or {})
        if action == "clarify_reply" else {},
        # Only when the turn is still a follow-up after every override above — nothing else
        # may act on it. memory_read_node clears it at the start of every turn.
        "topic_restore": topic_hit if action == "followup" else None,
    }


def smalltalk_node(state: ChatState) -> dict:
    """Direct reply for greetings/thanks/chit-chat — engine bypassed entirely."""
    message = state["message"]
    reply = _canned_smalltalk_reply(message)
    if reply is None and _SMALLTALK_LLM_REPLY:
        # Off by default since 2026-09-21. classify_node has ALREADY established that
        # this turn is smalltalk; a second call only chooses the WORDS, and it was
        # measured costing 5-20s of the 12.3s a typo'd greeting took end to end — on a
        # turn whose whole point is that it needs no work. It also drifted: "hola" came
        # back as "¡Hola! ... preguntas sobre数据分析", Spanish and Chinese mixed.
        # Set CHATBOT_SMALLTALK_LLM_REPLY=1 to restore it.
        reply = call_slm(
            build_smalltalk_system_prompt(),   # built fresh each call so "today" is always current
            message,
            max_tokens=60,
            model=CHATBOT_CLASSIFY_MODEL,
            purpose="smalltalk",
        )
    if not reply:
        # Tone-neutral, because this is reached by "okay" and "thnks" as well as by a
        # greeting the patterns could not spell-match.
        reply = SMALLTALK_FALLBACK_REPLY
    return {
        "reply_text": reply,
        "needs_clarification": False,
        "resolved_query": "",
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def classify_with_entry_gate(state: ChatState, config: RunnableConfig) -> dict:
    """classify_node plus the Turn Entry Gate's reporting. Registered as the graph's
    "classify" node so the gate observes EVERY path without classify_node's six early
    returns each having to remember to carry the fields.

    It adds no decision of its own — `entry_decision` reads the action classify_node
    already chose. The only reason this wrapper exists is that the gate must be
    observable (which tier settled the turn, and what it cost) and classify_node
    returns from six different places.
    """
    started = time.perf_counter()
    result = classify_node(state, config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    action = result.get("action") or ""
    decision = entry_decision(action, bool(state.get("history")))
    # Which tier settled it. Under ~2ms nothing reached the network, so the turn was
    # decided by L0 — measured rather than inferred from which branch ran, which would
    # have to be threaded out of all seven of them.
    if elapsed_ms < 2.0:
        path = "L0_DIRECT" if action == "smalltalk" else "L0_FASTPATH"
    else:
        path = "SLM_GATE"

    logger.info("entry_gate: path=%s requires_context=%s requires_veda=%s "
                "classification_latency_ms=%.1f action=%s",
                path, decision["requires_context"], decision["requires_veda"],
                elapsed_ms, action or "?")
    return {**result, "entry_path": path,
            "classification_latency_ms": round(elapsed_ms, 2), **decision}


def _fmt_filters(frame: dict) -> str:
    parts = [f"{f['field']} {f.get('operator', 'equals')} {f['value']}"
             for f in (frame.get("filters") or [])
             if f.get("field") and f.get("value") is not None]
    return ", ".join(parts)


def _fmt_ranking(frame: dict) -> str:
    bits = []
    for order in (frame.get("order_by") or []):
        if order.get("field"):
            bits.append(f"sorted by {order['field']} "
                        f"({'highest' if order.get('desc') else 'lowest'} first)")
    if frame.get("limit"):
        bits.append(f"limited to {frame['limit']} rows")
    return ", ".join(bits)


def _source_name(state: ChatState, frame: dict) -> str:
    """Display name for the source the last answer came from — the api tier resolved
    it once, per request, for sources the caller is authorised for (apps/query/
    scope.py::source_profiles_for). Falls back to nothing rather than to a bare id."""
    profile = (state.get("source_profiles") or {}).get(str(frame.get("source_id")))
    return (profile or {}).get("name") or ""


def no_match_node(state: ChatState) -> dict:
    """The message names nothing in the scoped data — say so at once, and say what CAN
    be answered, instead of spending a full engine round-trip on a refusal.

    Names the real connected sources and what they hold (source_profiles, already
    resolved per request for sources the caller is authorised for) so the reply is a
    route rather than a dead end. States no fact about the data itself."""
    profiles = state.get("source_profiles") or {}
    described = []
    for profile in profiles.values():
        name = (profile or {}).get("name")
        if not name:
            continue
        tags = [t for t in ((profile or {}).get("domain_tags") or []) if t]
        described.append(f"{name} ({', '.join(tags)})" if tags else name)

    if described:
        reply = ("I can only answer questions about your connected data, and I can't see "
                 "anything in that message that matches it. You're connected to "
                 + ", ".join(described)
                 + " — ask me something about it and I'll go and look.")
    else:
        reply = ("I can only answer questions about your connected data, and I can't see "
                 "anything in that message that matches it. Try naming what you want to "
                 "know about — a total, a count, a list, or a trend over time.")
    return {
        "reply_text": reply,
        "status": "no_match",
        "needs_clarification": False,
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def recall_node(state: ChatState) -> dict:
    """Answer a question about the CONVERSATION rather than about the data — no engine
    call, no SQL, no model call.

    Every fact here was harvested from an already-executed, already-validated result
    (chatbot/memory/frame.py::harvest_frame) and is quoted back verbatim. Nothing is
    re-derived and nothing is generated, so this can neither invent a number nor
    disagree with the answer the user is looking at. Memory is not written: nothing was
    executed this turn, so there is no new evidence and the frame must not advance
    (memory_write_node's evidence-only rule) — the graph routes this node to
    format_reply directly."""
    kind = state.get("recall_kind") or "trail"
    frame = state.get("frame") or {}
    if kind == "topics":
        # The topic index, as memory_read_node loaded it — already filtered to this turn's
        # authorised sources, so a revoked source's topic is never listed.
        _topics = [e for e in (state.get("topic_index") or []) if isinstance(e, dict)]
        if _topics:
            _lines = []
            for _i, _e in enumerate(_topics, 1):
                _src = _source_name(state, _e)
                _lines.append(f"{_i}. {memory_topics.display_name(_e)}"
                              + (f" — {_src}" if _src else ""))
            _reply = ("Here's what we've looked at so far, most recent first:\n"
                      + "\n".join(_lines)
                      + "\n\nSay \"go back to …\" with one of these and I'll pick it up again.")
        elif state.get("previous_topics"):
            _lines = []
            for _i, _e in enumerate(state.get("previous_topics") or [], 1):
                _src = _source_name(state, _e)
                _lines.append(f"{_i}. {memory_topics.display_name(_e)}"
                              + (f" — {_src}" if _src else ""))
            _reply = ("Nothing yet in this conversation. In your earlier conversations "
                      "we looked at:\n" + "\n".join(_lines)
                      + "\n\nSay \"continue with …\" or \"go back to …\" with one of "
                        "these and I'll pick it up again with today's data.")
        else:
            _reply = ("We haven't looked at anything yet in this conversation — ask me "
                      "something about your data to get started.")
        return {"reply_text": _reply, "status": "answered", "needs_clarification": False,
                "resolved_query": "", "engine_unavailable": False,
                "history": _turn_delta(state, _reply)}
    if kind == "presentation_not_applicable":
        _what = frame.get("entity_display") or frame.get("entity") or "a document"
        _reply = (f"There's nothing to chart here — this answer came from {_what}, "
                  f"which is written text rather than rows and figures. Ask me "
                  f"something else about it and I'll look it up.")
        return {"reply_text": _reply, "needs_clarification": False,
                "resolved_query": "", "engine_unavailable": False,
                "history": _turn_delta(state, _reply)}
    if kind == "shape_not_applicable":
        _what = frame.get("entity_display") or frame.get("entity") or "a document"
        _reply = (f"That one doesn't apply here — this answer came from {_what}, "
                  f"not a table of rows, so there's no ordering or row count to change. "
                  f"Ask me something else about it and I'll look it up.")
        return {"reply_text": _reply, "needs_clarification": False,
                "resolved_query": "", "engine_unavailable": False,
                "history": _turn_delta(state, _reply)}
    if kind == "access_revoked":
        _reply = ("I can't show that — the earlier answer came from data that is no longer "
                  "in your access or in this question's scope, so I've set it aside. Ask a "
                  "new question about the data you can see, or contact your Admin to "
                  "request access.")
        return {"reply_text": _reply, "needs_clarification": False,
                "resolved_query": "", "engine_unavailable": False,
                "history": _turn_delta(state, _reply)}
    if kind == "already_on_document":
        _what = frame.get("entity_display") or frame.get("entity") or "that document"
        _reply = (f"We're already on {_what} — ask me anything about it and I'll look "
                  f"it up.")
        return {"reply_text": _reply, "needs_clarification": False,
                "resolved_query": "", "engine_unavailable": False,
                "history": _turn_delta(state, _reply)}
    if kind == "drill_up_empty":
        # Reached with or without a frame: nothing has been narrowed, so there is no
        # level to return to either way.
        return {
            "reply_text": ("There's nothing to go back to — nothing has been narrowed "
                           "down yet in this conversation."),
            "needs_clarification": False, "resolved_query": "",
            "engine_unavailable": False,
            "history": _turn_delta(state, "There's nothing to go back to — nothing has "
                                          "been narrowed down yet in this conversation."),
        }
    if kind == "nothing" or not frame.get("entity"):
        reply = ("I don't have an earlier question to describe — nothing has been run "
                 "in this conversation yet. Ask me something about your data and I'll "
                 "be able to show you exactly how I answered it.")
        return {"reply_text": reply, "status": "answered", "needs_clarification": False,
                "engine_unavailable": False, "history": _turn_delta(state, reply)}
    entity, display = frame.get("entity"), frame.get("entity_display")
    # "Assets (assets_asset)" — the business name plus the raw table it came from, so a
    # reader can tie the answer to the schema. A DOCUMENT frame has no raw table: the
    # dataset IS the entity, and printing both gave "Samta-Employee Handbook April 2026
    # (Samta-Employee Handbook April 2026)".
    if display and entity and display != entity:
        dataset = f"{display} ({entity})"
    else:
        dataset = display or entity or ""
    source = _source_name(state, frame)
    in_source = f" in {source}" if source else ""
    filters, ranking = _fmt_filters(frame), _fmt_ranking(frame)
    rows = frame.get("last_row_count")

    if kind == "query":
        understanding = frame.get("understanding")
        if memory_frame.is_document_frame(frame):
            # `understanding` means two different things depending on the head that
            # produced it. On a SQL answer it restates the QUESTION ("Find the 100
            # cheapest sale listings"), which is exactly what this recall wants. On a
            # retrieval answer it describes the ANSWERING PROCESS — measured
            # 2026-09-21, "what did i ask earlier" replied "Your last question was:
            # Answered from Samta-Employee Handbook April 2026 using 5 relevant
            # passages", which is not a question and not what the user asked.
            #
            # The user's own words are already kept, verbatim, in the episodic buffer
            # (chatbot/memory/store.py, written by memory_write_node on answered turns
            # only) — so quote those instead of paraphrasing an answer back as a
            # question. Falls back to the dataset line when the buffer is empty.
            asked = [t.get("content") for t in (state.get("episodic") or [])
                     if t.get("role") == "user" and t.get("content")]
            reply = (f"You asked: {asked[-1]}" if asked
                     else f"Your last question read {dataset}{in_source}.")
        else:
            reply = (f"Your last question was: {understanding}" if understanding
                     else f"Your last question read {dataset}{in_source}.")
    elif kind == "sql":
        sql = frame.get("last_sql")
        if sql:
            reply = f"This is the SQL I ran:\n\n```sql\n{sql}\n```"
        elif memory_frame.is_document_frame(frame):
            # There is no SQL because none was run. Saying so and naming what WAS read
            # is the honest answer; "it came back without one" reads like a fault.
            reply = (f"I didn't run any SQL for that — the answer came from "
                     f"{dataset}{in_source}.")
        else:
            reply = "I don't have the SQL for the last answer — it came back without one."
    elif kind == "table":
        if dataset and memory_frame.is_document_frame(frame):
            reply = f"I read {dataset}{in_source} — a document, not a table."
        else:
            reply = f"I read {dataset}{in_source}." if dataset else (
                "I don't have a table recorded for the last answer.")
    elif kind == "filters":
        reply = f"Filters applied: {filters}." if filters else (
            "No filters were applied — the last answer covered every row.")
    elif kind == "rows":
        reply = (f"That returned {rows} row(s)." if rows is not None
                 else "I don't have a row count for the last answer.")
    else:                                        # "trail"
        steps = [f"read {dataset}{in_source}"] if dataset else []
        if filters:
            steps.append(f"applied {filters}")
        if ranking:
            steps.append(ranking)
        if rows is not None:
            steps.append(f"returned {rows} row(s)")
        reply = ("For that answer I " + ", ".join(steps) + "."
                 if steps else "I don't have the details of the last answer recorded.")

    return {
        "reply_text": reply,
        "status": "answered",
        "needs_clarification": False,
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def represent_node(state: ChatState) -> dict:
    """Re-render the PREVIOUS turn's result in the format just asked for — no engine
    call, no SQL, no model call.

    The rows are already in hand: re-running the pipeline to redraw them costs a full
    round-trip and, worse, can return a DIFFERENT result set than the one on screen
    (the underlying data can change between turns, and a non-deterministic plan can
    too), so "chart that" could redraw something other than "that". Memory is
    deliberately NOT written here — nothing new was executed, so there is no new
    evidence and the QueryFrame must not advance a turn (see memory_write_node's
    evidence-only rule); the graph routes this node straight to format_reply.

    `viz_override` rides along on the result dict so the api tier renders the format
    the user named instead of its own recommendation (apps/chat/services.py)."""
    kind = state.get("viz_override") or "chart"
    previous = dict(state.get("last_result") or state.get("engine_result") or {})
    previous["viz_override"] = kind
    row_count = len(previous.get("rows") or [])
    reply = {
        "table": f"Here are the same {row_count} row(s) as a table.",
        "csv": f"Here are the same {row_count} row(s) — use the table's own export "
               "to download them.",
    }.get(kind, f"Here are the same {row_count} row(s), drawn as a {kind}.")
    return {
        "engine_result": previous,
        "status": "answered",
        "reply_text": reply,
        "needs_clarification": False,
        "engine_unavailable": False,
        "history": _turn_delta(state, reply),
    }


def _frame_still_authorised(frame: dict, state: ChatState) -> bool:
    """Is the source this frame was harvested from still granted to the caller THIS turn?

    RBAC is already resolved fresh on every turn (apps/chat/views.py computes
    permitted_source_ids / resolve_query_scope / compute_data_scope before the service is
    built), and every engine-bound turn carries the resulting data_scope. But the frame is
    not metadata — it holds filter VALUES read out of the customer's data, the executed
    SQL, the row count, and (for a re-render) the result rows. Three paths answer from it
    without reaching the engine at all, so `data_scope` is never applied to them:
    recall_node, represent_node and context_resolve_node.

    Without this check, a grant revoked between turns left "what SQL did you run" and
    "show that as a table" serving content from the withdrawn source for as long as the
    memory lived — seven days. Authorisation from a previous turn is not authorisation.

    Fails OPEN only where there is nothing to decide: a frame with no recorded source
    (written before source pinning existed, or by a path that carries none) and a turn
    with no resolved scope (a non-HTTP caller, e.g. the CLI) both pass, because neither
    supplies a fact to compare. A frame WITH a source and a turn WITH a scope must match.
    """
    source_id = frame.get("source_id")
    if source_id is None:
        return True
    authorised = state.get("source_ids")
    if authorised is None:
        return True                    # no scope resolved at all (CLI / non-HTTP caller)
    if not authorised:
        # An EMPTY scope is a decision, not an absence: the view resolved the caller's
        # grants and found none. Treating it like "no scope supplied" let the single most
        # likely revocation shape — the last grant withdrawn — sail straight through the
        # guard and keep serving a seven-day-old frame.
        logger.warning("_frame_still_authorised: the caller has an EMPTY authorised "
                       "scope — discarding remembered source_id=%r", source_id)
        return False
    try:
        return int(source_id) in {int(s) for s in authorised}
    except (TypeError, ValueError):
        logger.warning("_frame_still_authorised: unreadable source ids "
                       "(frame=%r, scope=%r) — discarding the frame", source_id, authorised)
        return False


def _authorised_results(tenant: str, session_id: str, state: ChatState) -> list:
    """The earlier-result history, filtered to THIS turn's grants. Unlike a frame, an entry
    with no recorded source is dropped, not kept: a result holds shown row labels and
    values, and one whose source cannot be proven in scope is not shown to anything
    (measured 2026-09-26 by test_memory_rbac_guard — source-less entries survived a
    revocation)."""
    return [e for e in MemoryStore.read_results(tenant, session_id)
            if isinstance(e, dict) and e.get("source_id") is not None
            and _frame_still_authorised(e, state)]


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
        # ...and in the one memory that crosses sessions: "start over" asked to forget
        # this conversation, so an earlier-session lookup must not bring it back.
        if state.get("user_id"):
            MemoryStore.write_user_sessions(
                tenant, state.get("user_id"),
                memory_topics.without_session(
                    MemoryStore.read_user_sessions(tenant, state.get("user_id")), session_id))
        logger.info("memory_read_node: deterministic reset match, message=%r", message)
        # `memory_reset` ends the turn here (classify_node reads it first). Without it
        # the wipe was undone by its own turn: "start over" carried on to the engine,
        # which searched for a table named by those words, answered something, and
        # memory_write_node then wrote a BRAND NEW frame — measured, the frame came
        # back at version 1 with an entity in it, so the reset had no lasting effect.
        # Sending "start over" to a SQL engine was never meaningful anyway.
        return {"frame": {}, "drill_stack": [], "episodic": [], "memory_reset": True,
                "memory_revoked_source": None, "result_history": [],
                "pending_clarification": {}, "last_result": {}, "comparison": {},
                "result_reference": None, "topic_index": [], "topic_restore": None,
                "previous_topics": [],
                "memory_in": turn_telemetry.memory_summary({}, [], None, [], [])}

    # Type-guarded, not just falsiness-guarded: a Redis key holding a JSON string or
    # list (a bad write, a manual edit, a format change) previously raised
    # AttributeError out of this node and surfaced as HTTP 500. Memory is an
    # optimisation; unreadable memory means "no memory", never a failed turn.
    # Per-source memory (2026-09-18). The frame belongs to the source it was harvested
    # from, so a turn scoped to source B reads B's topic and leaves A's alone — before
    # this, one session-wide frame meant the newer source silently erased the older one.
    # A turn that names no source reads whichever source answered last (the store's
    # active pointer), which is what a single-source deployment and the CLI both get.
    source_id = state.get("source_id")
    frame = MemoryStore.read_frame(tenant, session_id, source_id)
    frame = frame if isinstance(frame, dict) else {}
    if not isinstance(frame.get("filters"), list):
        frame = {**frame, "filters": []}
    stack = MemoryStore.read_stack(tenant, session_id, source_id)
    stack = stack if isinstance(stack, list) else []
    # Scoped to THIS turn's grants. The buffer is session-wide by design (one
    # conversation, one thread), but each entry records the source that produced it, so a
    # revoked grant drops its own entries without discarding the rest of the thread —
    # the same precision _frame_still_authorised gives the frame.
    episodic = MemoryStore.read_episodic(tenant, session_id,
                                         authorised_source_ids=state.get("source_ids"))
    comparison = MemoryStore.read_comparison(tenant, session_id) or {}
    episodic = episodic if isinstance(episodic, list) else []

    if not _frame_still_authorised(frame, state):
        # The remembered source is no longer in THIS turn's authorised scope. Everything
        # derived from it goes with it — see _frame_still_authorised for why the frame is
        # data, not metadata. Handled here, once, because every downstream path
        # (recall_node, represent_node, context_resolve_node) reads what this node
        # returns; guarding them individually would leave the next one to be added
        # unguarded by default.
        _revoked = frame.get("source_id")
        logger.warning(
            "memory_read_node: remembered source_id=%r is outside this turn's authorised "
            "scope %r — discarding that source's analytical memory",
            _revoked, state.get("source_ids"))
        # Scoped to the revoked source, not the whole session (2026-09-18). Losing the
        # grant on one source is not a reason to throw away the user's work on another,
        # and per-source keys make that distinction expressible. The guard itself is
        # unchanged and still runs on every turn through this node — it got NARROWER in
        # what it destroys, never in what it catches. A frame with no recorded source has
        # no scoped key to delete, so that case still wipes the session.
        MemoryStore.reset(tenant, session_id, source_id=_revoked)
        # The reset above already dropped the revoked source's topics from the index; the
        # other sources' topics are still the user's to return to, still filtered here.
        _topics = [e for e in MemoryStore.read_topics(tenant, session_id)
                   if _frame_still_authorised(e, state)]
        # engine_result too. classify_node's presentation check reads
        # `last_result or engine_result`, so clearing only the first left the PREVIOUS
        # turn's checkpointed rows to resurrect: recall correctly refused after a
        # revocation while "show that as a table" still redrew the withdrawn source's
        # data. Everything the revoked source produced goes together or the guard has a
        # hole in it.
        return {"frame": {}, "drill_stack": [], "episodic": [], "memory_reset": False,
                "memory_revoked_source": _revoked,
                "result_history": _authorised_results(tenant, session_id, state),
                "last_result": {}, "pending_clarification": {}, "result_reference": None,
                "engine_result": {}, "sql": None, "rows": None, "status": None,
                "topic_index": _topics, "topic_restore": None,
                "previous_topics": _previous_topics(state, tenant, session_id),
                "memory_in": turn_telemetry.memory_summary({}, [], None, _topics, [])}
    # The previous answer's row identities — read only AFTER the authorisation check above,
    # and keyed to the frame's OWN source, so a reference can never outlive the frame it
    # describes or be read from a source this turn may not see.
    _ref_source = frame.get("source_id", source_id)
    reference = (MemoryStore.read_reference(tenant, session_id, _ref_source)
                 if frame.get("entity") else None)
    # Cleared EXPLICITLY on every non-reset turn. The checkpointer persists the whole
    # state across turns, so a flag only ever set True stays True: live test
    # 2026-09-17, the turn after "start over" was itself answered as a reset, and so
    # would every turn after that until the session ended.
    # The topic index, filtered by the SAME guard as the frame, entry by entry: a topic
    # holds filter values read out of the data, so a topic from a source this turn may not
    # see is invisible here — and therefore can never be matched, listed or restored.
    topics = [e for e in MemoryStore.read_topics(tenant, session_id)
              if _frame_still_authorised(e, state)]
    # `topic_restore` is cleared on EVERY turn (the reset path above too). Only
    # classify_node's final return sets it, and routes that skip that return
    # (clarify_reply, recall, represent) would otherwise hand memory_write_node the
    # PREVIOUS turn's restore.
    _previous = _previous_topics(state, tenant, session_id)
    # The source the conversation was last ON, when it is not in this turn's scope. The
    # frame read above is keyed by this turn's nominal source (apps/chat/views.py passes
    # source_ids[0]), so a narrowed scope reads an EMPTY frame instead of the revoked one —
    # nothing leaks, but nothing tells a "show me those results" why it has nothing to show.
    # Flag only: that source's memory is not read, and not deleted either (a deliberately
    # narrowed scope should get it back when widened again).
    _revoked_active = None
    if not frame.get("entity") and state.get("source_ids"):
        _active = MemoryStore.active_source(tenant, session_id)
        if _active is not None and str(_active) not in {str(x) for x in state["source_ids"]}:
            _revoked_active = _active
    _results = _authorised_results(tenant, session_id, state)
    return {"frame": frame, "drill_stack": stack, "episodic": episodic,
            "comparison": comparison, "memory_reset": False,
            "memory_revoked_source": _revoked_active, "result_history": _results,
            "result_reference": reference or None,
            "topic_index": topics, "topic_restore": None,
            "previous_topics": _previous,
            "memory_in": turn_telemetry.memory_summary(frame, stack, reference, topics,
                                                       _previous)}


def _previous_topics(state: ChatState, tenant: str, session_id: str) -> list:
    """Topics from this user's OTHER sessions, filtered by the SAME guard as the frame and
    the in-session index, entry by entry — a topic holds filter values read out of the
    data, so one from a source this turn may not see is invisible, unmatchable and
    unrestorable. Empty with no user (CLI) or on any read failure."""
    user_id = state.get("user_id")
    if not user_id:
        return []
    return [e for e in memory_topics.previous_topics(
                MemoryStore.read_user_sessions(tenant, user_id), session_id)
            if _frame_still_authorised(e, state)]


def _context_used(frame: dict, delta_type: str, delta_field: str, delta_value: str,
                  resolved: str, message: str) -> Optional[dict]:
    """What the conversation layer carried into this turn, for the user to see.

    Every field is something the turn ALREADY produced — the frame it merged and the
    delta it applied. Nothing here is re-derived, inferred or asked of a model, so it
    cannot claim an understanding the turn did not actually act on. That is the whole
    point: when a follow-up carries the wrong context, this is the only place the user
    could notice before reading the answer and believing it.

    Returns None when nothing was carried — a first question, or a turn whose resolved
    query is just the message. Showing "we understood: <your own words>" would be noise.
    """
    if not frame or not frame.get("entity"):
        return None
    if not resolved or resolved.strip() == (message or "").strip():
        return None
    carried = {
        "entity": frame.get("entity_display") or frame.get("entity"),
        "filters": [f"{f.get('field')} {f.get('operator', 'equals')} {f.get('value')}"
                    for f in (frame.get("filters") or [])
                    if f.get("field") and f.get("value") is not None],
        "source_id": frame.get("source_id"),
    }
    changed = None
    if delta_type in ("replace", "remove") and delta_field:
        changed = {"operation": delta_type, "field": delta_field}
        if delta_value:
            changed["value"] = delta_value
    elif delta_type in ("refine", "drill_down", "drill_up", "compare"):
        changed = {"operation": delta_type}
    return {"carried": carried, "changed": changed, "resolved_query": resolved}


_NO_RESULT_TO_POINT_AT = ("There's no earlier list for me to pick that row from — ask the "
                          "question first, and then you can point at a row of the answer.")
_ROWS_NOT_PICKABLE = ("I can't pick out a single row of that answer by its position. Name "
                      "the one you mean instead — for example \"only the DEBIT ones\".")


def _reference_refusal(message: str, *, delta_type: Optional[str], reason: str) -> dict:
    """End a turn whose reference cannot be resolved, honestly and WITHOUT the engine.

    Routed by graph._route_after_resolve to ask_clarification as a terminal answer (no
    pending slot is armed), and nothing is written — so whatever the conversation had
    before this turn is exactly what it has after it."""
    logger.info("context_resolve_node: result reference not resolvable for %r — %s",
                message, reason)
    return {"resolved_query": message, "delta_type": delta_type,
            "engine_result": {"status": "refuse", "route": "reference",
                              "refuse_reason": reason},
            "status": "refuse", "conversation_context": None, "context_used": None}


_NO_TOPIC_TO_RETURN_TO = "There's no earlier topic like that for me to go back to."


def _return_to_topic(state: ChatState, restore: dict) -> dict:
    """Restore a remembered topic as this turn's CANDIDATE frame + drill stack and replay it.

    Nothing is written here. The candidate reaches Redis only through memory_write_node, and
    only if the replay is answered — the same commit-on-success rule every other turn
    follows, so a failed restore leaves the conversation (and the topic index) exactly as
    it was. Always re-executed; the snapshot holds no rows to replay.

    What is sent is the boundary's usual pair: the user's OWN words — the topic's root
    question, recorded when they asked it (same principle as a drill-up replay) — and the
    snapshot's state as structured ConversationContext. A snapshot with no filters is its
    root question verbatim, so it goes out with no context at all, exactly as it did the
    first time (the same root-replay rule context_resolve_node applies to a drill-up)."""
    message = state["message"]
    if restore.get("kind") == "ambiguous":
        # Several remembered topics fit. Asking is the only honest answer: picking one is a
        # guess the user would only discover by reading a wrong answer. Terminal (no pending
        # slot), no engine call, nothing written.
        names = []
        for e in restore.get("candidates") or []:
            src = _source_name(state, e)
            names.append(memory_topics.display_name(e) + (f" in {src}" if src else ""))
        reason = ("More than one earlier topic matches that — which one do you want to go "
                  "back to? " + "; or ".join(names))
        logger.info("context_resolve_node: return-to-topic is ambiguous for %r — asking "
                    "(%d candidates)", message, len(names))
        return _reference_refusal(message, delta_type=None, reason=reason)

    entry = restore.get("topic") or {}
    candidate = memory_topics.frame_from_entry(entry)
    base_query = str(candidate.get("base_query") or "").strip()
    if restore.get("kind") == "document" or memory_frame.is_document_frame(candidate):
        return _return_to_document(state, restore, entry, candidate, base_query)
    if restore.get("kind") == "continue":
        return _continue_table_topic(state, entry, candidate)
    if not candidate.get("entity") or not base_query \
            or not _frame_still_authorised(candidate, state):
        # memory_read_node only ever hands over authorised, replayable entries, so this is
        # defence in depth — and it fails closed, never toward the engine.
        logger.warning("context_resolve_node: topic %r is not restorable this turn — "
                       "refusing rather than guessing", memory_topics.topic_key(entry))
        return _reference_refusal(message, delta_type=None, reason=_NO_TOPIC_TO_RETURN_TO)

    # The optimistic lock in MemoryStore.write_frame compares the version STORED under the
    # topic's source key, which is not the snapshot's version once that source has moved
    # on to another topic. Carry the live one, so an answered restore commits.
    tenant = state.get("tenant") or "default"
    session_id = state.get("session_id") or ""
    live = state.get("frame") or {}
    if str(live.get("source_id")) != str(candidate.get("source_id")) or not live.get("entity"):
        live = MemoryStore.read_frame(tenant, session_id, candidate.get("source_id")) or {}
    candidate.update({"version": live.get("version") or 0,
                      "turn_index": live.get("turn_index") or 0,
                      "tenant": tenant, "session_id": session_id})
    stack = [dict(lvl) for lvl in (entry.get("drill_stack") or [])]

    # Only filters the boundary can actually send (a column AND a value) make the replay
    # context-dependent. A value-less one ("transaction_type IS NOT NULL") is part of what
    # the root question itself produced, and re-asking the root reproduces it.
    carry = any(f.get("column") and f.get("value") is not None
                for f in candidate.get("filters") or [])
    conv_ctx = ConversationContext.from_frame(candidate, base_query, carry_state=carry,
                                              operation=_RESTORE_OPERATION)
    display = candidate.get("entity_display") or candidate.get("entity")
    used = {
        "carried": {
            "entity": display,
            "filters": [f"{f.get('field')} {f.get('operator', 'equals')} {f.get('value')}"
                        for f in candidate.get("filters") or []
                        if f.get("field") and f.get("value") is not None],
            "source_id": candidate.get("source_id"),
        },
        "changed": {"operation": "return_to_topic", "topic": display},
        "resolved_query": base_query,
    }
    logger.info("context_resolve_node: returning to topic %r (%d filter(s), stack depth %d) "
                "— replaying %r, context=%s", memory_topics.topic_key(entry),
                len(candidate.get("filters") or []), len(stack), base_query,
                "none" if conv_ctx.is_empty() else conv_ctx.to_payload())
    return {"resolved_query": base_query, "delta_type": _RESTORE_OPERATION,
            "frame": candidate, "drill_stack": stack,
            "conversation_context": conv_ctx.to_payload(), "context_used": used}


def _continue_table_topic(state: ChatState, entry: dict, candidate: dict) -> dict:
    """A values-only follow-up asked while the conversation sits on a document: it narrows
    the most recent TABLE topic instead (see classify_node). The topic's snapshot becomes the
    candidate frame and the message goes out as an ordinary refine of it — the same
    structured context any follow-up on that table carries. Nothing is written here."""
    message = state["message"]
    if not candidate.get("entity") or not _frame_still_authorised(candidate, state):
        return _reference_refusal(message, delta_type=None, reason=_NO_TOPIC_TO_RETURN_TO)
    tenant = state.get("tenant") or "default"
    session_id = state.get("session_id") or ""
    live = MemoryStore.read_frame(tenant, session_id, candidate.get("source_id")) or {}
    candidate.update({"version": live.get("version") or 0,
                      "turn_index": live.get("turn_index") or 0,
                      "tenant": tenant, "session_id": session_id})
    stack = [dict(lvl) for lvl in (entry.get("drill_stack") or [])]
    conv_ctx = ConversationContext.from_frame(candidate, message, operation="refine")
    display = candidate.get("entity_display") or candidate.get("entity")
    logger.info("context_resolve_node: values-only follow-up on a document — narrowing the "
                "latest table topic %r instead", memory_topics.topic_key(entry))
    return {"resolved_query": message, "delta_type": "refine",
            "frame": candidate, "drill_stack": stack,
            "conversation_context": conv_ctx.to_payload(),
            "context_used": {"carried": {"entity": display, "source_id": candidate.get("source_id"),
                                         "filters": [f"{f.get('field')} {f.get('operator', 'equals')} "
                                                     f"{f.get('value')}"
                                                     for f in candidate.get("filters") or []
                                                     if f.get("value") is not None]},
                             "changed": {"operation": "refine"}, "resolved_query": message}}


def _return_to_document(state: ChatState, restore: dict, entry: dict, candidate: dict,
                        base_query: str) -> dict:
    """Back into a document conversation: the remembered document becomes this turn's
    candidate frame, and the question goes to the engine anchored in it, with the frame's
    route carried as ConversationContext so the engine stays on the document lane.

    Two ways here. A question that NAMES the document ("what is the fee for repair in the
    maintenance policy?") is asked as written — it is its own question. A pure return
    ("go back to the maintenance policy") has no question of its own, so it re-asks the
    last one the conversation put to that document, the same principle as a table topic's
    replay. Nothing is written here; memory_write_node commits only an answered turn."""
    message = state["message"]
    asked = message if restore.get("kind") == "document" else base_query
    if not candidate.get("entity") or not asked \
            or not _frame_still_authorised(candidate, state):
        logger.warning("context_resolve_node: document topic %r is not restorable this "
                       "turn — refusing rather than guessing", memory_topics.topic_key(entry))
        return _reference_refusal(message, delta_type=None, reason=_NO_TOPIC_TO_RETURN_TO)
    tenant = state.get("tenant") or "default"
    session_id = state.get("session_id") or ""
    live = state.get("frame") or {}
    if str(live.get("source_id")) != str(candidate.get("source_id")) or not live.get("entity"):
        live = MemoryStore.read_frame(tenant, session_id, candidate.get("source_id")) or {}
    candidate.update({"version": live.get("version") or 0,
                      "turn_index": live.get("turn_index") or 0,
                      "tenant": tenant, "session_id": session_id})
    resolved = memory_frame.render_frame_as_query(candidate, asked, "refine",
                                                  referential=True)
    conv_ctx = ConversationContext.from_frame(candidate, resolved)
    display = candidate.get("entity_display") or candidate.get("entity")
    used = {"carried": {"entity": display, "filters": [],
                        "source_id": candidate.get("source_id")},
            "changed": {"operation": "return_to_topic", "topic": display},
            "resolved_query": resolved}
    logger.info("context_resolve_node: back into document %r — asking %r",
                candidate.get("entity"), resolved)
    return {"resolved_query": resolved, "delta_type": None,
            "frame": candidate, "drill_stack": [],
            "conversation_context": conv_ctx.to_payload(), "context_used": used}


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
    delta_field = state.get("delta_field") or ""
    delta_value = state.get("delta_value") or ""

    # RETURN TO AN EARLIER TOPIC — decided by classify_node from the authorised topic index.
    # First, because the topic may belong to a source whose frame this turn did not load
    # (the frame below can be empty while the index is not).
    _restore = state.get("topic_restore") or None
    if _restore:
        return _return_to_topic(state, _restore)

    if not frame.get("entity"):
        # NO FRAME AT ALL — the session has never had an answered analytical turn, so
        # there is no context to resolve this message against. It goes to the engine
        # exactly as the user wrote it.
        #
        # This used to spend a model call asking FOLLOWUP_SYSTEM_PROMPT to rewrite the
        # message into a self-contained question. Measured 2026-09-21: a first-turn
        # question cost TWO calls (classify, then this) — and the second one had
        # nothing to add, because a turn with no prior analytical context is
        # self-contained by definition. The rewrite could only paraphrase, and a
        # paraphrase of "how many assets are there" is a new chance to lose a word the
        # engine parses as data.
        #
        # The genuinely referential case ("what about the other one" with no frame) is
        # NOT handled here and must not be: classify_node's own backstop downgrades it
        # to smalltalk before it ever reaches this node, precisely so ungrounded text
        # cannot be forwarded to the engine.
        # ...except a message that only POINTS at a row of a result ("show the 9th one").
        # It names nothing else, so with no result behind it there is nothing it can mean,
        # and passing it through unchanged was measured 2026-09-25 to reach the engine and
        # come back narrating row 9 of an unrelated users table. Refused, not guessed.
        if memory_reference.result_pointer(message):
            return _reference_refusal(message, delta_type="new_topic",
                                      reason=_NO_RESULT_TO_POINT_AT)
        logger.info("context_resolve_node: no frame yet — passing %r through unchanged "
                    "(no rewrite call)", message)
        return {"resolved_query": message, "delta_type": "new_topic"}

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
        delta_type, _slot_candidates, _dfield = classify_delta(frame, message, episodic)
        delta_field = _dfield or delta_field
        delta_value = delta_value or (_slot_candidates[0] if _slot_candidates else "")
        # _slot_candidates itself is intentionally not threaded into the merge
        # (render_frame_as_query only ever uses `frame` + the verbatim
        # `message`, never a partially-extracted slot value — nothing gets
        # added that isn't either an old proven fact or a substring the user
        # actually typed). classify_delta()/parse_delta_response already
        # applied the H3 confidence gate before delta_type reaches here.

    if delta_type == "drill_up" and drill_stack:
        drill_stack = memory_frame.pop_drill(drill_stack)
        frame = memory_frame.rebuild_frame_from_stack(frame, drill_stack)

    # REPLACE / REMOVE happen HERE, in Python, on the structured frame — before any text
    # is rendered. The model contributed three small strings (delta_type, delta_field,
    # one grounded value); the mutation itself is a dict operation that cannot drift the
    # way a 7B model restating the whole context would. apply_context_delta refuses any
    # delta it cannot bind to a filter the frame actually holds, so a wrong or invented
    # classification degrades to "carry the context unchanged", never to a wrong filter.
    # A shape change ("make it top 10", "by month instead", "don't sort by amount") is
    # decided deterministically, from the message itself — see
    # chatbot/memory/frame.py::detect_shape_delta for why this is not a model call.
    # Asked ONLY when the model did not already produce a usable replace/remove, and
    # never when it called the turn a new topic, a comparison or a drill: correcting a
    # known operation bias is not the same as overruling a topic decision.
    # A measure ADDITION ("also include profit"). Deterministic and grounded against the
    # table's own measure columns — the closed delta set has no "add", and the model is
    # never asked for one, so this cannot put an invented column into memory. Tried
    # before the shape delta because "also include profit" is an addition, not a
    # re-shaping, and detect_shape_delta would not claim it either way.
    #
    # This updates MEMORY only. Measures are deliberately not rendered into the resolved
    # query (see frame.py::_describe_frame), so the engine learns about "profit" from the
    # user's own words, which pass through verbatim; what the frame gains is knowing the
    # question now has two measures when the NEXT turn re-shapes it.
    # `refine` only. Its definition in the prompt IS addition ("ADDS a filter/grouping,
    # keeps everything in the frame"), so the model has already drawn the add-vs-replace
    # line — "what about profit" comes back `replace` and never reaches here. `ambiguous`
    # is excluded deliberately: it means the classify call failed or was unsure, which is
    # not evidence of an addition.
    if delta_type == "refine":
        _added_measure = memory_frame.detect_measure_addition(frame, message)
        if _added_measure:
            frame = memory_frame.add_measure(frame, _added_measure)
            logger.info("context_resolve_node: measure addition — %r grounded against "
                        "the table's own measures (no SLM call): %r",
                        _added_measure, message)

    shape_delta = False
    if delta_type in ("refine", "ambiguous"):
        _shape = memory_frame.detect_shape_delta(frame, message)
        if _shape:
            delta_type, delta_field, delta_value = _shape
            shape_delta = True
            logger.info("context_resolve_node: deterministic shape delta — %s %s=%r "
                        "(no SLM call): %r", delta_type, delta_field, delta_value, message)

    if delta_type in ("replace", "remove"):
        _before = frame
        frame = memory_frame.apply_context_delta(
            frame, delta_type, field=delta_field, value=delta_value, message=message)
        # apply_context_delta returns the CALLER'S OWN object when it declines a delta
        # and a new dict whenever it acts, so identity is an exact did-anything-change
        # signal. It replaced a filters-length comparison, which could only see the one
        # slot: a shape delta ("don't sort by amount" -> order_by) leaves the filter
        # count untouched and was therefore reported as having matched nothing.
        _applied = frame is not _before
        logger.info("context_resolve_node: %s field=%r value=%r — %s",
                    delta_type, delta_field, delta_value,
                    "applied" if _applied else "declined, context carried unchanged")
        if delta_type == "remove" and not _applied:
            shape_delta = False
            # Nothing was actually removed — the named field/slot is not one the frame holds.
            # `remove` renders context-ONLY (the user's words are a navigation trigger,
            # not data), so leaving it as `remove` here threw the request away and
            # re-ran the previous query verbatim: the user asked for a change, got the
            # same answer back, and nothing said their request had been ignored.
            # Downgrade to a plain refinement so their own words still reach the engine.
            logger.info("context_resolve_node: remove field=%r matched nothing in the "
                        "frame — keeping the user's message instead of re-running the "
                        "previous query", delta_field)
            delta_type = "refine"

    # "new_topic"/"refine"/"drill_down"/"drill_up"/"compare" all merge
    # deterministically; "ambiguous" (judgment OR timeout) passes the message
    # through untouched — no second SLM call, see docstring above.
    # classify_node's own action label, reused — no extra model call. It is the only
    # continuation signal that works on a DOCUMENT frame, where delta_type is measurably
    # not one (see frame.py::_render_document_query). Routing into this node is
    # history-based rather than label-based (chatbot/graph.py::_route_after_classify),
    # so `action` still carries its real value here: an ordinary self-contained question
    # arrives as "answer" and is left unanchored.
    referential = state.get("action") == "followup"
    _comparison = state.get("comparison") or {}
    if _comparison and delta_type not in ("new_topic",) and not shape_delta:
        # A comparison is active and this turn continues it. Both sides go into the
        # resolved query — collapsing to one of them is the specific failure this
        # structure exists to prevent, and it is what happened before it existed:
        # "compare with 2024" rendered as an ordinary refinement of the 2025 frame, so
        # the comparand had nowhere to live and the next turn saw a single context.
        resolved = memory_frame.render_comparison_as_query(_comparison, message)
        logger.info("context_resolve_node: comparison active (%s) — both sides carried: "
                    "%r -> %r", _comparison.get("dimension"), message, resolved)
    elif delta_type == "compare" and (delta_value or memory_frame.detect_comparison_target(message)):
        # `delta_value` is the model's own grounded slot; measured 2026-09-22 that it
        # comes back empty on real "compare" classifications more often than not, so the
        # deterministic extractor (frame.py::detect_comparison_target) is the fallback,
        # never the other way — a value the model DID ground and verify is trusted first.
        delta_value = delta_value or memory_frame.detect_comparison_target(message)
        # FIRST turn of a comparison — no ComparisonContext exists yet (that is built in
        # memory_write_node, AFTER the engine answers). render_frame_as_query has no
        # branch for "compare" and falls through to its generic tail, which sent the
        # comparison INSTRUCTION itself to the engine:
        #   "compare that with Mumbai (for Assets (assets_asset), pune)"
        # — "compare" read as a possible column name (the engine asked "is 'compare' a
        # column name?"), the turn was refused, memory_write_node never runs on a refused
        # turn, and no comparison was ever built. Measured 2026-09-22: a full 5-turn live
        # attempt built nothing, and this is why.
        #
        # `delta_value` is the comparand ("Mumbai") — already grounded VERBATIM against
        # the user's message by the same gate replace/refine use
        # (memory/classify.py:94 includes "compare" in that check). Rendered against the
        # entity ALONE, filters cleared: this is exactly the shape a plain "what about
        # Mumbai" replace already renders and is proven to work — one value, not two
        # (the old Pune filter is deliberately dropped here, or the query would ask about
        # Pune AND Mumbai on the same field at once). The PRIMARY side (Pune) is not
        # lost — it is still sitting in `frame`, and memory_write_node builds the
        # comparison from that prev_frame plus whatever this turn's answer harvests.
        _entity_only = {**frame, "filters": []}
        resolved = memory_frame.render_frame_as_query(_entity_only, delta_value, "refine")
        logger.info("context_resolve_node: comparison turn (no context yet) — comparand "
                    "%r rendered alone: %r -> %r", delta_value, message, resolved)
    else:
        # THE BOUNDARY. What reaches the engine as the QUERY is only ever text the user
        # actually typed — this turn's message, or (for the two navigation deltas below)
        # their own pre-drill question, recorded verbatim when they asked it. The
        # remembered state travels beside it as ConversationContext, structured.
        #
        # This replaced `render_frame_as_query`, which glued the frame's description onto
        # the message: "only the debit ones (for Single Financial Transactions
        # (accounts_generalledger))". `Single Financial Transactions` is the engine's own
        # display label for the table, and veda/validation.py::qualifier_completeness —
        # whose stated contract is "every content token THE USER NAMED must appear in the
        # SQL" — had no way to know the user never said it, so it refused on `financial`
        # (which substring-matches the real column financial_year_id) while the user's
        # own word `debit` was a real value with 476 rows behind it.
        #
        # `remove` and `drill_up` carry no data of their own: their words name what to
        # STOP doing ("remove the year filter", "go back"), and the engine parses every
        # word of a query as data. They therefore replay the user's OWN earlier question
        # rather than this turn's trigger words — still never text this layer invented.
        if memory_frame.is_document_frame(frame):
            # DOCUMENT frames keep the existing rendering. The contamination this change
            # removes is a SQL-path problem — qualifier_completeness, the anchor and the
            # value arbiter all live there, and the structured context has no meaning for
            # a RAG answer (its "entity" is a document name, not a table). The document
            # anchoring in _render_document_query was built and live-verified separately;
            # bypassing it here would trade one measured bug for another. Recorded as a
            # known remaining gap rather than silently changed.
            resolved = memory_frame.render_frame_as_query(
                frame, message, delta_type, shape_delta=shape_delta, referential=referential)
        elif delta_type in ("remove", "drill_up") and not shape_delta:
            resolved = (frame.get("base_query") or "").strip() or message
        else:
            resolved = message

    # RESULT REFERENCE ("the second one", "its price"). Resolved against the identities of
    # the rows the user is looking at (chatbot/memory/reference.py), deterministically, and
    # only on a turn already placed as a continuation: an ordinal inside a self-contained
    # question ("the first transaction of 2024") names its own subject, not a row. The
    # selected row(s) join the CANDIDATE frame as ordinary structured filters — the id, or
    # the group's values — so the engine re-queries them with this turn's scope and current
    # data; nothing from the stored result is replayed. Nothing is written here: like every
    # other candidate change, it reaches memory only if the turn is answered.
    #
    # A message that is ONLY a pointer ("the 2nd one") is itself evidence of continuation —
    # it has no subject of its own — so it resolves whatever label the classifier gave it
    # (measured: "then the 2nd one" came back answer/ambiguous and was refused on "then").
    _ref_hit, _ref_terms = None, []
    # WHICH result the reference is to. The current one — unless the message names an
    # earlier one ("the 1st one from the price list", "in the earlier result"), matched on
    # the words of the question that produced it (memory/reference.py::earlier_result).
    _turn_ref = state.get("result_reference")
    _switched_ref = None
    _earlier = memory_reference.earlier_result(state.get("result_history"), message, _turn_ref)
    if _earlier and _earlier[0] == "refuse":
        return _reference_refusal(message, delta_type=delta_type, reason=_earlier[1])
    if _earlier and _earlier[0] == "ref" and delta_type != "drill_up":
        _entry = _earlier[1]
        if (str(_entry.get("entity")) != str(frame.get("entity"))
                or str(_entry.get("source_id")) != str(frame.get("source_id"))):
            # An earlier result on ANOTHER topic: that topic becomes this turn's frame.
            _topic = next((t for t in (state.get("topic_index") or [])
                           if str(t.get("entity")) == str(_entry.get("entity"))
                           and str(t.get("source_id")) == str(_entry.get("source_id"))), None)
            if _topic is None:
                return _reference_refusal(message, delta_type=delta_type,
                                          reason="That earlier list is no longer available "
                                                 "to pick from — could you ask it again?")
            _live = MemoryStore.read_frame(state.get("tenant") or "default",
                                           state.get("session_id") or "",
                                           _topic.get("source_id")) or {}
            frame = {**memory_topics.frame_from_entry(_topic),
                     "version": _live.get("version") or 0,
                     "turn_index": _live.get("turn_index") or 0,
                     "tenant": state.get("tenant") or "default",
                     "session_id": state.get("session_id") or ""}
            drill_stack = [dict(lvl) for lvl in (_topic.get("drill_stack") or [])]
        _turn_ref = _switched_ref = _entry
        _ref_terms = list(_earlier[2])
        logger.info("context_resolve_node: reference qualified to an earlier result %r (%s)",
                    _entry.get("question"), _entry.get("entity"))
    _pointer = memory_reference.result_pointer(message) or (
        memory_reference.parse_ordinal(message) if _switched_ref else None)
    _the_one = memory_reference.points_at_the_one_row(_turn_ref, message, frame)
    # A record named by its label as shown ("details of One & Only House") is evidence in
    # itself, like a pointer: the label came from the result on screen.
    # So is a SET of shown rows ("the first three", "those four"): it names no subject of
    # its own. Both are covered by selects_rows.
    _named_row = memory_reference.selects_rows(_turn_ref, message, frame)
    # Never on a drill-up: "go back" navigates the drill path, it points at no row.
    if ((referential or _pointer or _the_one or _named_row) and delta_type != "drill_up"
            and not memory_frame.is_document_frame(frame)):
        _ref_hit = memory_reference.resolve_reference(
            _turn_ref, message, frame=frame, referential=True)
        if _pointer and _ref_hit is None:
            # A pointer with nothing current to point at. Two different truths, told apart:
            # rows WERE shown but they are not individually pickable (the engine stated no
            # row key — measured 2026-09-25, edge E16: "the second one" after a 2-row list was
            # told "there's no earlier list", which the user could see was false), or there is
            # genuinely no current result to point at.
            _shown = (state.get("last_result") or state.get("engine_result") or {}).get("rows")
            _ref_hit = ("refuse", _ROWS_NOT_PICKABLE if _shown else _NO_RESULT_TO_POINT_AT)
    if _ref_hit and _ref_hit[0] == "refuse":
        # A position that is not there. Refused honestly WITHOUT reaching the engine — the
        # nearest row would be a guess — and routed as a terminal answer (no pending slot):
        # the reference survives untouched, so the user's next "the 2nd one" resolves.
        return _reference_refusal(message, delta_type=delta_type, reason=_ref_hit[1])
    if _ref_hit and _ref_hit[0] == "filters":
        _picked = _ref_hit[1]
        _ref_terms = _ref_terms + list(_ref_hit[2] if len(_ref_hit) > 2 else [])
        _cols = {f["column"] for f in _picked}
        if (_turn_ref or {}).get("kind") == "rows":
            # A record picked by KEY replaces whatever an earlier pick left on the SHOWN
            # columns ("details of Infotech Tower" → "and Shivsai Apartment?" kept
            # project_name = infotech tower beside the new id → 0 rows, measured
            # 2026-09-26). The key alone pins the row, so this can never widen it.
            _cols |= set(((_turn_ref or {}).get("texts") or {}).keys())
        frame = {**frame, "filters": [f for f in (frame.get("filters") or [])
                                      if f.get("column") not in _cols] + _picked}
        if delta_type in ("new_topic", "ambiguous"):
            delta_type = "refine"          # it narrows the current result, by evidence
        logger.info("context_resolve_node: result reference resolved to %s",
                    [(f["column"], f["value"]) for f in _picked])

    # ROOT replay: a remove/drill-up that leaves NO filters is the user's original question,
    # asked again — so it goes out exactly as it did the first time, with no context. Sent
    # WITH context it was treated as a context-dependent turn (no verified-cache lookup, a
    # different planner branch) and came back "lists rows without grouping" where the first
    # ask had answered (measured 2026-09-24, depth 1 → 0). Same distinction the previous
    # rendering drew (VEDA_DRILLDOWN_10LEVEL_FEASIBILITY.md §K1): replay base_query only when
    # nothing remains; otherwise send the remaining context.
    _root_replay = (delta_type in ("remove", "drill_up") and not shape_delta
                    and not memory_frame.is_document_frame(frame)
                    and not (frame.get("filters") or [])
                    and bool((frame.get("base_query") or "").strip())
                    and resolved == (frame.get("base_query") or "").strip())

    conv_ctx = ConversationContext.from_frame(
        frame, resolved,
        # A new topic is self-contained by definition and an `ambiguous` turn is one the
        # classifier could not place — neither may drag remembered state along. This is
        # the same rule the previous rendering applied when it returned the message
        # unchanged for both.
        carry_state=(not _root_replay
                     and (delta_type not in ("new_topic", "ambiguous") or referential)),
        operation=delta_type,
        resolved_terms=_ref_terms,
    )

    logger.info("context_resolve_node: delta_type=%s query=%r context=%s",
                delta_type, resolved,
                "none" if conv_ctx.is_empty() else conv_ctx.to_payload())
    _out_extra = {"result_reference": _switched_ref} if _switched_ref else {}
    return {"resolved_query": resolved, "delta_type": delta_type,
            "frame": frame, "drill_stack": drill_stack, **_out_extra,
            "conversation_context": conv_ctx.to_payload(),
            "context_used": _context_used(frame, delta_type, delta_field, delta_value,
                                          resolved, message)}


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
    # The head that answered ("deterministic" / "tier2" / "rag" / "nosql" / ...) lives
    # on item0, one level ABOVE res0, and was discarded here — so the chat front door
    # had no route to audit (traceability Part 19). Carried onto res0 under a
    # namespaced key so it cannot collide with a pipeline field. setdefault, never
    # overwrite: if a pipeline ever sets it itself, that value wins.
    if item0.get("route"):
        res0.setdefault("_route", item0["route"])
    # A DEFINITE refusal from the router (no access to the source that can answer,
    # NO_MATCH, ...) is minted by veda_core/veda_hybrid.py::_run_coordinator via
    # MultiResult.single(..., refuse_reason=...): it never runs the SQL pipeline, so
    # item0["result"] is null and the reason lives one level UP, on item0 itself.
    # res0 therefore ends up {} and ask_clarification_node found no feedback ->
    # every such refusal surfaced as the generic "Could you clarify what you're
    # asking about?" on the chat path, while /api/v1/query (which reads item0
    # directly) showed the real reason. Lift them onto res0 — the ONE place res0 is
    # assembled — so the graph downstream can surface them. Never clobbers a
    # pipeline that set its own.
    for _k in ("refuse_reason", "route"):
        if _k not in res0 and item0.get(_k) is not None:
            res0[_k] = item0[_k]
    status = res0.get("status")
    if status is None:
        # RAG/hybrid/nosql heads carry no pipeline-level status of their own
        # (veda_core/veda/pipeline.py::_done, which mints "answered"/"exec_error"/
        # "access_denied"/etc., is a SQL/Tier-1/Tier-2-only concept) — their res0
        # is just {answer, chunks, citations, ...} with no "status" key at all, so
        # the old `res0.get("status", "error")` always fell through to "error" and
        # a real answer got discarded as a generic clarify. Fall back to the
        # SubResult-level status (item0["status"], "ok"|"refused"|"error") ONLY
        # when res0 has none of its own — the docstring above still holds for
        # every route that DOES set one.
        status = "answered" if item0.get("status") == "ok" else "error"
        # Written BACK onto res0, not merely returned. Measured 2026-09-21 against the
        # real docs_contracts source: a document conversation had NO memory at all —
        # memory_write_node checks state["status"] == "answered" and proceeds, then
        # harvest_frame RE-CHECKS engine_result["status"], finds the key absent, and
        # returns None. Nothing was ever stored, so recall, re-present, drill and every
        # other frame-based feature were dead on the one source that answers reliably.
        # Normalising the derived status here, where res0 is assembled, is exactly what
        # this function already does for `cols`.
        res0["status"] = status
    return res0, status


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

    # The remembered state travels BESIDE the query, never inside it (see
    # context_resolve_node). `flags` is the request's existing extension point — it was
    # already accepted by inference/routes/hybrid.py and dropped there; it now carries
    # this. Absent/empty = exactly the previous behaviour, so every caller that builds no
    # context (clarify_reply, first turns, smalltalk) is unaffected.
    _conv_ctx = state.get("conversation_context") or None
    _flags = {"conversation_context": _conv_ctx} if _conv_ctx else None

    try:
        for kind, data in client.stream_hybrid_query(
            query,
            flags=_flags,
            source_id=state.get("source_id"),
            source_ids=state.get("source_ids"),
            tenant=state.get("tenant"),
            request_id=state.get("request_id"),
            data_scope=state.get("data_scope"),
            source_profiles=state.get("source_profiles"),
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

    # Recorded FIRST, before the harvest can bail out. harvest_frame returns None when
    # the result carries no explain block (a federated answer, or a server-side
    # business_explain failure), and the early return below then left `last_result`
    # holding an OLDER turn's rows — so a later "as a pie chart" charted data the user
    # was no longer looking at. The hazard is staleness, not failure: this node only
    # runs on an answered turn either way.
    _last_result = engine_result

    harvested = memory_frame.harvest_frame(engine_result)
    if harvested:
        # WHICH SOURCE ANSWERED — not which source the request happened to be pinned
        # to. Those used to be treated as the same fact; they are not.
        #
        # apps/chat/views.py resolves the request's nominal `source_id` as
        # `source_ids[0]` — the caller's FIRST authorised source, fixed for the whole
        # turn, regardless of which source in that scope actually produced the answer.
        # For a single-source deployment (or the CLI, which pins one explicitly) that
        # is harmless: source_ids[0] IS the only source. It stops being true the moment
        # a session is authorised for more than one source and a turn answers from
        # any source other than the first — measured 2026-09-22 live, through the real
        # API with no source pinned: a document answer (from docs_contracts) was
        # recorded under source_ids[0]'s key (homzhub), and a later SQL answer from the
        # SAME key would have silently overwritten it — reintroducing, via source
        # mislabeling, the exact "one frame erases another" failure per-source scoping
        # was built to prevent.
        #
        # The engine already knows better: `explain.sources` (business_explain.py's v2
        # extension, gated on EXPLAIN_V2_ENABLED) is built from `build_data_sources`,
        # which names a source ONLY with proof of participation — execution records
        # first, federation second, the routing decision only when it actually chose
        # (never under shadow-mode observation), never a candidate that was merely
        # considered. That is real evidence; state["source_id"] is a request-level
        # default.
        #
        # Used only when it is UNAMBIGUOUS — exactly one source named. Two or more
        # means a genuinely federated answer, which this architecture does not yet
        # have a multi-source frame to own (see PM_LOG/memory audit — reported as a
        # boundary, not worked around here); falling back to the old behavior there is
        # not a regression, since a federated turn already writes no entity (a route
        # name is not a business entity) and this line only decides which key it is
        # filed under. Zero named sources (v2 off, or nothing yet resolved) falls back
        # identically, so an environment without the extension behaves exactly as
        # before this change.
        _engine_sources = ((engine_result.get("explain") or {}).get("sources") or [])
        _engine_source_ids = {str(s.get("id")) for s in _engine_sources
                              if isinstance(s, dict) and s.get("id")}
        if len(_engine_source_ids) == 1:
            harvested["source_id"] = next(iter(_engine_source_ids))
        else:
            harvested["source_id"] = state.get("source_id")
    if not harvested:
        # business_explain failed server-side (already logged there) or the
        # result had no explain block — skip the write, the user's answer is
        # unaffected, memory just doesn't advance this turn — but the result the user
        # IS looking at is still recorded, or a presentation follow-up would redraw an
        # older one. For the same reason the row reference is CLEARED: it describes the
        # previous answer, and "the second one" must never mean a row of a result the
        # user is no longer looking at.
        MemoryStore.write_reference(
            tenant, session_id, None,
            source_id=(state.get("frame") or {}).get("source_id") or state.get("source_id"))
        return {"last_result": _last_result, "result_reference": None}

    prev_frame = state.get("frame") or {}
    prev_stack = state.get("drill_stack") or []
    delta_type = state.get("delta_type") or "new_topic"

    # Same continuation signal context_resolve_node anchors on, for the same reason:
    # delta_type is not informative on a document frame. A follow-up that still drew on
    # the document being discussed keeps it, rather than moving to whichever document
    # the engine happened to list first.
    # A FOLLOW-UP THAT LEFT THE CONVERSATION'S TABLE DOES NOT REPLACE IT. "answered" only
    # says the engine returned something, not that it answered THIS follow-up. Measured
    # 2026-09-26 (audit, scenario A): on a Noida/furnished frame, "Go back" was answered by
    # the federated route with "No matching rows", memory filed entity None with a
    # value-less `city` filter, and every later turn wandered into the employee handbook.
    # A follow-up on a table frame that comes back with no table of its own (federated,
    # documents) or with zero rows is not written: the conversation stays where it was,
    # exactly as after a refusal. New questions and topic returns are unaffected.
    _rows = engine_result.get("rows")
    if (state.get("action") == "followup" and not state.get("topic_restore")
            and prev_frame.get("entity") and not memory_frame.is_document_frame(prev_frame)
            and (not harvested.get("entity") or memory_frame.is_document_frame(harvested)
                 or (isinstance(_rows, list) and not _rows))):
        logger.info("memory_write_node: follow-up on %r answered with %s — not replacing the "
                    "conversation's state", prev_frame.get("entity"),
                    "no rows" if isinstance(_rows, list) and not _rows
                    else f"no table of its own ({harvested.get('entity')!r})")
        return {"last_result": _last_result}

    harvested = memory_frame.stabilise_document_entity(
        prev_frame, harvested, referential=state.get("action") == "followup")

    # The mirror case: a conversation on a TABLE whose follow-up came back from the
    # documents. That turn is `answered`, so it writes — and the frame moved to a document
    # nobody asked about, taking every later turn with it (measured 2026-09-23). The
    # entity is held; everything else the turn actually produced is still recorded.
    harvested = memory_frame.keep_entity_on_lane_change(
        prev_frame, harvested, referential=state.get("action") == "followup")

    # And the case neither of those covers: a turn the classifier could not place at all.
    # `ambiguous` carries no context, so the engine answers the bare fragment and lands
    # wherever that routes — which `is_topic_switch` below (and inside
    # merge_frame_post_execution) would otherwise read as a deliberate change of subject.
    harvested = memory_frame.hold_subject_on_unplaced_turn(
        prev_frame, harvested, delta_type, prev_stack)

    new_frame = memory_frame.merge_frame_post_execution(
        prev_frame, harvested, delta_type, tenant, session_id)

    reset = delta_type == "new_topic" or memory_frame.is_topic_switch(prev_frame, harvested)
    if reset:
        new_stack: list = []
    else:
        # EVERY turn, whatever the classifier called it, is decided by the evidence below:
        # the level pushed is the filter the executed query ADDED relative to the previous
        # one. A turn labelled `drill_down` used to take its own branch, push_drill(), which
        # pushes "the LAST filter in the list" — and that list's order is the SQL walker's,
        # not the order the user narrowed in. Measured 2026-09-25 (DRILLDOWN_QUERY_MATRIX
        # scenario 2): Pune, then EAST, labelled drill_down, pushed Location a second time
        # (stack ['Location', 'Location']) while the filters said Direction Facing = east.
        # Evidence also settles the case the label cannot: a new value on a field already
        # constrained is a replacement, not a deeper level.
        # A turn the classifier called something else (in practice almost always `refine`) still
        # drilled in if the executed query added a filter the previous one did not. Without this
        # the DrillStack stayed empty for every real narrowing — "only the open ones", "just the
        # Repair ones" — so drill_up had nothing to pop. Deterministic and evidence-based: it
        # reads the filters the SQL actually ran, never the classifier's label.
        _added = memory_frame.newly_added_filter(prev_frame, harvested)
        if _added is not None:
            new_stack = memory_frame.push_drill_level(prev_stack, _added)
        else:
            # No NEW field was constrained, but an existing one may now hold a different
            # value ("what about Mumbai" after Pune). Re-point that level instead of
            # deepening the path: the stack records how far in the user has drilled, and
            # a replacement does not change that. Leaving it stale mattered — drill_up
            # rebuilds the frame FROM the stack, so the old value came back as if the
            # replacement had never happened.
            new_stack = prev_stack
            for _f in (new_frame.get("filters") or []):
                _prev_val = next((p.get("value") for p in (prev_frame.get("filters") or [])
                                  if memory_frame._same_field(p.get("field"), _f.get("field"))),
                                 None)
                if _prev_val is not None and str(_prev_val) != str(_f.get("value")):
                    new_stack = memory_frame.update_drill_level(
                        new_stack, _f.get("field"), _f.get("value"))
    # A filter the user REMOVED must leave the drill stack with it. rebuild_frame_from_stack
    # derives filters FROM the stack on the next "go back", so a stale level put the
    # removed filter straight back: remove City -> "go back" -> City=Pune returns.
    # ... EXCEPT when the classifier had no opinion. `ambiguous` is what parse_delta_response
    # returns for a genuine judgment call AND for a model call that failed or timed out —
    # chatbot/llm.py returns None uniformly for both — so it is not evidence that the user
    # left the drill. Measured 2026-09-24, 3 runs out of 3: after a refused turn the next
    # follow-up came back `ambiguous` (the SLM host was failing DNS resolution), no context
    # was carried, the answer therefore had no filters, and this prune then erased a live
    # 1-level path. Keeping the stack costs nothing if the turn really was a new subject —
    # the `reset` branch above already empties it on a genuine topic switch.
    if delta_type != "ambiguous":
        new_stack = [lvl for lvl in new_stack
                     if any(memory_frame._same_field(lvl.get("dimension"), f.get("field"))
                            for f in (new_frame.get("filters") or []))]

    # The question the user asked BEFORE any narrowing, kept so "go back" can replay it
    # when it pops the last drill level. Recorded on any answered turn that carries no
    # filters — that IS the drill root — and never on a drill_up itself, whose message
    # ("go back") is a navigation trigger, not a question. Carried forward otherwise so
    # drilling in does not erase it.
    #
    # A RETURN TO AN EARLIER TOPIC keeps that topic's own root question: this turn's message
    # ("go back to the properties") is navigation, not a question, and the candidate frame
    # context_resolve_node restored already carries the root it replayed.
    #
    # A NEW TOPIC whose first question already carries a filter is its own root. It used to
    # inherit the PREVIOUS topic's base_query through the carry-forward branch (a reset
    # frame has none of its own), so "go back" to the root of the new topic — and, now, a
    # return to it from the topic index — would have replayed a question about a different
    # table.
    _restoring = (state.get("topic_restore") or {}).get("kind") == "restore"
    if _restoring and prev_frame.get("base_query"):
        new_frame["base_query"] = prev_frame["base_query"]
    elif not (new_frame.get("filters") or []) and delta_type != "drill_up":
        new_frame["base_query"] = state.get("message") or ""
    elif reset and delta_type != "drill_up":
        new_frame["base_query"] = state.get("message") or ""
    elif prev_frame.get("base_query") and not new_frame.get("base_query"):
        new_frame["base_query"] = prev_frame["base_query"]

    # COMPARISON. Built when the classifier called this turn a comparison and the
    # previous turn left a frame to compare against — both sides come from frames the
    # ENGINE produced, never from the model's prose. Dropped when the conversation moves
    # to an entity neither side is about, so a stale comparison cannot keep injecting two
    # contexts into an unrelated question.
    _prev_comparison = state.get("comparison") or {}
    _comparison = _prev_comparison
    if delta_type == "compare" and prev_frame.get("entity"):
        _built = memory_frame.build_comparison(prev_frame, new_frame,
                                               turn_index=new_frame.get("turn_index", 0))
        if _built:
            _comparison = _built
            logger.info("memory_write_node: comparison recorded — %s vs %s (dimension=%s)",
                        _built["primary"].get("label"), _built["comparison"].get("label"),
                        _built["dimension"])
    elif memory_frame.comparison_is_stale(_prev_comparison, new_frame):
        logger.info("memory_write_node: comparison dropped — the conversation moved to "
                    "%r, which neither side is about", new_frame.get("entity"))
        _comparison = {}
    if _comparison is not _prev_comparison:
        MemoryStore.write_comparison(tenant, session_id, _comparison or None)

    # The SAME resolved value the harvest above just decided — new_frame carries it
    # via `harvested["source_id"]` (merge_frame_post_execution folds harvested's fields
    # in). Re-reading state.get("source_id") here directly would undo that fix: the
    # FRAME's own source_id field would say the answer came from source 3 while the
    # Redis KEY it gets filed under still said source 2 — content and location
    # disagreeing, which is worse than the original bug, not better. One resolved
    # value, used for the key AND the field, every write below (frame, stack, and the
    # episodic entry this turn contributes) is filed under and stamped with it.
    _source_id = new_frame.get("source_id") or state.get("source_id")
    _committed = MemoryStore.write_frame(
        tenant, session_id, new_frame,
        expected_version=prev_frame.get("version") if prev_frame else None,
        source_id=_source_id)
    MemoryStore.write_stack(tenant, session_id, new_stack, source_id=_source_id)
    MemoryStore.push_episodic_turn(tenant, session_id, state.get("message", ""),
                                   _templated_gist(engine_result), source_id=_source_id)
    # The identities of the rows now on screen — or None, which CLEARS the slot: whatever
    # the user saw before this answer is no longer what "the second one" can mean.
    _reference = memory_reference.build_reference(engine_result, _source_id)
    # ...unless this turn PICKED from that answer ("details of the 3rd one", "the cheapest
    # one", "the first three"). Its own result is the picked row(s); the list the user is
    # still looking at is the one "which one is the cheapest?" or "the 5th one" points at
    # next. Measured 2026-09-26: after "details of the 3rd one" the one-row answer replaced
    # the 10-row list, and "which one has the highest amount?" was asked of that one row.
    _picked_from_shown = bool((state.get("conversation_context") or {}).get("resolved_terms"))
    if _picked_from_shown and state.get("result_reference"):
        # Written back, not just kept: when the pick was from an EARLIER result ("the 1st
        # one from the price list") that result is now the one on screen to point at.
        # WHICH row(s) were picked is recorded too, so the next "what is its carpet area?"
        # means that record — not "one of the N rows" (measured 2026-09-26).
        _reference = dict(state.get("result_reference"))
        _kc = _reference.get("key_column")
        _reference["picked"] = [str(f.get("value")) for f in (state.get("frame") or {}).get("filters") or []
                                if _kc and f.get("column") == _kc and f.get("value") is not None]
        MemoryStore.write_reference(tenant, session_id, _reference, source_id=_source_id)
    else:
        MemoryStore.write_reference(tenant, session_id, _reference, source_id=_source_id)
        if _reference:
            MemoryStore.write_results(tenant, session_id, memory_reference.remember_result(
                MemoryStore.read_results(tenant, session_id), _reference,
                state.get("message") or ""))

    # THE TOPIC INDEX. This topic's snapshot moves to the front; a topic the conversation
    # moved away from keeps the snapshot it had when last answered — that is what "go back
    # to it" restores. Only when the frame itself committed: an aborted write means a
    # concurrent turn won, and the index must not describe a frame that is not in memory.
    # Read-modify-write of the WHOLE list (MemoryStore.read_topics is unfiltered), so topics
    # of sources that are merely outside this turn's scope are kept, not dropped.
    _topics = state.get("topic_index") or []
    _entry = memory_topics.snapshot({**new_frame, "source_id": _source_id}, new_stack)
    if _committed and _entry:
        _stored = memory_topics.upsert(MemoryStore.read_topics(tenant, session_id), _entry)
        MemoryStore.write_topics(tenant, session_id, _stored)
        _topics = [e for e in _stored if _frame_still_authorised(e, state)]
        # SESSION MEMORY (step 6): this session's topic snapshots, filed under the USER, so a
        # later chat can list them ("what did we look at before?") or return to one
        # ("continue the property analysis"). Snapshots only — never rows or answers; a
        # restore always re-executes under the then-current authorisation.
        if state.get("user_id"):
            MemoryStore.write_user_sessions(
                tenant, state.get("user_id"),
                memory_topics.merge_sessions(
                    MemoryStore.read_user_sessions(tenant, state.get("user_id")),
                    memory_topics.session_summary(session_id, _stored)))

    # An answered turn resolves whatever was pending — nothing is left to complete.
    return {"frame": new_frame, "drill_stack": new_stack, "last_result": _last_result,
            "comparison": _comparison, "result_reference": _reference,
            "pending_clarification": {}, "topic_index": _topics}


def reset_node(state: ChatState) -> dict:
    """"Start over" — the session's analytical memory was wiped by memory_read_node and
    the turn ends here. It never reaches the engine: those words name no data, and
    sending them there previously produced an answer whose frame overwrote the very
    memory the user had just asked to clear."""
    reply = ("Cleared — I've forgotten the earlier context. Ask me anything about your "
             "data and we'll start fresh.")
    return {"reply_text": reply, "status": "answered", "needs_clarification": False,
            "engine_unavailable": False, "frame": {}, "drill_stack": [],
            "last_result": {}, "pending_clarification": {},
            "topic_index": [], "topic_restore": None,
            "history": _turn_delta(state, reply)}


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

    # A router-level refusal (no access to the source that can answer, no matching
    # source, or the router's own clarifying question) carries its reason at
    # refuse_reason — lifted onto res0 by _extract_engine_result. Surfacing it
    # verbatim matters most for "no_access": answering a permission denial with
    # "could you clarify what you're asking about?" actively misleads (it reads as
    # "I didn't understand" when the truth is "you're not allowed to see this"), and
    # it hid the denial completely on the chat path while /api/v1/query showed it
    # correctly all along.
    #
    # Matched on ROUTE, not on "has a refuse_reason": every SubResult carries one
    # (veda_hybrid.py::_to_subresult sets it from result["error"]/status), so keying
    # off its presence would surface raw engine codes like "qualifier_dropped" to the
    # user whenever the pipeline built no feedback — strictly worse than the generic
    # question. These three routes are minted ONLY by _run_coordinator, always with a
    # human-readable reason. "clarify" is a real question (keep needs_clarification);
    # "no_access"/"no_match" are terminal — no rephrasing changes the answer.
    # "reference" is context_resolve_node's own: a result position that does not exist
    # ("the 7th one" after 5 rows) — terminal, the reason already says which rows exist.
    _ROUTER_REFUSAL_ROUTES = ("no_access", "no_match", "clarify", "reference")
    refuse_reason = res0.get("refuse_reason")
    _route = res0.get("route")
    router_refusal = bool(refuse_reason) and _route in _ROUTER_REFUSAL_ROUTES
    definite = router_refusal and _route in ("no_access", "no_match", "reference")

    # The engine's own refusal text, when it has one. A refusal that reaches here through the
    # source-agent/coordinator path carries {answer, ok, refuse_reason, status} — the pipeline's
    # rich feedback dict is dropped when the AgentResult is mapped (veda_hybrid.py:318-319 sets
    # result=None for a refusal), but `answer` survives and is still real information
    # ("I couldn't find any data relevant to this question in the sources available to you.").
    # Showing it beats the generic question, which told the user nothing at all. NOT used for a
    # router refusal (that has its own explicit wording) and never over a real feedback text,
    # which is the engine's purpose-built clarifying question.
    _engine_answer = (res0.get("answer") or "").strip()

    feedback = res0.get("feedback")
    if status == "unavailable":
        # A transport/infra failure, NOT the engine declining to answer: call_engine_node sets
        # this for InferenceUnavailable or a stream that drops mid-response, and leaves
        # engine_result EMPTY — so with no feedback and no answer this fell through to
        # "Could you clarify what you're asking about?". That is the one case where the generic
        # question is actively wrong: nothing about the question needs clarifying, the service
        # did not respond, and inviting a rephrase sends the user to fix something that isn't
        # theirs. This node's own docstring already promised this branch; it was never written.
        # Retryable, so say so — and needs_clarification stays False (handled below), because
        # there is no question outstanding.
        question = ("I couldn't reach the query service just now, so I don't have an answer for "
                    "this yet. Please try again in a moment.")
    elif router_refusal:
        question = refuse_reason
    elif feedback:
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
        question = _engine_answer or "Could you clarify what you're asking about?"

    unavailable = status == "unavailable"
    # `definite` refusals are terminal, not questions — don't ask the user to clarify
    # something no rephrasing can fix (and don't leave the UI waiting on an answer).
    _no_clarify = unavailable or definite
    update = {
        "reply_text": question,
        "needs_clarification": not _no_clarify,
        "clarification_question": None if _no_clarify else question,
        "engine_unavailable": unavailable,
        # PENDING CLARIFICATION. A clarifying turn is not "answered", so
        # memory_write_node returns early and no frame is written — which left the
        # user's reply ("2024") with nothing structured to attach to, and it reached the
        # engine as that bare string. Recorded here instead, on the ONE node that knows
        # a question was asked. Carries the UNRESOLVED request verbatim so the next turn
        # can rebuild the whole question rather than send the answer alone.
        #
        # Cleared on: consumption (clarify_reply_node), any answered turn
        # (memory_write_node), and reset (reset_node / memory_read_node).
        "pending_clarification": {} if _no_clarify else {
            "question": question,
            # Keep the ORIGINAL request across repeated clarifications. Re-arming from
            # this turn's resolved_query let each round append to the last, so the
            # stored request grew one clause per turn until the engine choked on it.
            "original_query": ((state.get("pending_clarification") or {}).get("original_query")
                               or state.get("resolved_query") or state.get("message", "")),
            "missing": ((res0.get("feedback") or {}).get("missing")
                        or res0.get("missing") or ""),
            "turn_index": (state.get("frame") or {}).get("turn_index", 0),
        },
    }
    if not unavailable:
        update["history"] = _turn_delta(state, question)
    return update


def clarify_reply_node(state: ChatState) -> dict:
    """TASK 6 — consume a pending clarification and rebuild the ORIGINAL request.

    Deterministic: the stored request is quoted verbatim and the user's answer is quoted
    verbatim; nothing is generated. Sending only the answer ("2024") was the previous
    behaviour and gave the engine a string with no subject at all.

    This node does not call the engine itself — it produces the resolved query and the
    graph carries on to call_engine_node exactly as an ordinary turn would, so routing,
    agents, execution and RBAC are untouched."""
    pending = state.get("pending_clarification")
    pending = pending if isinstance(pending, dict) else {}
    answer = str(state.get("message") or "").strip().rstrip(".")
    original = str(pending.get("original_query") or "").strip().rstrip(".")
    if not original:
        # Nothing to attach to — treat the message as the question it is. Never silently
        # bind an answer-shaped message to an unrelated earlier request.
        logger.info("clarify_reply_node: no pending request to complete — passing %r "
                    "through unchanged", state.get("message"))
        return {"resolved_query": state.get("message"), "pending_clarification": {},
                "needs_clarification": False, "clarification_question": None}
    # ANSWER ALREADY RESTATES THE QUESTION — do not glue. Measured live, 2026-09-22: the
    # engine's own clarification was generic ("Could you clarify what you're asking
    # about?", `missing` empty), so the user did the natural thing and retyped their
    # exact original question ("top 5 debit transaction") as the "answer". Gluing
    # produced "top 5 debit transaction for top 5 debit transaction" — a self-duplicated
    # query the engine understandably could not answer, and since pending_clarification
    # is deliberately kept armed (see below) until an ANSWERED turn, retyping the same
    # question again repeats the identical corruption — an invisible infinite loop from
    # the user's side, who is doing nothing wrong.
    #
    # `_content_words`, the same word-level comparator this module already uses for
    # frame/message grounding: when every content word of the ORIGINAL already appears
    # in the ANSWER, the answer is not a missing value — it already IS a self-contained
    # restatement (verbatim or extended) of the request. Gluing adds nothing but a
    # duplicated fragment; the answer alone is what should reach the engine.
    _orig_words, _ans_words = _content_words(original), _content_words(answer)
    if _orig_words and _orig_words <= _ans_words:
        logger.info("clarify_reply_node: answer %r already restates the pending "
                    "question %r — using it alone rather than gluing", answer, original)
        resolved = answer
    else:
        joiner = "" if re.match(r"^\s*(for|in|from|during|between|on|at|by)\b", answer, re.I) else "for "
        resolved = f"{original} {joiner}{answer}".strip()
    logger.info("clarify_reply_node: completed the pending request -> %r", resolved)
    # The user typed only an answer ("2024"); what actually reaches the engine is their
    # earlier question with that answer attached. Reported for the same reason as
    # context_resolve_node's: a clarification bound to the WRONG earlier request is
    # otherwise invisible until the answer itself looks wrong.
    _used = {"carried": {"completing": original}, "changed": {"operation": "clarify_reply"},
             "resolved_query": resolved}
    # pending_clarification is deliberately NOT cleared here. Clearing it before the
    # engine runs meant ask_clarification_node saw an empty slot on the next round and
    # fell back to this turn's resolved_query — which is the already-concatenated
    # string — so the anti-compounding guard never executed and the chain grew exactly
    # as before ("show me the breakdown by city for 2024 for only the top 3").
    # memory_write_node clears it on the answered turn that actually resolves it.
    return {"resolved_query": resolved, "context_used": _used,
            "needs_clarification": False, "clarification_question": None}


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
