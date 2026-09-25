"""apps.query.data_vocabulary — the words this tenant's data actually contains.

One job: decide, in microseconds and with no model call, whether a message mentions
ANYTHING that exists in the scoped sources. A message that mentions nothing cannot be
answered from the data, so sending it to the engine buys a ~30s round-trip and then a
refusal anyway (measured 2026-09-17: "how many data sources do you have" spent 32.8s
being routed to assets_asset before returning "Could you clarify what you're asking
about?").

The vocabulary is read from the ingestion substrate the platform already maintains —
table names, column names, sampled column values and synonyms — so it grows with the
data and needs no curation. `business_purpose` prose is deliberately NOT included: a
dry run over it matched "life", "joke" and "world", which made the check useless.

CONSERVATIVE BY CONSTRUCTION. Everything here is tuned so a real question is never
held back, accepting that some junk gets through:
  · one matching content word is enough — never a threshold or a score;
  · sampled VALUES are in the vocabulary, so a question naming only a data value
    ("show me assets in Nagpur") still passes with no schema word in it at all;
  · singular/plural both count, so "owners" matches an `owner_id` column;
  · a load failure returns an EMPTY vocabulary, and an empty vocabulary never gates.
Dry run over 24 real questions and 14 unrelated messages: 0 real questions held back,
9/14 junk caught. The 5 that got through genuinely name schema words ("book", "email",
"delhi"), which is the check working as intended — the engine still refuses those.

This lives in the api tier because the substrate tables are Django's; the chatbot
package receives the result as plain data, the same way source_profiles already
crosses that boundary.
"""
from __future__ import annotations

import logging
import re
import threading
import time

from django.db import connection

logger = logging.getLogger(__name__)

# Function words, pleasantries and common conversational nouns. Stripped from BOTH the
# vocabulary and the message, because sampled free-text values drag them in: without
# this, "can u tell me abt my data" matched on "can"/"tell"/"the" and every junk
# message passed (measured — 1/14 caught before stripping, 9/14 after).
_STOPWORDS = frozenset("""
the and for are was were with that this from you your our their his her its
can could will would shall should may might must have has had been being
what which who whom whose when where why how many much more most less least
show tell give list find get set make take put see look want need help please
about into over under than then them they there here now today yesterday tomorrow
all any some none each every other another same different new old first last next
not but yet still also just only even very too quite really such own way thing
things something anything everything nothing someone anyone everyone nobody
hello hey yes okay sure thanks thank welcome bye goodbye please sorry
life world song poem joke politics vacation weather cup won win sing write plan
one two three four five six seven eight nine ten
ones
""".split())
# "ones" (2026-09-25): "only the Nagpur ones" / "only the Upi ones" — the commonest drill
# phrasing — carried "ones" as a CONTENT word, so names_only_values() could never say the
# message names only a value, and a run where the model called it a new question lost the
# conversation (demo chain C7: answered from the employee handbook). It names no data.

_WORD_RE = re.compile(r"[^a-z0-9]+")
_MIN_LEN = 3                 # shorter tokens carry no signal and collide with everything
_CACHE_TTL_SECS = 900        # the substrate only changes on re-ingest
_CACHE: dict = {}
_LOCK = threading.Lock()


def _tokens(text: str | None) -> set[str]:
    return {w for w in _WORD_RE.split((text or "").lower()) if len(w) >= _MIN_LEN}


def _content(text: str | None) -> set[str]:
    return _tokens(text) - _STOPWORDS


def _forms(word: str) -> set[str]:
    """Both number forms, so a question's "owners" matches an `owner_id` column and a
    question's "property" matches a `properties` table. Deliberately crude — a real
    stemmer would also fold unrelated words together."""
    out = {word, word + "s"}
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        out.add(word[:-1])
    return out


def _load(source_ids: tuple[int, ...]) -> set[str]:
    """Read the vocabulary for these sources. Any failure yields an empty set, which
    disables the check rather than risking a wrong answer — see module docstring."""
    structural, values = _load_split(source_ids)
    return structural | values


def _load_split(source_ids: tuple[int, ...]) -> tuple[set[str], set[str]]:
    """The same substrate read as _load, kept as its two halves: STRUCTURAL words (table
    and column names, both number forms) and VALUE words (sampled values, synonyms).
    Both empty on any failure."""
    vocab: set[str] = set()
    structural: set[str] = set()
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT name FROM substrate_schematable WHERE source_id IN %s",
                        [source_ids])
            for (name,) in cur.fetchall():
                structural |= _tokens(name)
            cur.execute(
                "SELECT c.name FROM substrate_schemacolumn c "
                "JOIN substrate_schematable t ON c.table_id = t.id "
                "WHERE t.source_id IN %s", [source_ids])
            for (name,) in cur.fetchall():
                structural |= _tokens(name)
            cur.execute("SELECT value FROM substrate_columnvaluesample WHERE source_id IN %s",
                        [source_ids])
            for (value,) in cur.fetchall():
                vocab |= _tokens(value)
            cur.execute("SELECT term FROM substrate_synonym WHERE source_id IN %s",
                        [source_ids])
            for (term,) in cur.fetchall():
                vocab |= _tokens(term)
    except Exception:
        logger.exception("data_vocabulary: load failed for sources=%s — the grounding "
                         "check stays disabled for this turn", source_ids)
        return set(), set()

    # Number forms are generated only for the STRUCTURAL half. Sampled values are real
    # data ("Nagpur", "Tax Invoice"); inventing plurals of them would widen the
    # vocabulary with strings that exist nowhere.
    structural |= {form for word in structural for form in _forms(word)}
    return structural - _STOPWORDS, vocab - _STOPWORDS


def vocabulary_for(source_ids) -> list[str]:
    """Cached vocabulary for an already-authorised source scope. Returns a list because
    this crosses into the chatbot package as plain JSON-safe data."""
    try:
        key = tuple(sorted(int(s) for s in (source_ids or [])))
    except (TypeError, ValueError):
        return []
    if not key:
        return []
    entry = _CACHE.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL_SECS:
        return entry[1]
    with _LOCK:
        entry = _CACHE.get(key)
        if entry and (time.monotonic() - entry[0]) < _CACHE_TTL_SECS:
            return entry[1]
        words = sorted(_load(key))
        _CACHE[key] = (time.monotonic(), words)
        logger.info("data_vocabulary: loaded %d tokens for sources=%s", len(words), list(key))
        return words


def mentions_the_data(message: str, vocabulary) -> bool:
    """True when the message names ANYTHING in the scoped data — a table, a column, a
    value or a synonym, in either number form.

    Two boundary cases, decided opposite ways on purpose:
      · no vocabulary (load failed, or no sources) → True, never gate. The check is an
        optimization; without its input it must not have an opinion.
      · no CONTENT words at all, i.e. the message is nothing but function words and
        pleasantries ("tell me a joke", "what can you do for me today") → False, gate
        it. A data question has to name its subject, so a message that names nothing
        whatsoever cannot be one. Returning True here was measured to let 3 of 4
        unrelated messages straight through to the engine.
    """
    if not vocabulary:
        return True
    content = _content(message)
    if not content:
        return False
    lookup = vocabulary if isinstance(vocabulary, (set, frozenset)) else set(vocabulary)
    return any(form in lookup for word in content for form in _forms(word))


_SPLIT_CACHE: dict = {}


def _split_for(source_ids) -> tuple[frozenset, frozenset]:
    """Cached (structural, values) for an authorised scope; both empty when unknown."""
    try:
        key = tuple(sorted(int(s) for s in (source_ids or [])))
    except (TypeError, ValueError):
        return frozenset(), frozenset()
    if not key:
        return frozenset(), frozenset()
    entry = _SPLIT_CACHE.get(key)
    if entry and (time.monotonic() - entry[0]) < _CACHE_TTL_SECS:
        return entry[1]
    with _LOCK:
        entry = _SPLIT_CACHE.get(key)
        if entry and (time.monotonic() - entry[0]) < _CACHE_TTL_SECS:
            return entry[1]
        st, vals = _load_split(key)
        out = (frozenset(st), frozenset(vals))
        _SPLIT_CACHE[key] = (time.monotonic(), out)
        return out


def names_only_values(message: str, source_ids) -> bool | None:
    """Does the message name ONLY data values — "Nagpur", "EAST", "DEBIT" — and no table
    or column at all?

    Such a message has no subject of its own. With a conversation in progress it can only
    be narrowing that conversation, whatever a classifier labels it (measured 2026-09-25:
    after "distribution of properties by facing", a bare "Nagpur" was labelled a new
    question on one run and a follow-up on the next — the new-question run grounded it on
    a city/phone-code lookup table instead of the properties being discussed).

    True only when every content word is a known VALUE and none is a table or column word
    in either number form ("vendors" names a table: a subject, so False). None — never
    False — when there is nothing to decide with: no vocabulary, or no content words.
    Values and structure come from the tenant's own ingested substrate; nothing here is a
    word list.
    """
    structural, values = _split_for(source_ids)
    if not values:
        return None
    content = _content(message)
    if not content:
        return None
    for word in content:
        forms = _forms(word)
        if forms & structural:
            return False
        if word not in values:
            return False
    return True
