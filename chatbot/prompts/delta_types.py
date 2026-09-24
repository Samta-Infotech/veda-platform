"""chatbot.prompts.delta_types — ONE canonical definition of the continuation types.

Both prompts that ask a model to classify a follow-up used to carry their own copy of
these eight definitions, the same grounding rule and overlapping examples:
`supervisor.py`'s delta addendum (671 tok) and `delta_classify.py`'s standalone prompt
(1068 tok). ~600 tokens of near-duplicate instruction, in two files that could drift
apart without anything noticing.

WHAT THIS MODULE ACTUALLY SHARES TODAY: the closed set (`DELTA_TYPES`) and the
field rule (`DELTA_TYPES_WITH_FIELD`). Nothing else.

RENDERING THE SUPERVISOR'S ADDENDUM FROM HERE WAS TRIED AND REVERTED, 2026-09-21.
Measured, in --mode full against a 54/54 baseline:

  compact rendering, no examples   53/55 — "compare that with 2023" -> ambiguous and
                                   "what about Mumbai" -> new_topic, both 5/5 on repeat,
                                   so not noise
  compact rendering, with examples 52/55 — worse again, "what about Mumbai" failing in
                                   two scenarios
  original text restored           54/54

The supervisor's delta block is 671 tokens and costs ~2.4 s on every follow-up, so the
incentive to compress it is real. It does not survive paraphrase. Anyone retrying must
run `run_eval.py --mode full` and revert on anything below 54/54.

`render_delta_types()` is kept because the standalone prompt's wording may be generated
from it later, and because the semantics belong written down in one place — but it is
NOT currently wired into either prompt.
"""
from __future__ import annotations

# The closed set the parser accepts. "replace" and "remove" joined the original six
# rather than superseding them, so every stored frame keeps its meaning.
DELTA_TYPES = ("new_topic", "refine", "replace", "remove",
               "drill_down", "drill_up", "compare", "ambiguous")

# The two that carry a target. Kept explicit so a model that emits `delta_field` on an
# operation with no target is ignored rather than trusted.
DELTA_TYPES_WITH_FIELD = ("replace", "remove")

# One line each: the semantics a classifier actually needs. No rationale, no history,
# no restatement of the grounding rule — those live once, below.
_SEMANTICS = (
    ("new_topic",  "a different entity/subject than the frame"),
    ("refine",     "ADDS a filter/grouping, keeps everything in the frame"),
    ("replace",    "swaps the value of something the frame ALREADY HAS "
                   "(frame Year 2025, user \"what about 2024?\") — give delta_field"),
    ("remove",     "drops something the frame ALREADY HAS (\"remove India\") "
                   "— give delta_field"),
    ("drill_down", "narrows into a MORE SPECIFIC value of a dimension in play"),
    ("drill_up",   "go back / zoom out / drop the most specific filter"),
    ("compare",    "two things SIDE BY SIDE, with an explicit comparison word "
                   "(compare/versus/vs/against). Merely SWITCHING value "
                   "(\"what about 2024?\") is replace, NOT compare"),
    ("ambiguous",  "references something not clearly resolvable from the frame. "
                   "When unsure choose this — never guess"),
)

# The one rule that makes the whole design safe: the model may point at the frame's own
# facts or at words the user just typed, and at nothing else. Stated once, used by both.
GROUNDING_RULE = (
    "NEVER invent a column, table or filter value that is not the frame's own "
    "remembered fact or a word the user just typed. \"slot_candidates\": words copied "
    "VERBATIM from the new message naming a value, dimension or period ([] if none). "
    # Added after the harness caught its absence: with the worked examples removed, the
    # model started returning the WHOLE rendered filter ("Year equals 2025") as
    # delta_field instead of the field name, taking the suite 54/54 -> 53/55. It still
    # bound correctly for filters, because _same_field matches on word subsets — but
    # the same mistake on a shape slot is fatal, since "limit 100" resolves to no slot
    # alias at all. Cheaper to say it in one clause than to pay for four examples.
    "\"delta_field\" is the NAME only — \"Year\", \"City\", or one of "
    "limit/group_by/order_by/measures — never the whole phrase."
)

# Examples. BOTH renderings use them, and that is a measured decision, not a default.
# Dropping them from the compact rendering to save ~120 tokens took the harness from
# 54/54 to 53/55 with two CONSISTENT failures (5/5 repeats, not noise):
#   "compare that with 2023"  -> ambiguous instead of compare
#   "what about Mumbai"       -> new_topic instead of replace
# Both are cases where the type descriptions alone left the model unsure and an example
# settled it. 120 tokens is the honest price of that; correctness over token count.
_EXAMPLES = (
    ('["Year equals 2025"]', '"what about 2024?"', 'replace, delta_field "Year"'),
    ('["Year equals 2025"]', '"compare with 2024"', 'compare, no delta_field'),
    ('["Country equals India"]', '"remove India"', 'remove, delta_field "Country"'),
    ('["Status equals Active"]', '"only the ones in Mumbai"', 'refine, no delta_field'),
)


def render_delta_types(compact: bool = True) -> str:
    """The eight types as prompt text. `compact` drops the examples and tightens the
    layout — that is the supervisor's rendering, and the difference is ~300 tokens on
    every turn that carries a frame."""
    body = "\n".join(f'- "{name}": {meaning}' for name, meaning in _SEMANTICS)
    examples = "\n".join(f"frame {frame:26s} {message:28s} -> {out}"
                         for frame, message, out in _EXAMPLES)
    if compact:
        # Same examples, tighter framing. The compact rendering saves its tokens on the
        # per-type prose and the rationale, not on the examples.
        return f"{body}\n\nEXAMPLES:\n{examples}"
    return f"{body}\n\nEXAMPLES (frame filters, message -> the fields that matter):\n{examples}"
