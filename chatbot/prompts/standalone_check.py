"""chatbot.prompts.standalone_check — classify_node's second-opinion prompt
(chatbot/nodes.py::_depends_on_history).

Generic (non-keyword) sanity check run only when the supervisor prompt
(supervisor.py) already said "smalltalk" and prior turns exist: asks the LLM
directly whether the message depends on the earlier conversation to mean
anything concrete, instead of pattern-matching specific phrasings — no fixed
word list generalizes to real production traffic.
"""
from __future__ import annotations

STANDALONE_CHECK_SYSTEM = (
    "A chatbot classifier just labeled a user's message as pure smalltalk "
    "(no data question at all). Before trusting that, sanity-check it: could "
    "this exact message mean something different, or refer back to something, "
    "if you also saw the conversation before it — i.e. does understanding it "
    "fully depend on the earlier turns? Answer with EXACTLY one word: "
    "'dependent' if it relies on the earlier conversation to mean anything "
    "concrete (regardless of the specific words used), or 'standalone' if it "
    "truly stands alone with no such dependency. Output only that one word."
)


def build_standalone_check_user_prompt(message: str, history: list) -> str:
    hist_lines = [f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-6:]]
    hist_block = "\n".join(hist_lines)
    return f"Conversation before this message:\n{hist_block}\n\nMessage: {message}"
