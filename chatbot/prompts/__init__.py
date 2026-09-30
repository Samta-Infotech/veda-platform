"""chatbot.prompts — one file per prompt, re-exported here for convenience.

    common.py            — today_str() (shared by every prompt below)
    supervisor.py        — classify_node's classification prompt
    smalltalk.py         — smalltalk_node's reply prompt
    followup.py          — resolve_followup_node's rewrite prompt (fallback path
                            only — see delta_classify.py / chatbot/memory/)
    standalone_check.py  — classify_node's second-opinion prompt (_depends_on_history)
    carryover_check.py   — classify_node's back-reference second opinion
                            (_carries_over_subject); see its docstring for the
                            measurement that made it a separate prompt
    delta_classify.py    — chatbot/memory/classify.py's structured-memory
                            continuation classifier (see docs/MEMORY_ARCHITECTURE.md)
"""
from .carryover_check import (CARRYOVER_CHECK_SYSTEM,
                              build_carryover_check_user_prompt)
from .common import today_str
from .delta_classify import (
    DELTA_TYPES,
    build_delta_classify_system_prompt,
    build_delta_classify_user_prompt,
)
from .followup import FOLLOWUP_SYSTEM_PROMPT, build_followup_user_prompt
from .smalltalk import (CAPABILITY_REPLY, FALLBACK_REPLY, IDENTITY_REPLY,
                        REPHRASE_REPLY,
                        SMALLTALK_FALLBACK_REPLY,
                        build_smalltalk_system_prompt)
from .standalone_check import STANDALONE_CHECK_SYSTEM, build_standalone_check_user_prompt
from .supervisor import build_supervisor_system_prompt, build_supervisor_user_prompt

__all__ = [
    "today_str",
    "build_supervisor_system_prompt",
    "build_supervisor_user_prompt",
    "build_smalltalk_system_prompt",
    "FALLBACK_REPLY",
    "SMALLTALK_FALLBACK_REPLY",
    "REPHRASE_REPLY",
    "IDENTITY_REPLY",
    "CAPABILITY_REPLY",
    "FOLLOWUP_SYSTEM_PROMPT",
    "build_followup_user_prompt",
    "STANDALONE_CHECK_SYSTEM",
    "build_standalone_check_user_prompt",
    "CARRYOVER_CHECK_SYSTEM",
    "build_carryover_check_user_prompt",
    "DELTA_TYPES",
    "build_delta_classify_system_prompt",
    "build_delta_classify_user_prompt",
]
