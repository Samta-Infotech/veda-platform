"""chatbot.prompts — one file per prompt, re-exported here for convenience.

    common.py            — today_str() (shared by every prompt below)
    supervisor.py        — classify_node's classification prompt
    smalltalk.py         — smalltalk_node's reply prompt
    followup.py          — resolve_followup_node's rewrite prompt
    standalone_check.py  — classify_node's second-opinion prompt (_depends_on_history)
"""
from .common import today_str
from .followup import FOLLOWUP_SYSTEM_PROMPT, build_followup_user_prompt
from .smalltalk import FALLBACK_REPLY, build_smalltalk_system_prompt
from .standalone_check import STANDALONE_CHECK_SYSTEM, build_standalone_check_user_prompt
from .supervisor import build_supervisor_system_prompt, build_supervisor_user_prompt

__all__ = [
    "today_str",
    "build_supervisor_system_prompt",
    "build_supervisor_user_prompt",
    "build_smalltalk_system_prompt",
    "FALLBACK_REPLY",
    "FOLLOWUP_SYSTEM_PROMPT",
    "build_followup_user_prompt",
    "STANDALONE_CHECK_SYSTEM",
    "build_standalone_check_user_prompt",
]
