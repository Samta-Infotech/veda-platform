"""chatbot.test_manual — quick manual smoke test, no pytest/framework needed.

call_engine_node reaches the engine via apps.query.inference_client
(INFERENCE_URL, default http://inference:8001), which only resolves on the
docker `veda_net` — run this INSIDE the api container:
    docker compose exec api python -m chatbot.test_manual
(Bare-metal still works if you set INFERENCE_URL to a locally-reachable
inference endpoint yourself.)

Tests, in one conversation (same session_id, so checkpointing/history applies):
  1. Smalltalk        -> should NOT touch the engine
  2. Data question     -> should route through veda_hybrid.run_hybrid_query
  3. Follow-up         -> should resolve using turn 2's context
  4. Ambiguous question -> should ask for clarification, not guess
"""
from __future__ import annotations

import json
import logging

from chatbot.run import run_chat_turn

# Configured here (a standalone script entrypoint), not inside chatbot/ itself
# — see chatbot/run.py's __main__ block for why the library stays silent by default.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | [%(name)s] %(message)s")

SESSION_ID = "manual-test-session"


def _print_turn(label: str, message: str, response: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"[{label}] USER: {message}")
    print(f"{'-' * 70}")
    print(json.dumps(response, indent=2, default=str))


def main() -> None:
    history = []

    turns = [
        ("SMALLTALK", "hi there!"),
        ("DATA QUESTION", "how many incidents are escalated"),
        ("FOLLOW-UP", "and how many are waived?"),
        ("AMBIGUOUS", "how many are there"),
    ]

    for label, message in turns:
        response = run_chat_turn(message, SESSION_ID, history=history)
        _print_turn(label, message, response)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.get("answer_text") or ""})

    print(f"\n{'=' * 70}")
    print("Done. Review each block above:")
    print("  - SMALLTALK reply should be conversational, no sql/rows.")
    print("  - DATA QUESTION should have status=answered, sql + rows populated.")
    print("  - FOLLOW-UP should resolve 'waived' against the incidents context.")
    print("  - AMBIGUOUS should have needs_clarification=true, not a guessed answer.")


if __name__ == "__main__":
    main()
