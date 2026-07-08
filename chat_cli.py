#!/usr/bin/env python
"""chat_cli.py — run one chatbot turn from the terminal.

Usage:
    python chat_cli.py "how many incidents are escalated"
    python chat_cli.py "and waived ones?" --session mysession   # follow-up, same session

Run from the repo root, with the venv active:
    source .venv/bin/activate
    python chat_cli.py "hi"
"""
from __future__ import annotations

import argparse
import json
import logging

from chatbot.run import run_chat_turn

# Configured here (the actual CLI entrypoint), not inside chatbot/ itself —
# chatbot.run is a plain importable library, so it never touches logging
# config on its own (see chatbot/run.py's __main__ block for why).
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | [%(name)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one turn through the chatbot supervisor graph.")
    parser.add_argument("message", help="the message/query to send")
    parser.add_argument("--session", default="cli-session", help="session id (same id = same conversation)")
    args = parser.parse_args()

    response = run_chat_turn(args.message, args.session)

    print(f"\nYou: {args.message}")
    print(f"Bot: {response.get('answer_text')}\n")
    print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
