"""Runs every filesystem (RAG / docs_contracts) test question used in this session
against a REAL deployed VEDA endpoint (default: https://vedademo.samta.ai:40443) —
not the local docker stack. Logs in for a real JWT, then hits
/api/v1/conversations/query for each question and prints the answer.

Covers the two documents actually exercised live in this session:
  - maintenance_policy.docx (fee-schedule questions)
  - msa_green_tower.pdf     (MSA / late-fee / amenities questions)

Usage:
    VEDA_TEST_USERNAME=ffffff VEDA_TEST_PASSWORD='Veda@12345678' \\
        python3 scripts/test_filesystem_queries_remote.py

Optional env vars:
    VEDA_TEST_BASE_URL       default: https://vedademo.samta.ai:40443
    VEDA_TEST_SOURCE_ID      default: 3   (docs_contracts)
    VEDA_TEST_TIMEOUT_SECS   default: 120

Credentials are never hardcoded here — pass them via env vars (or edit the two
constants below for a quick local run) so this file stays safe to commit.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VEDA_TEST_BASE_URL", "https://vedademo.samta.ai:40443")
SOURCE_ID = int(os.environ.get("VEDA_TEST_SOURCE_ID", "3"))
TIMEOUT_SECS = int(os.environ.get("VEDA_TEST_TIMEOUT_SECS", "120"))

LOGIN_URL = f"{BASE_URL}/api/v1/auth/login"
QUERY_URL = f"{BASE_URL}/api/v1/conversations/query"

USERNAME = os.environ.get("VEDA_TEST_USERNAME", "")
PASSWORD = os.environ.get("VEDA_TEST_PASSWORD", "")

# ---------------------------------------------------------------------------
# Questions — exactly what was run live against docs_contracts this session.
# ---------------------------------------------------------------------------
QUESTIONS = [
    # maintenance_policy.docx
    "What is the maintenance policy for asset 21?",
    "What is the rent fee under the maintenance policy?",
    "How much is the repair fee and how often is it charged?",
    "What is the insurance fee for the maintenance policy?",
    "What is the society charges amount for the maintenance policy?",
    "What is the tax amount under the maintenance policy?",
    "What is the loan payment amount under the maintenance policy?",
    # msa_green_tower.pdf
    "How much notice is required if either party wants to end the agreement?",
    "Is there any penalty for paying invoices late?",
    "Which types of invoices are subject to late payment charges?",
    "What is the monthly rate charged on overdue invoices?",
    "Which assets are included under this service arrangement?",
    "Which locations are associated with the covered assets?",
    "What amenities are included within the scope of services?",
    "Is intercom maintenance included in the covered services?",
    "Which financial categories are managed under this agreement?",
    "Does the agreement include repair-related expenses?",
]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return json.loads(body)
        except ValueError:
            return {"status_code": exc.code, "message": body}


def login(username: str, password: str) -> str:
    resp = _post(LOGIN_URL, {"username": username, "password": password})
    data = resp.get("data") or {}
    token = data.get("access_token")
    if not token:
        raise SystemExit(f"Login failed: {resp}")
    return token


def ask(token: str, message: str, chat_id: int | None, source_id: int = SOURCE_ID) -> tuple[str, int | None]:
    """Returns (answer, chat_id) — pass the returned chat_id into the next call
    to keep every question in the SAME conversation thread instead of each one
    silently creating a brand-new chat (the server's default when chat_id is
    omitted; see apps/chat/services.py::resolve_chat)."""
    payload = {"message": message, "stream": False, "source_id": source_id}
    if chat_id is not None:
        payload["chat_id"] = chat_id
    resp = _post(QUERY_URL, payload, token)
    data = resp.get("data") or {}
    answer = data.get("summary") or resp.get("message") or json.dumps(resp)
    return answer, data.get("chat_id", chat_id)


def main() -> int:
    username = USERNAME or input("Username: ").strip()
    password = PASSWORD or input("Password: ").strip()

    print(f"Logging in to {BASE_URL} as {username} ...")
    token = login(username, password)
    print("Login OK.\n")

    chat_id: int | None = None
    for i, q in enumerate(QUESTIONS, 1):
        print(f"Q{i}: {q}")
        answer, chat_id = ask(token, q, chat_id)
        print(f"A{i}: {answer}")
        print(f"    (chat_id={chat_id})")
        print("---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
