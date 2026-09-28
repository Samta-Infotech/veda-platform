#!/usr/bin/env python3
"""scripts/chat_probe.py — manual probe for the LIVE chat API (CHAT_API_CONTRACT.md).

Drives one real chat SESSION (all messages share one ``chat_id``, exactly like a
user typing turn after turn into one chat window) against
``POST {base}/api/v1/conversations/query`` and prints, per turn, the answer, the
generated SQL, the routing/classification facts, the scope the turn ran against,
how many SLM calls it spent, and wall time.

USAGE
-----
    python scripts/chat_probe.py "how many properties are there in each city" "only Mumbai" "top 3"
    python scripts/chat_probe.py --file session.txt [--sources 2,3,4,5] [--chat-id ID] \\
        [--base http://localhost:8080] [--token TOKEN | --user USERNAME] [--json] [--show-context]

Each positional argument is one turn, sent in order. ``--file`` reads one turn
per line instead (blank lines and ``#``-prefixed comment lines are skipped).

AUTH
----
The chat endpoints require an authenticated caller (AUTH_API_CONTRACT.md / DRF
Token or Session — see ``config/settings/base.py``'s ``DEFAULT_AUTHENTICATION_CLASSES``
and ``apps/chat/views.py::_resolve_user``). Resolved in this order:

    1. ``--token TOKEN``          — used verbatim as ``Authorization: Token TOKEN``
    2. ``VEDA_TOKEN`` env var     — same
    3. ``--user USERNAME``        — mints/reuses a token by running, in the api
                                     container:
                                       docker compose exec -T api python manage.py \\
                                           drf_create_token USERNAME
                                     (``rest_framework.authtoken`` is installed —
                                     see ``config/settings/base.py``.)

If none of the three is given, the script refuses to send a request that is
certain to 401 and tells you which flag/env-var to set instead.

WHERE THE FACTS COME FROM
--------------------------
* answer / SQL / usage / chat_id / message_id — the SSE stream itself (§1b of
  CHAT_API_CONTRACT.md): ``content``, ``explainability`` (``.sql.query``),
  ``usage``, ``completed`` events.
* classify_lane / routing_consumed / frame_source / slm_calls / continuity_op —
  NOT on the wire. These are engine-internal facts recorded in the engine's own
  compact per-query trace (``veda_core/veda/explain.py::ExplainTrace.compact``),
  appended to ``veda_core/logs/explain_trace.jsonl`` (host path; bind-mounted,
  so this script reads it directly with no docker exec). Matched by
  ``trace_id`` — which is exactly the ``X-Request-Id`` response header for that
  turn: ``apps/core/middleware.RequestIdMiddleware`` mints/echoes it, chatbot
  forwards it to inference as the ``X-Request-Id`` request header
  (``apps/query/inference_client.py``), and ``inference/routes/hybrid.py::
  _incoming_trace_id`` hands it to ``run_hybrid_query(trace_id=...)`` verbatim
  — see that function's own docstring ("reusing the caller's request id when
  one is passed"). ``continuity_op`` may be absent from a given trace record
  (being added separately) — printed as ``n/a`` when missing, never guessed.
* A turn the engine never saw at all (small talk, a canned greeting — see
  CHAT_API_CONTRACT.md's 2026-09-11 "no-engine turns" entry) writes NO trace
  record; this script says so rather than treating a lookup miss as an error.
* conversation_context (``--show-context``) — inference prints
  ``[inference] conversation_context trace_id=... keys=...`` to its own
  stdout (not a log framework line), so this greps
  ``docker compose logs inference --since <turn start>`` for the same
  trace_id. Optional and skipped by default since it shells out to docker.

Dependency-free (stdlib ``urllib`` only) — runs under the host ``python3`` or
``.venv/bin/python``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://localhost:8080"
QUERY_PATH = "/api/v1/conversations/query"

# veda_core/logs/explain_trace.jsonl is bind-mounted straight from the repo root
# (docker-compose.yml mounts ``.:/app``), so the host path below is the SAME file
# the api/inference containers write to — no docker exec needed to read it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_PATH = os.path.join(_REPO_ROOT, "veda_core", "logs", "explain_trace.jsonl")

_TOKEN_RE = re.compile(r"[Tt]oken\s+([0-9a-fA-F]{20,64})")

# The compact trace fields this probe surfaces per CHAT_API_CONTRACT.md's own
# terms — see ExplainTrace.compact() in veda_core/veda/explain.py. `continuity_op`
# is listed even though it may not exist yet in a given record (being added by a
# different change concurrently); its absence is handled explicitly, never assumed.
COMPACT_FIELDS = (
    "classify_lane", "routing_consumed", "frame_source", "slm_calls", "continuity_op",
)


# --------------------------------------------------------------------------- #
# SSE parsing — pure, network-free, unit-testable.
# --------------------------------------------------------------------------- #

def parse_sse_lines(lines):
    """Yield ``(event, data)`` for each SSE frame in an iterable of text lines.

    ``lines`` may be a list of ``str`` (already decoded, newline-stripped or
    not) or any iterable yielding one line at a time — this is exactly the
    shape ``http.client.HTTPResponse`` gives when iterated. A frame with no
    ``event:`` line defaults to event name ``"message"`` (the SSE spec
    default); ``data:`` lines are joined with ``\\n`` before JSON-decoding, and
    a non-JSON body is returned as ``{"_raw": <joined text>}`` rather than
    raising, so one malformed frame cannot kill the whole probe.
    """
    event = None
    data_lines: list[str] = []

    def _flush():
        if event is None and not data_lines:
            return None
        joined = "\n".join(data_lines)
        if joined == "":
            payload = {}
        else:
            try:
                payload = json.loads(joined)
            except json.JSONDecodeError:
                payload = {"_raw": joined}
        return (event or "message", payload)

    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.rstrip("\r\n")
        if line == "":
            frame = _flush()
            if frame is not None:
                yield frame
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue  # SSE comment/keepalive
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # any other field (id:, retry:) is ignored — not used by this contract

    frame = _flush()
    if frame is not None:
        yield frame


# --------------------------------------------------------------------------- #
# Engine trace lookup — pure, network-free, unit-testable.
# --------------------------------------------------------------------------- #

def find_trace(trace_path: str, trace_id: str) -> dict | None:
    """The LAST compact trace record in ``trace_path`` whose ``trace_id`` matches.

    Last, not first: a replayed/duplicate trace_id (should not happen, but the
    file is append-only and this script must never crash on one) should surface
    the most recent write. Returns None when the file is missing (nothing
    ingested/mounted yet) or holds no matching line — both are reported by the
    caller as "no engine trace for this turn", not as an error.
    """
    if not trace_id:
        return None
    match = None
    try:
        with open(trace_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or trace_id not in line:
                    continue  # cheap substring pre-filter before paying for json.loads
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("trace_id") == trace_id:
                    match = rec
    except FileNotFoundError:
        return None
    return match


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

class AuthError(RuntimeError):
    pass


def mint_token_for_user(username: str) -> str:
    """Reuse (idempotent) or create a DRF auth token for ``username`` by
    running ``manage.py drf_create_token`` inside the running ``api`` container.
    """
    cmd = ["docker", "compose", "exec", "-T", "api", "python", "manage.py",
           "drf_create_token", username]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise AuthError(f"could not run `docker compose` ({exc}); is docker running?") from exc
    except subprocess.TimeoutExpired as exc:
        raise AuthError("`docker compose exec ... drf_create_token` timed out after 60s") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise AuthError(
            f"drf_create_token {username!r} failed (exit {proc.returncode}):\n{output.strip()}")
    m = _TOKEN_RE.search(output)
    if not m:
        raise AuthError(
            f"drf_create_token {username!r} produced no recognizable token in:\n{output.strip()}")
    return m.group(1)


def resolve_token(args) -> str:
    if args.token:
        return args.token
    env_token = os.environ.get("VEDA_TOKEN")
    if env_token:
        return env_token
    if args.user:
        return mint_token_for_user(args.user)
    raise AuthError(
        "no credential given — pass --token TOKEN, set VEDA_TOKEN, or pass --user "
        "USERNAME (mints a token via `docker compose exec api manage.py "
        "drf_create_token`). The chat API requires an authenticated caller "
        "(AUTH_API_CONTRACT.md); an unauthenticated request always gets a 401.")


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #

class TurnResult:
    def __init__(self):
        self.request_id: str | None = None
        self.thinking: dict | None = None          # last thinking frame
        self.content: list[dict] = []
        self.visualizations: list[dict] = []
        self.explainability: dict | None = None
        self.usage: dict | None = None
        self.insights: dict | None = None
        self.completed: dict | None = None
        self.error: dict | None = None
        self.wall_s: float = 0.0

    @property
    def answer_text(self) -> str:
        if self.completed and self.completed.get("summary"):
            return self.completed["summary"]
        texts = [b.get("content", "") for b in self.content if isinstance(b, dict)]
        return "\n".join(t for t in texts if t)

    @property
    def sql(self) -> str | None:
        if not self.explainability:
            return None
        return (self.explainability.get("sql") or {}).get("query")


def run_turn(base: str, token: str, message: str, chat_id: int | None,
             source_ids: list[int] | None, timeout: float, no_cache: bool = False) -> TurnResult:
    body = {"message": message, "chat_id": chat_id, "stream": True}
    if source_ids:
        body["source_ids"] = source_ids
    if no_cache:
        body["no_cache"] = True

    req = urllib.request.Request(
        base.rstrip("/") + QUERY_PATH,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            # No explicit Accept header: DRF's DEFAULT_RENDERER_CLASSES is
            # JSONRenderer-only (config/settings/base.py), so an
            # `Accept: text/event-stream` request fails DRF's content
            # negotiation with a 406 before the view ever runs — the SSE
            # response is a raw StreamingHttpResponse the view constructs
            # itself, not something content negotiation is meant to gate.
            # `urllib` defaults to `Accept: */*`, which negotiates fine
            # (confirmed against the live stack; matches curl's default).
            "Authorization": f"Token {token}",
        },
    )

    result = TurnResult()
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result.request_id = resp.headers.get("X-Request-Id")
            for event, data in parse_sse_lines(resp):
                if event == "thinking":
                    result.thinking = data
                elif event == "content":
                    result.content.append(data)
                elif event == "visualization":
                    result.visualizations.append(data)
                elif event == "explainability":
                    result.explainability = data
                elif event == "usage":
                    result.usage = data
                elif event == "insights":
                    result.insights = data
                elif event == "completed":
                    result.completed = data
                elif event == "error":
                    result.error = data
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body_json = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            body_json = {"_raw": raw.decode("utf-8", errors="replace") if raw else ""}
        result.request_id = exc.headers.get("X-Request-Id") if exc.headers else None
        result.error = {"code": f"HTTP_{exc.code}", "message": body_json}
    finally:
        result.wall_s = time.monotonic() - t0
    return result


# --------------------------------------------------------------------------- #
# conversation_context (optional, best-effort)
# --------------------------------------------------------------------------- #

def grep_conversation_context(trace_id: str, since_iso: str) -> str | None:
    """The ``[inference] conversation_context trace_id=...`` line for this turn,
    from ``docker compose logs inference --since <since_iso>``. Best-effort:
    returns None on any failure (docker unavailable, no matching line, timeout)
    rather than raising — this is a diagnostic extra, never load-bearing."""
    if not trace_id:
        return None
    cmd = ["docker", "compose", "logs", "inference", "--since", since_iso, "--no-log-prefix"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        if "[inference] conversation_context" in line and f"trace_id={trace_id}" in line:
            return line.strip()
    return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def scope_summary(explainability: dict | None, requested: list[int] | None) -> dict:
    out = {"requested": requested if requested else "server-default (all ready sources)"}
    if explainability:
        sources = explainability.get("sources")
        if sources:
            out["answered_from"] = [
                {"name": s.get("name"), "type": s.get("type"), "rows": s.get("rows")}
                for s in sources
            ]
    return out


def build_turn_report(n: int, message: str, result: TurnResult, trace: dict | None,
                       requested_sources: list[int] | None, context_line: str | None) -> dict:
    report = {
        "turn": n,
        "message": message,
        "request_id": result.request_id,
        "wall_s": round(result.wall_s, 3),
    }
    if result.error:
        report["error"] = result.error
        return report

    report["chat_id"] = (result.completed or {}).get("chat_id")
    report["message_id"] = (result.completed or {}).get("message_id")
    report["answer"] = result.answer_text
    report["sql"] = result.sql
    report["scope"] = scope_summary(result.explainability, requested_sources)
    report["usage"] = result.usage
    if result.explainability is not None:
        report["confidence"] = result.explainability.get("confidence")

    if trace is None:
        report["engine_trace"] = None  # this turn never reached the engine, or the trace hasn't landed
    else:
        engine = {k: trace.get(k) for k in COMPACT_FIELDS}
        engine["slm_call_count"] = len(trace.get("slm_calls") or [])
        engine["total_ms"] = trace.get("total_ms")
        report["engine_trace"] = engine

    # Supervisor-side SLM call count: not exposed anywhere on the wire today
    # (checked apps/chat/turn_events.py / CHAT_API_CONTRACT.md's metadata shape) —
    # printed only if a future response ever carries it under this key.
    report["supervisor_slm_calls"] = (result.usage or {}).get("supervisor_slm_calls")

    if context_line is not None:
        report["conversation_context"] = context_line
    return report


def print_turn_report(report: dict) -> None:
    print(f"\n=== Turn {report['turn']}: {report['message']!r} ===")
    print(f"  request_id (trace_id) : {report['request_id']}")
    if "error" in report:
        err = report["error"]
        print(f"  ERROR                  : {err}")
        return
    print(f"  chat_id / message_id   : {report.get('chat_id')} / {report.get('message_id')}")
    print(f"  answer                 : {report.get('answer')}")
    print(f"  SQL                    : {report.get('sql')}")
    scope = report.get("scope") or {}
    print(f"  scope (requested)      : {scope.get('requested')}")
    if "answered_from" in scope:
        print(f"  scope (answered from)  : {scope['answered_from']}")
    print(f"  confidence             : {report.get('confidence')}")

    trace = report.get("engine_trace")
    if trace is None:
        print("  engine trace           : none (turn bypassed the engine, or trace not yet written)")
    else:
        print(f"  classify_lane          : {trace.get('classify_lane')}")
        print(f"  routing_consumed       : {trace.get('routing_consumed')}")
        print(f"  frame_source           : {trace.get('frame_source')}")
        op = trace.get("continuity_op")
        print(f"  continuity_op          : {op if op is not None else 'n/a (not in this trace record)'}")
        print(f"  slm_calls (engine)     : {trace.get('slm_call_count')}")

    sup = report.get("supervisor_slm_calls")
    print(f"  slm_calls (supervisor) : {sup if sup is not None else 'n/a (not exposed on the wire)'}")

    usage = report.get("usage") or {}
    print(f"  wall time              : {report['wall_s']}s"
          f"  (server latency_ms={usage.get('latency_ms')})")

    if "conversation_context" in report:
        print(f"  conversation_context   : {report['conversation_context']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def read_messages(args) -> list[str]:
    if args.file:
        messages = []
        with open(args.file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                messages.append(line)
        if not messages:
            raise SystemExit(f"--file {args.file!r} contained no usable turns")
        return messages
    if not args.messages:
        raise SystemExit("give at least one message, or use --file")
    return args.messages


def parse_sources(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"--sources must be a comma-separated list of ints, got {raw!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manually probe the live chat API with a real, multi-turn session.")
    p.add_argument("messages", nargs="*", help="turns to send, in order (one chat session)")
    p.add_argument("--file", help="read turns from this file instead, one per line "
                                  "(blank/# lines skipped)")
    p.add_argument("--sources", help="comma-separated source ids to pin, e.g. 2,3,4,5")
    p.add_argument("--chat-id", type=int, default=None,
                    help="continue an existing chat instead of starting a new one")
    p.add_argument("--base", default=DEFAULT_BASE,
                    help=f"api base URL (default: {DEFAULT_BASE}, nginx's published port — "
                         "see docker-compose.yml)")
    p.add_argument("--token", help="DRF auth token, sent as `Authorization: Token <token>`")
    p.add_argument("--user", help="username to mint/reuse a token for via drf_create_token")
    p.add_argument("--trace-path", default=DEFAULT_TRACE_PATH,
                    help="path to explain_trace.jsonl (default: the bind-mounted host path)")
    p.add_argument("--timeout", type=float, default=300.0,
                    help="per-turn HTTP timeout in seconds (default 300, matching nginx)")
    p.add_argument("--no-cache", action="store_true",
                    help="ask the engine to skip the verified-query cache for this session")
    p.add_argument("--show-context", action="store_true",
                    help="also grep `docker compose logs inference` for the "
                         "conversation_context line of each turn (slower, shells out to docker)")
    p.add_argument("--json", action="store_true", help="emit one JSON object per turn instead "
                                                         "of the human-readable report")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    messages = read_messages(args)
    source_ids = parse_sources(args.sources)

    try:
        token = resolve_token(args)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    chat_id = args.chat_id
    reports = []
    for i, message in enumerate(messages, start=1):
        turn_start_iso = None
        if args.show_context:
            # RFC3339 UTC, second resolution — good enough for `docker compose logs --since`.
            turn_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        result = run_turn(args.base, token, message, chat_id, source_ids,
                          timeout=args.timeout, no_cache=args.no_cache)

        if result.completed and result.completed.get("chat_id") is not None:
            chat_id = result.completed["chat_id"]

        trace = find_trace(args.trace_path, result.request_id) if result.request_id else None

        context_line = None
        if args.show_context and result.request_id:
            context_line = grep_conversation_context(result.request_id, turn_start_iso)

        report = build_turn_report(i, message, result, trace, source_ids, context_line)
        reports.append(report)

        if args.json:
            print(json.dumps(report))
        else:
            print_turn_report(report)

        if result.error:
            print("\n(turn failed — stopping the session here)", file=sys.stderr)
            break

    return 0 if not any("error" in r for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
