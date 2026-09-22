"""Query audit — the ONE ``QueryLog`` writer (traceability Part 19).

WHY THIS MODULE EXISTS
    ``QueryLog`` was written from exactly one place, ``apps.query.views.QueryView
    ._audit``, so the ``/api/v1/query`` front door was audited and the chat front
    door — the one the product actually ships — was not. Chat turns persisted only
    as ``ChatMessage`` rows, which are conversation content, not an audit trail:
    they carry no status, no route, no executed SQL, no latency, and no source.

    The fix is NOT a second copy of the audit write in the chat view. This
    codebase has been bitten repeatedly by exactly that (see
    ``apps.chat.turn_events``'s own docstring on the two byte-identical if/elif
    ladders that drifted). One writer, two callers.

BEST-EFFORT BY CONTRACT
    An audit-write failure is logged with its traceback and never propagated:
    losing an audit row must not turn a successfully answered query into a 500.
    It is logged rather than swallowed silently, so a broken audit table is
    visible instead of invisible.

WHAT IS SAFE TO STORE HERE
    ``executed_sql`` is the PARAMETERIZED text — placeholders, never interpolated
    values (a hard constraint, see ``QueryLog``'s own docstring). Warning CODES
    are stored, not their display messages. No credential, no grant payload, and
    no resource path ever reaches this table.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Table name the verified-query cache reports when it served the answer, instead of
#: a real table. Lives here — not in either view — because BOTH front doors have to
#: agree on this string or one of them silently records ``cache_hit=False`` forever,
#: and this module is the one thing both of them already import.
CACHED_TABLE_SENTINEL = "(cached)"

#: Terminal statuses the model accepts (apps.query.models.TerminalStatus). Anything
#: else is normalized rather than written raw, so a new engine status can never make
#: the column meaningless — or blow the 32-char limit.
_KNOWN_STATUSES = frozenset({
    "answered", "no_table", "clarify", "refuse", "ungrounded",
    "qualifier_dropped", "ir_mismatch", "invalid", "exec_error",
})

#: Engine/SubResult statuses that are not TerminalStatus members, mapped onto the
#: closest one. "ok" is the SubResult-level success label; "refused"/"error" are the
#: MultiResult ones (veda_core/query/multi_result.py).
_STATUS_ALIASES = {
    "ok": "answered",
    "refused": "refuse",
    "error": "exec_error",
    "access_denied": "refuse",
    "no_match": "refuse",
    "unavailable": "exec_error",
}


def normalize_status(status: Any) -> str:
    """An engine status mapped onto the frozen ``TerminalStatus`` vocabulary.

    Unknown values become ``"refuse"`` rather than being written verbatim: the
    column is a closed enum that dashboards group by, and an unmapped value would
    silently create a new bucket nobody is counting.
    """
    s = str(status or "").strip().lower()
    if s in _KNOWN_STATUSES:
        return s
    return _STATUS_ALIASES.get(s, "refuse" if s else "exec_error")


def _int_or_none(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return max(0, int(round(float(v))))
    except (TypeError, ValueError):
        return None


def _ids(values: Optional[Iterable]) -> list:
    out = []
    for v in (values or []):
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def record_query(*, query: str, tenant: str, user=None, source_id=None,
                 status: Any = "", route: str = "", sql: str = "",
                 refusal: str = "", latency_ms: Any = None, usage: Optional[dict] = None,
                 cache_hit: bool = False, request_id: str = "",
                 participating_sources: Optional[Iterable] = None,
                 partial: bool = False,
                 warning_codes: Optional[Iterable] = None) -> None:
    """Append one ``QueryLog`` row. Never raises.

    ``request_id`` is the correlation key: it is the same value the engine adopted
    as its ``trace_id``, so this row joins to the full engine trace and to the
    ``support.trace_id`` shown to the user.

    ``user`` may be ``None`` (unauthenticated / dev), which is recorded as-is
    rather than guessed at.
    """
    try:
        from .models import QueryLog

        usage = usage or {}
        # An AnonymousUser is not a persistable FK target — record it as "no user"
        # rather than letting the write fail and losing the whole row.
        if user is not None and not getattr(user, "pk", None):
            user = None

        QueryLog.objects.create(
            source_id=source_id,
            user=user,
            tenant=tenant or "",
            query_text=query or "",
            route=(route or "")[:16],
            status=normalize_status(status),
            executed_sql=sql or "",
            refusal_reason=refusal or "",
            latency_ms=_int_or_none(latency_ms),
            request_id=(request_id or "")[:64],
            cache_hit=bool(cache_hit),
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            participating_sources=_ids(participating_sources),
            partial=bool(partial),
            warning_codes=[str(c)[:64] for c in (warning_codes or []) if c],
        )
    except Exception:  # noqa: BLE001 — audit must never break the response
        logger.exception("query audit write failed request_id=%s tenant=%s status=%s",
                         request_id, tenant, status)


def audit_fields_from_explain(explain: Optional[dict]) -> dict:
    """The audit-relevant facts carried by an explainability payload.

    Reads the payload the turn ALREADY produced rather than re-deriving anything
    — one source of truth, and it means the audit row and what the user was shown
    can never disagree. Tolerates a v1 payload (no v2 blocks) and a refusal
    payload (no ``sql``/``result``) by simply returning fewer keys.
    """
    ex = explain if isinstance(explain, dict) else {}
    out: dict = {}

    sql_block = ex.get("sql")
    if isinstance(sql_block, dict) and sql_block.get("query"):
        out["sql"] = sql_block["query"]

    result = ex.get("result")
    if isinstance(result, dict):
        out["partial"] = bool(result.get("partial"))

    warnings = ex.get("warnings")
    if isinstance(warnings, list):
        out["warning_codes"] = [w.get("code") for w in warnings
                                if isinstance(w, dict) and w.get("code")]

    # Source identifiers moved to the level-3 `audit` block (they are internal keys,
    # not something a normal client renders — see safe_projection). The audit row
    # still needs them, so read them from there, falling back to the older shape for
    # a payload produced before the move.
    audit = ex.get("audit")
    if isinstance(audit, dict) and isinstance(audit.get("sources"), list):
        out["participating_sources"] = [s.get("id") for s in audit["sources"]
                                        if isinstance(s, dict) and s.get("id")]
    else:
        execution = ex.get("execution")
        if isinstance(execution, dict) and isinstance(execution.get("sources"), list):
            out["participating_sources"] = [s.get("id") for s in execution["sources"]
                                            if isinstance(s, dict) and s.get("id")]

    support = ex.get("support")
    if isinstance(support, dict) and support.get("trace_id"):
        out["request_id"] = support["trace_id"]

    # Present only on a refusal payload (build_refusal_explain).
    if ex.get("why"):
        out["refusal"] = ex["why"]
    if ex.get("status"):
        out["status"] = ex["status"]
    return out
