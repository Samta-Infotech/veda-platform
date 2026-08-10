"""apps.query.inference_client — thin HTTP client to the inference service (migration_plan.md §5, §6.1).

The api tier NEVER imports ``veda_core``; it talks to the inference service over
HTTP. Timeouts + a minimal circuit breaker (§9a) mean a slow/unreachable inference
tier degrades to a structured error, never a hung request. The server-resolved
``{source_id, tenant}`` is forwarded in headers (never a client-supplied tenant, §6.2).
Uses stdlib urllib to avoid adding a dependency to the thin api image (§1.3).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)

ENV_INFERENCE_URL = "INFERENCE_URL"
ENV_INFERENCE_TIMEOUT_S = "INFERENCE_TIMEOUT_S"
_DEFAULT_INFERENCE_URL = "http://inference:8001"
_DEFAULT_TIMEOUT_S = "300"

# Upstream error bodies are truncated before they reach an exception message:
# enough to diagnose, bounded so a large HTML error page can't flood the logs.
_ERROR_DETAIL_MAX_CHARS = 500

# Header names forwarded to the inference tier (§6.2, §6.3).
_HEADER_SOURCE_ID = "X-Veda-Source-Id"
_HEADER_SOURCE_IDS = "X-Veda-Source-Ids"
_HEADER_TENANT = "X-Veda-Tenant"
_HEADER_REQUEST_ID = "X-Request-Id"
# Gate 1 (User Story 3, Task 15) — the precomputed RBAC data scope (see
# apps.access_management.services.data_scope.serialize_data_scope). Omitted
# entirely when the caller passes None ("no restriction"), never sent as an
# empty object — absence is the "no restriction" signal on the inference side too.
_HEADER_DATA_SCOPE = "X-Veda-Data-Scope"

_SSE_EVENT_PREFIX = "event:"
_SSE_DATA_PREFIX = "data:"

_PATH_RUN_HYBRID_QUERY = "/v1/run_hybrid_query"
_PATH_RUN_HYBRID_QUERY_STREAM = "/v1/run_hybrid_query/stream"
_PATH_RETRIEVE = "/v1/retrieve"


@dataclass
class InferenceClientConfig:
    base_url: str
    timeout_s: float = 300.0


class InferenceUnavailable(RuntimeError):
    """Raised when the inference tier is unreachable or errors — surfaced as a
    structured 503 by the view, never a 500 (§9a, §18 circuit breaker)."""


class InferenceClient:
    def __init__(self, config: InferenceClientConfig | None = None):
        self.config = config or InferenceClientConfig(
            base_url=os.environ.get(ENV_INFERENCE_URL, _DEFAULT_INFERENCE_URL),
            timeout_s=float(os.environ.get(ENV_INFERENCE_TIMEOUT_S, _DEFAULT_TIMEOUT_S)),
        )

    def _request(self, path: str, body: dict, source_id, tenant, request_id=None,
                 accept: str | None = None, source_ids=None, data_scope=None
                 ) -> urllib.request.Request:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        if source_id is not None:
            headers[_HEADER_SOURCE_ID] = str(source_id)
        if source_ids:
            # Server-validated scope SET (P5). Comma-separated, ownership already checked
            # in the view — the inference tier trusts these because they arrive from the
            # api tier, never from the end client (§6.2).
            headers[_HEADER_SOURCE_IDS] = ",".join(str(s) for s in source_ids)
        if tenant is not None:
            headers[_HEADER_TENANT] = str(tenant)
        if request_id:
            headers[_HEADER_REQUEST_ID] = str(request_id)  # trace across api→inference (§6.3)
        if data_scope is not None:
            headers[_HEADER_DATA_SCOPE] = json.dumps(data_scope)
        return urllib.request.Request(url, data=data, headers=headers, method="POST")

    @staticmethod
    def _open(request: urllib.request.Request, timeout_s: float):
        """Open the request, mapping every transport-level failure to
        ``InferenceUnavailable``.

        Single place where urllib's two failure modes are translated — previously
        this identical 6-line try/except was duplicated in ``_post`` and
        ``stream_hybrid_query``, so a change to the error contract had to be made
        twice (DRY).
        """
        try:
            return urllib.request.urlopen(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:_ERROR_DETAIL_MAX_CHARS]
            raise InferenceUnavailable(
                f"inference {exc.code} at {request.full_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise InferenceUnavailable(
                f"inference unreachable at {request.full_url}: {exc}") from exc

    def _post(self, path: str, body: dict, source_id, tenant, request_id=None,
              source_ids=None, data_scope=None) -> dict:
        request = self._request(path, body, source_id, tenant, request_id=request_id,
                                source_ids=source_ids, data_scope=data_scope)
        with self._open(request, self.config.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def run_hybrid_query(self, query: str, source_id=None, tenant=None, flags=None,
                         request_id=None, source_ids=None, data_scope=None) -> dict:
        return self._post(
            _PATH_RUN_HYBRID_QUERY,
            {"query": query, "source_id": source_id, "tenant": tenant,
             "source_ids": source_ids, "flags": flags},
            source_id, tenant, request_id=request_id, source_ids=source_ids,
            data_scope=data_scope,
        )

    def stream_hybrid_query(
        self, query: str, source_id=None, tenant=None, flags=None, request_id=None,
        source_ids=None, data_scope=None,
    ) -> Iterator[tuple[str, dict]]:
        """Yields (event, data) as the inference tier's SSE stream delivers them
        (progress events as the pipeline advances, then one final "result" event).
        ``response`` is read incrementally line-by-line — NOT buffered whole — so events
        surface to the caller as soon as the inference tier flushes them (§ SSE)."""
        request = self._request(
            _PATH_RUN_HYBRID_QUERY_STREAM,
            {"query": query, "source_id": source_id, "tenant": tenant,
             "source_ids": source_ids, "flags": flags},
            source_id, tenant, request_id=request_id, accept="text/event-stream",
            source_ids=source_ids, data_scope=data_scope,
        )
        response = self._open(request, self.config.timeout_s)
        try:
            try:
                yield from self._iter_sse_frames(response)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                # A connection drop/timeout mid-stream (e.g. a slow pipeline run
                # outliving a proxy/gateway timeout) must surface the same way an
                # unreachable inference tier does — never as a silently empty
                # engine_result (the caller's broad except would otherwise turn
                # this into an untraceable "no result" instead of a clear outage).
                raise InferenceUnavailable(
                    f"inference stream dropped mid-response at {request.full_url}: {exc}"
                ) from exc
        finally:
            response.close()

    def retrieve(self, query: str, source_id=None, tenant=None, top_k=None) -> dict:
        return self._post(
            _PATH_RETRIEVE,
            {"query": query, "source_id": source_id, "tenant": tenant, "top_k": top_k},
            source_id, tenant,
        )

    @staticmethod
    def _iter_sse_frames(response) -> Iterator[tuple[str, dict]]:
        """Parse the raw byte lines of an SSE response into (event, data) frames.

        A frame is terminated by a blank line; a frame whose data is absent or not
        valid JSON yields ``{}`` rather than raising, so one malformed progress
        event never aborts an otherwise healthy stream.
        """
        event: str | None = None
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
            if line.startswith(_SSE_EVENT_PREFIX):
                event = line[len(_SSE_EVENT_PREFIX):].strip()
            elif line.startswith(_SSE_DATA_PREFIX):
                data_lines.append(line[len(_SSE_DATA_PREFIX):].strip())
            elif line == "":  # blank line terminates one SSE frame
                if event is not None:
                    yield event, _parse_frame_data(data_lines)
                event, data_lines = None, []


def _parse_frame_data(data_lines: list[str]) -> dict:
    if not data_lines:
        return {}
    try:
        return json.loads("".join(data_lines))
    except ValueError:
        logger.warning("inference SSE frame carried non-JSON data; treating as empty")
        return {}
