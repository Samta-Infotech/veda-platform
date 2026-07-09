"""apps.query.inference_client — thin HTTP client to the inference service (migration_plan.md §5, §6.1).

The api tier NEVER imports ``veda_core``; it talks to the inference service over
HTTP. Timeouts + a minimal circuit breaker (§9a) mean a slow/unreachable inference
tier degrades to a structured error, never a hung request. The server-resolved
``{source_id, tenant}`` is forwarded in headers (never a client-supplied tenant, §6.2).
Uses stdlib urllib to avoid adding a dependency to the thin api image (§1.3).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator


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
            base_url=os.environ.get("INFERENCE_URL", "http://inference:8001"),
            timeout_s=float(os.environ.get("INFERENCE_TIMEOUT_S", "300")),
        )

    def _request(self, path: str, body: dict, source_id, tenant, request_id=None,
                 accept: str | None = None, source_ids=None) -> urllib.request.Request:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        if source_id is not None:
            headers["X-Veda-Source-Id"] = str(source_id)
        if source_ids:
            # Server-validated scope SET (P5). Comma-separated, ownership already checked
            # in the view — the inference tier trusts these because they arrive from the
            # api tier, never from the end client (§6.2).
            headers["X-Veda-Source-Ids"] = ",".join(str(s) for s in source_ids)
        if tenant is not None:
            headers["X-Veda-Tenant"] = str(tenant)
        if request_id:
            headers["X-Request-Id"] = str(request_id)  # trace across api→inference (§6.3)
        return urllib.request.Request(url, data=data, headers=headers, method="POST")

    def _post(self, path: str, body: dict, source_id, tenant, request_id=None, source_ids=None) -> dict:
        req = self._request(path, body, source_id, tenant, request_id=request_id, source_ids=source_ids)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise InferenceUnavailable(f"inference {exc.code} at {req.full_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise InferenceUnavailable(f"inference unreachable at {req.full_url}: {exc}") from exc

    def run_hybrid_query(self, query: str, source_id=None, tenant=None, flags=None,
                         request_id=None, source_ids=None) -> dict:
        return self._post(
            "/v1/run_hybrid_query",
            {"query": query, "source_id": source_id, "tenant": tenant,
             "source_ids": source_ids, "flags": flags},
            source_id, tenant, request_id=request_id, source_ids=source_ids,
        )

    def stream_hybrid_query(
        self, query: str, source_id=None, tenant=None, flags=None, request_id=None,
    ) -> Iterator[tuple[str, dict]]:
        """Yields (event, data) as the inference tier's SSE stream delivers them
        (progress events as the pipeline advances, then one final "result" event).
        ``resp`` is read incrementally line-by-line — NOT buffered whole — so events
        surface to the caller as soon as the inference tier flushes them (§ SSE)."""
        req = self._request(
            "/v1/run_hybrid_query/stream",
            {"query": query, "source_id": source_id, "tenant": tenant, "flags": flags},
            source_id, tenant, request_id=request_id, accept="text/event-stream",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.config.timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise InferenceUnavailable(f"inference {exc.code} at {req.full_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise InferenceUnavailable(f"inference unreachable at {req.full_url}: {exc}") from exc

        try:
            event, data_lines = None, []
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                elif line == "":  # blank line terminates one SSE frame
                    if event is not None:
                        try:
                            data = json.loads("".join(data_lines)) if data_lines else {}
                        except ValueError:
                            data = {}
                        yield event, data
                    event, data_lines = None, []
        finally:
            resp.close()

    def retrieve(self, query: str, source_id=None, tenant=None, top_k=None) -> dict:
        return self._post(
            "/v1/retrieve",
            {"query": query, "source_id": source_id, "tenant": tenant, "top_k": top_k},
            source_id, tenant,
        )
