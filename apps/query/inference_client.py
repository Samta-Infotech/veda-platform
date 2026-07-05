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

    def _post(self, path: str, body: dict, source_id, tenant, request_id=None) -> dict:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if source_id is not None:
            headers["X-Veda-Source-Id"] = str(source_id)
        if tenant is not None:
            headers["X-Veda-Tenant"] = str(tenant)
        if request_id:
            headers["X-Request-Id"] = str(request_id)  # trace across api→inference (§6.3)
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise InferenceUnavailable(f"inference {exc.code} at {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise InferenceUnavailable(f"inference unreachable at {url}: {exc}") from exc

    def run_hybrid_query(self, query: str, source_id=None, tenant=None, flags=None, request_id=None) -> dict:
        return self._post(
            "/v1/run_hybrid_query",
            {"query": query, "source_id": source_id, "tenant": tenant, "flags": flags},
            source_id, tenant, request_id=request_id,
        )

    def retrieve(self, query: str, source_id=None, tenant=None, top_k=None) -> dict:
        return self._post(
            "/v1/retrieve",
            {"query": query, "source_id": source_id, "tenant": tenant, "top_k": top_k},
            source_id, tenant,
        )
