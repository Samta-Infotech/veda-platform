"""apps.core.middleware — request-id propagation (migration_plan.md §6.3).

Assigns/propagates an ``X-Request-Id`` so a single request can be traced across
api → inference (the api forwards it on the inference call) and into structured
logs and the QueryLog. If the client (or nginx) supplies one, it is honoured.
"""
from __future__ import annotations

import uuid


class RequestIdMiddleware:
    HEADER = "HTTP_X_REQUEST_ID"

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        rid = request.META.get(self.HEADER) or uuid.uuid4().hex
        request.request_id = rid
        response = self.get_response(request)
        response["X-Request-Id"] = rid
        return response
