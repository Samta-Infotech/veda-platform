"""Coverage for a 2026-08-08 audit finding: ``/v1/retrieve`` (a debug/tooling
endpoint on the inference tier) called ``get_engine().retrieve(...)`` directly
and returned raw results with NO RBAC filtering at all — unlike
``/v1/run_hybrid_query``, which applies ``filter_retrieval_results`` inside
``veda.pipeline.run_query``. Confirmed via full-repo grep to have no current
caller from either real entry point (``apps.query``/``apps.chat``), so this was
dormant rather than actively exploitable — but a live, mounted endpoint with no
RBAC awareness is exactly the kind of thing a future caller wires up without
anyone noticing it needs the same gate.

Fixed in ``inference/routes/retrieve.py`` by applying the same
``filter_retrieval_results`` Task 16 already built. This file proves the fix:
the route now narrows its results under a restrictive ambient context, and
stays unchanged when there's nothing to restrict.

Django-free (this route never touches Django); needs ``veda_core`` on
``sys.path`` for the route's own ``from veda....`` imports, and FastAPI's
``TestClient`` to drive the route over ASGI.

Run from repo root: ``pytest tests/test_inference_retrieve_rbac.py``
"""
import os
import sys
from dataclasses import dataclass
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from starlette.testclient import TestClient  # noqa: E402

import inference.main as main_mod  # noqa: E402
import veda.runtime as runtime_mod
import json


@dataclass
class _Result:
    col_id: str
    col_name: str = ""
    table_name: str = ""
    final_score: float = 0.0


_SM = {"tables": {"employee": {}, "department": {}}, "columns": {}}
_RESULTS = [_Result(col_id="employee.id", table_name="employee", col_name="id"),
           _Result(col_id="employee.salary", table_name="employee", col_name="salary"),
           _Result(col_id="department.name", table_name="department", col_name="name")]


class _StubEngine:
    def retrieve(self, query, intent, top_k):
        return list(_RESULTS)


def _client_with_stubbed_engine(monkeypatch):
    # get_engine/_load_scoped_sm are imported locally inside the route function
    # (`from veda.runtime import ...`), so patch them at the source module.
    monkeypatch.setattr(runtime_mod, "get_engine", lambda sm=None: _StubEngine())
    monkeypatch.setattr(runtime_mod, "_load_scoped_sm", lambda: dict(_SM))

    app = main_mod.create_app()
    return TestClient(app)


def test_retrieve_with_no_data_scope_header_returns_every_candidate(monkeypatch):
    client = _client_with_stubbed_engine(monkeypatch)

    resp = client.post("/v1/retrieve", json={
        "query": "salaries", "source_id": 1, "tenant": "default"})

    assert resp.status_code == 200
    col_ids = {c["col_id"] for c in resp.json()["columns"]}
    assert col_ids == {"employee.id", "employee.salary", "department.name"}


def test_retrieve_with_a_restrictive_data_scope_header_narrows_the_results(monkeypatch):
    client = _client_with_stubbed_engine(monkeypatch)
    data_scope = json.dumps({"1": {"open": False, "tables": {"employee": ["id"]}}})

    resp = client.post("/v1/retrieve",
                       headers={"x-veda-source-id": "1", "x-veda-tenant": "default",
                               "x-veda-data-scope": data_scope},
                       json={"query": "salaries", "source_id": 1, "tenant": "default"})

    assert resp.status_code == 200
    col_ids = {c["col_id"] for c in resp.json()["columns"]}
    assert col_ids == {"employee.id"}  # salary and department.name both dropped
