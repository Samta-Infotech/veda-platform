"""Regression coverage for the apps/ enterprise-refactor (2026-08).

The refactor restructured several api-tier modules WITHOUT changing behaviour.
The existing suite only covered apps/chat/{visualization,table_rendering,
thinking_messages}.py — precisely the modules that changed least — so the
heavily-restructured logic had no safety net at all. This file closes that gap
for everything reachable without a live DB / Celery / inference tier:

  1. apps/chat/turn_events.py       — the shared turn-event fold (was duplicated
                                      in views.py's JSON and SSE paths)
  2. apps/query/scope.py            — scope resolution, extracted from
                                      QueryView._resolve_scope
  3. apps/query/inference_client.py — SSE frame parsing
  4. apps/query/views.py            — MultiResult envelope destructuring
  5. apps/evaluation/tasks.py       — HTML escaping of the stored report (the
                                      stored-XSS fix)
  6. apps/ingestion/tasks.py        — engine-output → IngestionStage lifecycle

Django is configured in-process (no DB is ever opened) because 4-6 touch Django
models/enums. 1-3 are Django-free by construction and are exercised as such.

Run from repo root: ``pytest tests/test_apps_layer_refactor.py``
"""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apps.chat.turn_events import TurnEventAccumulator
from apps.query import scope
from apps.query.inference_client import InferenceClient
from apps.query.views import QueryView
from apps.evaluation.tasks import _report_row
from apps.evaluation.tasks import _build_report_html
from apps.ingestion.tasks import _StageTracker
from apps.ingestion.tasks import STAGE_ORDER, _ENGINE_STEP_TO_STAGE, _LAYER_STAGE_TO_ROW
from apps.ingestion.models import JobStatus
from apps.ingestion.tasks import _build_engine_command


def _setup_django():
    """Minimal in-process Django config: app registry only, no DB connection.

    ``config`` is imported first so the ``config/`` package wins over
    ``veda_core/config.py`` — the same name collision the ingestion task avoids
    by running the engine in a subprocess.
    """
    import config
    from django.conf import settings

    assert hasattr(config, "__path__"), (
        "the `config/` PACKAGE must win over veda_core/config.py on sys.path")

    if not settings.configured:
        settings.configure(
            INSTALLED_APPS=[
                "django.contrib.contenttypes", "django.contrib.auth",
                "apps.core", "apps.sources", "apps.substrate",
                "apps.ingestion", "apps.query", "apps.evaluation", "apps.chat",
            ],
            DATABASES={},
            USE_TZ=True,
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        )
        import django
        django.setup()


# ─────────────────────────────────────────────────────────────────────────────
# 1. TurnEventAccumulator — the fold shared by the JSON and SSE response paths
# ─────────────────────────────────────────────────────────────────────────────
def _accumulator():
    return TurnEventAccumulator()


def test_turn_accumulator_starts_empty():
    turn = _accumulator()
    assert turn.content_blocks == []
    assert turn.summary_text == "" and turn.thinking_text == ""
    assert turn.usage == {} and turn.explainability is None and turn.insights is None
    assert turn.metadata() == {"thinking": "", "explainability": None, "usage": {}}


def test_turn_accumulator_folds_a_full_answered_turn():
    """The canonical event order services.run_turn yields for an answered query."""
    turn = _accumulator()
    for kind, payload in [
        ("thinking", {"phase": "classify", "message": "Analyzing your question..."}),
        ("thinking", {"phase": "answer", "message": "Preparing your answer..."}),
        ("content", {"type": "markdown", "content": "You have 42 users.", "is_summary": True}),
        ("content", {"type": "markdown", "content": "| a |\n|---|"}),
        ("visualization", {"visualizations": [{"type": "bar"}]}),
        ("explainability", {"version": "1.0"}),
        ("usage", {"total_tokens": 7, "latency_ms": 12.5}),
        ("insights", {"insights": ["up 3%"], "follow_up_questions": ["why?"]}),
    ]:
        turn.consume(kind, payload)

    # LAST thinking message wins — it is the turn's final "what it did" line.
    assert turn.thinking_text == "Preparing your answer..."
    # content AND visualization both append, in arrival order (one response[] array).
    assert len(turn.content_blocks) == 3
    assert turn.content_blocks[-1] == {"visualizations": [{"type": "bar"}]}
    assert turn.summary_text == "You have 42 users."
    assert turn.usage == {"total_tokens": 7, "latency_ms": 12.5}
    assert turn.insights == {"insights": ["up 3%"], "follow_up_questions": ["why?"]}
    assert turn.metadata() == {"thinking": "Preparing your answer...",
                               "explainability": {"version": "1.0"},
                               "usage": {"total_tokens": 7, "latency_ms": 12.5}}


def test_turn_accumulator_only_is_summary_block_sets_summary():
    """A table block must never overwrite the summary set by the answer block."""
    turn = _accumulator()
    turn.consume("content", {"content": "the answer", "is_summary": True})
    turn.consume("content", {"content": "| table |"})
    assert turn.summary_text == "the answer"
    assert len(turn.content_blocks) == 2


def test_turn_accumulator_ignores_error_and_unknown_kinds():
    """`error` is handled by the views (terminal), never folded here; an unknown
    future event kind must be inert rather than raising."""
    turn = _accumulator()
    turn.consume("error", {"code": "MODEL_ERROR", "message": "boom"})
    turn.consume("some_future_event", {"anything": True})
    assert turn.content_blocks == []
    assert turn.metadata() == {"thinking": "", "explainability": None, "usage": {}}


def test_turn_accumulator_tolerates_missing_payload_keys():
    turn = _accumulator()
    turn.consume("thinking", {})                       # no "message"
    turn.consume("content", {"is_summary": True})      # no "content"
    assert turn.thinking_text == ""
    assert turn.summary_text == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. resolve_query_scope — server-side scope resolution (never trusts the body)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def scope_module(monkeypatch):
    def _with_ready(ready):
        monkeypatch.setattr(scope, "_ready_source_ids", lambda: list(ready))
        return scope
    return _with_ready


def test_scope_defaults_to_all_ready_sources(scope_module):
    scope = scope_module([3, 7])
    assert scope.resolve_query_scope({}, "default") == [3, 7]


def test_scope_falls_back_to_env_default_when_nothing_ready(scope_module, monkeypatch):
    scope = scope_module([])
    monkeypatch.setenv("VEDA_DEFAULT_SOURCE_ID", "5")
    assert scope.resolve_query_scope({}, "default") == [5]


def test_scope_intersects_request_pin_with_ownership(scope_module):
    """A pin the tenant does not own is dropped — the core §6.2 guarantee."""
    scope = scope_module([3, 7])
    assert scope.resolve_query_scope({"source_ids": [7, 3]}, "default") == [7, 3]
    assert scope.resolve_query_scope({"source_id": 3}, "default") == [3]
    # 9 is not ready → no usable pin → fall back to the full ready scope.
    assert scope.resolve_query_scope({"source_id": 9}, "default") == [3, 7]


def test_scope_preserves_pin_order_and_dedupes(scope_module):
    scope = scope_module([3, 7])
    assert scope.resolve_query_scope({"source_ids": [7, 3, 7]}, "default") == [7, 3]


def test_scope_trusts_pin_when_registry_unreadable(scope_module):
    """Registry down = unknown ownership: honour the explicit pin rather than
    failing the request (documented fallback)."""
    scope = scope_module([])
    assert scope.resolve_query_scope({"source_id": 9}, "default") == [9]


def test_scope_coerces_and_rejects_malformed_pins(scope_module):
    scope = scope_module([3, 7])
    assert scope.resolve_query_scope({"source_ids": ["3", "7"]}, "default") == [3, 7]
    # Malformed pin is ignored, NOT an error — falls through to the default scope.
    assert scope.resolve_query_scope({"source_ids": ["x"]}, "default") == [3, 7]
    assert scope.resolve_query_scope({"source_ids": "notalist"}, "default") == [3, 7]
    assert scope.resolve_query_scope({"source_ids": []}, "default") == [3, 7]


def test_scope_never_returns_empty(scope_module, monkeypatch):
    """The fail-closed context seam (§4.1) always needs at least one source."""
    monkeypatch.setenv("VEDA_DEFAULT_SOURCE_ID", "1")
    for ready in ([], [3, 7]):
        scope = scope_module(ready)
        for data in ({}, {"source_id": 999}, {"source_ids": ["bad"]}):
            assert scope.resolve_query_scope(data, "default")


# ─────────────────────────────────────────────────────────────────────────────
# 3. InferenceClient SSE frame parsing
# ─────────────────────────────────────────────────────────────────────────────
def _frames(raw: bytes):
    return list(InferenceClient._iter_sse_frames(io.BytesIO(raw)))


def test_sse_parses_frames_in_order():
    assert _frames(b'event: progress\ndata: {"a": 1}\n\n'
                   b'event: result\ndata: {"done": true}\n\n') == [
        ("progress", {"a": 1}), ("result", {"done": True})]


def test_sse_joins_multiline_data():
    assert _frames(b'event: e\ndata: {"b":\ndata:  2}\n\n') == [("e", {"b": 2})]


def test_sse_degrades_bad_or_missing_data_to_empty_dict():
    """One malformed progress frame must never abort an otherwise healthy stream."""
    assert _frames(b'event: nodata\n\nevent: bad\ndata: not-json\n\n') == [
        ("nodata", {}), ("bad", {})]


def test_sse_drops_frame_without_event_name():
    assert _frames(b'data: {"orphan": true}\n\n') == []


def test_sse_handles_crlf_and_ignores_unterminated_tail():
    assert _frames(b'event: a\r\ndata: {"x": 1}\r\n\r\nevent: never\ndata: {}') == [
        ("a", {"x": 1})]


# ─────────────────────────────────────────────────────────────────────────────
# 4. QueryView._first_item_fields — MultiResult envelope destructuring
# ─────────────────────────────────────────────────────────────────────────────
def _first_item_fields(result):
    _setup_django()
    return QueryView._first_item_fields(result)


def test_first_item_fields_reads_route_and_result():
    route, res = _first_item_fields(
        {"items": [{"route": "sql", "result": {"sql": "SELECT 1", "table": "(cached)"}}]})
    assert route == "sql"
    assert res["sql"] == "SELECT 1"


@pytest.mark.parametrize("envelope", [
    {}, {"items": []}, {"items": [None]}, {"items": ["nope"]},
    {"items": [{}]}, {"items": [{"result": None}]}, {"items": [{"result": "notadict"}]},
    "not-a-dict", None,
])
def test_first_item_fields_never_raises_on_unexpected_envelopes(envelope):
    """The view must degrade to empty values, never 500, on a malformed payload."""
    route, res = _first_item_fields(envelope)
    assert route == "" and res == {}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Eval report HTML escaping — the stored-XSS fix
# ─────────────────────────────────────────────────────────────────────────────
def test_eval_report_row_escapes_every_interpolated_value():
    """report_html is rendered back in Django admin, so caller-supplied query
    text and generated SQL must not be able to inject markup."""
    _setup_django()

    row = _report_row("D01", "DIRECT", "<script>alert(1)</script>", "ok", 12,
                      "SELECT * FROM t WHERE a < 5 AND b > 2")
    assert "<script>" not in row
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in row
    assert "a &lt; 5 AND b &gt; 2" in row
    # The row's own structural markup is still intact.
    assert row.startswith("<tr><td>") and row.endswith("</td></tr>")


def test_eval_report_html_escapes_the_header_fields():
    _setup_django()

    class _Run:
        label = '<img src=x onerror=alert(1)>'
        sql_success_rate = 0.5

    html = _build_report_html(_Run(), 1, "default", 1, 2, [])
    assert "<img" not in html
    assert "&lt;img" in html
    assert "<h2>VEDA Eval Run</h2>" in html


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ingestion: engine output → IngestionStage lifecycle
# ─────────────────────────────────────────────────────────────────────────────
class _FakeStage:
    """Stand-in for an IngestionStage row — records saves, never touches a DB."""

    def __init__(self, name):
        self.name = name
        self.status = "pending"
        self.started_at = None
        self.finished_at = None
        self.batch_checkpoint = {}
        self.saves = 0

    def save(self, update_fields=None):
        self.saves += 1


def _tracker(names):
    _setup_django()
    stages = {n: _FakeStage(n) for n in names}
    return _StageTracker(stages), stages


def test_engine_step_map_covers_every_step_and_only_real_stages():
    """Guards the table that replaced a per-output-line dict rebuild."""
    _setup_django()

    assert sorted(_ENGINE_STEP_TO_STAGE) == list(range(1, 13)), "engine steps 1..12"
    known = {name for _o, name, _q in STAGE_ORDER}
    assert set(_ENGINE_STEP_TO_STAGE.values()) <= known
    assert set(_LAYER_STAGE_TO_ROW.values()) <= known


def test_stage_tracker_marks_status_and_timestamps():
    tracker, stages = _tracker(["schema_scan"])

    tracker.mark(["schema_scan"], JobStatus.RUNNING)
    assert stages["schema_scan"].status == JobStatus.RUNNING
    assert stages["schema_scan"].started_at is not None
    assert stages["schema_scan"].finished_at is None

    started = stages["schema_scan"].started_at
    tracker.mark(["schema_scan"], JobStatus.SUCCESS)
    assert stages["schema_scan"].finished_at is not None
    assert stages["schema_scan"].started_at == started, "started_at must not be re-stamped"


def test_stage_tracker_ignores_unknown_stage_names():
    tracker, stages = _tracker(["schema_scan"])
    tracker.mark(["not_a_stage"], JobStatus.SUCCESS)   # must not raise
    tracker.update_checkpoint("not_a_stage", x=1)      # must not raise
    assert stages["schema_scan"].saves == 0


def test_stage_tracker_merges_checkpoint_without_dropping_keys():
    tracker, stages = _tracker(["embeddings"])
    tracker.update_checkpoint("embeddings", resume=True)
    tracker.update_checkpoint("embeddings", engine_step=8)
    assert stages["embeddings"].batch_checkpoint == {"resume": True, "engine_step": 8}


def test_step_markers_advance_stages_and_close_the_previous_one():
    from apps.ingestion.tasks import _consume_engine_output

    tracker, stages = _tracker(["schema_scan", "fk_adjacency", "data_graph"])
    process = type("P", (), {"stdout": [
        "[1/12] SchemaScan\n", "[1/12] SchemaScan again\n",   # same stage → no churn
        "[2/12] FkAdjacency\n", "[3/12] DataGraph\n",
    ]})()
    tail, active_marker, active_row = _consume_engine_output(process, tracker)

    assert stages["schema_scan"].status == JobStatus.SUCCESS
    assert stages["fk_adjacency"].status == JobStatus.SUCCESS
    assert stages["data_graph"].status == JobStatus.RUNNING, "last stage stays in flight"
    assert active_marker == "data_graph" and active_row is None
    assert stages["schema_scan"].batch_checkpoint["engine_step"] == 1
    assert len(tail) == 4


def test_layer_stage_events_roll_up_and_transition_rows():
    from apps.ingestion.tasks import _consume_engine_output

    tracker, stages = _tracker(["schema_scan", "embeddings", "vector_store"])
    process = type("P", (), {"stdout": [
        "[[STAGE]] L1 schema_scan ok\n",
        "[[STAGE]] L3 reg_graph ok\n",      # rolls up to embeddings
        "[[STAGE]] L3 join_paths ok\n",     # same row, re-confirms
        "[[STAGE]] L4 biencoder ok\n",      # rolls up to vector_store
    ]})()
    _tail, active_marker, active_row = _consume_engine_output(process, tracker)

    assert stages["schema_scan"].status == JobStatus.SUCCESS
    assert stages["embeddings"].status == JobStatus.SUCCESS
    assert stages["embeddings"].batch_checkpoint["layer_stage"] == "join_paths"
    assert active_row == "vector_store" and active_marker is None


def test_fatal_layer_event_marks_failed_and_is_not_overwritten_on_transition():
    """A row that went FATAL must not be flipped to SUCCESS by the next row's
    transition — the guard in _apply_stage_event."""
    from apps.ingestion.tasks import _consume_engine_output

    tracker, stages = _tracker(["embeddings", "vector_store"])
    process = type("P", (), {"stdout": [
        "[[STAGE]] L3 reg_graph fatal\n",
        "[[STAGE]] L4 biencoder ok\n",
    ]})()
    _consume_engine_output(process, tracker)
    assert stages["embeddings"].status == JobStatus.FAILED
    assert stages["vector_store"].status == JobStatus.SUCCESS


def test_unmapped_layer_stage_is_ignored():
    from apps.ingestion.tasks import _consume_engine_output

    tracker, stages = _tracker(["schema_scan"])
    process = type("P", (), {"stdout": ["[[STAGE]] L9 totally_unknown ok\n"]})()
    _tail, _m, active_row = _consume_engine_output(process, tracker)
    assert active_row is None
    assert stages["schema_scan"].saves == 0


def test_output_tail_is_bounded():
    """The failure message embeds this tail — it must not grow without bound."""
    from apps.ingestion.tasks import _OUTPUT_TAIL_MAX_LINES, _consume_engine_output

    tracker, _stages = _tracker(["schema_scan"])
    process = type("P", (), {"stdout": [f"line {i}\n" for i in range(1000)]})()
    tail, _m, _r = _consume_engine_output(process, tracker)
    assert len(tail) == _OUTPUT_TAIL_MAX_LINES
    assert tail[-1] == "line 999\n", "the tail must keep the MOST RECENT lines"


def test_engine_command_routes_by_source_kind():
    """Relational → run_ingestion; anything else → the dispatcher, with the
    source config passed via env rather than interpolated into the program."""
    _setup_django()

    class _Src:
        def __init__(self, kind):
            self._kind = kind

        def source_kind(self):
            return self._kind

        def as_source_config(self):
            return {"id": "1", "type": self._kind, "path": "/data/x"}

    env = {}
    code = _build_engine_command(_Src("relational"), env, skip_llm=True)
    assert "main.run_ingestion" in code and "skip_llm=True" in code
    assert "VEDA_SOURCE_JSON" not in env

    env = {}
    code = _build_engine_command(_Src("document"), env, skip_llm=False)
    assert "dispatch_ingestion" in code
    assert '"path": "/data/x"' in env["VEDA_SOURCE_JSON"]
    assert "/data/x" not in code, "source config must not be inlined into the program text"


def test_engine_command_defaults_to_relational_without_a_source():
    _setup_django()
    assert "main.run_ingestion" in _build_engine_command(None, {}, skip_llm=False)
