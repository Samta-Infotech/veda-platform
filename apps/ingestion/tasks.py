"""apps.ingestion.tasks — the API-triggered ingestion entrypoint (migration_plan.md §7).

``task_ingest_source`` is the single ingestion path: it records an ``IngestionJob``
(+ ordered ``IngestionStage`` rows from ``STAGE_ORDER`` for observability), injects
THIS source's connection, and runs the engine pipeline in an isolated subprocess —
routed by source type (relational → ``main.run_ingestion``; nosql/document/datalake
→ ``source_dispatcher.dispatch_ingestion``). ``task_warm_caches`` then syncs the
Django substrate from the engine store and rehydrates caches.

The engine step *logic* lives in ``veda_core/ingestion/``; the subprocess streams
its ``[N/NN] StageName`` markers back to live ``IngestionStage`` updates so admin
shows true per-stage progress.

Django / veda_core / Celery-app imports are kept function-local on purpose: this
module must stay importable in the thin api image, which omits the ML dependency
chain those imports pull in (§1.3). Stdlib imports have no such constraint and
live at module scope.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from django.utils import timezone
from django.conf import settings

try:
    from celery import shared_task
except ImportError:  # celery not installed in this environment — keep the module importable
    def shared_task(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        return _wrap
from veda_core.context import RequestContext, set_context
from storage_adapters import writer
from veda_core import config
from apps.access_management.services import CatalogDiscoveryService


logger = logging.getLogger(__name__)


# Ordered stage registry (§7 table). The task base class (Phase 3.5) sets the
# ambient (source, tenant) context before any veda_core function runs.
STAGE_ORDER = [
    (1, "schema_scan", "ingestion"),
    (2, "fk_adjacency", "ingestion"),
    (3, "data_graph", "ingestion"),
    (4, "semantic_types", "ingestion"),
    (5, "value_profiling", "ingestion"),
    (6, "embeddings", "ingestion"),      # batched commits (§4.2a)
    (7, "vector_store", "ingestion"),
    (8, "derived_language", "ingestion"),
    (9, "unified_graph", "ingestion"),
    (10, "warm_caches", "high"),
]

WARM_CACHES_STAGE = "warm_caches"

# Default embedding model id when veda_core.config does not declare one (WP3).
_DEFAULT_EMBEDDING_MODEL_ID = "bge-m3"

# Environment keys read/written around the engine subprocess.
_ENV_APP_DIR = "VEDA_APP_DIR"
_DEFAULT_APP_DIR = "/app"
_ENV_ARTIFACT_SCOPING = "VEDA_ARTIFACT_SCOPING"
_ARTIFACT_SCOPING_ENABLED = "1"

# How much engine output to retain for a failure message, and how much of that
# tail to actually surface — bounded so a chatty pipeline can't blow up the
# exception string or the worker's memory.
_OUTPUT_TAIL_MAX_LINES = 200
_ERROR_TAIL_MAX_CHARS = 1500
# Engine progress-marker label recorded on the stage checkpoint, truncated to
# keep the JSON checkpoint small.
_MARKER_LABEL_MAX_CHARS = 80


# Layered stage name (ingestion/layers) → STAGE_ORDER row name (§2.2). Multiple
# fine-grained layer stages roll up into one observable STAGE_ORDER row.
_LAYER_STAGE_TO_ROW = {
    "schema_scan": "schema_scan", "fk_adjacency": "fk_adjacency",
    "data_graph": "data_graph", "semantic_types": "semantic_types",
    "table_metadata": "semantic_types", "value_profiling": "value_profiling",
    "reg_graph": "embeddings", "join_paths": "embeddings",
    "graph_persist": "embeddings", "graph_embed": "embeddings",
    "sparse_index": "embeddings", "enrichment_index": "embeddings",
    "rerank_docs": "embeddings",
    # BGE biencoder is the live vector store (column_embeddings_v2) now that the
    # ensemble encoder + _lt/_hybrid store were removed — map it to the vector_store row.
    "biencoder": "vector_store",
    "semantic_layer": "derived_language",
    "relationship_graph": "derived_language", "semantic_registry": "derived_language",
    "value_mirror": "derived_language", "hnsw_tune": "derived_language",
    "unified_graph": "unified_graph",
}

# Engine step index (1..12, incl 7b/9b) → STAGE_ORDER row name, in order. Several
# engine steps roll up into one observable row (e.g. steps 10-12 → derived_language).
# Built once at import: this was previously rebuilt from a list of tuples via
# `dict(...)` on EVERY line of subprocess output.
_ENGINE_STEP_TO_STAGE = {
    1: "schema_scan", 2: "fk_adjacency", 3: "data_graph", 4: "semantic_types",
    5: "semantic_types", 6: "value_profiling", 7: "unified_graph",
    8: "embeddings", 9: "vector_store", 10: "derived_language",
    11: "derived_language", 12: "derived_language",
}

# "[N/NN] StageName" progress markers emitted by the engine's monolith path.
_MARKER_RE = re.compile(r"\[(\d+)[ab]?/\d+\]\s+([A-Za-z][^\(\n]+)")
# Layered mode (P4): "[[STAGE]] <layer> <stage> <ok|fail|fatal>" events drive
# IngestionStage rows from real lifecycle instead of regex-parsing progress bars.
_STAGE_EVENT_RE = re.compile(r"\[\[STAGE\]\]\s+(\S+)\s+(\S+)\s+(ok|fail|fatal)")
_STAGE_EVENT_OK = "ok"
_STAGE_EVENT_FATAL = "fatal"


@shared_task(queue="high")
def task_warm_caches(prev=None, source_id=None, tenant="default"):
    """Sync Django substrate from the engine store + publish sm + rehydrate fan-out (§8.4)."""

    set_context(RequestContext(source_id=int(source_id), tenant=str(tenant)))
    return writer.warm()


class _StageTracker:
    """Owns the ``IngestionStage`` rows of one job and their lifecycle transitions.

    Extracted from ``task_ingest_source``'s local ``_mark`` closure so the stage
    bookkeeping (status + timestamps + checkpoint JSON) is one testable unit with
    a name, rather than a closure over the task body's locals.
    """

    def __init__(self, stages_by_name: dict):
        self._stages = stages_by_name

    def mark(self, names, status) -> None:
        """Set ``status`` on each named stage, stamping start/finish timestamps.

        Unknown names are ignored — the engine may emit a layer stage that does
        not roll up to an observable row.
        """
        from apps.ingestion.models import JobStatus

        for name in names:
            stage = self._stages.get(name)
            if not stage:
                continue
            stage.status = status
            if status == JobStatus.RUNNING and not stage.started_at:
                stage.started_at = timezone.now()
            if status in (JobStatus.SUCCESS, JobStatus.FAILED):
                stage.finished_at = timezone.now()
            stage.save()

    def status_of(self, name):
        stage = self._stages.get(name)
        return stage.status if stage is not None else None

    def update_checkpoint(self, name: str, **fields) -> None:
        """Merge ``fields`` into a stage's ``batch_checkpoint`` JSON (§4.2a)."""
        stage = self._stages.get(name)
        if stage is None:
            return
        checkpoint = dict(stage.batch_checkpoint or {})
        checkpoint.update(fields)
        stage.batch_checkpoint = checkpoint
        stage.save(update_fields=["batch_checkpoint"])


@shared_task(queue="ingestion")
def task_ingest_source(source_id=None, tenant="default", verbose=True, force=False,
                       skip_llm=False, resume=False, ingestion_mode=None):
    """Run the preserved L0 orchestration and track it as an IngestionJob (§4.3).

    Calls ``veda_core.main.run_ingestion`` (the verbatim pipeline) directly rather
    than re-deriving the ten-stage chain — the logic is PRESERVED (§4.0). The job
    row records status/timing; Source.ready flips only on full success.

    Args:
        source_id: The Source row to ingest. Required — see the guard below.
        tenant: Tenant that owns the run; stamped on the job and the engine context.
        verbose: Accepted for signature compatibility; the engine subprocess is
            always run non-verbose and its output is streamed instead.
        force: Bypass the embedding-model-change guard (a changed embedding space
            invalidates every stored vector, §12).
        skip_llm: Skip the LLM-backed semantic-layer stage (relational path only).
        resume: Force VEDA_RESUME=1; also auto-detected from a prior failed job.
        ingestion_mode: Accepted for signature compatibility; the legacy monolith
            mode selector was removed in P7 (run_ingestion IS the layered pipeline).

    Returns:
        dict: ``{job_id, ok, source_id, warm}`` on success.

    Raises:
        ValueError: if ``source_id`` is not supplied.
        RuntimeError: on an embedding-model change without ``force``, or a non-zero
            engine subprocess exit.

    NOTE: embedding stages need torch/sentence-transformers, which the thin api/
    worker image intentionally omits (§1.3). Run this task on a worker built from
    the inference image (ML deps) or via the one-off inference-image runner used in
    dev. Kept import-lazy so the module still loads in the thin image.
    """

    from apps.ingestion.models import IngestionJob, IngestionStage, JobStatus
    from apps.sources.models import Source, SourceStatus

    # A missing source_id used to silently default to 1 — which stamped every
    # internal-DB row (embeddings, graph_nodes, …) and the request context as
    # source 1, and skipped the Source row update, so a job for another source
    # masqueraded as source 1. Fail loudly instead: the whole task assumes a real
    # Source row (job.source / as_engine_env), so None was never valid anyway.
    if source_id is None:
        raise ValueError(
            "task_ingest_source requires an explicit source_id; refusing to "
            "default to source 1 (that mislabels the job and its artifacts)."
        )
    set_context(RequestContext(source_id=int(source_id), tenant=str(tenant)))

    # WP3: stamp the embedding model id. The IngestionJob.encoder_mode
    # column is reused to hold it — same purpose: refuse a silent model change between
    # resume runs, which would mix incompatible embedding spaces.
    encoder_mode = getattr(config, "EMBEDDING_MODEL_ID", _DEFAULT_EMBEDDING_MODEL_ID)
    job = IngestionJob.objects.create(
        source_id=source_id, tenant=tenant, status=JobStatus.RUNNING,
        encoder_mode=encoder_mode, started_at=timezone.now(),
    )
    # Create the ordered stage rows (pending) for observability (§7 table).
    tracker = _StageTracker({
        name: IngestionStage.objects.create(job=job, order=order, name=name,
                                            status=JobStatus.PENDING)
        for order, name, _queue in STAGE_ORDER
    })
    logger.info("ingestion job started job_id=%s source_id=%s tenant=%s encoder=%s",
                job.pk, source_id, tenant, encoder_mode)

    _guard_embedding_model_change(job, encoder_mode, force)

    try:
        result_source_id = _run_engine_pipeline(job, tracker, source_id, tenant,
                                                skip_llm=skip_llm, resume=resume)
        tracker.mark([n for _o, n, _q in STAGE_ORDER if n != WARM_CACHES_STAGE],
                     JobStatus.SUCCESS)

        # warm stage: sync Django substrate + publish sm + rehydrate fan-out.
        tracker.mark([WARM_CACHES_STAGE], JobStatus.RUNNING)
        warm_counts = task_warm_caches(source_id=source_id, tenant=tenant)
        tracker.mark([WARM_CACHES_STAGE], JobStatus.SUCCESS)

        job.status = JobStatus.SUCCESS
        if source_id:
            Source.objects.filter(pk=source_id).update(
                ready=True, status=SourceStatus.READY, last_ingested_at=timezone.now(),
            )
            _sync_catalog_if_enabled(source_id)
            # Multi-source routing (Phase 1.3): generate a grounded source description from the
            # just-synced substrate schema. Flag-gated default-OFF and never-raises, so it can
            # never turn a successful ingestion into a failure.
            from apps.sources.source_profiler import profile_source_if_enabled
            profile_source_if_enabled(source_id, tenant=tenant)
            # Multi-source routing: build + profile the uniform SourceItem layer (per-item summary +
            # embedding for the query-time routing prior). Flag-gated default-OFF, never-raises.
            from apps.sources.item_profiler import build_source_items_if_enabled
            build_source_items_if_enabled(source_id)
        logger.info("ingestion job succeeded job_id=%s source_id=%s", job.pk, source_id)
        return {"job_id": job.pk, "ok": True, "source_id": result_source_id,
                "warm": warm_counts}
    except Exception:  # record failure, don't crash the worker
        logger.exception("ingestion job failed job_id=%s source_id=%s", job.pk, source_id)
        job.status = JobStatus.FAILED
        tracker.mark([n for _o, n, _q in STAGE_ORDER], JobStatus.FAILED)
        if source_id:
            Source.objects.filter(pk=source_id).update(status=SourceStatus.FAILED)
        raise
    finally:
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])


def _sync_catalog_if_enabled(source_id) -> None:
    """Reconcile the RBAC catalog projection for one just-ingested source.

    Flag-gated (``VEDA_AUTO_SYNC_CATALOG``, default OFF) — see the setting's own
    comment in ``config/settings/base.py`` for why. Off, this function does not
    even import Django's access_management models, so the flag-off path costs
    nothing beyond the getattr.

    Never allowed to fail the ingestion job that just succeeded: a broken catalog
    projection is a real problem, but it is a DIFFERENT, already-recoverable one
    (rerun ``manage.py sync_catalog``) — turning a successful ingestion into a
    failed job over it would be a strictly worse outcome for an operator to debug.
    """

    if not getattr(settings, "VEDA_AUTO_SYNC_CATALOG", False):
        return

    from apps.sources.models import Source

    try:
        source = Source.objects.get(pk=source_id)
        report = CatalogDiscoveryService().sync_source(source)
        logger.info("catalog auto-sync succeeded source_id=%s %s",
                   source_id, report.as_dict())
    except Exception:
        logger.exception("catalog auto-sync failed source_id=%s — ingestion job "
                         "still succeeded; run `manage.py sync_catalog` to retry",
                         source_id)


def _guard_embedding_model_change(job, encoder_mode: str, force: bool) -> None:
    """Embedding-model guard (§7): refuse if the model id differs from the persisted
    one without an explicit force flag (re-ingestion required — a changed embedding
    space invalidates every stored vector, per §12).

    Marks the job FAILED before raising so the row never lingers as RUNNING.
    """

    from apps.ingestion.models import JobStatus

    last_successful_job = job.source.ingestion_jobs.exclude(pk=job.pk).filter(
        status=JobStatus.SUCCESS).order_by("-id").first()
    if not (last_successful_job and last_successful_job.encoder_mode
            and last_successful_job.encoder_mode != encoder_mode and not force):
        return

    job.status = JobStatus.FAILED
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "finished_at"])
    raise RuntimeError(
        f"embedding model changed {last_successful_job.encoder_mode!r}→{encoder_mode!r}; "
        "re-ingestion required — pass force=True (§12)."
    )


def _run_engine_pipeline(job, tracker: _StageTracker, source_id, tenant, *,
                         skip_llm: bool, resume: bool) -> str:
    """Run the engine pipeline in a subprocess, driving IngestionStage rows from
    its streamed output. Returns the ``source_id`` string for the task result.

    Run in a SUBPROCESS because the engine imports a top-level ``config``
    (config.py) that collides with this Django project's ``config`` package in one
    interpreter. A subprocess gives it its own sys.modules — clean isolation.
    ``cwd=veda_core`` so the engine's relative paths (data/, schema/, client_bge)
    resolve.

    NOTE: the engine passes intermediate artifacts in-memory between steps, so true
    mid-run resume-from-stage-N would require the §4.0 artifact-persistence
    extraction; here a failed job records exactly which stage failed (the ones
    before it stay success), and a re-run restarts the idempotent pipeline. The
    batched-stage-6 checkpoint is recorded per encoder table.
    """
    from apps.ingestion.models import JobStatus

    source = job.source
    veda_core_dir = os.path.join(os.environ.get(_ENV_APP_DIR, _DEFAULT_APP_DIR), "veda_core")
    subprocess_env = _build_subprocess_env(job, tracker, source, source_id, tenant, resume=resume)
    python_code = _build_engine_command(source, subprocess_env, skip_llm=skip_llm)

    process = subprocess.Popen(
        ["python", "-u", "-c", python_code],
        cwd=veda_core_dir, env=subprocess_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output_tail, active_marker_stage, active_event_row = _consume_engine_output(process, tracker)
    process.wait()

    if process.returncode != 0:
        if active_event_row:
            tracker.mark([active_event_row], JobStatus.FAILED)
        if active_marker_stage:
            tracker.mark([active_marker_stage], JobStatus.FAILED)
        raise RuntimeError(f"run_ingestion subprocess failed (rc={process.returncode}): "
                           f"{''.join(output_tail)[-_ERROR_TAIL_MAX_CHARS:]}")

    if active_event_row:
        tracker.mark([active_event_row], JobStatus.SUCCESS)
    if active_marker_stage:
        tracker.mark([active_marker_stage], JobStatus.SUCCESS)
    # "primary_db" is the historical label for a run with no explicit source id.
    return str(source_id) if source_id else "primary_db"


def _build_subprocess_env(job, tracker: _StageTracker, source, source_id, tenant, *,
                          resume: bool) -> dict:
    """The environment handed to the engine subprocess.

    Carries THIS source's DB connection (§5) so ingestion targets the right source
    without any global env/code change, the tenant, the opt-in per-source artifact
    scope (P3/§3.1), and the resume flag.
    """
    subprocess_env = dict(os.environ)
    if source and source.host:
        subprocess_env.update(source.as_engine_env())

    # P7: the engine's run_ingestion IS the layered L1–L5 pipeline now (single
    # ingestion path — the legacy monolith + INGESTION_MODE flag were removed).
    # It emits "[[STAGE]] <layer> <stage> <status>" events parsed into real
    # per-stage lifecycle.
    subprocess_env["VEDA_TENANT"] = str(tenant)

    # Per-source artifact scope (P3/§3.1) is OPT-IN (VEDA_ARTIFACT_SCOPING=1): only
    # when the query tier is also scope-aware (P5) do artifacts move off the flat
    # data/ paths. Off by default so single-source behaviour stays byte-identical.
    if os.environ.get(_ENV_ARTIFACT_SCOPING) == _ARTIFACT_SCOPING_ENABLED:
        subprocess_env["VEDA_ARTIFACT_SCOPE"] = f"{tenant}/{source_id}/{job.pk}"

    # Resume (§4.2a/P8-B5): auto-detect from a prior failed job for this source, OR
    # explicit resume=True. VEDA_RESUME=1 makes the engine skip the expensive stages
    # (LLM semantic-layer, biencoder embeddings) when their persisted output exists,
    # while the fast prep stages re-run to rebuild the in-memory context.
    if _should_resume(job, resume):
        subprocess_env["VEDA_RESUME"] = "1"
        job.stages.filter(name__in=("embeddings", "derived_language")).update(
            batch_checkpoint={"resume": True})
    return subprocess_env


def _should_resume(job, resume: bool) -> bool:
    from apps.ingestion.models import JobStatus

    if resume:
        return True
    last_failed_job = job.source.ingestion_jobs.exclude(pk=job.pk).filter(
        status=JobStatus.FAILED).order_by("-id").first()
    return bool(last_failed_job)


def _build_engine_command(source, subprocess_env: dict, *, skip_llm: bool) -> str:
    """Source-type routing (§5) → the ``python -c`` program for the subprocess.

    Relational sources flow through the full, proven ``run_ingestion`` pipeline
    (their per-source connection is injected via ``as_engine_env``). Non-relational
    sources (nosql/document/datalake) — which ``run_ingestion`` cannot handle — are
    routed by type through ``source_dispatcher.dispatch_ingestion``, receiving this
    source's config as JSON (single source of truth = the DB Source row), passed via
    ``subprocess_env`` rather than interpolated into the program text.
    """
    source_kind = source.source_kind() if source else "relational"
    if source_kind == "relational":
        return f"import main; main.run_ingestion(verbose=False, skip_llm={bool(skip_llm)})"

    subprocess_env["VEDA_SOURCE_JSON"] = json.dumps(source.as_source_config())
    return (
        "import os, json; "
        "from ingestion.source_dispatcher import dispatch_ingestion; "
        "cfg = json.loads(os.environ['VEDA_SOURCE_JSON']); "
        "r = dispatch_ingestion(cfg, verbose=False); "
        "print('[dispatch] type=%s success=%s' % (r.source_type, r.success)); "
        "raise SystemExit(0 if r.success else 1)"
    )


def _consume_engine_output(process, tracker: _StageTracker) -> tuple[list[str], str | None, str | None]:
    """Stream the subprocess stdout, driving live IngestionStage updates.

    Returns ``(output_tail, active_marker_stage, active_event_row)`` — the retained
    tail for a failure message, plus whichever stage/row each of the two progress
    protocols left in flight, so the caller can close them out as SUCCESS or FAILED.
    """
    output_tail: list[str] = []
    active_marker_stage: str | None = None   # "[N/NN]" marker protocol
    active_event_row: str | None = None      # "[[STAGE]]" event protocol

    for line in process.stdout:
        # Echo the engine subprocess output to the worker's stdout so ingestion
        # progress is visible live via `docker compose logs -f ingest-worker`
        # (the lines are also parsed below into IngestionStage rows).
        print(line, end="", flush=True)
        output_tail.append(line)
        if len(output_tail) > _OUTPUT_TAIL_MAX_LINES:
            output_tail.pop(0)

        stage_event = _STAGE_EVENT_RE.search(line)
        if stage_event:
            active_event_row = _apply_stage_event(tracker, stage_event, active_event_row)
            continue

        marker = _MARKER_RE.search(line)
        if marker:
            active_marker_stage = _apply_step_marker(tracker, marker, active_marker_stage)

    return output_tail, active_marker_stage, active_event_row


def _apply_stage_event(tracker: _StageTracker, stage_event, active_event_row: str | None) -> str | None:
    """Handle one ``[[STAGE]] <layer> <stage> <status>`` event; returns the row now active."""
    from apps.ingestion.models import JobStatus

    layer, layer_stage, event_status = stage_event.group(1), stage_event.group(2), stage_event.group(3)
    row_name = _LAYER_STAGE_TO_ROW.get(layer_stage)
    if not row_name:
        return active_event_row

    # Row transition = the previous observable row finished all its constituent
    # layer stages → mark it SUCCESS *now* (real per-stage timing in admin), not
    # en masse at the end.
    if active_event_row and active_event_row != row_name:
        if tracker.status_of(active_event_row) not in (None, JobStatus.FAILED):
            tracker.mark([active_event_row], JobStatus.SUCCESS)

    # The pipeline emits "[[STAGE]] … ok" AFTER a stage completes, so ok → SUCCESS
    # (mark done). Rolled-up rows (several layer stages → one row) simply
    # re-confirm SUCCESS as each sub-stage lands.
    if event_status == _STAGE_EVENT_OK:
        tracker.mark([row_name], JobStatus.SUCCESS)
    elif event_status == _STAGE_EVENT_FATAL:
        tracker.mark([row_name], JobStatus.FAILED)
    tracker.update_checkpoint(row_name, layer_stage=layer_stage, layer=layer)
    return row_name


def _apply_step_marker(tracker: _StageTracker, marker, active_marker_stage: str | None) -> str | None:
    """Handle one ``[N/NN] StageName`` progress marker; returns the stage now active."""
    from apps.ingestion.models import JobStatus

    engine_step = int(marker.group(1))
    stage_name = _ENGINE_STEP_TO_STAGE.get(engine_step)
    if not stage_name or stage_name == active_marker_stage:
        return active_marker_stage

    if active_marker_stage:
        tracker.mark([active_marker_stage], JobStatus.SUCCESS)
    tracker.mark([stage_name], JobStatus.RUNNING)
    # Record which engine step is in-flight for stage-6 batch visibility (§4.2a).
    tracker.update_checkpoint(stage_name, engine_step=engine_step,
                              marker=marker.group(2).strip()[:_MARKER_LABEL_MAX_CHARS])
    return stage_name
