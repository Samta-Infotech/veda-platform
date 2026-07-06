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
"""
from __future__ import annotations

try:
    from celery import shared_task
except ImportError:  # celery not installed in this environment — keep the module importable
    def shared_task(*d_args, **d_kwargs):
        def _wrap(fn):
            return fn
        return _wrap


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


@shared_task(queue="high")
def task_warm_caches(prev=None, source_id=None, tenant="default"):
    """Sync Django substrate from the engine store + publish sm + rehydrate fan-out (§8.4)."""
    from veda_core.context import RequestContext, set_context
    from storage_adapters import writer

    set_context(RequestContext(source_id=int(source_id), tenant=str(tenant)))
    return writer.warm()


@shared_task(queue="ingestion")
def task_ingest_source(source_id=None, tenant="default", verbose=True, force=False,
                       skip_llm=False, resume=False):
    """Run the preserved L0 orchestration and track it as an IngestionJob (§4.3).

    Calls ``veda_core.main.run_ingestion`` (the verbatim pipeline) directly rather
    than re-deriving the ten-stage chain — the logic is PRESERVED (§4.0). The job
    row records status/timing; Source.ready flips only on full success.

    NOTE: embedding stages need torch/sentence-transformers, which the thin api/
    worker image intentionally omits (§1.3). Run this task on a worker built from
    the inference image (ML deps) or via the one-off inference-image runner used in
    dev. Kept import-lazy so the module still loads in the thin image.
    """
    import os

    from django.utils import timezone

    from apps.ingestion.models import IngestionJob, IngestionStage, JobStatus
    from apps.sources.models import Source, SourceStatus
    from veda_core import config
    from veda_core.context import RequestContext, set_context

    set_context(RequestContext(source_id=int(source_id or 1), tenant=str(tenant)))

    encoder_mode = getattr(config, "ENCODER_MODE", "ensemble")
    job = IngestionJob.objects.create(
        source_id=source_id, tenant=tenant, status=JobStatus.RUNNING,
        encoder_mode=encoder_mode, started_at=timezone.now(),
    )
    # Create the ordered stage rows (pending) for observability (§7 table).
    stages = {
        name: IngestionStage.objects.create(job=job, order=order, name=name, status=JobStatus.PENDING)
        for order, name, _q in STAGE_ORDER
    }

    # ENCODER_MODE guard (§7): refuse if the requested mode differs from the persisted
    # one without an explicit force flag (re-ingestion required, per §12).
    prev = job.source.ingestion_jobs.exclude(pk=job.pk).filter(
        status=JobStatus.SUCCESS).order_by("-id").first()
    if prev and prev.encoder_mode and prev.encoder_mode != encoder_mode and not force:
        job.status = JobStatus.FAILED
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])
        raise RuntimeError(
            f"ENCODER_MODE changed {prev.encoder_mode!r}→{encoder_mode!r}; re-ingestion "
            "required — pass force=True (§12)."
        )

    def _mark(names, status):
        for n in names:
            s = stages.get(n)
            if s:
                s.status = status
                if status == JobStatus.RUNNING and not s.started_at:
                    s.started_at = timezone.now()
                if status in (JobStatus.SUCCESS, JobStatus.FAILED):
                    s.finished_at = timezone.now()
                s.save()

    try:
        # Run the heavy engine pipeline in a SUBPROCESS: the engine imports a top-level
        # `config` (config.py) that collides with this Django project's `config` package
        # in one interpreter. A subprocess gives it its own sys.modules — clean isolation.
        # cwd=veda_core so the engine's relative paths (data/, schema/, client_bge) resolve.
        #
        # We STREAM the subprocess stdout and map the engine's real `[N/NN] StageName` step
        # markers to live IngestionStage updates, so admin shows true per-stage progress as it
        # runs (not all-at-once). NOTE: the engine passes intermediate artifacts in-memory
        # between steps, so true mid-run resume-from-stage-N would require the §4.0
        # artifact-persistence extraction; here a failed job records exactly which stage
        # failed (the ones before it stay success), and a re-run restarts the idempotent
        # pipeline. The batched-stage-6 checkpoint is recorded per encoder table below.
        import json
        import re
        import subprocess

        veda_core_dir = os.path.join(os.environ.get("VEDA_APP_DIR", "/app"), "veda_core")
        # Resume (§4.2a/P8-B5): auto-detect from a prior failed job for this source, OR
        # explicit resume=True. VEDA_RESUME=1 makes the engine skip the expensive stages
        # (LLM semantic-layer, biencoder embeddings) when their persisted output exists,
        # while the fast prep stages re-run to rebuild the in-memory context.
        prior_failed = job.source.ingestion_jobs.exclude(pk=job.pk).filter(
            status=JobStatus.FAILED).order_by("-id").first()
        do_resume = bool(resume or prior_failed)
        sub_env = dict(os.environ)
        # Per-source connection (§5): inject THIS source's DB connection from its Source row,
        # so ingestion targets the right source without any global env/code change.
        src = job.source
        if src and src.host:
            sub_env.update(src.as_engine_env())
        if do_resume:
            sub_env["VEDA_RESUME"] = "1"
            job.stages.filter(name__in=("embeddings", "derived_language")).update(
                batch_checkpoint={"resume": True})

        # Source-type routing (§5): relational sources flow through the full, proven
        # run_ingestion pipeline (its per-source connection is injected via as_engine_env
        # above). Non-relational sources (nosql/document/datalake) — which run_ingestion
        # cannot handle — are routed by type through source_dispatcher.dispatch_ingestion,
        # receiving this source's config as JSON (single source of truth = the DB Source row).
        src_kind = src.source_kind() if src else "relational"
        if src_kind == "relational":
            py_code = f"import main; main.run_ingestion(verbose=False, skip_llm={bool(skip_llm)})"
        else:
            sub_env["VEDA_SOURCE_JSON"] = json.dumps(src.as_source_config())
            py_code = (
                "import os, json; "
                "from ingestion.source_dispatcher import dispatch_ingestion; "
                "cfg = json.loads(os.environ['VEDA_SOURCE_JSON']); "
                "r = dispatch_ingestion(cfg, verbose=False); "
                "print('[dispatch] type=%s success=%s' % (r.source_type, r.success)); "
                "raise SystemExit(0 if r.success else 1)"
            )
        proc = subprocess.Popen(
            ["python", "-u", "-c", py_code],
            cwd=veda_core_dir, env=sub_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        # Map the engine's step index (1..12, incl 7b/9b) to our STAGE_ORDER names, in order.
        engine_to_stage = [
            (1, "schema_scan"), (2, "fk_adjacency"), (3, "data_graph"), (4, "semantic_types"),
            (5, "semantic_types"), (6, "value_profiling"), (7, "unified_graph"),
            (8, "embeddings"), (9, "vector_store"), (10, "derived_language"),
            (11, "derived_language"), (12, "derived_language"),
        ]
        marker_re = re.compile(r"\[(\d+)[ab]?/\d+\]\s+([A-Za-z][^\(\n]+)")
        tail = []
        current_stage = None
        for line in proc.stdout:
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
            m = marker_re.search(line)
            if not m:
                continue
            idx = int(m.group(1))
            stage_name = dict(engine_to_stage).get(idx)
            if stage_name and stage_name != current_stage:
                if current_stage:
                    _mark([current_stage], JobStatus.SUCCESS)
                _mark([stage_name], JobStatus.RUNNING)
                # Record which engine step is in-flight for stage-6 batch visibility (§4.2a).
                st = stages.get(stage_name)
                if st is not None:
                    cp = dict(st.batch_checkpoint or {})
                    cp["engine_step"] = idx
                    cp["marker"] = m.group(2).strip()[:80]
                    st.batch_checkpoint = cp
                    st.save(update_fields=["batch_checkpoint"])
                current_stage = stage_name
        proc.wait()
        if proc.returncode != 0:
            if current_stage:
                _mark([current_stage], JobStatus.FAILED)
            raise RuntimeError(f"run_ingestion subprocess failed (rc={proc.returncode}): "
                               f"{''.join(tail)[-1500:]}")
        if current_stage:
            _mark([current_stage], JobStatus.SUCCESS)
        result = {"source_id": str(source_id) if source_id else "primary_db"}
        _mark([n for _o, n, _q in STAGE_ORDER if n != "warm_caches"], JobStatus.SUCCESS)

        # warm stage: sync Django substrate + publish sm + rehydrate fan-out.
        _mark(["warm_caches"], JobStatus.RUNNING)
        warm_counts = task_warm_caches(source_id=source_id, tenant=tenant)
        _mark(["warm_caches"], JobStatus.SUCCESS)

        job.status = JobStatus.SUCCESS
        if source_id:
            Source.objects.filter(pk=source_id).update(
                ready=True, status=SourceStatus.READY, last_ingested_at=timezone.now(),
            )
        return {"job_id": job.pk, "ok": True, "source_id": result.get("source_id"),
                "warm": warm_counts}
    except Exception:  # record failure, don't crash the worker
        job.status = JobStatus.FAILED
        _mark([n for _o, n, _q in STAGE_ORDER], JobStatus.FAILED)
        if source_id:
            Source.objects.filter(pk=source_id).update(status=SourceStatus.FAILED)
        raise
    finally:
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "finished_at"])
