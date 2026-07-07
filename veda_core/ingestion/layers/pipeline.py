"""Layered ingestion composer — a faithful move of main.run_ingestion's body.

``run_layered_ingestion(ctx)`` runs L1→L5 in one process, threading the same
in-memory ``state`` dict the monolith used, and honours the per-stage fatal
semantics (a fatal stage aborts; a non-fatal stage logs and continues). Returns
the ``state`` dict so callers get the same context contract as run_ingestion.

Stage outcomes are surfaced via an optional ``on_stage`` callback so the Celery
orchestrator (apps.ingestion.tasks) can update IngestionStage rows from real
lifecycle events instead of parsing stdout markers (I-6 fix).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ingestion.contracts import SourceContext, StageOutcome
from ingestion.layers import l1_extract, l2_analyze, l3_enrich, l4_index, l5_publish

_LAYERS = [
    ("l1_extract", l1_extract.run),
    ("l2_analyze", l2_analyze.run),
    ("l3_enrich", l3_enrich.run),
    ("l4_index", l4_index.run),
    ("l5_publish", l5_publish.run),
]


def run_layered_ingestion(
    ctx: SourceContext,
    verbose: bool = False,
    on_stage: Optional[Callable[[StageOutcome], None]] = None,
) -> Dict:
    """Compose the five layers. Aborts on the first fatal stage failure."""
    state: Dict = {"source_id": ctx.source_id, "outcomes": []}
    t_total = time.time()

    for layer_name, layer_fn in _LAYERS:
        outcomes: List[StageOutcome] = layer_fn(ctx, state, verbose=verbose)
        for oc in outcomes:
            state["outcomes"].append(oc)
            if on_stage:
                on_stage(oc)
            # Stable, machine-readable stage event for the Celery orchestrator to map
            # to IngestionStage rows via real lifecycle (no stdout-regex — I-6 fix).
            _status = "ok" if oc.ok else ("fatal" if oc.fatal else "fail")
            print(f"[[STAGE]] {layer_name} {oc.name} {_status}", flush=True)
            if verbose:
                mark = "✓" if oc.ok else ("✗ FATAL" if oc.fatal else "✗")
                print(f"  [{layer_name}] {mark} {oc.name} — {oc.detail or oc.error or ''}")
            if oc.fatal and not oc.ok:
                state["failed_stage"] = oc.name
                raise RuntimeError(
                    f"Ingestion aborted at fatal stage {oc.name!r}: {oc.error}")

    state["duration_s"] = round(time.time() - t_total, 2)
    return state
