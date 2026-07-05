"""VEDA core library — the preserved engine, moved verbatim from the POC repo.

The packages under this directory (``veda``, ``query``, ``retrieval``,
``connectors``, ``graph``, ``semantic``, ``ingestion``, ``schema``, ``utils``)
plus ``config`` and ``veda_hybrid`` were relocated *without edits* (migration
plan §0.2 / PRESERVE). Their internal imports are top-level absolute
(``from config import ...``, ``from query.slm_layer import ...``,
``from utils.logger import ...``) exactly as they were in the original repo.

To keep those imports resolving *without touching a single line of the moved
code*, this package inserts its own directory onto ``sys.path`` at import time.
That makes the children importable both ways:

    from veda_core.veda_hybrid import run_hybrid_query   # package-qualified
    import config, query, utils                          # legacy top-level

This shim is the only new code inside ``veda_core`` that is not part of the
verbatim move. The later migration edits permitted here (plan §0.2) are confined
to storage call sites (Phase 3 → ``storage_adapters``) and SLM call sites
(Phase 3b → ``slm/_call_slm``); everything else stays byte-for-byte.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PKG_DIR not in _sys.path:
    # Prepend so the preserved top-level absolute imports resolve to the
    # relocated packages rather than any same-named module elsewhere.
    _sys.path.insert(0, _PKG_DIR)
