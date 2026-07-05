"""inference health routes — /healthz liveness, /readyz readiness (migration_plan.md §8.2).

/readyz returns 503 until the engine's warm-load state is green (semantic model
present); liveness is always 200 once the process is up.
"""
from __future__ import annotations

try:
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
except ImportError:  # keep importable without FastAPI
    APIRouter = None
    JSONResponse = None

if APIRouter is not None:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz():
        from inference.loaders import readiness

        state = readiness()
        code = 200 if state.get("ready") else 503
        return JSONResponse(status_code=code, content={"status": "ready" if state.get("ready") else "warming", **state})
else:  # pragma: no cover
    router = None
