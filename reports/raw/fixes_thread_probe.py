import sys, os, json, threading, contextvars, time
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/veda_core")
os.chdir("/app/veda_core")

PROBE = contextvars.ContextVar("probe_marker", default=None)
CALLS = []

from ingestion import m3_encoder as m3
_orig = m3._metal_post
def _mp(path, payload):
    t0=time.perf_counter()
    try:
        return _orig(path, payload)
    finally:
        import hashlib
        sha = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        CALLS.append({"path": path, "sha": sha,
                      "thread": threading.current_thread().name,
                      "tid": threading.get_ident(),
                      "ctxvar_visible": PROBE.get(),
                      "ms": round((time.perf_counter()-t0)*1000,1)})
m3._metal_post = _mp

from veda_core.context import RequestContext, set_context, set_source_profiles
set_context(RequestContext(source_id=2, tenant="default", source_ids=(2,)))
set_source_profiles({'2': {'source_type': 'relational'}})
PROBE.set("REQUEST-1")
from veda_hybrid import run_hybrid_query
try:
    run_hybrid_query("users created last month", verbose=False)
except Exception as e:
    print("raised:", type(e).__name__, e)

print("\n=== _metal_post calls ===")
for c in CALLS:
    print(f"  {c['path']:16s} sha={c['sha']} thread={c['thread']:<22s} tid={c['tid']} "
          f"ctxvar={c['ctxvar_visible']!r} {c['ms']:>7.0f}ms")
import collections
by_sha = collections.Counter((c["path"], c["sha"]) for c in CALLS)
print("\nduplicated payloads:", {f"{k[0]}:{k[1]}": v for k, v in by_sha.items() if v > 1})
print("distinct threads:", sorted({c["thread"] for c in CALLS}))
print("ctxvar visible in ALL calls:", all(c["ctxvar_visible"] == "REQUEST-1" for c in CALLS))
