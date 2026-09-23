"""GROUP 0.3 verification — assert at the WIRE that SLM_TEMPERATURE reaches the SLM.

Not a config read: this intercepts the actual POST to OLLAMA_URL and inspects
options.temperature on the two named call sites from the benchmark finding —
query/rag_layer.py::_call_ollama (purpose="rag_synthesis") and
query/slm_layer.py::_call_ollama (purpose="ir_emit").
"""
import json, os, sys, urllib.request
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/veda_core")
os.chdir("/app/veda_core")

CAPTURED = []
_HOST = (os.environ.get("OLLAMA_URL") or "").rstrip("/")
_orig = urllib.request.urlopen


def _urlopen(req, *a, **k):
    url = getattr(req, "full_url", None) or (req if isinstance(req, str) else "")
    if _HOST and str(url).startswith(_HOST) and getattr(req, "get_method", lambda: "GET")() == "POST":
        try:
            CAPTURED.append(json.loads(req.data.decode("utf-8")))
        except Exception as e:
            CAPTURED.append({"_parse_error": str(e)})
    return _orig(req, *a, **k)


urllib.request.urlopen = _urlopen

import config
print(f"config.SLM_TEMPERATURE = {config.SLM_TEMPERATURE!r}  (env SLM_TEMPERATURE={os.environ.get('SLM_TEMPERATURE')!r})")

results = {}

from query import rag_layer
CAPTURED.clear()
try:
    rag_layer._call_ollama("You are terse. Answer in one word.", "Say OK.")
except Exception as e:
    print(f"  rag_synthesis call raised (fine — we only need the payload): {type(e).__name__}: {e}")
results["rag_synthesis"] = list(CAPTURED)

from query import slm_layer
CAPTURED.clear()
try:
    slm_layer._call_ollama("Say OK.")
except Exception as e:
    print(f"  ir_emit call raised (fine — we only need the payload): {type(e).__name__}: {e}")
results["ir_emit"] = list(CAPTURED)

ok = True
for purpose, payloads in results.items():
    if not payloads:
        print(f"FAIL {purpose}: no POST captured"); ok = False; continue
    for p in payloads:
        temp = (p.get("options") or {}).get("temperature", "<ABSENT>")
        model = p.get("model")
        verdict = "PASS" if temp == 0 else "FAIL"
        if temp != 0:
            ok = False
        print(f"{verdict} {purpose:16s} wire options.temperature = {temp!r}  model={model}")

print("WIRE ASSERTION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
