"""VEDA branch-vs-master engine benchmark harness.

Runs the SAME query set through veda_hybrid.run_hybrid_query on whichever tree is
mounted at /app, capturing per-query wall clock, per-stage timing (from the engine's
own ExplainTrace), per-SLM-call latency + tokens, embedding/rerank round-trips and
DB connect/query counts.

Everything it patches exists byte-identically on both trees (verified before use):
  veda.pipeline.verified_cache_lookup / save_verified_query  -> neutralised
      (master has no cache_back kill switch; without this, reps 2-3 on master would
       replay a cached SQL and master would look artificially fast)
  veda.explain.ExplainTrace.finalize -> captures to_dict() (non-verbose, no extra work)
  ingestion.m3_encoder._metal_post   -> counts embedding round-trips
  query.reranker._RemoteReranker.predict -> counts rerank round-trips
  psycopg2.connect                   -> counts connects + wraps a counting cursor

Usage (inside a container with the tree at /app):
    python /bench/bench_engine.py --label branch --out /bench/out/branch.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

COUNTERS = defaultdict(int)
EVENTS: list = []
EMBED_CALLS: list = []
SLM_HTTP: list = []
RERANK_CALLS: list = []
CAPTURED: dict = {}


# ── instrumentation ─────────────────────────────────────────────────────────
def install_probes():
    installed = {}

    # 1. DB connects + queries (psycopg2 is the only driver in requirements/*.txt)
    try:
        import psycopg2
        import psycopg2.extensions as _pgext

        class _CountingCursor(_pgext.cursor):
            def execute(self, query, vars=None):
                COUNTERS["db_queries"] += 1
                return super().execute(query, vars)

            def executemany(self, query, vars_list):
                COUNTERS["db_queries"] += 1
                return super().executemany(query, vars_list)

        _orig_connect = psycopg2.connect

        def _connect(*a, **k):
            COUNTERS["db_connects"] += 1
            dsn = " ".join(str(x) for x in a) + json.dumps(k, default=str)
            if "veda_engine" in dsn:
                COUNTERS["db_connects_engine"] += 1
            elif "homzhub" in dsn:
                COUNTERS["db_connects_source"] += 1
            k.setdefault("cursor_factory", _CountingCursor)
            return _orig_connect(*a, **k)

        psycopg2.connect = _connect
        installed["psycopg2"] = True
    except Exception as e:
        installed["psycopg2"] = f"FAILED: {e}"

    # 2. embedding round-trips (host Metal server)
    try:
        from ingestion import m3_encoder as _m3
        _orig_mp = _m3._metal_post

        def _mp(path, payload):
            t0 = time.perf_counter()
            ok = True
            try:
                return _orig_mp(path, payload)
            except Exception:
                ok = False
                raise
            finally:
                ms = (time.perf_counter() - t0) * 1000.0
                COUNTERS["embed_roundtrips"] += 1
                n = len(payload.get("texts") or []) if isinstance(payload, dict) else 0
                import hashlib as _h
                try:
                    _blob = json.dumps(payload, sort_keys=True, default=str).encode()
                    _sig = _h.sha1(_blob).hexdigest()[:12]
                    _first = str((payload.get("texts") or [payload.get("text") or ""])[0])[:60]
                except Exception:
                    _sig, _first = "?", "?"
                EMBED_CALLS.append({"path": path, "n_texts": n, "payload_sha": _sig,
                                    "first_text": _first, "ms": round(ms, 1), "ok": ok})

        _m3._metal_post = _mp
        installed["m3_encoder._metal_post"] = True
    except Exception as e:
        installed["m3_encoder._metal_post"] = f"FAILED: {e}"

    # 3. rerank round-trips
    try:
        from query import reranker as _rr
        _orig_pred = _rr._RemoteReranker.predict

        def _pred(self, pairs, batch_size=64, **kw):
            t0 = time.perf_counter()
            ok = True
            try:
                return _orig_pred(self, pairs, batch_size=batch_size, **kw)
            except Exception:
                ok = False
                raise
            finally:
                ms = (time.perf_counter() - t0) * 1000.0
                COUNTERS["rerank_roundtrips"] += 1
                RERANK_CALLS.append({"n_pairs": len(list(pairs)) if pairs else 0,
                                     "ms": round(ms, 1), "ok": ok})

        _rr._RemoteReranker.predict = _pred
        installed["reranker.predict"] = True
    except Exception as e:
        installed["reranker.predict"] = f"FAILED: {e}"

    # 3b. TRANSPORT-LEVEL SLM probe. call_slm() is not the only way this codebase
    # reaches the SLM: query/answer_entity.py::_llm_relation_word does a hand-rolled
    # urllib POST to {SLM_OLLAMA_BASE_URL}/api/chat on master (branch routes it through
    # call_slm). Counting only call_slm would report master as making FEWER SLM calls
    # than it really does. This intercepts every POST to the SLM host on BOTH trees,
    # reads the body once, records tokens from Ollama's own prompt_eval_count/eval_count,
    # and hands the caller an equivalent response object. Non-SLM URLs pass straight
    # through untouched.
    try:
        import urllib.request as _ur
        import io as _io
        _slm_host = (os.environ.get("OLLAMA_URL") or "").rstrip("/")
        _orig_urlopen = _ur.urlopen

        class _Replay:
            def __init__(self, data, resp):
                self._buf = _io.BytesIO(data)
                self._resp = resp
            def read(self, *a):
                return self._buf.read(*a)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def __getattr__(self, n):
                return getattr(self._resp, n)

        def _urlopen(req, *a, **k):
            url = getattr(req, "full_url", None) or (req if isinstance(req, str) else "")
            method = getattr(req, "get_method", lambda: "GET")()
            if not (_slm_host and str(url).startswith(_slm_host) and method == "POST"):
                return _orig_urlopen(req, *a, **k)
            t0 = time.perf_counter()
            resp = _orig_urlopen(req, *a, **k)
            data = resp.read()
            ms = (time.perf_counter() - t0) * 1000.0
            pt = ct = None
            try:
                body = json.loads(data.decode("utf-8"))
                u = body.get("usage")
                if isinstance(u, dict):
                    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
                else:
                    pt, ct = body.get("prompt_eval_count"), body.get("eval_count")
            except Exception:
                pass
            COUNTERS["slm_http_posts"] += 1
            SLM_HTTP.append({"url": str(url), "ms": round(ms, 1),
                             "prompt_tokens": pt or 0, "completion_tokens": ct or 0})
            return _Replay(data, resp)

        _ur.urlopen = _urlopen
        installed["slm_http_probe"] = {"host": _slm_host}
    except Exception as e:
        installed["slm_http_probe"] = f"FAILED: {e}"

    # 4. verified-query cache OFF (parity: master has no cache_back switch)
    try:
        from veda import pipeline as _pl
        _pl.verified_cache_lookup = lambda q, threshold=0.85: (None, 0.0)
        _pl.save_verified_query = lambda *a, **k: None
        installed["verified_cache_disabled"] = True
    except Exception as e:
        installed["verified_cache_disabled"] = f"FAILED: {e}"

    # 5. capture the finalized trace (to_dict is non-verbose; no extra recording cost)
    try:
        from veda import explain as _ex
        _orig_fin = _ex.ExplainTrace.finalize

        def _fin(self, status, route=""):
            d = _orig_fin(self, status, route)
            try:
                CAPTURED["trace"] = self.to_dict()
            except Exception:
                CAPTURED["trace"] = None
            return d

        _ex.ExplainTrace.finalize = _fin
        installed["trace_capture"] = True
    except Exception as e:
        installed["trace_capture"] = f"FAILED: {e}"

    return installed


# ── query set (identical on both branches) ──────────────────────────────────
FULL_SCOPE = (2, 3, 4, 5)
PROFILES = {"2": {"source_type": "relational"}, "3": {"source_type": "filesystem"},
            "4": {"source_type": "csv_lake"}, "5": {"source_type": "parquet"}}

QUERIES = [
    # id, category, question, primary source, scope, expected shape
    ("DB1", "database_sql", "latest 10 payment transactions", 2, (2,),
     "answered; ORDER BY a date column, LIMIT 10, on payments_paymenttransaction"),
    ("DB2", "database_sql", "users created last month", 2, (2,),
     "answered; WHERE on a created-at column over last month, on users_user"),
    ("DL1", "data_lake", "total maintenance amount per category", 4, (4,),
     "answered; SUM(amount) GROUP BY category on maintenance (csv_lake)"),
    ("DL2", "data_lake", "average monthly fee per category", 5, (5,),
     "answered; AVG(monthly_fee) GROUP BY category on amenities_catalog (parquet)"),
    # Document questions run with the full ready scope — the context the product
    # actually serves them in (veda_core/doc_bench.py uses the same). Pinning to the
    # single document source trips veda_hybrid.py's permission pre-check; that case is
    # measured separately as probes FS1P/FS2P below, on BOTH trees.
    ("FS1", "filesystem_rag", "what is the late fee percentage", 3, FULL_SCOPE,
     "RAG answer naming 2 percent / 2%, cited to msa_green_tower"),
    ("FS2", "filesystem_rag", "summarize the employee handbook leave policy", 3, FULL_SCOPE,
     "RAG answer naming leave types (earned/sick/maternity), cited to the Employee Handbook"),
    ("XS1", "cross_source", "which vendors operate in cities where we have assets", 2, FULL_SCOPE,
     "federated: vendors@4 joined to assets_asset.city_name@2, or a typed refusal"),
    ("XS2", "cross_source", "total maintenance amount spent per city where we own assets", 2, FULL_SCOPE,
     "federated: SUM(maintenance.amount)@4 grouped by asset city@2, or a typed refusal"),
]

# Correctness probes — not part of the timed latency set. Same two document questions
# PINNED to the document source only, which is how a caller scoped to one source asks.
PROBES = [
    ("FS1P", "probe_pinned_fs", "what is the late fee percentage", 3, (3,),
     "same answer as FS1 (2 percent, cited msa_green_tower) — scope is narrower, not different"),
    ("FS2P", "probe_pinned_fs", "summarize the employee handbook leave policy", 3, (3,),
     "same answer as FS2 — scope is narrower, not different"),
]


def _f(r, name, default=None):
    if isinstance(r, dict):
        return r.get(name, default)
    return getattr(r, name, default)


def run_one(qid, question, source_id, scope, reset_counters=True):
    from veda_core.context import RequestContext, set_context, set_source_profiles
    from veda_hybrid import run_hybrid_query
    from slm._call_slm import collect_usage

    if reset_counters:
        COUNTERS.clear()
    EVENTS.clear()
    EMBED_CALLS.clear()
    RERANK_CALLS.clear()
    SLM_HTTP.clear()
    CAPTURED.clear()

    # Build the context with ONLY the fields this tree's RequestContext declares,
    # so the same harness runs on master (no cache_back / session_prior there).
    import dataclasses
    fields = {f.name for f in dataclasses.fields(RequestContext)}
    kw = {"source_id": source_id, "tenant": "default", "source_ids": tuple(scope)}
    if "cache_back" in fields:
        kw["cache_back"] = False
    set_context(RequestContext(**kw))
    set_source_profiles(PROFILES)

    t0 = time.perf_counter()

    def on_event(phase, message="", extra=None):
        # signature is (phase, message, extra) on BOTH trees
        # (veda/pipeline.py::_tick, veda/lifecycle.py::Timeline._emit)
        EVENTS.append({"t_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "phase": str(phase), "message": str(message)[:200],
                       "extra": json.loads(json.dumps(extra or {}, default=str))})

    row = {"qid": qid, "q": question, "source_id": source_id, "scope": list(scope)}
    usage_calls = []
    try:
        with collect_usage() as u:
            mr = run_hybrid_query(question, verbose=False, on_event=on_event)
            usage_calls = u.calls()
        wall = (time.perf_counter() - t0) * 1000.0
        items = getattr(mr, "items", None) or []
        it = items[0] if items else None
        res = it.result if it is not None else None
        row["wall_ms"] = round(wall, 1)
        row["item_status"] = str(getattr(it, "status", "")) if it is not None else "no-items"
        row["route"] = str(getattr(it, "route", "") or "")
        row["n_items"] = len(items)
        if isinstance(res, dict):
            row["kind"] = "sql"
            row["status"] = res.get("status")
            row["sql"] = (res.get("sql") or "")[:600]
            rows = res.get("rows")
            row["row_count"] = len(rows) if isinstance(rows, list) else res.get("row_count")
            row["answer"] = str(res.get("answer") or res.get("nl_summary") or "")[:400]
            row["reason"] = str(res.get("reason") or res.get("message") or "")[:300]
        else:
            row["kind"] = "rag" if _f(res, "citations") is not None else "other"
            row["status"] = _f(res, "status") or ("answered" if _f(res, "answer") else "no-answer")
            row["answer"] = str(_f(res, "answer") or "")[:600]
            row["citations"] = [str(c)[:120] for c in (_f(res, "citations") or [])][:6]
            row["error"] = str(_f(res, "error") or "")[:300]
    except Exception as e:
        row["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        row["status"] = "CRASH"
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-1500:]

    # ── trace-derived measurements ──
    tr = CAPTURED.get("trace") or {}
    secs = tr.get("sections") or {}
    totals = secs.get("totals") or {}
    row["trace_id"] = tr.get("trace_id")
    row["trace_total_ms"] = tr.get("total_ms")
    row["stage_durations_ms"] = totals.get("stage_durations_ms") or {}
    row["stage_starts_ms"] = {k: v.get("_ms") for k, v in secs.items()
                              if isinstance(v, dict) and "_ms" in v}
    row["slm_calls"] = (secs.get("slm") or {}).get("calls") or []
    row["slm_call_count"] = len(row["slm_calls"])
    row["llm_usage_section"] = secs.get("llm_usage") or {}
    row["retrieval_health"] = secs.get("retrieval_health") or {}
    row["firewall"] = secs.get("firewall") or {}
    row["routing"] = {k: v for k, v in (secs.get("routing") or {}).items()
                      if k in ("decision", "mode", "chosen", "source_ids", "reason",
                               "scope", "why", "_ms", "confidence", "candidates")}
    row["federated"] = secs.get("federated") or {}
    row["execution_plan"] = secs.get("execution_plan") or {}

    # ── harness-side measurements (identical on both trees) ──
    row["usage_calls"] = usage_calls
    row["tokens_prompt"] = sum(c.get("prompt_tokens", 0) or 0 for c in usage_calls)
    row["tokens_completion"] = sum(c.get("completion_tokens", 0) or 0 for c in usage_calls)
    row["tokens_total"] = row["tokens_prompt"] + row["tokens_completion"]
    row["slm_http_posts"] = COUNTERS.get("slm_http_posts", 0)
    row["slm_http_calls"] = list(SLM_HTTP)
    row["slm_http_tokens_prompt"] = sum(c["prompt_tokens"] for c in SLM_HTTP)
    row["slm_http_tokens_completion"] = sum(c["completion_tokens"] for c in SLM_HTTP)
    row["slm_http_ms_total"] = round(sum(c["ms"] for c in SLM_HTTP), 1)
    row["embed_roundtrips"] = COUNTERS.get("embed_roundtrips", 0)
    row["embed_calls"] = list(EMBED_CALLS)
    row["embed_ms_total"] = round(sum(c["ms"] for c in EMBED_CALLS), 1)
    row["rerank_roundtrips"] = COUNTERS.get("rerank_roundtrips", 0)
    row["rerank_calls"] = list(RERANK_CALLS)
    row["rerank_ms_total"] = round(sum(c["ms"] for c in RERANK_CALLS), 1)
    row["db_connects"] = COUNTERS.get("db_connects", 0)
    row["db_connects_engine"] = COUNTERS.get("db_connects_engine", 0)
    row["db_connects_source"] = COUNTERS.get("db_connects_source", 0)
    row["db_queries"] = COUNTERS.get("db_queries", 0)
    row["events"] = list(EVENTS)
    row["ttfb_ms"] = EVENTS[0]["t_ms"] if EVENTS else None
    row["event_count"] = len(EVENTS)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    os.chdir("/app/veda_core")
    probes = install_probes()
    meta = {"kind": "meta", "label": args.label, "probes": probes,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "env": {k: os.environ.get(k) for k in (
                "SLM_MODEL_NAME", "SLM_TEMPERATURE", "OLLAMA_URL", "METAL_EMBED_URL",
                "NL_SUMMARY_MODEL", "CHATBOT_CLASSIFY_MODEL", "VEDA_TOP_K",
                "VEDA_TOP_K_TO_LLM", "VEDA_ENCODER_MODE", "VEDA_HNSW_EF_SEARCH",
                "MULTISOURCE_ROUTING_ENABLED", "MULTISOURCE_ROUTING_SHADOW",
                "VEDA_FAST_PATH_ENABLED", "VEDA_QUERY_ROUTER_ENABLED",
                "SLM_NUM_CTX", "VEDA_SM_REDIS")}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, "w", buffering=1)
    fh.write(json.dumps(meta, default=str) + "\n")
    print(json.dumps({"probes": probes}), flush=True)

    sel = {s.strip() for s in args.only.split(",") if s.strip()}
    plan = [q for q in (QUERIES + PROBES) if (q[0] in sel if sel else q in QUERIES)]

    # ── warm-up: throwaway queries, never timed ──
    for i in range(args.warmup):
        wq = ("how many vendors are there", 4, (4,)) if i == 0 else \
             ("what does the site notes document say", 3, (3,))
        t = time.time()
        try:
            r = run_one(f"WARMUP{i+1}", wq[0], wq[1], wq[2])
            print(f"[warmup {i+1}] {wq[0]!r} -> {str(r.get('status'))} in {r.get('wall_ms')}ms",
                  flush=True)
            r["kind_row"] = "warmup"
            fh.write(json.dumps(r, default=str) + "\n")
        except Exception as e:
            print(f"[warmup {i+1}] FAILED {e}", flush=True)
        print(f"  (warmup elapsed {time.time()-t:.1f}s)", flush=True)

    # ── timed runs ──
    for rep in range(1, args.reps + 1):
        for (qid, cat, question, sid, scope, expect) in plan:
            r = run_one(qid, question, sid, scope)
            r.update({"kind_row": "timed", "rep": rep, "category": cat,
                      "expected_shape": expect, "label": args.label})
            fh.write(json.dumps(r, default=str) + "\n")
            print(f"[rep{rep}] {qid:4s} {str(r.get('status')):16s} "
                  f"wall={r.get('wall_ms')}ms slm={r.get('slm_call_count')} "
                  f"tok={r.get('tokens_total')} http={r.get('slm_http_posts')}/{r.get('slm_http_tokens_prompt')}+{r.get('slm_http_tokens_completion')} emb={r.get('embed_roundtrips')} "
                  f"rr={r.get('rerank_roundtrips')} dbq={r.get('db_queries')}", flush=True)
    fh.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
