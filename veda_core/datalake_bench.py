"""Datalake (source 4 invoices_csv + 5 catalog_parquet) benchmark — 54 grounded queries through the MAIN
pipeline with full routing (source_ids 2,3,4,5). Measures whether the coordinator routes each to a
DATALAKE table (vendors / maintenance / amenities_catalog) vs mis-routing to homzhub (source 2)."""
import json, time, sys, re
sys.path.insert(0, '/app'); sys.path.insert(0, '/app/veda_core')
from veda_core.context import RequestContext, set_context, set_source_profiles

QUESTIONS = [
    "How many vendors are there?", "What is the average vendor rating?", "What is the highest vendor rating?",
    "What is the lowest vendor rating?", "Which vendor has the highest rating?",
    "Which city has the highest-rated vendor?", "How many vendors are rated 4 or above?",
    "How many vendors are rated below 4?", "How many distinct cities have vendors?",
    "Which city has more than one vendor?", "What is the rating spread between the best and worst vendor?",
    "List all vendors with their ratings.", "Which vendors are located in Kochi?",
    "Which vendors are rated above 4.2?", "Show vendors with a rating of exactly 4.2.",
    "Which vendors are in Muddanahalli?", "How many maintenance records are there?",
    "What is the total maintenance amount?", "What is the average maintenance amount?",
    "How many maintenance records are repairs?", "How many maintenance records are paid?",
    "How many maintenance records are open?", "How many maintenance records are unpaid?",
    "What is the total repair amount?", "What is the total amount of paid maintenance records?",
    "What is the highest single maintenance amount?", "How many maintenance categories are there?",
    "Which category has the highest total maintenance amount?",
    "What is the distribution of maintenance records by category?",
    "What is the distribution of maintenance records by status?",
    "List all maintenance records with their amounts.", "Show maintenance records that are unpaid.",
    "Show repair maintenance records with an amount over 1000.",
    "Which maintenance records belong to asset 20?", "How many amenities are in the catalog?",
    "How many sports amenities are there?", "What is the total monthly fee of all catalog amenities?",
    "What is the average monthly fee of catalog amenities?", "Which amenity has the highest monthly fee?",
    "What is the highest monthly fee in the catalog?", "How many amenities have a zero monthly fee?",
    "How many amenity categories are there?", "Which category has the most amenities?",
    "List all catalog amenities with their monthly fees.", "Which amenities are in the Sports category?",
    "Which amenities cost more than 200 per month?", "What is the distribution of amenities by category?",
    "Which amenities are free of charge?", "Show amenities in the Security category.",
    "What is the combined monthly cost of all Sports amenities?",
    "If all vendors rated below 4.0 are removed, what is the new average rating?",
    "What percentage of vendors are rated at least 4.0?",
    "What proportion of maintenance records are paid?",
    "Which is greater, the total maintenance amount or the total catalog fees?",
]

_DL_RE = re.compile(r'\b(vendors|maintenance|amenities_catalog)\b', re.I)
_HZ_RE = re.compile(r'\b(assets_|accounts_|users_|leads_|generics_|worklists_)', re.I)


def _target(sql, ans, cites):
    blob = (sql or '') + ' ' + (ans or '') + ' ' + ' '.join(cites or [])
    if _DL_RE.search(sql or ''):
        return 'datalake'
    if _HZ_RE.search(sql or ''):
        return 'homzhub'
    if cites:
        return 'doc'
    return '?'


def run_one(q):
    set_context(RequestContext(source_id=2, tenant='default', source_ids=(2, 3, 4, 5)))
    set_source_profiles({'2': {'source_type': 'relational'}, '3': {'source_type': 'filesystem'},
                         '4': {'source_type': 'csv_lake'}, '5': {'source_type': 'parquet'}})
    from veda_hybrid import run_hybrid_query
    t = time.time()
    try:
        mr = run_hybrid_query(q); items = mr.items or []
        it = items[0] if items else None
        r = it.result if it else None
        sql = (r.get('sql') if isinstance(r, dict) else '') or ''
        ans = (r.get('answer') if isinstance(r, dict) else getattr(r, 'answer', '')) or ''
        cites = getattr(r, 'citations', None) or []
        return {"status": getattr(it, 'status', '?'), "target": _target(sql, str(ans), cites),
                "sql": sql[:120], "answer": str(ans)[:160], "secs": round(time.time() - t, 1)}
    except Exception as e:
        return {"status": "EXC", "target": "?", "error": str(e)[:150], "secs": round(time.time() - t, 1)}


out = []
for i, q in enumerate(QUESTIONS, 1):
    res = run_one(q); res.update({"n": i, "q": q})
    ok = (res.get("target") == "datalake")
    print(f"[{i:02d}/{len(QUESTIONS)}] {res['status']:7s} -> {res.get('target'):8s} "
          f"{'OK ' if ok else 'MIS'} {q[:42]}", flush=True)
    json.dump(out + [res], open('/app/datalake_bench_results.json', 'w'), indent=2)
    out.append(res)
print("DONE", len(out))
