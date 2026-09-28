"""Table-level retrieval check (the spine: retrieval_select.select_retrieval, rerank on)
before/after an embedding-sentence change. Per query: the ranked tables, the rank of the
first gold table; summary: table recall@3/@5 and MRR. Inference container:
    cd /app/veda_core && python /app/scripts/retrieval_tables_eval.py --source-id 2 \
        --golden /app/evaluation/retrieval_tables_homzhub.jsonl --out /app/reports/raw/agent2/ret_before.json
"""
import argparse, json, sys, time
sys.path.insert(0, "/app"); sys.path.insert(0, "/app/veda_core")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-id", default="2")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from veda_core.context import RequestContext, set_context
    set_context(RequestContext(source_id=int(a.source_id), tenant="default",
                               source_ids=(int(a.source_id),), cache_back=False))
    from query.retrieval_select import select_retrieval
    rows = [json.loads(l) for l in open(a.golden) if l.strip()]
    out = []
    for r in rows:
        t0 = time.time()
        sel = select_retrieval(query=r["query"], source_ids=[a.source_id], intent="sql", verbose=False)
        tabs = list(dict.fromkeys(getattr(sel, "tables", None) or []))
        gold = set(r["gold_tables"])
        rank = next((i + 1 for i, t in enumerate(tabs) if t in gold), None)
        out.append({"query": r["query"], "gold": r["gold_tables"], "tables": tabs[:10], "rank": rank,
                    "ms": round((time.time() - t0) * 1000)})
        print(f"{rank!s:>4}  {r['query'][:60]:<60} {tabs[:4]}", flush=True)
    n = len(out)
    summ = {"n": n, "recall@3": sum(1 for o in out if o["rank"] and o["rank"] <= 3) / n,
            "recall@5": sum(1 for o in out if o["rank"] and o["rank"] <= 5) / n,
            "mrr": sum(1.0 / o["rank"] for o in out if o["rank"]) / n}
    print(json.dumps(summ))
    json.dump({"summary": summ, "per_query": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
