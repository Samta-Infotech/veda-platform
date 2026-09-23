import json, statistics as st, os, sys
OUT="/private/tmp/claude-501/-Users-ekesel-samta-veda-platform/3d25578a-bdc8-4e8a-aadc-11fb0aaf74da/scratchpad/bench/out"

def load(p, kind="timed"):
    rows=[]
    for l in open(p):
        d=json.loads(l)
        if d.get("kind_row")==kind: rows.append(d)
    return rows

B=load(f"{OUT}/branch_v2.jsonl"); M=load(f"{OUT}/master_v2.jsonl")
QORDER=["DB1","DB2","DL1","DL2","FS1","FS2","XS1","XS2"]
CAT={"DB1":"Database / SQL","DB2":"Database / SQL","DL1":"Data Lake","DL2":"Data Lake",
     "FS1":"File System / RAG","FS2":"File System / RAG","XS1":"Cross-source","XS2":"Cross-source"}

def p90(v):
    v=sorted(v)
    if len(v)==1: return v[0]
    # linear interpolation, numpy 'linear' method
    k=(len(v)-1)*0.90
    f=int(k); c=min(f+1,len(v)-1)
    return v[f]+(v[c]-v[f])*(k-f)

def agg(rows,qid,field):
    v=[r[field] for r in rows if r["qid"]==qid and r.get(field) is not None]
    return v

def fmt(x, nd=0):
    if x is None: return "n/a"
    return f"{x:,.{nd}f}"

def delta(b,m):
    if b is None or m is None: return "n/a","n/a"
    d=b-m
    pct=(d/m*100) if m else float('inf')
    return f"{d:+,.0f}", (f"{pct:+.1f}%" if m else "n/a")

print("### TABLE A — latency (ms), master vs branch\n")
print("| Q | Category | master med | master p90 | branch med | branch p90 | Δ med | Δ med % | Δ p90 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
tot={}
for q in QORDER:
    mv=agg(M,q,"wall_ms"); bv=agg(B,q,"wall_ms")
    mm,mp=st.median(mv),p90(mv); bm,bp=st.median(bv),p90(bv)
    d,pc=delta(bm,mm); d9,_=delta(bp,mp)
    tot[q]=(mm,bm,mp,bp)
    print(f"| {q} | {CAT[q]} | {fmt(mm)} | {fmt(mp)} | {fmt(bm)} | {fmt(bp)} | {d} | {pc} | {d9} |")
allm=[tot[q][0] for q in QORDER]; allb=[tot[q][1] for q in QORDER]
print(f"| **sum of medians** | | **{fmt(sum(allm))}** | | **{fmt(sum(allb))}** | | {delta(sum(allb),sum(allm))[0]} | {delta(sum(allb),sum(allm))[1]} | |")

print("\n\n### TABLE B — SLM calls and tokens (median rep), master vs branch")
print("Counts are TRANSPORT-measured (every POST to the SLM host), not ledger-derived:")
print("master's query/answer_entity.py bypasses call_slm(), so its own trace under-reports.")
print("`ledger` shows what each tree's own trace claims, for comparison.\n")
print("| Q | master calls | master pt | master ct | master tot | master ledger | branch calls | branch pt | branch ct | branch tot | branch ledger | Δ calls | Δ tokens |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
tokm=tokb=0; callm=callb=0
for q in QORDER:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]
        rs.sort(key=lambda r:r["wall_ms"])
        return rs[len(rs)//2]
    m=pick(M); b=pick(B)
    mt=m["slm_http_tokens_prompt"]+m["slm_http_tokens_completion"]
    bt=b["slm_http_tokens_prompt"]+b["slm_http_tokens_completion"]
    tokm+=mt; tokb+=bt
    callm+=m["slm_http_posts"]; callb+=b["slm_http_posts"]
    print(f"| {q} | {m['slm_http_posts']} | {fmt(m['slm_http_tokens_prompt'])} | {fmt(m['slm_http_tokens_completion'])} | {fmt(mt)} | {m['slm_call_count']}c/{fmt(m['tokens_total'])}t "
          f"| {b['slm_http_posts']} | {fmt(b['slm_http_tokens_prompt'])} | {fmt(b['slm_http_tokens_completion'])} | {fmt(bt)} | {b['slm_call_count']}c/{fmt(b['tokens_total'])}t "
          f"| {b['slm_http_posts']-m['slm_http_posts']:+d} | {bt-mt:+,d} |")
print(f"| **total** | **{callm}** | | | **{tokm:,}** | **{callb}** | | | **{tokb:,}** | **{callb-callm:+d}** | **{tokb-tokm:+,d}** |")

print("\n\n### TABLE C — per-stage wall time (ms, median rep)\n")
STAGES=["query_understanding","routing","retrieval","retrieval_health","rrf","graph_expansion",
        "reranking","schema_linking","entity_resolution","join_planning","sql_planning",
        "firewall","validation","execution","execution_plan","source_execution","result_analysis",
        "summary","nl_summary","visualization","output","slm","lifecycle","federation","federated"]
for q in QORDER:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); return rs[len(rs)//2]
    m=pick(M); b=pick(B)
    ms=m.get("stage_durations_ms") or {}; bs=b.get("stage_durations_ms") or {}
    keys=[k for k in STAGES if (ms.get(k,0) or bs.get(k,0))]
    extra=sorted(set(list(ms)+list(bs))-set(STAGES))
    keys+= [k for k in extra if (ms.get(k,0) or bs.get(k,0))]
    if not keys: continue
    print(f"\n**{q}** ({CAT[q]}) — total master {fmt(m['wall_ms'])} / branch {fmt(b['wall_ms'])} ms\n")
    print("| stage | master | branch | Δ |")
    print("|---|---:|---:|---:|")
    for k in keys:
        a=ms.get(k,0); c=bs.get(k,0)
        if a==0 and c==0: continue
        print(f"| {k} | {fmt(a)} | {fmt(c)} | {c-a:+,.0f} |")

print("\n\n### TABLE D — correctness parity\n")
print("| Q | master status | branch status | master answer/SQL | branch answer/SQL | divergence |")
print("|---|---|---|---|---|---|")
for q in QORDER:
    m=[r for r in M if r["qid"]==q][0]; b=[r for r in B if r["qid"]==q][0]
    def desc(r):
        if r.get("sql"): return "`"+r["sql"][:90].replace("|","\\|").replace("\n"," ")+"`"
        a=(r.get("answer") or r.get("error") or "")[:90].replace("|","\\|").replace("\n"," ")
        return a or "—"
    div = "SAME" if (m.get("status")==b.get("status")) else "**DIFFERENT**"
    print(f"| {q} | {m.get('status')} | {b.get('status')} | {desc(m)} | {desc(b)} | {div} |")

print("\n\n### SLM decode cost (median rep) — transport-measured\n")
print("| Q | master SLM wall ms | master tok | master ms/tok | branch SLM wall ms | branch tok | branch ms/tok |")
print("|---|---:|---:|---:|---:|---:|---:|")
for q in QORDER:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); return rs[len(rs)//2]
    m=pick(M); b=pick(B)
    mt=m["slm_http_tokens_prompt"]+m["slm_http_tokens_completion"]
    bt=b["slm_http_tokens_prompt"]+b["slm_http_tokens_completion"]
    mr=(m["slm_http_ms_total"]/mt) if mt else None
    br=(b["slm_http_ms_total"]/bt) if bt else None
    print(f"| {q} | {fmt(m['slm_http_ms_total'])} | {fmt(mt)} | {fmt(mr,2) if mr else 'n/a'} | {fmt(b['slm_http_ms_total'])} | {fmt(bt)} | {fmt(br,2) if br else 'n/a'} |")

print("\n\n### Round-trip counts (median rep)\n")
print("| Q | emb rt (m/b) | emb ms (m/b) | rerank rt (m/b) | rerank ms (m/b) | db conn (m/b) | db q (m/b) |")
print("|---|---|---|---|---|---|---|")
for q in QORDER:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); return rs[len(rs)//2]
    m=pick(M); b=pick(B)
    print(f"| {q} | {m['embed_roundtrips']}/{b['embed_roundtrips']} | {fmt(m['embed_ms_total'])}/{fmt(b['embed_ms_total'])} "
          f"| {m['rerank_roundtrips']}/{b['rerank_roundtrips']} | {fmt(m['rerank_ms_total'])}/{fmt(b['rerank_ms_total'])} "
          f"| {m['db_connects']}/{b['db_connects']} | {m['db_queries']}/{b['db_queries']} |")

print("\n\n### SLM call ledger (median rep) — purpose, latency, tokens\n")
for q in QORDER:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); return rs[len(rs)//2]
    for lbl,rows in (("master",M),("branch",B)):
        r=pick(rows); sc=r.get("slm_calls") or []; uc=r.get("usage_calls") or []
        if not sc:
            print(f"- **{q} / {lbl}**: 0 SLM calls")
            continue
        parts=[]
        for i,c in enumerate(sc):
            u=uc[i] if i<len(uc) else {}
            parts.append(f"{c.get('purpose')} ({c.get('duration_ms'):.0f}ms, {u.get('prompt_tokens',0)}p/{u.get('completion_tokens',0)}c, ok={c.get('ok')}, timeout=no)")
        print(f"- **{q} / {lbl}**: {len(sc)} calls — " + "; ".join(parts))
