import json, statistics as st
OUT="/private/tmp/claude-501/-Users-ekesel-samta-veda-platform/3d25578a-bdc8-4e8a-aadc-11fb0aaf74da/scratchpad/bench/out"
def load(p,kind="timed"):
    return [json.loads(l) for l in open(p) if json.loads(l).get("kind_row")==kind]
B=load(f"{OUT}/branch_v2.jsonl"); M=load(f"{OUT}/master_v2.jsonl")
M1=load(f"{OUT}/master_v2.jsonl"); M0=load(f"{OUT}/master2.jsonl"); B0=load(f"{OUT}/branch.jsonl")
QORDER=["DB1","DB2","DL1","DL2","FS1","FS2","XS1","XS2"]
def p90(v):
    v=sorted(v); k=(len(v)-1)*0.9; f=int(k); c=min(f+1,len(v)-1)
    return v[f]+(v[c]-v[f])*(k-f)
def med(rows,q,f="wall_ms"): return st.median([r[f] for r in rows if r["qid"]==q])
def stat(rows,q):
    v=[r["wall_ms"] for r in rows if r["qid"]==q]
    return min(v), st.median(v), p90(v), max(v)

print("### Measurement noise — two independent master runs, same code, same config\n")
print("| Q | master run A med | master run B med | spread | spread % |")
print("|---|---:|---:|---:|---:|")
for q in QORDER:
    a=med(M0,q); b=med(M1,q)
    print(f"| {q} | {a:,.0f} | {b:,.0f} | {abs(b-a):,.0f} | {abs(b-a)/min(a,b)*100:.0f}% |")
sa=sum(med(M0,q) for q in QORDER); sb=sum(med(M1,q) for q in QORDER)
print(f"| **sum** | **{sa:,.0f}** | **{sb:,.0f}** | {abs(sb-sa):,.0f} | {abs(sb-sa)/min(sa,sb)*100:.0f}% |")

print("\n\n### Latency restricted to queries where BOTH trees produced an ANSWER\n")
both=[]
for q in QORDER:
    ms={r["status"] for r in M if r["qid"]==q}; bs={r["status"] for r in B if r["qid"]==q}
    if ms=={"answered"} and bs=={"answered"}: both.append(q)
print(f"Parity set (both answered on all 3 reps): {', '.join(both)}\n")
print("| Q | master min/med/p90 | branch min/med/p90 | Δ med | Δ med % |")
print("|---|---|---|---:|---:|")
tm=tb=0
for q in both:
    a=stat(M,q); b=stat(B,q); tm+=a[1]; tb+=b[1]
    print(f"| {q} | {a[0]:,.0f} / {a[1]:,.0f} / {a[2]:,.0f} | {b[0]:,.0f} / {b[1]:,.0f} / {b[2]:,.0f} | {b[1]-a[1]:+,.0f} | {(b[1]-a[1])/a[1]*100:+.1f}% |")
print(f"| **sum of medians** | **{tm:,.0f}** | **{tb:,.0f}** | {tb-tm:+,.0f} | {(tb-tm)/tm*100:+.1f}% |")

print("\n\n### Token cost restricted to the parity set (transport-measured)\n")
print("| Q | master tok | branch tok | Δ | master calls | branch calls |")
print("|---|---:|---:|---:|---:|---:|")
pm=pb=0
for q in both:
    def pick(rows):
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); return rs[len(rs)//2]
    a=pick(M); b=pick(B)
    at=a["slm_http_tokens_prompt"]+a["slm_http_tokens_completion"]
    bt=b["slm_http_tokens_prompt"]+b["slm_http_tokens_completion"]
    pm+=at; pb+=bt
    print(f"| {q} | {at:,} | {bt:,} | {bt-at:+,} | {a['slm_http_posts']} | {b['slm_http_posts']} |")
print(f"| **total** | **{pm:,}** | **{pb:,}** | {pb-pm:+,} | | |")

print("\n\n### Decode-seconds cost model (self-hosted; no $/token applies)\n")
allc=[]
for rows,lbl in ((M,"master"),(B,"branch")):
    for r in rows:
        for c in (r.get("slm_http_calls") or []):
            if c["completion_tokens"]:
                allc.append((lbl, c["ms"], c["prompt_tokens"], c["completion_tokens"]))
tot_ms=sum(c[1] for c in allc); tot_ct=sum(c[3] for c in allc); tot_pt=sum(c[2] for c in allc)
print(f"- Observed across all {len(allc)} timed SLM round-trips in both runs:")
print(f"  total SLM wall time {tot_ms/1000:,.1f} s, prompt tokens {tot_pt:,}, completion tokens {tot_ct:,}")
print(f"  aggregate {tot_ms/ (tot_pt+tot_ct):.2f} ms per total token, {tot_ms/tot_ct:.1f} ms per completion token")
print("\n| tree | SLM wall s / query (median-rep sum over 8 queries) | tokens | per-1000-queries SLM wall (hours) |")
print("|---|---:|---:|---:|")
for rows,lbl in ((M,"master"),(B,"branch")):
    s=0; t=0
    for q in QORDER:
        rs=[r for r in rows if r["qid"]==q]; rs.sort(key=lambda r:r["wall_ms"]); r=rs[len(rs)//2]
        s+=r["slm_http_ms_total"]; t+=r["slm_http_tokens_prompt"]+r["slm_http_tokens_completion"]
    print(f"| {lbl} | {s/1000:,.1f} | {t:,} | {s/1000*1000/8/3600:,.2f} |")
