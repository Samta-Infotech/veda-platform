import json, statistics as st
OUT="/private/tmp/claude-501/-Users-ekesel-samta-veda-platform/3d25578a-bdc8-4e8a-aadc-11fb0aaf74da/scratchpad/bench/out"
def load(p,kind="timed"):
    return [json.loads(l) for l in open(p) if json.loads(l).get("kind_row")==kind]
B=load(f"{OUT}/branch_v2.jsonl"); M=load(f"{OUT}/master_v2.jsonl")
QORDER=["DB1","DB2","DL1","DL2","FS1","FS2","XS1","XS2"]
print("### Does the latency delta separate from run-to-run noise?\n")
print("Criterion: the three branch runs and the three master runs do not overlap at all")
print("(branch min > master max, or branch max < master min). Anything overlapping is")
print("reported as NOT RESOLVED at n=3 — the two independent master runs of identical code")
print("differ by up to 100% on a single query, so medians alone prove nothing.\n")
print("| Q | master min..max | branch min..max | separated? | direction |")
print("|---|---|---|---|---|")
for q in QORDER:
    mv=sorted(r["wall_ms"] for r in M if r["qid"]==q)
    bv=sorted(r["wall_ms"] for r in B if r["qid"]==q)
    sep = bv[0] > mv[-1] or bv[-1] < mv[0]
    direction = ("branch SLOWER" if bv[0] > mv[-1] else "branch FASTER") if sep else "—"
    print(f"| {q} | {mv[0]:,.0f}..{mv[-1]:,.0f} | {bv[0]:,.0f}..{bv[-1]:,.0f} | {'**YES**' if sep else 'no (overlapping)'} | {direction} |")

print("\n\n### Follow-up turn (chat path)\n")
def loadc(p):
    return [json.loads(l) for l in open(p) if json.loads(l).get("kind_row")=="timed"]
CB=loadc(f"{OUT}/chat_branch.jsonl"); CM=loadc(f"{OUT}/chat_master.jsonl")
print("| script | turn | master status | master med ms | branch status | branch med ms | Δ ms | Δ % |")
print("|---|---|---|---:|---|---:|---:|---:|")
for sid in ("FU1","FU2"):
    for turn in (1,2):
        mr=[r for r in CM if r["script"]==sid and r["turn"]==turn]
        br=[r for r in CB if r["script"]==sid and r["turn"]==turn]
        if not mr or not br: continue
        mm=st.median([r["wall_ms"] for r in mr]); bm=st.median([r["wall_ms"] for r in br])
        ms=sorted({str(r.get("status")) for r in mr}); bs=sorted({str(r.get("status")) for r in br})
        role="opening" if turn==1 else "**follow-up**"
        print(f"| {sid} | {role} | {'/'.join(ms)} | {mm:,.0f} | {'/'.join(bs)} | {bm:,.0f} | {bm-mm:+,.0f} | {(bm-mm)/mm*100:+.1f}% |")

print("\n\n### Follow-up turn — tokens and session behaviour\n")
print("| script | turn | master tokens | branch tokens | master route_source | branch route_source | master sup_slm | branch sup_slm |")
print("|---|---|---:|---:|---|---|---|---|")
for sid in ("FU1","FU2"):
    for turn in (1,2):
        mr=[r for r in CM if r["script"]==sid and r["turn"]==turn]
        br=[r for r in CB if r["script"]==sid and r["turn"]==turn]
        if not mr or not br: continue
        def tok(rs):
            v=[(r.get("engine_usage") or {}).get("total_tokens") for r in rs]
            v=[x for x in v if x is not None]
            return st.median(v) if v else None
        def rs_(rs): return "/".join(sorted({str(r.get("route_source")) for r in rs}))
        def sup(rs): return "/".join(sorted({str(r.get("supervisor_slm_calls")) for r in rs}))
        print(f"| {sid} | {'opening' if turn==1 else '**follow-up**'} | {tok(mr):,.0f} | {tok(br):,.0f} | {rs_(mr)} | {rs_(br)} | {sup(mr)} | {sup(br)} |")
