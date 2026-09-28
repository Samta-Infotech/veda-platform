"""Planner agent — the F.3/F.4 tables from an eval_v3_inproc run (host; no engine needed).

    python3 scripts/agent_v3_report.py --compound reports/raw/agent/v3_compound.jsonl \
        --alone reports/raw/agent/v3_alone.jsonl --graded reports/raw/agent/v3_graded.json \
        [--prev reports/raw/v3inproc/graded.json] [--steps 1a,8a,…]

Prints (markdown): the 20 × 3 table (in-message / alone, with the agent marked), the scores
all-60 and homzhub-only against the previous run, per agent-planned part its steps / tool
calls / wall / validation rejections / firewall verdict, latency p50/p90 for agent-planned
vs fast-lane parts, tokens in/out per part, an OFFLINE re-validation of every accepted plan
against exactly the tool log it was accepted on (identifier provenance only), and the step
logs of the requested parts.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))

SYM = {"correct": "✔", "wrong": "✘", "clarify": "?", "timeout": "t", None: "·"}


def _load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p and os.path.exists(p) else []


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def revalidate(rec) -> list:
    """Every identifier of an accepted plan traced to a RESULT of its own logged tool calls."""
    from veda.agent.plan import Plan, seen_identifiers, split_ref, _vkey
    plan = rec.get("plan") or {}
    log = [{"id": c["id"], "tool": c["tool"], "args": c.get("args") or {}, "result": c.get("result") or {}}
           for c in rec.get("tool_calls") or []]
    s = seen_identifiers(log)
    bad = []
    for t in plan.get("tables") or []:
        if t not in s.tables:
            bad.append(f"table {t}")
    for j in plan.get("joins") or []:
        if j.get("via") not in s.routes:
            bad.append(f"join {j.get('via')}")
    refs = [(x.get("table"), x.get("column")) for k in ("projection", "group_by", "filters")
            for x in plan.get(k) or []]
    refs += [(a.get("table"), a.get("column")) for a in plan.get("aggregates") or [] if a.get("column")]
    for o in plan.get("order") or []:
        t, c = split_ref(o.get("expr"))
        if t is not None:
            refs.append((t, c))
    if plan.get("time"):
        refs.append((plan["time"].get("table"), plan["time"].get("column")))
    for t, c in refs:
        if (t, c) not in s.columns:
            bad.append(f"column {t}.{c}")
    for f in plan.get("filters") or []:
        if f.get("op") in ("is_null", "is_not_null"):
            continue
        t, c, v = f.get("table"), f.get("column"), f.get("value")
        probed = any(k[:2] == (t, c) for k in s.probed)
        vals = v if isinstance(v, list) else [v]
        known = s.values.get((t, c), set())
        if not probed and not all(_vkey(x) in known for x in vals):
            bad.append(f"value {t}.{c}={v}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compound")
    ap.add_argument("--alone")
    ap.add_argument("--graded")
    ap.add_argument("--prev", default="reports/raw/v3inproc/graded.json")
    ap.add_argument("--gt", default="reports/raw/qv3_ground_truth.json")
    ap.add_argument("--steps", default="1a,8a,11a,12a,13a,16a,18a,19a,20a")
    a = ap.parse_args()
    gt = {q["n"]: q for q in json.load(open(a.gt))["questions"]}
    comp = {r["n"]: r for r in _load(a.compound)}
    alone = {(r["n"], r["part"]): r for r in _load(a.alone) if not r.get("skipped")}
    graded = json.load(open(a.graded)) if a.graded and os.path.exists(a.graded) else {"compound": {}, "alone": {}}
    prev = json.load(open(a.prev)) if a.prev and os.path.exists(a.prev) else {}

    # which part of a message did the agent plan (and how)
    def agent_of_compound(n, gi):
        r = comp.get(n) or {}
        items = r.get("items") or []
        gp = gt[n]["parts"][gi]
        idx = gi if len(items) == len(gt[n]["parts"]) else next(
            (i for i, x in enumerate(items) if str(x.get("source_id")) == str(_src(gp))), None)
        if idx is None:
            return None, None
        ags = [x for x in (r.get("agent") or []) if x.get("part") == idx]
        return (ags[-1] if ags else None), (items[idx] if idx < len(items) else None)

    def agent_of_alone(n, p):
        r = alone.get((n, p)) or {}
        ags = r.get("agent") or []
        return (ags[-1] if ags else None), ((r.get("items") or [None])[0])

    print("## 20 × 3 — in the message / alone (✔ correct · ✘ wrong · ? clarify · t timeout; ᴬ = the agent planned it)\n")
    print("| # | (a) homzhub | (b) handbook | (c) tabular |")
    print("|---|---|---|---|")
    for n in sorted(gt):
        cells = []
        for gi, gp in enumerate(gt[n]["parts"]):
            key = f"{n}{gp['part']}"
            gc = (graded["compound"].get(key) or {}).get("grade")
            ga = (graded["alone"].get(key) or {}).get("grade")
            ac, _ = agent_of_compound(n, gi)
            aa, _ = agent_of_alone(n, gp["part"])
            mc = "ᴬ" if ac and ac.get("kind") == "sql" else ""
            ma = "ᴬ" if aa and aa.get("kind") == "sql" else ""
            cells.append(f"{SYM[gc]}{mc} / {SYM[ga]}{ma}")
        print(f"| {n} | " + " | ".join(cells) + " |")

    def score(mode, only=None):
        res = graded.get(mode) or {}
        keys = [k for k in res if only is None or k.endswith(only)]
        return {g: sum(1 for k in keys if res[k]["grade"] == g) for g in ("correct", "wrong", "clarify", "timeout")}

    def pscore(mode, only=None):
        key = {"compound": "in_message", "alone": "alone"}[mode]
        out = {"correct": 0, "wrong": 0, "clarify": 0, "timeout": 0}
        for n, v in prev.items():
            for p, g in (v.get(key) or {}).items():
                if only is None or p == only:
                    out[g] = out.get(g, 0) + 1
        return out
    print("\n## Scores\n")
    print("| run | all 60 | homzhub (a) | handbook (b) | tabular (c) | previous all 60 | previous homzhub |")
    print("|---|---|---|---|---|---|---|")
    for mode, label in (("compound", "in the message"), ("alone", "each part alone")):
        s_all, s_a, s_b, s_c = score(mode), score(mode, "a"), score(mode, "b"), score(mode, "c")
        p_all, p_a = pscore(mode), pscore(mode, "a")
        fmt = lambda d: f"{d['correct']} ✔ · {d['wrong']} ✘ · {d['clarify']} ? · {d['timeout']} t"
        print(f"| {label} | **{fmt(s_all)}** | {fmt(s_a)} | {fmt(s_b)} | {fmt(s_c)} | {fmt(p_all)} | {fmt(p_a)} |")

    # per agent-planned part
    print("\n## Every part the agent ran on\n")
    print("| part | mode | trigger | outcome | steps | tool calls | agent wall s | validation rejections | firewall | grade | unvalidated ids |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows, unval_total, accepted = [], 0, 0
    for n in sorted(gt):
        for gi, gp in enumerate(gt[n]["parts"]):
            key = f"{n}{gp['part']}"
            for mode, (ag, it) in (("msg", agent_of_compound(n, gi)), ("alone", agent_of_alone(n, gp["part"]))):
                if not ag:
                    continue
                g = (graded["compound" if mode == "msg" else "alone"].get(key) or {}).get("grade")
                rej = sum(1 for v in ag.get("validation") or [] if v.get("errors"))
                bad = revalidate(ag) if ag.get("kind") == "sql" else []
                if ag.get("kind") == "sql":
                    accepted += 1
                    unval_total += len(bad)
                fw = (it or {}).get("firewall")
                trig = next((x.get("trigger") for x in [ag] if x.get("trigger")), None) or (it or {}).get("agent_trigger")
                print(f"| {key} | {mode} | {trig or (it or {}).get('frame_reason') or ''} | {ag.get('kind')} ({ag.get('reason')}) | "
                      f"{len(ag.get('steps') or [])} | {(ag.get('budget') or {}).get('tool_calls')} | {ag.get('wall_s')} | "
                      f"{rej} | {fw or ''} | {SYM[g]} | {len(bad)}{(': ' + '; '.join(bad[:2])) if bad else ''} |")
                rows.append((mode, ag, it, g))
    print(f"\nAccepted plans: {accepted}; identifiers not traceable to a tool result of their own run: **{unval_total}**.")

    # latency: part wall, agent-planned vs fast lane (in the message)
    ag_ms, fl_ms, tok = [], [], []
    for n, r in comp.items():
        ags = {x.get("part") for x in (r.get("agent") or []) if x.get("kind") == "sql"}
        for i, ms in enumerate(r.get("parts_ms") or []):
            items = r.get("items") or []
            lane = (items[i] if i < len(items) else {}).get("lane")
            if lane == "rag":
                continue
            (ag_ms if i in ags else fl_ms).append(ms / 1000.0)
            pt = sum(int(c.get("prompt_tokens") or 0) for c in r.get("slm_calls") or [] if c.get("part") == i)
            ct = sum(int(c.get("completion_tokens") or 0) for c in r.get("slm_calls") or [] if c.get("part") == i)
            tok.append(("agent" if i in ags else "fast", pt, ct))
    walls = [x.get("wall_s") for r in comp.values() for x in (r.get("agent") or []) if x.get("wall_s") is not None]
    print("\n## Latency (in the message, SQL/tabular parts)\n")
    print("| parts | n | p50 s | p90 s | max s | tokens in p50 | tokens out p50 |")
    print("|---|---|---|---|---|---|---|")
    for lab, xs in (("agent-planned", ag_ms), ("fast lane / frame / other", fl_ms)):
        tin = [t[1] for t in tok if t[0] == ("agent" if lab.startswith("agent") else "fast")]
        tout = [t[2] for t in tok if t[0] == ("agent" if lab.startswith("agent") else "fast")]
        print(f"| {lab} | {len(xs)} | {_pct(xs, .5) and round(_pct(xs, .5), 1)} | {_pct(xs, .9) and round(_pct(xs, .9), 1)} | "
              f"{max(xs) if xs else None and round(max(xs), 1)} | {_pct(tin, .5)} | {_pct(tout, .5)} |")
    if walls:
        print(f"\nAgent loop alone (every run, incl. ones that did not plan): n={len(walls)} "
              f"p50={_pct(walls, .5)} s p90={_pct(walls, .9)} s max={max(walls)} s")

    # step logs
    print("\n## Agent step logs\n")
    for key in [k for k in a.steps.split(",") if k]:
        n, p = int(key[:-1]), key[-1]
        gi = next(i for i, x in enumerate(gt[n]["parts"]) if x["part"] == p)
        for mode, (ag, it) in (("in the message", agent_of_compound(n, gi)), ("alone", agent_of_alone(n, p))):
            g = (graded["compound" if mode != "alone" else "alone"].get(key) or {})
            print(f"### {key} — {mode}: {SYM[g.get('grade')]} {'; '.join(g.get('why') or [])[:160]}")
            if not ag:
                print(f"- agent not called (frame path: {(it or {}).get('frame_reason')}, head {(it or {}).get('head')})\n")
                continue
            print(f"- trigger `{(it or {}).get('agent_trigger') or ''}` → **{ag.get('kind')}** ({ag.get('reason')}), "
                  f"{len(ag.get('steps') or [])} steps, {(ag.get('budget') or {}).get('tool_calls')} tool calls, {ag.get('wall_s')} s")
            for c in ag.get("tool_calls") or []:
                print(f"  - `[{c['id']}{' auto' if c.get('auto') else ''}] {c['tool']}({json.dumps(c.get('args'), ensure_ascii=False)[:110]})`")
            for s in ag.get("steps") or []:
                act = json.dumps(s.get("action"), ensure_ascii=False)[:220]
                print(f"  - step {s.get('i')} ({s.get('ms')} ms, {s.get('prompt_tokens')}→{s.get('completion_tokens')} tok): "
                      f"_{s.get('thought', '')}_ `{act}` {s.get('error', '')}")
            for v in ag.get("validation") or []:
                if v.get("errors"):
                    print(f"  - rejected at step {v.get('step')}: {'; '.join(v['errors'])[:240]}")
            if ag.get("sql"):
                print(f"  - SQL: `{ag['sql'][:400]}`")
            if ag.get("message"):
                print(f"  - message: {ag['message'][:200]}")
            print()


def _src(gp):
    s = str(gp.get("source") or "").lower()
    if "invoices" in s:
        return 4
    if "catalog" in s or "parquet" in s:
        return 5
    if "docs" in s or ".pdf" in s or gp.get("type") == "doc":
        return 3
    return 2


if __name__ == "__main__":
    main()
