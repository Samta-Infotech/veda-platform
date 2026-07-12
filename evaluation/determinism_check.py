#!/usr/bin/env python3
"""Two-seed determinism check for anchor decisions.

Anchor/table choices must be a pure function of (query, schema artifacts) — never of
Python's per-process set/dict iteration order. A hash-order dependence is exactly the
"q25 anchor flips between processes" class on file. This script runs the same fixed
query battery in two subprocesses with PYTHONHASHSEED=0 and =1 and diffs the ranked
anchor decisions; any difference is a determinism defect.

Host-runnable: uses the on-disk semantic model / relationship graph (dev fallback
artifacts), no services required. The registry fast-path concept matcher is included
opportunistically (skipped silently when its stores aren't reachable from the host).

Exit code 0 = deterministic, 1 = divergence (printed).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "veda_core")

QUERIES = [
    "Which category contributes the highest value among all completed payments?",
    "Which category contributes the highest value among all captured payments?",
    "What is the total paid amount across all payment transactions?",
    "How many active listings do we have for each property, grouped by their market status?",
    "list recent payments made for our properties",
    "Which properties have the most expensive financial records logged to date?",
    "how many users are there",
    "approved sale listings per property",
    "show reminders by category",
    "average transaction amount for each property",
]

_CHILD = r"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sm = json.load(open("data/veda_semantic_model.json"))
from query.join_planner import score_anchors, load_graph
from semantic.name_tokens import table_tokens
from retrieval.query_enrichment import _singularize
graph = load_graph("data/veda_relationship_graph.json")
tabs = sorted(sm.get("tables", {}))
queries = json.loads(sys.argv[1])
out = {}
for q in queries:
    qt = {_singularize(w) for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 2}
    cand = [t for t in tabs if qt & table_tokens(t, sm)][:12]
    ranked = score_anchors(q, cand, {}, graph=graph, sm=sm)
    out[q] = [[r.table, r.score] for r in ranked[:3]]
    # registry fast-path concept match, when reachable from the host
    try:
        from semantic import registry as reg
        m = reg.match_concepts(q)
        out[q].append(["registry", str(m)[:120]])
    except Exception:
        pass
print(json.dumps(out, sort_keys=True))
"""


def run(seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    p = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(QUERIES)],
                       cwd=CORE, env=env, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        print(f"seed {seed}: child failed:\n{p.stderr[-1500:]}", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    a, b = run("0"), run("1")
    diverged = 0
    for q in QUERIES:
        if a.get(q) != b.get(q):
            diverged += 1
            print(f"DIVERGENCE: {q!r}\n  seed0: {a.get(q)}\n  seed1: {b.get(q)}")
    print(f"determinism: {len(QUERIES)} queries, {diverged} divergent")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
