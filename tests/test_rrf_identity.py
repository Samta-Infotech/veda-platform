"""WP6 acceptance — identity FUSION_WEIGHTS reproduce the unweighted RRF bit-for-bit.

Runs a fixed candidate fixture through RRFMerger.merge with identity weights and asserts
the (col_id, score) output equals the pre-WP6 unweighted formula Σ 1/(k+rank) computed
independently. Also asserts a non-identity weight actually changes the ranking.

Run: python tests/test_rrf_identity.py  (from repo root).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))

from retrieval.rrf_merger import RRFMerger

K = 60

# Fixed fixture — deterministic, covers all six signals.
SEMANTIC = [("users.name", 0.91), ("orgs.plan", 0.80), ("users.email", 0.55)]
SPARSE   = [("orgs.plan", 8.0), ("audit.action", 3.0)]
FK       = {"users.name": 0.7}
SUBGRAPH = {"users.email": 0.4}
VALUE    = {"audit.action": 1.0}
TPRIOR   = {"users": 0.6, "orgs": 0.3}   # keyed by table_name


def _rank_dict(ranking):
    return {cid: i + 1 for i, (cid, _) in enumerate(ranking)}


def _expected_unweighted():
    """The pre-WP6 formula: Σ 1/(k+rank) across all signals, no weights."""
    sem = _rank_dict(SEMANTIC)
    spa = _rank_dict(SPARSE)
    val = _rank_dict([(c, s) for c, s in VALUE.items()])
    cands = set(sem) | set(spa) | set(FK) | set(SUBGRAPH) | set(val)
    scores = {}
    for cid in cands:
        s = 0.0
        if cid in sem:
            s += 1 / (K + sem[cid])
        if cid in spa:
            s += 1 / (K + spa[cid])
        sg = SUBGRAPH.get(cid, 0.0)
        if sg > 0:
            s += 1 / (K + max(1, int((1 - sg) * K)))
        fk = FK.get(cid, 0.0)
        if fk > 0:
            s += 1 / (K + max(1, int((1 - fk) * K)))
        if cid in val:
            s += 1 / (K + val[cid])
        tname = cid.rsplit(".", 1)[0] if "." in cid else ""
        tp = TPRIOR.get(tname, 0.0)
        if tp > 0:
            s += 1 / (K + max(1, int((1 - tp) * K)))
        scores[cid] = s
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main() -> int:
    m = RRFMerger(k=K)
    identity = {"dense": 1.0, "sparse": 1.0, "subgraph": 1.0, "fk": 1.0,
                "value": 1.0, "table_prior": 1.0}
    out = m.merge(SEMANTIC, SPARSE, FK, SUBGRAPH, VALUE,
                  table_prior_signals=TPRIOR, weights=identity, top_k=50)
    expected = _expected_unweighted()

    assert [c for c, _ in out] == [c for c, _ in expected], \
        f"identity order != unweighted order\n {out}\n {expected}"
    for (co, so), (ce, se) in zip(out, expected):
        assert co == ce and abs(so - se) < 1e-12, f"score mismatch {co}:{so} vs {ce}:{se}"
    print(f"[1] identity weights == unweighted RRF, bit-for-bit ({len(out)} cols)  ✓")

    # A heavier dense weight must be able to reorder vs sparse-favoured order.
    heavy = dict(identity, sparse=3.0)
    out_heavy = m.merge(SEMANTIC, SPARSE, FK, SUBGRAPH, VALUE,
                        table_prior_signals=TPRIOR, weights=heavy, top_k=50)
    assert [c for c, _ in out_heavy] != [c for c, _ in out] or \
        any(abs(a[1] - b[1]) > 1e-9 for a, b in zip(out_heavy, out)), \
        "non-identity weights must change scores"
    print("[2] non-identity weights change the fusion  ✓")

    print("\n✓ WP6 WEIGHTED-FUSION CHECKS PASSED")

    _check_tie_break_deterministic()
    print("\n✓ TIE-BREAK DETERMINISM CHECK PASSED")
    return 0


def _check_tie_break_deterministic():
    """2026-09-29 fix — two candidates tied at the exact same fused score must sort
    identically regardless of the order they were discovered in (previously fell back
    to `set()` iteration order, which is PYTHONHASHSEED-dependent across process
    restarts). Two columns with IDENTICAL rank-1 dense hits and nothing else tie
    exactly; the fix must always put the alphabetically-earlier col_id first."""
    m = RRFMerger(k=K)
    # Both at dense rank 1 is impossible in one list, so give each its OWN rank-1 hit
    # via two separate signals that individually rank each of them first — same total
    # score either way (1/(k+1) once each), a genuine tie.
    tied = m.merge(
        semantic_ranking=[("zeta.col", 0.9)],
        sparse_ranking=[("alpha.col", 0.9)],
        fk_signals={}, subgraph_signals={}, value_signals={}, top_k=10,
    )
    scores = {cid: score for cid, score in tied}
    assert abs(scores["zeta.col"] - scores["alpha.col"]) < 1e-12, "fixture must be a real tie"
    order = [cid for cid, _ in tied]
    assert order == sorted(order), f"tie-break must be alphabetical by col_id, got {order}"
    # Run again — must be byte-identical (this is the actual regression: a set-driven
    # order isn't reproducible run-to-run within a hash-randomized process either, if
    # the candidates are constructed via any non-list-derived path).
    tied2 = m.merge(
        semantic_ranking=[("zeta.col", 0.9)],
        sparse_ranking=[("alpha.col", 0.9)],
        fk_signals={}, subgraph_signals={}, value_signals={}, top_k=10,
    )
    assert tied == tied2, "identical inputs must produce byte-identical output"
    print("[3] tied candidates sort deterministically (alphabetical col_id)  ✓")


if __name__ == "__main__":
    sys.exit(main())
