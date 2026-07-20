"""Model-free regression tests for the evidence-adaptive analytical summary pipeline.

No SLM / embeddings / DB — pure Python. Covers:
- verified findings enrichment (leader/laggard/spread/distribution) — deterministic
- summary MODES (brief vs analytical) and evidence-adaptive prompt/budget
- population consistency (RC-4) preserved AND bounded for large results (Phase 7)
- numerical grounding (every finding number traces to the result)
- schema independence (role-driven, not name-driven)
- explicit-identifier preservation
- no manufactured analytics when there is no verified pattern

Run from repo root: ``pytest tests/test_summary_analytics.py``
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))


def _stat(name, kind, role):
    from veda.result_analyzer import ColumnStat
    return ColumnStat(name=name, kind=kind, role=role)


def _capture_prompt(monkeypatch, query, columns, rows, **kw):
    """Run run_nl_answer with a stubbed SLM; return (prompt, num_predict, answer)."""
    import slm
    import query.result_explainer as re_mod
    seen = {}

    def _cap(p, **k):
        seen["p"] = p
        seen["np"] = k.get("num_predict")
        return "Stub answer."
    monkeypatch.setattr(slm, "call_slm", _cap)
    monkeypatch.setattr(re_mod, "NL_SUMMARY_NUMERIC_GUARD", False, raising=False)
    re_mod.run_nl_answer(query, columns, rows, **kw)
    return seen.get("p", ""), seen.get("np")


# ---------------------------------------------------------------------------
# summary modes
# ---------------------------------------------------------------------------
def test_mode_brief_for_scalar():
    from query.result_explainer import _summary_mode
    assert _summary_mode("SCALAR", 1) == "brief"
    assert _summary_mode("DETAIL_TABLE", 1) == "brief"
    assert _summary_mode(None, 5) == "brief"


def test_mode_analytical_for_grouped_multirow():
    from query.result_explainer import _summary_mode
    assert _summary_mode("GROUPED", 5) == "analytical"
    assert _summary_mode("RANKING", 10) == "analytical"
    assert _summary_mode("TREND", 12) == "analytical"
    # a single-group "grouped" result has nothing to compare → brief
    assert _summary_mode("GROUPED", 1) == "brief"


def test_scalar_prompt_is_brief(monkeypatch):
    prompt, npredict = _capture_prompt(monkeypatch, "total revenue", ["total"],
                                       [{"total": 5000}], result_shape="SCALAR")
    assert "1-2 sentences" in prompt
    assert "3-5 concise sentences" not in prompt


def test_grouped_prompt_is_analytical_and_richer(monkeypatch):
    rows = [{"region": r, "revenue": v} for r, v in
            [("North", 1200), ("West", 900), ("South", 600), ("East", 300)]]
    findings = ["North has the highest revenue at 1200", "East has the lowest revenue at 300"]
    prompt, npredict = _capture_prompt(monkeypatch, "revenue by region", ["region", "revenue"],
                                       rows, result_shape="GROUPED", patterns=findings,
                                       analytical_context={"operation": "SUM"})
    assert "3-5 concise sentences" in prompt
    assert "Resolved analytical context" in prompt
    assert "North has the highest revenue at 1200" in prompt
    assert npredict and npredict >= 300           # larger analytical budget
    # explicit-id + no-invented-calc guards still present
    assert "explicitly asked for an id" in prompt
    assert "growth rate" in prompt


# ---------------------------------------------------------------------------
# verified findings enrichment (deterministic)
# ---------------------------------------------------------------------------
def test_grouped_findings_leader_laggard_spread():
    from veda.result_analyzer import detect_patterns
    stats = [_stat("region", "categorical", "dimension"), _stat("revenue", "numeric", "measure")]
    rows = [{"region": r, "revenue": v} for r, v in
            [("North", 1200), ("West", 900), ("South", 600), ("East", 300)]]
    pats = {p.kind: p.detail for p in
            detect_patterns("GROUPED", stats, rows, [], ["revenue"], dimensions=["region"])}
    assert "leader" in pats and "North" in pats["leader"] and "1200" in pats["leader"]
    assert "laggard" in pats and "East" in pats["laggard"] and "300" in pats["laggard"]
    assert "spread" in pats and "300" in pats["spread"] and "1200" in pats["spread"]
    assert "distribution" in pats


def test_ranking_leader_and_gap():
    from veda.result_analyzer import detect_patterns
    stats = [_stat("customer", "categorical", "dimension"), _stat("spend", "numeric", "measure")]
    rows = [{"customer": c, "spend": v} for c, v in
            [("Acme", 5000), ("Beta", 1000), ("Gamma", 800)]]
    pats = {p.kind for p in
            detect_patterns("RANKING", stats, rows, [], ["spend"], dimensions=["customer"])}
    assert "leader" in pats
    assert "top_gap" in pats            # #1 leads #2 by a wide margin


def test_no_findings_when_single_group():
    from veda.result_analyzer import detect_patterns
    stats = [_stat("region", "categorical", "dimension"), _stat("revenue", "numeric", "measure")]
    rows = [{"region": "North", "revenue": 1200}]
    pats = detect_patterns("GROUPED", stats, rows, [], ["revenue"], dimensions=["region"])
    # one group → no leader/laggard/spread manufactured
    assert not any(p.kind in ("leader", "laggard", "spread") for p in pats)


def test_identifier_dimension_not_used_as_label():
    # dimension column is an identifier role → no leader/laggard label manufactured
    from veda.result_analyzer import detect_patterns
    stats = [_stat("region_id", "numeric", "identifier"), _stat("revenue", "numeric", "measure")]
    rows = [{"region_id": i, "revenue": v} for i, v in [(1, 1200), (2, 300)]]
    pats = detect_patterns("GROUPED", stats, rows, [], ["revenue"], dimensions=["region_id"])
    assert not any(p.kind in ("leader", "laggard") for p in pats)


# ---------------------------------------------------------------------------
# schema independence — role-driven, not name-driven
# ---------------------------------------------------------------------------
def test_findings_schema_independent():
    from veda.result_analyzer import detect_patterns
    # anti-convention names; roles come from ColumnStat (metadata), not suffixes
    stats = [_stat("display_label", "categorical", "dimension"),
             _stat("metric_value", "numeric", "measure")]
    rows = [{"display_label": n, "metric_value": v} for n, v in
            [("Alpha", 90), ("Zeta", 10)]]
    pats = {p.kind: p.detail for p in
            detect_patterns("GROUPED", stats, rows, [], ["metric_value"], dimensions=["display_label"])}
    assert "leader" in pats and "Alpha" in pats["leader"]
    assert "laggard" in pats and "Zeta" in pats["laggard"]


# ---------------------------------------------------------------------------
# numerical grounding — every finding number traces to the result
# ---------------------------------------------------------------------------
def test_finding_numbers_are_grounded():
    from veda.result_analyzer import detect_patterns
    stats = [_stat("region", "categorical", "dimension"), _stat("revenue", "numeric", "measure")]
    rows = [{"region": r, "revenue": v} for r, v in
            [("North", 1200), ("West", 900), ("South", 600)]]
    vals = [r["revenue"] for r in rows]
    pats = {p.kind: p.detail for p in
            detect_patterns("GROUPED", stats, rows, [], ["revenue"], dimensions=["region"])}
    assert str(max(vals)) in pats["leader"]
    assert str(min(vals)) in pats["laggard"]
    # distribution mean equals the real full mean of the scanned population
    assert str(round(statistics.fmean(vals), 2)) in pats["distribution"] or \
           str(int(statistics.fmean(vals))) in pats["distribution"]


# ---------------------------------------------------------------------------
# population consistency (RC-4) + bounded large-result scan (Phase 7)
# ---------------------------------------------------------------------------
def test_population_consistency_and_bound(monkeypatch):
    import query.result_explainer as re_mod
    from veda.result_analyzer import analyze_result
    monkeypatch.setattr(re_mod, "ANALYSIS_MAX_ROWS", 1000, raising=False)
    import veda.result_analyzer as ra_mod
    # 1400 rows: first 1000 (the bound) are 100.0; the tail is 999.0 and must NOT
    # shift the analyzed mean — both the analyzer pattern and the summary metric scan
    # only the first 1000, so they agree.
    rows = [{"grp": f"g{i}", "amt": 100.0} for i in range(1000)] + \
           [{"grp": f"g{i}", "amt": 999.0} for i in range(1000, 1400)]
    # patch the config bound the analyzer reads
    import config
    monkeypatch.setattr(config, "ANALYSIS_MAX_ROWS", 1000, raising=False)
    t0 = time.time()
    ctx = analyze_result("amt per grp", "SELECT grp, amt FROM t GROUP BY grp",
                         ["grp", "amt"], rows, max_rows=200)
    elapsed = time.time() - t0
    assert ctx.row_count == 1400                       # TRUE total reported
    # summary metric (result_explainer) over the SAME bound
    facts = re_mod._numeric_aggregates(["amt"], rows)
    assert facts["amt"]["mean"] == 100.0               # bounded to first 1000 → 100, not mixed
    assert elapsed < 5.0                               # bounded scan stays fast


def test_full_population_when_under_bound():
    # under the bound → exact full-population stats (no sampling divergence)
    import query.result_explainer as re_mod
    rows = [{"amt": v} for v in ([100.0] * 50 + [900.0] * 50)]
    facts = re_mod._numeric_aggregates(["amt"], rows)
    assert facts["amt"]["mean"] == 500.0               # full mean of all 100 rows


# ---------------------------------------------------------------------------
# explicit identifier preservation in prompt
# ---------------------------------------------------------------------------
def test_explicit_id_context_flows_to_prompt(monkeypatch):
    prompt, _ = _capture_prompt(monkeypatch, "which project id has the highest revenue",
                                ["project_id", "revenue"],
                                [{"project_id": 3691, "revenue": 5000},
                                 {"project_id": 3705, "revenue": 3000}],
                                result_shape="RANKING",
                                analytical_context={"explicit_identifier": True})
    assert "explicit id requested=True" in prompt
    # and the prompt still instructs to keep the id when explicitly requested
    assert "explicitly asked for an id" in prompt
