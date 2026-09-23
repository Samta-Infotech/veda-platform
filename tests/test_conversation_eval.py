"""Runs the conversation-layer regression harness in its deterministic mode as
an ordinary test, so the corpus and the fast-path assertions guard every run of
the suite rather than only a hand-invoked script.

Deterministic mode only: it forbids SLM calls outright, so this needs no model,
no network and no Redis, and its result cannot drift run to run. The `--mode
full` half stays a manual/CI-cron job — see evaluation/conversation/README.md.

Pure-python, no Django settings needed (chatbot.nodes imports standalone)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pytest

pytest.importorskip("langchain_core",
                    reason="chatbot/ needs langchain_core; run inside the api container")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation" / "conversation"))  # run_eval + build_suite

import run_eval  # noqa: E402

_HERE = Path(__file__).resolve().parent.parent / "evaluation" / "conversation"


@pytest.fixture(autouse=True)
def _no_slm(monkeypatch):
    monkeypatch.setattr(run_eval.nodes, "call_slm", run_eval._forbid_slm)


def test_no_fast_path_swallows_a_real_question():
    """299 real messages from chat_chatmessage: none of classify_node's
    deterministic short-circuits may fire on a real data question."""
    result = run_eval.run_real_corpus(_HERE / "corpus_real.jsonl")
    assert result["failures"] == [], (
        f"{len(result['failures'])} real messages are being short-circuited away from "
        f"the engine: {result['failures']}")


def test_conditional_clarification_risk_has_not_grown():
    """_is_clarification_answer matches five real one-and-two-word questions.
    They are only reachable while a clarification is pending, so they are not a
    failure — but the set must not grow silently, which is what this pins."""
    result = run_eval.run_real_corpus(_HERE / "corpus_real.jsonl")
    assert sorted(result["conditional"]) == [
        "highest transaction", "notice period", "notice period?",
        "property listed", "total users",
    ]


def test_deterministic_scenarios_all_pass():
    results = run_eval.run_scenarios(_HERE / "scenarios.jsonl", "deterministic")
    assert results["failed"] == [], results["failed"]
    assert results["passed"], "no deterministic scenario ran at all"


def test_target_behaviour_flags_are_not_stale():
    """A scenario marked expected_to_fail that now PASSES means the feature
    landed and the flag should be cleared — caught here rather than quietly
    hiding a working behaviour from the pass count."""
    results = run_eval.run_scenarios(_HERE / "scenarios.jsonl", "deterministic")
    assert results["target_met"] == [], (
        "these now pass — clear expected_to_fail in scenarios.jsonl: "
        f"{[r['id'] for r in results['target_met']]}")


# ---------------------------------------------------------------------------
# suite.jsonl — the broad single-message suite (346 messages, 16 categories)
# ---------------------------------------------------------------------------

def _suite():
    """Both fixtures, exactly as the CLI passes them. Omitting DOCUMENT_FRAME made
    every document row fall through to the model and fail as _SlmForbidden — the CLI
    passed while pytest did not, which is the sort of drift this wrapper exists to
    prevent."""
    import build_suite
    return run_eval.run_suite(_HERE / "suite.jsonl", "deterministic",
                              build_suite.ACTIVE_FRAME,
                              build_suite.DOCUMENT_FRAME), build_suite


def test_the_broad_suite_has_no_unrecorded_failures():
    """393 hand-written conversational messages — greetings and their typos, Hinglish,
    emoji, acknowledgements, recall, presentation, reset, drill up/down, shape changes,
    comparisons, clarification answers, junk, injection, 15 real document questions, the
    shape and drill phrasings that mean nothing asked of a document, and 20 real data
    questions as the control group. Anything failing that is NOT in build_suite.KNOWN_GAPS is a
    regression."""
    results, _ = _suite()
    assert results["failed"] == [], results["failed"]
    assert results["passed"], "the suite did not run at all"


def test_the_known_gap_set_has_not_grown():
    """The recorded coverage holes are pinned by COUNT and by MEMBERSHIP. A new phrasing
    that stops working shows up as an unrecorded failure above; one that starts working
    shows up here, so the markers cannot rot in either direction."""
    results, build_suite = _suite()
    assert results["gaps_now_passing"] == [], (
        "these now pass — remove them from build_suite.KNOWN_GAPS: "
        f"{[r['message'] for r in results['gaps_now_passing']]}")
    assert len(results["known_gaps"]) == len(build_suite.KNOWN_GAPS)


def test_every_real_data_question_reaches_the_engine():
    """The control group on its own: no conversational fast path may swallow a genuine
    question. This is the single most important property in the file."""
    results, _ = _suite()
    swallowed = [r for r in results["failed"] if r["category"] == "data_question"]
    assert swallowed == [], swallowed
