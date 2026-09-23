"""Run the whole-turn contract checker as part of the test suite.

WHY THIS FILE EXISTS. `tests/tools/explainability_invariants.py` is the only thing
that validates the SHAPE OF A COMPLETE TURN against CHAT_API_CONTRACT.md — closed
vocabularies, terminal consistency, leakage, cross-block agreement. An audit found
that nothing ran it: there is no CI in this repo (no `.github/workflows`, no
Makefile, no tox/pytest config), so it only ever executed when someone manually
captured a stream and remembered the command.

Two things run here, and they fail for different reasons:

  * the checker's own SELF-TEST — it feeds itself a payload with planted defects and
    fails unless it still catches them. This is what stops the checker rotting into
    something that passes everything.
  * the COMMITTED FIXTURES in tests/fixtures/sse/ — real captured streams from the
    live stack (ids normalised), one per answer shape. These are the regression
    half: a change that breaks the turn contract fails here without anyone needing
    a running stack.

Neither needs docker, a database, or the network.
"""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOL = os.path.join(_ROOT, "tests", "tools", "explainability_invariants.py")
_FIX = os.path.join(_ROOT, "tests", "fixtures", "sse")


def _run(args):
    return subprocess.run([sys.executable, _TOOL, *args],
                          capture_output=True, text=True, timeout=120)


def test_the_checker_still_catches_its_own_planted_defects():
    """If this fails the checker has gone blind and every clean report it has ever
    produced is worthless. It is deliberately the first assertion in the file."""
    r = _run(["--self-test"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CHECKER IS BLIND" not in r.stdout


def _fixtures():
    if not os.path.isdir(_FIX):
        return []
    return sorted(f for f in os.listdir(_FIX) if f.endswith(".sse"))


def test_there_are_fixtures_to_check():
    """A green suite with no fixtures would be a false sense of security."""
    names = {os.path.splitext(f)[0] for f in _fixtures()}
    for required in ("sql", "documents", "refusal", "denied", "smalltalk"):
        assert required in names, f"no captured stream for the {required} path"


@pytest.mark.parametrize("fixture", _fixtures())
def test_a_captured_turn_satisfies_the_contract(fixture):
    path = os.path.join(_FIX, fixture)
    label = os.path.splitext(fixture)[0]
    r = _run([f"{path}::{label}"])
    assert r.returncode == 0, (
        f"{label} violates the turn contract:\n{r.stdout}\n{r.stderr}")


def test_the_checker_rejects_a_stream_it_should_reject(tmp_path):
    """Proof the fixture check above can actually fail: hand it a turn that reports
    a terminal status and then keeps sending progress."""
    bad = tmp_path / "bad.sse"
    bad.write_text(
        'event: thinking\ndata: {"phase": "completed", "steps": {"type": "thinking", '
        '"status": "completed", "current_step": null, "total_steps": 4, "steps": [], '
        '"evidence": {}, "execution": {"type": "sql"}, "timing": {}}}\n\n'
        'event: thinking\ndata: {"phase": "visualization_prep", "message": "x"}\n\n')
    r = _run([f"{bad}::planted"])
    assert r.returncode != 0
    assert "TERMINAL" in r.stdout
