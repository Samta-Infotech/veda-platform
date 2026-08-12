"""Coverage for veda_core/veda/feedback.py's RBAC-aware refusal wording.

  veda_core/veda/feedback.py :: explain_failure, _restricted_match

Pure functions, no Django, no heavy ML deps — runs against ``veda_core`` on
``sys.path`` directly, matching how the inference tier imports it (same
pattern as ``tests/test_rbac_filter.py``).

The property under test: a "qualifier_dropped" refusal for a term that names a
REAL but RBAC-restricted table/column (``sm["_rbac_restricted"]``, populated by
``veda.rbac_filter.filter_sm``) must say so explicitly, instead of the generic
"I couldn't map 'X'..." wording used for a genuinely non-existent term — and
must never list other names as suggestions while doing it.

Run from repo root: ``pytest tests/test_feedback.py``
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))

from veda.feedback import (  # noqa: E402
    ACCESS_DENIED_WHAT,
    ACCESS_DENIED_WHY,
    _restricted_match,
    explain_failure,
)


def _sm(columns=None, restricted_tables=None, restricted_columns=None):
    return {
        "columns": columns or {},
        "_rbac_restricted": {
            "tables": restricted_tables or [],
            "columns": restricted_columns or [],
        },
    }


# ---------------------------------------------------------------------------
# _restricted_match
# ---------------------------------------------------------------------------


def test_exact_match_against_a_restricted_column():
    sm = _sm(restricted_columns=["priority"])
    assert _restricted_match("priority", sm) is True


def test_exact_match_is_case_insensitive():
    sm = _sm(restricted_columns=["priority"])
    assert _restricted_match("Priority", sm) is True


def test_exact_match_against_a_restricted_table():
    sm = _sm(restricted_tables=["payroll"])
    assert _restricted_match("payroll", sm) is True


def test_no_match_when_term_merely_shares_a_substring():
    """Deliberately NOT a fuzzy/substring match — 'pri' matching 'priority' would
    leak the restricted name's existence off a much weaker signal than the user
    actually typing it."""
    sm = _sm(restricted_columns=["priority"])
    assert _restricted_match("pri", sm) is False
    assert _restricted_match("priorities", sm) is False


def test_no_match_when_nothing_is_restricted():
    sm = _sm()
    assert _restricted_match("priority", sm) is False


def test_no_match_for_a_blank_term():
    sm = _sm(restricted_columns=["priority"])
    assert _restricted_match("", sm) is False
    assert _restricted_match(None, sm) is False


def test_no_rbac_restricted_key_at_all_is_treated_as_nothing_restricted():
    """A caller that never went through filter_sm (RBAC off, staff bypass) has no
    ``_rbac_restricted`` key — must not raise, must just find no match."""
    assert _restricted_match("priority", {"columns": {}}) is False


# ---------------------------------------------------------------------------
# explain_failure(status="qualifier_dropped")
# ---------------------------------------------------------------------------


def test_qualifier_dropped_for_a_restricted_column_says_permission_not_typo():
    sm = _sm(columns={"worklists_ticket.status": {}},
             restricted_columns=["priority"])

    out = explain_failure("qualifier_dropped", sm, missing="priority")

    assert out["why"] == "You don't have permission to access this data."
    assert "Admin" in out["what_needed"]


def test_qualifier_dropped_for_a_restricted_column_never_suggests_other_names():
    """The whole point of this path: never name a restricted resource, and never
    name ANY other resource either while explaining the refusal — a suggestions
    list here would leak real column names off an access-denied response."""
    sm = _sm(columns={"worklists_ticket.status": {}, "worklists_ticket.title": {}},
             restricted_columns=["priority"])

    out = explain_failure("qualifier_dropped", sm, missing="priority")

    assert out["suggestions"] == []
    assert "status" not in out["text"]
    assert "title" not in out["text"]


def test_qualifier_dropped_for_a_genuinely_unknown_term_is_unchanged():
    """A term that matches nothing at all (real OR restricted) keeps the
    original generic wording and its 'did you mean' suggestions — this path
    must not regress for the ordinary typo/non-existent-column case."""
    sm = _sm(columns={"worklists_ticket.priority_level": {}})

    out = explain_failure("qualifier_dropped", sm, missing="urgency")

    assert "urgency" in out["why"]
    assert out["why"] != "You don't have permission to access this data."


def test_qualifier_dropped_with_no_rbac_restricted_marker_is_unchanged():
    """RBAC off / staff bypass — sm never went through filter_sm, so there is no
    ``_rbac_restricted`` key at all. Must fall back to the ordinary wording, not
    raise, and not claim a permission problem that was never evaluated."""
    sm = {"columns": {"worklists_ticket.priority": {}}}

    out = explain_failure("qualifier_dropped", sm, missing="priority")

    assert out["why"] != "You don't have permission to access this data."
    assert "priority" in out["why"]


# ---------------------------------------------------------------------------
# explain_failure(status="access_denied")
# ---------------------------------------------------------------------------
#
# Reached from a different pipeline stage than "qualifier_dropped": the SQL
# was successfully GENERATED referencing a real column, then REJECTED by
# validate_and_parameterize because narrow_allowed() had already stripped that
# column from the allowlist. See pipeline.py's "invalid" branch — this is the
# earlier bail-out (T53's follow-up) that used to reach the user with no
# feedback object at all.


def test_access_denied_uses_the_shared_wording():
    out = explain_failure("access_denied", {})

    assert out["why"] == ACCESS_DENIED_WHY
    assert out["what_needed"] == ACCESS_DENIED_WHAT


def test_access_denied_never_carries_suggestions():
    out = explain_failure("access_denied", {})

    assert out["suggestions"] == []


def test_access_denied_and_restricted_qualifier_dropped_agree():
    """Both paths reach the same cause (a real, RBAC-restricted resource) via
    different pipeline stages — the user-facing wording must not differ."""
    sm = _sm(restricted_columns=["priority"])

    via_invalid = explain_failure("access_denied", sm)
    via_qualifier = explain_failure("qualifier_dropped", sm, missing="priority")

    assert via_invalid["why"] == via_qualifier["why"]
    assert via_invalid["what_needed"] == via_qualifier["what_needed"]
