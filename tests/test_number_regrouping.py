"""The summary SLM inserts thousands separators itself and sometimes gets them wrong.

Measured 2026-09-25 on the live engine: "The average expected price of sale listings is
122,244,1350.14" (true value 1,222,441,350.14) and "The total security deposit … is
23,493,131,4590.000" (true 234,931,314,590). The numeric guard strips commas, so the
VALUE passed; the rendering did not. Only an impossible grouping is rewritten.

Run: `PYTHONPATH=veda_core python -m pytest tests/test_number_regrouping.py -q`
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
from query.result_explainer import _regroup_numbers as R  # noqa: E402


@pytest.mark.parametrize("bad,good", [
    ("is 122,244,1350.14.", "is 1,222,441,350.14."),
    ("is 23,493,131,4590.000 units", "is 234,931,314,590.000 units"),
    ("value 1234,567 here", "value 1,234,567 here"),
])
def test_measured_misgroupings_are_fixed(bad, good):
    assert R(bad) == good


@pytest.mark.parametrize("fine", [
    "There are 1,234,567 rows.",            # correct Western grouping
    "12,34,567 in lakh style",              # correct Indian grouping
    "5,055 projects",
    "32,559,007.00 total",
    "values 1,2,3 and 10,20",               # a list, not a number — never merged
    "on 2026-09-25 at 3.14",
    "",
])
def test_valid_or_list_forms_are_untouched(fine):
    assert R(fine) == fine


def test_digits_never_change():
    out = R("a 122,244,1350.14 b")
    assert out.replace(",", "") == "a 1222441350.14 b"
