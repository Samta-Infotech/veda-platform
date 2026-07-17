"""apps.chat.table_rendering — the chat table sub-block's markdown rendering
(2026-07-17). Pure-python, no Django/redis/chatbot import chain needed (the
whole point of pulling this out of services.py — see that module's own
docstring). Run: ``pytest tests/test_table_rendering.py``"""
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.chat.table_rendering import (
    fmt_cell, fmt_header, project_display_columns, rows_to_markdown_table,
)


# ---------------------------------------------------------------------------
# fmt_cell
# ---------------------------------------------------------------------------

def test_none_and_empty_render_as_em_dash_not_literal_none():
    assert fmt_cell(None) == "—"
    assert fmt_cell("") == "—"


def test_decimal_formatted_with_commas_like_summary_side():
    assert fmt_cell(Decimal("98400.000")) == "98,400"


def test_int_and_float_thousands_separator():
    assert fmt_cell(98400) == "98,400"
    assert fmt_cell(5000.5) == "5,000.5"


def test_bool_renders_as_yes_no():
    assert fmt_cell(True) == "yes"
    assert fmt_cell(False) == "no"


def test_pipe_character_escaped_so_table_structure_survives():
    assert fmt_cell("Priya|Patel") == "Priya\\|Patel"


def test_long_text_is_capped_with_ellipsis():
    long_text = "A" * 200
    out = fmt_cell(long_text)
    assert len(out) <= 80
    assert out.endswith("…")


def test_short_text_passthrough_unchanged():
    assert fmt_cell("Rahul Sharma") == "Rahul Sharma"


# ---------------------------------------------------------------------------
# fmt_header
# ---------------------------------------------------------------------------

def test_header_humanizes_column_names():
    assert fmt_header("customer_name") == "Customer Name"
    assert fmt_header("amount") == "Amount"


# ---------------------------------------------------------------------------
# project_display_columns
# ---------------------------------------------------------------------------

def test_project_drops_identifier_columns():
    cols = ["account_id", "customer_name", "amount", "payment_status_id"]
    rows = [[1, "Rahul", 98400, 3], [2, "Priya", 5000, 1]]
    display_columns = ["customer_name", "amount"]   # engine already excluded the ids
    out_cols, out_rows = project_display_columns(cols, rows, display_columns)
    assert out_cols == ["customer_name", "amount"]
    assert out_rows == [["Rahul", 98400], ["Priya", 5000]]


def test_project_preserves_original_column_order():
    """display_columns may arrive in a different order (column_stats order) —
    the rendered table should keep the SQL's own column order, not re-sort."""
    cols = ["amount", "customer_name"]
    rows = [[98400, "Rahul"]]
    display_columns = ["customer_name", "amount"]   # reversed vs. cols
    out_cols, out_rows = project_display_columns(cols, rows, display_columns)
    assert out_cols == ["amount", "customer_name"]   # unchanged order
    assert out_rows == [[98400, "Rahul"]]


def test_project_fails_safe_when_display_columns_absent():
    cols = ["id", "amount"]
    rows = [[1, 100]]
    out_cols, out_rows = project_display_columns(cols, rows, None)
    assert out_cols == cols and out_rows == rows


def test_project_fails_safe_when_no_overlap():
    """A total name mismatch (e.g. analytics computed against different column
    names) must never render an empty table — fall back to showing everything."""
    cols = ["id", "amount"]
    rows = [[1, 100]]
    out_cols, out_rows = project_display_columns(cols, rows, ["totally_different_col"])
    assert out_cols == cols and out_rows == rows


def test_project_fails_safe_when_display_columns_is_everything():
    """No identifiers to drop — filtering would be a no-op, so it's skipped
    (same result either way, but exercises the len(idx)==len(cols) branch)."""
    cols = ["customer_name", "amount"]
    rows = [["Rahul", 100]]
    out_cols, out_rows = project_display_columns(cols, rows, ["customer_name", "amount"])
    assert out_cols == cols and out_rows == rows


def test_project_handles_short_rows_without_indexerror():
    """Defensive: a malformed/short row must not raise — missing cells become None."""
    cols = ["customer_name", "amount", "notes"]
    rows = [["Rahul", 100]]              # "notes" cell missing entirely
    display_columns = ["customer_name", "notes"]
    out_cols, out_rows = project_display_columns(cols, rows, display_columns)
    assert out_cols == ["customer_name", "notes"]
    assert out_rows == [["Rahul", None]]


# ---------------------------------------------------------------------------
# rows_to_markdown_table
# ---------------------------------------------------------------------------

def test_table_end_to_end_with_mixed_types():
    cols = ["customer_name", "amount", "notes", "last_login"]
    rows = [
        ["Rahul Sharma", Decimal("98400.000"),
         "A very long free-text note that goes on and on and could overflow a "
         "table cell in the UI if left unchecked", None],
        ["Priya|Patel", 5000.5, None, "2026-01-01"],
    ]
    out = rows_to_markdown_table(cols, rows)
    lines = out.splitlines()
    assert lines[0] == "| Customer Name | Amount | Notes | Last Login |"
    assert lines[1] == "| --- | ---: | --- | --- |"   # amount is all-numeric -> right-aligned
    assert "98,400" in lines[2]
    assert lines[2].endswith("| — |")           # trailing None -> explicit em-dash cell
    assert "…" in lines[2]                      # long note capped
    assert "Priya\\|Patel" in lines[3]         # pipe escaped, table stays 4 columns
    # unescaped separators only: 4 cols -> 5 raw "|" delimiters. The customer_name
    # row has no embedded pipe so its count is exact; the Priya row's literal cell
    # pipe is escaped (\|) and deliberately excluded from this count.
    assert lines[2].count("|") == 5


def test_truncation_note_appears_when_over_limit():
    cols = ["n"]
    rows = [[i] for i in range(25)]
    out = rows_to_markdown_table(cols, rows, limit=20)
    assert "Showing 20 of 25 rows" in out


def test_no_truncation_note_when_within_limit():
    cols = ["n"]
    rows = [[i] for i in range(5)]
    out = rows_to_markdown_table(cols, rows, limit=20)
    assert "Showing" not in out


# ---------------------------------------------------------------------------
# Numeric right-alignment
# ---------------------------------------------------------------------------

def test_all_numeric_column_right_aligned():
    cols = ["name", "amount"]
    rows = [["Rahul", 98400], ["Priya", 5000]]
    out = rows_to_markdown_table(cols, rows)
    assert out.splitlines()[1] == "| --- | ---: |"


def test_mixed_column_stays_left_aligned():
    """One non-numeric value anywhere disqualifies the whole column — a column
    holding a stray string among numbers must not be mis-aligned."""
    cols = ["code"]
    rows = [[1], [2], ["N/A"]]
    out = rows_to_markdown_table(cols, rows)
    assert out.splitlines()[1] == "| --- |"


def test_all_null_column_stays_left_aligned():
    cols = ["optional"]
    rows = [[None], [None]]
    out = rows_to_markdown_table(cols, rows)
    assert out.splitlines()[1] == "| --- |"


def test_bool_column_not_right_aligned():
    """Booleans render as yes/no text, not digits — must not be treated numeric."""
    cols = ["active"]
    rows = [[True], [False]]
    out = rows_to_markdown_table(cols, rows)
    assert out.splitlines()[1] == "| --- |"


def test_numeric_column_with_some_nulls_still_right_aligns():
    cols = ["amount"]
    rows = [[100], [None], [200]]
    out = rows_to_markdown_table(cols, rows)
    assert out.splitlines()[1] == "| ---: |"
