"""apps.chat.table_rendering — markdown-table rendering for the chat "content"
block's tabular sub-block (2026-07-17). Pulled out of services.py (which pulls
in a heavy import chain: chatbot.run -> langgraph checkpointer -> redis) so this
pure formatting logic is independently unit-testable, same reasoning as
apps/chat/visualization.py's own separation."""
from __future__ import annotations

from decimal import Decimal

_TABLE_CELL_MAX_CHARS = 80   # long free-text cells overflow/break the rendered table


_NULL_CELL = "—"   # em dash: explicit "no data" — a blank cell reads as a rendering glitch


def fmt_cell(v) -> str:
    """Markdown-table-safe, human-readable cell formatting:
    - None/empty -> "—" (an explicit "no data" marker, not a blank cell that
      reads as a glitch, and never the literal string "None")
    - Decimal (psycopg2's NUMERIC/SUM/AVG type) -> float, formatted with commas
    - int/float -> thousands separator (matches query/result_explainer.py's
      _fmt_value on the summary side, so a number reads the same in the table
      as in the prose above it)
    - a literal "|" would otherwise split into extra (broken) table columns —
      escaped to "\\|" per standard markdown table escaping
    - long free-text is capped (…) so one cell can't blow out the table width

    Deliberately does NOT prefix a currency symbol: the platform is
    multi-source/multi-tenant with currency itself a data column (see the
    semantic model's own "Currency" entity/currency_code column) — inventing a
    symbol here would risk showing the WRONG currency for a tenant/source whose
    data isn't in it, and would contradict query/result_explainer.py's own rule
    for the summary prose ("never introduce a currency/unit not in the data").
    """
    if v is None or v == "":
        return _NULL_CELL
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, int):
        s = f"{v:,}"
    elif isinstance(v, float):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
    else:
        s = str(v)
    s = s.replace("|", "\\|")
    if len(s) > _TABLE_CELL_MAX_CHARS:
        s = s[:_TABLE_CELL_MAX_CHARS - 1].rstrip() + "…"
    return s


def fmt_header(c) -> str:
    """Column names read as business labels, not raw identifiers — same
    underscore->space humanization query/result_explainer.py's
    _label_from_column applies to summary prose, so header and summary agree."""
    return str(c).replace("_", " ").strip().title()


def project_display_columns(cols: list, rows: list, display_columns: list | None) -> tuple[list, list]:
    """Drop non-business-relevant columns (2026-07-17) — e.g. internal ids/FK
    columns SQL needed for a join but that add no value in the rendered table —
    using the engine's OWN deterministic column-role classification
    (veda/result_analyzer.py's analytics_summary()["display_columns"], already
    computed server-side and excludes identifier-role columns; the same signal
    apps/chat/visualization.py's recommender already trusts for chart axes).
    Never re-derives roles here — purely a projection over an existing signal.

    `rows` must be POSITIONAL (index-aligned with `cols`) — see
    apps/chat/services.py's _positional_rows, called before this.

    Fails safe: if `display_columns` is absent, or filtering would drop every
    column (a name mismatch between the two lists), the original cols/rows are
    returned unchanged rather than ever rendering an empty table.
    """
    if not display_columns:
        return cols, rows
    keep = set(display_columns)
    idx = [i for i, c in enumerate(cols) if c in keep]
    if not idx or len(idx) == len(cols):
        return cols, rows
    filtered_cols = [cols[i] for i in idx]
    filtered_rows = [[row[i] if i < len(row) else None for i in idx] for row in rows]
    return filtered_cols, filtered_rows


def _is_numeric_column(rows: list, col_idx: int) -> bool:
    """A column right-aligns when EVERY non-null cell (across the rows actually
    being rendered) is a real number — booleans excluded (they render as
    yes/no text, not digits). An all-null column stays left-aligned (nothing
    to judge numeric-ness from); one non-numeric value anywhere disqualifies
    the whole column, so a mixed/id-like column never gets mis-aligned."""
    saw_value = False
    for row in rows:
        if col_idx >= len(row):
            continue
        v = row[col_idx]
        if v is None or v == "":
            continue
        saw_value = True
        if isinstance(v, bool) or not isinstance(v, (int, float, Decimal)):
            return False
    return saw_value


def rows_to_markdown_table(cols: list, rows: list, limit: int = 20) -> str:
    shown = rows[:limit]
    # Right-align numeric columns (markdown's ":---:"/"---:" alignment syntax) so
    # figures compare the way a spreadsheet reads them, instead of every column
    # ragged-left by default.
    aligns = [_is_numeric_column(shown, i) for i in range(len(cols))]
    header = "| " + " | ".join(fmt_header(c) for c in cols) + " |"
    sep = "| " + " | ".join("---:" if numeric else "---" for numeric in aligns) + " |"
    body_lines = [
        "| " + " | ".join(fmt_cell(v) for v in row[:len(cols)]) + " |" for row in shown
    ]
    lines = [header, sep, *body_lines]
    if len(rows) > limit:
        # Silent truncation previously gave no signal that more rows existed.
        lines.append(f"\n_Showing {limit} of {len(rows)} rows._")
    return "\n".join(lines)
