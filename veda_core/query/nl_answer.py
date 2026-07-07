# query/nl_answer.py
# VEDA — NL-back answer generation (Step 7: query -> NL)
# Gate: NL_ANSWER_ENABLED
# Turns result rows into a natural-language answer using the local SLM.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Optional

from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    SLM_TEMPERATURE,
    SLM_TIMEOUT_SECS,
    NL_ANSWER_MAX_ROWS,
)


@dataclass
class NLAnswerResult:
    answer:      str
    row_count:   int
    duration_ms: float
    error:       Optional[str] = None


def _fmt_value(v):
    """Human-friendly scalar formatting (thousands separators for ints)."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def _label_from_column(col: str) -> str:
    return str(col).replace("_", " ").strip().lower()


def template_answer(query: str, columns: List[str], rows: List[dict]) -> Optional[str]:
    """Q-7: deterministic NL answer for CANONICAL result shapes — no SLM call.

    Covers the empty set, single scalar (incl. count/aggregate), and single row.
    Returns None for multi-row narrative results, which still go to the SLM. This
    is the same phrasing the SLM would produce for these shapes, computed for free.
    """
    row_count = len(rows)
    if row_count == 0:
        return "No results found."

    if row_count == 1 and columns:
        row = rows[0]
        if len(columns) == 1:
            col = columns[0]
            val = _fmt_value(row.get(col))
            label = _label_from_column(col)
            # count/aggregate shapes: "count", "total", "n", "count(*)" …
            if any(k in label for k in ("count", "total", "number", "num ", "sum", "avg", "min", "max")):
                return f"The {label} is {val}."
            return f"{col}: {val}" if val != "" else "No results found."
        # single row, multiple columns — compact deterministic summary
        parts = [f"{_label_from_column(c)} {_fmt_value(row.get(c))}"
                 for c in columns[:6] if row.get(c) is not None]
        if parts:
            return "Result: " + ", ".join(parts) + "."
    return None


def deterministic_fallback_answer(query: str, columns: List[str], rows: List[dict]) -> str:
    """Row count + first rows summary — no SLM call. Used as the immediate answer
    on the F6 fast-return path, and as the SLM-failure fallback in run_nl_answer."""
    row_count = len(rows)
    if row_count == 0:
        return "No results found."
    first_vals = []
    if rows and columns:
        for c in columns[:3]:
            v = rows[0].get(c)
            if v is not None:
                first_vals.append(f"{c}={v}")
    return f"Returned {row_count} row(s)." + (
        f" First: {', '.join(first_vals)}." if first_vals else ""
    )


def run_nl_answer(
    query:   str,
    columns: List[str],
    rows:    List[dict],
    verbose: bool = False,
    timeout: Optional[float] = None,
) -> NLAnswerResult:
    """
    Converts SQL result rows into a natural-language prose answer using the local SLM.
    Deterministic fallback: row count + first rows summary if SLM unavailable.
    """
    import time
    t0 = time.time()

    row_count = len(rows)

    # Q-7: canonical shapes (empty / single scalar / single row) are answered
    # deterministically — skip the SLM round trip entirely (~1–3s saved). Gated so
    # it can be turned off for parity comparison. Multi-row results still use the SLM.
    try:
        from config import NL_TEMPLATE_ENABLED as _NL_TEMPLATE_ENABLED
    except Exception:
        _NL_TEMPLATE_ENABLED = True
    if _NL_TEMPLATE_ENABLED:
        _tmpl = template_answer(query, columns, rows)
        if _tmpl is not None:
            return NLAnswerResult(answer=_tmpl, row_count=row_count,
                                  duration_ms=round((time.time() - t0) * 1000, 2))

    sample_rows = rows[:NL_ANSWER_MAX_ROWS]

    # Build a compact table representation
    if columns and sample_rows:
        header = " | ".join(str(c)[:20] for c in columns[:6])
        table_lines = [header, "-" * len(header)]
        for row in sample_rows[:10]:
            line = " | ".join(str(row.get(c, ""))[:20] for c in columns[:6])
            table_lines.append(line)
        if row_count > 10:
            table_lines.append(f"... ({row_count} rows total)")
        table_text = "\n".join(table_lines)
    else:
        table_text = f"({row_count} rows, no data)"

    prompt = (
        f"User question: {query}\n\n"
        f"Query result ({row_count} rows):\n{table_text}\n\n"
        f"Write a single concise sentence summarising the result. "
        f"Do not repeat the column names verbatim. No markdown."
    )

    _slm_timeout = min(SLM_TIMEOUT_SECS, 60) if timeout is None else timeout
    try:
        from slm import call_slm
        answer = call_slm(
            prompt,
            purpose="nl_answer",
            temperature=0.1,
            num_predict=128,
            endpoint="generate",
            timeout=_slm_timeout,
        ).strip()
        if not answer:
            raise ValueError("Empty response from SLM")
    except Exception as e:
        answer = deterministic_fallback_answer(query, columns, rows)
        if verbose:
            print(f"  [NLAnswer] SLM unavailable ({e}) — using fallback answer")

    duration_ms = round((time.time() - t0) * 1000, 2)
    return NLAnswerResult(answer=answer, row_count=row_count, duration_ms=duration_ms)
