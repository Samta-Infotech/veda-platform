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


def run_nl_answer(
    query:   str,
    columns: List[str],
    rows:    List[dict],
    verbose: bool = False,
) -> NLAnswerResult:
    """
    Converts SQL result rows into a natural-language prose answer using the local SLM.
    Deterministic fallback: row count + first rows summary if SLM unavailable.
    """
    import time
    t0 = time.time()

    row_count = len(rows)
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

    payload = json.dumps({
        "model":  SLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 128},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{SLM_OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=min(SLM_TIMEOUT_SECS, 60)) as resp:
            data = json.loads(resp.read())
        answer = data.get("response", "").strip()
        if not answer:
            raise ValueError("Empty response from SLM")
    except Exception as e:
        # Deterministic fallback
        if row_count == 0:
            answer = "No results found."
        else:
            first_vals = []
            if rows and columns:
                for c in columns[:3]:
                    v = rows[0].get(c)
                    if v is not None:
                        first_vals.append(f"{c}={v}")
            answer = f"Returned {row_count} row(s)." + (
                f" First: {', '.join(first_vals)}." if first_vals else ""
            )
        if verbose:
            print(f"  [NLAnswer] SLM unavailable ({e}) — using fallback answer")

    duration_ms = round((time.time() - t0) * 1000, 2)
    return NLAnswerResult(answer=answer, row_count=row_count, duration_ms=duration_ms)
