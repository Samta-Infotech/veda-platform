from __future__ import annotations
import re, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimplifierResult:
    original_query:   str
    simplified_query: str
    was_simplified:   bool
    value_hints_used: List[str]
    duration_ms:      float
    error:            Optional[str] = None


def _get_value_hints(query: str) -> Dict[str, List[str]]:
    try:
        import ingestion.value_sampler as vs
        if not vs._VALUE_STORE:
            vs.rebuild_value_index_from_db()
        tokens = re.findall(r'\w+', query.lower())
        hints: Dict[str, List[str]] = {}
        for tok in tokens:
            if tok in vs._VALUE_INDEX:
                for col_id in vs._VALUE_INDEX[tok]:
                    sc = vs._VALUE_STORE.get(col_id)
                    if sc:
                        raw = sc.raw_values[:3]
                        key = hints.setdefault(tok, [])
                        info = f"{sc.table_name}.{sc.col_name}"
                        if raw: info += f" (e.g. {', '.join(raw)})"
                        key.append(info)
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i+1]
            if bigram in vs._VALUE_INDEX:
                for col_id in vs._VALUE_INDEX[bigram]:
                    sc = vs._VALUE_STORE.get(col_id)
                    if sc:
                        raw = sc.raw_values[:3]
                        key = hints.setdefault(bigram, [])
                        info = f"{sc.table_name}.{sc.col_name}"
                        if raw: info += f" (e.g. {', '.join(raw)})"
                        key.append(info)
        return hints
    except Exception as e:
        logger.debug("Value hints failed: %s", e)
        return {}


def _build_prompt(query: str, value_hints: Dict[str, List[str]]) -> str:
    lines = []
    for phrase, cols in list(value_hints.items())[:8]:
        for col_info in cols[:2]:
            lines.append(f'  "{phrase}" maps to: {col_info}')
    hints_text = ("\nSchema context (for reference only):\n" + "\n".join(lines)) if lines else ""

    return (
        f"Rewrite the user query as a clearer plain English question.\n"
        f"DO NOT write SQL. Output must be a natural language question only.\n"
        f"{hints_text}\n\n"
        f"User query: {query}\n\n"
        f"Rules:\n"
        f"- Output plain English only, NO SQL, NO code\n"
        f"- Keep same meaning, just clearer\n"
        f"- Use exact values from schema context if helpful\n"
        f"- One sentence only\n"
        f"- Example input: 'permissions related to level 1 queue for IT admin'\n"
        f"- Example output: 'show permission names containing level 1 for role named IT Admin'\n\n"
        f"Plain English rewrite:"
    )


def run_nl_simplifier(query: str, verbose: bool = False) -> SimplifierResult:
    t0 = time.time()
    try:
        from config import NL_SIMPLIFIER_ENABLED
        if not NL_SIMPLIFIER_ENABLED:
            return SimplifierResult(
                original_query=query, simplified_query=query,
                was_simplified=False, value_hints_used=[],
                duration_ms=0.0,
            )
    except ImportError:
        pass
    if len(query.split()) <= 4:
        return SimplifierResult(
            original_query=query, simplified_query=query,
            was_simplified=False, value_hints_used=[],
            duration_ms=0.0,
        )
    value_hints = _get_value_hints(query)
    hints_used  = list(value_hints.keys())
    prompt      = _build_prompt(query, value_hints)
    try:
        from config import SLM_OLLAMA_BASE_URL, SLM_MODEL_NAME
        import urllib.request as _req, json
        payload = {
            "model":  SLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 64, "num_ctx": 512},
        }
        req = _req.Request(
            f"{SLM_OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _req.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        raw        = data.get("response", "").strip()
        simplified = re.sub(
            r'^(rewritten[:\s]*|query[:\s]*)', '', raw,
            flags=re.IGNORECASE
        ).strip().strip('"').strip("'")
        if not simplified or len(simplified) < 3:
            simplified     = query
            was_simplified = False
        else:
            was_simplified = simplified.lower() != query.lower()
        duration_ms = round((time.time() - t0) * 1000, 1)
        if verbose:
            print(f"  [L0] Original:   '{query}'")
            if was_simplified:
                print(f"  [L0] Simplified: '{simplified}'")
            else:
                print(f"  [L0] No change")
            print(f"  [L0] Duration:   {duration_ms}ms | hints: {hints_used}")
        logger.debug("NL simplifier: '%s' -> '%s' (%dms)", query[:60], simplified[:60], duration_ms)
        return SimplifierResult(
            original_query=query, simplified_query=simplified,
            was_simplified=was_simplified, value_hints_used=hints_used,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = round((time.time() - t0) * 1000, 1)
        logger.warning("NL simplifier failed: %s", e)
        return SimplifierResult(
            original_query=query, simplified_query=query,
            was_simplified=False, value_hints_used=hints_used,
            duration_ms=duration_ms, error=str(e),
        )
