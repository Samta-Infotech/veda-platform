"""VEDA · L5 — LLM SQL generation (single-table + join-skeleton fill)."""
import os, re, sys, time, json, logging, threading
from config import SLM_MODEL_NAME, SLM_OLLAMA_BASE_URL


def _domain_line() -> str:
    """Optional domain primer for the SQL prompt — DESCRIPTIVE only (forbids inventing
    filters/rules; the IR-equivalence firewall enforces it). Empty when DOMAIN_CONTEXT unset."""
    try:
        from config import DOMAIN_CONTEXT
        if DOMAIN_CONTEXT:
            return (f" Domain (for interpreting terminology ONLY — do NOT add filters, "
                    f"conditions, or business rules the question did not explicitly ask "
                    f"for): {DOMAIN_CONTEXT}.")
    except Exception:
        pass
    return ""


def _column_glossary_block(col_glossary) -> str:
    """Render the in-scope columns' business_definition + aliases as a compact prompt
    block (vocab→column). Capped + truncated for token budget; only columns that HAVE
    metadata appear. Framed interpret-only; the IR-equivalence firewall still backstops."""
    if not col_glossary:
        return ""
    try:
        from config import (SQL_COLUMN_GLOSSARY_ENABLED, SQL_COLUMN_GLOSSARY_MAX_COLS,
                            SQL_COLUMN_GLOSSARY_DEF_LEN)
    except Exception:
        return ""
    if not SQL_COLUMN_GLOSSARY_ENABLED:
        return ""
    lines = []
    for col, m in list(col_glossary.items())[:SQL_COLUMN_GLOSSARY_MAX_COLS]:
        parts = []
        d = (m.get("def") or "").strip()
        if d:
            parts.append(d[:SQL_COLUMN_GLOSSARY_DEF_LEN])
        aliases = m.get("aliases") or []
        if aliases:
            parts.append("aka: " + ", ".join(aliases[:5]))
        if parts:
            lines.append(f"  {col} — " + " | ".join(parts))
    if not lines:
        return ""
    return ("\nColumn meanings (interpret the question's terminology ONLY — do NOT add "
            "filters or conditions not asked for):\n" + "\n".join(lines))


def _term_directive_block(term_map) -> str:
    """Render domain_synonyms-resolved phrase→column directives. When a query phrase maps
    (in the model's domain_synonyms) to a SPECIFIC in-scope column, tell the model to use
    that EXACT column — so "last logged in" → last_logged_in instead of the model guessing
    a sibling like last_login. Authoritative grounding, not a hint; keeps validation strict."""
    if not term_map:
        return ""
    seen, lines = set(), []
    for phrase, col in term_map:
        if (phrase, col) in seen:
            continue
        seen.add((phrase, col))
        lines.append(f'  "{phrase}" → {col}')
    if not lines:
        return ""
    return ("\nTerm → column mapping (the question's term on the LEFT means EXACTLY the "
            "column on the RIGHT — use that column, not a similarly-named one):\n"
            + "\n".join(lines))


def generate_sql(query, table, columns, temporal, col_glossary=None, term_map=None,
                 time_col=None):
    """Ask Qwen for ONE read-only SELECT over the chosen table's real columns."""
    import urllib.request

    date_line = ""
    if temporal and (temporal.start or temporal.end):
        # Name the EXACT temporal column (same authoritative grounding the term-map uses),
        # never "the appropriate datetime column" — that made the model GUESS, and a small
        # model reaches for the convention `created_at` (which then fails validation). The
        # caller passes the table's canonical temporal column; if there is none the caller
        # has already refused, so an empty time_col here means "do not add a date filter".
        if time_col:
            date_line = (f'\nApply the date filter on the "{time_col}" column ONLY '
                         f"(do not use any other column for the date), "
                         f"between '{temporal.start}' and '{temporal.end}'.")

    system = ("You are a PostgreSQL expert. Output ONE read-only SELECT statement "
              "and nothing else — no markdown, no commentary, no semicolon." + _domain_line())
    user = (f"Question: {query}\n"
            f"Table: {table}\n"
            f"Columns: {', '.join(columns)}{date_line}"
            f"{_column_glossary_block(col_glossary)}"
            f"{_term_directive_block(term_map)}\n"
            f"Rules: SELECT only, FROM {table}. Use only listed columns. "
            f"Always end with LIMIT 100.")

    payload = {
        "model": SLM_MODEL_NAME, "stream": False, "keep_alive": "24h",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        # temperature 0 + fixed seed → greedy, reproducible decoding. SQL generation
        # must be DETERMINISTIC: the same question had been returning different WHERE
        # clauses (and different counts) run-to-run at temperature 0.1.
        "options": {"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 2048},
    }
    req = urllib.request.Request(
        f"{SLM_OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        out = json.loads(resp.read().decode())
    sql = out.get("message", {}).get("content", "").strip()
    # strip markdown fences if the model added them
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql[sql.lower().find("select"):] if "select" in sql.lower() else sql
    return sql.strip().rstrip(";").strip()


_OVERRIDES_CACHE = {"v": None}


def _load_overrides():
    """semantic/overrides.json — human-declared facts, highest authority.
    Cached per process; absent file is fine (empty overrides)."""
    if _OVERRIDES_CACHE["v"] is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "semantic", "overrides.json")
        try:
            _OVERRIDES_CACHE["v"] = json.load(open(path)) if os.path.exists(path) else {}
        except Exception:
            _OVERRIDES_CACHE["v"] = {}
    return _OVERRIDES_CACHE["v"]


def _resolve_display_column(table, sm):
    """Pick a table's human-readable LABEL column, so the projection points at a
    REAL column (organizations -> name) instead of the LLM inventing one (org_name).

    Resolution chain (governance): 1) human override file — a person who checked
    the data declared it, always wins (fixes counterparty_details where the
    heuristic picks `title`, an honorific); 2) metadata/heuristic fallback below.
    Returns col_name or None."""
    ov = (_load_overrides().get("display_columns") or {}).get(table)
    if ov:
        return ov
    tcols = [(c.get("col_name", ""), c) for c in sm.get("columns", {}).values()
             if c.get("table_name") == table]
    if not tcols:
        return None
    for pref in ("name", "title", "display_name", "label"):
        if any(cn == pref for cn, _ in tcols):
            return pref
    for cn, _ in tcols:                          # role_name, category_name — strongest label
        if cn.endswith("_name"):
            return cn
    for cn, c in tcols:                          # business_role explicitly a name
        if "name" in (c.get("business_role", "") or "").lower():
            return cn
    for cn, _ in tcols:                          # human key (incident_no)
        if cn.endswith(("_no", "_number")):
            return cn
    for cn, _ in tcols:                          # abbreviation code, weaker
        if cn.endswith("_code"):
            return cn
    for cn, c in tcols:                          # last resort: a real categorical label
        if (c.get("analytics_role") == "DIMENSION"
                and c.get("semantic_type") in ("CATEGORY", "FREE_TEXT")
                and not cn.endswith("_id")):
            return cn
    return None


def _join_glossary_block(alias_map, sm) -> str:
    """Alias-keyed column glossary for the multi-table prompt (t0.col — def | aka: …).
    Capped TOTAL across all joined tables for token budget; only columns with metadata."""
    try:
        from config import (SQL_COLUMN_GLOSSARY_ENABLED, SQL_COLUMN_GLOSSARY_MAX_COLS,
                            SQL_COLUMN_GLOSSARY_DEF_LEN)
    except Exception:
        return ""
    if not SQL_COLUMN_GLOSSARY_ENABLED:
        return ""
    cols = sm.get("columns", {})
    # Fair per-table budget so every joined table gets glossary lines — else the first
    # table's columns consume the whole cap and the joined entity (the answer!) gets none.
    per_table = max(2, SQL_COLUMN_GLOSSARY_MAX_COLS // max(1, len(alias_map)))
    lines = []
    for al, tbl in alias_map.items():
        n = 0
        for key, m in cols.items():
            if not key.startswith(tbl + "."):
                continue
            m = m or {}
            if not (m.get("aliases") or m.get("business_definition")):
                continue
            parts = []
            d = (m.get("business_definition") or "").strip()
            if d:
                parts.append(d[:SQL_COLUMN_GLOSSARY_DEF_LEN])
            a = m.get("aliases") or []
            if a:
                parts.append("aka: " + ", ".join(a[:5]))
            if parts:
                lines.append(f"  {al}.{key.split('.', 1)[1]} — " + " | ".join(parts))
                n += 1
            if n >= per_table:
                break
    if not lines:
        return ""
    return ("\nColumn meanings (interpret terminology ONLY — do NOT add filters or "
            "conditions not asked for):\n" + "\n".join(lines))


def generate_join_sql(query, skeleton, alias_map, sm, tf):
    """LLM fills SELECT/WHERE/GROUP BY on a FIXED, deterministic FROM/JOIN block.
    The join keys + polymorphic predicates are never the LLM's to write, and the
    entity LABEL columns are resolved deterministically (not invented)."""
    import urllib.request
    cols = sm.get("columns", {})
    # alias_map is occurrence-keyed: alias -> table. The same table may appear under
    # two aliases (same-table-twice / self-join) — emit one block per OCCURRENCE.
    blocks = []
    for al, tbl in alias_map.items():
        tcols = [k.split(".", 1)[1] for k in cols if k.startswith(tbl + ".")][:20]
        blocks.append(f"  {al} = {tbl}: {', '.join(tcols)}")
    # Deterministic display-column guidance: tell the LLM the REAL label column per
    # table so it can't hallucinate (org_name). Validation remains the backstop.
    display_lines = []
    for al, tbl in alias_map.items():
        dc = _resolve_display_column(tbl, sm)
        if dc:
            display_lines.append(f"  {al}.{dc}  (label column for {tbl} — use this, do not invent)")
    display_block = ("\nEntity label columns (use these exact columns when projecting an "
                     "entity's name; never invent a column):\n" + "\n".join(display_lines)
                     ) if display_lines else ""
    date_line = ""
    if tf and (tf.start or tf.end):
        date_line = (f"\nIf the question implies a time window, filter a datetime column "
                     f"BETWEEN '{tf.start}' AND '{tf.end}'.")
    system = ("You are a PostgreSQL expert. You are given a FIXED FROM/JOIN block — copy it "
              "VERBATIM, never change tables/joins/ON. Add only SELECT (columns prefixed with "
              "the given aliases), and optional WHERE / GROUP BY / ORDER BY. Add GROUP BY ONLY "
              "when the question explicitly says 'per', 'by', or 'each' — then GROUP BY the "
              "anchor (t0) key and aggregate child rows. Otherwise do NOT add GROUP BY, COUNT, "
              "or DISTINCT: when the question asks to give/show/list attributes (even alongside "
              "'how many'), SELECT exactly those columns as a plain listing — the row count "
              "answers any 'how many'. Output ONLY SQL, no markdown, no semicolon."
              + _domain_line())
    # domain_synonyms-driven phrase→column directives (same as the single-table path):
    # a query phrase that the model maps to a SPECIFIC joined-table column ("last logged
    # in" → last_logged_in) is rendered as alias.column so the LLM uses the exact column
    # instead of a sibling (last_login). Word-boundary, len≥4 (avoid 'in'/'log' noise).
    _tbl_alias = {}
    for _al, _tbl in alias_map.items():
        _tbl_alias.setdefault(_tbl, _al)
    _join_term_map, _ql = [], query.lower()
    for _phrase, _cks in (sm.get("domain_synonyms", {}) or {}).items():
        if len(_phrase) < 4 or not re.search(rf"\b{re.escape(_phrase.lower())}\b", _ql):
            continue
        for _ck in (_cks or []):
            _t, _, _c = _ck.partition(".")
            if _t in _tbl_alias:
                _join_term_map.append((_phrase, f"{_tbl_alias[_t]}.{_c}"))
    user = (f"Question: {query}\n\nFIXED FROM/JOIN (use exactly):\n{skeleton}\n\n"
            f"Aliases → table: columns:\n" + "\n".join(blocks) + display_block
            + _join_glossary_block(alias_map, sm) + _term_directive_block(_join_term_map)
            + date_line +
            "\nRules: prefix every column with its alias; SELECT only; end with LIMIT 100.")
    payload = {"model": SLM_MODEL_NAME, "stream": False, "keep_alive": "24h",
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "options": {"temperature": 0, "seed": 0, "num_predict": 320, "num_ctx": 3072}}
    req = urllib.request.Request(f"{SLM_OLLAMA_BASE_URL}/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    sql = out.get("message", {}).get("content", "").strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql[sql.lower().find("select"):] if "select" in sql.lower() else sql
    return sql.strip().rstrip(";").strip()
