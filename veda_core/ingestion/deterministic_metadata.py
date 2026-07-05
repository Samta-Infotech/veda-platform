# =============================================================================
# ingestion/deterministic_metadata.py
# VEDA Semantic Layer v2 — deterministic (rule-based) metadata pass.
#
# Computes everything that is reproducible from schema + profile + naming rules,
# so the LLM never owns (and cannot corrupt) the structural contract that SQL
# generation depends on. The LLM pass only fills business-meaning fields.
#
#   semantic_type  : reused from semantic_type_inference (data_type + name)
#   analytics_role : rule from semantic_type + pk/fk + name
#   sql_usage      : pure function of analytics_role
#   importance_class: HIGH | MEDIUM | LOW  (ranking decides the number later)
#   base_aliases   : name split + acronym expansion + singular/plural
#   value_handling : keep | stats | pattern | remove   (leakage-safe)
# =============================================================================

import re
from typing import Dict, List, Optional, Tuple

try:
    from ingestion.column_text import _split_identifier, _expand_acronyms
except Exception:  # pragma: no cover - fallback if helpers move
    def _split_identifier(name: str) -> str:
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name.replace("_", " ")).lower().strip()

    def _expand_acronyms(g: str) -> str:
        return g

try:
    from retrieval.query_enrichment import _singularize
except Exception:  # pragma: no cover
    def _singularize(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _pluralize(w: str) -> str:
    if not w or w.endswith("s"):
        return w
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"      # category → categories
    return w + "s"


# --- audit / system column detection (these get importance LOW) ----------------
_AUDIT_RE = re.compile(
    r"^(created|updated|modified|deleted|changed)_(by|at|on|date|datetime|time)$"
    r"|^(created|updated|modified|deleted)_by_id$"
    r"|(_by_id)$|^row_version$|^version$|^etl_|_etl$|^dw_|^ingest",
    re.I,
)
# --- PII / sensitive name hints (raw values must NEVER flow on) ----------------
_PII_HINTS = (
    "email", "e_mail", "mail", "phone", "mobile", "fax", "ssn", "sin",
    "passport", "national_id", "tax_id", "pan", "aadhaar", "iban", "swift",
    "account_no", "account_number", "acct", "card", "cvv", "password", "secret",
    "token", "first_name", "last_name", "full_name", "middle_name", "surname",
    "address", "addr", "street", "zip", "postal", "dob", "birth", "gender",
    "latitude", "longitude", "ip_address", "username", "user_name",
)

# --- value-pattern regexes (detected from sample values) -----------------------
# Order matters: more-specific formats first; DATETIME before PHONE so dates
# like "2026-05-01" aren't mistaken for phone numbers.
_PATTERNS = [
    ("EMAIL", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
    ("URL", re.compile(r"^https?://", re.I)),
    ("UUID", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("IP", re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")),
    ("DATETIME", re.compile(r"^\d{4}-\d{2}-\d{2}([ T]|$)")),
    ("PHONE", re.compile(r"^\+?\(?\d[\d\-\s()]{6,18}$")),
]

_SQL_USAGE = {
    "IDENTIFIER":     {"groupable": True,  "filterable": True,  "sortable": False, "join_key": True},
    "MEASURE":        {"groupable": False, "filterable": True,  "sortable": True,  "join_key": False},
    "DIMENSION":      {"groupable": True,  "filterable": True,  "sortable": True,  "join_key": False},
    "TIME_DIMENSION": {"groupable": True,  "filterable": True,  "sortable": True,  "join_key": False},
    "ATTRIBUTE":      {"groupable": False, "filterable": True,  "sortable": False, "join_key": False},
}


def compute_analytics_role(col_name: str, semantic_type: str,
                           is_pk: bool = False, is_fk: bool = False) -> str:
    name = col_name.lower()
    st = (semantic_type or "").upper()
    if is_pk or is_fk or name.endswith("_id") or name == "id":
        return "IDENTIFIER"
    if st in ("TEMPORAL",):
        return "TIME_DIMENSION"
    if st in ("MONETARY", "METRIC"):
        return "MEASURE"
    if st in ("CATEGORICAL", "CATEGORY", "FLAG"):
        return "DIMENSION"
    return "ATTRIBUTE"


def compute_sql_usage(analytics_role: str) -> Dict[str, bool]:
    return dict(_SQL_USAGE.get(analytics_role, _SQL_USAGE["ATTRIBUTE"]))


def compute_importance_class(col_name: str, analytics_role: str) -> str:
    """HIGH = core business; MEDIUM = useful dimension/date; LOW = audit/surrogate id.

    The metadata only stores the CLASS — retrieval maps class→weight at query
    time (config.IMPORTANCE_WEIGHTS), so ranking is retunable with no re-ingest.
    """
    name = col_name.lower()
    if _AUDIT_RE.search(name):
        return "LOW"
    if name == "id" or (name.endswith("_id") and analytics_role == "IDENTIFIER"):
        return "LOW"
    if analytics_role == "MEASURE":
        return "HIGH"
    if any(k in name for k in ("status", "state", "name", "title", "label",
                               "category", "type", "amount", "total")):
        return "HIGH"
    if analytics_role in ("DIMENSION", "TIME_DIMENSION"):
        return "MEDIUM"
    return "MEDIUM"


# Single-token aliases that are too generic to identify a business concept —
# they'd match many unrelated columns and wreck retrieval precision. Multi-word
# aliases containing these are fine ("annotation id"); the bare token is not.
_GENERIC_ALIASES = {
    "id", "ids", "identifier", "identifiers", "key", "keys", "code", "codes",
    "name", "names", "type", "types", "kind", "date", "datetime", "time",
    "timestamp", "value", "values", "number", "no", "num", "status", "state",
    "flag", "data", "info", "detail", "details", "record", "records",
    "object", "objects", "item", "items", "field", "entry", "entries", "count",
}


def base_aliases(col_name: str, table_name: str = "") -> List[str]:
    """Name-derived aliases, generic single tokens removed for precision.

    Keeps specific multi-word phrases ("annotation id") and the full column
    phrase; drops bare generic tokens ("id", "type", "name", "date").
    """
    out = set()
    gloss = _expand_acronyms(_split_identifier(col_name))
    words = [w for w in gloss.split() if len(w) > 1]
    # No pluralization here — the query side singularizes tokens at retrieval
    # time, so plural aliases are redundant and only produce garbage
    # ("created" → "createds", "updation" → "updations").
    if words:
        out.add(" ".join(words))                       # full phrase, always kept
        if len(words) >= 2:                            # qualified single words only
            for w in words:
                if w not in _GENERIC_ALIASES:
                    out.add(w)
                    out.add(_singularize(w))
        else:                                          # single-word column: keep if specific
            w = words[0]
            if w not in _GENERIC_ALIASES:
                out.update({w, _singularize(w)})
    # drop any bare generic token that slipped in
    return sorted(a for a in out if a and len(a) > 1 and a not in _GENERIC_ALIASES)


def detect_value_pattern(samples: List) -> Optional[str]:
    """Return a pattern label (EMAIL/UUID/PHONE/…) if sample values match one."""
    vals = [str(v).strip() for v in (samples or []) if str(v).strip()]
    if not vals:
        return None
    for label, rx in _PATTERNS:
        if sum(1 for v in vals if rx.match(v)) >= max(1, len(vals) // 2):
            return label
    return None


def classify_value_handling(col_name: str, semantic_type: str, analytics_role: str,
                            samples: Optional[List] = None,
                            distinct_count: Optional[int] = None) -> Tuple[str, dict]:
    """Decide what (if anything) about the column's VALUES may flow to the LLM/embedding.

    Returns (mode, payload):
      'keep'    → safe low-cardinality enum/category: payload {"values": [...]}
      'stats'   → numeric/measure: payload {} (caller adds min/max/avg/null%)
      'pattern' → identifier/uuid: payload {"value_pattern": "<LABEL>"}
      'remove'  → PII/sensitive: payload {"value_pattern": "<LABEL or REDACTED>"}
    Raw values are emitted ONLY for 'keep'.
    """
    name = col_name.lower()
    st = (semantic_type or "").upper()
    pattern = detect_value_pattern(samples)

    # person-name columns (editor_name, customer_name, owner_name…) are PII even
    # though bare "name" (category_name, file_name) is not.
    is_person_name = bool(re.search(
        r"(first|last|full|middle|user|editor|owner|author|creator|customer|client|"
        r"employee|manager|contact|person|assignee|reviewer|approver|maker|checker)_name$",
        name)) or name in ("name", "username", "full_name", "fullname")

    # 1) PII by name hint, person-name, or detected sensitive pattern → never emit raw
    if (any(h in name for h in _PII_HINTS) or is_person_name
            or pattern in ("EMAIL", "PHONE", "IP", "URL")):
        return "remove", {"value_pattern": pattern or ("PERSON_NAME" if is_person_name else "REDACTED")}

    # 2) identifiers → pattern only (UUID/code), never the literal keys
    if analytics_role == "IDENTIFIER" or pattern == "UUID":
        return "pattern", {"value_pattern": pattern or "IDENTIFIER"}

    # 3) safe low-cardinality enum / status / category → keep the values (gold signal)
    if st in ("CATEGORICAL", "CATEGORY", "FLAG"):
        if distinct_count is None or distinct_count <= 25:
            vals = [str(v) for v in (samples or [])][:8]
            return "keep", {"values": vals}
        return "stats", {}

    # 4) everything else (measures, free text) → statistics only
    return "stats", {}


def compute_deterministic(col_name: str, data_type: str, semantic_type: str,
                          is_pk: bool = False, is_fk: bool = False,
                          table_name: str = "", samples: Optional[List] = None,
                          distinct_count: Optional[int] = None) -> dict:
    """One-call deterministic pass for a single column. semantic_type is provided
    by the existing semantic_type_inference (also deterministic)."""
    role = compute_analytics_role(col_name, semantic_type, is_pk, is_fk)
    mode, payload = classify_value_handling(col_name, semantic_type, role,
                                            samples, distinct_count)
    return {
        "analytics_role": role,
        "sql_usage": compute_sql_usage(role),
        "importance_class": compute_importance_class(col_name, role),
        "base_aliases": base_aliases(col_name, table_name),
        "value_handling": mode,
        "value_info": payload,
    }
