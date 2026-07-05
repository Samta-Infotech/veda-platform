"""
ingestion/domain_glossary.py
Builds and caches a domain-specific glossary for VEDA.

Three layers:
  Layer A: SLM-generated — schema columns → business synonyms (via Ollama)
  Layer B: HuggingFace datasets — BFSI/AML domain vocabulary
  Layer C: Static AML/KYC patterns — hardcoded regulatory terms

Output: glossary/domain_glossary.json
  {
    "term": ["col_name1", "col_name2", ...],
    ...
  }

One-time build — subsequent calls load from cache.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional

GLOSSARY_DIR  = "glossary"
COMBINED_PATH = os.path.join(GLOSSARY_DIR, "domain_glossary.json")
SLM_PATH      = os.path.join(GLOSSARY_DIR, "slm_glossary.json")
HF_PATH       = os.path.join(GLOSSARY_DIR, "hf_glossary.json")
STATIC_PATH   = os.path.join(GLOSSARY_DIR, "static_glossary.json")


# ── Layer C: Static AML/KYC/BFSI terms ────────────────────────────────────
STATIC_AML_GLOSSARY: Dict[str, List[str]] = {
    # AML core terms
    "sar":                  ["suspicious_activity", "report_type", "filing_date", "sar"],
    "ctr":                  ["cash_transaction", "currency_transaction", "threshold"],
    "pep":                  ["politically_exposed", "pep_flag", "is_pep", "pep_status"],
    "kyc":                  ["kyc_status", "kyc_date", "verification_status", "due_diligence"],
    "aml":                  ["aml_flag", "aml_score", "risk_rating", "aml_status"],
    "cdd":                  ["due_diligence", "cdd_status", "customer_risk", "cdd_date"],
    "edd":                  ["enhanced_due_diligence", "edd_status", "edd_date"],
    "ubo":                  ["beneficial_owner", "ownership_percentage", "ubo_name"],
    "sanction":             ["is_sanctioned", "sanction_list", "ofac_status", "sanctioned"],
    "ofac":                 ["ofac_status", "ofac_flag", "sanction_list"],
    "fatf":                 ["risk_rating", "jurisdiction_risk", "country_risk"],
    "risk score":           ["risk_score", "risk_rating", "fraud_score", "risk_level"],
    "risk rating":          ["risk_rating", "risk_level", "risk_score"],
    "counterparty":         ["counterparty_name", "counterparty_id", "beneficiary", "payee"],
    "beneficiary":          ["beneficiary_name", "beneficiary_id", "counterparty"],
    "transaction":          ["txn_amount", "amount", "debit_amount", "credit_amount", "transaction_id"],
    "wire transfer":        ["wire_amount", "transfer_amount", "txn_type", "transaction_type"],
    "remittance":           ["remittance_amount", "transfer_amount", "txn_type"],
    "flagged":              ["is_flagged", "flag_reason", "alert_type", "workflow_state"],
    "suspicious":           ["risk_score", "alert_type", "workflow_state", "is_suspicious"],
    "investigation":        ["incident", "case_type", "investigation_date", "case_status"],
    "alert":                ["incident", "alert_type", "alert_status", "workflow_state"],
    "case":                 ["incident", "case_id", "case_status", "case_type"],
    "escalated":            ["workflow_state", "escalation_status", "is_escalated"],
    "open":                 ["workflow_state", "status", "is_open", "is_active"],
    "closed":               ["workflow_state", "status", "closed_date", "resolution"],
    "queue":                ["workflow_state", "queue_name", "assigned_queue"],
    "sla":                  ["sla_hours", "sla_breach", "due_date", "sla_status"],
    "threshold":            ["threshold_amount", "limit_value", "max_amount"],
    "structuring":          ["txn_amount", "transaction_type", "structuring_flag"],
    "layering":             ["transaction_type", "transfer_count", "layering_flag"],
    "placement":            ["deposit_amount", "cash_deposit", "placement_flag"],
    # BFSI general
    "npa":                  ["is_npa", "npa_date", "npa_amount", "asset_classification"],
    "emi":                  ["emi_amount", "installment_amount", "monthly_payment"],
    "ltv":                  ["loan_to_value", "ltv_ratio", "collateral_value"],
    "cibil":                ["credit_score", "cibil_score", "bureau_score"],
    "collateral":           ["collateral_type", "collateral_value", "security_type"],
    "disbursement":         ["disbursement_date", "disbursed_amount", "release_date"],
    "delinquent":           ["is_delinquent", "days_past_due", "dpd", "overdue_amount"],
    "write off":            ["write_off_date", "write_off_amount", "is_written_off"],
    "recovery":             ["recovery_amount", "recovery_date", "recovery_status"],
    # Real estate
    "bhk":                  ["bedroom_count", "bhk_type", "unit_type", "configuration"],
    "carpet area":          ["carpet_area_sqft", "area_sqft", "built_up_area"],
    "possession":           ["possession_date", "handover_date", "completion_date"],
    "rera":                 ["rera_number", "rera_status", "project_approval"],
    "facing":               ["facing_direction", "facing", "property_facing"],
    "occupancy":            ["occupancy_rate", "is_occupied", "tenant_status"],
    "stamp duty":           ["stamp_duty", "registration_charges", "stamp_value"],
}


# ── Layer B: HuggingFace dataset extraction ────────────────────────────────

def _build_hf_glossary() -> Dict[str, List[str]]:
    """Download and process HuggingFace datasets to extract BFSI vocabulary."""
    glossary: Dict[str, List[str]] = {}

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Glossary] datasets library not installed — pip install datasets")
        print("[Glossary] Skipping HF glossary layer")
        return glossary

    # banking77: 77 banking intent labels
    try:
        print("[Glossary] Downloading banking77...")
        ds = load_dataset("PolyAI/banking77", split="train")
        label_names = ds.features["label"].names
        for label in label_names:
            parts = [p for p in label.split("_") if len(p) > 3]
            term  = " ".join(parts).lower()
            if term and len(term) > 4:
                glossary[term] = parts
        print(f"[Glossary] banking77: {len(label_names)} intents extracted")
    except Exception as e:
        print(f"[Glossary] banking77 failed: {e}")

    # financial_phrasebank: extract financial nouns
    try:
        print("[Glossary] Downloading financial_phrasebank...")
        import re
        ds2 = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
        finance_nouns = set()
        FINANCE_PATTERNS = [
            r'\b(revenue|profit|loss|earnings|dividend|equity|debt|asset|liability)\b',
            r'\b(transaction|payment|transfer|deposit|withdrawal|balance|account)\b',
            r'\b(fraud|risk|compliance|regulatory|audit|investigation|alert)\b',
            r'\b(customer|client|counterparty|beneficiary|holder|owner)\b',
        ]
        for row in ds2:
            text = row.get("sentence", "").lower()
            for pattern in FINANCE_PATTERNS:
                matches = re.findall(pattern, text)
                finance_nouns.update(matches)
        for noun in finance_nouns:
            if noun not in glossary:
                glossary[noun] = [noun, f"{noun}_id", f"{noun}_status", f"{noun}_type"]
        print(f"[Glossary] financial_phrasebank: {len(finance_nouns)} terms extracted")
    except Exception as e:
        print(f"[Glossary] financial_phrasebank failed: {e}")

    # finance-alpaca: extract AML/compliance terms
    try:
        print("[Glossary] Downloading finance-alpaca...")
        ds3 = load_dataset("gbharti/finance-alpaca", split="train")
        alpaca_terms = set()
        AML_KEYWORDS = {
            "money laundering", "suspicious", "fraud", "compliance",
            "aml", "kyc", "transaction monitoring", "risk assessment",
            "due diligence", "sanctions", "pep", "beneficial owner",
        }
        count = 0
        for row in ds3:
            text = (row.get("instruction", "") + " " + row.get("input", "")).lower()
            for kw in AML_KEYWORDS:
                if kw in text:
                    alpaca_terms.add(kw)
                    count += 1
                    break
            if count > 500:
                break
        for term in alpaca_terms:
            if term not in glossary:
                parts = term.split()
                glossary[term] = parts + [p for p in parts if len(p) > 3]
        print(f"[Glossary] finance-alpaca: {len(alpaca_terms)} AML terms extracted")
    except Exception as e:
        print(f"[Glossary] finance-alpaca failed: {e}")

    return glossary


# ── Layer A: SLM-generated glossary ───────────────────────────────────────

_SLM_GLOSSARY_PROMPT = """\
You are an AML/KYC domain expert analyzing a compliance database schema.

Given this column from an AML compliance system:
  Table: {table_name}
  Column: {col_name}
  Semantic type: {semantic_type}
  Sample values: {sample_values}

Generate a JSON object (no markdown, no explanation):
{{"synonyms": ["term1", "term2", "term3", "term4", "term5"], "query_patterns": ["pattern1", "pattern2", "pattern3"]}}

synonyms: 5 natural language terms a compliance analyst would use
query_patterns: 3 query fragments that would require this column
Keep all terms lowercase. No UUIDs. No column names — only business vocabulary.\
"""


def _call_slm_for_synonyms(
    table_name:    str,
    col_name:      str,
    semantic_type: str,
    sample_values: List[str],
    ollama_url:    str,
) -> Optional[Dict]:
    """Call Ollama to generate synonyms for a column."""
    import re as _re

    prompt = _SLM_GLOSSARY_PROMPT.format(
        table_name    = table_name,
        col_name      = col_name,
        semantic_type = semantic_type,
        sample_values = sample_values[:5] if sample_values else [],
    )
    payload = json.dumps({
        "model":    "qwen2.5-coder:7b",
        "stream":   False,
        "messages": [{"role": "user", "content": prompt}],
        "options":  {"temperature": 0.3, "num_predict": 200},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body    = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            match   = _re.search(r'\{.*\}', content, _re.DOTALL)
            if match:
                return json.loads(match.group())
    except Exception:
        pass
    return None


def _build_slm_glossary(
    inference_result,
    ollama_url: str,
    max_cols:   int = 50,
) -> Dict[str, List[str]]:
    """Generate column synonyms via Ollama. Processes CATEGORY/IDENTIFIER/FREE_TEXT columns."""
    glossary: Dict[str, List[str]] = {}

    eligible = [
        tc for tc in inference_result.typed_columns
        if tc.semantic_type in ("CATEGORY", "IDENTIFIER", "FREE_TEXT")
        and len(tc.col_name) > 3
    ][:max_cols]

    print(f"[Glossary] SLM generating synonyms for {len(eligible)} columns...")

    for i, tc in enumerate(eligible):
        sample_vals = getattr(tc, "sample_values", []) or []
        result = _call_slm_for_synonyms(
            table_name    = tc.table_name,
            col_name      = tc.col_name,
            semantic_type = tc.semantic_type,
            sample_values = [str(v) for v in sample_vals[:5]],
            ollama_url    = ollama_url,
        )
        if result:
            for term in result.get("synonyms", []) + result.get("query_patterns", []):
                term_lower = term.lower().strip()
                if len(term_lower) > 3:
                    if term_lower not in glossary:
                        glossary[term_lower] = []
                    if tc.col_name not in glossary[term_lower]:
                        glossary[term_lower].append(tc.col_name)

        if (i + 1) % 10 == 0:
            print(f"[Glossary] SLM progress: {i+1}/{len(eligible)}")

    print(f"[Glossary] SLM generated {len(glossary)} term mappings")
    return glossary


# ── Public API ─────────────────────────────────────────────────────────────

def build_glossary(
    inference_result=None,
    ollama_url:    str  = "http://localhost:11434",
    force_rebuild: bool = False,
) -> Dict[str, List[str]]:
    """
    Build and cache the domain glossary (one-time operation).
    Returns combined glossary: {term: [col_name, ...]}
    """
    os.makedirs(GLOSSARY_DIR, exist_ok=True)

    if not force_rebuild and os.path.exists(COMBINED_PATH):
        print(f"[Glossary] Loading from cache: {COMBINED_PATH}")
        with open(COMBINED_PATH) as f:
            return json.load(f)

    print("[Glossary] Building domain glossary (one-time operation)...")
    combined: Dict[str, List[str]] = {}

    # Layer C: Static
    print(f"[Glossary] Layer C: {len(STATIC_AML_GLOSSARY)} static AML/KYC terms")
    for term, cols in STATIC_AML_GLOSSARY.items():
        combined[term] = list(cols)
    with open(STATIC_PATH, "w") as f:
        json.dump(STATIC_AML_GLOSSARY, f, indent=2)

    # Layer B: HuggingFace
    if not os.path.exists(HF_PATH):
        print("[Glossary] Layer B: Building HF glossary...")
        hf_glossary = _build_hf_glossary()
        with open(HF_PATH, "w") as f:
            json.dump(hf_glossary, f, indent=2)
    else:
        print("[Glossary] Layer B: Loading HF glossary from cache")
        with open(HF_PATH) as f:
            hf_glossary = json.load(f)

    for term, cols in hf_glossary.items():
        if term not in combined:
            combined[term] = cols
        else:
            for col in cols:
                if col not in combined[term]:
                    combined[term].append(col)

    # Layer A: SLM-generated
    if inference_result is not None:
        if not os.path.exists(SLM_PATH):
            print("[Glossary] Layer A: Generating SLM synonyms via Ollama...")
            slm_glossary = _build_slm_glossary(inference_result, ollama_url)
            with open(SLM_PATH, "w") as f:
                json.dump(slm_glossary, f, indent=2)
        else:
            print("[Glossary] Layer A: Loading SLM glossary from cache")
            with open(SLM_PATH) as f:
                slm_glossary = json.load(f)

        for term, cols in slm_glossary.items():
            if term not in combined:
                combined[term] = cols
            else:
                for col in cols:
                    if col not in combined[term]:
                        combined[term].append(col)
    else:
        print("[Glossary] Layer A: Skipped (no inference_result provided)")

    with open(COMBINED_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[Glossary] Built: {len(combined)} total terms → {COMBINED_PATH}")
    return combined


def load_glossary() -> Dict[str, List[str]]:
    """Load glossary from cache. Returns empty dict if not built yet."""
    if not os.path.exists(COMBINED_PATH):
        return {}
    with open(COMBINED_PATH) as f:
        return json.load(f)


def expand_query_with_glossary(
    query_tokens: List[str],
    glossary:     Dict[str, List[str]],
) -> List[str]:
    """
    Expand query tokens using glossary.
    Returns additional col_name tokens to inject into search.
    """
    extra: List[str] = []
    query_lower = " ".join(query_tokens).lower()

    for term, col_names in glossary.items():
        if term in query_lower:
            for col in col_names:
                col_parts = col.replace("_", " ")
                if col_parts not in extra and col not in extra:
                    extra.append(col_parts)
                    extra.append(col)

    return list(dict.fromkeys(extra))
