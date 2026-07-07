# =============================================================================
# ingestion/glossary_builder.py
# VEDA Final Architecture — Stage 2: Domain Glossary Generation
#
# Purpose:
#   Call Qwen once to build business glossary mapping domain terms to definitions.
#   Glossary is cached and reused across ingestion runs.
#
# Output:
#   veda_glossary.json = {
#     "DEBIT": "Money going out, expense, cost, outflow",
#     "CREDIT": "Money coming in, income, received",
#     ...
#   }
# =============================================================================

import sys
import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    SLM_MODEL_NAME,
    SLM_OLLAMA_BASE_URL,
    GLOSSARY_GENERATION_ENABLED,
    GLOSSARY_TEMPERATURE,
    GLOSSARY_TIMEOUT,
    GLOSSARY_FILE,
    GLOSSARY_DOMAIN_DESCRIPTION,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _call_ollama(prompt: str, model: str = None, temperature: float = 0.5, timeout: int = 120) -> Optional[str]:
    """Call the SLM via the §10 seam. Returns response text or None on failure."""
    try:
        from slm import call_slm
        return call_slm(
            prompt, purpose="glossary", model=model,
            temperature=temperature, endpoint="generate", timeout=timeout,
        ).strip()
    except Exception as e:
        logger.error(f"SLM call failed: {e}")
        return None


def build_glossary_prompt(domain_description: str, table_names: List[str]) -> str:
    """Build prompt for Qwen to generate domain glossary."""
    tables_str = ", ".join(table_names[:10])
    if len(table_names) > 10:
        tables_str += f", ... and {len(table_names) - 10} more"

    prompt = f"""You are a business glossary builder.
Given a database schema and business domain, generate a JSON glossary mapping domain terms to their definitions.

Domain: {domain_description}
Database Schema Tables: {tables_str}

Generate a comprehensive glossary for common business terms in this domain.
Include synonyms and related concepts.
Format: {{"TERM": "definition with synonyms", ...}}

Output ONLY valid JSON, no explanations.
"""
    return prompt


def generate_glossary(
    domain_description: str = None,
    table_names: List[str] = None,
    model: str = None,
    temperature: float = None,
) -> Optional[Dict[str, str]]:
    """Call Qwen to generate domain glossary."""
    if not GLOSSARY_GENERATION_ENABLED:
        logger.info("Glossary generation disabled")
        return {}

    if domain_description is None:
        domain_description = GLOSSARY_DOMAIN_DESCRIPTION

    if table_names is None:
        table_names = ["payment_transaction", "user", "transaction_status"]

    if model is None:
        model = SLM_MODEL_NAME

    if temperature is None:
        temperature = GLOSSARY_TEMPERATURE

    logger.info("Generating domain glossary from Qwen...")
    start_time = time.time()

    prompt = build_glossary_prompt(domain_description, table_names)
    response = _call_ollama(
        prompt=prompt,
        model=model,
        temperature=temperature,
        timeout=GLOSSARY_TIMEOUT,
    )

    if response is None:
        logger.error("Glossary generation failed: Ollama call failed")
        return None

    # Strip markdown code blocks if present
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
        response = response.strip()

    try:
        glossary = json.loads(response)
        if not isinstance(glossary, dict):
            glossary = {}
    except json.JSONDecodeError as e:
        logger.error(f"Glossary generation failed: invalid JSON from Qwen: {e}")
        return None

    elapsed = time.time() - start_time
    logger.info(f"Glossary generated: {len(glossary)} terms in {elapsed:.1f}s")
    return glossary


def load_or_generate_glossary(
    domain_description: str = None,
    table_names: List[str] = None,
    glossary_file: str = None,
    force: bool = False,
) -> Dict[str, str]:
    """Load glossary from file if exists (unless force=True), otherwise generate and save."""
    if glossary_file is None:
        glossary_file = GLOSSARY_FILE

    if not force and os.path.exists(glossary_file):
        try:
            with open(glossary_file, "r") as f:
                glossary = json.load(f)
            logger.info(f"Glossary loaded from {glossary_file}: {len(glossary)} terms")
            return glossary
        except Exception as e:
            logger.warning(f"Could not load glossary from {glossary_file}: {e}")

    glossary = generate_glossary(domain_description=domain_description, table_names=table_names)
    if glossary is None:
        logger.warning("Glossary generation failed, using empty glossary")
        glossary = {}

    save_glossary(glossary, glossary_file)
    return glossary


def save_glossary(glossary: Dict[str, str], output_file: str = None):
    """Save glossary to JSON file."""
    if output_file is None:
        output_file = GLOSSARY_FILE

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(glossary, f, indent=2)

    logger.info(f"Glossary saved to {output_file}: {len(glossary)} terms")


def load_glossary(input_file: str = None) -> Dict[str, str]:
    """Load glossary from JSON file."""
    if input_file is None:
        input_file = GLOSSARY_FILE

    if not os.path.exists(input_file):
        logger.warning(f"Glossary file not found: {input_file}")
        return {}

    with open(input_file, "r") as f:
        glossary = json.load(f)

    logger.info(f"Glossary loaded from {input_file}: {len(glossary)} terms")
    return glossary


DEFAULT_GLOSSARY = {
    "DEBIT": "Money going out, expense, cost, outflow, payment, charge",
    "CREDIT": "Money coming in, income, received, inflow, deposit, receipt",
    "TRANSACTION": "Payment or financial event",
    "SETTLEMENT": "Completion of a transaction",
    "PENDING": "Awaiting processing",
    "COMPLETED": "Finished, done",
    "FAILED": "Error, not successful",
    "CANCELLED": "Terminated, revoked",
    "AMOUNT": "Monetary value, sum, total",
    "FEE": "Charge, commission, cost",
    "PAYER": "Party sending money",
    "RECEIVER": "Party receiving money",
    "TIMESTAMP": "Date and time of event",
    "STATUS": "Current state, condition",
}
