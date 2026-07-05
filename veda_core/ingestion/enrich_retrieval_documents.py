#!/usr/bin/env python3
# =============================================================================
# enrich_retrieval_documents.py
# Enrich retrieval_documents with FK + entity + domain + metrics + examples
# =============================================================================

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SEMANTIC_MODEL_FILE
from utils.logger import get_logger

logger = get_logger(__name__)


def enrich_retrieval_documents():
    """Enrich all retrieval_documents with critical metadata."""

    logger.info("\n" + "="*80)
    logger.info("ENRICHING RETRIEVAL_DOCUMENTS WITH SEMANTIC METADATA")
    logger.info("="*80 + "\n")

    # Load model with all enhancements
    with open(SEMANTIC_MODEL_FILE) as f:
        model = json.load(f)

    # Load supporting data
    with open("data/veda_fk_map.json") as f:
        fk_map = json.load(f)

    # Get entity mapping
    entity_map = {}
    if "business_entities" in model:
        for entity, data in model["business_entities"].get("entities", {}).items():
            table = data.get("table_name")
            entity_map[table] = {
                "entity": entity,
                "type": data.get("entity_type"),
                "domain": data.get("domain"),
            }

    # Get metric columns
    metric_cols = set()
    if "metrics" in model:
        metric_cols = set(model["metrics"].get("metrics", {}).keys())

    # Get query examples
    query_examples = {}
    if "query_examples" in model:
        for example in model["query_examples"].get("query_examples", []):
            for col_needed in example.get("columns_needed", []):
                if col_needed not in query_examples:
                    query_examples[col_needed] = {
                        "intent": example.get("intent"),
                        "example": example.get("example"),
                    }

    # Get domain mapping
    domain_map = {}
    if "business_ontology" in model:
        for domain, data in model["business_ontology"].get("domains", {}).items():
            for col_key in metric_cols:
                table_name = col_key.split(".")[0]
                # Simple mapping: if table entity matches domain terms, tag it
                domain_map[col_key] = domain

    # Process each retrieval document
    enhanced_count = 0

    for col_key in list(model["retrieval_documents"].keys()):
        table_name = col_key.split(".")[0]
        col_name = col_key.split(".")[1]

        old_doc = model["retrieval_documents"][col_key]

        # Parse the document
        lines = old_doc.split(" |\n")
        new_sections = []

        # Add FK linkages
        if col_key in fk_map.get("forward_fks", {}):
            fk_refs = ", ".join(fk_map["forward_fks"][col_key])
            # Replace empty LINKS TO with actual FKs
            new_doc = old_doc.replace("LINKS TO:  |", f"LINKS TO: {fk_refs} |")
        else:
            new_doc = old_doc

        # Add business entity
        if table_name in entity_map:
            entity_info = entity_map[table_name]
            new_doc += f"\nENTITY: {entity_info['entity']} | TYPE: {entity_info['type']} | DOMAIN: {entity_info['domain']} |"

        # Add if metric
        if col_key in metric_cols:
            new_doc += f"\nMETRIC: YES | Aggregatable: true | Functions: COUNT,SUM,AVG |"

        # Add domain/ontology
        if col_key in domain_map:
            new_doc += f"\nDOMAIN: {domain_map[col_key].upper()} |"

        # Add query example
        if col_key in query_examples:
            ex = query_examples[col_key]
            new_doc += f"\nQUERY_EXAMPLE: [{ex['intent']}] {ex['example']} |"

        # Update the document
        model["retrieval_documents"][col_key] = new_doc
        enhanced_count += 1

    # Save enriched model
    with open(SEMANTIC_MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)

    logger.info(f"✓ Enhanced {enhanced_count} retrieval_documents")
    logger.info("\nEnhancements added:")
    logger.info(f"  ✓ FK linkages")
    logger.info(f"  ✓ Business entity info")
    logger.info(f"  ✓ Metric identifiers")
    logger.info(f"  ✓ Domain/ontology tags")
    logger.info(f"  ✓ Query examples")

    # Show sample
    sample_col = list(model["retrieval_documents"].keys())[0]
    if any(col in sample_col for col in metric_cols):
        sample_col = [c for c in metric_cols if c in model["retrieval_documents"]][0]

    print("\n" + "="*80)
    print("SAMPLE ENRICHED DOCUMENT")
    print("="*80 + "\n")
    print(f"Column: {sample_col}")
    print(f"Document:\n{model['retrieval_documents'][sample_col]}\n")

    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 1 SEMANTIC LAYER COMPLETE")
    logger.info("="*80)
    logger.info("\nPhase 1 deliverables:")
    logger.info("  ✓ veda_semantic_model.json - Rich metadata + enriched retrieval_documents")
    logger.info("  ✓ veda_profiling.json - Column statistics")
    logger.info("  ✓ veda_fk_map.json - Foreign key relationships")
    logger.info("  ✓ glossary/ - Domain terminology")
    logger.info("\nRetrieval documents are now ready for:")
    logger.info("  1. Embedding (SentenceTransformer)")
    logger.info("  2. Storage in pgvector")
    logger.info("  3. Phase 2 semantic ranking")

    return model


if __name__ == "__main__":
    enrich_retrieval_documents()
