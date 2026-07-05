#!/usr/bin/env python3
# =============================================================================
# enhance_semantic_model.py
# Enhance semantic_model.json with missing critical components
# =============================================================================

import sys
import os
import json
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.real_schema import get_real_schema
from config import SEMANTIC_MODEL_FILE
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_relationship_graph(schema: Dict, fk_map: Dict) -> Dict:
    """Build explicit relationship graph from FK map."""
    logger.info("Building relationship graph...")

    graph = {
        "nodes": {},
        "edges": [],
        "adjacency": defaultdict(list),
    }

    # Extract tables as nodes
    for table in schema.get("tables", []):
        table_name = table.get("table_name")
        graph["nodes"][table_name] = {
            "table_name": table_name,
            "table_id": table.get("table_id"),
            "row_count": table.get("row_count"),
            "column_count": len(table.get("columns", [])),
        }

    # Extract FKs as edges
    for from_col, to_cols in fk_map.get("forward_fks", {}).items():
        from_table = from_col.split(".")[0]
        for to_col in to_cols:
            to_table = to_col.split(".")[0]

            graph["edges"].append({
                "from_table": from_table,
                "from_column": from_col,
                "to_table": to_table,
                "to_column": to_col,
                "relationship_type": "FOREIGN_KEY",
                "cardinality": "MANY_TO_ONE",
            })

            if to_table not in graph["adjacency"][from_table]:
                graph["adjacency"][from_table].append(to_table)

    logger.info(f"  ✓ {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    return graph


def extract_join_information(schema: Dict, semantic_model: Dict) -> Dict:
    """Build join paths between tables."""
    logger.info("Building join information...")

    joins = {
        "join_paths": {},
        "join_examples": [],
        "metrics": {
            "total_possible_joins": 0,
            "documented_joins": 0,
        }
    }

    # Scan for common join patterns
    tables_in_model = set(semantic_model.get("tables", {}).keys())

    for table_name in tables_in_model:
        joins["join_paths"][table_name] = []

    # Build join paths from FK relationships
    with open("data/veda_fk_map.json") as f:
        fk_map = json.load(f)

    for from_col, to_cols in fk_map.get("forward_fks", {}).items():
        from_table = from_col.split(".")[0]
        from_col_name = from_col.split(".")[1]

        for to_col in to_cols:
            to_table = to_col.split(".")[0]
            to_col_name = to_col.split(".")[1]

            if from_table in tables_in_model and to_table in tables_in_model:
                joins["join_paths"][from_table].append({
                    "target_table": to_table,
                    "join_condition": f"{from_col} = {to_col}",
                    "join_type": "LEFT JOIN",
                    "cardinality": "MANY_TO_ONE",
                })

                joins["metrics"]["total_possible_joins"] += 1
                joins["metrics"]["documented_joins"] += 1

    logger.info(f"  ✓ {joins['metrics']['total_possible_joins']} join paths documented")
    return joins


def model_business_entities(semantic_model: Dict) -> Dict:
    """Explicitly model business entities and their relationships."""
    logger.info("Modeling business entities...")

    entities = {
        "entities": {},
        "entity_hierarchy": {},
        "entity_metrics": {},
    }

    # Map tables to business entities
    entity_mapping = {
        "checklist_template": {"entity": "CHECKLIST", "type": "MASTER", "domain": "Compliance"},
        "checklist": {"entity": "CHECKLIST_INSTANCE", "type": "TRANSACTION", "domain": "Compliance"},
        "dashboard_items": {"entity": "DASHBOARD_ITEM", "type": "CONFIGURATION", "domain": "Analytics"},
        "dashboards": {"entity": "DASHBOARD", "type": "CONFIGURATION", "domain": "Analytics"},
        "comment": {"entity": "COMMENT", "type": "TRANSACTION", "domain": "Communication"},
        "change_request": {"entity": "CHANGE_REQUEST", "type": "TRANSACTION", "domain": "Change Management"},
        "annotation_record": {"entity": "ANNOTATION", "type": "TRANSACTION", "domain": "Documentation"},
        "counterparty_details": {"entity": "COUNTERPARTY", "type": "MASTER", "domain": "Risk Management"},
        "counterparty_supplementary_info": {"entity": "COUNTERPARTY_INFO", "type": "MASTER", "domain": "Risk Management"},
        "document_category_master": {"entity": "DOCUMENT_CATEGORY", "type": "REFERENCE", "domain": "Classification"},
    }

    for table_name, mapping in entity_mapping.items():
        if table_name in semantic_model.get("tables", {}):
            entities["entities"][mapping["entity"]] = {
                "table_name": table_name,
                "entity_type": mapping["type"],
                "domain": mapping["domain"],
                "description": semantic_model["tables"][table_name].get("business_purpose", ""),
            }

    logger.info(f"  ✓ {len(entities['entities'])} business entities modeled")
    return entities


def define_metrics(semantic_model: Dict) -> Dict:
    """Extract and define measurable metrics."""
    logger.info("Defining metrics...")

    metrics = {
        "metrics": {},
        "metric_types": ["COUNT", "SUM", "AVG", "MIN", "MAX", "RATE"],
    }

    # Identify metric columns
    metric_keywords = ["count", "sum", "total", "amount", "value", "score", "percentage", "rate", "avg", "min", "max"]

    for col_key, col_data in semantic_model.get("columns", {}).items():
        col_name = col_data.get("col_name", "").lower()
        semantic_type = col_data.get("semantic_type", "").lower()

        # Check if this looks like a metric
        is_metric = any(keyword in col_name for keyword in metric_keywords)
        if not is_metric and "MEASURE" in col_data.get("analytics_role", ""):
            is_metric = True

        if is_metric:
            metrics["metrics"][col_key] = {
                "column": col_key,
                "semantic_type": col_data.get("semantic_type"),
                "analytics_role": col_data.get("analytics_role"),
                "business_definition": col_data.get("business_definition", ""),
                "aggregation_functions": ["COUNT", "SUM", "AVG"],
            }

    logger.info(f"  ✓ {len(metrics['metrics'])} metrics defined")
    return metrics


def fix_semantic_role_misclassifications(semantic_model: Dict) -> int:
    """Fix incorrect semantic role assignments."""
    logger.info("Fixing semantic role misclassifications...")

    fixes = 0

    # Rules for role correction
    role_rules = {
        "created_by_id": "IDENTIFIER",
        "updated_by_id": "IDENTIFIER",
        "editor_id": "IDENTIFIER",
        "created_datetime": "TIME_DIMENSION",
        "updated_datetime": "TIME_DIMENSION",
        "deleted_datetime": "TIME_DIMENSION",
        "creation_date": "TIME_DIMENSION",
        "updation_date": "TIME_DIMENSION",
        "_id": "IDENTIFIER",
        "_date": "TIME_DIMENSION",
        "_datetime": "TIME_DIMENSION",
        "_count": "MEASURE",
        "_total": "MEASURE",
        "_amount": "MEASURE",
    }

    for col_key, col_data in semantic_model.get("columns", {}).items():
        col_name = col_data.get("col_name", "").lower()
        current_role = col_data.get("analytics_role", "")

        # Check if this column matches a rule
        for pattern, correct_role in role_rules.items():
            if pattern in col_name:
                if current_role != correct_role:
                    semantic_model["columns"][col_key]["analytics_role"] = correct_role
                    fixes += 1
                break

    logger.info(f"  ✓ Fixed {fixes} role misclassifications")
    return fixes


def handle_null_and_sparse_columns(semantic_model: Dict, profiling: Dict) -> int:
    """Handle columns with 100% null or 0 distinct values."""
    logger.info("Handling null/sparse columns...")

    handled = 0

    for col_key in semantic_model.get("columns", {}).keys():
        if col_key in profiling:
            prof = profiling[col_key]
            null_pct = prof.get("null_percentage", 0)
            distinct = prof.get("distinct_count", 0)

            # Mark sparse columns
            if null_pct > 95 or distinct == 0:
                semantic_model["columns"][col_key]["data_quality_flag"] = "SPARSE"
                semantic_model["columns"][col_key]["data_quality_reason"] = (
                    f"null%={null_pct}, distinct={distinct}"
                )
                handled += 1

    logger.info(f"  ✓ Flagged {handled} sparse/low-quality columns")
    return handled


def enhance_business_ontology(semantic_model: Dict, glossary: Dict) -> Dict:
    """Enhance business ontology coverage."""
    logger.info("Enhancing business ontology...")

    ontology = {
        "domains": {},
        "concepts": {},
        "semantic_tags": {},
    }

    # Map glossary terms to domain concepts
    domain_concept_map = {
        "compliance": ["COMPLIANCE", "AUDIT", "REGULATION", "RISK"],
        "kyc": ["CUSTOMER", "VERIFICATION", "IDENTITY", "AML"],
        "aml": ["SANCTIONS", "PEP", "MONEY_LAUNDERING", "SUSPICIOUS"],
        "incident": ["ALERT", "INVESTIGATION", "RESOLUTION", "TRACKING"],
        "fraud": ["DETECTION", "PREVENTION", "PATTERN", "ANOMALY"],
    }

    for term, definition in glossary.items():
        term_lower = term.lower()
        for domain, concepts in domain_concept_map.items():
            if domain in term_lower or any(c.lower() in term_lower for c in concepts):
                if domain not in ontology["domains"]:
                    ontology["domains"][domain] = {
                        "domain_name": domain.upper(),
                        "concepts": [],
                        "glossary_terms": [],
                    }

                if term not in ontology["domains"][domain]["glossary_terms"]:
                    ontology["domains"][domain]["glossary_terms"].append(term)

    # Add concepts
    for domain, domain_data in ontology["domains"].items():
        ontology["concepts"][domain] = domain_concept_map.get(domain, [])

    logger.info(f"  ✓ Enhanced ontology: {len(ontology['domains'])} domains")
    return ontology


def add_query_examples(semantic_model: Dict) -> Dict:
    """Add query examples and usage patterns."""
    logger.info("Adding query examples...")

    examples = {
        "query_examples": [],
        "usage_patterns": [],
    }

    # Add common query patterns
    query_patterns = [
        {
            "intent": "DIRECT",
            "example": "show me all checklist IDs",
            "tables_involved": ["checklist"],
            "columns_needed": ["checklist.id"],
        },
        {
            "intent": "TEMPORAL",
            "example": "list changes created in the last 30 days",
            "tables_involved": ["change_request"],
            "columns_needed": ["change_request.created_datetime"],
        },
        {
            "intent": "AGGREGATE",
            "example": "count checklists by template",
            "tables_involved": ["checklist", "checklist_template"],
            "columns_needed": ["checklist.checklist_id", "checklist_template.name"],
            "aggregation": "COUNT(*)",
        },
        {
            "intent": "MULTI_TABLE",
            "example": "find counterparties and their associated documents",
            "tables_involved": ["counterparty_details", "document_category_master"],
            "columns_needed": ["counterparty_details.party_name", "document_category_master.category_name"],
        },
    ]

    examples["query_examples"] = query_patterns
    examples["usage_patterns"] = [
        {"pattern": "ID_LOOKUP", "description": "Find records by ID"},
        {"pattern": "TIME_RANGE", "description": "Filter by date range"},
        {"pattern": "GROUP_BY_ENTITY", "description": "Aggregate by entity"},
        {"pattern": "JOIN_TABLES", "description": "Combine data across tables"},
    ]

    logger.info(f"  ✓ Added {len(examples['query_examples'])} query examples")
    return examples


def run_semantic_model_enhancement():
    """Run all enhancements."""
    logger.info("\n" + "="*80)
    logger.info("ENHANCING SEMANTIC_MODEL.JSON WITH CRITICAL COMPONENTS")
    logger.info("="*80 + "\n")

    # Load files
    with open(SEMANTIC_MODEL_FILE) as f:
        semantic_model = json.load(f)

    with open("data/veda_profiling.json") as f:
        profiling = json.load(f)

    # Load domain glossary
    glossary = {}
    glossary_files = [
        "glossary/domain_glossary.json",
        "glossary/static_glossary.json",
        "glossary/hf_glossary.json",
    ]
    for gf in glossary_files:
        try:
            with open(gf) as f:
                glossary.update(json.load(f))
        except FileNotFoundError:
            pass

    with open("data/veda_fk_map.json") as f:
        fk_map = json.load(f)

    schema = get_real_schema()

    # Run enhancements
    try:
        # 1. Relationship graph (CRITICAL)
        graph = extract_relationship_graph(schema, fk_map)
        semantic_model["relationship_graph"] = graph

        # 2. Join information (CRITICAL)
        joins = extract_join_information(schema, semantic_model)
        semantic_model["join_information"] = joins

        # 3. Business entities (HIGH)
        entities = model_business_entities(semantic_model)
        semantic_model["business_entities"] = entities

        # 4. Metric definitions (HIGH)
        metrics = define_metrics(semantic_model)
        semantic_model["metrics"] = metrics

        # 5. Fix semantic role misclassifications (MEDIUM)
        role_fixes = fix_semantic_role_misclassifications(semantic_model)

        # 6. Handle null/sparse columns (MEDIUM)
        sparse_flags = handle_null_and_sparse_columns(semantic_model, profiling)

        # 7. Business ontology (HIGH)
        ontology = enhance_business_ontology(semantic_model, glossary)
        semantic_model["business_ontology"] = ontology

        # 8. Query examples (HIGH)
        examples = add_query_examples(semantic_model)
        semantic_model["query_examples"] = examples

        # Save enhanced model
        with open(SEMANTIC_MODEL_FILE, "w") as f:
            json.dump(semantic_model, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("✓ SEMANTIC MODEL ENHANCEMENT COMPLETE")
        logger.info("="*80)
        logger.info("\nEnhancements applied:")
        logger.info(f"  1. ✓ Relationship graph: {len(graph['edges'])} FK relationships")
        logger.info(f"  2. ✓ Join information: {joins['metrics']['total_possible_joins']} join paths")
        logger.info(f"  3. ✓ Business entities: {len(entities['entities'])} entities modeled")
        logger.info(f"  4. ✓ Metric definitions: {len(metrics['metrics'])} metrics defined")
        logger.info(f"  5. ✓ Role fixes: {role_fixes} misclassifications corrected")
        logger.info(f"  6. ✓ Sparse columns flagged: {sparse_flags} columns")
        logger.info(f"  7. ✓ Business ontology: {len(ontology['domains'])} domains")
        logger.info(f"  8. ✓ Query examples: {len(examples['query_examples'])} patterns added")

        return semantic_model

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    run_semantic_model_enhancement()
