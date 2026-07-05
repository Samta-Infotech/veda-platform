#!/usr/bin/env python3
# =============================================================================
# gen_debug_files.py
# Generate 2 critical debug files for Phase 1-3 troubleshooting
# =============================================================================

import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.real_schema import get_real_schema
from config import SEMANTIC_MODEL_FILE
from utils.logger import get_logger

logger = get_logger(__name__)


def gen_fk_map():
    """Generate FK relationships map for debugging retrieval failures."""
    logger.info("Generating veda_fk_map.json...")

    schema = get_real_schema()
    fk_map = defaultdict(list)
    reverse_fk_map = defaultdict(list)

    # Extract FK relationships from schema
    for table in schema.get("tables", []):
        table_name = table.get("table_name")
        for col in table.get("columns", []):
            col_name = col.get("col_name") or col.get("name")

            # Heuristic: *_id columns reference other tables
            if col_name.endswith("_id") and col_name != "id":
                ref_table = col_name[:-3]
                from_col = f"{table_name}.{col_name}"
                to_col = f"{ref_table}.id"

                fk_map[from_col].append(to_col)
                reverse_fk_map[to_col].append(from_col)

    fk_debug = {
        "forward_fks": dict(fk_map),  # Which columns reference which
        "reverse_fks": dict(reverse_fk_map),  # Which columns are referenced by which
        "stats": {
            "total_fks": sum(len(v) for v in fk_map.values()),
            "columns_with_fks": len(fk_map),
            "columns_referenced": len(reverse_fk_map),
        },
        "purpose": "Debug retrieval when columns don't link across tables"
    }

    with open("data/veda_fk_map.json", "w") as f:
        json.dump(fk_debug, f, indent=2)

    logger.info(f"  ✓ FK map: {fk_debug['stats']['total_fks']} relationships")
    return fk_debug


def gen_retrieval_trace_template():
    """Generate template for Phase 2 retrieval trace logging."""
    logger.info("Generating veda_retrieval_trace_template.json...")

    with open(SEMANTIC_MODEL_FILE) as f:
        semantic_model = json.load(f)

    trace_template = {
        "query": "USER QUERY STRING",
        "intent": "DIRECT|TEMPORAL|AGGREGATE|SYNONYM|MULTI_TABLE",
        "phase_2_steps": {
            "step_1_semantic_ranking": {
                "description": "Bi-encoder embedding similarity",
                "top_results": [
                    {
                        "column": "table.column",
                        "score": 0.85,
                        "reason": "semantic similarity to query"
                    }
                ]
            },
            "step_2_bm25_ranking": {
                "description": "Keyword matching via BM25",
                "top_results": [
                    {
                        "column": "table.column",
                        "score": 0.72,
                        "reason": "keyword match: 'user'"
                    }
                ]
            },
            "step_3_signals": {
                "description": "FK + subgraph signals",
                "fk_signal": {"table.column": 0.5},
                "subgraph_signal": {"table.column": 0.3}
            },
            "step_4_rrf_merge": {
                "description": "RRF combines all signals",
                "top_results": [
                    {
                        "column": "table.column",
                        "rrf_score": 0.68,
                        "signal_breakdown": {
                            "semantic": 0.85,
                            "bm25": 0.72,
                            "fk": 0.5,
                            "subgraph": 0.3
                        }
                    }
                ]
            },
            "step_5_reranking": {
                "description": "Cross-encoder fine-grained reranking",
                "final_results": [
                    {
                        "column": "table.column",
                        "final_score": 0.70,
                        "reranked_by": "cross-encoder"
                    }
                ]
            }
        },
        "final_top_k": [
            {"rank": 1, "column": "table.column", "score": 0.70}
        ],
        "cache_hit": False,
        "response_time_ms": 150,
        "path": "full-pipeline",
        "purpose": "Debug when retrieval returns wrong columns"
    }

    with open("data/veda_retrieval_trace_template.json", "w") as f:
        json.dump(trace_template, f, indent=2)

    logger.info("  ✓ Retrieval trace template created")
    return trace_template


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING 2 CRITICAL DEBUG FILES")
    print("="*80 + "\n")

    try:
        fk_map = gen_fk_map()
        trace = gen_retrieval_trace_template()

        print("\n" + "="*80)
        print("✓ DEBUG FILES READY")
        print("="*80)
        print("\nFiles generated:")
        print(f"  1. data/veda_fk_map.json ({fk_map['stats']['total_fks']} FK relationships)")
        print(f"  2. data/veda_retrieval_trace_template.json (Phase 2 trace format)")
        print("\nWhen retrieval fails:")
        print("  - Check veda_fk_map.json for missing FK relationships")
        print("  - Enable retrieval trace logging to see signal scores step-by-step")

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        raise
