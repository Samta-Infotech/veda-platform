# VEDA MLflow Observability Platform — Enterprise Implementation

## Objective

Design and implement a **production-grade MLflow observability platform** for VEDA.

This is **NOT** basic experiment tracking.

The goal is to transform MLflow into an **AI Pipeline Analytics & Observability Platform**, allowing us to understand:

* Which pipeline layer contributed to the final answer.
* Which layer introduced an error.
* Latency of every stage.
* Confidence of every decision.
* Token consumption.
* Cost.
* SQL generation quality.
* Retrieval effectiveness.
* Routing effectiveness.
* Memory contribution.
* Summary contribution.
* Visualization contribution.

The system should make every query fully traceable and reproducible.

---

# High-Level Goals

For every query we should be able to answer:

* Why was this table selected?
* Which retrieval signal contributed most?
* Did graph expansion help?
* Did BM25 help?
* Did reranking change the winner?
* Did Tier2 improve Tier1?
* Was memory actually useful?
* Which validation fired?
* How much latency did every layer add?
* What is the overall pipeline accuracy?

---

# Design Principles

* Do NOT change the existing pipeline contracts.
* Do NOT change SSE responses.
* Observability should be additive.
* Logging failures must never break inference.
* MLflow should be optional (feature flag).
* Every layer should log independently.
* Every run should be reproducible.

---

# Every Query = One MLflow Run

Every user query creates one MLflow run.

Store:

Run Metadata

* run_id
* tenant_id
* conversation_id
* session_id
* user_id
* query
* normalized_query
* query_hash
* timestamp
* pipeline_version
* git_commit
* environment

---

# Global Metrics

Log

* total_latency_ms
* total_prompt_tokens
* total_completion_tokens
* total_tokens
* estimated_cost
* pipeline_status
* retry_count
* cpu_usage
* memory_usage

---

# Layer 1 — Query Understanding

Log

* latency
* intent
* business_intent
* aggregation
* metric
* dimension
* filters
* temporal
* sorting
* limit
* ambiguity
* confidence
* complexity

Store

query_understanding.json

---

# Layer 2 — Retrieval

## Latency

* retrieval_latency

## Candidate Counts

* embedding_candidates
* bm25_candidates
* graph_candidates
* value_candidates
* merged_candidates

## Signal Scores

Populate and log

* semantic_score
* bm25_score
* graph_score
* fk_score
* value_score
* rrf_score
* cross_encoder_score
* final_score

These fields already exist in RetrievalResult.

Populate them.

Do NOT create new ranking behaviour.

Pure observability.

---

## Contribution Metrics

Log

* embedding_used
* bm25_used
* graph_used
* value_used
* reranker_changed_top1

Store

retrieval_candidates.json

signal_scores.json

---

# Layer 2g — Graph Expansion

Log

* graph_latency
* graph_depth
* graph_edges
* seed_tables
* expanded_tables
* expanded_columns

Store

graph_expansion.json

---

# Layer 2b — Cross Encoder

Log

* rerank_model
* rerank_latency
* rerank_candidates
* rerank_top_score
* rerank_score_gap
* rerank_skipped

Also log

Top Before Rerank

Top After Rerank

This helps measure reranker contribution.

---

# Layer 3 — Routing

Log

* routing_latency
* primary_table
* routing_confidence
* alternative_tables
* routing_reason
* routing_changed

Store

routing.json

---

# Tier1 → Tier2

Log

* tier2_triggered
* execution_state_reused
* temporal_reused
* routing_reused
* candidate_reused
* repair_attempts
* execution_state_size

Store

execution_state.json

---

# SQL Generation

Log

* sql_model
* sql_latency
* prompt_tokens
* completion_tokens
* context_tokens
* sql_length
* joins
* tables_used
* columns_used
* group_by
* order_by
* limit_present

Store

generated_sql.sql

sql_prompt.txt

---

# SQL Validation

Log

* validation_latency
* ast_validation
* enum_validation
* value_validation
* read_only
* repair_required
* repair_count
* validation_status

Store

validation.json

---

# SQL Execution

Log

* execution_latency
* db_latency
* rows_returned
* timeout
* execution_status

---

# Memory

Log

* memory_enabled
* memory_hits
* memory_retrieved
* memory_injected
* compression_ratio
* summary_tokens
* latency

Store

memory.json

---

# Summary Layer

Log

* summary_model
* summary_latency
* summary_tokens
* summary_length

Store

summary.json

---

# Visualization

Log

* chart_type
* chart_confidence
* x_axis
* y_axis
* visualization_latency
* visualization_generated

Store

visualization.json

---

# Final Response

Store

final_response.json

This should contain

* final SQL
* final answer
* summary
* visualization
* citations
* metadata

---

# Layer Contribution Dashboard

Design MLflow dashboards showing

Pipeline Latency

Retrieval Contribution

Layer Contribution

Routing Confidence

SQL Generation Metrics

Validation Metrics

Memory Analytics

Summary Analytics

Visualization Analytics

End-to-End Accuracy

---

# Future Evaluation Support

Design the schema now, even if values are unavailable.

Future metrics

* correct_table
* correct_columns
* correct_sql
* correct_result
* summary_quality
* visualization_quality
* overall_accuracy

---

# Important Requirement

This is NOT a logging task.

This is an enterprise observability platform.

Every layer should become measurable.

Every architectural improvement should be validated using MLflow.

After implementation we should be able to answer questions like

* Which retrieval signal improved this query?
* Which layer introduced the wrong table?
* How often does graph expansion help?
* How often does reranking change the winner?
* Which routing decisions have low confidence?
* Which pipeline layer consumes the most latency?
* Which model consumes the most tokens?
* Which architectural changes actually improve SQL accuracy?

---

# Non-Goals

Do NOT redesign the pipeline.

Do NOT change inference behaviour.

Do NOT modify SQL generation.

Do NOT modify retrieval ranking.

Do NOT change SSE contracts.

Only add observability.

---

# Deliverables

Implement a reusable MLflow observability framework.

The design should be modular, production-ready, and easily extensible for future evaluation metrics and enterprise analytics.
 