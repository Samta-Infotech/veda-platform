"""query/lg_prompts.py — Focused prompts for LangGraph nodes."""

INTENT_PROMPT = """\
Classify this database query. Output ONLY JSON, no markdown.
INTENTS: SELECT | COUNT | AGGREGATE
COMPLEXITY: SIMPLE | MODERATE | COMPLEX
needs_clarification: true only if completely ambiguous or outside database domain.
{"intent":"SELECT","complexity":"SIMPLE","needs_clarification":false,"clarification_reason":null}
"""

ENTITY_PROMPT = """\
Select the PRIMARY table for this query.
Copy table_id EXACTLY from TABLE REFERENCE — never generate a UUID.
Output ONLY JSON, no markdown.
{"primary_table_id":"<uuid>","secondary_table_ids":["<uuid>"]}
"""

COLUMN_PROMPT = """\
Select relevant columns from COLUMN REFERENCE.
Copy col_id EXACTLY — never generate a UUID.
COUNT with no grouping: selected_col_ids = []
Self-check: every col_id must be in COLUMN REFERENCE.
Output ONLY JSON, no markdown.
{"selected_col_ids":["<uuid>"],"group_by_col_id":null,"order_by_col_id":null,"order_direction":"ASC"}
"""

FILTER_PROMPT = """\
Build filter_tree from COLUMN REFERENCE.
Copy col_id EXACTLY — never generate a UUID.
BETWEEN for dates. EQ for exact matches. Boolean: true/false not strings.
Never use placeholders like <literal>.
Self-check: every col_id must be in COLUMN REFERENCE.
Output ONLY JSON, no markdown.
{"filter_tree":{"type":"AND","children":[{"col_id":"<uuid>","operator":"EQ","value":"active"}]}}
OPERATORS: EQ|NEQ|GT|GTE|LT|LTE|BETWEEN|LIKE|IN|IS_NULL
"""
