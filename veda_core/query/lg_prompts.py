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
If a RECOMMENDED PROJECTION list is given, selected_col_ids should come from that list —
it's the business-relevant set to display. Only add a column outside it when the query
explicitly names it, or GROUP BY/ORDER BY structurally requires it. Do not add other
COLUMN REFERENCE columns "just in case".
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

# query/answer_entity.py::_llm_relation_word — narrow fallback used only when no
# closed-class pronoun cue (who/their/its/theirs/them) matches; asks the SLM for
# ONLY the relation word, never the table/column (see that function's docstring
# for why grounding is unaffected).
ANSWER_ENTITY_RELATION_PROMPT = """\
Does this question ask to SHOW a related person's name/identity via some \
relationship (e.g. who created/owns/handles/is assigned to something) — as \
opposed to filtering BY a specific named person? If yes, output the single \
English word that names the relationship (e.g. assigned, created, owner, \
handler). If no, output exactly: none. Output ONLY that one word, nothing else.\
"""
