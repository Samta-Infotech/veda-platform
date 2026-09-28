"""veda.agent — the planner agent (flag AGENT_PLANNER_ENABLED, default OFF).

The same local SLM that extracts frames, given VEDA's own ingestion semantics as TOOLS,
builds a typed Plan in short constrained steps over identifiers it RETRIEVED — never
recalled:

    tools.py    read-only, deterministic wrappers over what ingestion already publishes
                (entity cards, glossaries, semantic types, relationship graph, probes)
    plan.py     the Plan object, its validation against the run's tool log, and its
                compilation through the EXISTING builders (join_planner.build_skeleton,
                planning.build_aggregate_sql, frame_compiler.predicate) → QueryIR + SQL
    planner.py  the ReAct loop (constrained JSON at every step, budgets, reflection)
    examples.py the six worked traces shown to the model (also replayed by the tests)

The planner never writes SQL and never names an identifier a tool did not return; the
existing firewall judges the compiled statement exactly as it judges a frame answer.
"""
