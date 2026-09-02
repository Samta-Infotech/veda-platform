"""query/operation_classifier.py — bounded cross-source OPERATION classifier (routing Phase 2).

The single decision this module makes for a MULTI (cross-source) query is a CHOICE among a fixed,
closed set of supported operations — or UNSUPPORTED. It is the "reduce the problem until the 7B only
makes a bounded choice" layer: the SLM never writes SQL, never names a table/column/join key, never
invents a source. It receives the query plus ABSTRACT structural facts (is there a validated
relationship? are there documents? how many data sources?) and the allowed operation list, and returns
ONE operation label. Its output is then validated DETERMINISTICALLY against those same structural
facts, so an operation the data cannot support (e.g. SEMI_JOIN with no cross-source relationship) is
downgraded to UNSUPPORTED — never executed on a guess.

CODE (the caller) then dispatches the chosen operation to the matching EXISTING deterministic planner
(semi_join_planner / structured-plan / doc_data_planner). UNSUPPORTED → controlled clarify/refuse,
NEVER a free-form-SQL fallback.

Flag: OPERATION_CLASSIFIER_ENABLED (default OFF → this module is never entered; the legacy
sequential-fallback federated chain runs byte-identical).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Set

from slm._call_slm import call_slm

# ── the closed operation set ──────────────────────────────────────────────────────────────────────
OP_SEMI_JOIN = "SEMI_JOIN"                    # filter A's rows to those whose key exists in B
OP_AGG_AFTER_JOIN = "AGGREGATE_AFTER_JOIN"    # join across sources, then aggregate/group
OP_SET_INTERSECTION = "SET_INTERSECTION"      # the items present in BOTH sources
OP_DOC_TO_STRUCTURED = "DOC_TO_STRUCTURED"    # entities a document names, intersected with data
OP_LOOKUP_ENRICH = "LOOKUP_ENRICH"            # A's rows augmented with a related value from B
OP_UNSUPPORTED = "UNSUPPORTED"                # none of the above / not safely plannable

ALL_OPS: Set[str] = {
    OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION,
    OP_DOC_TO_STRUCTURED, OP_LOOKUP_ENRICH, OP_UNSUPPORTED,
}

# Operations reachable from the column-bearing federated route (a document source is not column-
# bearing, so DOC_TO_STRUCTURED is dispatched by its own pre-federated branch, not here).
FEDERATED_OPS: Set[str] = {
    OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION, OP_LOOKUP_ENRICH,
}

# Operations that structurally REQUIRE a validated cross-source relationship (a cross_source_fk join
# path). Without one, the SLM's choice is infeasible → UNSUPPORTED.
_RELATIONSHIP_REQUIRED: Set[str] = {
    OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION, OP_LOOKUP_ENRICH,
}


@dataclass
class OperationContext:
    """The ABSTRACT structural facts the classifier is validated against — derived deterministically by
    the caller from the assembled cross-source context (join hints / RAG chunks / source kinds). The
    SLM never sees raw schema; these booleans gate what it is allowed to have chosen."""
    has_relationship: bool          # a cross_source_fk join path exists between the data sources
    has_documents: bool             # RAG document chunks are in scope
    data_source_count: int          # number of column-bearing (relational/datalake) sources in scope


@dataclass
class OperationDecision:
    """Typed, validated classifier outcome. `valid` = the SLM label survived deterministic validation;
    when False, `operation` has been forced to UNSUPPORTED. `method` records provenance."""
    operation: str
    reason: str
    valid: bool
    method: str = "slm"             # "slm" | "deterministic" (forced without/over the SLM)


_OP_DEFS = (
    f"- {OP_SEMI_JOIN}: keep one source's rows only where its key ALSO appears in the other source "
    "(\"which X also have a Y\", \"X that appear in Y\").\n"
    f"- {OP_AGG_AFTER_JOIN}: join the sources, then COUNT/SUM/AVG/etc, usually grouped "
    "(\"total/average/count of X per Y\" where the grouping/measure spans both sources).\n"
    f"- {OP_SET_INTERSECTION}: the items present in BOTH sources (\"which X are in both A and B\").\n"
    f"- {OP_LOOKUP_ENRICH}: return one source's rows AUGMENTED with a related value looked up from the "
    "other (\"show each X along with its Y from the other source\").\n"
    f"- {OP_UNSUPPORTED}: none of the above cleanly fits, or the question needs reasoning no single "
    "supported operation covers."
)

_SYSTEM = (
    "You are a bounded OPERATION classifier for a cross-source data assistant. Given a question that "
    "needs data from more than one source, choose the ONE operation that best describes how the "
    "sources must be combined. Choose ONLY from the operations offered. Do NOT write SQL. Do NOT name "
    "tables, columns, or join keys. Reply with STRICT JSON only: "
    "{\"operation\": \"<one of the offered labels>\", \"reason\": \"<short, grounded in the question>\"}.\n"
    "Operation meanings:\n" + _OP_DEFS + "\n"
    "When unsure between a specific operation and UNSUPPORTED, prefer the specific operation ONLY if "
    "the question clearly matches it; otherwise choose UNSUPPORTED."
)


def _extract_json(raw) -> Optional[dict]:
    """Best-effort strict-JSON parse of an SLM reply (tolerant of a ```json fence)."""
    if not raw:
        return None
    s = str(raw).strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # last resort: first {...} span
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(s[i:j + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


# Language-layer CONSTRAINT signals (not schema/business vocabulary): a query carrying one of these
# expresses a filter that NO currently-supported deterministic cross-source operation can represent —
# semi-join is positive-existence only (no anti-join, no count threshold) and the structured aggregate
# template has no HAVING. Mapping such a query to the nearest op would SILENTLY drop the threshold or
# INVERT the negation (a wrong answer that looks right). The guard REFUSES these to UNSUPPORTED — the
# same grammar-signal discipline as the grouped/distinct answer-shape guards (refuse, never route). If
# an anti-join / HAVING template is added later, relax the matching set for that operation.
_NEGATION = (" not ", " without ", " never ", " excluding ", " except ", "n't have", "n't own",
             " don't have", " do not have", " with no ")
_COUNT_THRESHOLD = (" more than ", " at least ", " fewer than ", " greater than ", " less than ",
                    " at most ", " no more than ", " no fewer than ", " over ")


def _expressible(operation: str, query: str) -> tuple:
    """Deterministic expressiveness check: can ANY supported operation's template represent the
    CONSTRAINT this query carries? A negation ("assets that do NOT have tickets") needs an anti-join;
    a count threshold ("more than 5 tickets") needs HAVING — neither exists in the current planners, so
    the query is refused to UNSUPPORTED rather than answered with a constraint silently dropped/inverted.
    Returns (ok, reason). Language-signal vs the planners' known limits — no schema vocabulary, no
    routing; UNSUPPORTED itself is always expressible."""
    if operation == OP_UNSUPPORTED:
        return True, ""
    q = " " + (query or "").lower().strip() + " "
    if any(s in q for s in _NEGATION):
        return False, "negation / anti-join is not expressible by any supported cross-source operation"
    if any(s in q for s in _COUNT_THRESHOLD):
        return False, "a count threshold (HAVING) is not expressible by any supported cross-source operation"
    return True, ""


def _feasible(operation: str, ctx: OperationContext) -> bool:
    """Deterministic feasibility: can the structural context actually support this operation? A
    relationship-requiring operation needs a validated cross-source relationship AND ≥2 data sources.
    DOC_TO_STRUCTURED needs documents AND ≥1 data source. UNSUPPORTED is always 'feasible' (it is the
    safe outcome)."""
    if operation == OP_UNSUPPORTED:
        return True
    if operation == OP_DOC_TO_STRUCTURED:
        return ctx.has_documents and ctx.data_source_count >= 1
    if operation in _RELATIONSHIP_REQUIRED:
        return ctx.has_relationship and ctx.data_source_count >= 2
    return False


def classify_operation(query: str, ctx: OperationContext,
                       *, allowed_ops: Optional[Set[str]] = None) -> OperationDecision:
    """Classify a cross-source query into ONE supported operation (or UNSUPPORTED).

    `allowed_ops`: the operations the CALLER can actually dispatch (e.g. FEDERATED_OPS at the
    column-bearing federated route). UNSUPPORTED is always implicitly allowed. The SLM is offered only
    these; its answer is then validated against `ctx` — an out-of-set, unparseable, or structurally
    infeasible choice deterministically degrades to UNSUPPORTED (never guessed into execution).

    Pure aside from the single bounded SLM call; any SLM/parse failure → UNSUPPORTED (safe)."""
    allowed = set(allowed_ops) if allowed_ops else set(FEDERATED_OPS)
    allowed = (allowed & ALL_OPS) | {OP_UNSUPPORTED}

    offered = [op for op in (OP_SEMI_JOIN, OP_AGG_AFTER_JOIN, OP_SET_INTERSECTION,
                             OP_DOC_TO_STRUCTURED, OP_LOOKUP_ENRICH) if op in allowed]
    offered.append(OP_UNSUPPORTED)
    user = (f"Question: {query}\n\nOffered operations (choose exactly one): "
            f"{', '.join(offered)}\n\nReturn the JSON.")

    try:
        obj = _extract_json(call_slm(user, system=_SYSTEM, purpose="operation_classify",
                                     temperature=0.0, json_format=True))
    except Exception:
        obj = None

    if not obj or not isinstance(obj.get("operation"), str):
        return OperationDecision(OP_UNSUPPORTED, "no valid operation returned", False, "deterministic")

    op = obj["operation"].strip().upper()
    reason = str(obj.get("reason") or "")[:200]

    if op not in allowed:
        return OperationDecision(OP_UNSUPPORTED, f"'{op}' not an offered operation", False, "deterministic")
    if not _feasible(op, ctx):
        return OperationDecision(
            OP_UNSUPPORTED,
            f"'{op}' not structurally supported here (relationship={ctx.has_relationship}, "
            f"documents={ctx.has_documents}, data_sources={ctx.data_source_count})",
            False, "deterministic")
    ok_expr, expr_reason = _expressible(op, query)
    if not ok_expr:
        return OperationDecision(OP_UNSUPPORTED, expr_reason, False, "deterministic")
    return OperationDecision(op, reason, True, "slm")
