"""Qualifier-salvage acceptance — referent_tables + the re-anchor retry contract.

Covers the wrong-anchor refusal class generically (the "completed payments →
qualifier_dropped: 'payment'" failure, on ANY schema):
  S1: a dropped token that names an entity of this scope resolves to its owning
      table(s), ranked deterministically — the salvage retry's anchor candidates.
  S2: FOREIGN-schema case (the production shape: payment_id / payment_transaction_id
      columns on a table whose NAME never says payment) — the token must resolve via
      column-NAME ownership, with zero local-schema assumptions.
  S3: generic-word guard — a token matching more than QSR_REFERENT_MAX_COLS columns
      must NOT be credited through column names (request verbs stay out).
  S4: run_query exposes the anchor_hint retry contract (recursion guard's key).
  Safety: referent_tables never raises and returns [] for sub-content tokens.

Uses the on-disk dev artifacts (semantic model) like test_qsr_resolution.py.
Run: python tests/test_qualifier_salvage.py  (from repo root).
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.join(ROOT, "veda_core")
sys.path.insert(0, CORE)
os.chdir(CORE)  # artifact_path() unscoped fallback resolves data/ relative to cwd

SM = json.load(open("data/veda_semantic_model.json"))

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


from query.resolution import referent_tables

# ── S1: entity-word resolution on the dev scope ──────────────────────────────
refs = referent_tables("payment", SM)
check("'payment' resolves to referent tables", bool(refs), str(refs))
check("'payment' → accounts_paymenttransaction among referents",
      any(r["table"] == "accounts_paymenttransaction" for r in refs),
      str([r["table"] for r in refs][:5]))
check("referents ranked (scores non-increasing)",
      all(refs[i]["score"] >= refs[i + 1]["score"] for i in range(len(refs) - 1)))
check("deterministic across calls", refs == referent_tables("payment", SM))
check("plural form resolves like singular",
      any(r["table"] == "accounts_paymenttransaction"
          for r in referent_tables("payments", SM)))

# ── safety: sub-content tokens ────────────────────────────────────────────────
check("len<3 token → []", referent_tables("at", SM) == [])
check("empty/None → []", referent_tables("", SM) == [] and referent_tables(None, SM) == [])

# ── S2: FOREIGN schema — the production failure shape ────────────────────────
# A schema this codebase has never seen: no table is named after payment; the
# only evidence is engineer-written column names. This is exactly the remote
# refusal ("payment" → suggestions payment_transaction_id / payment_id) — the
# salvage must find the owning table from the scope's own vocabulary alone.
SM2 = {
    "tables": {"orders": {"business_purpose": "customer orders"},
               "customers": {"business_purpose": "customer master"}},
    "columns": {
        "orders.payment_id": {"semantic_type": "IDENTIFIER"},
        "orders.payment_transaction_id": {"semantic_type": "IDENTIFIER"},
        "orders.other_payment_type": {"semantic_type": "CATEGORY"},
        "orders.total_amount": {"analytics_role": "MEASURE"},
        "customers.customer_name": {"semantic_type": "TEXT"},
    },
}
refs2 = referent_tables("payment", SM2)
check("foreign schema: 'payment' → orders via column names",
      any(r["table"] == "orders" for r in refs2), str(refs2))
check("foreign schema: column-name ownership is the credited reason",
      any(r["table"] == "orders" and any("column-name" in w for w in r["why"])
          for r in refs2), str(refs2))
check("foreign schema: unrelated table not credited by column names",
      not any(r["table"] == "customers" and any("column-name" in w for w in r["why"])
              for r in refs2), str(refs2))

# ── S3: generic-word guard (QSR_REFERENT_MAX_COLS) ───────────────────────────
SM3 = {"tables": {"widgets": {}},
       "columns": {f"widgets.flag_{i}": {"semantic_type": "FLAG"} for i in range(25)}}
refs3 = referent_tables("flag", SM3)
check("token in >MAX_COLS columns gets no column-name credit",
      not any(r["table"] == "widgets" and any("column-name" in w for w in r["why"])
              for r in refs3), str(refs3))

# ── S4: the retry contract on run_query ──────────────────────────────────────
import inspect
from veda.pipeline import run_query
check("run_query accepts anchor_hint (salvage retry + recursion guard)",
      "anchor_hint" in inspect.signature(run_query).parameters)

# ── trigger-case interplay: 'completed' keeps its grounded-domain clarify ────
# 'completed' is NOT a value in this scope (test_qsr_resolution pins that); if it
# also has no referent tables, the salvage stays out of the way and the existing
# FK-domain clarify ("statuses here are captured/authorized/cancelled") wins.
refs_c = referent_tables("completed", SM)
print(f"INFO 'completed' referents: {[r['table'] for r in refs_c][:3] or 'none'}")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all qualifier-salvage checks passed")
