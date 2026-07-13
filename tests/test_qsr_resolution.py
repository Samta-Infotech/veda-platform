"""QSR (Phase A) acceptance — schema-vocab tokenizer + typed resolution + FK closure.

Covers the failure classes from ARCHITECTURE_ROOT_CAUSE_PLAN.md §1 at unit level:
  F1: fused Django table names must expose entity words (paymenttransaction → payment).
      Genuine words must NEVER be false-split (transaction ≠ trans+action).
  F4: 'captured' must resolve to the REFERENCING transaction table via FK closure;
      'completed' must carry NO value referent (it isn't in the data);
      domain_via must return the FK-scoped label set (captured/authorized/cancelled).
  Safety: typed_value_lookup returns DIRECT referents only (predicate material);
      closure stays routing-evidence.

Uses the on-disk dev artifacts (semantic model, relationship graph, value referents).
Run: python tests/test_qsr_resolution.py  (from repo root).
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


# ── F1: schema-vocab segmentation ────────────────────────────────────────────
from semantic.name_tokens import table_tokens, segment_token, token_table_idf

check("fused name segments (paymenttransaction)",
      table_tokens("accounts_paymenttransaction", SM) >= {"account", "payment", "transaction"},
      str(table_tokens("accounts_paymenttransaction", SM)))
check("fused name segments (assetverificationdocumenttype)",
      {"verification", "document"} <= table_tokens("assets_assetverificationdocumenttype", SM))
for w in ("transaction", "advertisement", "communication", "listing", "reminder"):
    check(f"no false split: {w}", segment_token(w, SM) == (w,), str(segment_token(w, SM)))

idf = token_table_idf(SM)
check("idf: entity word outranks app prefix",
      idf.get("payment", 0) > idf.get("asset", 1), f"payment={idf.get('payment')} asset={idf.get('asset')}")

# ── F4: typed value resolution + FK closure ──────────────────────────────────
from query.resolution import (resolve, value_referents, closed_value_tables,
                              typed_value_lookup, domain_via)

vr = value_referents("captured")
check("captured has direct referents", bool(vr["direct"]))
check("captured FK-closes to payment transactions",
      any(r["table"] == "accounts_paymenttransaction" and r["column"] == "payment_status_id"
          for r in vr["closed"]), str(vr["closed"][:2]))
check("captured closure is precise (no mode_of_payment pollution)",
      not any(r["column"] == "mode_of_payment_id" for r in vr["closed"]))

check("completed has NO value referent (grounded-clarify case)",
      not value_referents("completed")["direct"] and not value_referents("completed")["closed"])

check("typed_value_lookup is direct-only (predicate safety)",
      all(t == "list_of_values_listofvalue"
          for t, _, _, _ in typed_value_lookup()("captured")),
      str(typed_value_lookup()("captured")[:2]))

check("closed_value_tables exposes routing evidence",
      "accounts_paymenttransaction" in closed_value_tables("captured"))

dom = domain_via("accounts_paymenttransaction", "payment_status_id")
check("domain_via returns FK-scoped label set",
      set(dom) == {"authorized", "cancelled", "captured"}, str(dom))

# ── resolve(): typing on the trigger query ───────────────────────────────────
res = {tr.span: tr for tr in
       resolve("Which category contributes the highest value among all captured payments?", SM)}
check("'which' typed grammar", res["which"].grammar)
check("'payments' typed entity → paymenttransaction",
      any(t == "accounts_paymenttransaction" for t, _ in res["payments"].entities))
check("'value' keeps its measure reading despite grammar flag",
      res["value"].grammar and bool(res["value"].measures))
check("'category' typed dimension", bool(res["category"].dimensions))
check("'captured' value span resolved",
      bool(res["captured"].values["direct"]) and bool(res["captured"].values["closed"]))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all QSR resolution checks passed")
