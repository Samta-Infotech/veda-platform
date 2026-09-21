"""Routing eval — labelled questions vs. the source(s) that must answer them (Checkpoint B.4).

THE GATE this exists to enforce
-------------------------------
Routing cards (ingestion/routing_card.py) put a description of what each source IS into
the boundary SLM's prompt, and the plan makes a ROUTED/SINGLE decision authoritative only
if that change does not make routing worse. "Worse" needs a number, so this measures the
same labelled set twice in one run — cards OFF, then cards ON (ROUTING_CARDS_ENABLED) —
and reports precision/recall for each.

It calls `plan_route` DIRECTLY rather than asking whole questions through the chat API.
That is deliberate: routing is the thing under test, and a full answer folds in retrieval,
planning, SQL generation and summarisation, any of which can fail for reasons that have
nothing to do with which source was chosen. A routing regression hidden behind an
unrelated planner failure is exactly what a routing gate must not allow.

SCORING
-------
Per question, the decision's source set is compared to the labelled expectation:
  precision = |chosen ∩ expected| / |chosen|
  recall    = |chosen ∩ expected| / |expected|
reported as the mean over questions, plus exact-set accuracy (the strict measure: the
decision named exactly the right sources, no more and no fewer). A NO_MATCH/clarify is
scored as an empty choice — precision 1.0 by convention, recall 0.0 — so refusing
everything cannot win on precision alone. `expect_multi` questions additionally check
that the MODE was MULTI, because naming both sources while deciding SINGLE would execute
against one of them.

THE QUESTION SET
----------------
50 questions: 10 for each registered source (2/3/4/5) and 10 cross-source. The plan said
40 (10 per source + 10 cross-source), which assumed three sources; this deployment has
four, so per-source coverage is kept uniform rather than dropping a source.

Every question is written from what the source ACTUALLY holds, read off its own routing
card and semantic model — not invented. Source 2 is homzhub (relational: assets, users,
leases, invoices, payments, tickets, amenities); source 3 is the document set (employee
handbook, maintenance policy, site notes, MSA, readme); source 4 is the maintenance
datalake (maintenance: asset_id/category/status/amount; vendors: city/rating); source 5
is the amenities catalog (amenity_name/category/monthly_fee). The cross-source questions
are built on the joins the cards themselves declare — maintenance.asset_id ↔ assets_asset,
vendors.city ↔ assets_asset.city_name, amenities_catalog.amenity_name ↔ assets_amenity.name
— so a MULTI expectation is always structurally satisfiable, never wishful.

USAGE (inside the inference container; real retrieval, real SLM):
    cd /app/veda_core && python /app/scripts/eval_routing_cards.py
    Flags: --sources 2,3,4,5  --mode both|on|off  --only single|cross  -v

Exit 1 when cards-ON is worse than cards-OFF on exact-set accuracy OR mean recall.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/app")
_ENGINE_DIR = "/app/veda_core"
if os.path.isdir(_ENGINE_DIR):
    if _ENGINE_DIR in sys.path:
        sys.path.remove(_ENGINE_DIR)
    sys.path.insert(0, _ENGINE_DIR)


def Q(q, expect, multi=False, note=""):
    """expect: the source ids that must answer. multi: the decision must be MULTI."""
    return {"q": q, "expect": {str(s) for s in expect}, "multi": multi, "note": note}


# ── 10 per source, phrased in that source's own nouns ────────────────────────
QUESTIONS = [
    # ---- source 2: homzhub relational ----
    Q("how many properties are there", [2]),
    Q("how many users are registered", [2]),
    Q("list the assets in Mumbai", [2]),
    Q("how many payment transactions were made", [2]),
    Q("what is the total paid amount on user invoices", [2]),
    Q("show lease negotiations by status", [2]),
    Q("how many tickets are open", [2]),
    Q("list properties by project name", [2]),
    Q("which currencies do we support", [2]),
    Q("how many assets per city", [2]),

    # ---- source 3: document set ----
    Q("what does the employee handbook say about leave policy", [3]),
    Q("what is the maintenance policy for emergency repairs", [3]),
    Q("summarise the MSA for green tower", [3]),
    Q("what are the notice period rules in the handbook", [3]),
    Q("what does the maintenance policy say about response times", [3]),
    Q("what is in the site notes", [3]),
    Q("what are the payment terms in the MSA", [3]),
    Q("what does the handbook say about working hours", [3]),
    Q("are there any escalation rules documented for maintenance", [3]),
    Q("what does the readme describe", [3]),

    # ---- source 4: maintenance datalake ----
    Q("how many maintenance records are repairs", [4]),
    Q("which vendor has the highest rating", [4]),
    Q("show vendors with a rating above 4.2", [4]),
    Q("total maintenance amount by category", [4]),
    Q("how many maintenance tickets are unpaid", [4]),
    Q("list vendors in Kochi", [4]),
    Q("what is the average maintenance amount", [4]),
    Q("how many maintenance records per status", [4]),
    Q("which city has the most vendors", [4]),
    Q("show the maintenance tickets with their ids", [4]),

    # ---- source 5: amenities catalog ----
    Q("show amenities in the Security category", [5]),
    Q("what is the monthly fee for each amenity", [5]),
    Q("list all amenities in the catalog", [5]),
    Q("how many amenities are there per category", [5]),
    Q("which amenity has the highest monthly fee", [5]),
    Q("what categories of amenities exist", [5]),
    Q("total monthly fee across the amenities catalog", [5]),
    Q("show amenities costing more than 500 per month", [5]),
    Q("what is the average amenity monthly fee", [5]),
    Q("list amenity names and their categories", [5]),

    # ---- cross-source (built on the joins the cards declare) ----
    Q("show maintenance records with the property they belong to", [4, 2], multi=True,
      note="maintenance.asset_id -> assets_asset.id"),
    Q("which properties have open maintenance tickets", [2, 4], multi=True,
      note="asset filter needs source 2, ticket status needs source 4"),
    Q("total maintenance amount per property city", [4, 2], multi=True,
      note="amount in 4, city in 2"),
    Q("list vendors in cities where we have properties", [4, 2], multi=True,
      note="vendors.city -> assets_asset.city_name"),
    Q("which of our amenities appear in the amenities catalog", [2, 5], multi=True,
      note="assets_amenity.name -> amenities_catalog.amenity_name"),
    Q("monthly fee for the amenities our properties offer", [2, 5], multi=True,
      note="fee in 5, property amenities in 2"),
    Q("compare maintenance spend against property rent", [4, 2], multi=True),
    Q("how many properties have both maintenance records and amenities", [2, 4], multi=True,
      note="deliberately hard: 2+4 (+5 would also be defensible)"),
    Q("which vendors serve properties in Mumbai", [4, 2], multi=True),
    Q("amenity fees for properties with open maintenance tickets", [2, 5], multi=True,
      note="deliberately hard: spans 2/4/5; 2+5 is the minimum defensible set"),
]


def _score(chosen: set, expected: set):
    if not chosen:
        return 1.0, 0.0           # refusal: vacuously precise, recalls nothing
    hit = len(chosen & expected)
    return hit / len(chosen), hit / len(expected)


def run_pass(questions, source_ids, label, verbose=False):
    from query.source_coordinator import plan_route

    precs, recs, exact, mode_ok, rows = [], [], 0, 0, []
    t0 = time.time()
    for i, item in enumerate(questions, 1):
        try:
            d = plan_route(item["q"], source_ids)
            chosen = {str(s) for s in (d.source_ids or [])}
            status, mode, method = d.status, d.mode, d.decision_method
        except Exception as exc:
            chosen, status, mode, method = set(), f"ERROR:{type(exc).__name__}", "-", "-"
        p, r = _score(chosen, item["expect"])
        precs.append(p)
        recs.append(r)
        is_exact = chosen == item["expect"]
        exact += int(is_exact)
        if item["multi"]:
            mode_ok += int(mode == "MULTI")
        rows.append((item, chosen, status, mode, method, p, r, is_exact))
        if verbose:
            flag = "OK " if is_exact else "   "
            print(f"  {flag}[{i:>2}] want={sorted(item['expect'])} got={sorted(chosen)} "
                  f"{status}/{mode} ({method})  {item['q'][:58]}")

    n = len(questions)
    n_multi = sum(1 for q in questions if q["multi"])
    res = {
        "label": label,
        "n": n,
        "precision": sum(precs) / n if n else 0.0,
        "recall": sum(recs) / n if n else 0.0,
        "exact": exact / n if n else 0.0,
        "multi_mode": (mode_ok / n_multi) if n_multi else None,
        "secs": time.time() - t0,
        "rows": rows,
    }
    return res


def _print(res):
    mm = "  n/a" if res["multi_mode"] is None else f"{res['multi_mode']:>5.1%}"
    print(f"  {res['label']:<12} exact {res['exact']:>5.1%}   precision {res['precision']:>5.1%}   "
          f"recall {res['recall']:>5.1%}   MULTI-mode {mm}   ({res['secs']:.0f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="2,3,4,5")
    ap.add_argument("--mode", choices=["both", "on", "off"], default="both")
    ap.add_argument("--only", choices=["all", "single", "cross"], default="all")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    source_ids = [s.strip() for s in args.sources.split(",") if s.strip()]
    qs = QUESTIONS
    if args.only == "single":
        qs = [q for q in qs if not q["multi"]]
    elif args.only == "cross":
        qs = [q for q in qs if q["multi"]]

    print(f"\nrouting eval — {len(qs)} labelled questions over sources {source_ids}\n")

    results = {}
    for state, flag in (("off", "0"), ("on", "1")):
        if args.mode not in ("both", state):
            continue
        # The flag is read at module import (config.py), so it has to be set before the
        # engine config is imported AND re-read between passes — reload config and drop
        # the cached routing module so the second pass genuinely sees the new value.
        os.environ["ROUTING_CARDS_ENABLED"] = flag
        import importlib
        import config as _cfg
        importlib.reload(_cfg)
        for m in ("query.routing_slm", "query.source_coordinator"):
            if m in sys.modules:
                importlib.reload(sys.modules[m])
        results[state] = run_pass(qs, source_ids, f"cards {state.upper()}",
                                  verbose=args.verbose)
        _print(results[state])

    if "on" in results and "off" in results:
        on, off = results["on"], results["off"]
        d_exact = on["exact"] - off["exact"]
        d_rec = on["recall"] - off["recall"]
        print(f"\n  delta (ON - OFF): exact {d_exact:+.1%}   recall {d_rec:+.1%}   "
              f"precision {on['precision'] - off['precision']:+.1%}")
        if d_exact < 0 or d_rec < 0:
            print("\nFAIL: cards made routing worse — SINGLE must NOT be made authoritative "
                  "on this evidence.")
            return 1
        print("\nPASS: cards are at or above the no-card baseline on exact-set and recall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
