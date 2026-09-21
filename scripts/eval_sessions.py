"""Multi-turn session eval — does a conversation keep its thread? (Checkpoint B.3 / C.8)

WHAT THIS MEASURES
------------------
Every other battery in scripts/ asks ONE question per run. This one drives a real
multi-turn conversation through the supervisor graph (chatbot.run.run_chat_turn, the same
entry point apps/chat/services.py uses per user turn) and asserts, per turn:

  route_source    "coordinator" | "pin" on turn 1, "inherited" on follow-ups — i.e. the
                  session stayed on the source that answered the previous turn instead of
                  re-deciding from a message ("of those, only the Mumbai ones") that
                  carries almost no routing signal on its own.
  scope           the source ids the turn actually executed against.
  turn_slm_calls  how many SLM calls the turn spent. A follow-up that has to re-route and
                  re-classify costs several; the target is ≤ 1 (the prose summary).
  status          answered / clarify / refused.
  expect_text     optional: a substring the answer must contain (used for the document
                  source, where the assertion is about content, not SQL shape).

Turn state is real session state: the same `session_id` is passed to every turn, so the
LangGraph checkpointer restores the frame exactly as it would for a user typing into one
chat window.

SLM call counting reads the engine's own per-query trace log (the `slm_calls` the compact
record now carries — see veda/explain.py::compact), matched by the trace written during
that turn. It is therefore the SAME number scripts/slm_purpose_report.py reports, not a
second, separately-derived estimate.

USAGE (inside the API container — chatbot/ runs in the api tier, and only the api tier
can reach the inference service over veda_net):
    docker compose exec api python /app/scripts/eval_sessions.py
    Flags: --script s2|s4|s5|s3|cross|all   --sources 2,3,4,5   -v

Exit 1 on any failed assertion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "/app")

# Django must be initialised BEFORE apps.query.scope is imported, or that import raises
# and run_script's try/except silently falls back to `profiles = {}`. Empty profiles are
# not a harmless degradation: source_profiles is what tells the engine a source is
# document/datalake rather than relational (see chatbot/run.py::run_chat_turn), and
# without it the engine plans SQL against the primary source's schema and every
# non-pinned turn fails with "schema model isn't available" — which is exactly what this
# script reported for all of s3 before the setup call was added.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
try:
    import django
    django.setup()
except Exception as _exc:                       # pragma: no cover - dev convenience
    print(f"WARNING: django.setup() failed ({type(_exc).__name__}: {_exc}) — "
          f"source profiles will be empty and non-pinned turns will fail")


# ── the scripts ───────────────────────────────────────────────────────────────
# Each entry: (message, expectations). `first=True` marks the topic-opening turn, which
# must route via the coordinator; every later turn in the script is a follow-up and must
# inherit. `expect_source` is the source id the turn must execute on.
def T(msg, expect_source=None, first=False, expect_text=None, max_slm=None, note="",
      expect_scope_contains=None):
    """expect_source: the source that ANSWERED must be this one.
    expect_scope_contains: this source must be IN the scope the turn was allowed to use.

    The distinction matters once scope widening exists. Asserting that the executed SCOPE
    was exactly [4] conflates two different things — "the answer came from source 4" and
    "the other sources were withheld" — and only the first is a correctness property. A
    turn that is allowed to see 2 and 4 and answers from 4 is right; forbidding it from
    seeing 2 is what made cross-source follow-ups answer the wrong question."""
    return {"msg": msg, "expect_source": expect_source, "first": first,
            "expect_text": expect_text, "max_slm": max_slm, "note": note,
            "expect_scope_contains": expect_scope_contains}


SCRIPTS = {
    # 10-turn drill-down on the relational source
    "s2": [
        T("how many properties are there", expect_source=2, first=True),
        T("how many of those are in Mumbai", expect_source=2),
        T("of those, which are for sale", expect_source=2),
        T("break that down by property type", expect_source=2),
        T("show me just the top 3", expect_source=2),
        T("what is the average rent of those", expect_source=2),
        T("go back", expect_source=2),
        T("group by city instead", expect_source=2),
        T("only Pune", expect_source=2),
        T("how many of those have parking", expect_source=2),
    ],
    # 10-turn drill-down on the maintenance datalake
    "s4": [
        T("how many maintenance records are there", expect_source=4, first=True),
        T("how many of those are repairs", expect_source=4),
        T("break that down by status", expect_source=4),
        T("what is the total amount for those", expect_source=4),
        T("show the top 3 by amount", expect_source=4),
        T("go back", expect_source=4),
        T("which vendors handled them", expect_source=4),
        T("only the ones in Kochi", expect_source=4),
        T("what is their average rating", expect_source=4),
        T("who has the highest rating", expect_source=4),
    ],
    # 10-turn drill-down on the amenities catalog
    "s5": [
        T("list all amenities in the catalog", expect_source=5, first=True),
        T("how many are there", expect_source=5),
        T("group them by category", expect_source=5),
        T("only the Security ones", expect_source=5),
        T("what is their monthly fee", expect_source=5),
        T("which is the most expensive", expect_source=5),
        T("go back", expect_source=5),
        T("show the average fee per category", expect_source=5),
        T("only categories above 500", expect_source=5),
        T("how many amenities is that", expect_source=5),
    ],
    # document source: the drill is through RAG, so assertions are on TEXT
    "s3": [
        T("what does the employee handbook say about leave", expect_source=3, first=True,
          expect_text=["leave", "holiday", "absence"]),
        T("what about the notice period", expect_source=3),
        T("does it mention working hours", expect_source=3),
        T("what does the maintenance policy say about response times", expect_source=3),
        T("and for emergency repairs", expect_source=3),
        T("what are the escalation rules", expect_source=3),
        T("go back", expect_source=3),
        T("what are the payment terms in the MSA", expect_source=3),
        T("who are the parties to it", expect_source=3),
        T("what is the contract duration", expect_source=3),
    ],
    # starts on 4, joins to 2, drills, returns
    #
    # NOTE on turn 1: it is deliberately the PLAINEST question this source answers. The
    # opening turn's only job is to establish a frame for the rest of the script to drill
    # from; if it refuses, every later turn inherits nothing and the script measures the
    # opening question instead of the session. The original opener here ("how many
    # maintenance records are repairs" — a filtered count, a documented gap on this
    # source) refused, and all ten turns failed for that one reason.
    "cross": [
        T("how many maintenance records are there", expect_source=4, first=True),
        # These turns NAME an entity that lives in source 2 (properties, property type,
        # city), so the executed scope must WIDEN to include it. Asserting
        # expect_scope_contains rather than expect_source is the point of this script:
        # with the turns left unasserted, a run that stayed on source 4 and answered a
        # DIFFERENT question scored as a pass — which is exactly what happened, and is
        # the failure mode session-stickiness introduces if it never lets go.
        # expect_text as well as expect_scope_contains: the scope assertion proves the
        # turn COULD see source 2, but not that the answer is about properties. Without a
        # content check these two turns passed while answering from source 4's own rows —
        # "the total amount for maintenance records is 10,490" is a fluent answer to a
        # question nobody asked, and "8 maintenances are recorded in Mumbai" is asserted
        # off a table that has no city column at all.
        T("which properties do they belong to", expect_scope_contains=2,
          expect_text=["propert", "asset"],
          note="crosses to source 2 via maintenance.asset_id -> assets_asset.id"),
        # Two acceptable outcomes, one unacceptable one. City lives in source 2, so this
        # turn must either answer ABOUT properties/assets (a real cross-source answer) or
        # REFUSE. What it must never do is what it used to: return source 4's unfiltered
        # count dressed as a filtered one ("8 maintenances are recorded in Mumbai"), which
        # contains none of these strings.
        T("how many of those are in Mumbai", expect_scope_contains=2,
          expect_text=["propert", "asset", "couldn't", "could not", "cannot", "unable",
                       "no data", "not find", "don't have"],
          note="city lives in source 2; answering from source 4 alone is wrong, and "
               "refusing is an acceptable outcome — inventing a count is not"),
        T("break that down by property type", expect_scope_contains=2),
        T("what is the total maintenance amount for those", expect_source=4,
          note="amount is back in source 4"),
        T("go back"),
        T("show the vendors involved", expect_source=4),
        T("only the ones rated above 4", expect_source=4),
        T("which cities are they in", expect_source=4, note="vendors.city is in source 4"),
        T("how many maintenance records did they handle", expect_source=4),
    ],
}


def _trace_len(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# "Overhead" = the SUPERVISOR's own classify/route calls, which is what M4 removes. An
# engine-side call that does work the user sees (the prose summary) or an optional engine
# feature (answer-entity discovery) is not session overhead, and counting them together
# made the budget unfalsifiable — a turn could hit 2 for a reason the memory layer has no
# control over.


def _slm_calls_since(path, offset):
    """(total_calls, purposes) recorded in trace records written after `offset` lines."""
    n, purposes = 0, []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for c in (rec.get("slm_calls") or []):
                    n += 1
                    purposes.append(c.get("purpose") or "?")
    except OSError:
        pass
    return n, purposes


def run_script(name, turns, source_ids, trace_path, verbose=False, pin_first=False):
    from chatbot.run import run_chat_turn
    try:
        from apps.query.scope import source_profiles_for
        profiles = source_profiles_for(source_ids)
    except Exception as exc:
        print(f"  WARNING: source profiles unavailable ({type(exc).__name__}: {exc}) — "
              f"non-pinned turns will fail with 'schema model isn't available'")
        profiles = {}
    if not profiles:
        print("  WARNING: source profiles are EMPTY — the engine cannot tell a document "
              "or datalake source from a relational one")

    session = f"evalsess-{name}-{uuid.uuid4().hex[:8]}"
    failures, rows = [], []

    for i, t in enumerate(turns, 1):
        events = []

        def _on_event(phase, message, extra):
            events.append((phase, message, extra or {}))

        before = _trace_len(trace_path)
        t0 = time.time()
        # --pin-first pins ONLY the opening turn, never the follow-ups. This deployment
        # cannot currently answer an UNPINNED multi-source question from the chat path at
        # all (the engine reports "schema model isn't available" before routing gets a
        # say — reproduced with routing cards both on and off, so it predates them), which
        # would fail every turn for a reason that has nothing to do with session memory.
        # Pinning turn 1 gives the session a real frame to inherit FROM, so what the rest
        # of the script measures is exactly the mechanism under test: do turns 2..n stay
        # on that scope without being told to. Pinning any later turn would mask it.
        _pin = (t["expect_source"] if (pin_first and i == 1 and t["expect_source"]) else None)
        try:
            res = run_chat_turn(t["msg"], session, tenant="default",
                                source_id=_pin,
                                source_ids=([int(_pin)] if _pin else list(source_ids)),
                                source_profiles=profiles, on_event=_on_event)
        except Exception as exc:
            failures.append(f"[{name} t{i}] raised {type(exc).__name__}: {exc}")
            rows.append((i, t, "EXC", None, None, 0, time.time() - t0, ""))
            continue
        secs = time.time() - t0
        slm, slm_purposes = _slm_calls_since(trace_path, before)
        # The supervisor's own calls come from run_chat_turn (chatbot/llm.py), NOT from
        # the engine trace — the two tiers use separate SLM clients, so the trace-derived
        # count above is engine-side only and would silently report 0 overhead forever.
        overhead = list(res.get("supervisor_slm_purposes") or [])

        # route_source is emitted by chatbot/nodes.py::call_engine_node on EVERY turn
        route_source = next((e[2].get("route_source") for e in reversed(events)
                             if e[2].get("route_source")), None)
        # The scope the TURN WAS ALLOWED, from the chat tier's own route_source event —
        # not the last source_ids seen. The engine emits its own route event after the
        # coordinator narrows, so reading "the last one" reported the coordinator's
        # DECISION and made it impossible to tell "the turn could not see source 2" from
        # "the turn could see it and routing chose not to use it". Those are different
        # failures: the first is this tier's bug, the second is routing's.
        scope = next((e[2].get("source_ids") for e in reversed(events)
                      if e[2].get("route_source") and e[2].get("source_ids")), None)
        if scope is None:
            scope = next((e[2].get("source_ids") for e in reversed(events)
                          if e[2].get("source_ids")), None)
        status = res.get("status")
        # the source that actually produced the answer (engine payload), not the scope
        answered_by = (res.get("engine_result") or {}).get("source_id")
        answer = str(res.get("answer_text") or "")

        # ---- assertions ----
        if t["first"]:
            if route_source == "inherited":
                failures.append(f"[{name} t{i}] topic-opening turn inherited scope "
                                f"(should route via the coordinator)")
        else:
            if route_source != "inherited":
                failures.append(f"[{name} t{i}] follow-up did NOT inherit scope "
                                f"(route_source={route_source!r}) — the session lost its thread")
        if t["expect_source"] is not None and status == "answered":
            if answered_by is None or int(answered_by) != int(t["expect_source"]):
                failures.append(f"[{name} t{i}] answered by source {answered_by} != "
                                f"expected {t['expect_source']} (scope was {scope})")
        if t.get("expect_scope_contains") is not None:
            _want = int(t["expect_scope_contains"])
            if not scope or _want not in [int(s) for s in scope]:
                failures.append(
                    f"[{name} t{i}] executed scope {scope} does not include source {_want} — "
                    f"this turn names an entity that lives there, so an answer from the "
                    f"inherited scope alone is answering a different question")
        if t["expect_text"]:
            low = answer.lower()
            if not any(s.lower() in low for s in t["expect_text"]):
                failures.append(f"[{name} t{i}] answer contained none of {t['expect_text']}")
        # The budget assertion is on CLASSIFY/ROUTE overhead, which is what the IR stack
        # removes; total calls are still reported so a regression anywhere is visible.
        # The contract is "0 classify calls WHEN THE RULE LAYER RESOLVES THE TURN, at most
        # one otherwise" (C.3/C.4: the SLM runs only on `ambiguous`). Asserting 0 on every
        # follow-up contradicted that design — it failed turns the rules are supposed to
        # hand to the classifier — so the assertion is the cap, and the share of turns
        # needing NO call is reported as the metric instead.
        if not t["first"] and len(overhead) > 1:
            failures.append(f"[{name} t{i}] follow-up spent {len(overhead)} classify/route "
                            f"calls {overhead} — the budget is at most one "
                            f"(total engine calls this turn: {slm_purposes})")
        cap = t["max_slm"] if t["max_slm"] is not None else None
        if cap is not None and slm > cap:
            failures.append(f"[{name} t{i}] turn_slm_calls={slm} > {cap} {slm_purposes}")

        rows.append((i, t, status, route_source, scope, slm, secs, answer, overhead))
        if verbose:
            print(f"  t{i:<2} {status:<10} route={str(route_source):<12} "
                  f"by={answered_by} scope={scope} "
                  f"slm={slm}{('/' + str(len(overhead)) + 'ovh') if overhead else ''} "
                  f"{secs:>5.1f}s  {t['msg'][:44]}")
            print(f"       → {answer[:110]}")

    return rows, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", default="all",
                    choices=["all"] + sorted(SCRIPTS), nargs="+")
    ap.add_argument("--sources", default="2,3,4,5")
    ap.add_argument("--trace", default="/app/veda_core/logs/explain_trace.jsonl")
    ap.add_argument("--pin-first", action="store_true",
                    help="pin only turn 1 to its expected source, so follow-up inheritance "
                         "is measured in isolation from unpinned multi-source routing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    names = sorted(SCRIPTS) if "all" in args.script else args.script
    source_ids = [int(s) for s in args.sources.split(",") if s.strip()]

    all_fail, summary = [], []
    for name in names:
        print(f"\n=== session script: {name} ({len(SCRIPTS[name])} turns) ===")
        rows, failures = run_script(name, SCRIPTS[name], source_ids, args.trace,
                                    verbose=args.verbose, pin_first=args.pin_first)
        all_fail += failures
        inherited = sum(1 for r in rows if r[3] == "inherited")
        answered = sum(1 for r in rows if r[2] == "answered")
        deterministic = sum(1 for r in rows[1:] if not r[8])   # follow-ups with 0 overhead
        slms = [r[5] for r in rows[1:]]
        med = sorted(slms)[len(slms) // 2] if slms else 0
        summary.append((name, len(rows), answered, inherited, deterministic, len(failures)))
        print(f"  answered {answered}/{len(rows)}   inherited {inherited}/{max(0, len(rows) - 1)}"
              f"   no-classify follow-ups {deterministic}/{max(0, len(rows) - 1)}"
              f"   median SLM {med}   failures {len(failures)}")

    print("\n" + "=" * 78)
    for name, n, answered, inh, det, nf in summary:
        print(f"  {name:<8} turns {n:<3} answered {answered:<3} inherited {inh:<3} "
              f"no_classify {det:<3} failures {nf}")
    if all_fail:
        print(f"\n{len(all_fail)} FAILURE(S):")
        for f in all_fail[:40]:
            print(f"  - {f}")
        return 1
    print("\nAll session scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
