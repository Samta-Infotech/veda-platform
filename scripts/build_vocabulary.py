"""Build / rebuild the per-source BUSINESS VOCABULARY (meaning-first pass, Stage 1) offline.

Same artifacts the L5 `vocabulary` stage publishes on ingest — entity cards, value and
measure glossaries, synthetic questions (+ BGE-M3 embeddings), and the routing card
rebuilt from the cards — for sources that were ingested before the stage existed.

Usage (inside the inference container; the semantic model of sources 3/4/5 lives only
in Redis, so this needs the container):
    cd /app/veda_core && python /app/scripts/build_vocabulary.py --sources 2,4,5
        [--no-slm] [--no-embed] [--slm-budget 60] [--paraphrase-cards 25] [--live-values]
        [--print TABLE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="2,4,5")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--no-slm", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--slm-budget", type=int, default=None,
                    help="SLM-draft only the N most important tables (rest deterministic)")
    ap.add_argument("--paraphrase-cards", type=int, default=25)
    ap.add_argument("--live-values", action="store_true",
                    help="sample DISTINCT values from the source for CATEGORY columns with no offline domain")
    ap.add_argument("--print", dest="show", action="append", default=[])
    ap.add_argument("--reuse-cards", action="store_true",
                    help="reuse SLM card drafts checkpointed by a previous run (veda_entity_cards.draft.json)")
    a = ap.parse_args()

    from veda_core.context import RequestContext, set_context
    from veda.runtime import _load_one_sm
    from ingestion.vocabulary import publish_vocabulary, CARDS_ARTIFACT
    from config import source_artifact_path

    rc = 0
    for sid in [int(s) for s in a.sources.split(",") if s.strip()]:
        t0 = time.time()
        set_context(RequestContext(source_id=sid, tenant=a.tenant, source_ids=(sid,), cache_back=False))
        sm = _load_one_sm(sid, a.tenant)
        print(f"== source {sid}: {len((sm or {}).get('tables') or {})} tables", flush=True)
        try:
            r = publish_vocabulary(sid, a.tenant, sm, use_slm=not a.no_slm, embed=not a.no_embed,
                                   live_values=a.live_values, paraphrase_cards=a.paraphrase_cards,
                                   slm_budget=a.slm_budget, verbose=True,
                                   reuse_draft=a.reuse_cards)
        except Exception as e:
            print(f"!! source {sid} failed: {type(e).__name__}: {e}")
            rc = 1
            continue
        print(json.dumps({k: v for k, v in r.items() if k != "paths"}, default=str))
        for k, p in (r.get("paths") or {}).items():
            print(f"   {k:<16} {p}")
        cards = json.load(open(source_artifact_path(CARDS_ARTIFACT, sid, a.tenant))) if r.get("paths") else {}
        for t in a.show:
            if t in cards:
                print(json.dumps(cards[t], indent=1))
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
