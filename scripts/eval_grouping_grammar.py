"""Labelled precision/recall check for QUERY_GRAMMAR["grouping"] (veda_core/config.py),
consumed by veda/planning.py::grouped_mode.

Exists because docs/backlog/query-engine-open-items.md's P1-2 write-up explicitly flags
this word list's narrow phrasing coverage ("broken down by" isn't recognized even though
"breakdown" and "per" are) as a real, separate gap — and explicitly warns against a blind
addition: "retuning a grammar classifier without a labelled precision/recall check of its
own would be exactly the kind of blind change this whole exercise was trying to avoid."

This script is that check. It runs grouped_mode() over a hand-labelled set
(evaluation/grouping_grammar_labels.jsonl — true positives that SHOULD trigger a grouped
breakdown, and true negatives/distractors that share surface words ("by") but must not),
first against today's word list (BASELINE), then against a candidate-expanded list
(CANDIDATE) — never mutating config.py itself. Only once CANDIDATE shows recall gains with
NO precision loss (zero new false positives on the distractor set) is the expansion safe
to actually apply to QUERY_GRAMMAR["grouping"].

Usage (pure language-layer code, no DB/engine — runs on host or in-container):
    PYTHONPATH=veda_core:. python scripts/eval_grouping_grammar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "veda_core"))  # veda_core LAST-inserted = first on sys.path
                                              # (CLAUDE.md gotcha #1: else Django's own
                                              # `config` package shadows veda_core/config.py)

# Candidate additions being evaluated for veda_core/config.py::QUERY_GRAMMAR["grouping"].
# Each is a synonym of the already-accepted "grouped by" / "breakdown" — chosen for low
# collision risk with the negative/distractor set (generic bare "by" is deliberately never
# added — that's what would catch "sorted by"/"increased by"/"backed by"/etc.).
CANDIDATE_ADDITIONS = [
    "broken down by",
    "broken out by",
    "break down",       # covers "break down X by Y" / "can you break down ... by"
    "split by",
    "segmented by",
    "categorized by",
]


def _score(rows, grouping_words):
    import config as _cfg
    from veda.planning import grouped_mode

    original = _cfg.QUERY_GRAMMAR["grouping"]
    _cfg.QUERY_GRAMMAR["grouping"] = grouping_words
    try:
        tp = fp = tn = fn = 0
        misses = []
        for row in rows:
            predicted = grouped_mode(row["query"]) is not None
            actual = row["label"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
                misses.append(("FALSE POSITIVE", row))
            elif not predicted and actual:
                fn += 1
                misses.append(("FALSE NEGATIVE", row))
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision, "recall": recall, "misses": misses}
    finally:
        _cfg.QUERY_GRAMMAR["grouping"] = original


def main() -> int:
    rows = []
    with open(_REPO / "evaluation" / "grouping_grammar_labels.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    import config as _cfg
    baseline_words = list(_cfg.QUERY_GRAMMAR["grouping"])
    candidate_words = baseline_words + CANDIDATE_ADDITIONS

    for label, words in (("BASELINE", baseline_words), ("CANDIDATE", candidate_words)):
        r = _score(rows, words)
        print(f"[{label}] words={words}")
        print(f"  tp={r['tp']} fp={r['fp']} tn={r['tn']} fn={r['fn']}  "
              f"precision={r['precision']}  recall={r['recall']}")
        for kind, row in r["misses"]:
            print(f"    {kind}: {row['query']!r}  ({row['note']})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
