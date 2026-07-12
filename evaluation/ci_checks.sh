#!/bin/sh
# =============================================================================
# evaluation/ci_checks.sh — Phase D gate (ARCHITECTURE_ROOT_CAUSE_PLAN.md).
# Chains the harness checks a change must pass before it ships:
#   1. QSR unit acceptance   (tokenizer, closure, typed resolution)
#   2. Two-seed determinism  (anchor decisions independent of PYTHONHASHSEED)
#   3. Golden recheck        (no wrong-table ANSWERS in the newest suite results)
#   4. Latency assertions    (SLO: p50 < 5s answered fast-lane; see latency_assert)
# Usage: sh evaluation/ci_checks.sh [results.jsonl]
#   With no argument, uses the newest evaluation/nl_query_suite_*_results.jsonl.
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "── 1/4 QSR acceptance"
python3 tests/test_qsr_resolution.py > /dev/null && echo "ok"

echo "── 2/4 determinism (two-seed)"
python3 evaluation/determinism_check.py

RESULTS="$1"
if [ -z "$RESULTS" ]; then
    RESULTS=$(ls -t evaluation/nl_query_suite_*_results.jsonl 2>/dev/null | head -1)
fi
echo "── 3/4 golden recheck ($RESULTS)"
python3 evaluation/nl_query_suite.py --recheck "$RESULTS"
# recheck prints failures; enforce zero:
FAILS=$(python3 evaluation/nl_query_suite.py --recheck "$RESULTS" | tail -1 | sed 's/.* \([0-9]*\) GOLDEN-FAIL/\1/')
[ "$FAILS" = "0" ] || { echo "GOLDEN-FAILS present: $FAILS"; exit 1; }

echo "── 4/4 latency assertions"
python3 evaluation/latency_assert.py "$RESULTS"

echo "ALL CI CHECKS PASSED"
