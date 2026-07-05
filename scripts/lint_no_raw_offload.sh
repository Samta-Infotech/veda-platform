#!/bin/bash
# Lint gate (migration plan §4.1): offloaded work must carry the ambient
# (source, tenant) context, or a future code path silently reads a leaked/unset
# tenant. Two rules over inference/ and veda_core/:
#
#  1. HARD BAN: bare async `run_in_threadpool(` — the async hot path must use
#     inference.concurrency.run_in_threadpool_with_context (the one wrapped primitive).
#  2. CONTEXT-CARRYING: any file using ThreadPoolExecutor / .submit / .map must
#     also carry context (run_in_threadpool_with_context, or the sync helpers
#     veda_core.context.with_context / set_context). A bare offload fails.
#
# CI runs this; non-zero exit fails the build. Allowlist: the primitive's own file.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

# Rule 1 — bare async run_in_threadpool (not the wrapper).
bare_async=$(grep -rnE 'run_in_threadpool\(' inference veda_core --include='*.py' 2>/dev/null \
  | grep -v 'inference/concurrency.py' \
  | grep -v 'run_in_threadpool_with_context')
if [ -n "$bare_async" ]; then
  echo "LINT FAIL (rule 1) — bare run_in_threadpool; use run_in_threadpool_with_context (§4.1):"
  echo "$bare_async"; fail=1
fi

# Rule 2 — ThreadPoolExecutor without context-carrying in the same file.
# (Match the pool constructor only: any real .submit/.map needs one, and a bare
# `.map(` would false-match JS/pandas/list.map — e.g. JS in an HTML template.)
for f in $(grep -rlE 'ThreadPoolExecutor\(' inference veda_core --include='*.py' 2>/dev/null); do
  [ "$f" = "inference/concurrency.py" ] && continue
  if ! grep -qE 'run_in_threadpool_with_context|with_context|set_context' "$f"; then
    echo "LINT FAIL (rule 2) — offload without context-carrying: $f"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "LINT OK — all offload in inference/ and veda_core/ carries the ambient context (§4.1)"
exit 0
