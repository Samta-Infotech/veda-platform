# query/nl_answer.py
# VEDA — NL-back answer generation (Step 7: query -> NL)
# Gate: NL_ANSWER_ENABLED
#
# Compatibility shim: the implementation moved to query/result_explainer.py
# (Result Explanation Layer — deterministic-first, small-SLM fallback, optional
# semantic-metadata-aware prompting). Re-exported here unchanged so existing
# call sites (veda/pipeline.py, veda_hybrid.py) and tests keep working.
from query.result_explainer import (  # noqa: F401
    NLAnswerResult,
    template_answer,
    deterministic_fallback_answer,
    run_nl_answer,
)
