# apps/evaluation/ — tracked eval runs

One of four eval surfaces (see [../../docs/EVALUATION.md](../../docs/EVALUATION.md)) — this
is the Django-tracked Celery one.

| File | Role |
|------|------|
| `tasks.py` | `task_run_eval(source_id=1, tenant, label, queries=None)` (queue `default`). Runs each query through `InferenceClient.run_hybrid_query`, `_SUCCESS_STATUS="ok"`, stores an `EvalRun` + per-case `EvalCaseResult` + an HTML report (`_report_row` **HTML-escapes every value** — stored-XSS fix, query text is caller-supplied). Default 4-query flow set. Unreachable inference → an `exec_error` case, not a task failure. |
| `models.py` | `EvalRun` (FK source, tenant, label, `recall_at_k`, `hit_rate`, `sql_success_rate`, `report_html`) + `EvalCaseResult` (FK run, `query_id`, `query_type`, `difficulty`, `recall`, `hit`, `status`, `details` JSON). |
| `admin.py` | `EvalRunAdmin` + `EvalCaseResultInline`. |
| `apps.py` / `migrations/0001_initial.py` | config + tables. |

Trigger: `POST /api/v1/admin/eval` (`apps/query/views.EvalTriggerView`, staff-only).
The file-based harnesses (`evaluation/`), the WP0 retrieval eval (`scripts/`), and
`mlflow_observability evaluate` are separate — see `docs/EVALUATION.md`.
