# Ingestion Guide — Adding Documents & Datalake Files

Practical commands for adding new files to the **filesystem** source (`docs_contracts`) or
the **datalake** sources (`invoices_csv`, `catalog_parquet`) and getting them embedded/indexed.
Every command below was actually run and verified working this session.

## 0. Always use both compose files

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.demo.yml"
```

The base `docker-compose.yml` alone pins Postgres to `pg16`; the demo override pins it to
`pg17`, which is what the actual data directory was initialized with. Running any compose
command with **only** the base file recreates the `postgres` container on the wrong image and
crashes it (`FATAL: database files are incompatible with server` — this happened live during
this session and was recovered with zero data loss by re-running with both files). **Never**
run `docker compose ...` here without both `-f` flags.

---

## 1. Add a document to the filesystem source (`docs_contracts`)

**Step 1 — copy the file into the source's watched folder:**
```bash
$COMPOSE exec inference bash -c "ls /app/data/docs/"   # confirm the folder; source_path for docs_contracts
cp "/path/to/your/file.pdf" /Users/samta/veda-platform/<bind-mounted data/docs path>/
```
Supported formats: `.pdf`, `.docx`, `.txt`, `.md` (per `Source.doc_formats` for this source).

**Filename rule:** stick to lowercase letters, digits, `_`, `.`, `-`. A filename with spaces or
mixed case still ingests and is answerable, but its RBAC resource path only gets created
correctly since the sanitize fix landed this session (`CatalogDiscoveryService._sanitize_doc_segment`) —
prefer clean filenames anyway to avoid surprises.

**Step 2 — trigger ingestion:**
```bash
$COMPOSE exec api python3 manage.py shell -c "
from apps.ingestion.tasks import task_ingest_source
r = task_ingest_source.delay(source_id=3, tenant='default', force=True)
print('task:', r.id)
"
```
`source_id=3` is `docs_contracts` in this environment — confirm with:
```bash
$COMPOSE exec api python3 manage.py shell -c "
from apps.sources.models import Source
for s in Source.objects.all(): print(s.id, s.name, s.dialect)
"
```

**Step 3 — watch it run:**
```bash
$COMPOSE logs --tail=30 ingest-worker
```
Look for `[3] ✓  Chunk embedder — N chunks, M docs` and `ingestion job succeeded job_id=...`.
Also check the `catalog auto-sync` line — a non-empty `skipped: [...]` list means a document's
RBAC resource path failed to build (see the filename rule above).

**If `ingest-worker` isn't running** (`$COMPOSE ps` shows it `Exited`):
```bash
$COMPOSE up -d ingest-worker
```
It's a long-running Celery worker — if you edit Python source under `veda_core/`, `apps/`, etc.
**after** it started, it keeps the old code in memory. Restart it to pick up changes:
```bash
$COMPOSE restart ingest-worker
```

**Step 4 — verify the document is queryable:**
```bash
$COMPOSE exec inference python3 -c "
import sys; sys.path.insert(0, '/app/veda_core')
import context
ctx = context.RequestContext(source_id='3', tenant='default', source_ids=('3',), allowed_resources=None)
context.set_context(ctx)
import veda_core.context as vctx; vctx.set_context(ctx)
from query.rag_layer import run_rag_layer
r = run_rag_layer('<a question about the new document>', source_ids=['3'])
print(r.answer, r.citations)
"
```
(Both `context` module names are set deliberately — the engine can be imported under either
name depending on which container/entry point resolves `sys.path` first; setting both avoids a
known cross-module context-isolation issue.)

---

## 2. Add data to a datalake source (`invoices_csv` / `catalog_parquet`)

**Step 1 — drop the file into the source's folder:**
```bash
$COMPOSE exec inference bash -c "ls /app/data/invoices_csv/ /app/data/catalog_parquet/"
```
`invoices_csv` takes `.csv` files; `catalog_parquet` takes `.parquet` files. Copy your file into
the matching host-side folder the same way as step 1 above.

**Step 2 — trigger ingestion (same task, different `source_id`):**
```bash
$COMPOSE exec api python3 manage.py shell -c "
from apps.ingestion.tasks import task_ingest_source
task_ingest_source.delay(source_id=4, tenant='default', force=True)   # invoices_csv
task_ingest_source.delay(source_id=5, tenant='default', force=True)   # catalog_parquet
"
```

**Step 3 — watch it run**, same as above (`$COMPOSE logs --tail=30 ingest-worker`).

**Known limitation (as of this session — not yet fixed):** direct SQL queries against these
two sources' own schema currently fail — a stale, unscoped global cache
(`veda/runtime.py::get_graph()`) makes the SQL planner match against `homzhub`'s tables instead
of the datalake source's own. Ingestion itself succeeds; querying the ingested data via natural
language does not yet work reliably. See `HYBRID_PIPELINE_RCA.md` for the related, deeper
routing issues (federated cross-source queries can also mis-answer a document-only question
using datalake data by keyword coincidence).

---

## 3. Re-ingesting after editing ingestion code

If you change anything under `veda_core/ingestion/`, `veda_core/connectors/`, or
`apps/access_management/services/catalog.py`, the **long-running** `ingest-worker` process
does not see the change until restarted:
```bash
$COMPOSE restart ingest-worker
```
Then re-run the `task_ingest_source.delay(...)` call for the affected `source_id`. The `api`
container restarts fast enough during normal dev that this is usually not needed for it, but if
in doubt: `$COMPOSE restart api` too.

---

## 4. Quick sanity checks after any ingestion

```bash
# chunk counts per document (filesystem sources)
$COMPOSE exec postgres bash -c 'psql -U "$POSTGRES_USER" -d veda_engine -c \
  "select doc_name, count(*) from doc_chunks where source_id='"'"'3'"'"' group by doc_name;"'

# catalog resource paths + active state (RBAC addressability)
$COMPOSE exec api python3 manage.py shell -c "
from apps.access_management.models import CatalogResource
for r in CatalogResource.objects.filter(source_id=3).order_by('path'):
    print(r.path, r.is_active)
"
```

A document that doesn't show up in either should be re-ingested; one that shows `is_active=False`
means the last discovery run couldn't find it upstream (re-ingest, or check the filename rule).
