# Multi-source serving — deployment runbook

How to bring a **second (and third, …) data source** into service so a question is
*routed* to the right one and refused when the asker has no permission for it.

`DEPLOYMENT.md` covers the single-VM install; everything here is on top of a working
single-source deployment. `MULTI_SOURCE_SERVING.md` is the design/execution log of how the
capability was built (July) — it is not a deploy procedure and predates the flags below.

> **Everything in this file was learned by breaking it.** Each gotcha is a defect that
> actually shipped and cost a day. They are written as causes, not warnings, because the
> failure mode in every single case was **silent**: a plausible answer from the wrong
> source, or a confident number from unfiltered rows. Nothing raised.

---

## 0. The one-paragraph model

A question arrives with a **scope** (which source ids the asker may see) and a **profile per
source** (what kind each one is). The routing coordinator scores the question against every
in-scope source, picks a winner, and hands it to the agent for that source's *kind*. Four
kinds exist, and each has exactly one agent (`veda_core/query/agents.py::_AGENT_BY_KIND`):

| kind | agent | answers via |
|---|---|---|
| `relational` | `DatabaseAgent` | deterministic SQL head (`veda.pipeline.run_query`) |
| `datalake` | `DataLakeAgent` | same head; `veda/execution.py` runs the SQL on DuckDB over parquet |
| `document` | `FileSystemAgent` | RAG (`query.rag_layer.run_rag_layer`) |
| `nosql` | `NoSqlAgent` | `veda_hybrid._run_nosql` |

Anything that breaks the scope, the profiles, or the kind lookup does not error — it falls
through to the **legacy single-source path**, which answers from source 1. That fallback is
the root cause of most "why is it answering from the wrong data" reports.

---

## 1. Register each source with the right dialect

The `Source.dialect` field is the *only* input to `Source.source_kind()`
(`apps/sources/models.py::_DIALECT_TO_ENGINE`), and `source_kind()` is what selects the
agent. Supported dialects and the kind each maps to:

```
postgres | mysql | sqlite | oracle | sqlserver | duckdb   -> relational
mongo | es | dynamo                                       -> nosql
filesystem | s3_docs                                      -> document
delta | parquet | csv_lake | iceberg                      -> datalake
```

A dialect outside that table silently defaults to `("relational", "generic")`.

**Gotcha — the vocabulary trap.** The engine's agent registry is keyed by the *kind*
(`relational`/`datalake`/`document`/`nosql`), **not** by the dialect or connector type. Pass
a raw connector type (`csv_lake`, `parquet`, `filesystem`) as a source's `source_type` and
`resolve_agent()` returns `None` → `dispatch()` returns `None` → the coordinator falls
through to the legacy single-source path and answers from homzhub. The production path is
correct (`apps/query/scope.py::source_profiles_for` calls `s.source_kind()`), so this only
bites **callers that build profiles by hand** — which is exactly what the benchmark harnesses
in `veda_core/{dl_bench,fs_bench}.py` do. Three benchmark runs were invalidated by it before
anyone noticed the numbers were measuring the fallback path. If you write a harness, copy
the `source_profiles_for` shape, do not invent one.

---

## 2. Ingest, and confirm each source got its OWN semantic model

```bash
COMPOSE="docker compose"
# per source, then watch the IngestionJob to `ready`
curl -sk -X POST https://your.domain.com/api/v1/admin/ingest \
  -H "Authorization: Token <admin-token>" -H "Content-Type: application/json" \
  -d '{"source_id": 4, "force": true}'
```

`storage_adapters/writer.py::warm()` builds a per-source model from that source's own
`graph_nodes` for every **non-relational** source (`ingestion/lite_semantic_model.build_lite_sm`).
For data ingested *before* that landed, backfill instead of re-ingesting:

```bash
# needs Django + storage_adapters, so it runs in the api/ingest tier, not inference
$COMPOSE exec api python scripts/backfill_semantic_model.py --source-ids 4,5,3 --tenant default
```
It rebuilds each listed source's model from its **own** column nodes in `graph_nodes` and
republishes `veda:sm:{sid}:{tenant}`. Source 2 (homzhub, real `semantic_layer_v2` output) is
never touched.

Verify a source's model describes **its own** tables and nothing else:

```bash
$COMPOSE exec -w /app/veda_core -e PYTHONPATH=/app:/app/veda_core inference python -c "
from veda_hybrid import _load_sm_from_redis
sm = _load_sm_from_redis(scope=(4, 'default'))
print(sorted({k.split('.')[0] for k in (sm or {}).get('columns', {})}))"
```

Measured on the reference deployment:

```
2 []                                 <- expected: the relational source is not on the Redis SM path
4 ['maintenance', 'vendors']
5 ['amenities_catalog']
```

An empty list for the **relational** source is correct — it resolves its model through the
on-disk/substrate path, not `veda:sm:*`. An empty list for a datalake/document source, or one
listing another source's tables, is the bleed described below.

**Gotcha — semantic-model bleed.** Before the per-source build existed, every source
published the *global homzhub* model. A datalake question then retrieved homzhub columns,
passed the firewall (the columns were legitimately in the model), and answered from the wrong
schema. The symptom was 15 columns across 8 homzhub tables on a scope that owns 4 columns in
1 table. If the command above prints tables the source does not own, stop and fix this first
— nothing downstream can be trusted while it is wrong.

---

## 3. Set the routing flags — on **api and inference**, and recreate

`docker-compose.yml` shares `env_file: [.env]` across `api`, `worker`, `beat`,
`ingest-worker` and `inference`, so `.env` is the one place to set these. The full annotated
list is in `.env.example`; these are the four that decide whether multi-source works at all:

```bash
MULTISOURCE_ROUTING_ENABLED=1
MULTISOURCE_ROUTING_SHADOW=0          # code default is 1 — see below
REQUIRED_SOURCE_ESCALATION_ENABLED=0  # code default is 1 — see below
ROUTING_PERMISSION_DENY_GAP=0.12
```

```bash
$COMPOSE up -d          # NOT `restart`
```

**Gotcha — `SHADOW` defaults to observe-only.** `MULTISOURCE_ROUTING_SHADOW=1`
(`veda_core/config.py`, the default) makes the coordinator *log* its decision while the
legacy path still produces the answer. Routing looks correct in the trace and is not
actually in effect. There is no warning; the only tell is that answers come from source 1.
It **must** be `0` for routing to be authoritative.

**Gotcha — required-source escalation overrides permission.** With
`REQUIRED_SOURCE_ESCALATION_ENABLED=1` (also the default) a source named in the question is
forced into scope, *including one the asker's role does not grant* — so the permission
refusal never fires and the restricted source answers. Keep it `0` until re-validated. This
was the direct cause of the RBAC leak traced in `HYBRID_PIPELINE_RCA.md`.

**Gotcha — `restart` does not reload `.env`.** Env is read at container **create** time, so
`docker compose restart` keeps the old values while showing a healthy container. Use
`up -d`, which recreates. (Code is different: `.:/app` is bind-mounted, so a code edit needs
only a restart — but see §6.)

**Gotcha — `.env` is gitignored, and so is `.env.example`.** `.gitignore` ignores `.env` and
`.env.*`, so neither travels with the repo. On a fresh clone every flag above silently reverts
to its code default, which reinstates both failure modes in this section. Either add
`!.env.example` to `.gitignore` or hand the file over out of band.

---

## 4. Confirm the per-source profiles reach the engine

Scope and profiles cross a process boundary (api → inference) as headers:

```
X-Veda-Source-Ids       the authorised scope
X-Veda-Data-Scope       the precomputed RBAC scope (absent = no restriction)
X-Veda-Source-Profiles  {"4": {"source_type": "datalake", "is_canonical": false,
                               "domain_tags": [...], "description": "..."}}
```

The chain, all three links required:

| link | file |
|---|---|
| built from the `Source` rows | `apps/query/scope.py::source_profiles_for` |
| sent as a header | `apps/query/inference_client.py::_HEADER_SOURCE_PROFILES` |
| read into a contextvar | `inference/main.py` middleware → `set_source_profiles(...)` |
| threaded to the chat graph | `chatbot/run.py(source_profiles=...)` → `chatbot/state.py` |

**Gotcha — the header nobody read.** The api built the profiles and sent them correctly; the
inference middleware never parsed that one header (it read source-ids, tenant and data-scope
but not profiles). Every request therefore reached the coordinator with `{}` profiles, no
source had a known *kind*, and routing fell back. It looked like a routing-quality problem
for a long time because the profiles were visibly present on the wire.

**Gotcha — the streaming route loses contextvars.** `/v1/run_hybrid_query/stream` runs the
engine on a worker thread, which inherits none of the middleware's contextvars but
`RequestContext`. It needs `copy_context()` captured in the request thread and run inside the
worker (`inference/routes/hybrid.py`) — otherwise the non-streaming endpoint routes correctly
and the streaming one silently does not. **Any new contextvar must be added to that snapshot
or it will exist on one endpoint only.**

**Gotcha — two `context` modules.** `veda_core.context` and a bare `context` are separate
module objects with **separate** contextvars; setting one leaves the other empty. Read both
(`veda_hybrid.py::_current_ctx`, documented at `veda_hybrid.py:71-81`). This defaulted the
whole SQL head to source 1 once already.

---

## 5. Verify — three cases, all three required

Permission behaviour is not one case. Run all three; a deployment that passes only the first
two leaks data.

| # | setup | question | required outcome |
|---|---|---|---|
| 1 | asker has **all** sources | one question per source | each answers **from its own** source |
| 2 | asker **has** the datalake grant | a datalake question | answers from the datalake |
| 3 | asker **lacks** the datalake grant but holds another | the same datalake question | **refuses**: "you don't have permission" |

Case 3 is the whole point and the easiest to get wrong. Two specific requirements:

- The chunks/columns of the un-granted source **are** embedded and **are** scored — routing
  has to see them to know the question belongs to that source. Blocking happens *after*
  routing, at the permission boundary, not by hiding the source from the router.
- The refusal must **not name the source**. "You don't have permission to access this data.
  → Contact your Admin to request access." (`veda/feedback.py::ACCESS_DENIED_WHY`). Naming
  the source, or listing near-miss table names as suggestions, discloses the existence of
  data the asker may not know about — which is what the refusal exists to prevent.
  `_restricted_match` is deliberately exact-match, never fuzzy, for the same reason.

**Gotcha — the near-miss denial.** When a permitted source scores *close behind* an
un-granted one, answering from the permitted source is wrong (it answers a different
question) and refusing is right. `ROUTING_PERMISSION_DENY_GAP=0.12` is that margin, measured
from the live score-gap distribution — not a guess, and not portable to a different corpus
without re-measuring.

---

## 6. Two things that will waste your afternoon

**The chat path runs stale code after an edit.** `chatbot/` calls the inference tier over
HTTP (`apps/query/inference_client.py`), and the inference gunicorn worker holds the modules
it imported at container-create time. A code edit under `.:/app` is live for anything you run
with `docker exec ... python`, but the **chat path keeps executing the old modules** until:

```bash
$COMPOSE restart inference
```

This produces a genuinely confusing signal: an in-process test shows the fix and the chat
path shows the old behaviour, in the same container, at the same moment.

**`docs/multisource_routing/` does not exist.** `veda_core/config.py` cites ~10 reports
inside it (`SOURCE_ISOLATION_REPORT.md`, `REQUIRED_SOURCE_ESCALATION_REPORT.md`,
`CROSS_SOURCE_EDGE_MULTI.md`, …). None of them are in the repo. Don't spend time looking;
the flag comments in `config.py` are the surviving record.

---

## Post-deploy checklist

- [ ] Every `Source.dialect` maps to the intended kind (§1)
- [ ] Each source's semantic model lists only its own tables (§2)
- [ ] `MULTISOURCE_ROUTING_SHADOW=0` **and** `REQUIRED_SOURCE_ESCALATION_ENABLED=0` in `.env`
- [ ] Flags applied with `up -d`, not `restart`
- [ ] A request carries a non-empty `X-Veda-Source-Profiles`, and routing is correct on the
      **streaming** endpoint as well as the plain one
- [ ] All three permission cases in §5 pass, and case 3's refusal names no source
- [ ] `.env.example` is reachable by whoever deploys next (it is gitignored today)
