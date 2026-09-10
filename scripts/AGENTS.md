# scripts/ — operational scripts

Danger levels: **HIGH** = destroys state or touches prod · MED = wipes derived artifacts /
long / mutates live state · LOW = idempotent / read-only · none.

## Setup / onboarding
| Script | Role | Danger |
|--------|------|--------|
| `onboard_source.sh` | Fresh-setup a new relational source end-to-end: rewrites `.env` `VEDA_SOURCE_*`, **DROP DATABASE `veda_engine` + recreate**, TRUNCATE all `substrate_*` + verified cache, delete derived engine files, recreate the Django `Source` row, `up -d --force-recreate`. | **HIGH — destroys all ingested/derived state.** |
| `ingest_baremetal.sh <SOURCE_ID> [TENANT]` | Full ingestion on the Mac host (MPS + host Ollama), reads the connection from the `Source` row, writes into container Postgres `:15432`, then `writer.warm()` + flips `ready=True` + POSTs `/v1/rehydrate`. Hardcodes a `veda-poc` venv path. | MED |
| `deploy_semantic_bridge.sh` | Restart (or `REBUILD=1`) inference + ingest-worker, then run `backfill_semantic_bridge.py`. Idempotent. | LOW |
| `resync_catalog.py` | Delete stale `CatalogResource` children for non-relational sources, re-run `CatalogDiscoveryService`. | LOW-MED |

## The one that hits prod
| Script | Role | Danger |
|--------|------|--------|
| `../veda_core/run_homzhub_query.sh` | Exports `VEDA_SOURCE_*` at the live **DigitalOcean** `homzhub_prod` DB and runs `main.py --query`. `VEDA_SOURCE_PASSWORD=''` (filled locally). "Do not run casually" (`CLAUDE.md`). | **HIGH — live prod source DB.** |

## Demo bundle
| Script | Role | Danger |
|--------|------|--------|
| `demo/export.sh [OUT]` | `pg_dump -Fc` both DBs (direct `:15432`), tar `veda_core/data`, export the model/ollama volumes → `demo_bundle/`. | LOW (read-only) |
| `demo/restore.sh [BUNDLE]` | On the demo box: import volumes, reachability pre-check, untar data, `pg_restore --clean --if-exists` both DBs, per-source dialect preflight, `up -d --no-deps inference api`. | MED (`--clean`) |

## Eval / tuning (deterministic, no source-DB writes)
`build_golden_set.py`, `retrieval_eval.py` (WP0 gate), `parity_suite.py` (Phase 7.1
legacy-vs-migrated), `tune_fusion_weights.py` (WP6 — **never writes config**, prints the
dict), `hnsw_parity_sweep.py` (7.1a — writes an index table, LOW), `enrich_synonyms.py`
(mutates the synonyms file — recompile after, LOW).

## Backfills (idempotent, LOW)
`backfill_cross_source.py`, `backfill_semantic_bridge.py`, `backfill_semantic_model.py`
(overwrites `Sm*` + `veda:sm:*` for listed sources — LOW-MED; homzhub source 2 never touched).

## Host-side servers (run outside Docker)
| Script | Role |
|--------|------|
| `metal_embed_server.py --port 11435` | BGE-M3 + reranker on Apple MPS. Container reaches it via `METAL_EMBED_URL`. If down, every query CPU-falls-back and heavy-lane latency spikes. |
| `ollama_proxy.py --port 11434` | Round-robin LB across `OLLAMA_BACKENDS` (the `ollama-proxy` compose service, profile `proxy`, off by default). |

## CI / test
`lint_no_raw_offload.sh` (CI gate — bans bare offload in `inference/` + `veda_core/`),
`test_filesystem_queries_remote.py` (hits a real deployed endpoint, LOW),
`test_filesystem_rbac_live.py` (**mutates then restores** live RBAC grants — MED).
