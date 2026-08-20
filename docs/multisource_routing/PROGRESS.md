# Multi-Source Routing — Progress Log

> Living doc. Newest entry on top. One entry per work session: what changed, files touched,
> what's next, any blockers. Keep it factual (an index, not a retelling).

## Status snapshot
- **Current phase:** Phase 1 (Source Profiling & Catalog) — 1.1, 1.2 done
- **Next action:** Phase 1.3 — `ingestion/source_profiler.py` post-ingest hook
- **Flags live:** none yet (Phase 1 is additive schema, no query-path flag needed)
- **Branch:** `feat/multisource-routing`

---

## 2026-08-20 — Phase 1.2: Admin manual-entry surface
- `apps/sources/admin.py`: grouped `fieldsets` — new "Routing catalog" section holds
  `domain_tags`/`description`/`description_generated`/`is_canonical`; connection + doc fields
  collapsed. `description_generated` + timestamps read-only. `is_canonical` in list_display/filter.
- No source-CREATE/EDIT API exists (`views.py` POST is a list-with-body variant), so the read
  API contract (§5.2 `serialize_source`) is left untouched; the coordinator will read these
  fields off the `Source` model directly. `manage.py check` clean.
- **Next:** Phase 1.3 (profiler hook).

---

## 2026-08-20 — Phase 1.1: Source model profile fields
- Confirmed `Source` is a FLAT GLOBAL registry (not tenant-scoped; tenant lives on substrate
  rows via `TenantScopedModel`, and `_ready_source_ids` is global) → profile fields go directly
  on the `Source` row. Decision recorded in MEMORY.md.
- Added to `apps/sources/models.py`: `domain_tags` (JSON), `description` (text),
  `description_generated` (bool, provenance so profiler never overwrites manual), `is_canonical` (bool).
- Generated + applied migration `0005_source_description_...`. `manage.py check` clean.
- Env note: use `.venv/bin/python` (Django 5.0.14) for manage.py; system python has no Django.
- **Next:** Phase 1.2 (API/admin expose).

## 2026-08-20 — Planning & scaffolding
- Completed full architecture review grounded in codebase (3 parallel code-exploration passes).
  Output: `../../MULTISOURCE_ARCH_REVIEW.md`.
- Key verified facts: RBAC/scope, multi-source retrieval, cross-source federation, provenance,
  Tier-2 retry, ExplainTrace tracing all EXIST and are reusable (~70% of target already built).
- Identified 2 spine-blocking bugs: phase3 `RetrievalResult` has no `source_id`; graph adapter
  hardcodes `source_id=""` (`graph_retriever.py:543`) → candidates silently dropped from grouping.
- MISSING (net-new): source profile fields, canonical concept, presence-tier + deterministic
  routing, merge/conflict policy, per-source partial-failure, routing/merge trace sections.
- Created tracking docs: `PLAN.md`, `PROGRESS.md`, `MEMORY.md`, `PM.md` (this folder).
- **Next:** start Phase 1.1.
