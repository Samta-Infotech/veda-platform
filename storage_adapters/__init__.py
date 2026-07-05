"""storage_adapters

Migration plan §3, §4 — the seam between VEDA's storage call sites and the
Django-native substrate. Structured rows go through the ORM (`apps.substrate`
models); ANN/graph/arbitrary read SQL stays raw pgvector on the same Postgres
Django manages, reached through PgBouncer. Tenancy is applied here, and only
here, from the ambient `veda_core.context.current()` — engine signatures never
change (§4.1).
"""
