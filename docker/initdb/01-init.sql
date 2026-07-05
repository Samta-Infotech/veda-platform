-- Postgres init (migration_plan.md §1.1). Runs once on first cluster boot.
-- Enables pgvector and prepares the internal store. HNSW index tables (§6.4)
-- are created later by the Django RunSQL migration with tuned m/ef_construction.
CREATE EXTENSION IF NOT EXISTS vector;
