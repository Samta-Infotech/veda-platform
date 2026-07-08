-- Postgres init (migration_plan.md §1.1). Runs once on first cluster boot.
-- Enables pgvector and prepares the internal store. HNSW index tables (§6.4)
-- are created later by the Django RunSQL migration with tuned m/ef_construction.
CREATE EXTENSION IF NOT EXISTS vector;

-- Internal store (VEDA_INTERNAL_DBNAME, default "veda_engine" — see .env.example
-- and storage_adapters/writer.py): FK edges, glossary/synonyms, sampled column
-- values, etc. Separate database from POSTGRES_DB so ingestion's internal
-- bookkeeping stays isolated from the main app schema.
SELECT 'CREATE DATABASE veda_engine'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'veda_engine')\gexec

\c veda_engine
CREATE EXTENSION IF NOT EXISTS vector;
