# =============================================================================
# schema/simulate_schema.py
# VEDA POC — Synthetic Schema Generator
#
# Generates a realistic simulated relational schema that mimics an enterprise
# real-estate / property management database (similar to Homzhub domain used
# in the architecture document).
#
# Output: a plain Python dict — `SIMULATED_SCHEMA` — that every downstream
# ingestion step consumes. No real DB connection required.
#
# Replace this file with a real DB connector when moving to production.
# Nothing else in the pipeline needs to change.
# =============================================================================

import uuid
from config import (
    NUM_TABLES,
    NUM_FK_RELATIONS,
    SENSITIVE_PATTERNS,
)


# =============================================================================
# Schema definition
# Each table is a dict with:
#   - table_id   : stable UUID string
#   - table_name : string
#   - columns    : list of column dicts (see _make_column for shape)
#   - row_count  : approximate row count (simulated)
# =============================================================================

def _make_column(
    col_name: str,
    data_type: str,
    is_pk: bool = False,
    is_fk: bool = False,
    fk_ref_table: str = None,
    fk_ref_col: str = None,
    nullable: bool = True,
    cardinality: int = None,
) -> dict:
    """
    Returns a single column descriptor dict.
    cardinality: approximate number of distinct values (None = unknown / not sampled).
    """
    return {
        "col_id":        str(uuid.uuid4()),
        "col_name":      col_name,
        "data_type":     data_type,       # integer | varchar | numeric | timestamp | boolean
        "is_pk":         is_pk,
        "is_fk":         is_fk,
        "fk_ref_table":  fk_ref_table,    # table_name of referenced table (None if not FK)
        "fk_ref_col":    fk_ref_col,      # col_name of referenced column (None if not FK)
        "nullable":      nullable,
        "cardinality":   cardinality,     # int or None
    }


# =============================================================================
# Table definitions — 12 tables, realistic property-management domain
# Covers: tenants, properties, leases, payments, assets, maintenance, agents,
#         projects, invoices, units, owners, audit_log
# =============================================================================

def _build_raw_tables() -> list:
    """
    Returns the full list of table dicts with columns defined inline.
    FK references use table_name strings — resolved to UUIDs in a later pass.
    """
    tables = [

        # ------------------------------------------------------------------
        # 1. tenants
        # ------------------------------------------------------------------
        {
            "table_name": "tenants",
            "row_count":  8500,
            "columns": [
                _make_column("tenant_id",     "integer",   is_pk=True,  nullable=False, cardinality=8500),
                _make_column("full_name",      "varchar",                nullable=False, cardinality=8200),
                _make_column("email",          "varchar",                nullable=True,  cardinality=8100),
                _make_column("phone",          "varchar",                nullable=True,  cardinality=7900),
                _make_column("status",         "varchar",                nullable=False, cardinality=3),
                # sensitive — auto-excluded by schema scanner
                _make_column("aadhar",         "varchar",                nullable=True,  cardinality=8500),
                _make_column("created_at",     "timestamp",              nullable=False, cardinality=None),
            ],
        },

        # ------------------------------------------------------------------
        # 2. owners
        # ------------------------------------------------------------------
        {
            "table_name": "owners",
            "row_count":  1200,
            "columns": [
                _make_column("owner_id",       "integer",   is_pk=True,  nullable=False, cardinality=1200),
                _make_column("full_name",       "varchar",                nullable=False, cardinality=1180),
                _make_column("email",           "varchar",                nullable=True,  cardinality=1150),
                _make_column("pan_number",      "varchar",                nullable=True,  cardinality=1200),   # sensitive
                _make_column("created_at",      "timestamp",              nullable=False, cardinality=None),
            ],
        },

        # ------------------------------------------------------------------
        # 3. projects  (housing society / apartment complex)
        # ------------------------------------------------------------------
        {
            "table_name": "projects",
            "row_count":  320,
            "columns": [
                _make_column("project_id",     "integer",   is_pk=True,  nullable=False, cardinality=320),
                _make_column("project_name",   "varchar",                nullable=False, cardinality=318),
                _make_column("city",           "varchar",                nullable=False, cardinality=42),
                _make_column("locality",       "varchar",                nullable=True,  cardinality=180),
                _make_column("created_at",     "timestamp",              nullable=False, cardinality=None),
            ],
        },

        # ------------------------------------------------------------------
        # 4. units  (individual flats / offices within a project)
        # ------------------------------------------------------------------
        {
            "table_name": "units",
            "row_count":  14000,
            "columns": [
                _make_column("unit_id",        "integer",   is_pk=True,  nullable=False, cardinality=14000),
                _make_column("project_id",     "integer",   is_fk=True,
                             fk_ref_table="projects", fk_ref_col="project_id",
                             nullable=False, cardinality=320),
                _make_column("owner_id",       "integer",   is_fk=True,
                             fk_ref_table="owners", fk_ref_col="owner_id",
                             nullable=False, cardinality=1200),
                _make_column("unit_number",    "varchar",                nullable=False, cardinality=13800),
                _make_column("floor_number",   "integer",                nullable=True,  cardinality=40),
                _make_column("carpet_area_sqft","numeric",               nullable=True,  cardinality=220),
                _make_column("property_type",  "varchar",                nullable=False, cardinality=5),
                _make_column("furnishing_type","varchar",                nullable=True,  cardinality=3),
                _make_column("facing",         "varchar",                nullable=True,  cardinality=8),
                _make_column("status",         "varchar",                nullable=False, cardinality=4),
            ],
        },

        # ------------------------------------------------------------------
        # 5. assets  (physical assets attached to a unit)
        # ------------------------------------------------------------------
        {
            "table_name": "assets",
            "row_count":  31000,
            "columns": [
                _make_column("asset_uuid",     "varchar",   is_pk=True,  nullable=False, cardinality=31000),
                _make_column("unit_id",        "integer",   is_fk=True,
                             fk_ref_table="units", fk_ref_col="unit_id",
                             nullable=False, cardinality=14000),
                _make_column("asset_name",     "varchar",                nullable=False, cardinality=85),
                _make_column("asset_type",     "varchar",                nullable=False, cardinality=12),
                _make_column("purchase_date",  "timestamp",              nullable=True,  cardinality=None),
                _make_column("purchase_price", "numeric",                nullable=True,  cardinality=3200),
                _make_column("description",    "varchar",                nullable=True,  cardinality=28000),
            ],
        },

        # ------------------------------------------------------------------
        # 6. lease_transactions  (active and historical leases)
        # ------------------------------------------------------------------
        {
            "table_name": "lease_transactions",
            "row_count":  22000,
            "columns": [
                _make_column("lease_id",           "integer",  is_pk=True, nullable=False, cardinality=22000),
                _make_column("unit_id",            "integer",  is_fk=True,
                             fk_ref_table="units", fk_ref_col="unit_id",
                             nullable=False, cardinality=14000),
                _make_column("tenant_id",          "integer",  is_fk=True,
                             fk_ref_table="tenants", fk_ref_col="tenant_id",
                             nullable=False, cardinality=8500),
                _make_column("lease_start_date",   "timestamp",           nullable=False, cardinality=None),
                _make_column("lease_end_date",     "timestamp",           nullable=True,  cardinality=None),
                _make_column("rent",               "numeric",             nullable=False, cardinality=1800),
                _make_column("security_deposit",   "numeric",             nullable=True,  cardinality=900),
                _make_column("lease_period",       "integer",             nullable=True,  cardinality=24),
                _make_column("status",             "varchar",             nullable=False, cardinality=4),
                _make_column("remarks",            "varchar",             nullable=True,  cardinality=18000),
            ],
        },

        # ------------------------------------------------------------------
        # 7. payments
        # ------------------------------------------------------------------
        {
            "table_name": "payments",
            "row_count":  95000,
            "columns": [
                _make_column("payment_id",     "integer",   is_pk=True,  nullable=False, cardinality=95000),
                _make_column("lease_id",       "integer",   is_fk=True,
                             fk_ref_table="lease_transactions", fk_ref_col="lease_id",
                             nullable=False, cardinality=22000),
                _make_column("payment_date",   "timestamp",              nullable=False, cardinality=None),
                _make_column("paid_amount",    "numeric",                nullable=False, cardinality=4200),
                _make_column("payment_mode",   "varchar",                nullable=False, cardinality=6),
                _make_column("status",         "varchar",                nullable=False, cardinality=3),
                _make_column("notes",          "varchar",                nullable=True,  cardinality=61000),
                # sensitive
                _make_column("cvv",            "varchar",                nullable=True,  cardinality=900),
            ],
        },

        # ------------------------------------------------------------------
        # 8. invoices
        # ------------------------------------------------------------------
        {
            "table_name": "invoices",
            "row_count":  48000,
            "columns": [
                _make_column("invoice_id",     "integer",   is_pk=True,  nullable=False, cardinality=48000),
                _make_column("lease_id",       "integer",   is_fk=True,
                             fk_ref_table="lease_transactions", fk_ref_col="lease_id",
                             nullable=False, cardinality=22000),
                _make_column("invoice_date",   "timestamp",              nullable=False, cardinality=None),
                _make_column("due_date",       "timestamp",              nullable=False, cardinality=None),
                _make_column("total_value",    "numeric",                nullable=False, cardinality=5100),
                _make_column("service_fee",    "numeric",                nullable=True,  cardinality=280),
                _make_column("tax_amount",     "numeric",                nullable=True,  cardinality=640),
                _make_column("status",         "varchar",                nullable=False, cardinality=4),
            ],
        },

        # ------------------------------------------------------------------
        # 9. maintenance_requests
        # ------------------------------------------------------------------
        {
            "table_name": "maintenance_requests",
            "row_count":  17000,
            "columns": [
                _make_column("request_id",     "integer",   is_pk=True,  nullable=False, cardinality=17000),
                _make_column("unit_id",        "integer",   is_fk=True,
                             fk_ref_table="units", fk_ref_col="unit_id",
                             nullable=False, cardinality=14000),
                _make_column("tenant_id",      "integer",   is_fk=True,
                             fk_ref_table="tenants", fk_ref_col="tenant_id",
                             nullable=True,  cardinality=8500),
                _make_column("request_date",   "timestamp",              nullable=False, cardinality=None),
                _make_column("category",       "varchar",                nullable=False, cardinality=15),
                _make_column("priority",       "varchar",                nullable=False, cardinality=3),
                _make_column("status",         "varchar",                nullable=False, cardinality=5),
                _make_column("resolution_cost","numeric",                nullable=True,  cardinality=1200),
                _make_column("description",    "varchar",                nullable=True,  cardinality=16000),
                _make_column("resolved_at",    "timestamp",              nullable=True,  cardinality=None),
            ],
        },

        # ------------------------------------------------------------------
        # 10. agents
        # ------------------------------------------------------------------
        {
            "table_name": "agents",
            "row_count":  420,
            "columns": [
                _make_column("agent_id",       "integer",   is_pk=True,  nullable=False, cardinality=420),
                _make_column("full_name",       "varchar",                nullable=False, cardinality=415),
                _make_column("email",           "varchar",                nullable=True,  cardinality=410),
                _make_column("commission_rate", "numeric",                nullable=True,  cardinality=18),
                _make_column("status",          "varchar",                nullable=False, cardinality=2),
                _make_column("created_at",      "timestamp",              nullable=False, cardinality=None),
            ],
        },

        # ------------------------------------------------------------------
        # 11. agent_assignments  (which agent manages which unit)
        # ------------------------------------------------------------------
        {
            "table_name": "agent_assignments",
            "row_count":  9800,
            "columns": [
                _make_column("assignment_id",  "integer",   is_pk=True,  nullable=False, cardinality=9800),
                _make_column("agent_id",       "integer",   is_fk=True,
                             fk_ref_table="agents", fk_ref_col="agent_id",
                             nullable=False, cardinality=420),
                _make_column("unit_id",        "integer",   is_fk=True,
                             fk_ref_table="units", fk_ref_col="unit_id",
                             nullable=False, cardinality=14000),
                _make_column("assigned_date",  "timestamp",              nullable=False, cardinality=None),
                _make_column("is_active",      "boolean",                nullable=False, cardinality=2),
            ],
        },

        # ------------------------------------------------------------------
        # 12. audit_log
        # ------------------------------------------------------------------
        {
            "table_name": "audit_log",
            "row_count":  250000,
            "columns": [
                _make_column("log_id",         "integer",   is_pk=True,  nullable=False, cardinality=250000),
                _make_column("table_name",     "varchar",                nullable=False, cardinality=12),
                _make_column("operation",      "varchar",                nullable=False, cardinality=4),
                _make_column("changed_at",     "timestamp",              nullable=False, cardinality=None),
                _make_column("changed_by",     "varchar",                nullable=True,  cardinality=85),
                _make_column("notes",          "varchar",                nullable=True,  cardinality=210000),
                # sensitive
                _make_column("token",          "varchar",                nullable=True,  cardinality=250000),
            ],
        },
    ]

    return tables


# =============================================================================
# Post-processing passes
# =============================================================================

def _assign_table_ids(tables: list) -> dict:
    """
    Assigns a stable UUID to each table.
    Returns a dict: table_name -> table_id (used for FK resolution).
    """
    name_to_id = {}
    for table in tables:
        tid = str(uuid.uuid4())
        table["table_id"] = tid
        name_to_id[table["table_name"]] = tid
    return name_to_id


def _resolve_fk_table_ids(tables: list, name_to_id: dict) -> None:
    """
    Replaces fk_ref_table string names with their resolved UUID strings,
    adding fk_ref_table_id to each FK column.
    """
    for table in tables:
        for col in table["columns"]:
            if col["is_fk"] and col["fk_ref_table"]:
                ref_name = col["fk_ref_table"]
                col["fk_ref_table_id"] = name_to_id.get(ref_name, None)


def _exclude_sensitive_columns(tables: list) -> tuple:
    """
    Removes columns matching SENSITIVE_PATTERNS from every table.
    Returns (cleaned tables, list of excluded column names for audit).
    """
    excluded = []
    for table in tables:
        safe_cols = []
        for col in table["columns"]:
            is_sensitive = any(
                pattern in col["col_name"].lower()
                for pattern in SENSITIVE_PATTERNS
            )
            if is_sensitive:
                excluded.append(f"{table['table_name']}.{col['col_name']}")
            else:
                safe_cols.append(col)
        table["columns"] = safe_cols
    return tables, excluded


# =============================================================================
# Public entry point
# =============================================================================

def get_simulated_schema() -> dict:
    """
    Builds and returns the full simulated schema as a dict:

    {
        "tables": [ <table dict>, ... ],
        "name_to_id": { table_name: table_id, ... },
        "excluded_columns": [ "table.col", ... ],
        "stats": {
            "total_tables": int,
            "total_columns": int,
            "total_fk_edges": int,
            "excluded_count": int,
        }
    }

    This is the only function downstream modules should call.
    """
    tables = _build_raw_tables()
    name_to_id = _assign_table_ids(tables)
    _resolve_fk_table_ids(tables, name_to_id)
    tables, excluded = _exclude_sensitive_columns(tables)

    total_columns = sum(len(t["columns"]) for t in tables)
    total_fk_edges = sum(
        1 for t in tables
        for c in t["columns"]
        if c["is_fk"]
    )

    return {
        "tables":          tables,
        "name_to_id":      name_to_id,
        "excluded_columns": excluded,
        "stats": {
            "total_tables":   len(tables),
            "total_columns":  total_columns,
            "total_fk_edges": total_fk_edges,
            "excluded_count": len(excluded),
        },
    }


# =============================================================================
# Quick smoke test — run directly to inspect the schema
# python schema/simulate_schema.py
# =============================================================================

if __name__ == "__main__":
    schema = get_simulated_schema()
    stats  = schema["stats"]

    print("=" * 60)
    print("VEDA POC — Simulated Schema")
    print("=" * 60)
    print(f"  Tables          : {stats['total_tables']}")
    print(f"  Total columns   : {stats['total_columns']}")
    print(f"  FK edges        : {stats['total_fk_edges']}")
    print(f"  Excluded cols   : {stats['excluded_count']}")
    print()

    for table in schema["tables"]:
        print(f"  [{table['table_name']}]  ({table['row_count']:,} rows)  id={table['table_id'][:8]}...")
        for col in table["columns"]:
            fk_info = ""
            if col["is_fk"]:
                fk_info = f"  → FK:{col['fk_ref_table']}.{col['fk_ref_col']}"
            pk_info = " PK" if col["is_pk"] else ""
            print(f"      {col['col_name']:<25} {col['data_type']:<12}{pk_info}{fk_info}")
        print()

    print("Excluded sensitive columns:")
    for exc in schema["excluded_columns"]:
        print(f"  ✗  {exc}")