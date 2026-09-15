"""One-off live verification (not a permanent test) for the P0-2 follow-up: MySQL
dialect support in ingestion/relationship_graph.py, run against a throwaway MySQL 8
container (see docs/backlog/query-engine-open-items.md for the setup — a
customers/orders schema with a real FK).

Monkeypatches relationship_graph.get_real_schema for the duration of this call only
(the real dispatch, connectors.relational.get_real_schema(), reads the env-injected
"primary" source, not ctx.connection — a pre-existing single-source-per-process
design choice unrelated to this fix) so this can run without registering a throwaway
Django Source row.

Usage (inside the inference container, mysql-connector-python installed):
    python /app/scripts/verify_mysql_relationship_graph.py
"""
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

from ingestion.contracts import SourceContext
from connectors.relational import _build_relational_connector
import ingestion.relationship_graph as rg

MYSQL_CFG = {
    "id": "mysql_test", "engine": "mysql", "type": "relational",
    "host": "veda-test-mysql", "port": 3306, "dbname": "testdb",
    "user": "root", "password": "testpass",
}

def _fake_get_real_schema():
    connector = _build_relational_connector(MYSQL_CFG)
    status = connector.connect()
    assert status.ok, status.message
    try:
        return connector.get_raw_schema_dict()
    finally:
        connector.disconnect()

rg.get_real_schema = _fake_get_real_schema

ctx = SourceContext(source_id="mysql_test", tenant="default", type="relational",
                    engine="mysql", connection=dict(MYSQL_CFG))

graph = rg.build_relationship_graph(tables=["customers", "orders"], ctx=ctx, verbose=True)
print()
print("=== RESULT ===")
import json
print(json.dumps(graph, indent=2))

assert graph["stats"]["mode"] == "sql", f"expected full SQL introspection, got {graph['stats']}"
assert graph["stats"]["num_edges"] == 1, f"expected 1 FK edge (orders.customer_id -> customers.id), got {graph['stats']}"
edge = graph["edges"][0]
assert edge["source_table"] == "orders" and edge["source_column"] == "customer_id"
assert edge["target_table"] == "customers" and edge["target_column"] == "id"
assert edge["cardinality"] == "N:1", f"expected N:1 (many orders per customer), got {edge['cardinality']}"
print()
print("ALL ASSERTIONS PASSED — MySQL dialect path verified live: PK detection, "
      "declared-FK edge, and data-derived cardinality all correct.")
