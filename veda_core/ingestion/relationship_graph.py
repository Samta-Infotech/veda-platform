# =============================================================================
# ingestion/relationship_graph.py
# VEDA V1.5 — deterministic relationship graph (the foundation for join planning)
#
# Builds veda_relationship_graph.json at ingestion time. Runtime NEVER infers
# relationships — it only reads this file.
#
# Edge sources (in priority):
#   1. declared FK            (schema is_fk / fk_ref_table / fk_ref_col)
#   2. polymorphic            (object_id + object_type/model_name) — resolved by
#                             DATA CORRELATION, not string-matching, because:
#                               • the join key may be a business key, not the PK
#                                 (annotation_record.object_id = counterparty_details.counterparty_id)
#                               • some discriminator values are categories, not
#                                 entity pointers (e.g. 'SAR', 'Level 1') → non-joinable
#
# Each edge carries: type, weight (deterministic), cardinality (inferred),
# polymorphic flag, requires_predicate, discovery, confidence.
# =============================================================================

import sys
import os
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_primary_relational_source

RELATIONSHIP_GRAPH_FILE = "data/veda_relationship_graph.json"

# deterministic weights by edge type (lower = preferred path)
_WEIGHT = {
    "business_core": 1, "bridge": 1, "reference": 2, "lookup": 2,
    "polymorphic": 2, "data_inferred": 3, "audit": 10, "history": 10,
}
_AUDIT_TABLE_RE = re.compile(r"(_history|_log|_audit|_archive)$", re.I)
# Ownership / "who touched this row" FK columns. These are a near-universal naming
# convention (NOT db-specific), and they all point at the users table — which turns
# `user` into a hub that falsely connects every table to every other table via
# "edited by the same person". Classifying them as audit (weight 10) keeps the edge
# usable for a DIRECT "X and its <owner>" join but stops the planner routing THROUGH
# user to bridge two unrelated business tables (10+10 ≫ any real path).
_AUDIT_COL_RE = re.compile(r"_(by|by_id|by_group)$|^(owned_by|assigned_to)", re.I)
_POLY_VALUE_SAMPLE = 200      # object_id values sampled per discriminator value
_MATCH_FLOOR = 0.80           # min correlation to accept a polymorphic/inferred edge


def _conn():
    cfg = get_primary_relational_source()
    import psycopg2
    return psycopg2.connect(host=cfg["host"], port=cfg["port"], dbname=cfg["dbname"],
                            user=cfg["user"], password=cfg.get("password", ""))


def _q(cur, sql, args=None):
    cur.execute(sql, args or [])
    return cur.fetchall()


def _table_meta(cur, tables):
    """Per table: columns, PK, unique 'key-like' columns, and column data types."""
    meta = {}
    for t in tables:
        rows = _q(cur,
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s", [t])
        cols = [r[0] for r in rows]
        dtypes = {r[0]: r[1] for r in rows}
        pks = [r[0] for r in _q(cur,
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey) "
            "WHERE i.indrelid=%s::regclass AND i.indisprimary", [t])]
        rowcount = _q(cur, f'SELECT count(*) FROM "{t}"')[0][0]
        # key-like = PK + columns that are unique (candidate business keys)
        key_cols = list(pks)
        for c in cols:
            if c in key_cols:
                continue
            # cheap uniqueness check only for id/key/code/no-suffixed columns
            if re.search(r"(_id|_no|_code|_key|_number)$", c) or c.endswith("id"):
                d = _q(cur, f'SELECT count(DISTINCT "{c}") FROM "{t}"')[0][0]
                if rowcount and d >= rowcount * 0.95:
                    key_cols.append(c)
        meta[t] = {"columns": cols, "pk": pks, "rowcount": rowcount,
                   "key_cols": key_cols, "dtypes": dtypes}
    return meta


_STRING_TYPES = ("character varying", "varchar", "text", "char", "uuid", "citext")


def _name_affinity(disc_value, target_table):
    """Does the discriminator value relate to the target table name?
    e.g. 'counterparty' ↔ counterparty_details ✓ ; 'User' ↔ counterparty_details ✗"""
    try:
        from retrieval.query_enrichment import _singularize
    except Exception:
        def _singularize(w): return w.rstrip("s")
    v = _singularize(re.sub(r"[^a-z]", "", str(disc_value).lower()))
    if len(v) < 3:
        return False
    tbl_tokens = {_singularize(tok) for tok in target_table.lower().split("_")}
    return v in tbl_tokens or any(v in tok or tok in v for tok in tbl_tokens)


def _cardinality(cur, child_t, child_col, parent_t, parent_col):
    """1:1 / N:1 / 1:N from distinctness on each side."""
    try:
        ch_rows = _q(cur, f'SELECT count(*) FROM "{child_t}"')[0][0]
        ch_distinct = _q(cur, f'SELECT count(DISTINCT "{child_col}") FROM "{child_t}"')[0][0]
        pa_distinct = _q(cur, f'SELECT count(DISTINCT "{parent_col}") FROM "{parent_t}"')[0][0]
        pa_rows = _q(cur, f'SELECT count(*) FROM "{parent_t}"')[0][0]
        child_unique = ch_rows and ch_distinct >= ch_rows * 0.95
        parent_unique = pa_rows and pa_distinct >= pa_rows * 0.95
        if child_unique and parent_unique:
            return "1:1"
        if parent_unique:
            return "N:1"      # many child rows → one parent
        return "N:M"
    except Exception:
        return "unknown"


def _declared_fk_edges(schema_tables):
    edges = []
    for t in schema_tables:
        tname = t["table_name"]
        for c in t.get("columns", []):
            if c.get("is_fk") and c.get("fk_ref_table"):
                edges.append({
                    "source_table": tname, "source_column": c["col_name"],
                    "target_table": c["fk_ref_table"], "target_column": c.get("fk_ref_col") or "id",
                    "relationship_type": "audit" if (_AUDIT_TABLE_RE.search(tname)
                        or _AUDIT_COL_RE.search(c["col_name"])) else "business_core",
                    "discovery": "declared_fk", "polymorphic": False,
                    "requires_predicate": None, "confidence": 1.0,
                })
    return edges


def _polymorphic_edges(cur, tables, meta):
    """Detect *_id + (*_type|model_name) pairs and resolve each discriminator value
    to a target table.column by DATA CORRELATION (not string matching)."""
    edges = []
    for t in tables:
        cols = meta[t]["columns"]
        id_cols = [c for c in cols if c == "object_id" or c.endswith("_object_id")]
        disc_cols = [c for c in cols if c in ("object_type", "model_name") or c.endswith("_type")]
        if not id_cols or not disc_cols:
            continue
        id_col, disc_col = id_cols[0], disc_cols[0]

        for (val,) in _q(cur, f'SELECT DISTINCT "{disc_col}" FROM "{t}" '
                              f'WHERE "{disc_col}" IS NOT NULL'):
            sample = [r[0] for r in _q(cur,
                f'SELECT "{id_col}" FROM "{t}" WHERE "{disc_col}"=%s '
                f'AND "{id_col}" IS NOT NULL LIMIT {_POLY_VALUE_SAMPLE}', [val])]
            if not sample:
                continue
            sample = [str(x) for x in sample]

            best = None
            for cand_t in tables:
                if cand_t == t:
                    continue
                for kc in meta[cand_t]["key_cols"]:
                    present = _q(cur,
                        f'SELECT count(DISTINCT "{kc}"::text) FROM "{cand_t}" '
                        f'WHERE "{kc}"::text = ANY(%s)', [sample])[0][0]
                    rate = present / len(set(sample))
                    if rate < _MATCH_FLOOR:
                        continue
                    # Collision guard: a numeric surrogate-PK match is only trustworthy
                    # if the discriminator value also NAMES the target table. String/
                    # business-key matches are collision-resistant on their own.
                    is_string_key = meta[cand_t]["dtypes"].get(kc, "") in _STRING_TYPES
                    affinity = _name_affinity(val, cand_t)
                    if not (is_string_key or affinity):
                        continue   # reject numeric-overlap-without-name-affinity (User→cp.id)
                    score = rate + (0.5 if affinity else 0) + (0.3 if is_string_key else 0)
                    if best is None or score > best["_score"]:
                        best = {"target_table": cand_t, "target_column": kc,
                                "confidence": round(rate, 3), "_score": score}
            if best:
                best.pop("_score", None)

            if best:
                edges.append({
                    "source_table": t, "source_column": id_col,
                    "target_table": best["target_table"], "target_column": best["target_column"],
                    "relationship_type": "polymorphic", "discovery": "data_inferred",
                    "polymorphic": True,
                    "requires_predicate": f"{t}.{disc_col} = '{val}'",
                    "confidence": best["confidence"],
                })
            # else: discriminator value is categorical (e.g. 'SAR','Level 1') → no edge
    return edges


def build_relationship_graph(tables=None, verbose=False):
    """Build and persist veda_relationship_graph.json for the given tables
    (default: the tables in the current semantic model)."""
    from schema.real_schema import get_real_schema
    raw = get_real_schema()
    schema_tables = raw.get("tables", [])
    by_name = {t["table_name"]: t for t in schema_tables}

    if tables is None:
        # default to whatever the semantic model covers
        try:
            from config import SEMANTIC_MODEL_FILE
            sm = json.load(open(SEMANTIC_MODEL_FILE))
            tables = sorted(sm.get("tables", {}).keys())
        except Exception:
            tables = list(by_name.keys())
    tables = [t for t in tables if t in by_name]

    conn = _conn()
    cur = conn.cursor()
    try:
        meta = _table_meta(cur, tables)

        edges = _declared_fk_edges([by_name[t] for t in tables])
        # keep only edges whose both ends are in scope
        edges = [e for e in edges if e["target_table"] in tables]
        edges += _polymorphic_edges(cur, tables, meta)

        # cardinality + weight per edge
        for e in edges:
            e["cardinality"] = _cardinality(cur, e["source_table"], e["source_column"],
                                            e["target_table"], e["target_column"])
            e["weight"] = _WEIGHT.get(e["relationship_type"], 3)

        graph = {
            "tables": tables,
            "edges": edges,
            "stats": {
                "num_tables": len(tables), "num_edges": len(edges),
                "declared": sum(1 for e in edges if e["discovery"] == "declared_fk"),
                "polymorphic": sum(1 for e in edges if e["polymorphic"]),
            },
        }
    finally:
        cur.close(); conn.close()

    os.makedirs(os.path.dirname(RELATIONSHIP_GRAPH_FILE) or ".", exist_ok=True)
    json.dump(graph, open(RELATIONSHIP_GRAPH_FILE, "w"), indent=2)
    if verbose:
        print(json.dumps(graph["stats"], indent=2))
    return graph


if __name__ == "__main__":
    g = build_relationship_graph(verbose=True)
    print(f"\n✓ wrote {RELATIONSHIP_GRAPH_FILE}: {g['stats']}")
    for e in g["edges"]:
        pred = f"  [{e['requires_predicate']}]" if e["requires_predicate"] else ""
        print(f"  {e['source_table']}.{e['source_column']} → "
              f"{e['target_table']}.{e['target_column']}  "
              f"({e['relationship_type']}, {e['cardinality']}, w={e['weight']}, "
              f"conf={e['confidence']}){pred}")
