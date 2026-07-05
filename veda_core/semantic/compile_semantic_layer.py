#!/usr/bin/env python3
# =============================================================================
# semantic/compile_semantic_layer.py
# VEDA — offline compiler for the deterministic semantic layer (Phase 1 slice).
#
# Pure function of data/veda_semantic_model.json. Emits three static, versioned
# registries the runtime fast path resolves against — NO LLM, NO retrieval:
#
#   semantic/concepts.json    business nouns  → entity table   (e.g. "case" → incident)
#   semantic/dimensions.json  groupable cols  + their values   (e.g. incident_status)
#   semantic/metrics.json     named counts/aggregations + grain (e.g. incident_count)
#
# Run once per ingestion:   python3 -m semantic.compile_semantic_layer
# Inspect without writing:  python3 -m semantic.compile_semantic_layer --dry-run
#
# Every artifact is stamped with source_hash (of the semantic model that produced
# it) so the runtime can assert it isn't serving a stale registry.
# =============================================================================

import os
import json
import hashlib

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
SEMANTIC_MODEL_FILE = os.path.join(_ROOT, "data", "veda_semantic_model.json")
CONCEPTS_FILE   = os.path.join(_HERE, "concepts.json")
DIMENSIONS_FILE = os.path.join(_HERE, "dimensions.json")
METRICS_FILE    = os.path.join(_HERE, "metrics.json")
MANIFEST_FILE   = os.path.join(_HERE, "MANIFEST.json")

VERSION = "1.0"

# Entity tables get a COUNT metric + a concept; bridge tables never do (they are
# join glue, never the subject of a question).
_ENTITY_TABLE_TYPES = {"TRANSACTION", "MASTER", "EVENT", "REFERENCE"}

# VEDA's own internal stores must NEVER become queryable business concepts.
# They leaked into the semantic model once (table_embeddings showed up in the
# discoverability report) — guard here so a scan-hygiene slip upstream cannot
# turn the embedding index into an answerable entity.
_VEDA_INTERNAL_TABLES = {
    "table_embeddings", "column_embeddings", "column_embeddings_lt",
    "column_embeddings_hybrid", "column_embeddings_v2", "table_embeddings_v2",
    "doc_chunks", "fk_adjacency", "table_metadata", "column_values",
    "graph_nodes", "graph_edges", "graph_node_embeddings", "source_registry",
}


def _is_internal(table: str) -> bool:
    return table in _VEDA_INTERNAL_TABLES

# Tokens stripped when turning a snake_case table name into matchable entity tokens.
# Connectives + ultra-generic suffixes only — leave real entity words intact so
# "counterparty_details" still matches on "counterparty".
_STRIP_NAME_TOKS = {"and", "or", "of", "to", "by", "the",
                    "master", "mapping", "cfg", "config", "data", "tbl", "table"}


def _singularize(word: str) -> str:
    # Reuse the project's enrichment singularizer so match-time and compile-time
    # tokenization agree exactly; fall back to a tiny rule set if unavailable.
    try:
        from retrieval.query_enrichment import _singularize as _s
        return _s(word)
    except Exception:
        w = word.lower()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ses") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w


def _name_tokens(table_name: str) -> list:
    toks = [_singularize(t) for t in table_name.split("_")
            if len(t) > 2 and t.lower() not in _STRIP_NAME_TOKS]
    # de-dup, keep order
    return list(dict.fromkeys(toks))


def _source_hash(model: dict) -> str:
    blob = json.dumps(model, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dimensions(sm: dict) -> dict:
    """One entry per DIMENSION column: labels (aliases) + known values + owner."""
    cols = sm.get("columns", {})
    out = {}
    for col_id, c in cols.items():
        if c.get("analytics_role") != "DIMENSION":
            continue
        if _is_internal(c.get("table_name", "")):
            continue
        tname = c.get("table_name", "")
        cname = c.get("col_name", "")
        labels = [l.lower() for l in (c.get("aliases") or []) if l]
        labels.append(cname.replace("_", " "))
        # Contentless single-word aliases ('is' sits on 53 dimensions) can only
        # mis-match grouping; real labels like 'status' stay.
        _junk = {"is", "has", "the", "and", "or", "of", "in", "on", "for",
                 "with", "a", "an", "to", "by"}
        labels = [l for l in dict.fromkeys(labels) if l not in _junk]
        # PII / value-handling gate: the semantic model already decided per column
        # whether raw values may be persisted (value_handling) and whether the
        # column is PII (contains_pii). The registry must honor that decision —
        # first_name / party_name sample values must never land in dimensions.json.
        # A redacted dimension stays groupable/filterable; it just offers no value
        # list, so value-matching falls through to the (DB-probed) main pipeline.
        _values_ok = (not c.get("contains_pii")
                      and c.get("value_handling", "keep") in ("keep", "", None))
        out[col_id] = {
            "dimension_id":        col_id,
            "owner_table":         tname,
            "column":              col_id,
            "col_name":            cname,
            "labels":              labels,
            "business_definition": c.get("business_definition", ""),
            "semantic_type":       c.get("semantic_type", ""),
            "values":              ([str(v) for v in (c.get("sample_values") or [])]
                                    if _values_ok else []),
            "values_redacted":     not _values_ok,
            "groupable":           bool((c.get("sql_usage") or {}).get("groupable", True)),
            "filterable":          bool((c.get("sql_usage") or {}).get("filterable", True)),
            "confidence":          c.get("confidence", 1.0),
        }
    return out


def build_concepts(sm: dict) -> dict:
    """One entry per entity table: business-noun labels → table + default count."""
    tabs = sm.get("tables", {})
    cols = sm.get("columns", {})
    has_id = {f"{c.get('table_name')}" for k, c in cols.items() if c.get("col_name") == "id"}

    # display columns: prefer a name/identity-ish column, else the incident_no-style key
    def _display_cols(t):
        picks = []
        for k, c in cols.items():
            if c.get("table_name") != t:
                continue
            role = (c.get("business_role") or "").lower()
            name = c.get("col_name", "")
            if name in ("name", "title") or role in ("name", "identifier") or name.endswith("_no"):
                picks.append(k)
        return picks[:3]

    out = {}
    for t, m in tabs.items():
        if m.get("table_type") not in _ENTITY_TABLE_TYPES or _is_internal(t):
            continue
        toks = _name_tokens(t)
        if not toks:
            continue
        labels = [t.replace("_", " ")] + toks
        labels = list(dict.fromkeys(l.lower() for l in labels))
        out[t] = {
            "concept_id":      t,
            "labels":          labels,
            "match_tokens":    toks,
            "resolves_to":     {"type": "ENTITY", "table": t,
                                 "primary_key": f"{t}.id" if t in has_id else None},
            "primary_entity":  m.get("primary_entity", ""),
            "business_domain": (next((c.get("business_domain", "")
                                      for k, c in cols.items() if c.get("table_name") == t), "")),
            "default_metric":  f"{t}_count",
            "default_display_columns": _display_cols(t),
            "table_type":      m.get("table_type", ""),
        }
    return out


def build_metrics(sm: dict, concepts: dict, dimensions: dict):
    """COUNT metric per entity + SUM/AVG metrics for MEASURE columns.

    Phase-1 metrics are single-table by construction (the fast path never joins),
    so COUNT(*) is fan-out-safe; COUNT(DISTINCT pk) is used when a pk is known as a
    defensive default for any future sliced use.

    GRAIN SAFETY: a table holding several entries per entity (counterparty_details:
    id = entry, counterparty_id = counterparty) makes a row count a WRONG entity
    count. Names alone cannot prove which situation holds, and COUNT(DISTINCT key)
    silently drops NULL keys — so this compiler only NOMINATES suspects
    (grain_suspect=true; expression UNCHANGED) and emits the DB probe queries to
    the MANIFEST. The fast path declines suspect metrics; the real decision is
    made on live data or via a human override.

    Returns (metrics, grain_report)."""
    tabs = sm.get("tables", {})
    cols = sm.get("columns", {})
    out = {}
    grain_report = {}

    # FK source columns from the relationship graph (when present): a *_id column
    # that is a known FK to ANOTHER table is not an entity-key candidate.
    fk_sources = set()
    try:
        _gpath = os.path.join(_ROOT, "data", "veda_relationship_graph.json")
        if os.path.exists(_gpath):
            for e in json.load(open(_gpath)).get("edges", []):
                if e.get("source_table") != e.get("target_table"):
                    fk_sources.add((e["source_table"], e["source_column"]))
    except Exception:
        pass

    def _grain_suspect(t):
        """Nominate (never decide): table has an 'id' pk AND a non-FK *_id column
        whose name overlaps the table's own name → entries may repeat per entity."""
        tcols = {c.get("col_name"): c for c in cols.values() if c.get("table_name") == t}
        if "id" not in tcols:
            return None
        ttoks = [x for x in t.split("_") if len(x) > 3]
        for cn in tcols:
            if (cn.endswith("_id") and cn != "id" and (t, cn) not in fk_sources
                    and any(tok in cn for tok in ttoks)):
                return cn
        return None

    def _soft_delete_filter(t):
        """Conventional live-rows predicate — STORED ONLY, never applied silently.
        Runtime application is gated by COUNT_EXCLUDE_SOFT_DELETED (default off)."""
        tcols = {c.get("col_name") for c in cols.values() if c.get("table_name") == t}
        if "is_deleted" in tcols:
            return '"is_deleted" = false'
        if "deleted_datetime" in tcols:
            return '"deleted_datetime" IS NULL'
        if "deleted_at" in tcols:
            return '"deleted_at" IS NULL'
        return None

    dims_by_table = {}
    for d in dimensions.values():
        dims_by_table.setdefault(d["owner_table"], []).append(d["dimension_id"])

    def _time_dim(t):
        # Only a real date/timestamp column (semantic_type TEMPORAL) is usable as a time
        # dimension. A bigint-epoch column is often listed as a temporal candidate but
        # typed METRIC; selecting it builds date-literal comparisons that fail at execution
        # ("invalid input syntax for type bigint"). No TEMPORAL candidate → None, so date
        # queries refuse rather than emit broken SQL.
        cand = (tabs.get(t, {}) or {}).get("candidate_temporal_columns") or []
        for c in cand:
            cid = f"{t}.{c}"
            if cid in cols and (cols[cid].get("semantic_type") or "").upper() == "TEMPORAL":
                return cid
        # FALLBACK: candidate_temporal_columns sometimes lists ONLY the bigint-epoch
        # variants (typed METRIC, correctly rejected above) while the real datetime column
        # exists but was never listed — an ingestion inconsistency that left 10 tables
        # (permission, role, user, …) with no time dimension, so "X added last month"
        # count queries fell through to the LLM (which hallucinated a 'created_at' column).
        # Scan the table's actual TEMPORAL columns and prefer a CREATION timestamp (that is
        # what "added"/"created" means); avoid soft-delete columns. Schema decides, no lists.
        # CONSERVATIVE: a COUNT metric's time dimension answers "how many X added/created
        # in <period>", so ONLY a genuine CREATION timestamp is correct. Picking any other
        # temporal column (an updated/deleted ts, or a domain date like a lookback window)
        # would produce a confident WRONG temporal count — the silent-wrong-answer we refuse
        # to risk. No clear creation column → return None (fall through; never guess).
        creation = [k for k, m in cols.items()
                    if k.startswith(t + ".")
                    and (m.get("semantic_type") or "").upper() == "TEMPORAL"
                    and any(w in k.split(".", 1)[1].lower() for w in ("creat", "added", "joined"))]
        return creation[0] if creation else None

    # ── entity COUNT metrics ──────────────────────────────────────────────
    for t, concept in concepts.items():
        pk = concept["resolves_to"].get("primary_key")
        expr = f"COUNT(DISTINCT {pk})" if pk else "COUNT(*)"
        mid = f"{t}_count"
        # Labels cover the full entity phrase AND each distinctive token, so "count of
        # counterparties" and "how many counterparty details" both surface this metric.
        # (COUNT metrics are primarily resolved via the entity concept at runtime; these
        # labels are a secondary convenience.)
        entity_words = [t.replace("_", " ")] + concept["match_tokens"]
        labels = [mid.replace("_", " ")]
        for w in entity_words:
            labels += [f"number of {w}", f"count of {w}", f"how many {w}", f"total {w}"]

        ek = _grain_suspect(t)
        if ek:
            grain_report[t] = {
                "entity_key_candidate": f"{t}.{ek}",
                "row_pk": f"{t}.id",
                "probes": [
                    f'SELECT COUNT(*) FROM "{t}"',
                    f'SELECT COUNT(DISTINCT "id") FROM "{t}"',
                    f'SELECT COUNT(DISTINCT "{ek}") FROM "{t}"',
                    f'SELECT COUNT(*) FROM "{t}" WHERE "{ek}" IS NULL',
                ],
                "decision_rule": "all four equal-ish → 1:1, keep row count; "
                                 "distinct entity_key < distinct id and no NULLs → "
                                 "entity grain differs, decide via override; "
                                 "NULLs present → human must decide orphan policy",
            }
        out[mid] = {
            "metric_id":            mid,
            "kind":                 "COUNT",
            "labels":               list(dict.fromkeys(l.lower() for l in labels)),
            "entity_concept":       t,
            "expression":           expr,
            "source_table":         t,
            "grain":                pk or f"{t}.*",
            "grain_suspect":        bool(ek),
            "entity_key_candidate": f"{t}.{ek}" if ek else None,
            "soft_delete_filter":   _soft_delete_filter(t),
            "default_filters":      [],
            "allowed_dimensions":   dims_by_table.get(t, []),
            "allowed_time_dimension": _time_dim(t),
            "aggregation":          "COUNT_DISTINCT" if pk else "COUNT",
            "fanout_safe":          True,
        }

    # ── MEASURE column metrics (SUM/AVG) ──────────────────────────────────
    for col_id, c in cols.items():
        if c.get("analytics_role") != "MEASURE":
            continue
        t = c.get("table_name", "")
        if t not in concepts:           # only over recognised entities
            continue
        cname = c.get("col_name", "")
        aggs = [a.upper() for a in (c.get("allowed_aggregations") or [])]
        defn = (c.get("business_definition") or cname.replace("_", " ")).lower()
        for agg, verb in (("SUM", "total"), ("AVG", "average")):
            if agg not in aggs:
                continue
            mid = f"{agg.lower()}_{t}_{cname}"
            out[mid] = {
                "metric_id":            mid,
                "kind":                 agg,
                "labels":               [f"{verb} {cname.replace('_',' ')}",
                                          f"{verb} {defn}"],
                "entity_concept":       t,
                "expression":           f"{agg}({col_id})",
                "source_table":         t,
                "grain":                concepts[t]["resolves_to"].get("primary_key") or f"{t}.*",
                # SUM/AVG over a duplicate-entry table double-counts exactly like
                # COUNT does — suspect tables decline the fast path for these too.
                "grain_suspect":        t in grain_report,
                "soft_delete_filter":   _soft_delete_filter(t),
                "default_filters":      [],
                "allowed_dimensions":   dims_by_table.get(t, []),
                "allowed_time_dimension": _time_dim(t),
                "aggregation":          agg,
                "fanout_safe":          True,
            }
    return out, grain_report


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

def compile_all(write: bool = True) -> dict:
    with open(SEMANTIC_MODEL_FILE) as f:
        sm = json.load(f)

    src_hash   = _source_hash(sm)
    dimensions = build_dimensions(sm)
    concepts   = build_concepts(sm)
    metrics, grain_report = build_metrics(sm, concepts, dimensions)

    def _stamp(payload, kind):
        return {"version": VERSION, "kind": kind, "source_hash": src_hash,
                "count": len(payload), "items": payload}

    artifacts = {
        CONCEPTS_FILE:   _stamp(concepts,   "concepts"),
        DIMENSIONS_FILE: _stamp(dimensions, "dimensions"),
        METRICS_FILE:    _stamp(metrics,    "metrics"),
    }
    manifest = {
        "version": VERSION, "source_hash": src_hash,
        "source_model": os.path.relpath(SEMANTIC_MODEL_FILE, _ROOT),
        "stats": {"concepts": len(concepts), "dimensions": len(dimensions),
                  "metrics": len(metrics),
                  "grain_suspects": len(grain_report),
                  "redacted_dimensions": sum(1 for d in dimensions.values()
                                             if d.get("values_redacted")),
                  "soft_delete_metrics": sum(1 for m in metrics.values()
                                             if m.get("soft_delete_filter"))},
        # Run these on the live DB; each suspect becomes an evidence-based decision
        # (keep row count / switch to entity grain via override / orphan policy).
        # Until decided, the fast path DECLINES these metrics — never a wrong count.
        "grain_suspects": grain_report,
    }

    # Discoverability: columns with thin metadata are the ones queries can't find
    # ("nature of business" → counterparty_supplementary_info). This is the
    # metadata-enrichment backlog, surfaced as data instead of production misses.
    low = []
    for col_id, c in sm.get("columns", {}).items():
        score = sum([bool(c.get("business_definition")),
                     len(c.get("aliases") or []) >= 2,
                     bool(c.get("user_query_patterns")),
                     bool(c.get("business_role"))])
        if score <= 1:
            low.append({"column": col_id, "score": score})
    low.sort(key=lambda x: x["score"])
    manifest["low_discoverability"] = {"count": len(low), "columns": low[:50]}

    if write:
        for path, payload in artifacts.items():
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

    return {"concepts": concepts, "dimensions": dimensions,
            "metrics": metrics, "manifest": manifest}


def _main():
    import sys
    dry = "--dry-run" in sys.argv
    r = compile_all(write=not dry)
    m = r["manifest"]["stats"]
    print(f"  source_hash={r['manifest']['source_hash']}  "
          f"concepts={m['concepts']}  dimensions={m['dimensions']}  metrics={m['metrics']}")
    if dry:
        print("\n  [dry-run] no files written. Samples:")
        cs = list(r["concepts"].values())[:3]
        for c in cs:
            print(f"    CONCEPT {c['concept_id']:<24} labels={c['labels'][:4]} → {c['resolves_to']}")
        for mk in ("incident_count", "counterparty_details_count"):
            if mk in r["metrics"]:
                mm = r["metrics"][mk]
                print(f"    METRIC  {mk:<28} {mm['expression']:<28} labels={mm['labels'][:3]}")
        ds = [d for d in r["dimensions"].values() if "status" in d["col_name"]][:2]
        for d in ds:
            print(f"    DIM     {d['dimension_id']:<28} values={d['values'][:5]}")
    else:
        print(f"  ✓ wrote concepts.json, dimensions.json, metrics.json, MANIFEST.json to {_HERE}")


if __name__ == "__main__":
    _main()
