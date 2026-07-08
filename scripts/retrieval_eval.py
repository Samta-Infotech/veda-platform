"""WP0 — retrieval-quality evaluation harness (before/after number for every later WP).

Runs the ONE canonical column-selection path — ``query/retrieval_select.py:select_retrieval``
— over the golden query set and reports gold-column recall@5 / recall@15 / MRR, gold-table
recall@3, mean candidate-set size, and wall-clock per stage. Writes a JSON report to
``evaluation/results/retrieval_<git-sha>.json``.

Runs in the inference container (needs the engine, models, psycopg2, and the Django
substrate for the ambient (source, tenant) context). The golden set is produced by
``scripts/build_golden_set.py`` from ``VerifiedQueryCache`` — run that first.

Determinism: retrieval reads a fixed index and this script introduces no randomness, so
two runs against the same build + ingestion produce byte-identical reports (the WP0
acceptance gate). The output is sorted and the timestamp lives OUTSIDE the graded numbers.

Usage:
    python scripts/retrieval_eval.py --source-id 1 --tenant default \
        [--golden evaluation/golden_queries.jsonl] [--out PATH] [--label baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# select_retrieval and the engine import their siblings as top-level modules
# (``from config import ...``, ``from query... import ...``), so veda_core must resolve
# FIRST — otherwise the repo-root Django ``config`` package shadows veda_core/config.py
# and every engine import (db_abstraction → ``from config import VEDA_INTERNAL_DB``) breaks.
# Insert veda_core LAST so it lands at sys.path[0]; the repo root stays available (index 1)
# for the best-effort Django context in _setup_django.
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "veda_core"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO)
        ).decode().strip()
    except Exception:
        return "nogit"


def _norm_col(ref: str) -> str:
    """Normalize any column reference to a comparable ``table.col`` (lowercased).

    Gold labels are ``schema.table.col`` (from parsed verified SQL); retrieved columns
    are ``table_name.col_name``. Collapsing both to the trailing two dotted components
    makes them comparable without needing the schema qualifier the engine doesn't carry.
    """
    parts = [p for p in str(ref).lower().split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return parts[0] if parts else ""


def _norm_table(ref: str) -> str:
    parts = [p for p in str(ref).lower().split(".") if p]
    return parts[-1] if parts else ""


def _retrieved_cols(selected) -> list[str]:
    out = []
    for r in selected.columns:
        t = (getattr(r, "table_name", "") or "").lower()
        c = (getattr(r, "col_name", "") or "").lower()
        if t and c:
            out.append(f"{t}.{c}")
    return out


def _recall_at_k(gold: set[str], ranked: list[str], k: int) -> float | None:
    if not gold:
        return None
    topk = set(ranked[:k])
    return len(gold & topk) / len(gold)


def _mrr(gold: set[str], ranked: list[str]) -> float | None:
    if not gold:
        return None
    for i, c in enumerate(ranked, start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def _setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        import django
        django.setup()
    except Exception as e:  # pragma: no cover - context is set best-effort
        print(f"[warn] django.setup() failed ({e}); continuing without substrate context",
              file=sys.stderr)


def _load_golden(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


# ===========================================================================
# Cross-source metrics (Cross-source plan, Phase 6.2)
#
# Four graph/answer-level metrics that the column-recall harness above cannot
# express, computed over evaluation/golden_cross_source.jsonl:
#   1. cross-source JOIN precision — discovered `cross_source_fk` edges audited
#      against the golden join pairs (HIGH-tier precision is the plan's ≥0.9
#      target; ALL-tier reported too since HIGH may be empty on small corpora),
#      plus a negative check that `must_not_link` pairs never reach HIGH.
#   2. entity-linking precision/recall — admitted entity nodes vs the hand
#      labels in evaluation/cross_source_entity_labels.jsonl.
#   3. subgraph source-coverage — for each cross-source query, the fraction of
#      expected_sources represented in the retrieved subgraph.
#   4. end-to-end federated accuracy — one real federated JOIN executed through
#      query/federated_executor over the live sources, asserted to return rows.
# All are read-only and degrade to {ok: false, reason} rather than crashing.
# ===========================================================================

def _norm_pair_member(ref: str) -> str:
    """'maintenance.asset_id@4' -> 'maintenance.asset_id@4' (table.col@src, lowercased).
    Tolerates a missing @src and extra dotted schema qualifiers."""
    ref = str(ref).strip().lower()
    src = ""
    if "@" in ref:
        ref, src = ref.rsplit("@", 1)
    parts = [p for p in ref.split(".") if p]
    tc = ".".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")
    return f"{tc}@{src}" if src else tc


def _pair_key(a: str, b: str) -> frozenset:
    return frozenset({_norm_pair_member(a), _norm_pair_member(b)})


def _load_discovered_cross_edges():
    """[{pair: frozenset, tier, jaccard, containment, a, b}] for every persisted
    cross_source_fk edge, labelled 'table.col@src' on both ends."""
    from ingestion.db_abstraction import (
        get_internal_connection, release_internal_connection)
    conn = get_internal_connection()
    out = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ns.table_name, ns.name, ns.source_id,
                       nd.table_name, nd.name, nd.source_id,
                       (e.attrs::jsonb->>'tier'), (e.attrs::jsonb->>'jaccard'),
                       (e.attrs::jsonb->>'containment')
                FROM graph_edges e
                JOIN graph_nodes ns ON ns.node_id = e.src_node_id
                JOIN graph_nodes nd ON nd.node_id = e.dst_node_id
                WHERE e.edge_type = 'cross_source_fk'
            """)
            for st, sn, ssid, dt, dn, dsid, tier, jac, cont in cur.fetchall():
                a = f"{st}.{sn}@{ssid}"
                b = f"{dt}.{dn}@{dsid}"
                out.append({"pair": _pair_key(a, b), "a": a, "b": b, "tier": tier,
                            "jaccard": float(jac or 0), "containment": float(cont or 0)})
    finally:
        release_internal_connection(conn)
    return out


def eval_join_precision(golden: list[dict], discovered: list[dict]) -> dict:
    gold_pairs, neg_pairs = set(), set()
    for g in golden:
        for e in g.get("gold_cross_source_edges", []) or []:
            gold_pairs.add(_pair_key(e["a"], e["b"]))
        for e in g.get("must_not_link", []) or []:
            neg_pairs.add(_pair_key(e["a"], e["b"]))

    high = [d for d in discovered if (d["tier"] or "").upper() == "HIGH"]
    disc_all_pairs = {d["pair"] for d in discovered}
    disc_high_pairs = {d["pair"] for d in high}

    def _prec(disc_pairs):
        if not disc_pairs:
            return None
        return round(len(disc_pairs & gold_pairs) / len(disc_pairs), 4)

    return {
        "ok": True,
        "n_discovered": len(discovered),
        "n_discovered_high": len(high),
        "n_gold_pairs": len(gold_pairs),
        "gold_pairs_found": sorted("|".join(sorted(p)) for p in (disc_all_pairs & gold_pairs)),
        "recall_gold_pairs": (round(len(disc_all_pairs & gold_pairs) / len(gold_pairs), 4)
                              if gold_pairs else None),
        "precision_high": _prec(disc_high_pairs),          # plan target >= 0.9
        "precision_all_tiers": _prec(disc_all_pairs),      # includes MEDIUM noise
        "negative_violations_high": sorted(
            "|".join(sorted(p)) for p in (disc_high_pairs & neg_pairs)),
        "negative_violations_any": sorted(
            "|".join(sorted(p)) for p in (disc_all_pairs & neg_pairs)),
    }


def eval_entity_linking(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {"ok": False, "reason": f"labels not found: {labels_path}"}
    labels = {}
    for row in _load_golden(labels_path):
        labels[str(row["name"]).lower()] = bool(row.get("is_bridge"))
    from ingestion.db_abstraction import (
        get_internal_connection, release_internal_connection)
    conn = get_internal_connection()
    admitted = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT gn.name, count(e.edge_id)
                           FROM graph_nodes gn
                           LEFT JOIN graph_edges e
                             ON e.src_node_id = gn.node_id AND e.edge_type='value_of'
                           WHERE gn.node_type='entity' GROUP BY gn.name""")
            for name, nlinks in cur.fetchall():
                admitted[str(name).lower()] = int(nlinks)
    finally:
        release_internal_connection(conn)

    tp = sum(1 for n in admitted if labels.get(n) is True)
    fp = sum(1 for n in admitted if labels.get(n) is False)
    unlabeled = [n for n in admitted if n not in labels]
    total_bridges = sum(1 for v in labels.values() if v)
    zero_link_admits = sorted(n for n, k in admitted.items() if k == 0)
    return {
        "ok": True,
        "n_admitted": len(admitted),
        "n_labeled_bridges": total_bridges,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / total_bridges, 4) if total_bridges else None,
        "unlabeled_admitted": sorted(unlabeled),
        "zero_link_admits": zero_link_admits,   # admission-rule violations (must-bridge)
    }


def eval_subgraph_coverage(golden: list[dict], all_source_ids: list[str],
                           name_by_id: dict) -> dict:
    """For each multi-source query, fraction of expected_sources whose columns/chunks
    appear in the retrieved subgraph (retrieval run across ALL ready sources)."""
    from query.retrieval_select import select_retrieval
    id_by_name = {v: k for k, v in name_by_id.items()}
    rows, covs = [], []
    for g in golden:
        exp = [id_by_name.get(s) for s in g.get("expected_sources", [])]
        exp = [s for s in exp if s]
        if len(exp) < 2:            # coverage only meaningful for cross-source queries
            continue
        try:
            sel = select_retrieval(query=g["query"], source_ids=all_source_ids,
                                   intent=g.get("intent", "sql"), verbose=False)
            present = {str(getattr(c, "source_id", "") or "") for c in sel.columns}
            gr = getattr(sel, "graph_result", None)
            for ch in (getattr(gr, "chunks", None) or []):
                present.add(str(getattr(ch, "source_id", "") or ""))
            hit = [s for s in exp if s in present]
            cov = len(hit) / len(exp)
        except Exception as e:
            rows.append({"query": g["query"], "error": str(e)})
            continue
        covs.append(cov)
        rows.append({"query": g["query"], "expected": exp,
                     "present": sorted(present), "coverage": round(cov, 4)})
    return {"ok": True, "n_queries": len(covs),
            "mean_source_coverage": round(sum(covs) / len(covs), 4) if covs else None,
            "per_query": rows}


def eval_federated_e2e(tenant: str) -> dict:
    """Execute one real federated JOIN (maintenance CSV × homzhub assets_asset) through
    the firewall + executor, asserting it validates and returns rows."""
    try:
        from query.cross_source_composer import resolve_surface
        from query.federated_executor import FederatedExecutor, catalog_name
    except Exception as e:
        return {"ok": False, "reason": f"import failed: {e}"}
    s_csv = resolve_surface("4", tenant)   # invoices_csv -> parquet (maintenance, vendors)
    s_pg = resolve_surface("2", tenant)    # homzhub -> postgres
    if not s_csv or not s_pg:
        return {"ok": False, "reason": f"surface unresolved (csv={bool(s_csv)}, pg={bool(s_pg)})"}
    ex = FederatedExecutor([s_csv, s_pg])
    c4, c2 = catalog_name("4"), catalog_name("2")
    sql = (f'SELECT a.city_name, count(*) AS n_tickets, sum(m.amount) AS total_amount '
           f'FROM {c4}.maintenance m '
           f'JOIN {c2}.public.assets_asset a ON CAST(m.asset_id AS BIGINT) = a.id '
           f'GROUP BY a.city_name ORDER BY total_amount DESC')
    try:
        res = ex.execute(sql)
        return {"ok": True, "sql": sql, "row_count": res["row_count"],
                "columns": res["columns"], "catalogs": res.get("catalogs"),
                "sample_rows": res["rows"][:5]}
    except Exception as e:
        return {"ok": False, "reason": str(e), "sql": sql}


def run_cross_source(args) -> int:
    _setup_django()
    from context import RequestContext, set_context
    set_context(RequestContext(source_id=int(args.source_id), tenant=args.tenant))

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[error] golden set not found: {golden_path}", file=sys.stderr)
        return 2
    golden = _load_golden(golden_path)

    # source name<->id map from the registry (for coverage + reporting)
    name_by_id: dict = {}
    try:
        from storage_adapters import reader
        conn = reader._connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, ready FROM sources_source ORDER BY id")
            regrows = cur.fetchall()
        name_by_id = {str(r[0]): r[1] for r in regrows}
        ready_ids = [str(r[0]) for r in regrows if r[2]]
    except Exception as e:
        print(f"[warn] registry read failed ({e}); coverage will be limited", file=sys.stderr)
        ready_ids = [str(args.source_id)]

    discovered = _load_discovered_cross_edges()
    report = {
        "git_sha": _git_sha(),
        "label": args.label,
        "tenant": args.tenant,
        "golden_set": str(golden_path.relative_to(_REPO)) if golden_path.is_relative_to(_REPO)
                      else str(golden_path),
        "ready_sources": {sid: name_by_id.get(sid) for sid in ready_ids},
        "join_precision": eval_join_precision(golden, discovered),
        "entity_linking": eval_entity_linking(
            _REPO / "evaluation" / "cross_source_entity_labels.jsonl"),
        "subgraph_coverage": eval_subgraph_coverage(golden, ready_ids, name_by_id),
        "federated_e2e": eval_federated_e2e(args.tenant),
        "generated_at_epoch": int(time.time()),
    }
    out_path = Path(args.out) if args.out else (
        _REPO / "evaluation" / "results" / f"cross_source_{report['git_sha']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    jp, el, sc, fe = (report["join_precision"], report["entity_linking"],
                      report["subgraph_coverage"], report["federated_e2e"])
    print("[cross_source_eval] join: %d edges (%d HIGH)  prec_high=%s prec_all=%s "
          "recall_gold=%s  neg_violations_high=%d"
          % (jp["n_discovered"], jp["n_discovered_high"], jp["precision_high"],
             jp["precision_all_tiers"], jp["recall_gold_pairs"],
             len(jp["negative_violations_high"])))
    print("[cross_source_eval] entity: admitted=%s bridges=%s prec=%s recall=%s "
          "zero_link_admits=%d" % (el.get("n_admitted"), el.get("n_labeled_bridges"),
          el.get("precision"), el.get("recall"), len(el.get("zero_link_admits", []))))
    print("[cross_source_eval] subgraph coverage (cross-source qs): mean=%s over %s queries"
          % (sc.get("mean_source_coverage"), sc.get("n_queries")))
    print("[cross_source_eval] federated e2e: ok=%s %s"
          % (fe.get("ok"), (("rows=%s" % fe.get("row_count")) if fe.get("ok")
                            else fe.get("reason"))))
    print(f"[cross_source_eval] wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="VEDA retrieval-quality eval (WP0)")
    ap.add_argument("--source-id", type=int, default=int(os.environ.get("VEDA_EVAL_SOURCE_ID", "1")))
    ap.add_argument("--tenant", default=os.environ.get("VEDA_EVAL_TENANT", "default"))
    ap.add_argument("--golden", default=str(_REPO / "evaluation" / "golden_queries.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--cross-source", action="store_true",
                    help="run the Phase 6 cross-source metric suite instead of WP0 column recall")
    args = ap.parse_args()

    if args.cross_source:
        return run_cross_source(args)

    _setup_django()
    from context import RequestContext, set_context  # veda_core/context.py
    set_context(RequestContext(source_id=args.source_id, tenant=args.tenant))

    from query.retrieval_select import select_retrieval

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[error] golden set not found: {golden_path}\n"
              f"        run scripts/build_golden_set.py first.", file=sys.stderr)
        return 2
    golden = _load_golden(golden_path)

    per_query = []
    graded = 0  # queries with gold_columns (recall is defined)
    for g in golden:
        query = g["query"]
        gold_cols = {_norm_col(c) for c in g.get("gold_columns", []) if c}
        gold_tables = {_norm_table(t) for t in g.get("gold_tables", []) if t}
        intent = g.get("intent", "sql")

        t0 = time.perf_counter()
        try:
            selected = select_retrieval(
                query=query, source_ids=[str(args.source_id)], intent=intent, verbose=False,
            )
            err = None
        except Exception as e:  # keep going — a single-query crash shouldn't void the run
            selected = None
            err = str(e)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if selected is None:
            per_query.append({"query": query, "error": err, "elapsed_ms": elapsed_ms})
            continue

        ranked = _retrieved_cols(selected)
        ranked_tables = [(_norm_table(t)) for t in selected.tables]
        has_gold = bool(gold_cols)
        if has_gold:
            graded += 1
        entry = {
            "query": query,
            "intent": intent,
            "expected_kind": g.get("expected_kind"),
            "source": selected.source,
            "short_circuit": selected.short_circuit,
            "n_candidates": len(ranked),
            "recall@5": _recall_at_k(gold_cols, ranked, 5),
            "recall@15": _recall_at_k(gold_cols, ranked, 15),
            "mrr": _mrr(gold_cols, ranked),
            "table_recall@3": (
                len(gold_tables & set(ranked_tables[:3])) / len(gold_tables)
                if gold_tables else None
            ),
            "elapsed_ms": elapsed_ms,
            "stage_stats": selected.stats,
            "graded": has_gold,
        }
        per_query.append(entry)

    graded_rows = [e for e in per_query if e.get("graded")]
    summary = {
        "n_queries": len(golden),
        "n_graded": graded,
        "recall@5": _mean([e["recall@5"] for e in graded_rows]),
        "recall@15": _mean([e["recall@15"] for e in graded_rows]),
        "mrr": _mean([e["mrr"] for e in graded_rows]),
        "table_recall@3": _mean([e["table_recall@3"] for e in graded_rows
                                 if e["table_recall@3"] is not None]),
        "mean_candidate_set_size": _mean([float(e["n_candidates"]) for e in per_query
                                          if "n_candidates" in e]),
        "mean_elapsed_ms": _mean([float(e["elapsed_ms"]) for e in per_query
                                  if "elapsed_ms" in e]),
    }
    # Per-class breakdown so WP acceptance can gate "every query class".
    by_class: dict[str, list[dict]] = {}
    for e in graded_rows:
        by_class.setdefault(e.get("intent", "sql"), []).append(e)
    per_class = {
        cls: {
            "n": len(rows),
            "recall@15": _mean([r["recall@15"] for r in rows]),
            "mrr": _mean([r["mrr"] for r in rows]),
        }
        for cls, rows in sorted(by_class.items())
    }

    sha = _git_sha()
    report = {
        "git_sha": sha,
        "label": args.label,
        "source_id": args.source_id,
        "tenant": args.tenant,
        "golden_set": str(golden_path.relative_to(_REPO)) if golden_path.is_relative_to(_REPO)
                      else str(golden_path),
        "summary": summary,
        "per_class": per_class,
        # graded numbers above are timestamp-free & deterministic; wall-clock stamp is
        # recorded separately so it never affects a diff of the graded report.
        "generated_at_epoch": int(time.time()),
        "per_query": sorted(per_query, key=lambda e: e["query"]),
    }

    out_path = Path(args.out) if args.out else (
        _REPO / "evaluation" / "results" / f"retrieval_{sha}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    s = summary
    print(f"[retrieval_eval] {s['n_graded']}/{s['n_queries']} graded  "
          f"recall@5={s['recall@5']:.4f} recall@15={s['recall@15']:.4f} "
          f"mrr={s['mrr']:.4f} table_recall@3={s['table_recall@3']:.4f}  "
          f"mean|cand|={s['mean_candidate_set_size']:.1f}  "
          f"mean_ms={s['mean_elapsed_ms']:.1f}")
    print(f"[retrieval_eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
