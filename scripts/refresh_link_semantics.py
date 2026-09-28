"""Apply the link-table / FK-relationship sentences (ingestion/link_text.py) to an ALREADY
ingested source, touching only what they change (inference container):

    cd /app/veda_core && python /app/scripts/refresh_link_semantics.py --source 2 [--dry-run]
    cd /app/veda_core && python /app/scripts/refresh_link_semantics.py --source 2 --revert <backup.jsonl>

1. entity cards: link cards + split run-together names (vocabulary.apply_link_semantics)
2. column_embeddings_v2 / column_sparse_v1: every declared-FK column's passage gains
   "RELATIONSHIP: the user this ticket is assigned to" → re-embedded (dense + sparse)
3. table_embeddings_v2 / table_sparse_v1: every link table's passage gains its link sentence
4. rerank docs rebuilt (they read the same sentences)
5. synthetic questions: the changed cards' template questions regenerated + embedded
6. routing card rebuilt from the cards

Every other row stays byte-identical (a full re-ingest would re-embed all 1,902 columns).
The old text + vectors of every touched row are written to a backup first
(reports/raw/agent2/link_refresh_backup_<source>.jsonl) and --revert restores them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

MARK = "\nRELATIONSHIP: "
TMARK = ". LINK TABLE: "


def _strip(text: str, mark: str) -> str:
    i = text.find(mark)
    return text if i < 0 else text[:i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", default=None)
    ap.add_argument("--backup-dir", default="/app/reports/raw/agent2")
    a = ap.parse_args()
    sid, tenant = str(a.source), a.tenant
    from config import (BIENCODER_COL_TABLE, BIENCODER_TABLE_TABLE, COLUMN_SPARSE_TABLE,
                        TABLE_SPARSE_TABLE, BIENCODER_PASSAGE_PREFIX, source_artifact_path,
                        resolve_source_artifact)
    from ingestion.biencoder import _get_pg_conn
    conn = _get_pg_conn()
    cur = conn.cursor()

    if a.revert:
        n = 0
        for line in open(a.revert):
            r = json.loads(line)
            if r["kind"] == "col":
                cur.execute(f"UPDATE {BIENCODER_COL_TABLE} SET text=%s, embedding=%s::vector "
                            f"WHERE source_id=%s AND table_name=%s AND col_name=%s",
                            (r["text"], r["embedding"], sid, r["table"], r["col"]))
                cur.execute(f"UPDATE {COLUMN_SPARSE_TABLE} SET weights=%s WHERE source_id=%s AND col_id=%s",
                            (json.dumps(r["sparse"]), sid, f"{r['table']}.{r['col']}"))
            elif r["kind"] == "table":
                cur.execute(f"UPDATE {BIENCODER_TABLE_TABLE} SET text=%s, embedding=%s::vector "
                            f"WHERE source_id=%s AND table_name=%s", (r["text"], r["embedding"], sid, r["table"]))
                cur.execute(f"UPDATE {TABLE_SPARSE_TABLE} SET weights=%s WHERE source_id=%s AND table_name=%s",
                            (json.dumps(r["sparse"]), sid, r["table"]))
            elif r["kind"] == "file":
                open(r["path"], "wb").write(bytes.fromhex(r["hex"]))
            n += 1
        conn.commit()
        print(f"reverted {n} rows/files")
        return

    from ingestion.vocabulary import (_read_json, _write_json, _graph, apply_link_semantics,
                                      template_questions, embed_questions, routing_card_from_cards,
                                      CARDS_ARTIFACT, QUESTIONS_ARTIFACT, QUESTIONS_EMB_ARTIFACT,
                                      VALUE_GLOSSARY_ARTIFACT, MEASURE_GLOSSARY_ARTIFACT)
    from ingestion.link_text import link_tables, fk_phrases, table_sentence
    from ingestion import m3_encoder
    t0 = time.time()
    sm = _read_json(resolve_source_artifact("veda_semantic_model.json", sid, tenant)) or {}
    graph = _graph(sid, tenant)
    cpath = source_artifact_path(CARDS_ARTIFACT, sid, tenant)
    raw_cards = _read_json(cpath) or {}
    about = {k: v for k, v in raw_cards.items() if str(k).startswith("_")}
    cards = {k: v for k, v in raw_cards.items() if not str(k).startswith("_")}
    changed = apply_link_semantics(cards, sm, graph)
    links = link_tables(sm, graph, cards)
    phrases = fk_phrases(sm, graph, cards, links)
    print(f"[refresh] source {sid}: {len(links)} link tables, {len(phrases)} FK phrases, "
          f"{len(changed)} cards changed")
    if a.dry_run:
        for t in sorted(links)[:10]:
            print("  link", t, links[t]["one_row_is"], links[t]["aliases"][:5])
        return

    os.makedirs(a.backup_dir, exist_ok=True)
    bpath = os.path.join(a.backup_dir, f"link_refresh_backup_{sid}.jsonl")
    bak = open(bpath, "w")

    # files first (cards, questions, emb, routing card, rerank docs)
    files = [cpath, source_artifact_path(QUESTIONS_ARTIFACT, sid, tenant),
             source_artifact_path(QUESTIONS_EMB_ARTIFACT, sid, tenant),
             source_artifact_path("veda_routing_card.json", sid, tenant),
             source_artifact_path("veda_rerank_docs.json", sid, tenant)]
    for f in files:
        if os.path.exists(f):
            bak.write(json.dumps({"kind": "file", "path": f, "hex": open(f, "rb").read().hex()}) + "\n")

    # ── columns ──
    cur.execute(f"SELECT table_name, col_name, text, embedding::text FROM {BIENCODER_COL_TABLE} WHERE source_id=%s", (sid,))
    rows = {(t, c): (txt, emb) for t, c, txt, emb in cur.fetchall()}
    cur.execute(f"SELECT col_id, weights FROM {COLUMN_SPARSE_TABLE} WHERE source_id=%s", (sid,))
    sparse = {k: w for k, w in cur.fetchall()}
    todo = []
    for key, ph in phrases.items():
        t, c = key.split(".", 1)
        if (t, c) not in rows:
            continue
        old = rows[(t, c)][0] or ""
        new = _strip(old, MARK) + MARK + ph
        if new != old:
            todo.append((t, c, old, rows[(t, c)][1], new))
    print(f"[refresh] {len(todo)} column passages change")
    if todo:
        dense = m3_encoder.encode_dense([x[4] for x in todo])
        sw = m3_encoder.encode_sparse([x[4] for x in todo])
        for (t, c, old, emb, new), d, w in zip(todo, dense, sw):
            bak.write(json.dumps({"kind": "col", "table": t, "col": c, "text": old, "embedding": emb,
                                  "sparse": sparse.get(f"{t}.{c}")}) + "\n")
            cur.execute(f"UPDATE {BIENCODER_COL_TABLE} SET text=%s, embedding=%s::vector "
                        f"WHERE source_id=%s AND table_name=%s AND col_name=%s",
                        (new, str(list(map(float, d))), sid, t, c))
            cur.execute(f"UPDATE {COLUMN_SPARSE_TABLE} SET weights=%s WHERE source_id=%s AND col_id=%s",
                        (json.dumps(w), sid, f"{t}.{c}"))

    # ── tables ──
    cur.execute(f"SELECT table_name, text, embedding::text FROM {BIENCODER_TABLE_TABLE} WHERE source_id=%s", (sid,))
    trows = {t: (txt, emb) for t, txt, emb in cur.fetchall()}
    cur.execute(f"SELECT table_name, weights FROM {TABLE_SPARSE_TABLE} WHERE source_id=%s", (sid,))
    tsparse = {k: w for k, w in cur.fetchall()}
    ttodo = []
    for t in links:
        if t not in trows:
            continue
        old = trows[t][0] or ""
        new = _strip(old, TMARK) + TMARK + table_sentence(t, links)[len("LINK TABLE: "):]
        if new != old:
            ttodo.append((t, old, trows[t][1], new))
    print(f"[refresh] {len(ttodo)} table passages change")
    if ttodo:
        dense = m3_encoder.encode_dense([x[3] for x in ttodo])
        sw = m3_encoder.encode_sparse([x[3] for x in ttodo])
        for (t, old, emb, new), d, w in zip(ttodo, dense, sw):
            bak.write(json.dumps({"kind": "table", "table": t, "text": old, "embedding": emb,
                                  "sparse": tsparse.get(t)}) + "\n")
            cur.execute(f"UPDATE {BIENCODER_TABLE_TABLE} SET text=%s, embedding=%s::vector "
                        f"WHERE source_id=%s AND table_name=%s", (new, str(list(map(float, d))), sid, t))
            cur.execute(f"UPDATE {TABLE_SPARSE_TABLE} SET weights=%s WHERE source_id=%s AND table_name=%s",
                        (json.dumps(w), sid, t))
    conn.commit()
    bak.close()

    # ── cards, questions, routing card, rerank docs ──
    _write_json(cpath, {**about, **cards})
    qpath = source_artifact_path(QUESTIONS_ARTIFACT, sid, tenant)
    epath = source_artifact_path(QUESTIONS_EMB_ARTIFACT, sid, tenant)
    qs = [json.loads(l) for l in open(qpath)] if os.path.exists(qpath) else []
    import numpy as np
    emb = np.load(epath) if os.path.exists(epath) else None
    keep = [i for i, q in enumerate(qs) if not (q.get("table") in changed and q.get("origin") == "template")]
    vg = {k: v for k, v in (_read_json(source_artifact_path(VALUE_GLOSSARY_ARTIFACT, sid, tenant)) or {}).items()
          if not str(k).startswith("_")}
    mg = {k: v for k, v in (_read_json(source_artifact_path(MEASURE_GLOSSARY_ARTIFACT, sid, tenant)) or {}).items()
          if not str(k).startswith("_")}
    newq = [q for q in template_questions({t: cards[t] for t in changed if t in cards}, vg, mg)]
    qs2 = [qs[i] for i in keep] + newq
    new_emb = embed_questions(newq) if newq else None
    if emb is not None and emb.shape[0] == len(qs):
        parts = [emb[keep]] + ([np.asarray(new_emb, dtype=emb.dtype)] if new_emb is not None else [])
        emb2 = np.vstack(parts)
        with open(epath + ".tmp", "wb") as f:
            np.save(f, emb2)
        os.replace(epath + ".tmp", epath)
    with open(qpath + ".tmp", "w") as f:
        for q in qs2:
            f.write(json.dumps(q, default=str) + "\n")
    os.replace(qpath + ".tmp", qpath)
    print(f"[refresh] questions: {len(qs)} → {len(qs2)} ({len(newq)} regenerated for {len(changed)} cards)")
    routing_card_from_cards(sid, tenant, cards, qs2)
    try:
        if not sm.get("tables"):
            raise RuntimeError("no semantic model for this source — its rerank docs are not ours to write")
        from ingestion.rerank_docs import build_rerank_docs
        print("[refresh] rerank docs", build_rerank_docs(source_id=sid, tenant=tenant, semantic_model=sm))
    except Exception as e:
        print(f"[refresh] rerank docs skipped: {e}")
    print(f"[refresh] done in {time.time() - t0:.1f}s; backup {bpath}")


if __name__ == "__main__":
    main()
