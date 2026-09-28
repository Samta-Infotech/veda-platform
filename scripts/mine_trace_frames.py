"""Stage 0.2 — mine verified (question, SQL) pairs into (question, FRAME) examples.

Sources (the platform DB `veda`, read-only):
  * query_querylog     rows with status answered/ok and a non-empty executed_sql — these
                       passed the firewall (the log is written after it) and executed;
  * chat_chatmessage   explainability messages carrying the executed SQL, joined to the
                       user message of the same turn, and their `feedback` (like = accepted);
  * substrate_verifiedquerycache   verified (question, SQL) pairs.

A frame is derived from each SQL with business_explain.extract_sql_facts (entities, filters,
orderings, limit, aggregations, groupings, distinct) and expressed in CONCEPTS through the
source's entity cards (entity = the card's business name; columns → humanized phrases).

The 20 external questions (question.txt) are EXCLUDED — they are the evaluation set, and
several of their logged answers are known wrong (2/20 correct); using them as few-shots
would both leak and teach wrong frames.

Writes evaluation/frames/<source>.jsonl  ({question, frame, sql, origin, accepted, table}).

Usage (inside the inference container):
    cd /app/veda_core && python /app/scripts/mine_trace_frames.py [--sources 2,4,5]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/veda_core")

_OP = {"EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=", "In": "in",
       "Between": "between", "Is": "is_null", "NotIs": "is_not_null", "Like": "=", "ILike": "="}
_AGG = {"COUNT": "count", "SUM": "sum", "AVG": "avg", "MIN": "min", "MAX": "max"}


def _conn():
    import psycopg2
    return psycopg2.connect(host=os.environ.get("VEDA_INTERNAL_HOST", "pgbouncer"),
                            port=int(os.environ.get("VEDA_INTERNAL_PORT", "6432")),
                            dbname=os.environ.get("POSTGRES_DB", "veda"),
                            user=os.environ.get("POSTGRES_USER", "veda"),
                            password=os.environ.get("POSTGRES_PASSWORD"))


def _h(c):
    return str(c or "").replace("_", " ").strip().lower()


def frame_from_sql(sql, cards):
    from veda.business_explain import extract_sql_facts
    f = extract_sql_facts(sql)
    anchor = f.get("from_table") or f.get("anchor")
    if not anchor:
        return None, None
    card = cards.get(anchor) or {}
    fr = {"entity": card.get("business_name") or _h(anchor), "secondaries": [],
          "measure": None, "aggregation": "none", "filters": [], "group_by": [],
          "order": None, "limit": f.get("limit"), "time": None,
          "distinct": bool(f.get("distinct")), "confidence": 1.0}
    for t in f.get("entities") or []:
        if t != anchor and t in cards:
            fr["secondaries"].append(cards[t].get("business_name") or _h(t))
    aggs = f.get("aggregations") or []
    if aggs:
        fn, col = aggs[0]
        fr["aggregation"] = _AGG.get(str(fn).upper(), "none")
        if fr["aggregation"] == "count" and fr["distinct"] and col:
            fr["aggregation"] = "count_distinct"
        fr["measure"] = _h(col) if col and col != "*" else None
        if fr["limit"] == 100 and not f.get("groupings"):
            fr["limit"] = None
    for col, opc, val in f.get("filters") or []:
        op = _OP.get(opc)
        if not op:
            continue
        if isinstance(val, tuple):
            val = list(val)
        fr["filters"].append({"concept": _h(col), "op": op, "value": val})
    fr["group_by"] = [_h(g) for g in (f.get("groupings") or [])]
    ords = f.get("orderings") or []
    if ords:
        col, desc = ords[0]
        fr["order"] = {"concept": _h(col), "dir": "desc" if desc else "asc"}
    if fr["limit"] == 100 and not fr["order"]:
        fr["limit"] = None          # the engine's default page, not the question's
    return fr, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="2,4,5")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--questions", default="/app/question.txt")
    a = ap.parse_args()
    exclude = set()
    try:
        with open(a.questions) as fh:
            exclude = {l.strip().lower() for l in fh if l.strip() and l.strip().lower() != "question"}
    except Exception:
        pass
    from veda.understanding.vocabulary import load_source
    from veda.runtime import _load_one_sm
    conn = _conn()
    cur = conn.cursor()
    out_dir = "/app/evaluation/frames"
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for sid in [s.strip() for s in a.sources.split(",") if s.strip()]:
        cards = load_source(sid, a.tenant, sm_for_fallback=_load_one_sm(int(sid), a.tenant)).cards
        pairs = []
        cur.execute("SELECT query_text, executed_sql, status FROM query_querylog "
                    "WHERE source_id = %s AND status IN ('answered','ok') AND executed_sql <> '' "
                    "ORDER BY id", (int(sid),))
        pairs += [(q, s, "querylog", False) for q, s, _st in cur.fetchall()]
        cur.execute("SELECT query_text, verified_sql FROM substrate_verifiedquerycache WHERE source_id = %s",
                    (int(sid),))
        pairs += [(q, s, "verified_cache", True) for q, s in cur.fetchall()]
        # chat turns: the explainability message of an answered turn carries its SQL
        try:
            cur.execute("""
                SELECT u.content, e.metadata, COALESCE(a.feedback, '')
                FROM chat_chatmessage e
                JOIN chat_chatmessage u ON u.session_id = e.session_id AND u.type = 'user'
                     AND u.created_at <= e.created_at
                     AND NOT EXISTS (SELECT 1 FROM chat_chatmessage u2 WHERE u2.session_id = e.session_id
                                     AND u2.type = 'user' AND u2.created_at > u.created_at
                                     AND u2.created_at <= e.created_at)
                LEFT JOIN chat_chatmessage a ON a.session_id = e.session_id AND a.type = 'assistant'
                     AND abs(extract(epoch from (a.created_at - e.created_at))) < 5
                WHERE e.type = 'explainability'""")
            for q, meta, fb in cur.fetchall():
                m = meta if isinstance(meta, dict) else {}
                s = ((m.get("sql") or {}).get("query") if isinstance(m.get("sql"), dict) else None) or ""
                srcs = [str(x.get("source_id")) for x in (m.get("sources") or []) if isinstance(x, dict)]
                if s and (not srcs or sid in srcs):
                    pairs.append((q, s, "chat", fb.lower() in ("like", "positive", "up", "thumbs_up")))
        except Exception as e:
            conn.rollback()
            print(f"  chat mining skipped: {type(e).__name__}: {str(e)[:120]}")
        seen, rows, reasons = set(), [], Counter()
        for q, s, origin, accepted in pairs:
            ql = (q or "").strip()
            if not ql or not s:
                continue
            if ql.lower() in exclude:
                reasons["excluded_eval_question"] += 1
                continue
            if ql.lower() in seen:
                reasons["duplicate"] += 1
                continue
            if "%s" in s:
                # the log stores the PARAMETERISED statement; its bound values are not
                # recoverable, and a filter with no value teaches the extractor nothing
                reasons["parameterised_values_lost"] += 1
                continue
            try:
                fr, anchor = frame_from_sql(s, cards)
            except Exception:
                fr, anchor = None, None
            if not fr:
                reasons["unparseable_sql"] += 1
                continue
            if anchor not in cards:
                reasons["anchor_not_in_source"] += 1
                continue
            seen.add(ql.lower())
            rows.append({"question": ql, "frame": fr, "sql": s, "origin": origin,
                         "accepted": bool(accepted), "table": anchor})
        path = os.path.join(out_dir, f"{sid}.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        report[sid] = {"candidates": len(pairs), "written": len(rows),
                       "accepted": sum(1 for r in rows if r["accepted"]),
                       "by_origin": dict(Counter(r["origin"] for r in rows)),
                       "dropped": dict(reasons), "path": path}
        print(json.dumps({sid: report[sid]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
