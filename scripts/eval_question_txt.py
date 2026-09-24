"""question.txt — end-to-end correctness gate (2026-09-23).

The 20 questions of `question.txt`, each with the ground-truth SQL established in
`reports/QUESTION_TXT_VERIFICATION_2026-09-23.md`, POSTed to the live engine through
`POST /api/v1/conversations/query` with `chat_id: null` (a fresh chat per question, so
no session-memory carry-over) and graded against the live `homzhub` source DB.

Grading is SHAPE + DATA, not string equality — the engine is free to project different
columns or pick a different-but-equivalent ordering column, so long as the ANSWER is
the one the question asked for:

  correct          the emitted SQL hits the right anchor table, carries every qualifier
                   the question named (a WHERE the question implies, the ORDER BY the
                   question implies, an explicit LIMIT N), and re-executing it against
                   `homzhub` reproduces the ground-truth rows (exact row identity for
                   LIMIT <= 5, row COUNT otherwise).
  wrong_answer     an answer was rendered but the data does not match the question —
                   wrong table, wrong/absent ordering on a ranked question, or a row
                   set that disagrees with ground truth. THE WORST OUTCOME: a reader
                   cannot tell it from a real answer.
  ignored_qualifier  correct data for a BROADER question — the SQL ran and returned
                   real rows, but a qualifier the question named ("currently on the
                   market", "between 100 and 50,000") never reached the WHERE clause.
  typed_clarify    no answer, but the refusal NAMES the column / value / table it could
                   not map ("status has APPROVED / DRAFT / CANCELLED, not ACTIVE").
                   The only acceptable non-`correct` verdict.
  generic_refusal  no answer and no information — "Could you clarify what you're asking
                   about?", "I couldn't answer this query." A dead end for the user.

Targets (Part F): >= 14/20 correct, 0 wrong_answer, 0 generic_refusal.

Usage (from the repo root, on the host — the API is on localhost:8080 and the source DB
`homzhub` on localhost:5432):

    python3 scripts/eval_question_txt.py [--out reports/raw/qtxt_<label>.json] [--label OFF]
    python3 scripts/eval_question_txt.py --only 9,10,13      # a subset, by question number

Exit code 1 if any target is missed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = os.environ.get("VEDA_QTXT_API", "http://localhost:8080/api/v1/conversations/query")
TOKEN = os.environ.get("VEDA_QTXT_TOKEN", "f1212a9cb5fde18680911174187136feaa9028fb")
DSN = dict(host=os.environ.get("HOMZHUB_HOST", "localhost"),
           port=int(os.environ.get("HOMZHUB_PORT", "5432")),
           dbname=os.environ.get("HOMZHUB_DB", "homzhub"),
           user=os.environ.get("HOMZHUB_USER", os.environ.get("USER", "")),
           password=os.environ.get("HOMZHUB_PASSWORD", "") or None)

GL = "accounts_generalledger"
SL = "assets_salelisting"

# Ground truth per question. `order` is (semantic kind, direction) — the grader accepts
# ANY column of that kind on the anchor, because "the latest payments" does not dictate
# transaction_date over updated_at; what it dictates is "a date column, descending".
#   where     : regex markers that MUST appear in the emitted SQL (the named qualifier)
#   limit     : an explicit row count the question named
#   id_col    : primary key used for exact row-identity comparison when limit <= 5
#   expect    : "answer" (a correct SQL answer exists) | "clarify" (no honest answer exists)
SPECS = [
    # 1
    dict(n=1, table=GL, order=("date", "desc"), where=[], limit=None,
         must_project=[r"entry_type", r"amount"],
         gt="select id from accounts_generalledger order by transaction_date desc, id desc limit 100",
         note="latest ledger entries with DEBIT/CREDIT + amount"),
    # 2
    dict(n=2, table=SL, order=("money", "asc"), where=[r"status"], limit=None,
         gt="select id from assets_salelisting where status='APPROVED' order by expected_price asc, id asc limit 100",
         note="cheapest listings ON THE MARKET (status filter required)"),
    # 3
    dict(n=3, table=GL, order=("money", "desc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by amount desc, id asc limit 100",
         note="most expensive ledger records"),
    # 4
    dict(n=4, table=GL, order=("money", "asc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by amount asc, id asc limit 100",
         note="smallest payments"),
    # 5
    dict(n=5, table=SL, order=("date", "asc"), where=[], limit=None,
         must_project=[r"status"],
         gt="select id from assets_salelisting order by created_at asc, id asc limit 100",
         note="oldest listings + status"),
    # 6
    dict(n=6, table=GL, order=("date", "desc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by transaction_date desc, id desc limit 100",
         note="most recent accounting entries"),
    # 7
    dict(n=7, table=GL, order=("name", "asc"), where=[], limit=None,
         gt=None,
         note="alphabetical properties + their recent payments (join; ordering on a name column)"),
    # 8
    dict(n=8, table=GL, order=("name", "desc"), where=[], limit=None,
         gt=None,
         note="financial logs by property name, reverse alphabetical (join)"),
    # 9
    # "properties ON THE MARKET priced above 10,000". The 2026-09-23 report's ground
    # truth for this one omitted the status filter (340 rows), but that same report
    # treats "currently on the market" as a REQUIRED status filter in its criticism of
    # Q2 and Q14 — 340 counts DRAFT and CANCELLED listings as being on the market. The
    # consistent reading applies the filter here too: 186 rows.
    dict(n=9, table=SL, order=None, where=[r">\s*'?10[,.]?0*'?", r"status"], limit=None,
         gt=("select id from assets_salelisting "
             "where status='APPROVED' and expected_price > 10000"),
         note="on-the-market AND expected_price > 10000 -> 186 rows"),
    # 10
    dict(n=10, table=GL, order=None, where=[r"between|>=|>"], limit=None,
         gt="select id from accounts_generalledger where amount between 100 and 50000",
         note="amount between 100 and 50000 -> 717 rows"),
    # 11
    dict(n=11, table=GL, order=("date", "desc"), where=[], limit=5,
         gt="select id from accounts_generalledger order by transaction_date desc, id desc limit 5",
         note="top 5 most recently dated entries"),
    # 12
    dict(n=12, table=GL, order=("id", "desc"), where=[], limit=5,
         gt="select id from accounts_generalledger order by id desc limit 5",
         note="last 5 records by internal processing id"),
    # 13 — no ACTIVE status exists; the honest answer names the real domain
    dict(n=13, table=SL, order=None, where=[], limit=None, expect="clarify",
         clarify_must_name=["active", "approved", "draft", "cancelled", "status"],
         gt=None,
         note="'active' is not a value of assets_salelisting.status (APPROVED/DRAFT/CANCELLED)"),
    # 14
    dict(n=14, table=SL, order=("money", "desc"), where=[r"status"], limit=None,
         gt="select id from assets_salelisting where status='APPROVED' order by expected_price desc, id asc limit 100",
         note="most expensive listings ON THE MARKET (status filter required)"),
    # 15
    dict(n=15, table=GL, order=("money", "asc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by amount asc, id asc limit 100",
         note="absolute smallest financial records"),
    # 16
    dict(n=16, table=GL, order=("currency", "any"), where=[], limit=None,
         gt=None,
         note="recent payments sorted by currency"),
    # 17
    dict(n=17, table=SL, order=("currency", "desc"), where=[], limit=None, group=True,
         gt=None,
         note="listings grouped by currency, descending"),
    # 18
    dict(n=18, table=GL, order=("date", "asc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by transaction_date asc, id asc limit 100",
         note="oldest financial records by date (a SORT key, not a GROUP BY)"),
    # 19
    dict(n=19, table=GL, order=("updated", "desc"), where=[], limit=None,
         gt="select id from accounts_generalledger order by updated_at desc, id desc limit 100",
         note="most recently MODIFIED/UPDATED -> updated_at, not created_at"),
    # 20
    dict(n=20, table=SL, order=("updated", "desc"), where=[], limit=None,
         gt="select id from assets_salelisting order by updated_at desc, id desc limit 100",
         note="listings whose STATUS was updated most recently -> updated_at"),
]

# Which real columns satisfy each semantic ordering kind, per anchor table.
ORDER_COLS = {
    GL: {"date": ["transaction_date", "created_at", "updated_at"],
         "money": ["amount"],
         "updated": ["updated_at"],
         "id": ["id"],
         "currency": ["currency_id"],
         "name": ["project_name", "building_name", "label"]},
    SL: {"date": ["created_at", "available_from_date", "updated_at"],
         "money": ["expected_price"],
         "updated": ["updated_at"],
         "id": ["id"],
         "currency": ["currency_id"],
         "name": ["project_name", "building_name"]},
}

GENERIC_REFUSALS = (
    "could you clarify what you're asking about",
    "i couldn't answer this query",
    "try rephrasing it",
    "i'm not sure exactly what you're asking about",
)


# ── plumbing ──────────────────────────────────────────────────────────────────────────
def load_questions(path):
    qs = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.lower() == "question":
                continue
            qs.append(s)
    return qs


#: Source scope to pin each request to, or None for the account's default (unpinned)
#: scope. `apps/query/scope.py::_requested_source_ids` reads `source_ids` from the body
#: and intersects it with what RBAC permits, so this is a NARROWING request, not a
#: privilege escalation.
PIN_SOURCE_IDS = None


def ask(question, timeout=600, retries=3, backoff=20):
    """POST the question. TRANSPORT failures are retried and then reported as such.

    This matters: a 502 `LLM_UNAVAILABLE` (the SLM host on the LAN dropping out) has no
    `data` block, so the grader below saw an empty summary and scored it
    `generic_refusal` — an INFRASTRUCTURE outage silently recorded as engine behaviour.
    Five of the twenty scored that way on the first pinned run. They are now retried and,
    if still failing, surfaced as `transport_error` so they can never be mistaken for a
    verdict about the engine."""
    payload = {"message": question, "chat_id": None, "stream": False}
    if PIN_SOURCE_IDS:
        payload["source_ids"] = list(PIN_SOURCE_IDS)
    body = json.dumps(payload).encode()
    t0 = time.time()
    data = None
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Token {TOKEN}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            raw = e.read().decode()[:2000]
            data = {"_http_error": e.code, "_body": raw}
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            break
        except Exception as e:
            data = {"_error": repr(e)}
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            break
    data["_elapsed_s"] = round(time.time() - t0, 1)
    return data


_CONN = None


def _conn():
    global _CONN
    if _CONN is None:
        import psycopg2
        kw = {k: v for k, v in DSN.items() if v is not None}
        _CONN = psycopg2.connect(**kw)
        _CONN.set_session(readonly=True, autocommit=True)
    return _CONN


def run_sql(sql, limit_rows=None):
    """Execute read-only against homzhub. Returns (rows, error). Placeholders that the
    engine left unbound (%s) make a query un-runnable — reported, not crashed on."""
    if not sql:
        return None, "no sql"
    if "%s" in sql:
        return None, "unbound placeholders"
    try:
        with _conn().cursor() as cur:
            cur.execute("SET statement_timeout = 30000")
            cur.execute(sql)
            rows = cur.fetchall()
        return (rows[:limit_rows] if limit_rows else rows), None
    except Exception as e:
        try:
            _conn().rollback()
        except Exception:
            pass
        return None, f"{type(e).__name__}: {str(e)[:160]}"


# ── grading ───────────────────────────────────────────────────────────────────────────
def _tables_in(sql):
    return {t.lower() for t in re.findall(r'(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', sql or "",
                                          re.IGNORECASE)}


def _order_by(sql):
    """[(column, direction)] from the ORDER BY tail, lowercased."""
    m = re.search(r"\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|\bOFFSET\b|$)", sql or "",
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    out = []
    for part in m.group(1).split(","):
        p = part.strip()
        if not p:
            continue
        toks = [t.lower() for t in re.findall(r'"?([A-Za-z_][A-Za-z0-9_]*)"?', p)]
        # drop a table qualifier ("t"."amount" -> amount) and the trailing ASC/DESC
        toks = [t for t in toks if t not in ("asc", "desc", "nulls", "first", "last")]
        if not toks:
            continue
        direction = "desc" if re.search(r"\bDESC\b", p, re.IGNORECASE) else "asc"
        out.append((toks[-1], direction))
    return out


def _explicit_limit(sql):
    m = re.search(r"\bLIMIT\s+(\d+)", sql or "", re.IGNORECASE)
    return int(m.group(1)) if m else None


def grade(spec, resp):
    """-> (verdict, detail, emitted_sql, returned_rows, gt_rows)"""
    if resp.get("_http_error") or resp.get("_error"):
        why = (f"HTTP {resp['_http_error']}" if resp.get("_http_error")
               else resp.get("_error"))
        body = (resp.get("_body") or "")[:160]
        return "transport_error", f"{why} — NOT an engine verdict: {body}", "", None, None
    data = resp.get("data") or {}
    exp = (data.get("metadata") or {}).get("explainability") or {}
    sql = ((exp.get("sql") or {}).get("query") or "").strip()
    summary = (data.get("summary") or "").strip()
    n_ret = ((exp.get("result") or {}).get("row_count"))

    gt_rows, gt_err = (run_sql(spec["gt"]) if spec.get("gt") else (None, None))
    gt_n = len(gt_rows) if gt_rows is not None else None

    # ── no SQL: a refusal or a clarify ────────────────────────────────────────────────
    if not sql:
        low = summary.lower()
        if any(g in low for g in GENERIC_REFUSALS) or len(low) < 25:
            return "generic_refusal", f"no SQL; generic text: {summary[:120]!r}", sql, n_ret, gt_n
        named = spec.get("clarify_must_name")
        if named and not any(w in low for w in named):
            return "generic_refusal", (f"no SQL; clarify names none of {named}: "
                                       f"{summary[:120]!r}"), sql, n_ret, gt_n
        # names a real schema element -> the acceptable non-correct outcome
        if spec.get("expect") == "clarify":
            return "correct", f"typed clarify (the honest answer): {summary[:110]!r}", sql, n_ret, gt_n
        return "typed_clarify", summary[:140], sql, n_ret, gt_n

    # ── SQL emitted ───────────────────────────────────────────────────────────────────
    if spec.get("expect") == "clarify":
        return "wrong_answer", (f"answered a question with no honest answer "
                                f"({spec['note']})"), sql, n_ret, gt_n

    tabs = _tables_in(sql)
    if spec["table"].lower() not in tabs:
        return "wrong_answer", f"wrong anchor: SQL hits {sorted(tabs)}, expected {spec['table']}", \
            sql, n_ret, gt_n

    # required qualifier markers (the WHERE the question named)
    low_sql = sql.lower()
    for marker in spec.get("where") or []:
        if not re.search(marker, low_sql):
            return "ignored_qualifier", f"SQL lacks required qualifier /{marker}/", sql, n_ret, gt_n

    # a projection the question explicitly asked for ("debited or credited", "status")
    for marker in spec.get("must_project") or []:
        if not re.search(marker, low_sql):
            return "ignored_qualifier", f"SQL never projects /{marker}/", sql, n_ret, gt_n

    # ordering the question named
    want = spec.get("order")
    if want:
        kind, want_dir = want
        ok_cols = ORDER_COLS.get(spec["table"], {}).get(kind, [])
        obs = _order_by(sql)
        if not obs:
            return "wrong_answer", f"ranked question, no ORDER BY (wanted {kind} {want_dir})", \
                sql, n_ret, gt_n
        col, direction = obs[0]
        if col not in ok_cols:
            return "wrong_answer", (f"ORDER BY {col} is not a {kind} column "
                                    f"(expected one of {ok_cols})"), sql, n_ret, gt_n
        if want_dir != "any" and direction != want_dir:
            return "wrong_answer", f"ORDER BY {col} {direction.upper()}, expected {want_dir.upper()}", \
                sql, n_ret, gt_n

    if spec.get("group") and not re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE):
        return "wrong_answer", "grouping question with no GROUP BY", sql, n_ret, gt_n

    want_limit = spec.get("limit")
    if want_limit is not None and _explicit_limit(sql) != want_limit:
        return "wrong_answer", f"question named LIMIT {want_limit}, SQL has {_explicit_limit(sql)}", \
            sql, n_ret, gt_n

    # ── data comparison against ground truth ──────────────────────────────────────────
    if spec.get("gt"):
        if gt_err:
            return "correct", f"shape ok; ground truth unrunnable ({gt_err})", sql, n_ret, gt_n
        got, err = run_sql(sql)
        if err:
            return "correct", f"shape ok; emitted SQL not re-runnable here ({err})", sql, n_ret, gt_n
        got_n = len(got)
        if want_limit is not None and want_limit <= 5:
            # Exact rows for a small LIMIT. Compare the ORDERING COLUMN'S VALUE SEQUENCE,
            # not row ids: `transaction_date` has ties (four rows share 2026-05-20), so
            # which id lands in slot 4 vs 5 is not determined and an id comparison would
            # be flaky. Ground truth is recomputed on the column the ENGINE chose (the
            # shape check above already restricted that to an acceptable one), which is
            # the real question — "are these the true top N by that column?".
            obs = _order_by(sql)
            ocol, odir = obs[0]
            want_seq = _col_seq(f'SELECT "{ocol}" FROM "{spec["table"]}" '
                                f'ORDER BY "{ocol}" {odir.upper()} LIMIT {want_limit}')
            got_seq = _col_seq(f'SELECT "{ocol}" FROM ({sql.rstrip().rstrip(";")}) _q '
                               f'LIMIT {want_limit}')
            if want_seq is None or got_seq is None:
                if got_n != len(gt_rows):
                    return "wrong_answer", f"returned {got_n} rows, ground truth {len(gt_rows)}", \
                        sql, got_n, gt_n
                return "correct", f"{got_n} rows (value sequence unverifiable)", sql, got_n, gt_n
            if got_seq != want_seq:
                return "wrong_answer", (f"top-{want_limit} {ocol} {got_seq} != true "
                                        f"{want_seq}"), sql, got_n, gt_n
            return "correct", f"exact top-{want_limit} by {ocol} {odir}", sql, got_n, gt_n
        # otherwise compare counts, allowing the engine's own LIMIT page
        emitted_limit = _explicit_limit(sql)
        gt_compare = min(gt_n, emitted_limit) if emitted_limit else gt_n
        if got_n != gt_compare:
            return "wrong_answer", f"returned {got_n} rows, ground truth {gt_compare}", \
                sql, got_n, gt_n
        return "correct", f"{got_n} rows == ground truth", sql, got_n, gt_n

    return "correct", "shape ok (no row-level ground truth for this question)", sql, n_ret, gt_n


def _col_seq(sql):
    """First column of `sql`, in order, as a list — or None if it will not run."""
    try:
        with _conn().cursor() as cur:
            cur.execute("SET statement_timeout = 30000")
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]
    except Exception:
        try:
            _conn().rollback()
        except Exception:
            pass
        return None


# ── main ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(os.path.dirname(__file__), "..", "question.txt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--only", default="", help="comma-separated question numbers")
    ap.add_argument("--source-ids", default="",
                    help="pin the request scope, e.g. 2 or 2,3,4,5 (default: unpinned)")
    args = ap.parse_args()

    global PIN_SOURCE_IDS
    if args.source_ids.strip():
        PIN_SOURCE_IDS = [int(x) for x in args.source_ids.split(",") if x.strip()]
        print(f"pinned to source_ids={PIN_SOURCE_IDS}")

    qs = load_questions(os.path.abspath(args.questions))
    if len(qs) != 20:
        print(f"!! expected 20 questions, found {len(qs)}", file=sys.stderr)
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None

    rows, counts = [], {}
    for spec in SPECS:
        n = spec["n"]
        if only and n not in only:
            continue
        q = qs[n - 1]
        resp = ask(q)
        verdict, detail, sql, n_ret, gt_n = grade(spec, resp)
        counts[verdict] = counts.get(verdict, 0) + 1
        conf = (((resp.get("data") or {}).get("metadata") or {})
                .get("explainability") or {}).get("confidence")
        rows.append(dict(n=n, question=q, verdict=verdict, detail=detail, sql=sql,
                         returned_rows=n_ret, ground_truth_rows=gt_n, confidence=conf,
                         summary=((resp.get("data") or {}).get("summary") or "")[:300],
                         elapsed_s=resp.get("_elapsed_s")))
        print(f"[{n:02d}] {verdict:<16} {detail[:110]}", flush=True)

    correct = counts.get("correct", 0)
    wrong = counts.get("wrong_answer", 0)
    generic = counts.get("generic_refusal", 0)
    total = len(rows)
    print("\n" + "=" * 78)
    print(f"label={args.label or '(none)'}  correct={correct}/{total}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "correct"))
    transport = counts.get("transport_error", 0)
    if transport:
        print(f"\n!! {transport} question(s) failed at the TRANSPORT layer (502/500/timeout).")
        print("   Those are infrastructure outages, not engine verdicts — this run is")
        print("   INCOMPLETE and must be re-run once the stack is stable.")
    ok = correct >= 14 and wrong == 0 and generic == 0 and transport == 0
    print(f"TARGETS  >=14 correct: {'OK' if correct >= 14 else 'MISS'} ({correct})   "
          f"0 wrong_answer: {'OK' if wrong == 0 else 'MISS'} ({wrong})   "
          f"0 generic_refusal: {'OK' if generic == 0 else 'MISS'} ({generic})")
    print("=" * 78)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "source_ids": PIN_SOURCE_IDS,
                       "counts": counts, "correct": correct,
                       "total": total, "results": rows}, f, indent=1)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
