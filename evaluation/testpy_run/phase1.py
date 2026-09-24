"""PHASE 1 — which of test.py's questions actually work on the LOCAL main pipeline.

Every question gets its OWN chat, so nothing here is a memory test: a failure is the
engine's, not a leaked frame from the previous question. Results are appended to disk after
every question, so a worker death loses one row, not the run.
"""
import json, os, sys, time, urllib.error, urllib.request

sys.path.insert(0, "/app")
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

BASE = "http://localhost:8000/api/v1"
OUT = "/tmp/phase1_results.jsonl"
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import AccessToken
TOK = str(AccessToken.for_user(get_user_model().objects.get(username="veda")))
H = {"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"}


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(body).encode(), headers=H)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}
    except Exception as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


# test.py's lists. `src` records which list a question came from so the report can say
# whether a whole CATEGORY works rather than only counting individual questions.
LEDGER = [
    "Show top five general ledger entries",
    'Show the 10 transactions made by the payer "Tenant" with transaction details such as receiver name, entry type, date, amount, and currency.',
    "Show top 5 highest CREDIT transactions",
    "Show top 5 highest DEBIT transactions",
    "Display all transactions from the ledger where the transaction amount exceeds 10,000.",
    "Show the latest 10 ledger entries.",
    "How many lease transactions are recorded with currency ID 30?",
    "Can you show me the 10 most recently created tickets along with their details?",
    "How many tickets are there in each status category?",
    "What is the average time taken to resolve closed tickets?",
]
PROPERTY_DB = [
    "How many properties are listed in the database?",
    "Which floor has the maximum number of properties?",
    "How many properties have power backup vs no power backup?",
    "How many properties allow all day access?",
    "How many properties are corner properties?",
    "How many properties are in a gated community vs non-gated community?",
    "Count transactions where exchange rate is greater than 74.",
    "What is the total settlement expected, paid, and balance amount?",
    "How many new properties have been added after the year 2012?",
    "How many ledger transactions exist?",
    "How many payment transaction history records exist?",
    "How many settlement history records exist?",
]
SHAPES = [
    "How many properties were there in the last 12 months by created at?",
    "What is the monthly trend of properties based on created at?",
    "What is the distribution of properties by furnishing?",
    "What is the distribution of properties by facing?",
    "What is the distribution of lease listings by status?",
    "What is the distribution of payment transactions by transaction type?",
    "Which properties have the highest carpet area?",
    "Which lease listings have the highest expected monthly rent?",
    "Which payment transactions have the highest paid amount?",
    "What is the average carpet area sqft of properties?",
    "What is the total expected monthly rent across all lease listings?",
    "What is the average expected price of sale listings?",
    "Show properties where is gated is true.",
    "List properties with furnishing 'FULL'.",
    "List sale listings with status 'APPROVED'.",
    "How many properties are there?",
    "How many projects do we have?",
    "How many sale listings are there?",
    "List all amenities.",
    "Show all asset types.",
    "What are the names of all projects?",
    "show properties with their owners",
    "show lease listings and their property",
    "show properties with their amenities",
    "top 5 properties by number of payment transactions",
    "top 5 payment transactions by paid amount",
    "top 10 users by total paid amount",
    "list all tenants",
    "List all amenity categories.",
    "List all payment types.",
]
BUSINESS = [
    "What is the total outstanding amount for all maintenance items that are not yet completed or settled?",
    "Which category contributes the highest value among all completed payments?",
    "What percentage of all maintenance records are related to repair work?",
    "How much money has already been collected through completed payments?",
    "Which assets are currently covered under the maintenance arrangement?",
    "Which city has the highest-rated vendor, and what is the vendor's rating?",
    "What is the average vendor rating across all service locations?",
    "What percentage of vendors have ratings of at least 4.0?",
]

PLAN = ([("ledger", q) for q in LEDGER]
        + [("property_db", q) for q in PROPERTY_DB]
        + [("shapes", q) for q in SHAPES]
        + [("business", q) for q in BUSINESS])

done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            done.add(json.loads(ln)["question"])
        except Exception:
            pass
print(f"plan={len(PLAN)} already_done={len(done)}", flush=True)

fh = open(OUT, "a")
for i, (src, q) in enumerate(PLAN, 1):
    if q in done:
        continue
    row = {"n": i, "src": src, "question": q}
    for attempt in (1, 2):                      # one retry: gunicorn drops a worker sometimes
        _, created = post("/conversations/create", {"title": f"p1-{i}"})
        chat = (created.get("data") or {}).get("chat_id")
        t0 = time.perf_counter()
        code, body = post("/conversations/query",
                          {"message": q, "chat_id": chat, "stream": False})
        took = round(time.perf_counter() - t0, 1)
        if code == 200 or attempt == 2:
            break
        time.sleep(5)
    d = body.get("data") or {}
    text = ""
    for b in (d.get("response") or []):
        if isinstance(b, dict) and b.get("content"):
            text = str(b["content"]); break
    ex = (d.get("metadata") or {}).get("explainability") or {}
    row.update({
        "chat_id": chat, "http": code, "took_s": took, "answer": text,
        "sql": ((ex.get("sql") or {}).get("query")),
        "row_count": ((ex.get("result") or {}).get("row_count")),
        "sources": [s.get("name") for s in (ex.get("sources") or []) if isinstance(s, dict)],
        "cross_source": bool((ex.get("cross_source") or {}).get("used")),
        "action": (d.get("metadata") or {}).get("action"),
    })
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print(f"{i:3d}/{len(PLAN)} [{code}] {took:5.1f}s {src:11s} {q[:58]!r}", flush=True)
    print(f"      -> {text[:110]!r}", flush=True)
fh.close()
print("PHASE1 DONE", flush=True)
