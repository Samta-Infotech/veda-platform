"""Filesystem/document (source 3) benchmark — 57 grounded queries through the MAIN pipeline.
Runs with all ready sources in context so the coordinator ROUTES each query (query embed -> source
route -> retrieve -> answer). Captures routed source + answer. Run in the container, CWD=/app/veda_core:
  docker exec -w /app/veda_core -e PYTHONPATH=/app <endpoints...> veda-platform-inference-1 python -u doc_bench.py
"""
import json, time, sys
sys.path.insert(0, '/app'); sys.path.insert(0, '/app/veda_core')
from veda_core.context import RequestContext, set_context, set_source_profiles

QUESTIONS = [
    "What is the late fee on overdue Rent and Society Charges invoices?",
    "What percentage late fee applies per month on overdue invoices?",
    "How much notice is required to terminate the Green Tower agreement?",
    "Can either party terminate the MSA, and under what condition?",
    "Which assets does the Green Tower MSA cover?",
    "In which cities are the assets covered by the MSA located?",
    "Which amenities are in scope under the Green Tower MSA?",
    "Which ledger categories are applicable under the MSA?",
    "Is there a penalty for paying invoices late under the MSA?",
    "Does the MSA cover asset 20?",
    "Is Intercom in scope under the Green Tower MSA?",
    "Which assets does the maintenance policy govern?",
    "Which courts or facilities are maintained under the policy and where?",
    "What is the monthly Society Charges fee?",
    "How much is the Repair fee and how often is it charged?",
    "What is the fee and frequency for Rent under the policy?",
    "What is the annual Insurance fee?",
    "What is the Taxes fee and how often is it charged?",
    "What is the monthly Loan payment amount?",
    "List all maintenance fee categories with their fees.",
    "Which fee categories are charged monthly?",
    "Which fee categories are charged annually?",
    "What is the total of all monthly maintenance fees?",
    "Which category has the highest fee in the maintenance policy?",
    "Which asset was inspected in the site notes?",
    "What is pending for the inspected asset?",
    "Which amenities does the inspected property list?",
    "In which city is the inspected asset located?",
    "Which asset has a follow-up scheduled?",
    "Is there a Repair request noted in the site inspection?",
    "Is an Insurance renewal pending for the inspected asset?",
    "What are male employees entitled to under paternity leave?",
    "Can employees take leave during their notice period?",
    "Which bank account is required for salary credit?",
    "How is annual salary structured and paid?",
    "What must employees NOT do under confidentiality and data protection?",
    "What is the probation policy for new recruits and interns?",
    "What is Samta's grievance redressal approach?",
    "What does the code of conduct say about fraternisation?",
    "Are public holidays fixed, or as per an annual list?",
    "What is the career mobility policy about?",
    "What does the business travel policy provide?",
    "What happens to health and insurance coverage while an employee is on leave?",
    "Summarize the information security policies in the handbook.",
    "Which amenities appear in both the MSA and the maintenance policy?",
    "Which assets are mentioned across the MSA, maintenance policy, and site notes?",
    "Which city appears in both the MSA and the site notes?",
    "Which ledger or fee categories are common to the MSA and the maintenance policy?",
    "Compare the Repair fee in the maintenance policy with the late-fee terms in the MSA.",
    "For asset 21, which documents mention it and what does each say?",
    "A tenant in Kochi has an overdue Rent invoice — which document governs the late fee, and how much is it?",
    "Which document should I check for the maintenance fee of the Football court?",
    "Does the knowledge-base readme contain any property or financial records?",
    "What topics does the general onboarding readme cover?",
    "How do employees contact the help desk?",
    "Does the readme link to any structured data source?",
    "If Rent and Society Charges invoices are overdue by 2 months, what total late-fee percentage applies?",
]


def _f(r, name, default=None):
    """Read a field from either a dict (SQL result) or a RAGResult/dataclass (doc result)."""
    if isinstance(r, dict):
        return r.get(name, default)
    return getattr(r, name, default)


def run_one(q):
    # full routing: coordinator picks among all ready sources
    set_context(RequestContext(source_id=2, tenant='default', source_ids=(2, 3, 4, 5)))
    set_source_profiles({'2': {'source_type': 'relational'}, '3': {'source_type': 'filesystem'},
                         '4': {'source_type': 'csv_lake'}, '5': {'source_type': 'parquet'}})
    from veda_hybrid import run_hybrid_query
    t = time.time()
    try:
        mr = run_hybrid_query(q)
        items = mr.items or []
        it = items[0] if items else None
        r = it.result if it else None
        cites = _f(r, 'citations') or []
        route = _f(r, 'route') or _f(r, 'table')
        kind = 'doc' if (_f(r, 'citations') is not None or _f(r, 'chunks') is not None) else 'sql'
        return {"status": getattr(it, 'status', '?'), "kind": kind,
                "route": route, "citations": cites[:4],
                "answer": (str(_f(r, 'answer') or ''))[:400],
                "error": str(_f(r, 'error') or ''), "secs": round(time.time() - t, 1)}
    except Exception as e:
        return {"status": "EXC", "error": str(e)[:200], "answer": "", "secs": round(time.time() - t, 1)}


out = []
for i, q in enumerate(QUESTIONS, 1):
    res = run_one(q); res["n"] = i; res["q"] = q; out.append(res)
    ans = (res.get('answer') or res.get('error') or '')[:75]
    cite = ",".join(str(c) for c in (res.get('citations') or []))[:30]
    print(f"[{i:02d}/{len(QUESTIONS)}] {res['status']:8s} {res.get('kind','?'):3s} [{cite}] {q[:38]}", flush=True)
    print(f"        -> {ans}", flush=True)
    json.dump(out, open('/app/doc_bench_results.json', 'w'), indent=2)
print("DONE", len(out))
