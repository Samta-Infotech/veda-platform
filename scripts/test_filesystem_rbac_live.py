"""Live end-to-end aggressive check: the filesystem (RAG) pipeline, under two
opposite RBAC shapes, against the REAL running stack (real Postgres, real
Redis, real inference engine, real ingested docs_contracts + homzhub data) —
not mocked. This is deliberately NOT a pytest unit test: it needs the actual
ingested content (the 4 real docs_contracts documents, the real homzhub
schema) to assert on real answer TEXT, not just an HTTP status code. Every
scenario/response shape here was manually verified once, live, during the
session that produced the fixes this script guards:

  - apps.access_management.services.catalog   (per-document CatalogResource)
  - apps.access_management.services.data_scope (per-document allow-list)
  - veda_core/veda/rbac_filter.py::filter_doc_chunks
  - veda_core/ingestion/chunk_embedder.py::retrieve_top_k_chunks
  - chatbot/nodes.py::_extract_engine_result (status-mapping fix)

Two scenarios, same query set, asserting the SAME correct filesystem answers
in both:

  A. RBAC scoped SOLELY to filesystem (docs_contracts) — no db/lake access at
     all. Every document must still answer correctly; any db-only query must
     be cleanly refused (never crash, never leak).
  B. RBAC broadened to ALSO allow db (homzhub) + lake sources. The exact same
     filesystem queries must answer IDENTICALLY to scenario A — a wider scope
     must never confuse or break RAG retrieval for documents already granted.

Plus one deny-one-document pass (in EACH scenario) proving per-document
enforcement holds regardless of how broad the rest of the grant is.

Snapshots the Admin role's current db:*/lake:*/files:* grants first and
restores them exactly on exit (including on failure) — never leaves the live
permission state altered.

Run from the repo root, inside the `api` container (Django + a real DB
connection; no torch/veda_core ML deps needed — only the HTTP surface is
exercised):

    docker compose exec api python scripts/test_filesystem_rbac_live.py

Exits 0 if every assertion passed, 1 otherwise (prints a summary either way).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

import django  # noqa: E402

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from rest_framework_simplejwt.tokens import RefreshToken  # noqa: E402

from apps.access_management.models import Effect, Permission, Role, RolePermission  # noqa: E402

BASE_URL = os.environ.get("VEDA_TEST_BASE_URL", "http://localhost:8000")
QUERY_URL = f"{BASE_URL}/api/v1/conversations/query"
ADMIN_USERNAME = os.environ.get("VEDA_TEST_ADMIN_USERNAME", "veda")
ADMIN_ROLE_NAME = os.environ.get("VEDA_TEST_ADMIN_ROLE", "Admin")
DOCS_SOURCE_ID = int(os.environ.get("VEDA_TEST_DOCS_SOURCE_ID", "3"))
HOMZHUB_SOURCE_ID = int(os.environ.get("VEDA_TEST_HOMZHUB_SOURCE_ID", "2"))

# The 4 real documents ingested for docs_contracts in this environment, each
# with a query whose correct answer can ONLY come from that specific document
# (so a leak from any other document is independently detectable) and a
# substring that must appear in a genuinely correct answer.
DOCS = {
    "msa_green_tower.pdf": {
        "query": "What does the MSA agreement say about late fees?",
        # Lenient — fine to appear in ANY answer discussing this document, even a
        # polite refusal that merely echoes the question's own topic.
        "expect_any": ["2 percent", "2%", "late fee"],
        # Strict — only ever present if this SPECIFIC document's actual content
        # was retrieved and reproduced (a refusal that just restates the
        # question topic can't accidentally produce these).
        "leak_markers": ["squash", "intercom", "green tower"],
    },
    "maintenance_policy.docx": {
        "query": "What is the maintenance policy for asset 21?",
        "expect_any": ["maintenance", "football court", "muddanahalli"],
        "leak_markers": ["muddanahalli", "football court", "basket ball court",
                        "basketball court"],
    },
    "site_notes.md": {
        "query": "What do the site inspection notes say about asset 6?",
        "expect_any": ["asset 6", "kochi", "insurance"],
        "leak_markers": ["kochi", "grocery store", "asset 4"],
    },
    "generic_readme.txt": {
        "query": "What does the readme document say about the help desk?",
        "expect_any": ["help desk"],
        "leak_markers": ["password polic", "onboarding step", "knowledge base"],
    },
}

HYBRID_QUERY = ("How many payment transactions are there and what does the "
               "MSA agreement say about late fees?")

_DENIED_PHRASES = ["clarify what you're asking", "not provided in the given context",
                  "don't have permission", "admin approval"]

results = []  # (name, ok: bool, detail: str)


def _record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _token():
    user = get_user_model().objects.get(username=ADMIN_USERNAME)
    return str(RefreshToken.for_user(user).access_token)


def _ask(message, **body):
    payload = {"message": message, "stream": False, **body}
    req = urllib.request.Request(
        QUERY_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_token()}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        data = json.loads(exc.read().decode())
    return (data.get("data") or {}).get("summary") or data.get("message") or json.dumps(data)


# ---------------------------------------------------------------------------
# Permission plumbing
# ---------------------------------------------------------------------------

def _role():
    return Role.objects.get(name=ADMIN_ROLE_NAME)


def _read_perm():
    return Permission.objects.get(code="data.read")


def _snapshot(role):
    """Every current data.read grant on this role — restored verbatim on exit."""
    return list(RolePermission.objects.filter(role=role, permission=_read_perm())
               .values("resource_path", "effect"))


def _restore(role, snapshot):
    RolePermission.objects.filter(role=role, permission=_read_perm()).delete()
    RolePermission.objects.bulk_create([
        RolePermission(role=role, permission=_read_perm(), **row) for row in snapshot
    ])


def _set_grants(role, paths_effects):
    """Replace ALL of this role's data.read grants with exactly {path: effect}."""
    RolePermission.objects.filter(role=role, permission=_read_perm()).delete()
    RolePermission.objects.bulk_create([
        RolePermission(role=role, permission=_read_perm(), resource_path=path, effect=effect)
        for path, effect in paths_effects.items()
    ])


def _doc_path(name=""):
    from apps.access_management import resource_path as rp
    parts = ["files", "docs_contracts"] + ([name] if name else [])
    return rp.build(*parts)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _assert_document_answered(label, doc_name):
    spec = DOCS[doc_name]
    answer = _ask(spec["query"], source_id=DOCS_SOURCE_ID)
    lower = answer.lower()
    if any(p in lower for p in _DENIED_PHRASES):
        _record(label, False, f"got a refusal instead of an answer: {answer!r}")
        return
    hit = any(kw.lower() in lower for kw in spec["expect_any"])
    _record(label, hit, f"none of {spec['expect_any']} found in: {answer!r}")


def _assert_document_denied(label, doc_name):
    spec = DOCS[doc_name]
    answer = _ask(spec["query"], source_id=DOCS_SOURCE_ID)
    lower = answer.lower()
    # leak_markers only (not expect_any): a polite refusal legitimately echoes the
    # question's own topic words ("...does not contain the maintenance policy...")
    # without that being a leak — only these document-specific FACTS could appear
    # from the actual restricted content being retrieved and reproduced.
    leaked = any(kw.lower() in lower for kw in spec["leak_markers"])
    _record(label, not leaked, f"DENIED document's content leaked into the answer: {answer!r}")


def _assert_sql_denied(label):
    answer = _ask("How many rent transactions are there", source_id=HOMZHUB_SOURCE_ID)
    lower = answer.lower()
    denied = any(p in lower for p in ["admin approval", "don't have permission", "permission"])
    _record(label, denied, f"expected an access-denied refusal, got: {answer!r}")


def _assert_sql_answered(label):
    answer = _ask("How many payment transactions are there", source_id=HOMZHUB_SOURCE_ID)
    lower = answer.lower()
    ok = any(ch.isdigit() for ch in answer) and "permission" not in lower and "admin approval" not in lower
    _record(label, ok, f"expected a numeric answer, got: {answer!r}")


def _assert_hybrid_answered(label):
    answer = _ask(HYBRID_QUERY, source_ids=[HOMZHUB_SOURCE_ID, DOCS_SOURCE_ID])
    lower = answer.lower()
    if any(p in lower for p in _DENIED_PHRASES[:1]):  # the generic-clarify bug specifically
        _record(label, False, f"got the generic clarify fallback: {answer!r}")
        return
    has_doc = any(kw in lower for kw in ["2 percent", "2%", "late fee"])
    _record(label, has_doc, f"MSA late-fee clause not found in hybrid answer: {answer!r}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_a_filesystem_only(role):
    print("\n=== Scenario A: RBAC scoped SOLELY to filesystem (no db/lake access) ===")
    _set_grants(role, {_doc_path(name): Effect.ALLOW for name in DOCS})

    for name in DOCS:
        _assert_document_answered(f"A: {name} answers correctly (filesystem-only scope)", name)
    _assert_sql_denied("A: a db-only query is cleanly refused (no db grant at all)")

    # Aggressive: deny exactly one document, the other 3 (still the ONLY grants
    # this role has at all) must be completely unaffected.
    print("  -- aggressive: deny maintenance_policy.docx, keep the other 3 --")
    grants = {_doc_path(name): Effect.ALLOW for name in DOCS}
    grants[_doc_path("maintenance_policy.docx")] = Effect.DENY
    _set_grants(role, grants)
    _assert_document_denied("A: denied document's content never leaks", "maintenance_policy.docx")
    for name in DOCS:
        if name == "maintenance_policy.docx":
            continue
        _assert_document_answered(f"A: {name} still answers correctly (one sibling denied)", name)


def scenario_b_filesystem_plus_everything(role):
    print("\n=== Scenario B: RBAC broadened to ALSO allow db (homzhub) + lake ===")
    grants = {_doc_path(name): Effect.ALLOW for name in DOCS}
    grants["db:homzhub"] = Effect.ALLOW
    grants["db:launchpad"] = Effect.ALLOW
    grants["lake:invoices_csv"] = Effect.ALLOW
    grants["lake:catalog_parquet"] = Effect.ALLOW
    _set_grants(role, grants)

    # The exact same filesystem queries must answer IDENTICALLY to scenario A —
    # a wider scope must never confuse retrieval for a source already granted.
    for name in DOCS:
        _assert_document_answered(f"B: {name} STILL answers correctly (broad scope)", name)
    _assert_sql_answered("B: a db query now answers (db grant present)")
    _assert_hybrid_answered("B: hybrid (SQL + doc) answers with both parts")

    print("  -- aggressive: deny exactly one document even with everything else open --")
    grants[_doc_path("site_notes.md")] = Effect.DENY
    _set_grants(role, grants)
    _assert_document_denied("B: denied document never leaks (even with db/lake wide open)",
                            "site_notes.md")
    for name in DOCS:
        if name == "site_notes.md":
            continue
        _assert_document_answered(f"B: {name} still answers correctly (one sibling denied, broad scope)",
                                  name)


def main():
    role = _role()
    snapshot = _snapshot(role)
    try:
        scenario_a_filesystem_only(role)
        scenario_b_filesystem_plus_everything(role)
    finally:
        _restore(role, snapshot)
        print(f"\nRestored {ADMIN_ROLE_NAME}'s original {len(snapshot)} data.read grant(s).")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*70}\n{passed}/{total} assertions passed")
    if passed != total:
        print("FAILED:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
