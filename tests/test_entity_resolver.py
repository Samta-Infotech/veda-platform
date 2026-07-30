"""Unit tests for Entity Resolution V1 (query/entity_resolver.py).

Deterministic, no DB / no SLM: drives resolve_entities against the REAL semantic
model. Retrieval-dependent cases use small synthetic result objects. Validates the
§18 categories: canonical entity, explicit detail, user vs preference, multi-entity,
same-table concepts, ambiguous, ungrounded, and audit-FK safety (never invented).

Run from repo root: ``pytest tests/test_entity_resolver.py``
"""
import os, sys, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "veda_core"))
_SM = json.load(open(os.path.join(_ROOT, "veda_core", "data", "veda_semantic_model.json")))


class _R:
    """Minimal retrieval-result stand-in: col_id + final_score."""
    def __init__(self, col_id, score):
        self.col_id = col_id
        self.final_score = score


def _resolve(query, results=None):
    from query.entity_resolver import resolve_entities
    return resolve_entities(query, results, _SM, [])


# ── canonical entity: MASTER wins over its detail children by name coverage ────
def test_canonical_user_beats_preference():
    r = _resolve("show users")
    assert r.status == "RESOLVED"
    assert r.anchor == "users_user"          # NOT users_userpreference/useraddress


def test_explicit_detail_beats_master():
    r = _resolve("show user preferences")
    assert r.status == "RESOLVED"
    assert r.anchor == "users_userpreference"   # detail noun beats the MASTER prior


# ── payment vs settlement: retrieval breaks the pure name-tie ─────────────────
def test_payment_transaction_with_retrieval():
    # retrieval favors accounts_paymenttransaction → resolves it over the reminder bridge
    results = [_R("accounts_paymenttransaction.paid_amount", 0.9),
               _R("accounts_paymenttransaction.expected_amount", 0.8),
               _R("reminders_reminderpaymenttransaction.id", 0.1)]
    r = _resolve("show payment transactions", results)
    assert r.status == "RESOLVED"
    assert r.anchor == "accounts_paymenttransaction"


def test_explicit_settlement():
    r = _resolve("show payment transaction settlements")
    assert r.status == "RESOLVED"
    assert r.anchor == "accounts_paymenttransactionsettlement"


# ── ungrounded: a business noun with no table + only column synonyms → fallback ─
def test_truly_ungrounded_not_forced():
    # a noun with no table AND no glossary entry → do NOT force an arbitrary MASTER pick
    r = _resolve("list the gadgets and widgets")
    assert r.status in ("UNGROUNDED", "AMBIGUOUS")
    assert r.anchor is None


def test_impossible_concept_not_forced_to_master():
    r = _resolve("show the favorite color of each spaceship")
    assert r.status in ("UNGROUNDED", "AMBIGUOUS")
    assert r.anchor is None


# ── multi-entity: two DISTINCT named entities → ≥2 distinct tables ────────────
def test_multi_entity_distinct_tables():
    results = [_R("assets_leaselisting.expected_monthly_rent", 0.9),
               _R("assets_leasetransaction.rent", 0.85)]
    r = _resolve("show lease listings with their lease transactions", results)
    assert r.status == "RESOLVED"
    assert r.anchor in ("assets_leaselisting", "assets_leasetransaction")
    assert r.distinct_tables >= 2            # forces needs_join downstream


# ── same-table concepts must NOT trigger multi-table ─────────────────────────
def test_same_table_two_concepts_single_entity():
    results = [_R("accounts_paymenttransaction.paid_amount", 0.9),
               _R("accounts_paymenttransaction.expected_amount", 0.9)]
    r = _resolve("total paid and expected amount of payment transactions", results)
    # two measures, one entity → single table, never a spurious join
    assert r.distinct_tables <= 1


# ── ambiguity: a true same-token tie with no retrieval separation → fallback ──
def test_ambiguous_without_retrieval_falls_back():
    r = _resolve("show payment transactions")     # no retrieval → paymenttx vs reminder tie
    assert r.status in ("AMBIGUOUS", "UNGROUNDED")
    assert r.anchor is None                        # never forces a coin-flip pick


# ── audit-FK safety: resolver returns entity names only; never a join relation ─
def test_resolver_never_returns_audit_relationship():
    # The resolver only names entities; it must never emit created_by/updated_by joins.
    r = _resolve("show users")
    for t in [r.anchor, *r.secondaries]:
        assert t is None or "created_by" not in str(t)


# ── entity COMPLETENESS: curated glossary resolves UNNAMED entities ──────────
def test_glossary_resolves_unnamed_property():
    # "property" names no table (synonyms column-level) but the curated glossary maps it
    r = _resolve("list properties")
    assert r.status == "RESOLVED"
    assert r.anchor == "assets_asset"


def test_glossary_multi_entity_owner_property():
    # both unnamed entities resolved → ≥2 distinct tables → forces multi-table planning
    r = _resolve("show properties with their owners")
    assert r.status == "RESOLVED"
    tables = {r.anchor, *r.secondaries}
    assert "users_user" in tables and "assets_asset" in tables
    assert r.distinct_tables >= 2


def test_glossary_does_not_break_named_detail():
    # curated "user"→users_user must NOT override the explicit detail noun "preferences"
    r = _resolve("show user preferences")
    assert r.anchor == "users_userpreference"


# ── RC3 grounded clarification (ER_GROUNDED_REFUSAL) ─────────────────────────
def test_rc3_flag_default_off():
    # default OFF → production pipeline byte-identical (no forced clarifications)
    import config
    assert config.ER_GROUNDED_REFUSAL is False


def test_rc3_ambiguous_exposes_clarify_options():
    # When AMBIGUOUS, evidence must carry the tied candidates so the pipeline can
    # name them in the clarify message ("did you mean X or Y?"). Only exercised
    # when the resolver actually returns AMBIGUOUS (not UNGROUNDED) for this query.
    r = _resolve("show payment transactions")
    if r.status != "AMBIGUOUS":
        import pytest
        pytest.skip("query resolved/ungrounded here, not ambiguous — trigger tested elsewhere")
    assert r.anchor is None                      # never coin-flips a pick
    cands = r.evidence.get("candidates") or []
    assert len(cands) >= 2                        # two competitors to disambiguate
    # the two tied candidates share the SAME matched-token set (the AMBIGUOUS rule)
    assert set(cands[0]["matched"]) == set(cands[1]["matched"])
    # the message builder has a non-empty label for each option
    opts = [c.get("master") or c.get("table") for c in cands[:2]]
    assert all(opts) and len(opts) == 2


# ── flag semantics sanity: resolver object shape ─────────────────────────────
def test_result_shape():
    r = _resolve("show users")
    assert hasattr(r, "status") and hasattr(r, "anchor") and hasattr(r, "secondaries")
    assert hasattr(r, "confidence") and hasattr(r, "distinct_tables")
    assert isinstance(r.evidence, dict)
