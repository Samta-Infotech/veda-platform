"""Unit tests for the meaning-first pass (veda/understanding/frame*.py, producers, vocabulary).

Pure functions only — no SLM, no DB, no Redis. Run:
    python -m pytest tests/test_frame_path.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "veda_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from veda.understanding.frame import Frame, FrameOrder, normalise, json_schema, slot_values  # noqa: E402
from veda.understanding import producers as P  # noqa: E402
from veda.understanding.vocabulary import ScopeVocab, norm_phrase  # noqa: E402
from veda.understanding import frame_grounding as G  # noqa: E402
from veda.understanding.frame_compiler import compile_frame, predicate, Declined  # noqa: E402


# ── a tiny two-table scope: ledger entries (child) → property (parent) ────────────────
SM = {
    "tables": {"ledger": {"table_type": "TRANSACTION"}, "prop": {"table_type": "MASTER"},
               "listing": {"table_type": "MASTER"}},
    "columns": {
        "ledger.id": {"semantic_type": "IDENTIFIER"},
        "ledger.amount": {"semantic_type": "MONETARY"},
        "ledger.entry_type": {"semantic_type": "CATEGORY", "sample_values": ["DEBIT", "CREDIT"]},
        "ledger.transaction_date": {"semantic_type": "TEMPORAL"},
        "ledger.updated_at": {"semantic_type": "TEMPORAL"},
        "ledger.currency_id": {"semantic_type": "IDENTIFIER"},
        "ledger.prop_id": {"semantic_type": "IDENTIFIER"},
        "prop.id": {"semantic_type": "IDENTIFIER"},
        "prop.project_name": {"semantic_type": "CATEGORY"},
        "prop.created_at": {"semantic_type": "TEMPORAL"},
        "listing.id": {"semantic_type": "IDENTIFIER"},
        "listing.status": {"semantic_type": "CATEGORY", "sample_values": ["APPROVED", "DRAFT", "CANCELLED"]},
        "listing.expected_price": {"semantic_type": "MONETARY"},
        "listing.created_at": {"semantic_type": "TEMPORAL"},
        "listing.prop_id": {"semantic_type": "IDENTIFIER"},
    },
}


def _vocab():
    v = ScopeVocab()
    v.cards = {
        "ledger": {"table": "ledger", "business_name": "ledger entry", "plural": "ledger entries",
                   "aliases": ["payment", "payments", "financial record", "financial records"],
                   "business_date_column": "transaction_date", "lifecycle_column": "entry_type",
                   "key_measures": ["amount"], "parent_entities": ["prop"], "importance": 10},
        "prop": {"table": "prop", "business_name": "property", "plural": "properties",
                 "aliases": [], "business_date_column": "created_at", "display_column": "project_name",
                 "key_measures": [], "parent_entities": [], "importance": 9},
        "listing": {"table": "listing", "business_name": "sale listing", "plural": "sale listings",
                    "aliases": ["properties for sale", "properties on the market"],
                    "business_date_column": "created_at", "lifecycle_column": "status",
                    "key_measures": ["expected_price"], "parent_entities": ["prop"], "importance": 8},
    }
    v.value_glossary = {"listing.status": {"APPROVED": ["on the market", "currently on the market"],
                                           "DRAFT": ["draft"], "CANCELLED": ["cancelled"]},
                        "ledger.entry_type": {"DEBIT": ["debit"], "CREDIT": ["credit"]}}
    v.measure_glossary = {"listing.expected_price": {"type": "MONETARY", "phrases": ["price", "priced", "expensive", "cheapest"]},
                          "ledger.amount": {"type": "MONETARY", "phrases": ["amount", "smallest"]}}
    v.source_of = {"ledger": "2", "prop": "2", "listing": "2"}
    return v


def test_normalise_clamps_off_schema():
    fr = normalise({"entity": " ledger entry ", "aggregation": "average", "limit": "5",
                    "filters": [{"concept": "status", "op": "==", "value": "x"}, {"op": ">"}],
                    "order": {"concept": "date", "dir": "sideways"}, "confidence": 7})
    assert fr.entity == "ledger entry" and fr.aggregation == "avg" and fr.limit == 5
    assert len(fr.filters) == 1 and fr.filters[0].op == "="
    assert fr.order.dir == "desc" and fr.confidence == 1.0


def test_schema_closes_entity_enum():
    s = json_schema(["ledger entry", "property"])
    ent = s["properties"]["entity"]
    assert {"type": "null"} in ent["anyOf"]
    assert ent["anyOf"][1]["enum"] == ["ledger entry", "property"]


def test_norm_phrase_plural_insensitive():
    assert norm_phrase("Ledger Entries") == norm_phrase("ledger entry")


def test_name_hits_gapped_and_specific():
    v = _vocab()
    hits = G.name_hits(v, "What are the oldest properties we put up for sale?")
    # "properties … for sale" (gap 3) subsumes the bare "properties"
    assert hits[0]["table"] == "listing"
    assert all(h["table"] != "prop" for h in hits)


def test_producer_ranking_limit_and_updated():
    fr = P.p_ranking("Which property payments were most recently modified or updated?")
    assert fr["order"]["concept"] == "updated" and fr["order"]["dir"] == "desc"
    fr = P.p_ranking("Show me the top 5 most recently dated accounting entries")
    assert fr["limit"] == 5


def test_producer_numeric_between():
    fr = P.p_numeric("Are there any recent payments processed that fall between 100 and 50,000?", _vocab())
    f = fr["filters"][0]
    assert f["op"] == "between" and f["value"] == [100.0, 50000.0]


def test_producer_values_glossary():
    fr = P.p_values("the cheapest properties currently on the market", _vocab())
    f = fr["filters"][0]
    assert f["value"] == "APPROVED" and f["_table"] == "listing"


def test_merge_fills_and_flags_conflicts():
    fr = Frame(entity="ledger entry", limit=10)
    fr = P.merge(fr, {"ranking": {"limit": 5, "order": {"concept": "@date", "dir": "desc"}}})
    assert fr.limit == 10 and fr.provenance.get("conflict:limit") == "producer:ranking"
    assert fr.order.concept == "@date"


def test_license_turns_unlicensed_min_into_order():
    fr = Frame(entity="ledger entry", aggregation="min", measure="amount")
    G.license_aggregation(fr, "What were the smallest payments processed recently?")
    assert fr.aggregation == "none" and fr.order.dir == "asc"


def test_coverage_moves_named_parent_to_child_that_hosts_slots():
    v = _vocab()
    fr = Frame(entity="property", order=FrameOrder("price", "asc"))
    fr.filters = [__import__("veda.understanding.frame", fromlist=["FrameFilter"]).FrameFilter("status", "=", "APPROVED")]
    fr.filters[0].__dict__.update({"_table": "listing", "_producer": "values", "_glossary_value": "APPROVED",
                                    "_phrase": "currently on the market"})
    t, method, clar = G.choose_entity(fr, v, SM, "Show me the cheapest properties currently on the market")
    assert t == "listing" and clar is None


def test_ground_and_compile_list_with_parent_name_order():
    v = _vocab()
    fr = Frame(entity="ledger entry", order=FrameOrder("property name", "desc"))
    fr.provenance["entity_table"] = "ledger"
    g = G.ground_frame(fr, v, SM, "Please show the financial logs ordered by property name in reverse alphabetical order.")
    assert isinstance(g, G.GroundedFrame), g
    assert g.order == ("prop", "project_name", "desc")
    c = compile_frame(g, SM)
    assert not isinstance(c, Declined)
    assert 'LEFT JOIN "prop" t1 ON t1."id" = t0."prop_id"' in c.sql
    assert 'ORDER BY t1."project_name" DESC' in c.sql
    assert c.ir.ir_partial is False and c.ir.head.startswith("frame")


def test_unmapped_value_is_a_clarify_with_domain():
    v = _vocab()
    from veda.understanding.frame import FrameFilter
    fr = Frame(entity="sale listing", filters=[FrameFilter("status", "=", "active")])
    fr.provenance["entity_table"] = "listing"
    g = G.ground_frame(fr, v, SM, "Which sale listings have an active status?")
    assert isinstance(g, G.FrameClarify)
    assert "APPROVED" in g.message and "active" in g.message


# ── §10.7: no LIMIT unless the frame named one explicitly ─────────────────────────────
def test_compile_list_no_limit_when_none_asked():
    fr = Frame(entity="ledger entry")
    g = G.GroundedFrame(frame=fr, anchor="ledger", anchor_method="test",
                        projection=[("ledger", "id")])
    c = compile_frame(g, SM)
    assert not isinstance(c, Declined)
    assert "LIMIT" not in c.sql
    assert c.ir.limit is None


def test_compile_list_keeps_explicit_limit():
    fr = Frame(entity="ledger entry", limit=5)
    g = G.GroundedFrame(frame=fr, anchor="ledger", anchor_method="test",
                        projection=[("ledger", "id")], limit=5)
    c = compile_frame(g, SM)
    assert not isinstance(c, Declined)
    assert "LIMIT 5" in c.sql
    assert c.ir.limit == 5


def test_compile_grouped_no_limit_when_none_asked():
    fr = Frame(entity="sale listing", aggregation="count", group_by=["status"])
    g = G.GroundedFrame(frame=fr, anchor="listing", anchor_method="test",
                        group_by=[("listing", "status")], measure=("count", None))
    c = compile_frame(g, SM)
    assert not isinstance(c, Declined)
    assert "LIMIT" not in c.sql
    assert c.ir.limit is None


def test_compile_grouped_keeps_explicit_limit():
    fr = Frame(entity="sale listing", aggregation="count", group_by=["status"], limit=3)
    g = G.GroundedFrame(frame=fr, anchor="listing", anchor_method="test",
                        group_by=[("listing", "status")], measure=("count", None), limit=3)
    c = compile_frame(g, SM)
    assert not isinstance(c, Declined)
    assert "LIMIT 3" in c.sql
    assert c.ir.limit == 3


def test_predicate_forms():
    f = G.GFilter("listing", "status", "=", "APPROVED", grounding="glossary")
    assert predicate(f, "t0") == "LOWER(CAST(t0.\"status\" AS TEXT)) = 'approved'"
    f = G.GFilter("ledger", "amount", "between", [100, 50000], grounding="numeric")
    assert predicate(f, "t0") == 't0."amount" BETWEEN 100 AND 50000'


def test_vote_marks_disagreement_uncertain():
    a = Frame(entity="x", limit=5)
    b = Frame(entity="x", limit=10)
    c = Frame(entity="x", limit=None)
    fr, unc = P.vote([a, b, c])
    assert "limit" in unc and "entity" not in unc
    assert slot_values(fr)["entity"] == "x"


# ── validation pass (2026-09-25) ─────────────────────────────────────────────────────
def test_unlicensed_status_filter_is_dropped():
    """The SLM restating 'properties for sale' as status = APPROVED is not the user asking."""
    v = _vocab()
    from veda.understanding.frame import FrameFilter
    fr = Frame(entity="sale listing", filters=[FrameFilter("status", "=", "APPROVED")],
               order=FrameOrder("created at", "desc"))
    fr.provenance["entity_table"] = "listing"
    g = G.ground_frame(fr, v, SM, "Show the sale listings")
    assert isinstance(g, G.GroundedFrame) and g.filters == []
    assert g.evidence.get("filters_unlicensed")


def test_audit_date_needs_its_verb():
    v = _vocab()
    fr = Frame(entity="ledger entry", order=FrameOrder("updated", "desc"))
    fr.provenance["entity_table"] = "ledger"
    g = G.ground_frame(fr, v, SM, "Show the most recently dated ledger entries")
    assert g.order == ("ledger", "transaction_date", "desc")
    g = G.ground_frame(fr, v, SM, "Which ledger entries were most recently updated?")
    assert g.order == ("ledger", "updated_at", "desc")


def test_identifier_shaped_measures_rejected_by_name():
    from ingestion.vocabulary import measure_ok
    for col in ("ifsc_code", "review_id", "latitude", "construction_year", "title"):
        assert measure_ok("t", col, {}, live=False) is False, col
    for col in ("amount", "expected_price", "total_floors", "monthly_fee"):
        assert measure_ok("t", col, {}, live=False) is True, col


def test_siblings_share_a_content_word():
    from ingestion.vocabulary import siblings
    cards = {"a_salelisting": {"table": "a_salelisting", "business_name": "sale listing", "plural": "sale listings"},
             "a_saletransaction": {"table": "a_saletransaction", "business_name": "sale transaction", "plural": "sale transactions"},
             "a_user": {"table": "a_user", "business_name": "user", "plural": "users"}}
    assert siblings(cards, "a_salelisting") == ["a_saletransaction"]


# ══ front-door decomposition (2026-09-25): one message → N intents → N sources ═══════
# The SLM is stubbed with realistic replies (including its typical mistakes: a missed
# part, a policy question labelled 'sql', one question split into its clauses); the
# segmenter, reconcile, source grounding and the compound plan are the real code.
import json as _json  # noqa: E402
from veda.understanding import frame_extractor as FE  # noqa: E402
from veda.understanding import compound as C  # noqa: E402
from veda.understanding.vocabulary import document_card  # noqa: E402
from veda.understanding.frame import Intents, normalise_intents, intents_json_schema  # noqa: E402


def _scope_vocab():
    v = _vocab()
    v.cards.update({
        "vendors": {"table": "vendors", "business_name": "vendor", "plural": "vendors",
                    "aliases": ["supplier"], "key_measures": ["rating"], "key_dimensions": ["city"],
                    "importance": 5},
        "maintenance": {"table": "maintenance", "business_name": "maintenance record",
                        "plural": "maintenance records", "aliases": ["maintenance log"],
                        "key_measures": ["amount"], "lifecycle_column": "status",
                        "key_dimensions": ["category", "status"], "importance": 5},
        "amenities_catalog": {"table": "amenities_catalog", "business_name": "amenity",
                              "plural": "amenities", "aliases": ["facility"],
                              "key_measures": ["monthly_fee"], "key_dimensions": ["category"],
                              "importance": 5},
        "prop_amenity": {"table": "prop_amenity", "business_name": "amenity", "plural": "amenities",
                         "aliases": [], "key_measures": [], "parent_entities": ["prop"], "importance": 3},
    })
    v.source_of.update({"vendors": "4", "maintenance": "4", "amenities_catalog": "5",
                        "prop_amenity": "2"})
    v.measure_glossary.update({"amenities_catalog.monthly_fee": {"type": "MONETARY",
                                                                 "phrases": ["monthly fee", "fee"]}})
    chunks = ["**HANDBOOK** > **LEAVE POLICY**:\nLeaves...",
              "**HANDBOOK** > **LEAVE POLICY** > **SICK LEAVES (SLS):**:\n6 per year",
              "**HANDBOOK** > **LEAVE POLICY** > **EARNED / PRIVILEGED LEAVES (ELS/PLS):**:\n15",
              "**HANDBOOK** > **LEAVE POLICY** > **PATERNITY LEAVE**:\n5 days",
              "**HANDBOOK** > **WORK GUIDELINES** > **WORK FROM HOME (WFH):**:\n4 per quarter",
              "**HANDBOOK** > **CODE OF CONDUCT** > **WORKING WEEK & TIMINGS**:\nMon-Fri",
              "**HANDBOOK** > **CODE OF CONDUCT** > **DRESS CODE**:\nbusiness casual"]
    v.__dict__["doc_cards"] = [document_card("Handbook.pdf", chunks, "3")]
    v.__dict__["source_kind"] = {"2": "sql", "3": "rag", "4": "tabular", "5": "tabular"}
    v.__dict__.pop("_name_idx", None)
    return v


def _i(part, kind="sql", entity=None, **kw):
    return {"part": part, "kind": kind, "entity": entity, "aggregation": kw.pop("aggregation", "none"), **kw}


# message → (the stubbed SLM reply, expected [(kind, source_id, entity or None, outcome)])
COMPOUND_CASES = {
    # ── single ──
    "How many sale listings are there?": (
        {"intents": [_i("How many sale listings are there", entity="sale listing", aggregation="count")],
         "relation": "independent"},
        [("sql", "2", "listing", "grounded")]),
    "What is the dress code policy?": (
        {"intents": [_i("What is the dress code policy", "rag", "Handbook", topics=["dress code"])],
         "relation": "independent"},
        [("rag", "3", "Handbook", "grounded")]),
    "List the ledger entries of each property ordered by amount": (
        # the SLM splits ONE join question into its clauses; no clause boundary → single
        {"intents": [_i("List the ledger entries", entity="ledger entry"),
                     _i("of each property ordered by amount", entity="property")],
         "relation": "independent"},
        [("sql", "2", "ledger", "grounded"), ("sql", "2", "prop", "grounded")]),
    # ── two parts ──
    "How many vendors are there, and what does the handbook say about sick leaves?": (
        {"intents": [_i("How many vendors are there", entity="vendor", aggregation="count"),
                     _i("what does the handbook say about sick leaves", "rag", "Handbook",
                        topics=["sick leaves"])], "relation": "independent"},
        [("tabular", "4", "vendors", "grounded"), ("rag", "3", "Handbook", "grounded")]),
    "Show the most recent ledger entries; what is the average monthly fee of amenities?": (
        {"intents": [_i("Show the most recent ledger entries", entity="ledger entry",
                        order={"concept": "date", "dir": "desc"}),
                     _i("what is the average monthly fee of amenities", entity="amenity",
                        measure="monthly fee", aggregation="avg")], "relation": "independent"},
        [("sql", "2", "ledger", "grounded"), ("tabular", "5", "amenities_catalog", "grounded")]),
    "Which maintenance records are unpaid, and how many days of paternity leave do I get?": (
        # the SLM calls the policy part 'sql' with no entity → the document's topic wins
        {"intents": [_i("Which maintenance records are unpaid", entity="maintenance record",
                        filters=[{"concept": "status", "op": "=", "value": "unpaid"}]),
                     _i("how many days of paternity leave do I get", "sql", None,
                        aggregation="count")], "relation": "independent"},
        [("tabular", "4", "maintenance", "grounded"), ("rag", "3", "Handbook", "grounded")]),
    "List the sale listings priced above 5000 and also show vendors with a rating above 4": (
        {"intents": [_i("List the sale listings priced above 5000", entity="sale listing",
                        filters=[{"concept": "price", "op": ">", "value": 5000}]),
                     _i("show vendors with a rating above 4", entity="vendor",
                        filters=[{"concept": "rating", "op": ">", "value": 4}])],
         "relation": "independent"},
        [("sql", "2", "listing", "grounded"), ("tabular", "4", "vendors", "grounded")]),
    "Can I work from home on Fridays? How many properties do we have?": (
        # the SLM misses the second question → the segmenter adds it
        {"intents": [_i("Can I work from home on Fridays", "rag", "Handbook",
                        topics=["work from home"])], "relation": "independent"},
        [("rag", "3", "Handbook", "grounded"), ("sql", "2", "prop", "grounded")]),
    "Show the top 5 ledger entries by amount, and for those, what is the total amount?": (
        {"intents": [_i("Show the top 5 ledger entries by amount", entity="ledger entry",
                        order={"concept": "amount", "dir": "desc"}, limit=5),
                     _i("for those, what is the total amount", entity="ledger entry",
                        measure="amount", aggregation="sum", depends_on=0)],
         "relation": "dependent"},
        [("sql", "2", "ledger", "grounded"), ("sql", "2", "ledger", "grounded")]),
    # ── three parts ──
    "How many sale listings are there, how many sick leaves do I get each year, and which vendor has the highest rating?": (
        {"intents": [_i("How many sale listings are there", entity="sale listing", aggregation="count"),
                     _i("how many sick leaves do I get each year", "rag", "Handbook",
                        topics=["sick leaves"]),
                     _i("which vendor has the highest rating", entity="vendor",
                        order={"concept": "rating", "dir": "desc"}, limit=1)],
         "relation": "independent"},
        [("sql", "2", "listing", "grounded"), ("rag", "3", "Handbook", "grounded"),
         ("tabular", "4", "vendors", "grounded")]),
    "What is the total maintenance amount per category, what are the office timings, and list the amenities in the sports category?": (
        {"intents": [_i("What is the total maintenance amount per category", entity="maintenance record",
                        measure="amount", aggregation="sum", group_by=["category"]),
                     _i("what are the office timings", "rag", "Handbook", topics=["working week & timings"]),
                     _i("list the amenities in the sports category", entity="amenity",
                        filters=[{"concept": "category", "op": "=", "value": "sports"}])],
         "relation": "independent"},
        [("tabular", "4", "maintenance", "grounded"), ("rag", "3", "Handbook", "grounded"),
         ("tabular", "5", "amenities_catalog", "grounded")]),
    "How many amenities are there, how many vendors are there, and can I carry over earned leaves?": (
        # 'amenities' names a card in source 2 AND source 5 and nothing separates them →
        # a clarify for THAT part only
        {"intents": [_i("How many amenities are there", entity="amenity", aggregation="count"),
                     _i("how many vendors are there", entity="vendor", aggregation="count"),
                     _i("can I carry over earned leaves", "rag", "Handbook",
                        topics=["earned leaves"])], "relation": "independent"},
        [(None, None, None, "clarify"), ("tabular", "4", "vendors", "grounded"),
         ("rag", "3", "Handbook", "grounded")]),
}


def _run_extract(monkeypatch, msg):
    reply = COMPOUND_CASES[msg][0]
    monkeypatch.setattr(FE, "_call", lambda *a, **k: _json.loads(_json.dumps(reply)))
    v = _scope_vocab()
    st = {}
    its = FE.extract_intents(msg, v, doc_cards=v.__dict__["doc_cards"], stats=st)
    return v, its, st


def test_intents_schema_is_a_bounded_list():
    s = intents_json_schema(["vendor"], ["Handbook"])
    assert s["properties"]["intents"]["minItems"] == 1 and s["properties"]["intents"]["maxItems"] == 5
    assert set(s["properties"]["relation"]["enum"]) == {"independent", "dependent"}
    ent = s["properties"]["intents"]["items"]["properties"]["entity"]
    assert "Handbook" in ent["anyOf"][1]["enum"] and "vendor" in ent["anyOf"][1]["enum"]


def test_normalise_intents_clamps_dependency():
    its = normalise_intents({"intents": [{"part": "a", "depends_on": 3}, {"part": "b", "depends_on": 0}],
                             "relation": "dependent"})
    assert its.intents[0].depends_on is None and its.intents[1].depends_on == 0
    its = normalise_intents({"intents": [{"part": "a"}, {"part": "b", "depends_on": 0}],
                             "relation": "independent"})
    assert its.intents[1].depends_on is None


def test_segmenter_splits_on_second_interrogative_not_inside_parentheses():
    segs = FE.segment("List the payments (debits, what kind, credits), what are the office "
                      "hours, and which vendor is best?")
    assert segs == ["List the payments (debits, what kind, credits)", "what are the office hours",
                    "which vendor is best"]
    assert FE.segment("List the ledger entries of each property ordered by amount") == [
        "List the ledger entries of each property ordered by amount"]


def test_compound_messages_frame_count_and_kind(monkeypatch):
    """Part A exit: 12 messages → the right number of frames and the right kind each."""
    for msg, (_reply, want) in COMPOUND_CASES.items():
        v, its, st = _run_extract(monkeypatch, msg)
        assert its is not None, msg
        gs = C.ground_intents(its, v)
        plan = C.plan_compound(its, gs, st["segments"])
        n = 1 if plan == "single" else len(its.intents)
        exp_n = 1 if msg.startswith("List the ledger entries of each") else len(want)
        assert n == exp_n, (msg, n, plan, [f.part for f in its.intents])
        if plan == "compound":
            assert [g.kind for g in gs] == [w[0] for w in want], (msg, [g.kind for g in gs])


def test_compound_messages_ground_each_frame_to_its_source(monkeypatch):
    """Part B exit: each frame's source and entity."""
    for msg, (_reply, want) in COMPOUND_CASES.items():
        v, its, _st = _run_extract(monkeypatch, msg)
        gs = C.ground_intents(its, v)
        got = [(g.kind, g.source_id, g.entity if g.kind != "rag" else g.entity_name, g.outcome)
               for g in gs]
        assert got == want, (msg, got, [g.evidence for g in gs])


def test_dependent_frame_inherits_parent_grounding(monkeypatch):
    v, its, _ = _run_extract(monkeypatch, "Show the top 5 ledger entries by amount, and for "
                                          "those, what is the total amount?")
    assert its.relation == "dependent" and its.intents[1].depends_on == 0
    gs = C.ground_intents(its, v)
    assert gs[1].method == "INHERITED" and gs[1].source_id == gs[0].source_id


def test_clarify_names_business_names_not_tables(monkeypatch):
    v, its, _ = _run_extract(monkeypatch, "How many amenities are there, how many vendors are "
                                          "there, and can I carry over earned leaves?")
    g = C.ground_intents(its, v)[0]
    assert g.outcome == "clarify"
    assert "amenities" in g.message and "prop_amenity" not in g.message \
        and "amenities_catalog" not in g.message


def test_compose_reply_orders_parts_and_drops_doc_negative_on_data_parts():
    parts = [{"part": "how many vendors are there", "outcome": "answered", "lane": "tabular",
              "answer": "There are 6 vendors. The documents do not contain vendor ratings."},
             {"part": "can I work from home", "outcome": "answered", "lane": "rag",
              "answer": "Yes, up to 4 days a quarter.", "citations": ["Handbook.pdf (p.12)"]},
             {"part": "list the amenities", "outcome": "clarify", "lane": None,
              "answer": "'amenities' could mean amenities in homzhub or amenities in catalog."}]
    txt = C.compose_reply(parts, C.fallback_summary(parts))
    assert txt.index("1. How many vendors") < txt.index("2. Can I work") < txt.index("3. List the amenities")
    assert "do not contain" not in txt and "Handbook.pdf" in txt
    assert "2 of 3 parts answered" in txt
    assert not C.summary_is_safe("There are 7 vendors.", parts)
    assert not C.summary_is_safe("The documents do not contain amenities.", parts)


def test_humanize_refusal_uses_business_names(monkeypatch):
    from veda import feedback
    monkeypatch.setattr(feedback, "_business_names", lambda: {"assets_asset": "properties",
                                                              "vendors": "vendors"})
    assert feedback.humanize_refusal("join confidence 0.35 < 0.55 (assets_asset ↔ vendors)") == \
        "I couldn't find a reliable link between properties and vendors."
    out = feedback.humanize_refusal("assets_asset and vendors are not directly related in the "
                                    "schema — no join path exists")
    assert out == "I couldn't find a reliable link between properties and vendors."
    assert "0.55" not in feedback.humanize_refusal("join confidence 0.35 < 0.55")


def test_compound_run_yields_labelled_parts_with_a_clarify(monkeypatch):
    """Part C exit: a stubbed three-frame message → a MultiResult with three labelled
    parts, one of them a clarify, and a composed reply that names all three."""
    import time as _t
    import veda_hybrid as VH
    from query.multi_result import SubResult, OUTCOME_STATUS
    v, its, _ = _run_extract(monkeypatch, "How many amenities are there, how many vendors are "
                                          "there, and can I carry over earned leaves?")
    gs = C.ground_intents(its, v)

    def fake_part(fr, g, parent, deadline, verbose=False, on_event=None):
        if g.outcome == "clarify":
            return SubResult(g.part, OUTCOME_STATUS["clarify"], "clarify",
                             {"ok": False, "status": "clarify", "answer": g.message}, g.message,
                             part=g.part, outcome="clarify", lane=g.kind)
        ans = {"tabular": "There are 6 vendors.", "rag": "No, earned leaves are not carried forward."}[g.kind]
        return SubResult(g.part, "ok", g.kind, {"ok": True, "answer": ans}, part=g.part,
                         outcome="answered", lane=g.kind, source_id=g.source_id)
    monkeypatch.setattr(VH, "_run_part", fake_part)
    import slm
    monkeypatch.setattr(slm, "call_slm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no slm")))
    mr = VH._run_compound("msg", its, gs, _t.time())
    assert mr.compound and len(mr.items) == 3
    assert [it.outcome for it in mr.items] == ["clarify", "answered", "answered"]
    assert all(it.part for it in mr.items)
    for i, it in enumerate(mr.items):
        assert f"{i + 1}. " in mr.summary and it.part.split()[1].lower() in mr.summary.lower()
    assert "6 vendors" in mr.summary and "not carried forward" in mr.summary


def test_injected_frame_is_used_without_extraction(monkeypatch):
    """A part runs the frame path with ITS frame: extract_frame is never called."""
    from veda.understanding import frame_path as FP
    from veda.understanding import frame_extractor as FX
    import config as _cfg
    monkeypatch.setattr(FX, "extract_frame", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-extracted")))
    monkeypatch.setattr(_cfg, "FRAME_PROBES_ENABLED", False, raising=False)
    import veda.understanding.vocabulary as VOC
    monkeypatch.setattr(VOC, "scope_vocab", lambda sm, *a, **k: _vocab())
    fr = Frame(entity="sale listing", aggregation="count")
    fr.provenance["entity_table"] = "listing"
    tok = FP.inject_frame(fr)
    try:
        res = FP.run_frame_path("How many sale listings are there", SM)
    finally:
        FP.reset_injected(tok)
    assert res.trace.get("extract", {}).get("injected") is True
    assert res.kind in ("sql", "clarify", "degrade") and res.reason != "extract_failed"


# ══ B.3 (2026-09-27): front-door reuse filter parity ══════════════════════════════════
# The compound intents extractor (extract_intents) dropped value/numeric filters the
# single-frame extractor (extract_frame) kept, because its schema/prompt/few-shot never
# taught it the filters slot the same way. See reports/VEDA_INTEGRATION_2026-09-27.md §B.1.

def test_intents_item_schema_shares_filter_slots_with_frame_schema():
    """The per-intent schema is built FROM json_schema() (frame_extractor.py's
    intents_json_schema), so the filter/group_by/order/limit/time/distinct shapes are
    byte-identical between the two extractors — not just similarly worded."""
    from veda.understanding.frame import json_schema as frame_schema
    single = frame_schema(entity_enum=["vendor"])
    item = intents_json_schema(["vendor"], ["Handbook"])["properties"]["intents"]["items"]
    for slot in ("filters", "group_by", "order", "limit", "time", "distinct"):
        assert item["properties"][slot] == single["properties"][slot], slot


def test_intents_sys_prompt_carries_the_same_filter_rules_as_frame_sys():
    """_INTENTS_SYS used to summarise the filter/order/time rules in one line; it now
    shares the SAME text _SYS uses (§10.3 — the summary was too thin for the model to
    reliably keep a WHERE clause)."""
    assert FE._SLOT_RULES in FE._SYS
    assert FE._SLOT_RULES in FE._INTENTS_SYS
    assert FE._AGG_RULE in FE._SYS
    assert FE._AGG_RULE in FE._INTENTS_SYS
    # the concrete regression case: the op vocabulary and the "never a bare filter" rule
    for phrase in ("= != > >= < <= between in is_null is_not_null",
                  "Never turn 'recent', 'latest', 'oldest' into a filter"):
        assert phrase in FE._INTENTS_SYS


def test_compound_examples_demonstrate_the_filters_slot():
    """ingestion.vocabulary.compound_examples used to have NO example with a non-empty
    `filters` list — the intent extractor had never seen the shape it was asked to fill.
    Given a value glossary (as the real scope vocab carries), it now demonstrates both a
    value filter (op '=') and a numeric filter (op '>')."""
    from ingestion.vocabulary import compound_examples
    v = _scope_vocab()
    exs = compound_examples(v.cards, v.__dict__["doc_cards"], v.source_of,
                            vg=v.value_glossary, mg=v.measure_glossary)
    ops = {f["op"] for ex in exs for it in ex["intents"] for f in (it.get("filters") or [])}
    assert "=" in ops and ">" in ops


def test_extract_intents_prompt_includes_a_filter_example(monkeypatch):
    """End-to-end: the actual prompt handed to the SLM (captured via a fake _call)
    contains a rendered filters example, not just aggregation/order examples."""
    captured = {}

    def _fake_call(user, system, schema, **kw):
        captured["user"] = user
        captured["system"] = system
        return {"intents": [{"part": "how many vendors are there", "kind": "sql",
                             "entity": "vendor", "aggregation": "count"}],
                "relation": "independent"}

    monkeypatch.setattr(FE, "_call", _fake_call)
    v = _scope_vocab()
    FE.extract_intents("how many vendors are there", v, doc_cards=v.__dict__["doc_cards"])
    assert '"op":"="' in captured["user"] or '"op": "="' in captured["user"]
    assert captured["system"] is FE._INTENTS_SYS


def test_intents_prompt_stays_within_the_ctx_budget():
    """A rough char/4 token estimate for a realistic filled prompt (concepts + documents
    + few-shot + message) against SLM_NUM_CTX, leaving room for FRAME_INTENTS_NUM_PREDICT
    of output. Not exact tokenization, but catches an accidental blow-up from the new
    few-shot examples."""
    import config as _cfg
    from ingestion.vocabulary import compound_examples
    v = _scope_vocab()
    lines, shown = FE._scope_concepts(v, "how many vendors are there", 40)
    doc_lines = FE._doc_lines(v.__dict__["doc_cards"], "how many vendors are there")
    exs = compound_examples(v.cards, v.__dict__["doc_cards"], v.source_of,
                            vg=v.value_glossary, mg=v.measure_glossary)
    parts = ["CONCEPTS (records in the data):", *lines, "", "DOCUMENTS (policies and text):",
             *doc_lines, "", "EXAMPLES:"]
    for ex in exs:
        parts.append(f"Q: {ex['question']}\nINTENTS: {ex}")
    parts += ["MESSAGE: how many vendors are there", "INTENTS:"]
    user = "\n".join(parts)
    est_tokens = (len(FE._INTENTS_SYS) + len(user)) / 4
    budget = getattr(_cfg, "SLM_NUM_CTX", 4096) - getattr(_cfg, "FRAME_INTENTS_NUM_PREDICT", 700)
    assert est_tokens < budget, est_tokens
