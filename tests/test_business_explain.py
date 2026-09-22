"""Tests for veda/business_explain.py's extract_sql_facts() — the public
wrapper onto the existing zero-LLM sqlglot AST pass, exposed for reuse by
veda/result_analyzer.py. Pure-python, no DB, no network."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "veda_core"))
from veda.business_explain import build_explain
from veda.business_explain import build_refusal_explain
from veda.feedback import explain_failure


def test_extract_sql_facts_matches_private_extract():
    from veda.business_explain import extract_sql_facts, _extract
    sql = ('SELECT payer_name, SUM(amount) AS total FROM ledger '
           'WHERE entry_type = \'CREDIT\' GROUP BY payer_name '
           'ORDER BY total DESC LIMIT 5')
    assert extract_sql_facts(sql) == _extract(sql)


def test_extract_sql_facts_aggregation_and_grouping():
    from veda.business_explain import extract_sql_facts
    sql = 'SELECT status, COUNT(*) AS n FROM incidents GROUP BY status'
    facts = extract_sql_facts(sql)
    assert facts["entities"] == ["incidents"]
    assert facts["groupings"] == ["status"]
    assert ("COUNT", None) in facts["aggregations"]


def test_extract_sql_facts_orderings_and_limit():
    from veda.business_explain import extract_sql_facts
    sql = 'SELECT id FROM ledger ORDER BY amount DESC LIMIT 10'
    facts = extract_sql_facts(sql)
    assert facts["orderings"] == [("amount", True)]
    assert facts["limit"] == 10


def test_extract_sql_facts_filters():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE amount > 100"
    facts = extract_sql_facts(sql)
    assert ("amount", "GT", "100") in facts["filters"]


def test_extract_sql_facts_invalid_sql_returns_safe_empty_shape():
    from veda.business_explain import extract_sql_facts
    facts = extract_sql_facts("not valid sql at all !!!")
    assert facts["entities"] == []
    assert facts["limit"] is None


# ---------------------------------------------------------------------------
# Phase 2 gap-fill: build_explain() surfaces the Insight Engine's validated
# visualization reasoning — additive only, omitted when None (existing
# callers/consumers unaffected).
# ---------------------------------------------------------------------------

def test_build_explain_omits_visualization_key_by_default():
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert "visualization" not in out


def test_build_explain_includes_validated_visualization():
    """Reasoning is deterministic/standardized (Final Polish, Section 9) — the
    SLM's own free-text "reason" is NOT surfaced verbatim; a known chart type
    always gets the same, LLM-free phrasing."""
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    out = build_explain(
        sql='SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name',
        table="ledger", sm=sm,
        visualization={"type": "bar", "x_axis": "payer_name", "y_axis": "total",
                       "reason": "categorical vs numeric comparison"},
    )
    assert out["visualization"]["type"] == "bar"
    assert out["visualization"]["reason"] == (
        "Bar chart selected because the query compares a numeric measure "
        "across discrete categories."
    )


def test_build_explain_unknown_chart_type_falls_back_to_slm_reason():
    out = build_explain(
        sql='SELECT a FROM t', table="t", sm=None,
        visualization={"type": "scatter", "x_axis": None, "y_axis": None,
                       "reason": "a free-text reason with no deterministic template"},
    )
    assert out["visualization"]["reason"] == "a free-text reason with no deterministic template"


# ---------------------------------------------------------------------------
# Filter-value resolution from PARAMETERIZED sql: validate_and_parameterize()
# rewrites every filter literal into a %s placeholder (bound separately in
# `params`) for safe execution — _extract()/build_explain() must resolve the
# real value back from `params` by position, not just report None.
# ---------------------------------------------------------------------------

def test_extract_sql_facts_resolves_placeholder_value_from_params():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql, params=["open"])
    assert ("status", "EQ", "open") in facts["filters"]


def test_extract_sql_facts_without_params_still_degrades_to_none():
    """Regression guard: existing callers that don't pass `params` (the
    default) must see identical behavior to before this fix — None, not a
    crash or a changed shape."""
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql)
    assert ("status", "EQ", None) in facts["filters"]


def test_extract_sql_facts_multiple_placeholders_resolve_by_position():
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s AND entry_type = %s"
    facts = extract_sql_facts(sql, params=["open", "DEBIT"])
    assert ("status", "EQ", "open") in facts["filters"]
    assert ("entry_type", "EQ", "DEBIT") in facts["filters"]


def test_extract_sql_facts_placeholder_index_out_of_range_degrades_to_none():
    """Fewer params than placeholders (shouldn't happen in practice, but must
    never crash) — degrades to None for the unresolvable one, same as today."""
    from veda.business_explain import extract_sql_facts
    sql = "SELECT id FROM ledger WHERE status = %s"
    facts = extract_sql_facts(sql, params=[])
    assert ("status", "EQ", None) in facts["filters"]


def test_build_explain_filter_value_resolved_with_params():
    out = build_explain(
        sql="SELECT id FROM ledger WHERE entry_type = %s", table="ledger", sm=None,
        params=["DEBIT"],
    )
    applied = out["filters"]["applied"]
    assert len(applied) == 1
    assert applied[0]["value"] == "DEBIT"
    assert "DEBIT" in out["filters"]["summary"]


def test_build_explain_filter_value_none_without_params_unchanged():
    """Regression guard: existing callers of build_explain() that don't pass
    `params` (every caller before this fix) see the exact same None-value
    behavior as before — this fix is additive/opt-in only."""
    out = build_explain(sql="SELECT id FROM ledger WHERE entry_type = %s",
                        table="ledger", sm=None)
    applied = out["filters"]["applied"]
    assert len(applied) == 1
    assert applied[0]["value"] is None


# ---------------------------------------------------------------------------
# Explainability-gap fixes: refusal explain (Item 1), confidence placeholder
# (Item 2), timeline (Item 3) — all additive, existing fields/behavior
# unaffected.
# ---------------------------------------------------------------------------

def test_build_explain_confidence_key_always_present_and_none():
    """Item 2: schema-only placeholder — never a computed/fake number."""
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert "confidence" in out
    assert out["confidence"] is None


def test_build_explain_timeline_defaults_to_empty_list():
    """Item 3: omitting `timeline` must not change any existing caller's output
    shape beyond adding this one always-present, empty-by-default key."""
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert out["timeline"] == []


def test_build_explain_timeline_relays_ticks_verbatim_in_order():
    """Item 3: timeline is a passive relay of the run's own _tick() checkpoints
    — same messages, same order, no re-derivation."""
    ticks = [("schema_linking", "Using ledger for this"),
             ("sql_planning", "Narrowing to that time period"),
             ("output", "Done — here's your answer")]
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None, timeline=ticks)
    assert out["timeline"] == [
        {"phase": "schema_linking", "message": "Using ledger for this"},
        {"phase": "sql_planning", "message": "Narrowing to that time period"},
        {"phase": "output", "message": "Done — here's your answer"},
    ]


def test_build_explain_existing_fields_unaffected_by_new_keys():
    """Regression guard: adding confidence/timeline must not change any
    existing field's value for a caller that predates both."""
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    out = build_explain(
        sql='SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name',
        table="ledger", sm=sm,
    )
    assert out["version"] == "1.0"
    assert out["data_used"]["datasets"] == ["Ledgers"]
    assert out["operations"] == [
        {"type": "total", "summary": "Calculate total Amount"},
        {"type": "group", "summary": "Group by Payer Name"},
    ]
    # The SQL block's SHAPE is part of the v1 contract and must always be present;
    # whether it is POPULATED is EXPLAIN_EXPOSE_SQL's call, so assert against the
    # flag rather than against whatever the default happens to be. That default has
    # moved (on -> off under D2 -> back on 2026-09-11 at the user's request), which
    # is exactly why this test pins the SHAPE and leaves the value to the flag.
    assert "sql" in out and "query" in out["sql"] and "enabled" in out["sql"]


def test_sql_block_follows_the_expose_flag_in_both_states(monkeypatch):
    """D2: the SQL is exposed only when EXPLAIN_EXPOSE_SQL says so, and the block
    keeps its shape either way so an existing v1 consumer never sees a missing key."""
    import config
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    sql = 'SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name'

    monkeypatch.setattr(config, "EXPLAIN_EXPOSE_SQL", True, raising=False)
    on = build_explain(sql=sql, table="ledger", sm=sm)
    assert on["sql"]["enabled"] is True
    assert on["sql"]["query"].startswith("SELECT payer_name")

    monkeypatch.setattr(config, "EXPLAIN_EXPOSE_SQL", False, raising=False)
    off = build_explain(sql=sql, table="ledger", sm=sm)
    assert off["sql"]["enabled"] is False
    assert off["sql"]["query"] is None
    # And nothing else in the payload may carry the SQL text as a side effect.
    import json
    assert "SELECT payer_name" not in json.dumps(off)


def test_build_refusal_explain_returns_none_without_feedback():
    """Item 1: no feedback dict (e.g. FEEDBACK_ENABLED=False, or the
    invalid/exec_error _done() call sites that never build one) -> None,
    same "no explain" signal the answered path already uses."""
    assert build_refusal_explain("no_table", None) is None
    assert build_refusal_explain("invalid", {}) is None


def test_build_refusal_explain_no_table():
    fb = explain_failure("no_table", {}, candidates=["ledger", "invoice"])
    out = build_refusal_explain("no_table", fb)
    assert out["version"] == "1.0"
    assert out["status"] == "no_table"
    assert out["understanding"]["summary"] == fb["why"]
    assert "couldn't confidently match" in out["why"]
    assert out["what_would_help"] == fb["what_needed"]
    assert out["suggestions"] == ["ledger", "invoice"]


def test_build_refusal_explain_qualifier_dropped():
    """A second, differently-shaped refusal status — proves the fix isn't
    special-cased to just one status."""
    fb = explain_failure("qualifier_dropped", {"columns": {"ledger.status": {}}},
                         missing="pending")
    out = build_refusal_explain("qualifier_dropped", fb)
    assert out["status"] == "qualifier_dropped"
    assert "pending" in out["why"]
    assert out["what_would_help"]


# ---------------------------------------------------------------------------
# Item 4 (optional) — understanding.breakdown, additive alongside summary.
# ---------------------------------------------------------------------------

def test_build_explain_understanding_breakdown_is_additive():
    """`summary` (the single prose sentence) must stay byte-identical; the
    new `breakdown` list is assembled from the same operations/filter_phrases
    already computed, not a new derivation."""
    sm = {"columns": {"ledger.status": {"business_role": "Status"}}}
    out = build_explain(
        sql="SELECT status, COUNT(*) AS n FROM ledger WHERE status = 'open' "
            "GROUP BY status ORDER BY n DESC LIMIT 5",
        table="ledger", sm=sm,
    )
    # "by N" was the COUNT's SELECT alias leaking into the sentence — an ORDER BY on a
    # ranked aggregate names the alias, so the label has to name the MEASURE it computes.
    assert out["understanding"]["summary"] == (
        "Find the top 5 Statuses by record count where Status equals open."
    )
    assert out["understanding"]["breakdown"] == [
        "Count records",
        "Group by Status",
        "Sort by record count (highest first)",
        "Return top 5",
        "Status equals open",
    ]


def test_build_explain_understanding_breakdown_empty_for_bare_list():
    """No aggregation/grouping/ordering/limit/filter -> operations defaults
    to a single 'List <dataset>' entry, filter_phrases is empty -> breakdown
    is that one phrase, matching `operations` exactly."""
    out = build_explain(sql="SELECT id FROM ledger", table="ledger", sm=None)
    assert out["understanding"]["breakdown"] == ["List records"]


# ---------------------------------------------------------------------------
# Alias-qualified columns resolve to the table they ACTUALLY came from.
# Regression: a join's columns carry SQL aliases ("t2"."name"), which resolved
# against neither the alias nor the right table, so the business name came from
# a model-wide suffix scan — a ticket category's `name` was labelled "Device
# Name" off an unrelated fcm-device table, and the whole panel described a query
# that never ran ("Find the top 100 Device Names by Ticket Identifier").
# ---------------------------------------------------------------------------

_JOIN_SQL = (
    'SELECT "t2"."name" AS "category_name", COUNT("t0"."id") AS "activity_count" '
    'FROM "worklists_ticketupdate" AS "t0" '
    'JOIN "worklists_ticket" AS "t1" ON "t1"."id" = "t0"."ticket_id" '
    'JOIN "worklists_ticketcategory" AS "t2" ON "t2"."id" = "t1"."ticket_category_id" '
    'GROUP BY "t2"."name" ORDER BY "activity_count" DESC LIMIT 100'
)
_JOIN_SM = {
    "tables": {
        "worklists_ticketupdate": {"primary_entity": "An update to a ticket."},
        "worklists_ticket": {"primary_entity": "A support ticket."},
        "worklists_ticketcategory": {"primary_entity": "A single category for tickets."},
    },
    "columns": {
        # the unrelated table whose `name` the old suffix scan reached first
        "fcm_django_fcmdevice.name": {"business_role": "Device Name"},
        "worklists_ticketcategory.name": {"business_role": "Category Name"},
        "worklists_ticket.id": {"business_role": "Ticket Identifier"},
        "worklists_ticketupdate.id": {"business_role": "Update Identifier"},
    },
}


def test_extract_resolves_column_aliases_to_real_tables():
    from veda.business_explain import extract_sql_facts
    facts = extract_sql_facts(_JOIN_SQL)
    assert facts["column_tables"]["name"] == "worklists_ticketcategory"
    assert facts["column_tables"]["id"] == "worklists_ticketupdate"   # COUNT("t0"."id")
    assert facts["from_table"] == "worklists_ticketupdate"
    assert facts["alias_aggs"]["activity_count"] == ("COUNT", "id")


def test_build_explain_labels_join_columns_by_their_own_table():
    out = build_explain(sql=_JOIN_SQL, table="worklists_ticketupdate", sm=_JOIN_SM)
    assert out["understanding"]["summary"] == "Find the top 100 Category Names by record count."
    assert out["operations"][1]["summary"] == "Group by Category Name"
    assert out["operations"][2]["summary"] == "Sort by record count (highest first)"
    assert "Device Name" not in json.dumps(out)


def test_build_explain_ambiguous_bare_column_is_humanized_not_guessed():
    """An unqualified column that isn't on the driving table can't be attributed to any
    one table. The model-wide scan must NOT then pick an arbitrary same-named column —
    a plain humanized name is the honest answer."""
    sm = {"columns": {"b.status": {"business_role": "Lease Status"},
                      "c.status": {"business_role": "Payment Status"}}}
    out = build_explain(sql="SELECT status FROM a JOIN b ON b.id = a.b_id GROUP BY status",
                        table="a", sm=sm)
    assert out["operations"][0]["summary"] == "Group by Status"
    # ...while a bare column the DRIVING table does own still gets its business name.
    sm2 = {"columns": {"a.status": {"business_role": "Lease Status"}}}
    out2 = build_explain(sql="SELECT status FROM a JOIN b ON b.id = a.b_id GROUP BY status",
                         table="a", sm=sm2)
    assert out2["operations"][0]["summary"] == "Group by Lease Status"


def test_business_table_name_pluralizes_the_head_noun_once():
    from veda.business_explain import _business_table_name
    sm = {"tables": {
        "t_update": {"primary_entity": "An update to a ticket."},
        "t_cat": {"primary_entity": "A single category for tickets."},     # already plural
        "t_att": {"primary_entity": "A single attachment associated with a ticket."},
        "t_plain": {"primary_entity": "A support ticket."},
    }}
    assert _business_table_name("t_update", sm) == "Updates To A Ticket"
    assert _business_table_name("t_cat", sm) == "Single Categories For Tickets"
    assert _business_table_name("t_att", sm) == "Single Attachments Associated With A Ticket"
    assert _business_table_name("t_plain", sm) == "Support Tickets"


def test_build_explain_not_included_lists_omitted_entities():
    out = build_explain(sql=_JOIN_SQL, table="worklists_ticketupdate", sm=_JOIN_SM,
                        not_included=["worklists_ticketattachment"])
    assert out["data_used"]["not_included"] == ["Ticketattachments"]
    assert "not_included" not in build_explain(sql=_JOIN_SQL, table="", sm=_JOIN_SM)["data_used"]


def test_build_explain_entity_coverage_check_is_labelled():
    out = build_explain(sql=_JOIN_SQL, table="worklists_ticketupdate", sm=_JOIN_SM,
                        checks=[{"name": "entity_coverage", "status": "fail"}])
    assert out["validation"]["passed"] is False
    assert out["validation"]["checks"] == [
        {"label": "Every entity you asked about was included", "passed": False}]


def test_repeated_stage_check_is_stated_once():
    """The firewall runs in stages and records a check per stage — the raw internal name
    "firewall" rendered four times in the end-user panel. One guarantee, stated once."""
    checks = [{"name": "firewall", "status": "pass"},
              {"name": "value_grounding", "status": "pass"},
              {"name": "firewall", "status": "pass"},
              {"name": "firewall", "status": "pass"}]
    out = build_explain(sql=_JOIN_SQL, table="worklists_ticketupdate", sm=_JOIN_SM, checks=checks)
    labels = [c["label"] for c in out["validation"]["checks"]]
    assert labels == ["Passed every SQL safety check", "All filter values exist in the data"]
    assert "firewall" not in json.dumps(out["validation"])


def test_a_failed_stage_is_never_hidden_by_an_earlier_pass():
    out = build_explain(sql=_JOIN_SQL, table="worklists_ticketupdate", sm=_JOIN_SM,
                        checks=[{"name": "firewall", "status": "pass"},
                                {"name": "firewall", "status": "fail"}])
    assert out["validation"]["checks"] == [
        {"label": "Passed every SQL safety check", "passed": False}]
    assert out["validation"]["passed"] is False
def test_validation_passed_is_unknown_when_nothing_was_checked():
    """`all_passed` starts True and an empty check list never falsifies it, so a
    payload with `checks: []` used to claim `passed: true` — telling the reader the
    result cleared checks that never ran. Observed live on a document answer."""
    out = build_explain(sql="", table="", sm=None)
    assert out["validation"]["checks"] == []
    assert out["validation"]["passed"] is None, (
        "no checks ran, so `passed` must be unknown — not True")


def test_validation_passed_still_reflects_real_checks():
    sm = {"columns": {"ledger.total": {"business_role": "Total Amount"}}}
    sql = "SELECT payer_name, SUM(amount) AS total FROM ledger GROUP BY payer_name"
    ok = build_explain(sql=sql, table="ledger", sm=sm,
                       checks=[{"name": "value_grounding", "status": "pass"}])
    assert ok["validation"]["checks"] and ok["validation"]["passed"] is True

    bad = build_explain(sql=sql, table="ledger", sm=sm,
                        checks=[{"name": "value_grounding", "status": "fail"}])
    assert bad["validation"]["passed"] is False


# ===================================================== catalog names, not ids
class TestExposedSqlCarriesNoSourceIds:
    """The federated executor attaches every source under a catalog named
    `src_<id>`, so a cross-source statement reads `src_2.homzhub."assets_asset"`.
    With the SQL restored to the payload (2026-09-11) that identifier would walk
    straight past `demote_source_ids`, which exists precisely to keep source ids
    out of the user-facing blocks."""

    def _profiles(self, prof=None):
        import importlib
        prof = prof if prof is not None else {
            "2": {"name": "homzhub", "source_type": "relational"},
            "4": {"name": "invoices_csv", "source_type": "datalake"}}
        for name in ("context", "veda_core.context"):
            try:
                importlib.import_module(name).set_source_profiles(prof)
            except Exception:
                pass

    def _f(self):
        from veda.business_explain import _name_catalogs
        return _name_catalogs

    def test_a_catalog_becomes_the_display_name(self):
        self._profiles()
        assert self._f()('SELECT COUNT(*) FROM src_2.homzhub."assets_asset"') == \
            'SELECT COUNT(*) FROM "homzhub".homzhub."assets_asset"'

    def test_every_catalog_in_a_join_is_replaced(self):
        self._profiles()
        out = self._f()('SELECT a.x FROM src_2.homzhub."assets" a '
                        'FULL JOIN src_4."invoices" b USING ("city")')
        assert "src_2" not in out and "src_4" not in out
        assert '"homzhub"' in out and '"invoices_csv"' in out

    def test_a_single_source_statement_is_untouched(self):
        self._profiles()
        sql = 'SELECT "amount" FROM "accounts_txn" WHERE id = %s'
        assert self._f()(sql) == sql

    def test_an_unauthorised_source_gets_the_generic_label_never_its_id(self):
        """display_name never falls back to the raw id — that is what makes this
        safe rather than cosmetic."""
        self._profiles()
        out = self._f()('SELECT * FROM src_99."secret_table"')
        assert "src_99" not in out and "99" not in out
        assert '"a data source"' in out

    def test_nothing_is_published_raw_when_the_lookup_itself_fails(self, monkeypatch):
        import veda.source_names as sn
        monkeypatch.setattr(sn, "display_name",
                            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
        out = self._f()('SELECT * FROM src_2."t"')
        assert "src_2" not in out, "a failed lookup must not fall through to the id"

    def test_empty_and_missing_sql_stay_none(self):
        assert self._f()("") is None and self._f()(None) is None
