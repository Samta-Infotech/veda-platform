"""Tests for apps.sources.source_profiler (multi-source routing, Phase 1.3/1.4).

Covers: grounded description generation (document + tabular), manual-override precedence,
regeneration of a previously auto-generated description, and the flag-gated no-op hook.

Run: `.venv/bin/python manage.py test apps.sources`
"""
from django.test import TestCase, override_settings

from apps.sources.models import Source
from apps.sources import source_profiler as P


class SourceProfilerTests(TestCase):

    def _doc_source(self, **kw):
        return Source.objects.create(
            name=kw.pop("name", "docs"), dialect="filesystem", connector_type="filesystem",
            source_path="/data/contracts", doc_formats=["pdf", "docx"], **kw)

    def _db_source(self, **kw):
        return Source.objects.create(
            name=kw.pop("name", "finance_db"), dialect="postgres", connector_type="postgresql", **kw)

    # ── grounding ──────────────────────────────────────────────────────────────
    def test_document_source_gets_grounded_description(self):
        s = self._doc_source()
        r = P.profile_source(s.pk)
        s.refresh_from_db()
        self.assertTrue(r.updated)
        self.assertTrue(s.description_generated)
        self.assertIn("/data/contracts", s.description)
        self.assertIn("pdf", s.description)

    def test_tabular_description_names_tables_and_capabilities(self):
        from apps.substrate.models import SchemaTable, SchemaColumn
        s = self._db_source()
        t = SchemaTable.objects.create(source=s, tenant="default", name="monthly_revenue",
                                       row_count=1000)
        SchemaColumn.objects.create(source=s, tenant="default", table=t, name="actual_revenue",
                                    data_type="numeric", semantic_type="MONETARY")
        SchemaColumn.objects.create(source=s, tenant="default", table=t, name="month",
                                    data_type="date", semantic_type="TEMPORAL")
        r = P.profile_source(s.pk, tenant="default")
        s.refresh_from_db()
        self.assertTrue(r.updated)
        self.assertIn("monthly_revenue", s.description)
        self.assertIn("aggregation", s.description)   # MONETARY capability
        self.assertIn("trend", s.description)          # TEMPORAL capability

    def test_tabular_with_no_observed_schema_is_skipped(self):
        s = self._db_source()
        r = P.profile_source(s.pk, tenant="default")
        s.refresh_from_db()
        self.assertFalse(r.updated)
        self.assertEqual(r.reason, "no_observed_schema")
        self.assertEqual(s.description, "")

    # ── manual-wins ────────────────────────────────────────────────────────────
    def test_manual_description_is_never_overwritten(self):
        s = self._doc_source(description="Our finance contracts vault", description_generated=False)
        r = P.profile_source(s.pk)
        s.refresh_from_db()
        self.assertEqual(r.reason, "manual_description_kept")
        self.assertEqual(s.description, "Our finance contracts vault")

    def test_previously_generated_description_is_refreshed(self):
        s = self._doc_source(description="stale auto text", description_generated=True)
        r = P.profile_source(s.pk)
        s.refresh_from_db()
        self.assertTrue(r.updated)
        self.assertNotEqual(s.description, "stale auto text")

    def test_profiler_never_sets_domain_or_canonical(self):
        s = self._doc_source()
        P.profile_source(s.pk)
        s.refresh_from_db()
        self.assertEqual(s.domain_tags, [])
        self.assertFalse(s.is_canonical)

    # ── flag gating ────────────────────────────────────────────────────────────
    @override_settings(SOURCE_PROFILER_ENABLED=False)
    def test_hook_is_noop_when_flag_off(self):
        s = self._doc_source()
        P.profile_source_if_enabled(s.pk)
        s.refresh_from_db()
        self.assertEqual(s.description, "")

    @override_settings(SOURCE_PROFILER_ENABLED=True)
    def test_hook_runs_when_flag_on(self):
        s = self._doc_source()
        P.profile_source_if_enabled(s.pk)
        s.refresh_from_db()
        self.assertTrue(s.description_generated)
        self.assertNotEqual(s.description, "")
