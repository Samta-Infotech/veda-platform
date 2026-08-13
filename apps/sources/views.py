"""Thin DRF view for Data Sources listing.

Follows the coding standard defined in ``apps/access_management/views/roles.py``:
inherits from ``AdminView``, uses ``self.validate(request)``, and returns via ``apps.core.api``.

When the Django ``Source`` table has rows, those are returned directly (raw DB).
When it is empty, sources configured via ``veda_core/config.py`` env are returned
as a fallback — so the frontend always has data to render.

Deliberately un-paginated and CONNECTED-only (user's call): the number of data
sources an org onboards is small and admin-curated, never user-generated, so a
page control adds nothing — and a source that isn't actually connected yet
(``ready=False``) isn't a "meaningful" source to show in this list at all, it's
onboarding-in-progress noise. This is a deliberate deviation from
``api_contract.md`` §5.2, which still documents pagination + the ERROR status —
update that doc alongside this file if the two need to stay in sync.
"""
from __future__ import annotations

from apps.access_management.codes import PermissionCode
from apps.access_management.views.base import AdminView
from apps.core import api
from apps.sources.models import Source
from apps.sources.serializers import (
    DataSourceListSerializer,
    dialect_to_source_type,
    get_config_sources,
    group_by_type,
    serialize_source,
)


class DataSourceListView(AdminView):
    """POST /api/v1/data-sources/list & GET /api/v1/data-sources/list

    Lists connected data sources with their metadata. No pagination, no
    not-yet-connected entries — see the module docstring for why.
    """

    serializer_class = DataSourceListSerializer
    action = "data sources list"
    required_permission = PermissionCode.SOURCE_MANAGE

    def _handle_list(self, data):
        # 1. Try Django Source table first (the real DB)
        db_count = Source.objects.count()

        if db_count > 0:
            # Source table has rows — use them directly
            items = self._from_db(data)
        else:
            # Source table empty — fallback to veda_core/config.py env sources
            items = self._from_config(data)

        return api.success("data source status fetched successfully", group_by_type(items))

    def _from_db(self, data):
        """Fetch from Django Source model (real DB rows) — connected only."""
        qs = Source.objects.filter(ready=True).order_by("pk")

        source_type = data.get("source_type")
        if source_type:
            target_st = source_type.upper()
            all_sources = list(qs)
            qs_ids = [s.pk for s in all_sources if dialect_to_source_type(s.dialect) == target_st]
            qs = Source.objects.filter(pk__in=qs_ids).order_by("pk")

        search = data.get("search")
        if search:
            qs = qs.filter(name__icontains=search)

        return [serialize_source(s) for s in qs]

    def _from_config(self, data):
        """Fallback: fetch from veda_core/config.py env-injected sources.

        ``get_config_sources()`` only ever surfaces enabled/connected engines, so
        every item here is already "connected" — nothing to filter out.
        """
        items = get_config_sources()

        source_type = data.get("source_type")
        if source_type:
            target_st = source_type.upper()
            items = [i for i in items if i["source_type"] == target_st]

        search = data.get("search")
        if search:
            items = [i for i in items if search.lower() in i["name"].lower()]

        return items

    def post(self, request):
        data, failure = self.validate(request)
        if failure:
            return failure
        return self._handle_list(data)

    def get(self, request):
        params = {
            "source_type": request.query_params.get("source_type"),
            "search": request.query_params.get("search", ""),
        }
        serializer = self.serializer_class(data=params)
        if not serializer.is_valid():
            return api.invalid_payload(serializer.errors)
        return self._handle_list(serializer.validated_data)
