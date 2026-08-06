"""Shared request-validation pieces for every access-management domain.

Extracted when roles became the second consumer of the same paging contract — not
ahead of it. One definition means users and roles cannot drift into different caps,
different defaults, or one of them forgetting to allowlist ``ordering``.
"""
from __future__ import annotations

from rest_framework import serializers


class PaginatedListSerializer(serializers.Serializer):
    """Base body for any ``<resource>/list`` endpoint.

    Paging and filters travel in the body like every other endpoint's parameters, so
    there is no second place (a query string) for a client to look.

    Validated like any other input for two specific reasons:

      * an unvalidated ``ordering`` goes straight to ``order_by()``, where an
        arbitrary string can traverse relations or raise a 500;
      * an uncapped ``page_size`` lets one caller ask for every row in the table.

    Subclasses declare ``ORDERING_FIELDS`` and re-declare ``ordering`` with their own
    default. ``search`` semantics (which columns it matches) belong to the service,
    not here — this only guarantees a string arrived.
    """

    #: Allowlisted sort keys, each usable with a "-" prefix for descending. Subclasses
    #: MUST narrow this to columns that are indexed or cheap to sort.
    ORDERING_FIELDS: tuple[str, ...] = ("id",)

    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 25

    page = serializers.IntegerField(required=False, min_value=1, default=1)
    page_size = serializers.IntegerField(
        required=False, min_value=1, max_value=MAX_PAGE_SIZE, default=DEFAULT_PAGE_SIZE)
    search = serializers.CharField(required=False, allow_blank=True, default="")
    # Tri-state: None means "no filter", which is NOT the same as False.
    is_active = serializers.BooleanField(required=False, default=None, allow_null=True)
    ordering = serializers.CharField(required=False, default="id")

    def validate_ordering(self, value):
        if value.lstrip("-") not in self.ORDERING_FIELDS:
            raise serializers.ValidationError(
                f"Must be one of: {', '.join(self.ORDERING_FIELDS)} "
                "(optionally prefixed with '-' for descending).")
        return value
