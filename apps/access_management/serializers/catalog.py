"""Catalog request validation — read-only.

The catalog is a *projection*: rows are produced by discovery from the authoritative
catalog, never authored through the API. So there is no create/update serializer, for
the same reason there is none for permissions — an operator-invented resource would be
a name nothing upstream corresponds to, and granting on it would be granting on
nothing.
"""
from __future__ import annotations

from rest_framework import serializers

from .. import resource_path
from .base import PaginatedListSerializer


class CatalogResourceDetailSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/catalog/detail`` — which resource to fetch.

    Addressed by ``path``, not by id: the path IS the identity a grant references, and
    it is stable across environments where an autoincrement id is not.
    """

    path = serializers.CharField(max_length=resource_path.MAX_LENGTH)


class CatalogResourceListSerializer(PaginatedListSerializer):
    """Body of ``POST /api/v1/catalog/list``.

    Adds the tree-navigation and narrowing filters to the shared paging contract.
    ``search`` matches the path substring — which, because paths are hierarchical,
    doubles as a subtree search.
    """

    ORDERING_FIELDS = ("id", "path", "kind")

    ordering = serializers.CharField(required=False, default="path")

    #: Exact parent match — a lazy tree loads one level per call. Tri-state: omitted
    #: means "no level filter", "" selects the source-level roots, a path selects that
    #: node's children.
    parent_path = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None)
    kind = serializers.CharField(required=False, allow_blank=True, default="")
    source_id = serializers.IntegerField(required=False, min_value=1,
                                         allow_null=True, default=None)

    def validate_kind(self, value):
        """Reject an unknown kind rather than silently returning nothing — an empty
        page and a typo look identical to a client otherwise."""
        if value and value not in set(resource_path.KIND_BY_DIALECT.values()):
            raise serializers.ValidationError(
                f"Must be one of: {', '.join(sorted(set(resource_path.KIND_BY_DIALECT.values())))}.")
        return value

    def validate_parent_path(self, value):
        """Canonicalise, so a caller navigating with a differently-cased path still
        finds the children stored under the canonical one."""
        if value is None or value == "":
            return value      # None = no filter, "" = roots. Both pass through.
        try:
            return resource_path.validate(value)
        except resource_path.InvalidResourcePath as exc:
            raise serializers.ValidationError(str(exc)) from exc
