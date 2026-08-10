"""Effective-permission request validation."""
from __future__ import annotations

from rest_framework import serializers

from .. import resource_path as rp


class EffectivePermissionsSerializer(serializers.Serializer):
    """Body of ``POST /api/v1/users/permissions/effective``.

    ``permission_code`` + ``resource_path`` are optional: supplying them asks the
    resolver a *specific* question ("may this user do X on Y?") and adds a ``decision``
    block to the response. Omitting them returns the whole effective set.

    Answering the specific question server-side matters — a client that re-implements
    prefix inheritance and DENY precedence will eventually disagree with the resolver,
    and a UI that shows "allowed" where the server denies is worse than no UI.
    """

    user_id = serializers.IntegerField(min_value=1)
    permission_code = serializers.CharField(
        required=False, allow_blank=True, default="", max_length=100)
    resource_path = serializers.CharField(
        required=False, allow_blank=True, default="", max_length=rp.MAX_LENGTH)

    def validate_resource_path(self, value):
        """Canonicalise, so the question is asked in the same shape the grants are
        stored in."""
        if not value:
            return ""
        try:
            return rp.validate(value)
        except rp.InvalidResourcePath as exc:
            raise serializers.ValidationError(str(exc)) from exc

    def validate(self, attrs):
        if attrs.get("resource_path") and not attrs.get("permission_code"):
            raise serializers.ValidationError({
                "permission_code": [
                    "Required when resource_path is given — a resource alone is not a "
                    "question the resolver can answer."]})
        return attrs
