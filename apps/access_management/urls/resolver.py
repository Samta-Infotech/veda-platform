"""Effective-permission route."""
from django.urls import path

from ..views import EffectivePermissionsView

urlpatterns = [
    path("users/permissions/effective", EffectivePermissionsView.as_view(),
         name="user-effective-permissions"),
]
