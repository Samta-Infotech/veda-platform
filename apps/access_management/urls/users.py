"""User-administration routes.

``<resource>/<action>``, all POST — the convention ``apps/chat`` already uses
(``conversations/create``, ``conversations/list``, ``conversations/history``). Not
REST verbs: one style across the platform beats a textbook style in one app, and
every parameter travels in the body rather than being split between query string
and path.
"""
from django.urls import path

from ..views import (
    UserCreateView,
    UserDeleteView,
    UserDetailView,
    UserListView,
    UserUpdateView,
)

urlpatterns = [
    path("users/create", UserCreateView.as_view(), name="user-create"),
    path("users/detail", UserDetailView.as_view(), name="user-detail"),
    path("users/list", UserListView.as_view(), name="user-list"),
    path("users/update", UserUpdateView.as_view(), name="user-update"),
    path("users/delete", UserDeleteView.as_view(), name="user-delete"),
]
