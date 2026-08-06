"""Auth routes, mounted under ``api/v1/`` by ``config/urls.py``.

``auth/login`` keeps the exact path the pre-JWT view served from
``apps/chat/urls.py``, so moving authentication into its own app is invisible to
the frontend.
"""
from django.urls import path

from .views import LoginView, LogoutView, TokenRefreshView

urlpatterns = [
    path("auth/login", LoginView.as_view(), name="auth-login"),
    path("auth/refresh", TokenRefreshView.as_view(), name="auth-refresh"),
    path("auth/logout", LogoutView.as_view(), name="auth-logout"),
]
