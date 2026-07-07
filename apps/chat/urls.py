from django.urls import path

from .views import (
    ConversationHistoryView,
    ConversationQueryView,
    CreateConversationView,
    ListConversationsView,
    LoginView,
)

urlpatterns = [
    path("auth/login", LoginView.as_view(), name="login"),
    path("conversations/query", ConversationQueryView.as_view(), name="conversation-query"),
    path("conversations/create", CreateConversationView.as_view(), name="conversation-create"),
    path("conversations/list", ListConversationsView.as_view(), name="conversation-list"),
    path("conversations/history", ConversationHistoryView.as_view(), name="conversation-history"),
]
