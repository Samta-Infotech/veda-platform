from __future__ import annotations

from django.conf import settings
from django.db import models
from django.utils import timezone


class MessageType(models.TextChoices):
    USER = "user", "User"
    ASSISTANT = "assistant", "Assistant"
    SYSTEM = "system", "System"
    TOOL = "tool", "Tool"


class ChatSession(models.Model):
    name = models.CharField(max_length=255, blank=True)  # chat title
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chat_sessions",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_datetime = models.DateTimeField(null=True, blank=True, default=None)
    is_deleted = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=["user", "is_deleted"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        return f"chatsession#{self.pk} {self.name or '(untitled)'}"

    def soft_delete(self) -> None:
        self.is_deleted = True
        self.deleted_datetime = timezone.now()
        self.save(update_fields=["is_deleted", "deleted_datetime"])


class ChatMessage(models.Model):
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages",
    )
    type = models.CharField(max_length=50, choices=MessageType.choices, db_index=True)
    content = models.TextField()
    metadata = models.JSONField(null=True, blank=True)
    feedback = models.CharField(max_length=50, null=True, blank=True)
    comment = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_datetime = models.DateTimeField(null=True, blank=True, default=None)
    is_deleted = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=["session", "is_deleted"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self) -> str:
        return f"chatmessage#{self.pk} [{self.type}] session={self.session_id}"

    def soft_delete(self) -> None:
        self.is_deleted = True
        self.deleted_datetime = timezone.now()
        self.save(update_fields=["is_deleted", "deleted_datetime"])
