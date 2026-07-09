from rest_framework import serializers


class LoginRequestSerializer(serializers.Serializer):
    username = serializers.CharField(required=True, allow_blank=False)
    password = serializers.CharField(required=True, allow_blank=False)


class ConversationQuerySerializer(serializers.Serializer):
    message = serializers.CharField(required=True, allow_blank=False)
    chat_id = serializers.IntegerField(required=False, allow_null=True, default=None)
    stream = serializers.BooleanField(required=False, default=True)


class CreateConversationSerializer(serializers.Serializer):
    conversation_title = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, default=None
    )


class ConversationHistorySerializer(serializers.Serializer):
    chat_id = serializers.IntegerField(required=True)
