from rest_framework import serializers
from django.conf import settings

class RequestSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, min_length=1)
    service = serializers.ChoiceField(
        choices=['openai'],  # List valid services
        required=True
    )

    def validate_text(self, value):
        if not value.strip():
            raise serializers.ValidationError("Text cannot be empty or whitespace")
        return value 