from django.db import models
from django.contrib.auth import get_user_model

class PIIAuditLog(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    original_text_hash = models.CharField(max_length=64)
    num_entities_detected = models.IntegerField()
    entity_types = models.JSONField() 