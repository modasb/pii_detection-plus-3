from django.db import models
from django.contrib.auth import get_user_model

# Create your models here.

class PIIAuditLog(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    original_text_hash = models.CharField(max_length=64)
    num_entities_detected = models.IntegerField()
    entity_types = models.JSONField()

    def __str__(self):
        return f"PII Audit Log {self.timestamp} - {self.num_entities_detected} entities"
