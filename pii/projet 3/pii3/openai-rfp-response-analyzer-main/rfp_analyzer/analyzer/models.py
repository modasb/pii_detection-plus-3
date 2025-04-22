from django.db import models
from django.utils import timezone

class Document(models.Model):
    DOCUMENT_TYPES = [
        ('RFP', 'Request for Proposal'),
        ('RESPONSE', 'Response Document')
    ]
    
    title = models.CharField(max_length=255)
    document_type = models.CharField(max_length=10, choices=DOCUMENT_TYPES)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    faiss_index_path = models.CharField(max_length=255, null=True, blank=True)
    
    def __str__(self):
        return f"{self.title} ({self.document_type})"

class Analysis(models.Model):
    rfp_document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='rfp_analyses')
    response_document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='response_analyses')
    report = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Analysis: {self.rfp_document.title} - {self.response_document.title}"

class ChatMessage(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='chat_messages')
    query = models.TextField()
    response = models.TextField()
    has_pii = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chat: {self.query[:50]}..." 