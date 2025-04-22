from typing import List, Optional
from pydantic import BaseModel, Field

class PIIEntity(BaseModel):
    """Model for a detected PII entity."""
    type: str = Field(..., description="Type of PII entity (e.g., CREDIT_CARD, EMAIL, etc.)")
    start: int = Field(..., description="Start position of the entity in text")
    end: int = Field(..., description="End position of the entity in text")
    value: str = Field(..., description="The detected PII value")
    confidence: float = Field(..., description="Confidence score of the detection")

class PIIRequest(BaseModel):
    """Model for PII detection request."""
    text: str = Field(..., description="Text to analyze for PII")
    language: Optional[str] = Field(None, description="Optional language code (e.g., 'en', 'fr')")
    anonymize: bool = Field(True, description="Whether to return anonymized text")

class PIIResponse(BaseModel):
    """Model for PII detection response."""
    entities: List[PIIEntity] = Field(..., description="List of detected PII entities")
    anonymized_text: Optional[str] = Field(None, description="Anonymized version of the input text")

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version") 