from fastapi import APIRouter, HTTPException
from core.pii_detector import PIIDetector
from .models import PIIRequest, PIIResponse, HealthResponse
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize PII detector
pii_detector = PIIDetector()

@router.post("/detect", response_model=PIIResponse, tags=["PII Detection"])
async def detect_pii(request: PIIRequest):
    """
    Detect PII in the provided text.
    
    Returns both the detected entities and optionally anonymized text.
    """
    try:
        result = pii_detector.detect_and_anonymize(request.text, request.language)
        
        # If anonymization is not requested, set anonymized_text to None
        if not request.anonymize:
            result['anonymized_text'] = None
            
        return PIIResponse(**result)
    except Exception as e:
        logger.error(f"Error processing PII detection request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and version.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )

@router.get("/supported-types", tags=["PII Detection"])
async def get_supported_types():
    """
    Get the list of supported PII types and their patterns.
    
    Returns a dictionary of supported PII types and their detection patterns.
    """
    return {
        "supported_types": list(pii_detector.patterns.keys()),
        "context_words": pii_detector.context_words
    } 