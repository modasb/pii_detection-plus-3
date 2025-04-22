"""FastAPI application for PII detection service."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PIIDetector class (not the function)
from pii_detector import PIIDetector
from pii_analyzer import PIIAnalyzer
from models import ModelConfig
from constants import ENGLISH_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PII Detection Service",
    description="API for detecting and analyzing Personally Identifiable Information (PII) in text",
    version="1.0.0"
)

# Initialize PII components
detector = PIIDetector()
analyzer = PIIAnalyzer()

class PIIRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

class BatchRequest(BaseModel):
    """Request model for batch text analysis."""
    texts: List[str]
    language: Optional[str] = "en"

class PIIEntity(BaseModel):
    type: str
    value: str
    start: int
    end: int

class PIIResponse(BaseModel):
    has_pii: bool
    entities: List[PIIEntity]
    masked_text: str
    language: str
    metadata: Dict[str, Any]

@app.post("/detect", response_model=PIIResponse)
async def detect_pii(request: PIIRequest):
    """Detect PII in the given text."""
    try:
        logger.debug(f"Received request with text: {request.text}")
        # Update the global detector's configuration
        detector.config = ModelConfig(
            model_name=ENGLISH_MODEL_NAME,
            language=request.language,
            confidence_threshold=0.7
        )
        # Call detect_pii with just the text parameter
        detection_result = detector.detect_pii(request.text)
        
        # Create metadata
        metadata = {
            "source": "PII Detector v1",
            "processing_time": detection_result.get("metadata", {}).get("processing_time", 0),
            "entity_counts": detection_result.get("metadata", {}).get("entity_counts", {}),
            "total_entities": len(detection_result.get("entities", []))
        }
        
        return PIIResponse(
            has_pii=detection_result["has_pii"],
            entities=detection_result["entities"],
            masked_text=detection_result["masked_text"],
            language=request.language,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_pii(request: PIIRequest):
    """Analyze PII in the given text."""
    try:
        detection_result = detector.detect_pii(request.text, request.language)
        analysis_result = analyzer.analyze(detection_result)
        return analysis_result
    except Exception as e:
        logger.error(f"Error analyzing PII: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_detect_pii(request: BatchRequest):
    """Process multiple texts in parallel."""
    try:
        result = detector.batch_detect_pii(request.texts, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 