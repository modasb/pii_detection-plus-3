from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .pii import PIIProtectionLayer, PIIDetectionConfig
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PII Detection API",
    description="API for detecting and redacting personally identifiable information in text",
    version="1.0.0"
)

# Initialize PII Protection Layer
pii_layer = PIIProtectionLayer()

# Define supported languages explicitly since it's not in the PIIProtectionLayer class
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic"
}

# Pydantic models for request/response validation
class PIIRequest(BaseModel):
    text: str
    language: str = "en"
    config: Optional[Dict[str, Any]] = None

class BatchPIIRequest(BaseModel):
    texts: List[str]
    language: str = "en"
    config: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect")
async def detect_pii(request: PIIRequest):
    """
    Detect and redact PII in a single text.
    
    Args:
        request: PIIRequest object containing text and configuration
        
    Returns:
        Dict containing detection results
    """
    try:
        # Create custom config if provided
        config = None
        if request.config:
            config = PIIDetectionConfig(**request.config)
            
        # Process the text
        result = pii_layer.analyze_text(
            text=request.text,
            language=request.language,
            config=config
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-detect")
async def batch_detect_pii(request: BatchPIIRequest):
    """
    Detect and redact PII in multiple texts.
    """
    try:
        # Create custom config if provided
        config = None
        if request.config:
            config = PIIDetectionConfig(**request.config)
            
        # Process the texts
        results = []
        total_pii_found = 0
        total_processing_time = 0
        
        for text in request.texts:
            result = pii_layer.analyze_text(
                text=text,
                language=request.language,
                config=config
            )
            results.append(result)
            total_pii_found += len(result.get("detected_entities", []))
            total_processing_time += result.get("processing_time", 0)
        
        return {
            "results": results,
            "summary": {
                "total_processed": len(request.texts),
                "total_pii_found": total_pii_found,
                "total_processing_time": total_processing_time,
                "average_time_per_text": total_processing_time / len(request.texts) if request.texts else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return {
        "languages": list(SUPPORTED_LANGUAGES.keys()),
        "descriptions": SUPPORTED_LANGUAGES,
        "default": "en"
    }

@app.get("/entity-types")
async def get_entity_types():
    """Get list of detectable entity types."""
    entity_types = {
        "PERSON": "Personal names",
        "EMAIL": "Email addresses",
        "PHONE": "Phone numbers",
        "ADDRESS": "Physical addresses",
        "CREDIT_CARD": "Credit card numbers",
        "SSN": "Social Security numbers",
        "IP_ADDRESS": "IP addresses",
        "URL": "URLs",
        "DATE": "Dates",
        "ORGANIZATION": "Organization names",
        "FINANCIAL": "Financial information",
        "ID": "Identification numbers"
    }
    
    return {
        "entity_types": list(entity_types.keys()),
        "descriptions": entity_types
    }

def start_api():
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api() 