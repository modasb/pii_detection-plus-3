from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pii_detector import MultilingualPIIDetector

app = FastAPI(
    title="PII Detection Service",
    description="API for detecting and anonymizing PII in English and French text",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str
    language: str = "en"

class BatchTextRequest(BaseModel):
    texts: List[str]
    language: str = "en"

# Initialize the PII detector
pii_detector = MultilingualPIIDetector()

@app.post("/detect_and_anonymize/")
async def detect_and_anonymize(request: TextRequest):
    """
    Detect and anonymize PII in a single text
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if request.language not in ['en', 'fr']:
        raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
    
    result = pii_detector.detect_and_anonymize(request.text, request.language)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])
    
    return result

@app.post("/batch_detect_and_anonymize/")
async def batch_detect_and_anonymize(request: BatchTextRequest):
    """
    Detect and anonymize PII in multiple texts
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if request.language not in ['en', 'fr']:
        raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
    
    results = pii_detector.batch_detect_and_anonymize(request.texts, request.language)
    return {"results": results}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 