from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from api.v1 import endpoints

# Create FastAPI app
app = FastAPI(
    title="PII Detection API",
    description="""
    API for detecting and anonymizing Personally Identifiable Information (PII) in text.
    
    Supported PII types:
    - Credit Card Numbers
    - Email Addresses
    - Phone Numbers
    - Social Security Numbers (SSN)
    - IP Addresses
    
    Features:
    - Multi-language support
    - Context-aware detection
    - Confidence scoring
    - Text anonymization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    endpoints.router,
    prefix="/api/v1",
    tags=["PII Detection"]
)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - redirects to API documentation.
    """
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 