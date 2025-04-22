# Multilingual PII Detection Service

A FastAPI-based service for detecting and anonymizing Personally Identifiable Information (PII) in both English and French text.

## Features

- Detects and anonymizes various types of PII:
  - Person names
  - Organizations
  - Locations
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - Social Security numbers
  - URLs
  - IP addresses
- Supports both English and French text
- Provides both single text and batch processing endpoints
- RESTful API interface
- Comprehensive test suite

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy language models:
```bash
python -m spacy download en_core_web_md
python -m spacy download fr_core_news_md
```

## Running the Service

Start the FastAPI server:
```bash
cd api
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Detect and Anonymize Single Text
- **Endpoint**: `/detect_and_anonymize/`
- **Method**: POST
- **Request Body**:
```json
{
    "text": "My name is John Smith",
    "language": "en"  // or "fr" for French
}
```

### 2. Batch Process Multiple Texts
- **Endpoint**: `/batch_detect_and_anonymize/`
- **Method**: POST
- **Request Body**:
```json
{
    "texts": ["Text 1", "Text 2"],
    "language": "en"  // or "fr" for French
}
```

### 3. Health Check
- **Endpoint**: `/health`
- **Method**: GET

## Running Tests

```bash
pytest tests/
```

## Integration with Main Project

To integrate this service with the main RFP analyzer project:

1. Copy the `pii_service` directory to the main project
2. Import and use the `MultilingualPIIDetector` class:
```python
from pii_service.pii_detector import MultilingualPIIDetector

detector = MultilingualPIIDetector()
result = detector.detect_and_anonymize("Your text here", "en")
```

## Error Handling

The service includes comprehensive error handling:
- Invalid language codes
- Empty text
- Processing errors
- Invalid requests

## Future Improvements

- Add more language support
- Enhance French entity recognition
- Add custom entity types
- Implement caching for better performance
- Add authentication and rate limiting 