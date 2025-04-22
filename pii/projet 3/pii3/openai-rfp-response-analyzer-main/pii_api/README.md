# PII Detection API

A FastAPI-based REST API for detecting and anonymizing Personally Identifiable Information (PII) in text.

## Features

- Detect multiple types of PII:
  - Credit Card Numbers
  - Email Addresses
  - Phone Numbers
  - Social Security Numbers
  - IP Addresses
- Language detection and support for multiple languages
- Context-aware detection with confidence scores
- Text anonymization
- Easy to integrate with any application

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pii_api
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative documentation: `http://localhost:8000/redoc`

### Endpoints

#### POST /api/v1/detect
Detect PII in text and return both detected entities and anonymized text.

Request body:
```json
{
    "text": "My email is john@example.com and my phone is 123-456-7890",
    "language": "en"  // optional
}
```

Response:
```json
{
    "entities": [
        {
            "type": "EMAIL",
            "start": 12,
            "end": 27,
            "value": "john@example.com",
            "confidence": 0.9
        },
        {
            "type": "PHONE_NUMBER",
            "start": 42,
            "end": 54,
            "value": "123-456-7890",
            "confidence": 0.85
        }
    ],
    "anonymized_text": "My email is [EMAIL] and my phone is [PHONE_NUMBER]"
}
```

#### GET /api/v1/health
Health check endpoint.

Response:
```json
{
    "status": "healthy"
}
```

## Configuration

The PII detector can be configured by providing a JSON configuration file with custom patterns and context words. See the example in `app/core/pii_detector.py`.

## Integration

To integrate with your application, simply make HTTP requests to the API endpoints. Example using Python requests:

```python
import requests

api_url = "http://localhost:8000/api/v1/detect"
response = requests.post(api_url, json={
    "text": "My credit card is 4532-0158-7845-9623",
    "language": "en"
})

result = response.json()
print(result["anonymized_text"])
```

## License

MIT License 