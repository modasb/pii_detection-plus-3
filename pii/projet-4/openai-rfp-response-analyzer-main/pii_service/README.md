# PII Service

A service for detecting and analyzing Personally Identifiable Information (PII) in text.

## Features

- Detects various types of PII:
  - Email addresses
  - Phone numbers
  - Social Security Numbers (SSN)
  - Credit card numbers
  - Bank account numbers
- Provides risk assessment for detected PII
- Offers recommendations for handling sensitive information
- Includes masking functionality to anonymize text

## Installation

```bash
pip install -e .
```

## Usage

### As a Service

Start the service:
```bash
python run.py
```

The service will be available at `http://localhost:8001`

### API Endpoints

- `POST /detect`: Detect PII in text
- `GET /health`: Health check endpoint

### Example Request

```python
import aiohttp
import json

async def detect_pii():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8001/detect",
            json={"text": "My email is john.doe@example.com"}
        ) as response:
            return await response.json()
```

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest
``` 