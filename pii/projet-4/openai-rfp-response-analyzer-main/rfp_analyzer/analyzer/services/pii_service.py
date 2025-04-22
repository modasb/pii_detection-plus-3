import httpx
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class PIIServiceClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def detect_pii(self, content: str, language: str = "en") -> dict:
        """
        Detect PII in the given content using the PII service API.
        
        Args:
            content: The text content to analyze
            language: Language code (default: "en")
            
        Returns:
            Dict containing PII detection results
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/detect",
                json={"text": content}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error calling PII service: {e}")
            return {
                'has_pii': False,
                'pii_count': 0,
                'entities': [],
                'anonymized_text': content,
                'risk_level': 'UNKNOWN',
                'recommendations': ['PII detection service unavailable']
            } 