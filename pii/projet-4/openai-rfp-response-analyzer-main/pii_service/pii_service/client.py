import aiohttp
import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PIIServiceClient:
    """Client for making requests to the PII service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
    
    async def detect_pii(self, text: str) -> Dict:
        """
        Send text to the PII service for detection.
        
        Args:
            text: Text to analyze for PII
            
        Returns:
            Dict containing detection results
        """
        try:
            logger.debug(f"Sending request to {self.base_url}/detect")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/detect",
                    json={"text": text}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"PII service error: {error_text}")
                        raise Exception(f"PII service error: {error_text}")
                    result = await response.json()
                    logger.debug(f"Received response: {result}")
                    return result
        except Exception as e:
            logger.error(f"Error in detect_pii: {str(e)}", exc_info=True)
            raise
    
    async def health_check(self) -> bool:
        """
        Check if the PII service is healthy.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            logger.debug(f"Checking health at {self.base_url}/health")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Health check error: {error_text}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Error in health_check: {str(e)}", exc_info=True)
            return False 