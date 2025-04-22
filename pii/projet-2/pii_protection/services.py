import httpx
from django.conf import settings
from typing import Dict, Any
from .exceptions import APIGatewayError

class APIGateway:
    """
    API Gateway for routing requests to external services
    """
    def __init__(self):
        self.endpoints = settings.API_GATEWAY['ENDPOINTS']
        self.timeout = settings.API_GATEWAY['TIMEOUT']
        self.retry_attempts = settings.API_GATEWAY['RETRY_ATTEMPTS']

    async def route_request(self, service: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a request to the appropriate external API
        """
        if service not in self.endpoints:
            raise APIGatewayError(f"Unknown service: {service}")

        async with httpx.AsyncClient() as client:
            for attempt in range(self.retry_attempts):
                try:
                    response = await client.post(
                        self.endpoints[service],
                        json=payload,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as e:
                    if attempt == self.retry_attempts - 1:
                        raise APIGatewayError(f"Failed to process request: {str(e)}") 