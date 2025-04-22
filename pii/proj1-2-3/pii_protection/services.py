import httpx
from django.conf import settings
from typing import Dict, Any, List
from .exceptions import APIGatewayError
import logging
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class AIService:
    """Base class for AI services"""
    def __init__(self, endpoint: str, api_key: str = None):
        self.endpoint = endpoint
        self.api_key = api_key
        
    async def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """Process text using the AI service"""
        raise NotImplementedError

class SummarizationService(AIService):
    async def process(self, text: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json={'text': text, **kwargs},
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            return response.json()

class TranslationService(AIService):
    async def process(self, text: str, target_language: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json={'text': text, 'target_language': target_language, **kwargs},
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            return response.json()

class NERService(AIService):
    async def process(self, text: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json={'text': text, **kwargs},
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            return response.json()

class APIGateway:
    """
    API Gateway for routing requests to various AI services
    """
    def __init__(self):
        self.services = self._initialize_services()
        self.timeout = settings.API_GATEWAY.get('TIMEOUT', 30)
        self.retry_attempts = settings.API_GATEWAY.get('RETRY_ATTEMPTS', 3)
        
    @lru_cache()
    def _initialize_services(self) -> Dict[str, AIService]:
        """Initialize available AI services"""
        services = {}
        service_configs = settings.API_GATEWAY.get('SERVICES', {})
        
        service_classes = {
            'summarization': SummarizationService,
            'translation': TranslationService,
            'ner': NERService,
        }
        
        for service_name, config in service_configs.items():
            if service_name in service_classes:
                services[service_name] = service_classes[service_name](
                    endpoint=config['endpoint'],
                    api_key=config.get('api_key')
                )
                
        return services
        
    async def route_request(self, service: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the appropriate AI service"""
        if service not in self.services:
            raise APIGatewayError(f"Unknown service: {service}")
            
        service_instance = self.services[service]
        
        for attempt in range(self.retry_attempts):
            try:
                return await service_instance.process(**payload)
            except httpx.HTTPError as e:
                logger.error(f"HTTP error processing request: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise APIGatewayError(f"Failed to process request: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                raise APIGatewayError(f"Internal server error: {str(e)}")
                
    async def batch_route_requests(
        self, 
        service: str, 
        payloads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Route multiple requests to an AI service"""
        if service not in self.services:
            raise APIGatewayError(f"Unknown service: {service}")
            
        tasks = [
            self.route_request(service, payload)
            for payload in payloads
        ]
        
        try:
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise APIGatewayError(f"Batch processing failed: {str(e)}")
            
    def get_available_services(self) -> List[str]:
        """Get list of available AI services"""
        return list(self.services.keys()) 