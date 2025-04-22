from django.conf import settings
from .pii import PIIProtectionLayer
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PIIProcessingMiddleware:
    """
    Middleware to process PII in requests before they reach the views
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.pii_layer = PIIProtectionLayer()
        
    def __call__(self, request):
        if not self._should_process_request(request):
            return self.get_response(request)
            
        try:
            self._process_request_pii(request)
        except Exception as e:
            logger.error(f"PII processing error: {str(e)}")
            
        response = self.get_response(request)
        
        try:
            self._process_response_pii(response)
        except Exception as e:
            logger.error(f"PII processing error in response: {str(e)}")
            
        return response
        
    def _should_process_request(self, request) -> bool:
        """
        Determine if the request should be processed for PII
        """
        # Skip processing for static files, admin pages, etc.
        excluded_paths = ['/static/', '/admin/', '/media/']
        return not any(request.path.startswith(path) for path in excluded_paths)
        
    def _process_request_pii(self, request) -> None:
        """
        Process incoming request data for PII
        """
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                if request.content_type == 'application/json':
                    body = json.loads(request.body)
                    processed_body = self._process_dict_pii(body)
                    request._body = json.dumps(processed_body).encode('utf-8')
                elif request.POST:
                    for key, value in request.POST.items():
                        if isinstance(value, str):
                            request.POST[key] = self._process_text_pii(value)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON body")
                
    def _process_response_pii(self, response) -> None:
        """
        Process outgoing response data for PII
        """
        if hasattr(response, 'content'):
            try:
                content = response.content.decode('utf-8')
                data = json.loads(content)
                processed_data = self._process_dict_pii(data, is_response=True)
                response.content = json.dumps(processed_data).encode('utf-8')
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
                
    def _process_dict_pii(self, data: Dict[str, Any], is_response: bool = False) -> Dict[str, Any]:
        """
        Recursively process dictionary values for PII
        """
        if isinstance(data, dict):
            return {k: self._process_dict_pii(v, is_response) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_dict_pii(item, is_response) for item in data]
        elif isinstance(data, str):
            return self._process_text_pii(data, is_response)
        return data
        
    def _process_text_pii(self, text: str, is_response: bool = False) -> str:
        """
        Process a single text string for PII
        """
        try:
            result = self.pii_layer.analyze_text(text)
            return result['redacted_text'] if is_response else text
        except Exception as e:
            logger.error(f"Error processing text for PII: {str(e)}")
            return text 