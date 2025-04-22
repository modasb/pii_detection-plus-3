from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings
from .models import PIIAuditLog
from .serializers import RequestSerializer
from .exceptions import PIIDetectionError, APIGatewayError
from .services import APIGateway
from .pii import PIIProtectionLayer
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import asyncio

logger = logging.getLogger(__name__)

# Create your views here.

class AIServiceViewSet(viewsets.ViewSet):
    """
    API endpoint that handles requests to AI services with PII protection
    """
    pii_layer = PIIProtectionLayer()
    api_gateway = APIGateway()
    
    @action(detail=False, methods=['POST'])
    async def process(self, request):
        """
        Process text through PII protection and AI service
        """
        try:
            # Validate request data
            service = request.data.get('service')
            text = request.data.get('text')
            options = request.data.get('options', {})
            
            if not service or not text:
                return Response(
                    {'error': 'Service and text are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            # Process PII
            pii_result = self.pii_layer.analyze_text(text)
            redacted_text = pii_result['redacted_text']
            
            # Log PII detection if entities found
            if pii_result['detected_entities'] and request.user.is_authenticated:
                PIIAuditLog.objects.create(
                    user=request.user,
                    service=service,
                    original_text_hash=hash(text),
                    num_entities_detected=len(pii_result['detected_entities']),
                    entity_types=[e['type'] for e in pii_result['detected_entities']]
                )
                
            # Process through AI service
            payload = {'text': redacted_text, **options}
            service_response = await self.api_gateway.route_request(service, payload)
            
            return Response({
                'service': service,
                'pii_detected': bool(pii_result['detected_entities']),
                'service_response': service_response
            })
            
        except APIGatewayError as e:
            logger.error(f"API Gateway error: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_502_BAD_GATEWAY
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    @action(detail=False, methods=['POST'])
    async def batch_process(self, request):
        """
        Process multiple texts through PII protection and AI service
        """
        try:
            service = request.data.get('service')
            texts = request.data.get('texts', [])
            options = request.data.get('options', {})
            
            if not service or not texts:
                return Response(
                    {'error': 'Service and texts are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            # Process PII for all texts
            processed_texts = []
            for text in texts:
                pii_result = self.pii_layer.analyze_text(text)
                processed_texts.append({
                    'redacted_text': pii_result['redacted_text'],
                    'pii_detected': bool(pii_result['detected_entities'])
                })
                
            # Prepare payloads for AI service
            payloads = [
                {'text': item['redacted_text'], **options}
                for item in processed_texts
            ]
            
            # Process through AI service
            service_responses = await self.api_gateway.batch_route_requests(service, payloads)
            
            # Combine results
            results = []
            for i, response in enumerate(service_responses):
                results.append({
                    'pii_detected': processed_texts[i]['pii_detected'],
                    'service_response': response
                })
                
            return Response({
                'service': service,
                'results': results
            })
            
        except APIGatewayError as e:
            logger.error(f"API Gateway error in batch processing: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_502_BAD_GATEWAY
            )
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    @action(detail=False, methods=['GET'])
    def available_services(self, request):
        """
        Get list of available AI services
        """
        try:
            services = self.api_gateway.get_available_services()
            return Response({'services': services})
        except Exception as e:
            logger.error(f"Error getting available services: {str(e)}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

def home(request):
    return JsonResponse({
        'status': 'ok',
        'message': 'PII Protection API is running',
        'endpoints': {
            'pii_detection': '/api/detect/'
        }
    })

# Initialize the PII protection layer
pii_layer = PIIProtectionLayer()

@csrf_exempt
@require_http_methods(["POST"])
def detect_pii(request):
    """API endpoint to detect and redact PII in text."""
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
            
        # Process the text
        result = pii_layer.analyze_text(text, language)
        
        return JsonResponse({
            'original_text': result['original_text'],
            'redacted_text': result['redacted_text'],
            'entity_count': len(result['detected_entities']),
            'entities': [
                {
                    'type': entity['type'],
                    'text': entity['text'],
                    'method': entity['method']
                }
                for entity in result['detected_entities']
            ]
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def batch_process(request):
    """API endpoint to process multiple texts at once."""
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        language = data.get('language', 'en')
        
        if not texts:
            return JsonResponse({'error': 'No texts provided'}, status=400)
            
        # Process each text
        results = []
        for text in texts:
            result = pii_layer.analyze_text(text, language)
            results.append({
                'original_text': result['original_text'],
                'redacted_text': result['redacted_text'],
                'entity_count': len(result['detected_entities'])
            })
        
        return JsonResponse({'results': results})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
