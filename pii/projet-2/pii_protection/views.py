from django.shortcuts import render
from rest_framework import viewsets
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

logger = logging.getLogger(__name__)

# Create your views here.

class AIRequestViewSet(viewsets.ViewSet):
    """
    API endpoint that handles requests to external AI services with PII protection.
    """
    pii_layer = PIIProtectionLayer()
    api_gateway = APIGateway()
    
    def create(self, request):
        serializer = RequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        input_text = serializer.validated_data['text']
        service = serializer.validated_data['service']
        
        logger.info(f"Processing text: {input_text}")
        
        try:
            # Use Presidio to detect and redact PII
            redacted_text, detected_entities = self.pii_layer.detect_and_redact(input_text)
            
            logger.info(f"Detected entities: {detected_entities}")
            logger.info(f"Redacted text: {redacted_text}")
            
            # Log what was found
            if request.user.is_authenticated and detected_entities:
                PIIAuditLog.objects.create(
                    user=request.user,
                    original_text_hash=hash(input_text),
                    num_entities_detected=len(detected_entities),
                    entity_types=[e['entity_type'] for e in detected_entities]
                )
            
            return Response({
                'processed_text': redacted_text,
                'ai_response': {'message': 'AI service response would go here'},
                'pii_detected': bool(detected_entities),
                'detected_entities': detected_entities  # Add this for debugging
            })
            
        except (PIIDetectionError, APIGatewayError) as e:
            logger.error(f"Request processing failed: {str(e)}")
            return Response(
                {'error': str(e)},
                status=500
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
