from django.conf import settings
from .pii import PIIProtectionLayer
import json
from typing import Dict, List
import re

class PIIProtectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.pii_layer = PIIProtectionLayer()
        # Initialize language-specific PII detectors
        self.supported_languages = ['en', 'fr', 'ar']  # English, French, Arabic

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and formatting
        """
        # Remove markdown formatting
        text = re.sub(r'[*#_\[\]]', '', text)
        
        # Remove multiple spaces and newlines
        text = ' '.join(text.split())
        
        # Truncate if too long (5000 char limit for translation)
        if len(text) > 4900:
            text = text[:4900] + "..."
            
        return text

    def chunk_text(self, text: str, max_length: int = 4900) -> List[str]:
        """
        Split text into smaller chunks for translation
        """
        if len(text) <= max_length:
            return [text]
            
        chunks = []
        sentences = text.split('. ')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 2  # +2 for '. '
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks

    def process_text(self, text: str, language: str) -> Dict:
        """
        Process text for PII detection in specific language
        Returns dict with detected PII and redacted text
        """
        try:
            # Clean and prepare text
            clean_text = self.clean_text(text)
            text_chunks = self.chunk_text(clean_text)
            
            # Process each chunk
            all_detections = []
            redacted_chunks = []
            
            for chunk in text_chunks:
                # Detect and redact PII
                chunk_detections = self.pii_layer.detect_pii(chunk, language)
                redacted_chunk = self.pii_layer.redact_pii(chunk, chunk_detections)
                
                all_detections.extend(chunk_detections)
                redacted_chunks.append(redacted_chunk)
            
            return {
                'original_text': text,
                'redacted_text': ' '.join(redacted_chunks),
                'detected_pii': all_detections,
                'language': language,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language
            }

    def __call__(self, request):
        # Process request body if it contains text data
        if request.method in ['POST', 'PUT'] and request.content_type == 'application/json':
            try:
                body_data = json.loads(request.body)
                if 'text' in body_data and 'language' in body_data:
                    if body_data['language'] in self.supported_languages:
                        results = self.process_text(
                            body_data['text'], 
                            body_data['language']
                        )
                        # Attach results to request for downstream processing
                        request.pii_results = results
            except json.JSONDecodeError:
                pass

        response = self.get_response(request)
        return response 