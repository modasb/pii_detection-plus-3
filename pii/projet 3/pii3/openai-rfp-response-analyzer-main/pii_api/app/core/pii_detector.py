import re
import json
from typing import List, Dict, Optional
from pathlib import Path
import logging
from langdetect import detect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PIIDetector:
    """PII Detection class with configurable patterns and rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.patterns = self.config.get('patterns', {})
        self.context_words = self.config.get('context_words', {})
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'patterns': {
                'CREDIT_CARD': [
                    # Visa
                    r'\b4[0-9]{12}(?:[0-9]{3})?\b',
                    # Mastercard
                    r'\b5[1-5][0-9]{14}\b',
                    # American Express
                    r'\b3[47][0-9]{13}\b',
                    # Discover
                    r'\b6(?:011|5[0-9]{2})[0-9]{12}\b',
                    # Any card with dashes or spaces
                    r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
                ],
                'EMAIL': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                'PHONE_NUMBER': [
                    r'\b(?:\+?1[-.]?)?\s*\(?([0-9]{3})\)?[-.\s]*([0-9]{3})[-.\s]*([0-9]{4})\b'
                ],
                'SSN': [
                    r'\b[0-9]{3}[-]?[0-9]{2}[-]?[0-9]{4}\b'
                ],
                'IP_ADDRESS': [
                    r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
                ]
            },
            'context_words': {
                'CREDIT_CARD': ['card', 'credit', 'visa', 'mastercard', 'amex', 'payment', 'cc', 'credit card', 'card number'],
                'EMAIL': ['email', 'e-mail', 'mail', 'contact', '@'],
                'PHONE_NUMBER': ['phone', 'tel', 'telephone', 'mobile', 'cell', 'call'],
                'SSN': ['ssn', 'social security', 'social insurance', 'tax'],
                'IP_ADDRESS': ['ip', 'address', 'server', 'host']
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            return 'en'
    
    def _check_context(self, text: str, entity_type: str, start: int, end: int, window: int = 100) -> float:
        """Check surrounding context for relevant words."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()
        
        context_words = self.context_words.get(entity_type, [])
        matches = sum(1 for word in context_words if word.lower() in context)
        
        # Increase base confidence if context words are found
        return min(matches * 0.1, 0.5)
    
    def _luhn_check(self, card_number: str) -> bool:
        """
        Implement Luhn algorithm for credit card validation.
        """
        digits = [int(d) for d in card_number if d.isdigit()]
        if not digits:
            return False
            
        # Starting from the rightmost digit and moving left
        for i in range(len(digits)-2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
                
        return sum(digits) % 10 == 0
    
    def _normalize_match(self, match_text: str, entity_type: str) -> str:
        """Normalize matched text based on entity type."""
        if entity_type == 'PHONE_NUMBER':
            # Remove any non-digit characters for phone numbers
            digits = ''.join(c for c in match_text if c.isdigit())
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif entity_type == 'CREDIT_CARD':
            # Remove any non-digit characters
            digits = ''.join(c for c in match_text if c.isdigit())
            if 13 <= len(digits) <= 19 and self._luhn_check(digits):
                # Format as XXXX-XXXX-XXXX-XXXX
                parts = [digits[i:i+4] for i in range(0, len(digits), 4)]
                return '-'.join(parts)
        return match_text
    
    def detect_pii(self, text: str, language: Optional[str] = None) -> List[Dict]:
        """Detect PII entities in the given text."""
        if not language:
            language = self.detect_language(text)
            
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            if isinstance(patterns, dict):
                patterns = patterns.get(language.upper(), patterns.get('EN', []))
            
            if not isinstance(patterns, list):
                patterns = [patterns]
                
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start, end = match.span()
                    matched_text = text[start:end]
                    
                    # Base confidence score
                    confidence = 0.8
                    
                    # Add context-based confidence boost
                    context_boost = self._check_context(text, entity_type, start, end)
                    confidence = min(confidence + context_boost, 1.0)
                    
                    # Additional validation for credit cards
                    if entity_type == 'CREDIT_CARD':
                        digits = ''.join(c for c in matched_text if c.isdigit())
                        if not (13 <= len(digits) <= 19) or not self._luhn_check(digits):
                            continue
                        confidence = min(confidence + 0.1, 1.0)  # Boost confidence if Luhn check passes
                    
                    # Normalize the matched text
                    normalized_text = self._normalize_match(matched_text, entity_type)
                    
                    entities.append({
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'value': normalized_text,
                        'confidence': confidence
                    })
        
        return sorted(entities, key=lambda x: x['start'])
    
    def anonymize_text(self, text: str, entities: List[Dict]) -> str:
        """Replace detected PII with anonymized placeholders."""
        if not entities:
            return text
            
        result = list(text)
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            placeholder = f" [{entity['type']}]"  # Added space before placeholder
            start, end = entity['start'], entity['end']
            # If there's already a space before the entity, don't add another
            if start > 0 and result[start - 1].isspace():
                placeholder = placeholder[1:]
            result[start:end] = placeholder
            
        return ''.join(result)
    
    def detect_and_anonymize(self, text: str, language: Optional[str] = None) -> Dict:
        """Detect PII and return both entities and anonymized text."""
        entities = self.detect_pii(text, language)
        anonymized_text = self.anonymize_text(text, entities)
        
        return {
            'entities': entities,
            'anonymized_text': anonymized_text
        } 