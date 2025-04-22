import logging
import re
import json
import arabic_reshaper
from bidi.algorithm import get_display
from typing import List, Dict, Any, Optional
from functools import lru_cache
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIProtectionLayerLight:
    """
    Memory-optimized PII protection system for document processing in French, Arabic, and English.
    This version uses less memory by avoiding loading large models.
    """
    
    REGEX_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_number': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'date': r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
        # Added document-specific patterns
        'passport': r'\b[A-Z]{1,2}[0-9]{6,9}\b',
        'social_security': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',
        'swift_bic': r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',
    }
    
    LANGUAGE_REGEX_PATTERNS = {
        "fr": {
            "fr_phone": r'\b(?:\+33|0)\s?[1-9](?:\s?\d{2}){4}\b',
            "fr_postcode": r'\b\d{5}\b',
            "fr_national_id": r'\b\d{13}\b',
            "fr_siret": r'\b\d{3}\s?\d{3}\s?\d{3}\s?\d{5}\b',
            "fr_siren": r'\b\d{3}\s?\d{3}\s?\d{3}\b',
            "fr_tva": r'\bFR\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b',
        },
        "ar": {
            # Basic Arabic patterns
            "ar_phone": r'\b(?:\+\d{1,3}|0)\d{9,10}\b',
            "ar_date": r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
            "ar_national_id": r'\b\d{14}\b',
            "ar_passport": r'\b[A-Z]\s?\d{7}\b',
            
            # Enhanced country-specific patterns
            "ar_phone_sa": r'\b(?:\+966|00966|0)5[0-9]{8}\b',  # Saudi Arabia
            "ar_phone_ae": r'\b(?:\+971|00971|0)5[0-9]{8}\b',  # UAE
            "ar_phone_eg": r'\b(?:\+20|0020|0)1[0-9]{9}\b',    # Egypt
            "ar_phone_qa": r'\b(?:\+974|00974|0)(?:3|5|6|7)[0-9]{7}\b',  # Qatar
            
            # Enhanced ID patterns
            "ar_national_id_sa": r'\b[12]\d{9}\b',  # Saudi Arabia (10 digits)
            "ar_national_id_eg": r'\b[123]\d{13}\b',  # Egypt (14 digits)
            "ar_national_id_ae": r'\b7\d{13}\b',  # UAE (15 digits starting with 7)
            
            # Enhanced passport patterns
            "ar_passport_sa": r'\b[A-Z]\s?\d{7,9}\b',  # Saudi Arabia
            "ar_passport_eg": r'\b[A-Z]\s?\d{7,8}\b',  # Egypt
            
            # Arabic names and addresses
            "ar_name": r'(?:[\u0600-\u06FF]+\s+){1,4}[\u0600-\u06FF]+',  # Up to 5 Arabic name parts
            "ar_address": r'(?:شارع|طريق|حي|منطقة|مدينة|قرية)\s+[\u0600-\u06FF\s،,0-9]+',
            
            # Arabic email domains
            "ar_email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:sa|eg|ae|qa|kw|bh|om|ma|dz|tn|lb|jo|ps|iq|sy)\b',
        },
        "en": {
            "uk_nino": r'\b[A-Z]{2}\d{6}[A-Z]\b',
            "uk_passport": r'\b[0-9]{9}\b',
            "us_ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }
    }

    MASKING_STRATEGIES = {
        'default': lambda x: '[REDACTED]',
        'email': lambda x: x.split('@')[0][:2] + '***@' + x.split('@')[1],
        'phone_number': lambda x: x[:4] + '*' * (len(x) - 4),
        'date': lambda x: '[DATE]',
        'credit_card': lambda x: x[:4] + '*' * 8 + x[-4:],
        'passport': lambda x: '[PASSPORT]',
        'person': lambda x: '[PERSON]',
        'location': lambda x: '[LOCATION]',
        'organization': lambda x: '[ORGANIZATION]'
    }

    def __init__(self, redaction_strategy: str = 'mask', confidence_threshold: Optional[Dict[str, float]] = None):
        """Initialize memory-optimized PII detection system for document processing."""
        logger.info("Initializing Memory-Optimized PII Protection Layer...")
        
        self.confidence_thresholds = confidence_threshold or {'en': 0.6, 'fr': 0.4, 'ar': 0.7}
        self.redaction_strategy = redaction_strategy
        
        # Entity mapping for consistency
        self.entity_mapping = {
            # Common mappings
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "EMAIL": "EMAIL",
            "PHONE_NUMBER": "PHONE_NUMBER",
            "PASSPORT": "PASSPORT",
            
            # French-specific
            "fr_phone": "PHONE_NUMBER",
            "fr_postcode": "POSTCODE",
            "fr_siret": "SIRET",
            "fr_siren": "SIREN",
            "fr_tva": "VAT",
            
            # Arabic-specific
            "ar_phone": "PHONE_NUMBER",
            "ar_national_id": "NATIONAL_ID",
            "ar_passport": "PASSPORT",
            "ar_phone_sa": "PHONE_NUMBER_SA",
            "ar_phone_ae": "PHONE_NUMBER_AE",
            "ar_phone_eg": "PHONE_NUMBER_EG",
            "ar_phone_qa": "PHONE_NUMBER_QA",
            "ar_national_id_sa": "NATIONAL_ID_SA",
            "ar_national_id_eg": "NATIONAL_ID_EG",
            "ar_national_id_ae": "NATIONAL_ID_AE",
            "ar_passport_sa": "PASSPORT_SA",
            "ar_passport_eg": "PASSPORT_EG",
            "ar_name": "ARABIC_NAME",
            "ar_address": "ARABIC_ADDRESS",
            "ar_email": "ARABIC_EMAIL",
        }

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to a consistent format."""
        # Convert to uppercase for consistency
        entity_type = entity_type.upper()
        # Use mapping if available, otherwise keep original
        return self.entity_mapping.get(entity_type, entity_type)

    @lru_cache(maxsize=100)
    def detect_pii(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Detect PII entities using regex patterns."""
        detected_entities = []
        
        # 1. Apply common regex patterns
        for pattern_name, pattern in self.REGEX_PATTERNS.items():
            for match in re.finditer(pattern, text):
                detected_entities.append({
                    'type': self._normalize_entity_type(pattern_name),
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 1.0,
                    'method': 'regex'
                })

        # 2. Apply language-specific regex patterns
        if language in self.LANGUAGE_REGEX_PATTERNS:
            for pattern_name, pattern in self.LANGUAGE_REGEX_PATTERNS[language].items():
                for match in re.finditer(pattern, text):
                    detected_entities.append({
                        'type': self._normalize_entity_type(pattern_name),
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'score': 1.0,
                        'method': f'regex_{language}'
                    })
        
        # 3. Add enhanced Arabic detection if needed
        if language == 'ar':
            detected_entities.extend(self._enhance_arabic_detection(text))

        # Remove overlapping entities, keeping those with higher confidence
        return self._remove_overlapping_entities(detected_entities)

    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping those with higher confidence scores."""
        if not entities:
            return []

        # Sort by start position and then by score (descending)
        sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['score']))
        filtered_entities = []
        last_end = -1

        for entity in sorted_entities:
            if entity['start'] >= last_end:
                filtered_entities.append(entity)
                last_end = entity['end']

        return filtered_entities

    def mask_pii(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Mask detected PII entities using appropriate strategies."""
        if not entities:
            return text

        # Sort entities by start position in reverse order
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        masked_text = text
        for entity in entities:
            strategy = self.MASKING_STRATEGIES.get(
                entity['type'].lower(),
                self.MASKING_STRATEGIES['default']
            )
            masked_text = (
                masked_text[:entity['start']] +
                strategy(entity['text']) +
                masked_text[entity['end']:]
            )
        
        return masked_text

    def process_document(self, text: str, language: str) -> Dict[str, Any]:
        """Process document with enhanced Arabic support."""
        try:
            original_text = text
            
            # Handle Arabic text if needed
            if language == 'ar':
                # Store original text for display but use reshaped text for processing
                try:
                    reshaped_text = arabic_reshaper.reshape(text)
                    display_text = get_display(reshaped_text)
                    # Process both the original and the reshaped text for better coverage
                    detected_entities_original = self.detect_pii(text, language)
                    detected_entities_reshaped = self.detect_pii(display_text, language)
                    
                    # Merge the results, adjusting positions for reshaped text
                    detected_entities = detected_entities_original
                    for entity in detected_entities_reshaped:
                        # Only add if we don't have a similar entity already
                        if not any(e['text'] == entity['text'] for e in detected_entities):
                            detected_entities.append(entity)
                except Exception as e:
                    logger.warning(f"Arabic text reshaping failed: {str(e)}")
                    detected_entities = self.detect_pii(text, language)
            else:
                # For non-Arabic languages, proceed normally
                detected_entities = self.detect_pii(text, language)
            
            # Mask PII
            masked_text = self.mask_pii(text, detected_entities)
            
            # Group entities by type and method
            entity_stats = {}
            entity_types = set()
            entity_methods = set()
            
            for entity in detected_entities:
                entity_type = entity['type']
                method = entity['method']
                
                # Track unique entity types and methods
                entity_types.add(entity_type)
                entity_methods.add(method)
                
                # Create combined key for statistics
                key = f"{entity_type}_{method}"
                if key not in entity_stats:
                    entity_stats[key] = 0
                entity_stats[key] += 1
            
            return {
                'original_text': original_text,
                'masked_text': masked_text,
                'detected_entities': detected_entities,
                'statistics': {
                    'total_entities': len(detected_entities),
                    'entity_types': entity_stats,
                    'unique_types': list(entity_types),
                    'detection_methods': list(entity_methods)
                },
                'language': language,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'original_text': text,
                'masked_text': text,
                'detected_entities': [],
                'statistics': {
                    'total_entities': 0,
                    'entity_types': {},
                    'unique_types': [],
                    'detection_methods': []
                },
                'language': language,
                'success': False,
                'error': str(e)
            }

    def batch_process(self, texts: List[str], language: str) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        return [self.process_document(text, language) for text in texts]

    def analyze_text(self, text: str, language: str, redaction_strategy: str = None) -> Dict[str, Any]:
        """Alias for process_document with configurable redaction strategy."""
        old_strategy = self.redaction_strategy
        if redaction_strategy:
            self.redaction_strategy = redaction_strategy
        
        result = self.process_document(text, language)
        
        # Restore original strategy
        if redaction_strategy:
            self.redaction_strategy = old_strategy
            
        return result
    
    def batch_analyze(self, texts: List[str], language: str) -> List[Dict[str, Any]]:
        """Alias for batch_process."""
        return self.batch_process(texts, language)

    # Enhanced Arabic text handling
    ARABIC_REGEX_PATTERNS = {
        # Enhanced Arabic phone patterns covering Saudi, UAE, Egypt, etc.
        "ar_phone_sa": r'\b(?:\+966|00966|0)5[0-9]{8}\b',  # Saudi Arabia
        "ar_phone_ae": r'\b(?:\+971|00971|0)5[0-9]{8}\b',  # UAE
        "ar_phone_eg": r'\b(?:\+20|0020|0)1[0-9]{9}\b',    # Egypt
        "ar_phone_qa": r'\b(?:\+974|00974|0)(?:3|5|6|7)[0-9]{7}\b',  # Qatar
        
        # Enhanced Arabic national ID patterns
        "ar_national_id_sa": r'\b[12]\d{9}\b',  # Saudi Arabia (10 digits)
        "ar_national_id_eg": r'\b[123]\d{13}\b',  # Egypt (14 digits)
        "ar_national_id_ae": r'\b7\d{13}\b',  # UAE (15 digits starting with 7)
        
        # Enhanced Arabic passport patterns
        "ar_passport_sa": r'\b[A-Z]\s?\d{7,9}\b',  # Saudi Arabia
        "ar_passport_eg": r'\b[A-Z]\s?\d{7,8}\b',  # Egypt
        
        # Arabic names pattern (more comprehensive)
        "ar_name": r'(?:[\u0600-\u06FF]+\s+){1,4}[\u0600-\u06FF]+',  # Up to 5 Arabic name parts
        
        # Arabic addresses and locations
        "ar_address": r'(?:شارع|طريق|حي|منطقة|مدينة|قرية)\s+[\u0600-\u06FF\s،,0-9]+',
        
        # Arabic email domains
        "ar_email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:sa|eg|ae|qa|kw|bh|om|ma|dz|tn|lb|jo|ps|iq|sy)\b',
    }
    
    def _enhance_arabic_detection(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced detection specifically for Arabic text."""
        detected_entities = []
        
        # Apply enhanced Arabic regex patterns
        for pattern_name, pattern in self.ARABIC_REGEX_PATTERNS.items():
            for match in re.finditer(pattern, text):
                detected_entities.append({
                    'type': self._normalize_entity_type(pattern_name),
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.9,  # High confidence for specific patterns
                    'method': 'enhanced_arabic_regex'
                })
        
        return detected_entities 