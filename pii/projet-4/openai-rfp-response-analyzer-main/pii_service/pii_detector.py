"""PII detection and analysis module."""

from typing import List, Dict, Optional, Set, Tuple
import re
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from constants import (
    ENTITY_EMAIL, ENTITY_PHONE, ENTITY_SSN, ENTITY_CREDIT_CARD,
    ENTITY_BANK_ACCOUNT, ENTITY_IBAN, ENTITY_SWIFT_BIC, ENTITY_VAT,
    ENTITY_INVOICE, ENTITY_PERSON, ENTITY_LOCATION, ENTITY_ORGANIZATION,
    ENTITY_CRYPTO, ENTITY_PAYMENT_TOKEN, ENTITY_CURRENCY,
    ENTITY_TYPE_MAPPING, CONFIDENCE_THRESHOLDS, CONTEXT_WORDS,
    EMAIL_REGEX_SIMPLE, EMAIL_REGEX_COMPLEX, PHONE_REGEX,
    SSN_REGEX, CREDIT_CARD_REGEX, IBAN_REGEX,
    LANG_EN, LANG_FR, LANG_AR,
    ARABIC_MODEL_NAME, ENGLISH_MODEL_NAME, FRENCH_MODEL_NAME, UNIVERSAL_MODEL_NAME
)
from models import Entity, DetectionResult, EvaluationMetrics, BatchResult, ModelConfig, AnonymizationConfig

logger = logging.getLogger(__name__)

class PIIDetector:
    """Enhanced PII detector with multilingual support."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the PII detector with configuration."""
        self.config = config or ModelConfig(
            model_name=ENGLISH_MODEL_NAME,
            language=LANG_EN,
            confidence_threshold=0.7
        )
        self._initialize_models()
        self._initialize_recognizers()
        
    def _initialize_models(self):
        """Initialize NLP models with fallback handling."""
        try:
            import spacy
            import transformers
            
            # Initialize spaCy models
            try:
                self.nlp_en = spacy.load(ENGLISH_MODEL_NAME)
            except OSError:
                logger.warning(f"Downloading English model: {ENGLISH_MODEL_NAME}")
                spacy.cli.download(ENGLISH_MODEL_NAME)
                self.nlp_en = spacy.load(ENGLISH_MODEL_NAME)
            
            try:
                self.nlp_fr = spacy.load(FRENCH_MODEL_NAME)
            except OSError:
                logger.warning(f"Downloading French model: {FRENCH_MODEL_NAME}")
                spacy.cli.download(FRENCH_MODEL_NAME)
                self.nlp_fr = spacy.load(FRENCH_MODEL_NAME)
            
            # Initialize Arabic model with fallback
            try:
                self.ar_tokenizer = transformers.AutoTokenizer.from_pretrained(ARABIC_MODEL_NAME)
                self.ar_model = transformers.AutoModelForTokenClassification.from_pretrained(ARABIC_MODEL_NAME)
                self.ar_ner = transformers.pipeline("ner", model=self.ar_model, tokenizer=self.ar_tokenizer)
            except Exception as e:
                logger.warning(f"Arabic model not loaded. Skipping Arabic NER: {e}")
                self.ar_ner = None
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _initialize_recognizers(self):
        """Initialize entity recognizers."""
        self.recognizers = {
            ENTITY_EMAIL: self._recognize_email,
            ENTITY_PHONE: self._recognize_phone,
            ENTITY_SSN: self._recognize_ssn,
            ENTITY_CREDIT_CARD: self._recognize_credit_card,
            ENTITY_BANK_ACCOUNT: self._recognize_bank_account,
            ENTITY_IBAN: self._recognize_iban,
            ENTITY_SWIFT_BIC: self._recognize_swift_bic,
            ENTITY_VAT: self._recognize_vat,
            ENTITY_INVOICE: self._recognize_invoice,
            ENTITY_PERSON: self._recognize_person,
            ENTITY_LOCATION: self._recognize_location,
            ENTITY_ORGANIZATION: self._recognize_organization,
            ENTITY_CRYPTO: self._recognize_crypto,
            ENTITY_PAYMENT_TOKEN: self._recognize_payment_token,
            ENTITY_CURRENCY: self._recognize_currency
        }
    
    def detect_pii(self, text: str, language: str = LANG_EN) -> Dict:
        """Detect PII in the given text.
        
        Args:
            text: The text to analyze for PII
            language: The language code (default: 'en')
            
        Returns:
            Dict containing detection results including entities and masked text
        """
        start_time = time.time()
        
        # Normalize text
        text = self._preprocess_text(text)
        
        # Detect entities
        entities = []
        for entity_type, recognizer in self.recognizers.items():
            try:
                results = recognizer(text, language)
                entities.extend(results)
            except Exception as e:
                logger.error(f"Error in {entity_type} recognition: {e}")
        
        # Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        # Anonymize text
        anonymized_text = self._anonymize_text(text, entities)
        
        # Create metadata
        metadata = {
            'processing_time': time.time() - start_time,
            'entity_counts': self._count_entities(entities),
            'total_entities': len(entities),
            'language': language
        }
        
        # Create result
        result = DetectionResult(
            original_text=text,
            anonymized_text=anonymized_text,
            entities=entities,
            language=language,
            metadata=metadata
        )
        
        return {
            "has_pii": len(entities) > 0,
            "entities": [entity.__dict__ for entity in entities],
            "masked_text": anonymized_text,
            "language": language,
            "metadata": metadata
        }
    
    def batch_detect_pii(self, texts: List[str], language: str = LANG_EN, max_workers: int = 4) -> BatchResult:
        """Process multiple texts in parallel."""
        start_time = time.time()
        errors = []
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.detect_pii, text=text, language=language) for text in texts]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append({'error': str(e)})
        
        return BatchResult(
            results=results,
            total_processed=len(texts),
            success_count=len(results),
            error_count=len(errors),
            processing_time=time.time() - start_time,
            errors=errors if errors else None
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove zero-width spaces
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        return text.strip()
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities keeping the ones with higher confidence."""
        if not entities:
            return entities
        
        # Sort by position and confidence
        sorted_entities = sorted(entities, key=lambda x: (x.start, -x.score))
        filtered = []
        current = sorted_entities[0]
        
        for entity in sorted_entities[1:]:
            if entity.start >= current.end:
                filtered.append(current)
                current = entity
            elif entity.score > current.score:
                current = entity
        
        filtered.append(current)
        return filtered

    def _anonymize_text(self, text: str, entities: List[Entity]) -> str:
        """Anonymize detected entities in text."""
        # Sort entities by end position (descending) to handle overlapping
        sorted_entities = sorted(entities, key=lambda x: -x.end)
        
        for entity in sorted_entities:
            # Get the appropriate mask
            mask = self._get_entity_mask(entity)
            # Replace the entity with the mask
            text = text[:entity.start] + mask + text[entity.end:]
        
        return text
    
    def _get_entity_mask(self, entity: Entity) -> str:
        """Get the appropriate mask for an entity type."""
        if entity.type == ENTITY_EMAIL:
            return "[EMAIL]"
        elif entity.type == ENTITY_PHONE:
            return "[PHONE]"
        elif entity.type == ENTITY_SSN:
            return "[SSN]"
        elif entity.type == ENTITY_CREDIT_CARD:
            return "[CREDIT_CARD]"
        elif entity.type == ENTITY_BANK_ACCOUNT:
            return "[BANK_ACCOUNT]"
        elif entity.type == ENTITY_IBAN:
            return "[IBAN]"
        elif entity.type == ENTITY_SWIFT_BIC:
            return "[SWIFT_BIC]"
        elif entity.type == ENTITY_VAT:
            return "[VAT]"
        elif entity.type == ENTITY_INVOICE:
            return "[INVOICE]"
        elif entity.type == ENTITY_PERSON:
            return "[PERSON]"
        elif entity.type == ENTITY_LOCATION:
            return "[LOCATION]"
        elif entity.type == ENTITY_ORGANIZATION:
            return "[ORGANIZATION]"
        elif entity.type == ENTITY_CRYPTO:
            return "[CRYPTO]"
        elif entity.type == ENTITY_PAYMENT_TOKEN:
            return "[PAYMENT_TOKEN]"
        elif entity.type == ENTITY_CURRENCY:
            return "[CURRENCY]"
        else:
            return "[PII]"
    
    def _count_entities(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities by type."""
        counts = {}
        for entity in entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
        return counts
    
    # Entity recognition methods
    def _recognize_email(self, text: str, language: str) -> List[Entity]:
        """Recognize email addresses."""
        entities = []
        # Try simple regex first
        for match in EMAIL_REGEX_SIMPLE.finditer(text):
            if self._validate_email(match.group()):
                entities.append(Entity(
                    type=ENTITY_EMAIL,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.8
                ))
        # Try complex regex if needed
        if not entities:
            for match in EMAIL_REGEX_COMPLEX.finditer(text):
                if self._validate_email(match.group()):
                    entities.append(Entity(
                        type=ENTITY_EMAIL,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        score=0.9
                    ))
        return entities
    
    def _recognize_phone(self, text: str, language: str) -> List[Entity]:
        """Recognize phone numbers."""
        entities = []
        for match in PHONE_REGEX.finditer(text):
            if self._validate_phone(match.group()):
                entities.append(Entity(
                    type=ENTITY_PHONE,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.8
                ))
        return entities
    
    def _recognize_ssn(self, text: str, language: str) -> List[Entity]:
        """Recognize SSNs."""
        entities = []
        for match in SSN_REGEX.finditer(text):
            if self._validate_ssn(match.group()):
                entities.append(Entity(
                    type=ENTITY_SSN,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        return entities
    
    def _recognize_credit_card(self, text: str, language: str) -> List[Entity]:
        """Recognize credit card numbers."""
        entities = []
        for match in CREDIT_CARD_REGEX.finditer(text):
            if self._validate_credit_card(match.group()):
                entities.append(Entity(
                    type=ENTITY_CREDIT_CARD,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        return entities
    
    def _recognize_bank_account(self, text: str, language: str) -> List[Entity]:
        """Recognize bank account numbers."""
        entities = []
        # Use IBAN regex for international accounts
        for match in IBAN_REGEX.finditer(text):
            if self._validate_iban(match.group()):
                entities.append(Entity(
                    type=ENTITY_IBAN,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        return entities
    
    def _recognize_iban(self, text: str, language: str) -> List[Entity]:
        """Recognize IBAN numbers."""
        entities = []
        for match in IBAN_REGEX.finditer(text):
            if self._validate_iban(match.group()):
                entities.append(Entity(
                    type=ENTITY_IBAN,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        return entities
    
    def _recognize_swift_bic(self, text: str, language: str) -> List[Entity]:
        """Recognize SWIFT/BIC codes."""
        entities = []
        # Implementation depends on specific SWIFT/BIC format
        return entities
    
    def _recognize_vat(self, text: str, language: str) -> List[Entity]:
        """Recognize VAT numbers."""
        entities = []
        # Implementation depends on specific VAT format
        return entities
    
    def _recognize_invoice(self, text: str, language: str) -> List[Entity]:
        """Recognize invoice numbers."""
        entities = []
        # Implementation depends on specific invoice format
        return entities
    
    def _recognize_person(self, text: str, language: str) -> List[Entity]:
        """Recognize person names."""
        entities = []
        if language == LANG_EN:
            doc = self.nlp_en(text)
        elif language == LANG_FR:
            doc = self.nlp_fr(text)
        else:
            return entities
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'PER']:
                entities.append(Entity(
                    type=ENTITY_PERSON,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.8
                ))
        return entities
    
    def _recognize_location(self, text: str, language: str) -> List[Entity]:
        """Recognize locations."""
        entities = []
        if language == LANG_EN:
            doc = self.nlp_en(text)
        elif language == LANG_FR:
            doc = self.nlp_fr(text)
        else:
            return entities
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'LOCATION']:
                entities.append(Entity(
                    type=ENTITY_LOCATION,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.8
                ))
        return entities
    
    def _recognize_organization(self, text: str, language: str) -> List[Entity]:
        """Recognize organizations."""
        entities = []
        if language == LANG_EN:
            doc = self.nlp_en(text)
        elif language == LANG_FR:
            doc = self.nlp_fr(text)
        else:
            return entities
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'ORGANIZATION']:
                entities.append(Entity(
                    type=ENTITY_ORGANIZATION,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.8
                ))
        return entities
    
    def _recognize_crypto(self, text: str, language: str) -> List[Entity]:
        """Recognize cryptocurrency addresses."""
        entities = []
        # Implementation depends on specific crypto format
        return entities
    
    def _recognize_payment_token(self, text: str, language: str) -> List[Entity]:
        """Recognize payment tokens."""
        entities = []
        # Implementation depends on specific token format
        return entities
    
    def _recognize_currency(self, text: str, language: str) -> List[Entity]:
        """Recognize currency amounts."""
        entities = []
        # Implementation depends on specific currency format
        return entities
    
    # Validation methods
    def _validate_email(self, email: str) -> bool:
        """Validate email address."""
        # Basic format check
        if not EMAIL_REGEX_SIMPLE.match(email):
            return False
            
        # Check for common invalid patterns
        invalid_patterns = [
            r'example\.com$',
            r'test\.com$',
            r'domain\.com$',
            r'@.*@',
            r'\.{2,}',
            r'^[0-9]+@',
            r'@.*_.*\.'
        ]
        
        if any(re.search(pattern, email, re.I) for pattern in invalid_patterns):
            return False
            
        return True
    
    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number."""
        # Remove non-digits except + at start
        if phone.startswith('+'):
            digits = '+' + re.sub(r'\D', '', phone[1:])
        else:
            digits = re.sub(r'\D', '', phone)
        
        # Basic length check
        if len(digits) < 8 or len(digits) > 15:
            return False
        
        # Check for repeating patterns
        if re.search(r'(\d)\1{4,}', digits):
            return False
            
        # Check for sequential numbers
        if re.search(r'(?:0123|1234|2345|3456|4567|5678|6789|7890){2,}', digits):
            return False
        
        return True

    def _validate_ssn(self, ssn: str) -> bool:
        """Validate SSN."""
        # Basic format check
        if not SSN_REGEX.match(ssn):
            return False
        
        # Additional validation rules
        parts = ssn.split('-')
        if len(parts) != 3:
            return False
        
        area, group, serial = parts
        
        # Area code validation
        if area == '000' or area == '666' or int(area) > 899:
            return False
        
        # Group code validation
        if group == '00':
            return False
        
        # Serial number validation
        if serial == '0000':
            return False
        
        return True
    
    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove spaces and dashes
        number = re.sub(r'[\s-]', '', number)
        
        # Check length
        if len(number) not in [13, 14, 15, 16]:
            return False
        
        # Luhn algorithm
        digits = [int(d) for d in number]
        checksum = 0
        is_even = len(digits) % 2 == 0
        
        for i in range(len(digits) - 1, -1, -1):
            d = digits[i]
            if (len(digits) - i) % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        
        return checksum % 10 == 0
    
    def _validate_iban(self, iban: str) -> bool:
        """Validate IBAN number."""
        # Remove spaces and convert to uppercase
        iban = iban.replace(' ', '').upper()
        
        # Basic format check
        if not IBAN_REGEX.match(iban):
            return False
        
        # Country code validation
        country_code = iban[:2]
        if country_code not in self._get_valid_country_codes():
            return False

        # Move first 4 characters to end
        iban = iban[4:] + iban[:4]
        
        # Convert letters to numbers
        iban = ''.join(str(int(c, 36)) if c.isalpha() else c for c in iban)
        
        # Check if divisible by 97
        return int(iban) % 97 == 1
    
    def _get_valid_country_codes(self) -> Set[str]:
        """Get set of valid country codes for IBAN."""
        import pycountry
        return {country.alpha_2 for country in pycountry.countries} 