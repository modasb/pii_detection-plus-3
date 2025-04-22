import logging
import re
import time
from typing import List, Dict, Any, Tuple

# LangChain imports
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIILangChainDetector:
    """
    A PII detection system that uses LangChain and Hugging Face models
    for multilingual PII detection.
    """
    
    def __init__(self, huggingface_api_token=None):
        """Initialize the LangChain-based PII detection system."""
        logger.info("Initializing PII LangChain Detector...")
        
        self.supported_languages = ['en', 'fr', 'ar']
        self.huggingface_api_token = huggingface_api_token
        
        # Initialize LangChain with Hugging Face
        try:
            self.llm = HuggingFaceHub(
                repo_id="bigscience/bloom", 
                huggingfacehub_api_token=huggingface_api_token,
                model_kwargs={"temperature": 0.1, "max_length": 64}
            )
            logger.info("Successfully initialized Hugging Face LLM")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face LLM: {e}")
            self.llm = None
        
        # Create prompt templates for different languages
        self.prompts = {
            'en': PromptTemplate(
                input_variables=["text"],
                template="""
                Identify all personally identifiable information (PII) in the following text.
                Return only the PII entities as a list in the format: [ENTITY_TYPE: text]
                
                Text: {text}
                
                PII entities:
                """
            ),
            'fr': PromptTemplate(
                input_variables=["text"],
                template="""
                Identifiez toutes les informations personnelles identifiables (PII) dans le texte suivant.
                Retournez uniquement les entités PII sous forme de liste au format: [TYPE_ENTITÉ: texte]
                
                Texte: {text}
                
                Entités PII:
                """
            ),
            'ar': PromptTemplate(
                input_variables=["text"],
                template="""
                حدد جميع المعلومات الشخصية القابلة للتعريف (PII) في النص التالي.
                قم بإرجاع كيانات PII فقط كقائمة بالتنسيق: [نوع_الكيان: النص]
                
                النص: {text}
                
                كيانات PII:
                """
            )
        }
        
        # Create LLM chains for each language
        self.chains = {}
        if self.llm:
            for lang in self.supported_languages:
                self.chains[lang] = LLMChain(llm=self.llm, prompt=self.prompts[lang])
        
        # Fallback regex patterns for when LLM is not available
        self.patterns = {
            'en': {
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'PHONE_NUMBER': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
                'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            },
            'fr': {
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'PHONE_NUMBER': r'\b(?:\+33|0)\s?[1-9](?:[\s.-]?\d{2}){4}\b',
                'PERSON': r'\b[A-ZÀ-Ÿ][a-zà-ÿ]+\s+[A-ZÀ-Ÿ][a-zà-ÿ]+\b',
            },
            'ar': {
                'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'PHONE_NUMBER': r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
                'PERSON': r'[\u0600-\u06FF]+\s+[\u0600-\u06FF]+',
            }
        }

    def detect_pii(self, text: str, language: str = 'en') -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect PII using LangChain.
        
        Returns:
            Tuple of (detected entities, processing time)
        """
        if language not in self.supported_languages:
            language = 'en'  # Default to English if language not supported
            
        start_time = time.time()
        detected_entities = []
        
        # Try using LangChain first
        if self.llm and language in self.chains:
            try:
                chain = self.chains[language]
                response = chain.run(text=text)
                
                # Parse the response
                # Expected format: [ENTITY_TYPE: text]
                entity_matches = re.finditer(r'\[(.*?):\s*(.*?)\]', response)
                
                for match in entity_matches:
                    entity_type = match.group(1).strip()
                    entity_text = match.group(2).strip()
                    
                    # Find the position in the original text
                    text_pos = text.find(entity_text)
                    if text_pos >= 0:
                        detected_entities.append({
                            'type': entity_type,
                            'text': entity_text,
                            'start': text_pos,
                            'end': text_pos + len(entity_text),
                            'score': 0.8,  # Default confidence score
                            'method': 'langchain'
                        })
            except Exception as e:
                logger.error(f"LangChain detection error: {str(e)}")
                # Fall back to regex if LangChain fails
                detected_entities = self._detect_with_regex(text, language)
        else:
            # Fall back to regex if LLM is not available
            detected_entities = self._detect_with_regex(text, language)
        
        end_time = time.time()
        return detected_entities, end_time - start_time
    
    def _detect_with_regex(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Detect PII using regex patterns as a fallback."""
        detected_entities = []
        
        for entity_type, pattern in self.patterns[language].items():
            matches = re.finditer(pattern, text)
            for match in matches:
                start, end = match.span()
                detected_entities.append({
                    'type': entity_type,
                    'text': match.group(),
                    'start': start,
                    'end': end,
                    'score': 0.7,  # Lower confidence for regex
                    'method': 'regex'
                })
        
        return detected_entities
    
    def redact_pii(self, text: str, detected_entities: List[Dict[str, Any]]) -> str:
        """
        Redact detected PII entities from the text.
        """
        # Sort entities by start position in reverse order to avoid index shifting
        sorted_entities = sorted(detected_entities, key=lambda x: x['start'], reverse=True)
        
        # Create a mutable list of characters from the text
        chars = list(text)
        
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['type']
            
            # Different redaction strategies based on entity type
            if entity_type == 'EMAIL' or entity_type == 'EMAIL_ADDRESS':
                redacted = '[EMAIL REDACTED]'
            elif entity_type == 'PHONE_NUMBER':
                redacted = '[PHONE REDACTED]'
            elif entity_type == 'ADDRESS':
                redacted = '[ADDRESS REDACTED]'
            elif entity_type == 'PERSON' or entity_type == 'PERSON_NAME':
                # Partial redaction for names (first initial, then asterisks)
                name = text[start:end]
                name_parts = name.split()
                redacted_parts = []
                for part in name_parts:
                    if len(part) > 1:
                        redacted_parts.append(part[0] + '*' * (len(part) - 1))
                    else:
                        redacted_parts.append(part)
                redacted = ' '.join(redacted_parts)
            elif entity_type == 'CREDIT_CARD':
                redacted = '[CREDIT CARD REDACTED]'
            elif entity_type == 'SSN':
                redacted = '[SSN REDACTED]'
            else:
                # Default full redaction
                redacted = '[' + entity_type + ' REDACTED]'
            
            # Replace the entity with its redacted version
            chars[start:end] = list(redacted)
        
        return ''.join(chars)
    
    def analyze_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Complete PII analysis workflow: detect and redact PII."""
        detected_entities, processing_time = self.detect_pii(text, language)
        redacted_text = self.redact_pii(text, detected_entities)
        
        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'detected_entities': detected_entities,
            'language': language,
            'entity_count': len(detected_entities),
            'processing_time': processing_time
        }
    
    def batch_detect_pii(self, texts: List[str], language: str = 'en', batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process multiple texts in batches."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.analyze_text(text, language))
        
        return results 