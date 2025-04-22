from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from transformers import pipeline
import spacy
from typing import List, Dict
import os

class MultilingualPIIDetector:
    """PII detector supporting both English and French"""
    
    def __init__(self):
        # Initialize analyzers for both languages
        self.analyzer_en = AnalyzerEngine()
        self.analyzer_fr = AnalyzerEngine()
        
        # Initialize the anonymizer engine
        self.anonymizer = AnonymizerEngine()
        
        # Define custom operators for anonymization
        self.operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMISÉ>"}),
            "PHONE_NUMBER": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "*",
                    "chars_to_mask": 12,
                    "from_end": True,
                },
            ),
            "EMAIL_ADDRESS": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "@",
                    "chars_to_mask": 10,
                    "from_end": True,
                },
            ),
            "CREDIT_CARD": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "#",
                    "chars_to_mask": 16,
                    "from_end": True,
                },
            ),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSONNE>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LIEU>"}),
            "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ORGANISATION>"}),
            "US_SSN": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "#",
                    "chars_to_mask": 11,
                    "from_end": False,
                },
            ),
        }
        
        # Add custom recognizers
        self._add_custom_recognizers()
    
    def _add_custom_recognizers(self):
        """Add custom recognizers for better French support"""
        # Add French-specific recognizers here if needed
        pass
    
    def detect_and_anonymize(self, text: str, language: str = 'en') -> Dict:
        """
        Detect and anonymize PII in the given text
        
        Args:
            text: Input text to analyze
            language: Language code ('en' or 'fr')
            
        Returns:
            Dict containing original text, anonymized text, and detected entities
        """
        # Select appropriate analyzer based on language
        analyzer = self.analyzer_en if language == 'en' else self.analyzer_fr
        
        try:
            # Analyze text for PII
            analyzer_results = analyzer.analyze(
                text=text,
                language=language,
                entities=[
                    "PERSON", "LOCATION", "ORGANIZATION", 
                    "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
                    "US_SSN", "DATE_TIME", "URL", "IP_ADDRESS"
                ]
            )
            
            # Anonymize detected PII
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self.operators
            )
            
            # Extract detected entities
            entities = [
                {
                    'type': result.entity_type,
                    'text': text[result.start:result.end],
                    'start': result.start,
                    'end': result.end,
                    'score': result.score
                }
                for result in analyzer_results
            ]
            
            return {
                'original_text': text,
                'anonymized_text': anonymized_result.text,
                'entities': entities,
                'language': language
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_text': text,
                'anonymized_text': text,
                'entities': [],
                'language': language
            }
    
    def batch_detect_and_anonymize(self, texts: List[str], language: str = 'en') -> List[Dict]:
        """Process multiple texts in batch"""
        return [self.detect_and_anonymize(text, language) for text in texts] 