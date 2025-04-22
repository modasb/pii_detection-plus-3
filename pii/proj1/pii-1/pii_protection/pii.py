import logging
import re
import threading
import csv
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import arabic_reshaper
from bidi.algorithm import get_display
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpEngineProvider
import spacy
import torch
import time
import os
import hashlib
from dataclasses import dataclass
import json
import gc
from farasa.segmenter import FarasaSegmenter
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import bisect
from fuzzywuzzy import fuzz
from nltk import sent_tokenize

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PIIEntity:
    """Data class for PII entities"""
    text: str
    type: str
    score: float
    start: int
    end: int
    method: str

@dataclass
class PIIDetectionConfig:
    """Configuration for PII detection"""
    # Language settings
    language: str = "en"
    use_language_specific_masking: bool = False
    
    # Performance settings
    use_gpu: bool = torch.cuda.is_available()
    max_workers: int = 5
    cache_size: int = 1000
    
    # Model selection and method enablement
    enable_hf: bool = True
    enable_presidio: bool = True
    enable_spacy: bool = True
    enable_regex: bool = True
    enable_financial: bool = True
    
    # Confidence thresholds - Lowered for better detection
    min_confidence: float = 0.3  # Lowered from 0.5 for better detection
    financial_confidence: float = 0.4  # Lowered from 0.7
    
    # Cache settings
    cache_dir: Optional[str] = None
    
    # Logging settings
    debug_mode: bool = True  # Enable debug mode by default
    log_method_results: bool = True
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".pii_cache")
        
        # Set language-specific confidence threshold with lower values
        if hasattr(PIIConfig, 'CONFIDENCE_THRESHOLDS'):
            self.min_confidence = PIIConfig.CONFIDENCE_THRESHOLDS.get(self.language, 0.3)
            
        # Validate configuration
        if not any([self.enable_hf, self.enable_presidio, self.enable_spacy, self.enable_regex]):
            logger.warning("No detection methods are enabled! Enable at least one method.")

class PIIConfig:
    """Configuration for PII detection with support for English, French, and Arabic"""
    
    MODELS = {
        "en": {
            "spacy": "en_core_web_trf",
            "ner": [
                "dbmdz/bert-large-cased-finetuned-conll03-english",
                "dslim/bert-base-NER"
            ]
        },
        "fr": {
            "spacy": "fr_core_news_lg",
            "ner": [
                "Jean-Baptiste/camembert-ner-with-dates",
                "cmarkea/distilcamembert-base-ner"
            ]
        },
        "ar": {
            "spacy": None,  # Not using spaCy for Arabic
            "ner": [
                "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
                "aubmindlab/bert-base-arabertv2"
            ],
            "word_embeddings": "cc.ar.300.bin"  # FastText Arabic embeddings
        }
    }
    
    # Adjusted confidence thresholds for supported languages
    CONFIDENCE_THRESHOLDS = {
        "en": 0.4,
        "fr": 0.4,
        "ar": 0.5
    }
    
    # GPU memory settings
    GPU_SETTINGS = {
        "max_memory": "2GB",
        "device_map": "auto",
        "torch_dtype": "float16"
    }
    
    # Common regex patterns
    REGEX_PATTERNS = {
        "EMAIL": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
        "PHONE_NUMBER": re.compile(r'\b(?:\+?\d{1,3}[-. ]?)?\d{2,4}[-. ]?\d{3}[-. ]?\d{3,4}\b'),
        "ADDRESS": re.compile(r'\b\d{1,5}\s+(?:[A-Za-z0-9.-]+\s*)+(?:Street|St|Drive|Dr|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Court|Ct|Circle|Cir|Way|Place|Pl|Square|Sq)[.,]?\s*(?:[A-Za-z]+(?:\s+[A-Za-z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?', re.IGNORECASE),
        "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
        "DATE": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4})\b', re.IGNORECASE),
        "SSN": re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),
        "IP_ADDRESS": re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
        "URL": re.compile(r'\b(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:/[^\s]*)?', re.IGNORECASE),
    }
    
    # Language-specific regex patterns
    LANGUAGE_REGEX_PATTERNS = {
        "en": {
            "UK_PHONE": re.compile(r'\b(?:(?:\+44\s?|0)(?:1[0-9]{8,9}|[23478]\d{9}|[58]0\d{8}))\b'),
            "US_PHONE": re.compile(r'\b(?:\+1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),
        },
        "fr": {
            "FR_PHONE": re.compile(r'\b(?:(?:\+33|0)\s?[1-9](?:\s?\d{2}){4})\b'),
            "FR_POSTAL_CODE": re.compile(r'\b\d{2}\s?\d{3}\b'),
        },
        "ar": {
            "AR_PHONE": re.compile(r'\b(?:\+\d{3}|0)\d{8,10}\b'),
            "AR_NAME": re.compile(r'(?:[\u0600-\u06FF]+\s+){1,3}[\u0600-\u06FF]+'),
        }
    }
    
    # Standardized masking strategies
    MASKING_STRATEGIES = {
        "default": lambda x: "[MASKED]",
        "EMAIL": lambda x: "[EMAIL]",
        "PHONE_NUMBER": lambda x: "[PHONE]",
        "ADDRESS": lambda x: "[ADDR]",
        "CREDIT_CARD": lambda x: "[CC]",
        "SSN": lambda x: "[SSN]",
        "DATE": lambda x: "[DATE]",
        "PERSON": lambda x: "[PER]",
        "URL": lambda x: "[URL]",
        "IP_ADDRESS": lambda x: "[IP]",
        "ORGANIZATION": lambda x: "[ORG]",
        "LOCATION": lambda x: "[LOC]",
    }
    
    # Entity type mapping for normalization
    ENTITY_TYPE_MAPPING = {
        # Common mappings across all supported languages
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "PERSON": "PERSON",
        "NORP": "ORGANIZATION",
        "FAC": "LOCATION",
        "PRODUCT": "PRODUCT",
        
        # French-specific mappings
        "NOM": "PERSON",
        "LIEU": "LOCATION",
        "ORGANISATION": "ORGANIZATION",
        
        # Arabic-specific mappings
        "شخص": "PERSON",
        "مكان": "LOCATION",
        "منظمة": "ORGANIZATION",
        
        # Presidio mappings
        "PERSON": "PERSON",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
        "DATE_TIME": "DATE",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "CREDIT_CARD": "CREDIT_CARD",
        "CRYPTO": "CRYPTO",
        "IP_ADDRESS": "IP_ADDRESS",
        "URL": "URL",
    }
    
    @staticmethod
    def get_model_config(language: str) -> Dict[str, Any]:
        """Get model configuration with memory management settings"""
        if language not in {'en', 'fr', 'ar'}:
            logger.warning(f"Unsupported language {language}. Using English configuration.")
            language = 'en'
        
        base_config = {
            "model_max_length": 512,  # Prevent excessive memory usage
            "truncation": True,
            "padding": True,
            "aggregation_strategy": "simple"
        }
        
        if torch.cuda.is_available():
            base_config.update({
                "device_map": "auto",
                "torch_dtype": torch.float16
            })
        else:
            base_config["device"] = "cpu"
        
        return base_config

class ArabicPIIConfig:
    """Configuration for Arabic-specific PII detection with improved patterns"""
    
    # Common Arabic name prefixes and titles (expanded)
    NAME_PREFIXES = r'(?:الشيخ|الدكتور|الأستاذ|السيد|الحاج|المهندس|د\.|م\.|أ\.|الشيخة|الدكتورة|السيدة|الآنسة)'
    
    # Improved Arabic name pattern with MUCH stricter matching
    # Now requires either a name prefix or specific name structure
    ARABIC_NAME_PATTERN = fr'\b(?:{NAME_PREFIXES})\s+[\u0600-\u06FF]{{3,}}(?:\s+[\u0600-\u06FF]{{3,}}){{0,3}}\b|'
    ARABIC_NAME_PATTERN += fr'\b[\u0600-\u06FF]{{3,}}\s+(?:بن|بنت|ابن)\s+[\u0600-\u06FF]{{3,}}\b|'
    ARABIC_NAME_PATTERN += fr'\b(?:جون|ماري|ديفيد|مايكل|روبرت|جيمس|باربرا|ليندا|إليزابيث|سارة|جوزيف|توماس|تشارلز|كريستوفر)(?:\s+[\u0600-\u06FF]{{3,}}){{1,2}}\b'
    
    # Arabic address components with expanded patterns
    ADDRESS_COMPONENTS = {
        'STREET': r'شارع|طريق|حي|منطقة|جادة|ميدان',
        'BUILDING': r'عمارة|مبنى|برج|فيلا|منزل|مجمع',
        'AREA': r'حي|مدينة|قرية|منطقة|محافظة|ولاية',
        'NUMBER': r'\d+',
    }
    
    # Completely revised regex patterns for different types of Arabic PII
    PATTERNS = {
        # Completely revised name pattern - MUCH stricter
        "AR_NAME": re.compile(
            # Pattern 1: Must have a name prefix
            fr'\b(?:{NAME_PREFIXES})\s+[\u0600-\u06FF]{{3,}}(?:\s+[\u0600-\u06FF]{{3,}}){{0,3}}\b|'
            # Pattern 2: Must have name connector
            fr'\b[\u0600-\u06FF]{{3,}}\s+(?:بن|بنت|ابن)\s+[\u0600-\u06FF]{{3,}}\b|'
            # Pattern 3: Foreign names transliterated to Arabic (typically have specific patterns)
            fr'\b(?:جون|ماري|ديفيد|مايكل|روبرت|جيمس|باربرا|ليندا|إليزابيث|سارة|جوزيف|توماس|تشارلز|كريستوفر)(?:\s+[\u0600-\u06FF]{{3,}}){{1,2}}\b',
            re.UNICODE
        ),
        
        # Improved address pattern - requires specific address components
        "AR_ADDRESS": re.compile(
            fr'\b(?:{ADDRESS_COMPONENTS["AREA"]})\s+[\u0600-\u06FF]{{3,}}\s+'
            fr'(?:{ADDRESS_COMPONENTS["STREET"]})\s+[\u0600-\u06FF\s,]{{5,}}\b',
            re.UNICODE | re.IGNORECASE
        ),
        
        # Phone patterns remain the same as they're more reliable
        "AR_PHONE_SA": re.compile(r'\b(?:\+966|0)?[0-5]\d{8,9}\b'),
        "AR_PHONE_UAE": re.compile(r'\b(?:\+971|0)?(?:5[0-9]|2[0-9]|3[0-9]|4[0-9]|6[0-9]|7[0-9]|9[0-9])[0-9]{7}\b'),
        "AR_PHONE_EG": re.compile(r'\b(?:\+20|0)?1[0125][0-9]{8}\b'),
        
        # Improved organization pattern - requires org prefix
        "AR_ORGANIZATION": re.compile(
            r'\b(?:شركة|مؤسسة|مجموعة|بنك|جامعة|معهد|وزارة|هيئة|دائرة|مركز|مستشفى|مدرسة)\s+[\u0600-\u06FF\s]{5,}\b',
            re.UNICODE | re.IGNORECASE
        ),
    }
    
    # Increased confidence thresholds to reduce false positives
    CONFIDENCE_THRESHOLDS = {
        "AR_NAME": 0.8,  # Increased from 0.7
        "AR_EMAIL": 0.9,
        "AR_ORGANIZATION": 0.8,  # Increased from 0.7
        "AR_PHONE_SA": 0.9,
        "AR_PHONE_UAE": 0.9,
        "AR_PHONE_EG": 0.9,
    }
    
    # Enhanced context indicators for better confidence scoring
    CONTEXT_INDICATORS = {
        "AR_NAME": [
            "يدعى", "اسمه", "اسمها", "السيد", "السيدة", "الدكتور", "الدكتورة", 
            "المهندس", "المهندسة", "الأستاذ", "الأستاذة", "قال", "قالت", "صرح", "أكد",
            "ذكر", "أفاد", "أوضح", "أشار", "أعلن", "كتب", "نشر", "ألف", "ترأس", "يعمل"
        ],
        "AR_ORGANIZATION": [
            "شركة", "مؤسسة", "مجموعة", "بنك", "جامعة", "معهد", "وزارة", "هيئة", 
            "دائرة", "مركز", "مستشفى", "مدرسة", "مقرها", "مكاتبها", "فرع", "مدير"
        ],
        "AR_ADDRESS": [
            "يقع في", "عنوانه", "عنوانها", "مقره", "مقرها", "يقيم في", "تقيم في",
            "العنوان", "الموقع", "المكان", "الشارع", "الحي", "المنطقة", "المدينة"
        ],
    }
    
    # Common Arabic words that should never be considered names
    COMMON_WORDS = {
        "مبرمج", "مرتبط", "تطوير", "واجهة", "إنشاء", "بديهية", "المستخدمين",
        "التنقل", "برمجة", "تصميم", "تحليل", "بيانات", "معالجة", "نظام",
        "برنامج", "تطبيق", "موقع", "خدمة", "مشروع", "عمل", "شركة", "مؤسسة",
        "مدير", "موظف", "عامل", "مهندس", "مطور", "مصمم", "محلل", "مستخدم",
        "زبون", "عميل", "منتج", "خدمة", "سعر", "تكلفة", "ربح", "خسارة",
        "استثمار", "تمويل", "ميزانية", "مال", "سوق", "تجارة", "صناعة",
        "زراعة", "تعليم", "صحة", "نقل", "اتصال", "إعلام", "ثقافة", "رياضة",
        "سياحة", "سفر", "طعام", "شراب", "ملابس", "سكن", "منزل", "سيارة",
        "حاسوب", "هاتف", "جهاز", "أداة", "آلة", "معدات", "مواد", "منتجات",
        "بتطوير", "واجهة", "المستخدم", "تحسين", "تجربة", "المستخدم", "تطبيقات",
        "الويب", "الهاتف", "المحمول", "الذكي", "الحاسوب", "البرمجة", "التصميم",
        "التحليل", "التطوير", "البرمجيات", "الأنظمة", "الشبكات", "البيانات",
        "المعلومات", "الاتصالات", "الإنترنت", "الموقع", "الإلكتروني", "الرقمي",
        "الذكاء", "الاصطناعي", "التعلم", "الآلي", "البيانات", "الضخمة", "الحوسبة",
        "السحابية", "الأمن", "السيبراني", "الشبكة", "العنكبوتية", "التكنولوجيا",
        "المعلومات", "الاتصالات", "الرقمية", "الإلكترونية", "الحديثة", "المتطورة",
        "الجديدة", "القديمة", "الحالية", "المستقبلية", "الماضية", "الحاضرة"
    }

class FrenchPIIConfig:
    """Configuration for French-specific PII detection"""
    
    # French name prefixes
    NAME_PREFIXES = r'(?:M\.|Mme\.|Mlle\.|Dr\.|Prof\.|)'
    
    # French address components
    ADDRESS_COMPONENTS = {
        'STREET': r'rue|avenue|boulevard|place|allée|impasse',
        'NUMBER': r'\d+',
        'POSTAL': r'\d{5}'
    }
    
    PATTERNS = {
        "FR_NAME": re.compile(
            fr'\b{NAME_PREFIXES}\s*[A-ZÉÈÊËÎÏÔŒ][a-zéèêëîïôœ]+(?:\s+[A-ZÉÈÊËÎÏÔŒ][a-zéèêëîïôœ]+){{1,3}}\b'
        ),
        
        "FR_ADDRESS": re.compile(
            fr'\b(?:\d+)\s+'  # Number
            fr'(?:rue|avenue|boulevard|place|allée|impasse)\s+'  # Street type
            fr'[A-Za-zÉÈÊËÎÏÔŒéèêëîïôœ\s,]+\s+'  # Street name
            fr'(?:\d{{5}})\s+'  # Postal code
            fr'[A-Za-zÉÈÊËÎÏÔŒéèêëîïôœ\s]+',  # City
            re.IGNORECASE
        ),
        
        # French social security number (INSEE)
        "FR_SSN": re.compile(r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b'),
        
        # French tax identification number (SPI)
        "FR_TAX_ID": re.compile(r'\b[0-9]{13}\b'),
        
        # French phone numbers
        "FR_PHONE": re.compile(r'\b(?:(?:\+33|0)\s?[1-9](?:\s?\d{2}){4})\b'),
        
        # French company registration (SIRET)
        "FR_SIRET": re.compile(r'\b\d{14}\b'),
    }
    
    REDACTION_STRATEGIES = {
        "FR_NAME": lambda x: "[PER]",
        "FR_ADDRESS": lambda x: "[ADDR]",
        "FR_SSN": lambda x: "[SSN]",
        "FR_TAX_ID": lambda x: "[TAX]",
        "FR_PHONE": lambda x: "[PHONE]",
        "FR_SIRET": lambda x: "[SIRET]",
    }
    
    # Add context indicators for French
    CONTEXT_INDICATORS = {
        "FR_NAME": [
            "nommé", "appelé", "s'appelle", "je suis", "c'est", "prénom",
            "nom", "monsieur", "madame", "mademoiselle"
        ],
        "FR_ADDRESS": [
            "habite", "demeure", "adresse", "domicilié", "résidant",
            "situé", "localisé", "se trouve"
        ],
        "FR_PHONE": [
            "téléphone", "portable", "fixe", "numéro", "tel", "tél",
            "contactez", "joindre", "appeler"
        ],
        "FR_EMAIL": [
            "email", "courriel", "adresse mail", "contact", "@",
            "électronique", "e-mail"
        ]
    }
    
    # Add proximity weights for confidence adjustment
    PROXIMITY_WEIGHTS = {
        "immediate": 0.3,  # Same sentence
        "near": 0.2,      # Adjacent sentence
        "far": 0.1        # Within 3 sentences
    }


class FinancialPIIConfig:
    """Common financial PII patterns across languages"""
    
    # IBAN patterns by country
    IBAN_PATTERNS = {
        'FR': r'FR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}',  # French IBAN
        'SA': r'SA\d{2}\s?\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}',  # Saudi IBAN
        'AE': r'AE\d{2}\s?\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}',          # UAE IBAN
        'GB': r'GB\d{2}\s?[A-Z]{4}\s?\d{6}\s?\d{8}',                        # UK IBAN
    }
    
    # BIC/SWIFT codes
    BIC_PATTERN = r'[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?'
    
    # Credit card patterns
    CREDIT_CARD_PATTERNS = {
        'VISA': r'4[0-9]{12}(?:[0-9]{3})?',
        'MASTERCARD': r'5[1-5][0-9]{14}',
        'AMEX': r'3[47][0-9]{13}',
        'DISCOVER': r'6(?:011|5[0-9]{2})[0-9]{12}',
    }
    
    # Bank account numbers (generic pattern)
    ACCOUNT_NUMBER_PATTERN = r'\b\d{8,12}\b'

class ArabicFinancialPIIConfig:
    """Arabic-specific financial PII patterns"""
    PATTERNS = {
        "AR_IBAN_SA": re.compile(r'\bSA\d{2}\s?\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b'),
        "AR_IBAN_UAE": re.compile(r'\bAE\d{2}\s?\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b'),
        "AR_BANK_ACCOUNT": re.compile(r'\b\d{8,12}\b'),
        "AR_BANK_NAME": re.compile(r'\b(?:بنك|مصرف)\s+[\u0600-\u06FF\s]+\b', re.UNICODE),
        "AR_AMOUNT": re.compile(r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:ريال|درهم|دينار|جنيه)\b'),
        "AR_CREDIT_CARD": re.compile(r'\b(?:بطاقة\s+(?:ائتمان|فيزا|ماستركارد))\s*:\s*[0-9\s-]{16,19}\b')
    }
    
    REDACTION_STRATEGIES = {
        "AR_IBAN_SA": "[IBAN-SA]",
        "AR_IBAN_UAE": "[IBAN-UAE]",
        "AR_BANK_ACCOUNT": "[حساب]",
        "AR_BANK_NAME": "[بنك]",
        "AR_AMOUNT": "[مبلغ]",
        "AR_CREDIT_CARD": "[بطاقة]"
    }

class FrenchFinancialPIIConfig:
    """French-specific financial PII patterns"""
    PATTERNS = {
        "FR_IBAN": re.compile(r'FR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}'),
        "FR_BIC": re.compile(r'[A-Z]{4}FR[A-Z0-9]{2}(?:[A-Z0-9]{3})?'),
        "FR_ACCOUNT_NUMBER": re.compile(r'\b\d{11,23}\b'),
        "FR_BANK_NAME": re.compile(r'\b(?:Banque|Crédit|Caisse|Société)\b', re.IGNORECASE),
        "FR_AMOUNT": re.compile(r'\b\d+(?:,\d{2})?\s?(?:EUR|€)\b')
    }
    
    REDACTION_STRATEGIES = {
        "FR_IBAN": "[IBAN]",
        "FR_BIC": "[BIC]",
        "FR_ACCOUNT_NUMBER": "[COMPTE]",
        "FR_BANK_NAME": "[BANQUE]",
        "FR_AMOUNT": "[MONTANT]"
    }

class EnglishFinancialPIIConfig:
    """English-specific financial PII patterns"""
    PATTERNS = {
        "EN_IBAN": re.compile(r'GB\d{2}\s?[A-Z]{4}\s?\d{6}\s?\d{8}'),
        "EN_SORT_CODE": re.compile(r'\b\d{2}-\d{2}-\d{2}\b'),
        "EN_ACCOUNT_NUMBER": re.compile(r'\b\d{8,12}\b'),
        "EN_BIC": re.compile(r'[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?'),
        "EN_BANK_NAME": re.compile(r'\b(?:Bank|Trust|Savings|Federal|Credit Union)\b', re.IGNORECASE),
        "EN_AMOUNT": re.compile(r'\b\d+(?:\.\d{2})?\s?(?:USD|GBP|EUR)\b')
    }
    
    REDACTION_STRATEGIES = {
        "EN_IBAN": "[IBAN]",
        "EN_SORT_CODE": "[SORT]",
        "EN_ACCOUNT_NUMBER": "[ACC]",
        "EN_BIC": "[BIC]",
        "EN_BANK_NAME": "[BANK]",
        "EN_AMOUNT": "[AMT]"
    }

def detect_arabic_pii(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced Arabic PII detection function
    """
    detected_entities = []
    
    # Apply each pattern
    for entity_type, pattern in ArabicPIIConfig.PATTERNS.items():
        matches = pattern.finditer(text)
        for match in matches:
            # Get the matched text
            matched_text = match.group()
            
            # Calculate confidence based on pattern strength and context
            confidence = _calculate_arabic_confidence(
                matched_text, 
                entity_type,
                text[max(0, match.start()-20):match.end()+20]  # Get context
            )
            
            # Check if confidence meets threshold
            if confidence >= ArabicPIIConfig.CONFIDENCE_THRESHOLDS[entity_type]:
                detected_entities.append({
                    'text': matched_text,
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': confidence,
                    'method': 'arabic_regex'
                })
    
    return detected_entities

def _calculate_arabic_confidence(text: str, entity_type: str, context: str) -> float:
    """Calculate confidence score for Arabic PII entities with improved context validation"""
    base_confidence = ArabicPIIConfig.CONFIDENCE_THRESHOLDS.get(entity_type, 0.6)
    
    # Initialize confidence adjustments
    context_boost = 0
    length_penalty = 0
    dictionary_penalty = 0
    
    # Check for context indicators (improved)
    context_indicators = ArabicPIIConfig.CONTEXT_INDICATORS.get(entity_type, [])
    if context_indicators:
        # Check both before and after context with appropriate window sizes
        context_window = context.lower()
        for indicator in context_indicators:
            if indicator.lower() in context_window:
                context_boost += 0.15
                break
    
    # Apply length-based adjustments
    if entity_type == "AR_NAME":
        # Penalize very short names (likely false positives)
        if len(text.split()) == 1 and len(text) < 5:
            length_penalty = 0.3
        # Boost confidence for multi-word names with appropriate length
        elif len(text.split()) >= 2:
            context_boost += 0.1
    
    # Check if the text is a common word (improved dictionary check)
    if self._is_common_word(text, "ar"):
        dictionary_penalty = 0.4
    
    # Apply name prefix boost
    if entity_type == "AR_NAME":
        name_prefixes = ["السيد", "الدكتور", "الشيخ", "المهندس", "الأستاذ", "السيدة", "الآنسة"]
        if any(text.startswith(prefix) for prefix in name_prefixes):
            context_boost += 0.2
    
    # Calculate final confidence with all adjustments
    final_confidence = base_confidence + context_boost - length_penalty - dictionary_penalty
    
    # Ensure confidence is within valid range
    return max(0.0, min(1.0, final_confidence))

def redact_arabic_pii(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Redact detected Arabic PII using appropriate strategies
    """
    # Sort entities by start position in reverse order
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Create a list of characters from the text
    chars = list(text)
    
    # Replace each entity with its redacted version
    for entity in entities:
        entity_type = entity['type']
        original_text = entity['text']
        
        # Get redaction strategy
        redaction_func = ArabicPIIConfig.REDACTION_STRATEGIES.get(
            entity_type,
            lambda x: "محجوب"  # Default redaction
        )
        
        # Apply redaction
        redacted_text = redaction_func(original_text)
        chars[entity['start']:entity['end']] = list(redacted_text)
    
    return ''.join(chars)

class AdaptiveThresholds:
    """Manages adaptive confidence thresholds based on detection performance"""
    def __init__(self, initial_thresholds: Dict[str, float], learning_rate: float = 0.1):
        self.thresholds = initial_thresholds
        self.learning_rate = learning_rate
        self.false_positive_counts = {method: 0 for method in initial_thresholds}
        self.total_detections = {method: 0 for method in initial_thresholds}
    
    def adjust_threshold(self, method: str, is_false_positive: bool):
        """Adjust threshold based on false positive detection"""
        self.total_detections[method] += 1
        if is_false_positive:
            self.false_positive_counts[method] += 1
            
            # Calculate false positive rate
            fp_rate = self.false_positive_counts[method] / self.total_detections[method]
            
            # Adjust threshold if false positive rate is too high
            if fp_rate > 0.1:  # More than 10% false positives
                self.thresholds[method] = min(
                    0.95,  # Maximum threshold
                    self.thresholds[method] + self.learning_rate
                )
    
    def get_threshold(self, method: str) -> float:
        """Get current threshold for a method"""
        return self.thresholds.get(method, 0.5)  # Default threshold if method not found

class EntityLinker:
    """ML-based entity linking using word embeddings"""
    def __init__(self, language: str):
        self.language = language
        self.word_embeddings = None
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load appropriate word embeddings based on language"""
        try:
            if self.language == 'ar':
                model_path = PIIConfig.MODELS['ar']['word_embeddings']
                self.word_embeddings = fasttext.load_model(model_path)
            else:
                # Use appropriate embeddings for other languages
                model_path = f"cc.{self.language}.300.bin"
                self.word_embeddings = fasttext.load_model(model_path)
        except Exception as e:
            logger.error(f"Error loading word embeddings: {str(e)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get word embedding for text"""
        if self.word_embeddings is None:
            return None
        return self.word_embeddings.get_sentence_vector(text)
    
    def should_merge_entities(self, entity1: PIIEntity, entity2: PIIEntity, threshold: float = 0.8) -> bool:
        """Determine if entities should be merged based on semantic similarity"""
        if self.word_embeddings is None:
            return False
            
        # Get embeddings
        emb1 = self.get_embedding(entity1.text)
        emb2 = self.get_embedding(entity2.text)
        
        if emb1 is None or emb2 is None:
            return False
            
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        return similarity >= threshold

class PIIProtectionLayer:
    """Main class for PII detection and protection"""
    
    def __init__(self, 
                 model_name: str = "jean-baptiste/roberta-large-ner-english",
                 config: Optional[PIIDetectionConfig] = None,
                 use_gpu: Optional[bool] = None,
                 cache_dir: Optional[str] = None):
        """Initialize the PII Protection Layer."""
        
        # Add supported languages attribute
        self.supported_languages = ["en", "fr", "ar"]
        
        # Initialize configuration
        self.config = config or PIIDetectionConfig()
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.cache_dir = cache_dir
        
        # Initialize Presidio components
        self._initialize_presidio()
        
        # Initialize other components
        if self.config.enable_hf:
            self._initialize_models(model_name)
        if self.config.enable_spacy:
            self._initialize_spacy()
        
        self._compile_regex_patterns()
        self._setup_logging()
        
        # Initialize Farasa segmenter for Arabic
        self.farasa_segmenter = FarasaSegmenter() if self.config.language == 'ar' else None
        
        # Initialize adaptive thresholds
        self.adaptive_thresholds = AdaptiveThresholds({
            'regex': 0.7,
            'spacy': 0.6,
            'huggingface': 0.5,
            'presidio': 0.6
        })
        
        # Initialize entity linker
        self.entity_linker = EntityLinker(self.config.language)
        
        # Initialize pre-filtering patterns
        self.prefilter_patterns = {
            'email': r'@',
            'phone': r'\d{3}',
            'credit_card': r'\d{4}',
            'date': r'\d{2}[/-]',
            'ip': r'\d{1,3}\.\d{1,3}',
        }
    
    def _initialize_models(self, model_name: str) -> None:
        """Initialize the NER models."""
        try:
            # Initialize HuggingFace models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
        except Exception as e:
            logger.error(f"Error initializing HuggingFace models: {str(e)}")
            self.ner_pipeline = None
        
    def _initialize_presidio(self) -> None:
        """Initialize Presidio analyzer and anonymizer."""
        try:
            # Create NLP engine provider
            provider = NlpEngineProvider(nlp_engine_name="spacy")
            nlp_engine = provider.create_engine()
            
            # Initialize analyzer with registry
            registry = RecognizerRegistry()
            self.presidio_analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry
            )
            
            # Initialize anonymizer
            self.presidio_anonymizer = AnonymizerEngine()
            
            logger.info("Presidio components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {str(e)}")
            self.presidio_analyzer = None
            self.presidio_anonymizer = None
        
    def _initialize_spacy(self) -> None:
        """Initialize Spacy models for different languages"""
        self.spacy_models = {}
        models = {
            'en': 'en_core_web_trf',  # Using transformer model for better accuracy
            'fr': 'fr_core_news_lg',
            'ar': 'xx_ent_wiki_sm'  # Multilingual model for Arabic
        }
        
        for lang, model in models.items():
            try:
                self.spacy_models[lang] = spacy.load(model)
                logger.info(f"Loaded Spacy model for {lang}")
            except Exception as e:
                logger.warning(f"Could not load Spacy model for {lang}: {str(e)}")
                
    def _compile_regex_patterns(self) -> None:
        """Initialize regex patterns for PII detection"""
        # Use the patterns from PIIConfig
        self.patterns = PIIConfig.REGEX_PATTERNS.copy()
        
        # Add language-specific patterns
        for lang, patterns in PIIConfig.LANGUAGE_REGEX_PATTERNS.items():
            for pattern_name, pattern in patterns.items():
                self.patterns[pattern_name] = pattern
        
        logger.info(f"Compiled {len(self.patterns)} regex patterns")
        
    def _setup_logging(self) -> None:
        """Configure logging for the PII detection system"""
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug_mode else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler for debug logging if enabled
        if self.config.debug_mode:
            fh = logging.FileHandler('pii_detection_debug.log')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        logger.info(f"Initialized PII detection with config: {self.config}")
        logger.info(f"Language: {self.config.language}, Min confidence: {self.config.min_confidence}")
        logger.info(f"Enabled methods: HF={self.config.enable_hf}, Presidio={self.config.enable_presidio}, "
                    f"Spacy={self.config.enable_spacy}, Regex={self.config.enable_regex}")
        
    @lru_cache(maxsize=1000)
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for text caching"""
        return hashlib.sha256(text.encode()).hexdigest()
        
    def _detect_with_hf(self, text: str) -> List[PIIEntity]:
        """Detect PII entities using Hugging Face models with improved offset handling"""
        if not self.tokenizer or not self.model or not self.config.enable_hf:
            return []
        
        entities = []
        
        try:
            # Clean text for better processing
            cleaned_text = self._clean_hf_text(text)
            
            # Track character mapping between cleaned and original text
            char_mapping = self._create_char_mapping(text, cleaned_text)
            
            # Handle long texts by chunking
            max_length = self.tokenizer.model_max_length - 10
            chunks = []
            offsets = []
            
            # Create chunks with proper overlap to avoid entity splitting
            for i in range(0, len(cleaned_text), max_length // 2):
                chunk = cleaned_text[i:i + max_length]
                chunks.append(chunk)
                offsets.append(i)
            
            # Process each chunk
            for chunk_idx, (chunk, offset) in enumerate(zip(chunks, offsets)):
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                
                # Move to GPU if available
                if self.config.use_gpu and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    self.model = self.model.to("cuda")
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert predictions to entities
                predictions = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                
                # Extract entities from predictions
                current_entity = None
                
                for idx, (token, label_idx) in enumerate(zip(predictions, outputs.logits[0].argmax(dim=-1))):
                    label = self.model.config.id2label[label_idx.item()]
                    
                    # Skip special tokens
                    if token in self.tokenizer.special_tokens_map.values():
                        continue
                    
                    # Handle entity start
                    if label.startswith("B-"):
                        # Save previous entity if exists
                        if current_entity:
                            entities.append(current_entity)
                        
                        # Start new entity
                        entity_type = label[2:]  # Remove B- prefix
                        
                        # Get token position in original text
                        token_text = self.tokenizer.convert_tokens_to_string([token])
                        token_start = chunk.find(token_text, 0 if idx == 0 else chunk.find(self.tokenizer.convert_tokens_to_string([predictions[idx-1]])))
                        
                        if token_start != -1:
                            # Calculate global position with offset
                            global_start = offset + token_start
                            
                            # Map back to original text position
                            original_start = self._map_to_original_position(global_start, char_mapping)
                            
                            current_entity = {
                                "text": token_text,
                                "type": entity_type,
                                "start": original_start,
                                "end": original_start + len(token_text),
                                "score": outputs.logits[0][idx][label_idx].item()
                            }
                    
                    # Handle entity continuation
                    elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
                        token_text = self.tokenizer.convert_tokens_to_string([token])
                        
                        # Extend current entity
                        current_entity["text"] += token_text
                        current_entity["end"] = current_entity["start"] + len(current_entity["text"])
                
                # Add last entity if exists
                if current_entity:
                    entities.append(current_entity)
            
            # Convert to PIIEntity objects
            pii_entities = []
            for entity in entities:
                # Verify entity positions in original text
                entity_text = text[entity["start"]:entity["end"]]
                if entity_text.strip() and self._verify_entity_text(entity_text, entity["text"]):
                    pii_entities.append(PIIEntity(
                        text=entity_text,
                        type=self._normalize_entity_type(entity["type"], entity["score"], entity_text)[0],
                        score=entity["score"],
                        start=entity["start"],
                        end=entity["end"],
                        method="huggingface"
                    ))
            
            return pii_entities
        
        except Exception as e:
            if self.config.debug_mode:
                logger.error(f"Error in HuggingFace detection: {str(e)}")
            return []

    def _create_char_mapping(self, original_text: str, cleaned_text: str) -> Dict[int, int]:
        """Create mapping between cleaned text positions and original text positions"""
        mapping = {}
        orig_idx = 0
        clean_idx = 0
        
        while orig_idx < len(original_text) and clean_idx < len(cleaned_text):
            if original_text[orig_idx] == cleaned_text[clean_idx]:
                mapping[clean_idx] = orig_idx
                orig_idx += 1
                clean_idx += 1
            else:
                # Character was removed during cleaning
                orig_idx += 1
        
        # Fill any gaps in mapping
        for i in range(len(cleaned_text)):
            if i not in mapping:
                # Find nearest mapped position
                prev_mapped = max([k for k in mapping.keys() if k < i], default=0)
                mapping[i] = mapping[prev_mapped] + (i - prev_mapped)
        
        return mapping

    def _map_to_original_position(self, cleaned_pos: int, char_mapping: Dict[int, int]) -> int:
        """Map position in cleaned text back to original text"""
        if cleaned_pos in char_mapping:
            return char_mapping[cleaned_pos]
        
        # Find nearest mapped position
        keys = sorted(char_mapping.keys())
        idx = bisect.bisect_left(keys, cleaned_pos)
        
        if idx == 0:
            return char_mapping[keys[0]]
        
        if idx == len(keys):
            return char_mapping[keys[-1]] + (cleaned_pos - keys[-1])
        
        # Interpolate between mapped positions
        prev_key = keys[idx - 1]
        next_key = keys[idx]
        
        prev_val = char_mapping[prev_key]
        next_val = char_mapping[next_key]
        
        # Linear interpolation
        return prev_val + int((cleaned_pos - prev_key) * (next_val - prev_val) / (next_key - prev_key))

    def _verify_entity_text(self, original_text: str, entity_text: str) -> bool:
        """Verify that entity text matches or is contained in the original text"""
        original_normalized = self._normalize_text_for_comparison(original_text)
        entity_normalized = self._normalize_text_for_comparison(entity_text)
        
        return (
            entity_normalized in original_normalized or
            original_normalized in entity_normalized or
            fuzz.ratio(original_normalized, entity_normalized) > 80
        )

    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for comparison by removing extra spaces and lowercasing"""
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _detect_with_presidio(self, text: str, language: str = 'en') -> List[PIIEntity]:
        """Detect PII using Presidio."""
        if not self.presidio_analyzer:
            logger.warning("Presidio analyzer not initialized")
            return []
            
        try:
            # Get analyzer results
            results = self.presidio_analyzer.analyze(
                text=text,
                language=language,
                entities=self._get_supported_recognizers(language),
                score_threshold=self.config.min_confidence
            )
            
            # Convert to PIIEntity format
            entities = []
            for result in results:
                entity = PIIEntity(
                    text=text[result.start:result.end],
                    type=result.entity_type,
                    score=result.score,
                    start=result.start,
                    end=result.end,
                    method="presidio"
                )
                entities.append(entity)
                
            return entities
            
        except Exception as e:
            logger.error(f"Error in Presidio detection: {str(e)}")
            return []

    def _get_supported_recognizers(self, language: str) -> List[str]:
        """Get list of supported recognizers for a given language"""
        # Common recognizers that work well across languages
        common_recognizers = [
            "CreditCardRecognizer",
            "EmailRecognizer",
            "IpRecognizer",
            "PhoneRecognizer",
            "UrlRecognizer"
        ]
        
        # Language-specific recognizers
        language_recognizers = {
            'en': [
                "UsSsnRecognizer",
                "UsLicenseRecognizer",
                "UsPassportRecognizer",
                "UsItinRecognizer",
                "UsPhoneRecognizer"
            ]
        }
        
        # Get recognizers for the specified language
        supported = common_recognizers + language_recognizers.get(language, [])
        logger.debug(f"Supported recognizers for {language}: {supported}")
        return supported

    def _empty_result(self, language: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'original_text': '',
            'redacted_text': '',
            'detected_entities': [],
            'language': language,
            'statistics': {
                'processing_time': 0,
                'entity_count': 0,
                'methods_used': [],
                'method_stats': {}
            },
            'success': True
        }

    def _add_french_recognizers(self):
        """Add French-specific pattern recognizers"""
        try:
            # Phone number pattern
            phone_pattern = Pattern(
                name="french_phone",
                regex=FrenchPIIConfig.PATTERNS["FR_PHONE"].pattern,
                score=0.8
            )
            self.presidio_analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="PHONE_NUMBER",
                    supported_language="fr",
                    patterns=[phone_pattern]
                )
            )
            
            # Address pattern
            address_pattern = Pattern(
                name="french_address",
                regex=FrenchPIIConfig.PATTERNS["FR_ADDRESS"].pattern,
                score=0.7
            )
            self.presidio_analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="ADDRESS",
                    supported_language="fr",
                    patterns=[address_pattern]
                )
            )
            
        except Exception as e:
            logger.error(f"Error adding French recognizers: {str(e)}")

    def _add_arabic_recognizers(self):
        """Add Arabic-specific pattern recognizers"""
        try:
            # Name pattern
            name_pattern = Pattern(
                name="arabic_name",
                regex=ArabicPIIConfig.PATTERNS["AR_NAME"].pattern,
                score=0.8
            )
            self.presidio_analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="PERSON",
                    supported_language="ar",
                    patterns=[name_pattern]
                )
            )
            
            # Phone patterns
            phone_patterns = [
                Pattern(
                    name="arabic_phone_sa",
                    regex=ArabicPIIConfig.PATTERNS["AR_PHONE_SA"].pattern,
                    score=0.9
                ),
                Pattern(
                    name="arabic_phone_uae",
                    regex=ArabicPIIConfig.PATTERNS["AR_PHONE_UAE"].pattern,
                    score=0.9
                ),
                Pattern(
                    name="arabic_phone_eg",
                    regex=ArabicPIIConfig.PATTERNS["AR_PHONE_EG"].pattern,
                    score=0.9
                )
            ]
            self.presidio_analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="PHONE_NUMBER",
                    supported_language="ar",
                    patterns=phone_patterns
                )
            )
            
        except Exception as e:
            logger.error(f"Error adding Arabic recognizers: {str(e)}")

    def _detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns."""
        entities = []
        
        # Use patterns from PIIConfig
        for entity_type, pattern in PIIConfig.REGEX_PATTERNS.items():
            try:
                matches = list(pattern.finditer(text))
                for match in matches:
                    confidence = self._calculate_pattern_confidence(entity_type, match.group())
                    if confidence >= self.config.min_confidence:
                        entity = PIIEntity(
                        text=match.group(),
                        type=entity_type,
                        score=confidence,
                        start=match.start(),
                        end=match.end(),
                            method="regex"
                        )
                        entities.append(entity)
            except Exception as e:
                logger.error(f"Error in regex detection for {entity_type}: {str(e)}")
                continue
        
        return entities

    def _calculate_pattern_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence score for regex matches."""
        if entity_type == "EMAIL":
            # Higher confidence for well-formed email addresses
            if "@" in text and "." in text.split("@")[1]:
                return 0.95
            return 0.5
        
        elif entity_type == "PHONE_NUMBER":
            # Higher confidence for well-structured phone numbers
            if re.match(r'^\+?1?\d{3}[-.]?\d{3}[-.]?\d{4}$', text):
                return 0.9
            return 0.7
        
        elif entity_type == "CREDIT_CARD":
            # Higher confidence for valid credit card numbers
            if self._validate_credit_card(text.replace("-", "").replace(" ", "")):
                return 0.95
            return 0.6
        
        elif entity_type == "SSN":
            # Higher confidence for well-formed SSNs
            if re.match(r'^\d{3}-\d{2}-\d{4}$', text):
                return 0.9
            return 0.7
        
        # Default confidence for other patterns
        return 0.8

    def _detect_with_spacy(self, text: str, language: str = 'en') -> List[PIIEntity]:
        """Detect PII using Spacy with improved accuracy"""
        if language not in self.spacy_models:
            return []
            
        try:
            doc = self.spacy_models[language](text)
            entities = []
            
            # Define minimum lengths for different entity types
            min_lengths = {
                'PERSON': 4,
                'ORG': 3,
                'GPE': 2,
                'LOC': 2,
            }
            
            for ent in doc.ents:
                # Skip entities that are too short
                if len(ent.text.strip()) < min_lengths.get(ent.label_, 2):
                    continue
                    
                # Skip common words that are often misclassified
                if self._is_common_word(ent.text, language):
                    continue
                    
                # Normalize type and adjust confidence
                normalized_type, adjusted_score = self._normalize_entity_type(
                    ent.label_,
                    ent._.confidence if hasattr(ent._, 'confidence') else 0.8,
                    ent.text
                )
                
                if adjusted_score >= self.config.min_confidence:
                    entities.append(PIIEntity(
                        text=ent.text,
                        type=normalized_type,
                        score=adjusted_score,
                        start=ent.start_char,
                        end=ent.end_char,
                        method='spacy'
                    ))
                
            return entities
        except Exception as e:
            logger.error(f"Error in Spacy detection: {str(e)}")
            return []
            
    def _is_common_word(self, text: str, language: str) -> bool:
        """Check if text is a common word that should not be considered PII
        
        Args:
            text: Text to check
            language: Language code
            
        Returns:
            bool: True if text is a common word
        """
        # Normalize text
        text = text.lower().strip()
        
        # Skip very short text
        if len(text) < 3:
            return True
        
        # For Arabic
        if language == "ar":
            # Common Arabic words that are often false positives
            common_words = {
                "مبرمج", "مرتبط", "تطوير", "واجهة", "إنشاء", "بديهية", "المستخدمين",
                "التنقل", "برمجة", "تصميم", "تحليل", "بيانات", "معالجة", "نظام",
                "برنامج", "تطبيق", "موقع", "خدمة", "مشروع", "عمل", "شركة", "مؤسسة",
                "مدير", "موظف", "عامل", "مهندس", "مطور", "مصمم", "محلل", "مستخدم",
                "زبون", "عميل", "منتج", "خدمة", "سعر", "تكلفة", "ربح", "خسارة",
                "استثمار", "تمويل", "ميزانية", "مال", "سوق", "تجارة", "صناعة",
                "زراعة", "تعليم", "صحة", "نقل", "اتصال", "إعلام", "ثقافة", "رياضة",
                "سياحة", "سفر", "طعام", "شراب", "ملابس", "سكن", "منزل", "سيارة",
                "حاسوب", "هاتف", "جهاز", "أداة", "آلة", "معدات", "مواد", "منتجات"
            }
            
            # Check if text is in common words
            if text in common_words:
                return True
            
            # Check if text is part of common phrases
            common_phrases = [
                "تطوير واجهة", "إنشاء واجهة", "المستخدمين التنقل", "تحليل البيانات",
                "معالجة البيانات", "تطوير النظام", "تصميم الموقع", "إدارة المشروع",
                "تنفيذ العمل", "تطوير البرمجيات", "تصميم التطبيق", "إدارة الموقع"
            ]
            
            for phrase in common_phrases:
                if text in phrase or phrase in text:
                    return True
        
        # For English
        elif language == "en":
            # Common English words (abbreviated list)
            common_words = {
                "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
                "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
                "there", "their", "what", "out", "about", "who", "get", "which", "when", "make",
                "can", "like", "time", "just", "him", "know", "take", "people", "into", "year",
                "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
                "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
                "two", "how", "our", "work", "first", "well", "way", "even", "new", "want"
            }
            
            if text in common_words:
                return True
        
        # For French
        elif language == "fr":
            # Common French words (abbreviated list)
            common_words = {
                "le", "la", "les", "un", "une", "des", "et", "est", "sont", "pour",
                "dans", "avec", "sur", "que", "qui", "pas", "plus", "par", "mais", "ou",
                "donc", "car", "si", "ce", "cette", "ces", "mon", "ton", "son", "notre",
                "votre", "leur", "tout", "tous", "toute", "toutes", "quel", "quelle",
                "quels", "quelles", "autre", "autres", "même", "mêmes", "tel", "telle",
                "tels", "telles", "cet", "chaque", "plusieurs", "quelque", "quelques"
            }
            
            if text in common_words:
                return True
        
        return False

    def _clean_entity_text(self, text: str) -> str:
        """Clean up tokenization artifacts from entity text"""
        # Remove Hugging Face tokenizer artifacts
        text = text.replace('Ġ', '').replace('▁', '')
        # Remove leading/trailing spaces and punctuation
        text = text.strip('.,;: ')
        return text

    def _normalize_entity_type(self, entity_type: str, score: float, text: str) -> Tuple[str, float]:
        """Normalize entity types and adjust confidence scores based on content."""
        # Convert to uppercase for consistency
        entity_type = entity_type.upper()
        
        # Start with base score
        confidence = score
        
        # Length-based adjustments (less aggressive)
        if len(text) < 2:
            confidence *= 0.5  # Reduced penalty for very short text
        elif len(text) < 3:
            confidence *= 0.7  # Reduced penalty for short text
        
        # Common word check (skip for certain types)
        if entity_type not in {'PERSON', 'ORG', 'GPE', 'LOC'} and self._is_common_word(text, self.config.language):
            confidence *= 0.8  # Reduced penalty for common words
        
        # Boost confidence for strong patterns
        if entity_type in {'EMAIL', 'PHONE_NUMBER', 'SSN', 'CREDIT_CARD'}:
            if any(pattern.fullmatch(text) for pattern in PIIConfig.REGEX_PATTERNS.values()):
                confidence = min(1.0, confidence * 1.2)
        
        # Boost for multiple detection methods
        if hasattr(self, '_method_counts') and text in self._method_counts:
            method_count = self._method_counts[text]
            if method_count > 1:
                confidence = min(1.0, confidence * (1.0 + 0.1 * method_count))
        
        # Filter out common non-PII CARDINAL and TIME entities
        if entity_type in ['CARDINAL', 'TIME']:
            # Skip standalone numbers that don't match sensitive patterns
            if re.match(r'^\d+$', text.strip()):
                return None, 0.0
                
            # Skip time-related phrases that aren't dates
            time_skip_patterns = [
                r'countless\s+(?:hours|days|weeks|months|years)',
                r'(?:few|many|several)\s+(?:hours|days|weeks|months|years)',
                r'(?:early|late|mid)',
            ]
            if any(re.search(pattern, text.lower()) for pattern in time_skip_patterns):
                return None, 0.0

        # Only keep CARDINAL if it matches sensitive number patterns
        if entity_type == 'CARDINAL':
            sensitive_patterns = [
                PIIConfig.REGEX_PATTERNS['CREDIT_CARD'],
                PIIConfig.REGEX_PATTERNS['PHONE_NUMBER'],
                PIIConfig.REGEX_PATTERNS['SSN'],
            ]
            if not any(pattern.search(text) for pattern in sensitive_patterns):
                return None, 0.0

        # Reduce confidence for very short entities
        if len(text.strip()) < 3 and entity_type not in ['ID', 'AGE']:
            confidence *= 0.5

        return entity_type, confidence

    def _merge_entities(self, entities: List[Union[PIIEntity, List[PIIEntity]]]) -> List[PIIEntity]:
        """Merge entities from different detection methods with improved duplicate handling
        
        Args:
            entities: List of entities or lists of entities from different detection methods
            
        Returns:
            List[PIIEntity]: Merged list of entities
        """
        # Flatten list of entities
        flat_entities = []
        for entity_or_list in entities:
            if isinstance(entity_or_list, list):
                flat_entities.extend(entity_or_list)
            else:
                flat_entities.append(entity_or_list)
        
        # Skip if no entities
        if not flat_entities:
            return []
        
        # Sort entities by position
        sorted_entities = sorted(flat_entities, key=lambda e: (e.start, e.end))
        
        # Group overlapping entities
        entity_groups = []
        current_group = [sorted_entities[0]]
        
        for entity in sorted_entities[1:]:
            # Check if current entity overlaps with any entity in current group
            overlaps = False
            for group_entity in current_group:
                overlap_type = self._check_overlap(group_entity, entity)
                if overlap_type != "none":
                    overlaps = True
                    break
            
            # If overlaps, add to current group
            if overlaps:
                current_group.append(entity)
            # Otherwise, start new group
            else:
                entity_groups.append(current_group)
                current_group = [entity]
        
        # Add last group
        entity_groups.append(current_group)
        
        # Merge each group
        merged_entities = []
        
        for group in entity_groups:
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Check for exact duplicates first (same text, type, start, end)
                unique_entities = {}
                for entity in group:
                    key = (entity.text, entity.type, entity.start, entity.end)
                    if key not in unique_entities or entity.score > unique_entities[key].score:
                        unique_entities[key] = entity
                
                # If only one unique entity after deduplication
                if len(unique_entities) == 1:
                    merged_entities.append(list(unique_entities.values())[0])
                    continue
                
                # Group by entity type
                type_groups = {}
                for entity in unique_entities.values():
                    base_type = entity.type.split('_')[0] if '_' in entity.type else entity.type
                    if base_type not in type_groups:
                        type_groups[base_type] = []
                    type_groups[base_type].append(entity)
                
                # Process each type group
                for type_name, type_entities in type_groups.items():
                    if len(type_entities) == 1:
                        merged_entities.append(type_entities[0])
                    else:
                        # For PERSON entities, use specialized merging
                        if type_name == "PERSON" or "NAME" in type_name:
                            merged_person = self._merge_person_entities(type_entities)
                            if merged_person:
                                merged_entities.append(merged_person)
                        else:
                            # For other types, merge based on confidence and coverage
                            merged_entity = self._merge_type_entities(type_entities)
                            if merged_entity:
                                merged_entities.append(merged_entity)
        
        # Final pass to merge adjacent entities based on rules
        merged_entities = self._merge_adjacent_entities(merged_entities)
        
        return merged_entities

    def _merge_adjacent_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge adjacent entities based on rule-based logic
        
        Args:
            entities: List of entities to check for adjacency
            
        Returns:
            List[PIIEntity]: List with adjacent entities merged
        """
        if not entities or len(entities) < 2:
            return entities
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))
        
        # Iteratively merge adjacent entities
        i = 0
        while i < len(sorted_entities) - 1:
            entity1 = sorted_entities[i]
            entity2 = sorted_entities[i + 1]
            
            # Check if entities should be merged based on rules
            if self._check_rule_based_merge(entity1, entity2):
                # Get text between entities
                between_text = self._get_text_between(entity1, entity2)
                
                # Create merged entity
                merged_text = entity1.text + between_text + entity2.text
                merged_entity = PIIEntity(
                    text=merged_text,
                    type=entity1.type,  # Use type from first entity
                    score=max(entity1.score, entity2.score),  # Use higher confidence
                    start=entity1.start,
                    end=entity2.end,
                    method=f"{entity1.method}+{entity2.method}"
                )
                
                # Replace entities with merged entity
                sorted_entities[i] = merged_entity
                sorted_entities.pop(i + 1)
            else:
                i += 1
        
        return sorted_entities

    def _check_rule_based_merge(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """Fallback rule-based merging logic for entities that should be merged based on rules
        
        Args:
            entity1: First entity to check
            entity2: Second entity to check
            
        Returns:
            bool: True if entities should be merged based on rules
        """
        # Check if entities are adjacent or very close (within 3 characters)
        if abs(entity2.start - entity1.end) <= 3:
            # Check for name parts that should be merged
            if entity1.type == "PERSON" and entity2.type == "PERSON":
                return True
                
            # Check for Arabic name parts
            if entity1.type == "AR_NAME" and entity2.type == "AR_NAME":
                return True
                
            # Check for address components
            if (entity1.type == "LOCATION" or "ADDRESS" in entity1.type) and \
               (entity2.type == "LOCATION" or "ADDRESS" in entity2.type):
                return True
                
            # Check for organization name parts
            if entity1.type == "ORGANIZATION" and entity2.type == "ORGANIZATION":
                return True
                
            # Check for date components
            if entity1.type == "DATE" and entity2.type == "DATE":
                return True
        
        # Check for specific patterns like phone numbers with country codes
        if (entity1.type == "PHONE_NUMBER" and entity2.type == "PHONE_NUMBER") or \
           (entity1.type.endswith("_PHONE") and entity2.type.endswith("_PHONE")):
            return True
            
        # Check for name with title (e.g., "Dr." + "John Smith")
        if entity1.type == "TITLE" and entity2.type == "PERSON":
            return True
            
        # For Arabic specifically
        if self.config.language == "ar":
            # Check for Arabic name components (first name + last name)
            if entity1.type == "AR_NAME" and entity2.type == "AR_NAME":
                # Get text between entities
                between_text = self._get_text_between(entity1, entity2)
                # Check if the text between contains connectors like "بن" or "ابن"
                if "بن" in between_text or "ابن" in between_text or between_text.strip() == "":
                    return True
        
        return False

    def _is_valid_name_combination(self, prev_text: str, next_text: str) -> bool:
        """
        Check if two name fragments form a valid name combination
        """
        # Name pattern validation
        name_patterns = [
            r'^[A-Z][a-z]+$',  # Single capitalized word
            r'^[A-Z]\.$',      # Initial with period
            r'^[A-Z]$',        # Single capital letter
            r'^(?:van|de|der|el|al|bin|ibn|mc|mac|von|van der)\s[A-Z][a-z]+$',  # Common prefixes
            r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)*$',  # Hyphenated names
            r'^\p{Lu}\p{Ll}+$',  # Unicode aware capitalization for non-English names
        ]
        
        return (any(re.match(pattern, prev_text) for pattern in name_patterns) and
                any(re.match(pattern, next_text) for pattern in name_patterns))

    def _merge_name_group(self, entities: List[PIIEntity]) -> PIIEntity:
        """
        Merge a group of name entities into a single entity with improved confidence scoring
        """
        if not entities:
            return None
        
        # Combine texts with appropriate spacing
        combined_text = ''
        for i, entity in enumerate(entities):
            if i > 0:
                # Add space unless it's an initial
                if not (len(entity.text) == 2 and entity.text.endswith('.')):
                    combined_text += ' '
            combined_text += entity.text
        
        # Calculate combined score with bias towards multiple detections
        base_score = max(e.score for e in entities)
        method_bonus = len(set(e.method for e in entities)) * 0.1  # Bonus for multiple methods
        length_bonus = min(0.1, len(entities) * 0.05)  # Bonus for multiple name parts
        final_score = min(1.0, base_score + method_bonus + length_bonus)
        
        # Combine methods
        methods = '+'.join(sorted(set(e.method for e in entities)))
        
        logger.debug(f"Merged name entities: {[e.text for e in entities]} -> {combined_text}")
        
        return PIIEntity(
            text=combined_text.strip(),
            type='PERSON',
            score=final_score,
            start=entities[0].start,
            end=entities[-1].end,
            method=methods
        )

    def _merge_type_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Merge entities of the same type with improved handling
        """
        if not entities:
            return []
            
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            if self._should_merge_entities(current, next_entity):
                # Merge entities
                current = self._merge_entity_pair(current, next_entity)
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged

    def _post_process_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Post-process detected entities to remove false positives and improve accuracy."""
        if not entities:
            return []

        filtered_entities = []
        
        for entity in entities:
            # Skip if normalized type returned None
            norm_type, norm_score = self._normalize_entity_type(entity.type, entity.score, entity.text)
            if norm_type is None:
                continue

            # Skip entities with very low confidence
            if norm_score < self.config.min_confidence:
                continue

            # Skip common words unless they're part of a larger entity
            if self._is_common_word(entity.text, self.config.language):
                if len(entity.text.split()) <= 1:  # Skip only single common words
                    continue

            # Update entity with normalized values
            entity.type = norm_type
            entity.score = norm_score
            filtered_entities.append(entity)

        # Additional filtering for numeric entities
        filtered_entities = [
            entity for entity in filtered_entities
            if not (
                entity.type in ['CARDINAL', 'TIME'] and
                len(entity.text.strip()) < 4 and  # Very short numbers
                not any(char.isalpha() for char in entity.text)  # Pure numbers
            )
        ]

        return filtered_entities

    def _remove_repetitive_patterns(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Remove repetitive short entity patterns
        """
        if len(entities) < 2:
            return entities
        
        # Group entities by type and text
        pattern_groups = {}
        for entity in entities:
            key = (entity.type, entity.text)
            pattern_groups.setdefault(key, []).append(entity)
        
        # Filter out repetitive short patterns
        filtered = []
        for (entity_type, text), group in pattern_groups.items():
            if len(text) < 5 and len(group) > 2:
                # Keep only the highest scoring instance
                best_entity = max(group, key=lambda x: x.score)
                filtered.append(best_entity)
            else:
                filtered.extend(group)
        
        return filtered

    def _filter_overlapping_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Filter overlapping entities with improved logic
        """
        if len(entities) < 2:
            return entities
        
        # Sort by start position and score
        entities.sort(key=lambda x: (x.start, -x.score))
        
        filtered = []
        last_end = -1
        
        for entity in entities:
            if entity.start >= last_end:
                filtered.append(entity)
                last_end = entity.end
            else:
                # Check if current entity is more specific or higher confidence
                prev = filtered[-1]
                if (entity.score > prev.score + 0.1 or  # Significantly higher confidence
                    (len(entity.text) > len(prev.text) and entity.score >= prev.score)):  # More specific with same/better confidence
                    filtered[-1] = entity
                    last_end = entity.end
        
        return filtered

    def _adjust_entity_boundaries(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Adjust entity boundaries based on context and patterns
        """
        adjusted = []
        for entity in entities:
            # Trim common prefixes/suffixes that shouldn't be included
            text = entity.text.strip()
            start_adj = entity.start
            end_adj = entity.end
            
            # Remove common artifacts
            prefixes = ['"', "'", "«", "(", "[", "{"]
            suffixes = ['"', "'", "»", ")", "]", "}", ".", ",", ";", ":"]
            
            while text and text[0] in prefixes:
                text = text[1:]
                start_adj += 1
            while text and text[-1] in suffixes:
                text = text[:-1]
                end_adj -= 1
            
            if text:  # Only add if there's still text after trimming
                adjusted.append(PIIEntity(
                    text=text,
                    type=entity.type,
                    score=entity.score,
                    start=start_adj,
                    end=end_adj,
                    method=entity.method
                ))
        
        return adjusted

    def _is_mergeable_type(self, entity_type: str) -> bool:
        """
        Determine if entity type should be considered for merging
        """
        mergeable_types = {
            'PERSON', 'ORGANIZATION', 'LOCATION', 'ADDRESS',
            'PRODUCT', 'MONEY', 'DATE', 'TIME'
        }
        return entity_type in mergeable_types

    def _has_separator_between(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """
        Check if there are major separators between entities
        """
        # Major separators that should prevent merging
        separators = {'.', ';', '!', '?', '\n', '\r', '|', '--'}
        text_between = self._get_text_between(entity1, entity2)
        return any(sep in text_between for sep in separators)

    def _check_overlap(self, entity1: PIIEntity, entity2: PIIEntity) -> str:
        """
        Check how two entities overlap
        Returns: 'NONE', 'CONTAINS', 'CONTAINED', 'PARTIAL'
        """
        if entity1.start <= entity2.start and entity1.end >= entity2.end:
            return 'CONTAINS'
        elif entity2.start <= entity1.start and entity2.end >= entity1.end:
            return 'CONTAINED'
        elif (entity1.start <= entity2.start < entity1.end or 
              entity1.start < entity2.end <= entity1.end):
            return 'PARTIAL'
        return 'NONE'

    def _get_text_between(self, entity1: PIIEntity, entity2: PIIEntity) -> str:
        """Get the text between two entities"""
        # This is a placeholder - in a real implementation, you would need access to the original text
        # For now, we'll just return a space
        return ' '

    def _redact_text(self, text: str, entities: List[PIIEntity]) -> str:
        """Redact detected PII entities from text."""
        if not entities:
            return text
            
        # Sort entities by start position in reverse order
        entities = sorted(entities, key=lambda x: x.start, reverse=True)
        
        # Create a copy of the text
        redacted = text
        
        # Replace each entity with its mask
        for entity in entities:
            mask = PIIConfig.MASKING_STRATEGIES.get(
                entity.type,
                PIIConfig.MASKING_STRATEGIES["default"]
            )(entity.text)
            
            redacted = redacted[:entity.start] + mask + redacted[entity.end:]
        
        return redacted

    def analyze_text(self, text: str, language: str = 'en', config: Optional[PIIDetectionConfig] = None) -> Dict[str, Any]:
        """Analyze text for PII entities."""
        start_time = time.time()
        
        try:
            # Use provided config or default
            cfg = config or self.config
            
            # Initialize results
            all_entities = []
            method_stats = {}
            
            # Regex detection (first)
            try:
                regex_entities = self._detect_with_regex(text)
                all_entities.extend(regex_entities)
                method_stats["regex"] = {
                    "total": len(regex_entities),
                    "filtered": len(regex_entities),
                    "types": list(set(e.type for e in regex_entities))
                }
            except Exception as e:
                logger.error(f"Error in regex detection: {str(e)}")
            
            # Combine all detected entities
            pre_merge_count = 0
            for method, method_entities in method_results.items():
                pre_merge_count += len(method_entities)
                entities.extend(method_entities)
            
            logger.info(f"Total entities found before merging: {pre_merge_count}")
            
            # Early return if no entities found
            if not entities:
                logger.info("No entities detected in text")
                return []
            
            # Balance method contributions
            entities = self._balance_method_contributions(entities)
            logger.debug(f"Entities after balancing: {len(entities)}")
            
            # Merge and deduplicate entities
            pre_merge_entities = entities.copy()
            entities = self._merge_entities(entities)
            logger.info(f"Merged {len(pre_merge_entities)} entities into {len(entities)} unique entities")
            
            if len(entities) < len(pre_merge_entities) / 2:
                logger.warning(f"Significant entity loss during merging: {len(pre_merge_entities)} -> {len(entities)}")
            
            # Filter by confidence threshold
            pre_filter_count = len(entities)
            entities = [e for e in entities if e.score >= self.config.min_confidence]
            if len(entities) < pre_filter_count:
                logger.info(f"Filtered out {pre_filter_count - len(entities)} entities below confidence threshold")
            
            # Sort by position
            entities.sort(key=lambda x: x.start)
            
            elapsed = time.time() - start_time
            logger.info(f"PII detection completed in {elapsed:.2f}s. Found {len(entities)} final entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error in PII detection: {str(e)}")
            return []

    def detect_arabic_pii(self, text: str) -> List[PIIEntity]:
        """Enhanced Arabic PII detection with Farasa and CamelBERT"""
        try:
            # Pre-process text with Farasa
            processed_text = self._process_arabic_text(text)
            
            detected_entities = []
            
            # Use CamelBERT for NER
            if hasattr(self, 'ner_pipeline') and self.ner_pipeline is not None:
                try:
                    # Get model configuration without aggregation_strategy
                    model_config = PIIConfig.get_model_config('ar')
                    model_config.pop('aggregation_strategy', None)  # Remove if present
                    
                    # Run NER with proper configuration
                    ner_results = self.ner_pipeline(
                        processed_text,
                        aggregation_strategy="simple",
                        **model_config
                    )
                    
                    for result in ner_results:
                        confidence = result['score']
                        # Get adaptive threshold for the method
                        threshold = self.adaptive_thresholds.get_threshold('huggingface')
                        
                        if confidence >= threshold:
                            entity = PIIEntity(
                                text=result['word'],
                                type=result['entity_group'],  # Use entity_group instead of entity
                                score=confidence,
                                start=result['start'],
                                end=result['end'],
                                method='camelbert'
                            )
                            detected_entities.append(entity)
                except Exception as e:
                    logger.error(f"Error in CamelBERT NER: {str(e)}")
            
            # Apply regex patterns
            try:
                for pattern_name, pattern in ArabicPIIConfig.PATTERNS.items():
                    matches = pattern.finditer(text)  # Use original text for regex
                    for match in matches:
                        confidence = self._calculate_arabic_confidence(
                            match.group(),
                            pattern_name,
                            text[max(0, match.start()-20):match.end()+20]
                        )
                        threshold = self.adaptive_thresholds.get_threshold('regex')
                        
                        if confidence >= threshold:
                            detected_entities.append(PIIEntity(
                                text=match.group(),
                                type=pattern_name,
                                score=confidence,
                                start=match.start(),
                                end=match.end(),
                                method='arabic_regex'
                            ))
            except Exception as e:
                logger.error(f"Error in Arabic regex detection: {str(e)}")
            
            # Merge entities using ML-based linking
            if detected_entities:
                merged_entities = self._merge_entities(detected_entities)
                return merged_entities
            
            return []
            
        except Exception as e:
            logger.error(f"Error in Arabic PII detection: {str(e)}")
            return []
    
    def _detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Enhanced regex detection with pre-filtering and Arabic support"""
        try:
            entities = []
            
            # Process Arabic text if needed
            if self.config.language == 'ar':
                text = self._process_arabic_text(text)
            
            # Get potential PII regions
            potential_regions = self._prefilter_text(text)
            
            # Get patterns based on language
            if self.config.language == 'ar':
                patterns = ArabicPIIConfig.PATTERNS
            elif self.config.language == 'fr':
                patterns = FrenchPIIConfig.PATTERNS
            else:
                patterns = PIIConfig.REGEX_PATTERNS
            
            # Add language-specific patterns
            lang_patterns = PIIConfig.LANGUAGE_REGEX_PATTERNS.get(self.config.language, {})
            patterns.update(lang_patterns)
            
            for start, end, pii_type in potential_regions:
                region_text = text[start:end]
                
                # Get relevant patterns for the PII type
                relevant_patterns = self._get_patterns_for_type(pii_type)
                
                for pattern_name, pattern in relevant_patterns.items():
                    matches = pattern.finditer(region_text)
                    for match in matches:
                        confidence = self._calculate_pattern_confidence(
                            match.group(),
                            pattern_name,
                            region_text
                        )
                        
                        # Get adaptive threshold
                        threshold = self.adaptive_thresholds.get_threshold('regex')
                        
                        if confidence >= threshold:
                            entity = PIIEntity(
                                text=match.group(),
                                type=pattern_name,
                                score=confidence,
                                start=start + match.start(),
                                end=start + match.end(),
                                method='regex'
                            )
                            entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in regex detection: {str(e)}")
            return []

    def _get_patterns_for_type(self, pii_type: str) -> Dict[str, re.Pattern]:
        """Get relevant patterns based on PII type and language"""
        patterns = {}
        
        if self.config.language == 'ar':
            if pii_type == 'phone':
                patterns.update({
                    'AR_PHONE_SA': ArabicPIIConfig.PATTERNS['AR_PHONE_SA'],
                    'AR_PHONE_UAE': ArabicPIIConfig.PATTERNS['AR_PHONE_UAE'],
                    'AR_PHONE_EG': ArabicPIIConfig.PATTERNS['AR_PHONE_EG']
                })
            elif pii_type == 'name':
                patterns['AR_NAME'] = ArabicPIIConfig.PATTERNS['AR_NAME']
            elif pii_type == 'address':
                patterns['AR_ADDRESS'] = ArabicPIIConfig.PATTERNS['AR_ADDRESS']
            # Add other Arabic-specific patterns...
        else:
            # Handle other languages...
            if pii_type == 'email':
                patterns['EMAIL'] = PIIConfig.REGEX_PATTERNS['EMAIL']
            elif pii_type == 'phone':
                patterns['PHONE_NUMBER'] = PIIConfig.REGEX_PATTERNS['PHONE_NUMBER']
                patterns.update({k: v for k, v in PIIConfig.LANGUAGE_REGEX_PATTERNS.get(self.config.language, {}).items() 
                               if 'PHONE' in k})
            # Add other patterns...
        
        return patterns
    
    def _calculate_pattern_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence based on pattern matching strength."""
        pattern_confidence = 0.0
        
        if entity_type == "EMAIL":
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', text):
                pattern_confidence = 0.9
        elif entity_type == "PHONE":
            if re.match(r'^\+?[\d\s-]{10,}$', text):
                pattern_confidence = 0.8
        elif entity_type == "CREDIT_CARD":
            if re.match(r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$', text):
                pattern_confidence = 0.9
        elif entity_type == "SSN":
            if re.match(r'^\d{3}-\d{2}-\d{4}$', text):
                pattern_confidence = 0.9
            
        return pattern_confidence

    def _prefilter_text(self, text: str) -> List[Tuple[int, int, str]]:
        """Pre-filter text to identify potential PII-containing regions"""
        potential_regions = []
        
        # Split text into lines
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            # Check each pattern
            for pii_type, pattern in self.prefilter_patterns.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Add context window around match
                    start = max(0, current_pos + match.start() - 20)
                    end = min(len(text), current_pos + match.end() + 20)
                    potential_regions.append((start, end, pii_type))
            
            current_pos += len(line) + 1  # +1 for newline
        
        return self._merge_overlapping_regions(potential_regions)
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Merge overlapping text regions"""
        if not regions:
            return []
            
        # Sort by start position
        regions.sort(key=lambda x: x[0])
        
        merged = []
        current = regions[0]
        
        for next_region in regions[1:]:
            if next_region[0] <= current[1]:
                # Regions overlap, merge them
                current = (
                    current[0],
                    max(current[1], next_region[1]),
                    f"{current[2]}+{next_region[2]}"
                )
            else:
                merged.append(current)
                current = next_region
        
        merged.append(current)
        return merged
    
    def _process_arabic_text(self, text: str) -> str:
        """Process Arabic text for better detection"""
        try:
            if not text or not isinstance(text, str):
                return text
                
            # Basic normalization
            text = text.strip()
            
            # Normalize Arabic characters
            text = re.sub('[إأٱآا]', 'ا', text)  # Normalize alef
            text = re.sub('ى', 'ي', text)  # Normalize yeh
            text = re.sub('ة', 'ه', text)  # Normalize teh marbuta
            
            # Handle common spacing issues
            text = re.sub(r'\s+', ' ', text)  # Normalize spaces
            text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Join split numbers
            
            # Use Farasa for morphological analysis if available
            if hasattr(self, 'farasa_segmenter') and self.farasa_segmenter:
                try:
                    text = self.farasa_segmenter.segment(text)
                    # Clean up Farasa markers
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                except Exception as e:
                    logger.warning(f"Farasa processing failed: {str(e)}")
            
            return text
        except Exception as e:
            logger.error(f"Error in Arabic text processing: {str(e)}")
            return text
    
    def _merge_entities(self, entities: List[Union[PIIEntity, List[PIIEntity]]]) -> List[PIIEntity]:
        """Merge entities from different detection methods with improved duplicate handling
        
        Args:
            entities: List of entities or lists of entities from different detection methods
            
        Returns:
            List[PIIEntity]: Merged list of entities
        """
        # Flatten list of entities
        flat_entities = []
        for entity_or_list in entities:
            if isinstance(entity_or_list, list):
                flat_entities.extend(entity_or_list)
            else:
                flat_entities.append(entity_or_list)
        
        # Skip if no entities
        if not flat_entities:
            return []
        
        # Sort entities by position
        sorted_entities = sorted(flat_entities, key=lambda e: (e.start, e.end))
        
        # Group overlapping entities
        entity_groups = []
        current_group = [sorted_entities[0]]
        
        for entity in sorted_entities[1:]:
            # Check if current entity overlaps with any entity in current group
            overlaps = False
            for group_entity in current_group:
                overlap_type = self._check_overlap(group_entity, entity)
                if overlap_type != "none":
                    overlaps = True
                    break
            
            # If overlaps, add to current group
            if overlaps:
                current_group.append(entity)
            # Otherwise, start new group
            else:
                entity_groups.append(current_group)
                current_group = [entity]
        
        # Add last group
        entity_groups.append(current_group)
        
        # Merge each group
        merged_entities = []
        
        for group in entity_groups:
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Check for exact duplicates first (same text, type, start, end)
                unique_entities = {}
                for entity in group:
                    key = (entity.text, entity.type, entity.start, entity.end)
                    if key not in unique_entities or entity.score > unique_entities[key].score:
                        unique_entities[key] = entity
                
                # If only one unique entity after deduplication
                if len(unique_entities) == 1:
                    merged_entities.append(list(unique_entities.values())[0])
                    continue
                
                # Group by entity type
                type_groups = {}
                for entity in unique_entities.values():
                    base_type = entity.type.split('_')[0] if '_' in entity.type else entity.type
                    if base_type not in type_groups:
                        type_groups[base_type] = []
                    type_groups[base_type].append(entity)
                
                # Process each type group
                for type_name, type_entities in type_groups.items():
                    if len(type_entities) == 1:
                        merged_entities.append(type_entities[0])
                    else:
                        # For PERSON entities, use specialized merging
                        if type_name == "PERSON" or "NAME" in type_name:
                            merged_person = self._merge_person_entities(type_entities)
                            if merged_person:
                                merged_entities.append(merged_person)
                        else:
                            # For other types, merge based on confidence and coverage
                            merged_entity = self._merge_type_entities(type_entities)
                            if merged_entity:
                                merged_entities.append(merged_entity)
        
        # Final pass to merge adjacent entities based on rules
        merged_entities = self._merge_adjacent_entities(merged_entities)
        
        return merged_entities

    def _merge_adjacent_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge adjacent entities based on rule-based logic
        
        Args:
            entities: List of entities to check for adjacency
            
        Returns:
            List[PIIEntity]: List with adjacent entities merged
        """
        if not entities or len(entities) < 2:
            return entities
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: (e.start, e.end))
        
        # Iteratively merge adjacent entities
        i = 0
        while i < len(sorted_entities) - 1:
            entity1 = sorted_entities[i]
            entity2 = sorted_entities[i + 1]
            
            # Check if entities should be merged based on rules
            if self._check_rule_based_merge(entity1, entity2):
                # Get text between entities
                between_text = self._get_text_between(entity1, entity2)
                
                # Create merged entity
                merged_text = entity1.text + between_text + entity2.text
                merged_entity = PIIEntity(
                    text=merged_text,
                    type=entity1.type,  # Use type from first entity
                    score=max(entity1.score, entity2.score),  # Use higher confidence
                    start=entity1.start,
                    end=entity2.end,
                    method=f"{entity1.method}+{entity2.method}"
                )
                
                # Replace entities with merged entity
                sorted_entities[i] = merged_entity
                sorted_entities.pop(i + 1)
            else:
                i += 1
        
        return sorted_entities

    def _check_rule_based_merge(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """Fallback rule-based merging logic for entities that should be merged based on rules
        
        Args:
            entity1: First entity to check
            entity2: Second entity to check
            
        Returns:
            bool: True if entities should be merged based on rules
        """
        # Check if entities are adjacent or very close (within 3 characters)
        if abs(entity2.start - entity1.end) <= 3:
            # Check for name parts that should be merged
            if entity1.type == "PERSON" and entity2.type == "PERSON":
                return True
                
            # Check for Arabic name parts
            if entity1.type == "AR_NAME" and entity2.type == "AR_NAME":
                return True
                
            # Check for address components
            if (entity1.type == "LOCATION" or "ADDRESS" in entity1.type) and \
               (entity2.type == "LOCATION" or "ADDRESS" in entity2.type):
                return True
                
            # Check for organization name parts
            if entity1.type == "ORGANIZATION" and entity2.type == "ORGANIZATION":
                return True
                
            # Check for date components
            if entity1.type == "DATE" and entity2.type == "DATE":
                return True
        
        # Check for specific patterns like phone numbers with country codes
        if (entity1.type == "PHONE_NUMBER" and entity2.type == "PHONE_NUMBER") or \
           (entity1.type.endswith("_PHONE") and entity2.type.endswith("_PHONE")):
            return True
            
        # Check for name with title (e.g., "Dr." + "John Smith")
        if entity1.type == "TITLE" and entity2.type == "PERSON":
            return True
            
        # For Arabic specifically
        if self.config.language == "ar":
            # Check for Arabic name components (first name + last name)
            if entity1.type == "AR_NAME" and entity2.type == "AR_NAME":
                # Get text between entities
                between_text = self._get_text_between(entity1, entity2)
                # Check if the text between contains connectors like "بن" or "ابن"
                if "بن" in between_text or "ابن" in between_text or between_text.strip() == "":
                    return True
        
        return False

    def _validate_arabic_entity_context(self, entity: PIIEntity, text: str) -> float:
        """Validate Arabic entity based on surrounding context
        
        Args:
            entity: The entity to validate
            text: The full text
            
        Returns:
            float: Context validation score (0.0-1.0)
        """
        # Get context window around entity
        start = max(0, entity.start - 50)
        end = min(len(text), entity.end + 50)
        context = text[start:end]
        
        validation_score = 0.5  # Base score
        
        # Check for context indicators based on entity type
        if entity.type == "AR_NAME":
            name_indicators = [
                "يدعى", "اسمه", "اسمها", "السيد", "السيدة", "الدكتور", "الدكتورة", 
                "المهندس", "المهندسة", "الأستاذ", "الأستاذة", "قال", "قالت", "صرح", "أكد",
                "ذكر", "أفاد", "أوضح", "أشار", "أعلن", "كتب", "نشر", "ألف", "ترأس", "يعمل"
            ]
            
            # Check if any indicators are present in context
            for indicator in name_indicators:
                if indicator in context:
                    validation_score += 0.2
                    break
            
            # Check if entity text has multiple words (more likely to be a name)
            if len(entity.text.split()) > 1:
                validation_score += 0.1
            
            # Penalize very short names (likely false positives)
            if len(entity.text) < 5:
                validation_score -= 0.2
        
        elif "ADDRESS" in entity.type:
            address_indicators = [
                "يقع في", "عنوانه", "عنوانها", "مقره", "مقرها", "يقيم في", "تقيم في",
                "العنوان", "الموقع", "المكان", "الشارع", "الحي", "المنطقة", "المدينة"
            ]
            
            for indicator in address_indicators:
                if indicator in context:
                    validation_score += 0.2
                    break
        
        # Ensure score is within valid range
        return max(0.0, min(1.0, validation_score))

    def _calculate_french_confidence(self, text: str, entity_type: str, entity_text: str) -> float:
        """Calculate confidence score for French entities based on context and patterns."""
        base_confidence = 0.7
        context_boost = 0.0
        
        # Get context indicators for this entity type
        indicators = FrenchPIIConfig.CONTEXT_INDICATORS.get(entity_type, [])
        
        # Get surrounding context (3 sentences)
        sentences = sent_tokenize(text)
        entity_sentence_idx = None
        
        # Find which sentence contains our entity
        for idx, sentence in enumerate(sentences):
            if entity_text in sentence:
                entity_sentence_idx = idx
                break
                
        if entity_sentence_idx is not None:
            # Check immediate context (same sentence)
            current_sentence = sentences[entity_sentence_idx]
            for indicator in indicators:
                if indicator.lower() in current_sentence.lower():
                    context_boost += FrenchPIIConfig.PROXIMITY_WEIGHTS["immediate"]
                    
            # Check adjacent sentences
            start_idx = max(0, entity_sentence_idx - 1)
            end_idx = min(len(sentences), entity_sentence_idx + 2)
            
            for idx in range(start_idx, end_idx):
                if idx == entity_sentence_idx:
                    continue
                for indicator in indicators:
                    if indicator.lower() in sentences[idx].lower():
                        context_boost += FrenchPIIConfig.PROXIMITY_WEIGHTS["near"]
        
        # Pattern-based confidence adjustments
        pattern_confidence = self._calculate_pattern_confidence(entity_type, entity_text)
        
        # Combine scores
        final_confidence = min(1.0, base_confidence + context_boost + pattern_confidence)
        
        return final_confidence

    def _calculate_pattern_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence based on pattern matching strength."""
        pattern_confidence = 0.0
        
        if entity_type == "FR_EMAIL":
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', text):
                pattern_confidence = 0.3
        elif entity_type == "FR_PHONE":
            if re.match(r'^\+33\s?[1-9](\s?\d{2}){4}$', text):
                pattern_confidence = 0.3
        elif entity_type == "FR_NAME":
            # Check for proper capitalization and length
            words = text.split()
            if all(word[0].isupper() and word[1:].islower() for word in words):
                pattern_confidence = 0.2
        
        return pattern_confidence

    def _enhance_cross_method_detection(self, text: str) -> List[PIIEntity]:
        """Enhance detection by cross-checking between different methods."""
        entities = []
        
        # Get entities from all methods
        hf_entities = self._detect_with_hf(text)
        spacy_entities = self._detect_with_spacy(text)
        regex_entities = self._detect_with_regex(text)
        
        # Create method confidence weights
        method_weights = {
            'regex': 0.9,    # High confidence for structured patterns
            'hf': 0.8,       # Good for general NER
            'spacy': 0.7     # Backup method
        }
        
        # Combine and cross-validate entities
        all_entities = []
        for entity in hf_entities + spacy_entities + regex_entities:
            # Boost confidence if multiple methods detect same entity
            overlapping = self._find_overlapping_entities(entity, all_entities)
            if overlapping:
                # Average confidence weighted by method reliability
                methods = set([e.method for e in overlapping] + [entity.method])
                confidence = sum(method_weights[m] for m in methods) / len(methods)
                entity.score = min(1.0, confidence)
            
            all_entities.append(entity)
        
        # Filter and merge overlapping entities
        final_entities = self._merge_overlapping_detections(all_entities)
        
        return final_entities

    def _find_overlapping_entities(self, entity: PIIEntity, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Find entities that overlap with the given entity."""
        overlapping = []
        for e in entities:
            # Check for text or position overlap
            if (e.text == entity.text or
                (e.start <= entity.end and entity.start <= e.end)):
                overlapping.append(e)
        return overlapping

    def _merge_overlapping_detections(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge overlapping entities, preferring higher confidence detections."""
        if not entities:
            return []
        
        # Sort by start position and confidence
        entities.sort(key=lambda x: (x.start, -x.score))
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            if current.end >= next_entity.start:
                # Overlapping entities - keep the higher confidence one
                if next_entity.score > current.score:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged