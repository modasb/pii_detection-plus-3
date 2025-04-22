import logging
import re
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import json
from groq import Groq
import google.generativeai as genai
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import tiktoken
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import timeout_decorator
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
import spacy

DetectorFactory.seed = 0

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMPIIConfig:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "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")
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "AIzaSyDgtMtegycrQL-cAC-pZ-NH1or16Jq0Kn4")
    model_provider: Literal["groq", "gemini", "both", "alternate"] = "groq"  # Default to Groq for speed
    groq_model: str = "mixtral-8x7b-32768"
    gemini_model: str = "gemini-2.0-flash"
    max_workers: int = 10  # Increased for concurrency
    chunk_size: int = 4000  # Larger chunks to reduce API calls
    overlap: int = 0  # No overlap to minimize chunks
    max_retries: int = 1  # Reduced retries
    iou_threshold: float = 0.5
    min_confidence: float = 0.3
    allow_type_merging: bool = True
    supported_languages: List[str] = None
    timeout: int = 5  # Shorter timeout

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "fr", "ar"]
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

@dataclass
class ProcessingStats:
    groq_calls: int = 0
    gemini_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_retries: int = 0
    cache_hits: int = 0
    processing_time: float = 0.0

class LLMPIIDetector:
    ENTITY_TYPES = {
        "PERSON": {"tag": "[PER]"},
        "EMAIL": {"tag": "[EMAIL]"},
        "PHONE": {"tag": "[PHONE]"},
        "ADDRESS": {"tag": "[ADDR]"},
        "DATE": {"tag": "[DATE]"},
        "ORGANIZATION": {"tag": "[ORG]"},
        "CREDIT_CARD": {"tag": "[CC]"},
        "SSN": {"tag": "[SSN]"},
        "IP_ADDRESS": {"tag": "[IP]"},
        "URL": {"tag": "[URL]"},
        "FINANCIAL": {"tag": "[FIN]"},
        "ID": {"tag": "[ID]"}
    }

    LANGUAGE_MAPPING = {
        'ar': 'ar', 'arb': 'ar', 'ara': 'ar',  # Arabic variants
        'fr': 'fr', 'fra': 'fr', 'fre': 'fr',  # French variants
        'en': 'en', 'eng': 'en'                 # English variants
    }

    GROQ_TOKEN_LIMIT = 32768
    GEMINI_TOKEN_LIMIT = 32000

    REGEX_PATTERNS = {
        "en": {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}",
            "CREDIT_CARD": r"(?:\d{4}[- ]?){3}\d{4}|\d{16}",
            "SSN": r"\d{3}-\d{2}-\d{4}",
            "IP_ADDRESS": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            "URL": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
            "ID": r"[A-Za-z0-9-]{8,}"
        },
        "fr": {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}",  # French phone format
            "CREDIT_CARD": r"(?:\d{4}[- ]?){3}\d{4}|\d{16}",
            "SSN": r"\d{1}\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{3}\s*\d{3}\s*\d{2}",  # French social security
            "IP_ADDRESS": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            "URL": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
            "ID": r"[A-Za-z0-9-]{8,}"
        },
        "ar": {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"(?:(?:\+|00)(?:20|961|962|963|964|965|966|967|968|971|972|973|974|975|976|977|98)[-\s]?)?(?:\d[-\s]?){8,12}",  # Middle Eastern phone formats
            "CREDIT_CARD": r"(?:\d{4}[- ]?){3}\d{4}|\d{16}",
            "IP_ADDRESS": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            "URL": r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
            "ID": r"[A-Za-z0-9-]{8,}"
        }
    }

    EXAMPLES = {
        "en": {
            "text": "John Smith's email is john.smith@company.com and phone is +1 (555) 123-4567.",
            "entities": [
                {"text": "John Smith", "type": "PERSON", "start": 0, "end": 10, "confidence": 0.95},
                {"text": "john.smith@company.com", "type": "EMAIL", "start": 21, "end": 42, "confidence": 0.99},
                {"text": "+1 (555) 123-4567", "type": "PHONE", "start": 56, "end": 73, "confidence": 0.98}
            ]
        },
        "fr": {
            "text": "Marie Dubois habite au 15 Rue de la Paix, Paris. Son numéro est 06 12 34 56 78.",
            "entities": [
                {"text": "Marie Dubois", "type": "PERSON", "start": 0, "end": 12, "confidence": 0.95},
                {"text": "15 Rue de la Paix, Paris", "type": "ADDRESS", "start": 24, "end": 45, "confidence": 0.97},
                {"text": "06 12 34 56 78", "type": "PHONE", "start": 60, "end": 73, "confidence": 0.98}
            ]
        },
        "ar": {
            "text": "محمد أحمد يسكن في شارع الملك فهد، الرياض. رقم هاتفه +966 50 123 4567",
            "entities": [
                {"text": "محمد أحمد", "type": "PERSON", "start": 0, "end": 9, "confidence": 0.95},
                {"text": "شارع الملك فهد، الرياض", "type": "ADDRESS", "start": 17, "end": 38, "confidence": 0.97},
                {"text": "+966 50 123 4567", "type": "PHONE", "start": 51, "end": 66, "confidence": 0.98}
            ]
        }
    }

    def __init__(self, config: Optional[LLMPIIConfig] = None):
        self.config = config or LLMPIIConfig()
        self.current_model = self.config.model_provider if self.config.model_provider in ["groq", "gemini"] else "groq"

        if not self.config.groq_api_key:
            raise ValueError("Groq API key not provided.")
        
        logger.info(f"Initializing Groq client with API key: {self.config.groq_api_key[:8]}...")
        self.groq_client = Groq(api_key=self.config.groq_api_key)
        self._test_connection(self.groq_client, "Groq")

        self.gemini_model = None
        if self.config.gemini_api_key and self.config.model_provider in ["gemini", "alternate", "both"]:
            logger.info(f"Initializing Gemini client with API key: {self.config.gemini_api_key[:8]}...")
            try:
                self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
                self._test_connection(self.gemini_model, "Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {str(e)}. Using Groq only.")
                self.gemini_model = None

        self.stats = ProcessingStats()
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            logger.warning("Tiktoken unavailable. Using character-based token estimation.")
            self.tokenizer = None
        
        self.presidio_analyzer = AnalyzerEngine()
        self.presidio_anonymizer = AnonymizerEngine()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("SpaCy NER model unavailable. NER detection disabled.")
            self.nlp = None
        
        self._build_entity_mappings()

    def _test_connection(self, client, name: str):
        try:
            if name == "Groq":
                client.chat.completions.create(model=self.config.groq_model, messages=[{"role": "user", "content": "test"}], max_tokens=10)
            else:
                client.generate_content("test")
            logger.info(f"✓ {name} client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize {name} client: {str(e)}")

    def _build_entity_mappings(self):
        self.type_mapping = {}
        for canonical in self.ENTITY_TYPES:
            self.type_mapping[canonical] = canonical

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return min(len(text) // 4, self.GROQ_TOKEN_LIMIT)

    def _escape_text(self, text: str) -> str:
        return json.dumps(text)[1:-1]

    def _create_prompt(self, text: str, language: str) -> str:
        """Create a language-specific prompt for PII detection."""
        entity_list = ", ".join(self.ENTITY_TYPES.keys())
        escaped_text = self._escape_text(text)
        example = self.EXAMPLES.get(language, self.EXAMPLES["en"])
        
        prompts = {
            "en": f"""You are a specialized PII detector for English text. Identify ALL instances of personal identifiable information in the text.
Example text: {example['text']}
Example output: {json.dumps({"entities": example['entities']}, ensure_ascii=False, indent=2)}

Analyze this text and return ONLY valid JSON with ALL detected PII entities ({entity_list}):
{escaped_text}""",
            
            "fr": f"""Tu es un détecteur spécialisé d'informations personnelles en français. Identifie TOUTES les instances d'informations personnelles dans le texte.
Exemple de texte: {example['text']}
Exemple de sortie: {json.dumps({"entities": example['entities']}, ensure_ascii=False, indent=2)}

Analyse ce texte et renvoie UNIQUEMENT du JSON valide avec TOUTES les entités PII détectées ({entity_list}):
{escaped_text}""",
            
            "ar": f"""أنت كاشف متخصص للمعلومات الشخصية في النص العربي. حدد جميع حالات المعلومات الشخصية في النص.
مثال النص: {example['text']}
مثال المخرجات: {json.dumps({"entities": example['entities']}, ensure_ascii=False, indent=2)}

حلل هذا النص وأعد JSON صالح فقط مع جميع الكيانات الشخصية المكتشفة ({entity_list}):
{escaped_text}"""
        }
        
        return prompts.get(language, prompts["en"])

    def _retry_llm_call(self, call_func, *args, max_retries: int = 1) -> List[Dict]:
        for attempt in range(max_retries):
            try:
                return call_func(*args)
            except (TimeoutError, Exception) as e:
                wait_time = min(2 ** attempt, 10)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                self.stats.total_retries += 1
                time.sleep(wait_time)
        logger.error(f"All {max_retries} attempts failed for {call_func.__name__}.")
        return []

    @timeout_decorator.timeout(5, timeout_exception=TimeoutError)
    def _process_chunk_with_groq(self, chunk: Dict[str, Any], language: str) -> List[Dict]:
        self.stats.groq_calls += 1
        prompt = self._create_prompt(chunk["text"], language)
        token_count = self._count_tokens(prompt)
        if token_count > self.GROQ_TOKEN_LIMIT:
            logger.warning(f"Prompt too long ({token_count} tokens). Skipping.")
            return []

        self.stats.total_tokens += token_count
        result = self.groq_client.chat.completions.create(
            model=self.config.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000
        )
        raw_response = result.choices[0].message.content
        logger.debug(f"Groq raw response: {raw_response}")
        parsed = json.loads(raw_response) if raw_response else {"entities": []}
        entities = [e | {"method": "groq"} for e in parsed.get("entities", []) if self._validate_entity(e, chunk["text"])]
        self.stats.successful_calls += 1 if entities else 0
        self.stats.failed_calls += 1 if not entities else 0
        return entities

    @timeout_decorator.timeout(5, timeout_exception=TimeoutError)
    def _process_chunk_with_gemini(self, chunk: Dict[str, Any], language: str) -> List[Dict]:
        if not self.gemini_model:
            return []
        self.stats.gemini_calls += 1
        prompt = self._create_prompt(chunk["text"], language)
        token_count = self._count_tokens(prompt)
        if token_count > self.GEMINI_TOKEN_LIMIT:
            logger.warning(f"Prompt too long ({token_count} tokens). Skipping.")
            return []

        self.stats.total_tokens += token_count
        result = self.gemini_model.generate_content(prompt)
        raw_response = result.text
        logger.debug(f"Gemini raw response: {raw_response}")
        parsed = json.loads(raw_response) if raw_response else {"entities": []}
        entities = [e | {"method": "gemini"} for e in parsed.get("entities", []) if self._validate_entity(e, chunk["text"])]
        self.stats.successful_calls += 1 if entities else 0
        self.stats.failed_calls += 1 if not entities else 0
        return entities

    def _process_with_regex(self, text: str, language: str) -> List[Dict]:
        """Process text with language-specific regex patterns."""
        entities = []
        patterns = self.REGEX_PATTERNS.get(language, self.REGEX_PATTERNS["en"])
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entity = {
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.95,
                    "method": "regex"
                }
                if self._validate_entity(entity, text):
                    entities.append(entity)
        logger.debug(f"Regex detected entities ({language}): {entities}")
        return entities

    def _process_with_ner(self, text: str) -> List[Dict]:
        if not self.nlp:
            return []
        try:
            doc = self.nlp(text)
            entities = []
            spacy_to_pii = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "DATE": "DATE",
                "NORP": "ORGANIZATION",
            }
            for ent in doc.ents:
                entity_type = spacy_to_pii.get(ent.label_)
                if entity_type and entity_type in self.ENTITY_TYPES:
                    entity = {
                        "text": ent.text,
                        "type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.85,
                        "method": "ner"
                    }
                    if self._validate_entity(entity, text):
                        entities.append(entity)
            logger.debug(f"NER detected entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"NER processing failed: {str(e)}")
            return []

    def _process_with_presidio(self, text: str, language: str) -> List[Dict]:
        presidio_language = "en" if language not in ["en"] else language
        try:
            results = self.presidio_analyzer.analyze(text=text, language=presidio_language, score_threshold=self.config.min_confidence)
            entities = []
            for result in results:
                entity = {
                    "text": text[result.start:result.end],
                    "type": result.entity_type.upper(),
                    "start": result.start,
                    "end": result.end,
                    "confidence": result.score,
                    "method": "presidio"
                }
                if self._validate_entity(entity, text):
                    entities.append(entity)
            logger.debug(f"Presidio detected entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Presidio error: {str(e)}")
            return []

    def _validate_entity(self, entity: Dict, text: str) -> bool:
        required = {"text", "type", "start", "end", "confidence"}
        if not all(k in entity for k in required):
            logger.debug(f"Entity missing fields: {entity}")
            return False
        try:
            entity["start"] = int(entity["start"])
            entity["end"] = int(entity["end"])
            entity["confidence"] = float(entity["confidence"])
            if not (0 <= entity["start"] <= entity["end"] <= len(text) and 0 <= entity["confidence"] <= 1):
                logger.debug(f"Entity out of bounds or invalid confidence: {entity}")
                return False
            extracted = text[entity["start"]:entity["end"]]
            if extracted.strip() != entity["text"].strip():
                logger.debug(f"Text mismatch: extracted='{extracted}', entity='{entity['text']}'")
                window = 15
                search_start = max(0, entity["start"] - window)
                search_end = min(len(text), entity["end"] + window)
                search_text = text[search_start:search_end]
                match = re.search(re.escape(entity["text"].strip()), search_text)
                if match:
                    entity["start"] = search_start + match.start()
                    entity["end"] = search_start + match.end()
                    logger.debug(f"Adjusted entity position: {entity}")
                else:
                    return False
            entity["type"] = self._normalize_entity_type(entity["type"])
            return True
        except (ValueError, TypeError) as e:
            logger.debug(f"Entity validation error: {str(e)}, entity: {entity}")
            return False

    def _normalize_entity_type(self, entity_type: str) -> str:
        return self.type_mapping.get(entity_type.upper(), entity_type.upper())

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            if end < len(text):
                while end > start and not text[end].isspace():
                    end -= 1
            chunks.append({"text": text[start:end], "offset": start})
            start = end  # No overlap
        return chunks

    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        if not entities:
            return []
        entities.sort(key=lambda x: (x["start"], -x["confidence"]))
        merged = []
        for entity in entities:
            if entity["confidence"] < self.config.min_confidence:
                logger.debug(f"Entity below confidence threshold: {entity}")
                continue
            overlap = any(self._calculate_iou(entity, e) >= self.config.iou_threshold for e in merged)
            if not overlap:
                merged.append(entity)
            elif self.config.allow_type_merging:
                for i, existing in enumerate(merged):
                    if self._calculate_iou(entity, existing) >= self.config.iou_threshold and entity["confidence"] > existing["confidence"]:
                        merged[i] = entity
                        break
        return merged

    def _calculate_iou(self, e1: Dict, e2: Dict) -> float:
        start = max(e1["start"], e2["start"])
        end = min(e1["end"], e2["end"])
        if start >= end:
            return 0.0
        intersection = end - start
        union = (e1["end"] - e1["start"]) + (e2["end"] - e2["start"]) - intersection
        return intersection / union

    def _redact_text(self, text: str, entities: List[Dict]) -> str:
        try:
            analyzer_results = [
                RecognizerResult(
                    entity_type=e["type"],
                    start=e["start"],
                    end=e["end"],
                    score=e["confidence"]
                ) for e in entities
            ]
            anonymized = self.presidio_anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results
            )
            return anonymized.text
        except Exception as e:
            logger.error(f"Presidio anonymization failed: {str(e)}. Falling back to manual redaction.")
            chars = list(text)
            for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
                tag = self.ENTITY_TYPES.get(entity["type"], {}).get("tag", f"[{entity['type']}]")
                chars[entity["start"]:entity["end"]] = list(tag)
            return ''.join(chars)

    def _detect_language(self, text: str) -> str:
        """Detect text language with fallback and validation."""
        try:
            detected = detect(text)
            normalized = self.LANGUAGE_MAPPING.get(detected)
            if normalized in ["en", "fr", "ar"]:
                logger.info(f"Detected language: {normalized} (from {detected})")
                return normalized
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}")
        
        # Fallback: Try to guess based on character sets
        if re.search(r'[\u0600-\u06FF]', text):  # Arabic script
            return "ar"
        elif re.search(r'[éèêëàâäôöùûüçîïÉÈÊËÀÂÄÔÖÙÛÜÇÎÏ]', text):  # French accents
            return "fr"
        
        logger.info("Defaulting to English")
        return "en"

    def _process_chunk(self, chunk: Dict[str, Any], language: str) -> List[Dict]:
        """Process a chunk with improved model alternation and language support."""
        entities = []
        
        # Always run regex first for baseline detection
        regex_entities = self._process_with_regex(chunk["text"], language)
        entities.extend(regex_entities)
        
        # Determine which model to use
        if self.config.model_provider == "alternate":
            # Alternate between models based on chunk position
            if chunk.get("index", 0) % 2 == 0:
                groq_entities = self._retry_llm_call(self._process_chunk_with_groq, chunk, language)
                entities.extend(groq_entities)
                if not groq_entities and self.gemini_model:  # Fallback to Gemini if Groq fails
                    entities.extend(self._retry_llm_call(self._process_chunk_with_gemini, chunk, language))
            elif self.gemini_model:
                gemini_entities = self._retry_llm_call(self._process_chunk_with_gemini, chunk, language)
                entities.extend(gemini_entities)
                if not gemini_entities:  # Fallback to Groq if Gemini fails
                    entities.extend(self._retry_llm_call(self._process_chunk_with_groq, chunk, language))
        elif self.config.model_provider == "both":
            # Run both models and combine results
            groq_entities = self._retry_llm_call(self._process_chunk_with_groq, chunk, language)
            entities.extend(groq_entities)
            if self.gemini_model:
                entities.extend(self._retry_llm_call(self._process_chunk_with_gemini, chunk, language))
        else:
            # Use specified model or default to Groq
            if self.config.model_provider == "gemini" and self.gemini_model:
                entities.extend(self._retry_llm_call(self._process_chunk_with_gemini, chunk, language))
            else:
                entities.extend(self._retry_llm_call(self._process_chunk_with_groq, chunk, language))
        
        return entities

    def analyze_text(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text with improved language handling and performance tracking."""
        start_time = time.time()
        
        try:
            # Detect or validate language
            if language:
                language = self.LANGUAGE_MAPPING.get(language, language)
                if language not in ["en", "fr", "ar"]:
                    logger.warning(f"Unsupported language: {language}. Detecting automatically.")
                    language = self._detect_language(text)
            else:
                language = self._detect_language(text)
            
            logger.info(f"Processing text in {language}")
            
            # Process text in chunks
            chunks = self._chunk_text(text)
            all_entities = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk, chunk, language): chunk
                    for chunk in chunks
                }
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_entities = future.result()
                        all_entities.extend(chunk_entities)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
            
            # Merge and sort entities
            merged_entities = self._merge_entities(all_entities)
            merged_entities.sort(key=lambda x: x["start"])
            
            # Create redacted text
            redacted_text = self._redact_text(text, merged_entities)
            
            self.stats.processing_time = time.time() - start_time
            
            return {
                "success": True,
                "detected_entities": merged_entities,
                "redacted_text": redacted_text,
                "language": language,
                "statistics": {
                    "model_stats": {
                        "groq_calls": self.stats.groq_calls,
                        "gemini_calls": self.stats.gemini_calls,
                        "successful_calls": self.stats.successful_calls,
                        "failed_calls": self.stats.failed_calls,
                        "total_tokens": self.stats.total_tokens,
                        "total_retries": self.stats.total_retries,
                        "cache_hits": self.stats.cache_hits,
                        "processing_time": self.stats.processing_time
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "language": language if 'language' in locals() else None
            }

if __name__ == "__main__":
    config = LLMPIIConfig(
        model_provider="groq",  # Optimized for single provider
        max_workers=10,
        chunk_size=4000,
        overlap=0,
        min_confidence=0.3,
        max_retries=1,
        timeout=5
    )
    detector = LLMPIIDetector(config)
    
    test_texts = [
        "Martin Kobayashi's email is martin-kobayashi@gmail.org and phone number is 0394 627 259, address: 128 East Oak Street.",
        "Marie Dubois, email: marie.dubois@email.fr, tél: 01 23 45 67 89, adresse: 15 Rue de la Paix.",
        "محمد أحمد، البريد الإلكتروني: mohammed.ahmed@email.com، هاتف: ٠١٢٣٤٥٦٧٨٩، العنوان: شارع الملك فهد."
    ]

    for text in test_texts:
        logger.info(f"\nTesting text: {text}")
        result = detector.analyze_text(text)
        logger.info(f"Full result: {json.dumps(result, indent=2, ensure_ascii=False)}")