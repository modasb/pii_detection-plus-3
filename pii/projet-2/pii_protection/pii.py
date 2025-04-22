import logging
import re
import threading
from typing import List, Dict, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import arabic_reshaper
from bidi.algorithm import get_display
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
import spacy
import torch
import time
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIConfig:
    MODELS = {
        "en": {"spacy": "en_core_web_sm", "ner": "dslim/bert-base-NER"},
        "fr": {"spacy": "fr_core_news_sm", "ner": "Jean-Baptiste/camembert-ner"},
        "ar": {"spacy": "xx_ent_wiki_sm", "ner": "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"},
    }
    CONFIDENCE_THRESHOLDS = {"en": 0.6, "fr": 0.5, "ar": 0.7}
    REGEX_PATTERNS = {
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "PHONE_NUMBER": re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'),
        "ADDRESS": re.compile(r'\b\d{1,5}\s+[A-Za-z0-9\s]+(?:Street|Drive|Avenue|St\.|Dr\.|Ave\.)\b', re.IGNORECASE),
    }
    LANGUAGE_REGEX_PATTERNS = {
        "en": {"AU_PHONE": re.compile(r'\b0\d{3}\s\d{3}\s\d{3}\b')},
        "fr": {"FR_PHONE": re.compile(r'\b0\d{1}\s?(?:\d{2}\s?){4}\b')},
        "ar": {"AR_PHONE": re.compile(r'\b(?:\+\d{3}|0)\d{8,10}\b'), "AR_NAME": re.compile(r'(?:[\u0600-\u06FF]+\s+){1,3}[\u0600-\u06FF]+')},
    }
    MASKING_STRATEGIES = {
        "default": lambda x: "[REDACTED]",
        "EMAIL": lambda x: x.split('@')[0][:2] + "***@" + x.split('@')[1],
        "PHONE_NUMBER": lambda x: x[:4] + "*" * (len(x) - 4),
        "ADDRESS": lambda x: "[ADDRESS]",
        "AR_NAME": lambda x: "[ARABIC_NAME]",
    }

class PIIProtectionLayer:
    def __init__(self, config: PIIConfig = PIIConfig(), force_gpu: bool = False, redaction_strategy: str = "mask", confidence_threshold: Optional[Dict[str, float]] = None):
        logger.info("Initializing PII Protection Layer...")
        self.config = config
        self.force_gpu = force_gpu
        
        # Override confidence thresholds if provided
        if confidence_threshold:
            for lang, threshold in confidence_threshold.items():
                self.config.CONFIDENCE_THRESHOLDS[lang] = threshold
                
        # Set redaction strategy if provided (for backward compatibility)
        self.redaction_strategy = redaction_strategy
        
        self.executor = ThreadPoolExecutor(max_workers=4)  # Increased workers for better parallelism
        self._nlp_models: Dict[str, Any] = {}
        self._ner_models: Dict[str, Any] = {}
        self._analyzer: Optional[AnalyzerEngine] = None
        self._anonymizer: Optional[AnonymizerEngine] = None
        self._lock = threading.Lock()
        self.audit_log: List[Dict[str, Any]] = []
        
        # Check GPU availability at initialization
        self._check_gpu_availability()

    def _check_gpu_availability(self) -> None:
        """Check if GPU is available and print detailed information."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU is available! Found {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
                logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
                
            # Set default device to the first GPU
            torch.cuda.set_device(0)
            logger.info(f"Default GPU set to: {torch.cuda.current_device()}")
            
            # Try to allocate a small tensor to verify GPU works
            try:
                test_tensor = torch.zeros(1000, 1000, device=0)
                del test_tensor  # Free memory
                logger.info("Successfully allocated test tensor on GPU")
            except Exception as e:
                logger.error(f"Error allocating tensor on GPU: {e}")
        else:
            logger.warning("No GPU available. Using CPU for processing.")
            logger.info("To enable GPU, ensure PyTorch is installed with CUDA support")

    def _load_model(self, lang: str, model_type: str) -> Optional[Any]:
        with self._lock:
            target = self._nlp_models if model_type == "spacy" else self._ner_models
            if lang not in target:
                try:
                    if model_type == "spacy":
                        target[lang] = spacy.load(self.config.MODELS[lang]["spacy"], disable=["parser"])
                        logger.info(f"Loaded {model_type} model for {lang} on CPU")
                    else:
                        # Determine device based on availability and force_gpu setting
                        if self.force_gpu:
                            if torch.cuda.is_available():
                                device = 0  # Use first GPU
                                logger.info(f"Forcing GPU usage as requested: {torch.cuda.get_device_name(0)}")
                                # Set CUDA device explicitly
                                torch.cuda.set_device(device)
                            else:
                                logger.warning("force_gpu=True but CUDA is not available! Falling back to CPU")
                                device = -1
                        else:
                            # Use GPU if available, otherwise CPU
                            if torch.cuda.is_available():
                                logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
                                device = 0  # Use first GPU
                                # Set CUDA device explicitly
                                torch.cuda.set_device(device)
                            else:
                                logger.warning("CUDA is not available, falling back to CPU")
                                device = -1
                            
                        # Check transformers version to handle parameter differences
                        import transformers
                        transformers_version = transformers.__version__
                        logger.info(f"Transformers version: {transformers_version}")
                        
                        # Create pipeline with parameters compatible with the installed version
                        pipeline_kwargs = {
                            "model": self.config.MODELS[lang]["ner"],
                            "aggregation_strategy": "simple",
                            "device": device,
                        }
                        
                        # Add truncation parameter (compatible with newer versions)
                        if hasattr(pipeline, "PIPELINE_INIT_ARGS") and "truncation" in getattr(pipeline, "PIPELINE_INIT_ARGS").get("ner", []):
                            pipeline_kwargs["truncation"] = True
                        
                        # Add max_length parameter (compatible with newer versions)
                        if hasattr(pipeline, "PIPELINE_INIT_ARGS") and "max_length" in getattr(pipeline, "PIPELINE_INIT_ARGS").get("ner", []):
                            pipeline_kwargs["max_length"] = 512
                        
                        # Add batch_size parameter (compatible with newer versions)
                        if hasattr(pipeline, "PIPELINE_INIT_ARGS") and "batch_size" in getattr(pipeline, "PIPELINE_INIT_ARGS").get("ner", []):
                            pipeline_kwargs["batch_size"] = 1
                        
                        logger.info(f"Creating NER pipeline with parameters: {pipeline_kwargs}")
                        target[lang] = pipeline("ner", **pipeline_kwargs)
                        logger.info(f"Loaded {model_type} model for {lang} on {'GPU' if device == 0 else 'CPU'}")
                        
                        # Verify GPU usage
                        if device == 0:
                            # Try to run a small test to verify GPU usage
                            try:
                                test_text = "This is a test."
                                start_time = time.time()
                                _ = target[lang](test_text)
                                elapsed = time.time() - start_time
                                logger.info(f"GPU test inference took {elapsed:.4f} seconds")
                            except Exception as e:
                                logger.error(f"GPU test failed: {e}")
                except Exception as e:
                    logger.error(f"Failed to load {model_type} for {lang}: {e}")
                    target[lang] = None
            return target.get(lang)

    def _init_presidio(self) -> None:
        with self._lock:
            if self._analyzer is None and any(self._load_model(lang, "spacy") for lang in self.config.MODELS):
                models_config = {lang: {"lang_code": lang, "model": self._load_model(lang, "spacy")} for lang in self.config.MODELS if self._load_model(lang, "spacy")}
                nlp_engine = SpacyNlpEngine(models=models_config)
                self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                self._anonymizer = AnonymizerEngine()
                for lang, patterns in self.config.LANGUAGE_REGEX_PATTERNS.items():
                    for name, pattern in patterns.items():
                        self._analyzer.registry.add_recognizer(
                            PatternRecognizer(supported_language=lang, patterns=[Pattern(name=name, regex=pattern, score=0.9)])
                        )
                logger.info("Presidio initialized")

    def _detect_with_regex(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Run regex-based detection in a separate thread."""
        entities = []
        
        # Apply common regex patterns
        for name, pattern in self.config.REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append({
                    "type": name, 
                    "text": match.group(), 
                    "start": match.start(), 
                    "end": match.end(), 
                    "score": 1.0, 
                    "method": "regex"
                })
                
        # Apply language-specific regex patterns
        for name, pattern in self.config.LANGUAGE_REGEX_PATTERNS.get(language, {}).items():
            for match in pattern.finditer(text):
                entities.append({
                    "type": name, 
                    "text": match.group(), 
                    "start": match.start(), 
                    "end": match.end(), 
                    "score": 1.0, 
                    "method": f"regex_{language}"
                })
                
        # Arabic enhancement
        if language == "ar" and any('\u0600' <= c <= '\u06FF' for c in text):
            try:
                reshaped = arabic_reshaper.reshape(text)
                display_text = get_display(reshaped)
                for name, pattern in self.config.LANGUAGE_REGEX_PATTERNS["ar"].items():
                    for match in pattern.finditer(display_text):
                        entities.append({
                            "type": name, 
                            "text": match.group(), 
                            "start": match.start(), 
                            "end": match.end(), 
                            "score": 0.9, 
                            "method": "arabic_regex"
                        })
            except Exception as e:
                logger.warning(f"Arabic text reshaping failed: {e}")
                
        return entities

    def _detect_with_presidio(self, text: str, language: str, threshold: float) -> List[Dict[str, Any]]:
        """Run Presidio detection in a separate thread."""
        entities = []
        if self._analyzer or self._init_presidio():
            try:
                results = self._analyzer.analyze(text=text, language=language, score_threshold=threshold)
                entities.extend(
                    {"type": r.entity_type, "text": text[r.start:r.end], "start": r.start, "end": r.end, "score": float(r.score), "method": "presidio"}
                    for r in results
                )
            except Exception as e:
                logger.warning(f"Presidio failed: {e}")
        return entities

    def _detect_with_ner(self, text: str, language: str, threshold: float) -> List[Dict[str, Any]]:
        """Run NER detection with GPU optimization."""
        entities = []
        ner = self._load_model(language, "ner")
        if ner:
            try:
                # For longer texts, split into chunks to avoid CUDA memory issues
                if len(text) > 500 and torch.cuda.is_available():
                    logger.info(f"Processing long text ({len(text)} chars) in chunks")
                    # Split text into chunks of approximately 500 characters
                    # Try to split at sentence boundaries when possible
                    chunks = []
                    sentences = text.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 500:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Process each chunk
                    all_results = []
                    for i, chunk in enumerate(chunks):
                        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                        try:
                            chunk_results = ner(chunk)
                            
                            # Adjust entity positions based on chunk position in original text
                            offset = text.find(chunk)
                            if offset >= 0:  # Only adjust if chunk is found in original text
                                for result in chunk_results:
                                    result["start"] += offset
                                    result["end"] += offset
                            
                            all_results.extend(chunk_results)
                        except Exception as chunk_e:
                            logger.warning(f"Error processing chunk {i+1}: {str(chunk_e)}")
                    
                    results = all_results
                else:
                    # Process the entire text at once
                    results = ner(text)
                
                # Filter and format results
                for e in results:
                    # Handle different result formats based on transformers version
                    entity_type = e.get("entity_group", e.get("entity", "UNKNOWN"))
                    entity_text = e.get("word", e.get("text", ""))
                    entity_score = e.get("score", 0.0)
                    entity_start = e.get("start", 0)
                    entity_end = e.get("end", 0)
                    
                    if entity_score >= threshold:
                        entities.append({
                            "type": entity_type,
                            "text": entity_text,
                            "start": entity_start,
                            "end": entity_end,
                            "score": entity_score,
                            "method": f"ner_{language}"
                        })
                
                # Log GPU memory usage after processing
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
                    logger.debug(f"GPU memory after NER: allocated={memory_allocated:.2f}MB, reserved={memory_reserved:.2f}MB")
                    
            except Exception as e:
                logger.warning(f"{language.capitalize()} NER error: {str(e)}")
                # If we get CUDA out of memory, try again with CPU
                if "CUDA out of memory" in str(e) and torch.cuda.is_available():
                    logger.warning("CUDA out of memory, falling back to CPU for this text")
                    try:
                        # Create a CPU version of the pipeline just for this text
                        import transformers
                        transformers_version = transformers.__version__
                        
                        # Create pipeline with parameters compatible with the installed version
                        pipeline_kwargs = {
                            "model": self.config.MODELS[language]["ner"],
                            "aggregation_strategy": "simple",
                            "device": -1,  # Force CPU
                        }
                        
                        # Add compatible parameters based on transformers version
                        if hasattr(pipeline, "PIPELINE_INIT_ARGS"):
                            pipeline_args = getattr(pipeline, "PIPELINE_INIT_ARGS").get("ner", [])
                            if "truncation" in pipeline_args:
                                pipeline_kwargs["truncation"] = True
                            if "max_length" in pipeline_args:
                                pipeline_kwargs["max_length"] = 512
                        
                        cpu_ner = pipeline("ner", **pipeline_kwargs)
                        results = cpu_ner(text)
                        
                        # Filter and format results
                        for e in results:
                            # Handle different result formats based on transformers version
                            entity_type = e.get("entity_group", e.get("entity", "UNKNOWN"))
                            entity_text = e.get("word", e.get("text", ""))
                            entity_score = e.get("score", 0.0)
                            entity_start = e.get("start", 0)
                            entity_end = e.get("end", 0)
                            
                            if entity_score >= threshold:
                                entities.append({
                                    "type": entity_type,
                                    "text": entity_text,
                                    "start": entity_start,
                                    "end": entity_end,
                                    "score": entity_score,
                                    "method": f"ner_{language}_cpu_fallback"
                                })
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
        return entities

    @lru_cache(maxsize=500)
    def detect_pii(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Detect PII entities using multiple methods in parallel."""
        threshold = self.config.CONFIDENCE_THRESHOLDS.get(language, 0.5)
        
        # Use ThreadPoolExecutor to run detection methods in parallel
        futures = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit detection tasks
            futures.append(executor.submit(self._detect_with_regex, text, language))
            futures.append(executor.submit(self._detect_with_presidio, text, language, threshold))
            futures.append(executor.submit(self._detect_with_ner, text, language, threshold))
        
        # Collect results
        entities = []
        for future in futures:
            try:
                entities.extend(future.result())
            except Exception as e:
                logger.error(f"Error in detection thread: {e}")
        
        return self._merge_entities(entities)

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return []
        entities.sort(key=lambda x: (x["start"], -x["score"]))
        merged = []
        current = None
        for entity in entities:
            if not current:
                current = entity
            elif entity["start"] <= current["end"]:
                current["end"] = max(current["end"], entity["end"])
                if entity["score"] > current["score"]:
                    current.update({k: v for k, v in entity.items() if k != "end"})
            else:
                merged.append(current)
                current = entity
        if current:
            merged.append(current)
        return merged

    def process_document(self, text: str, language: str) -> Dict[str, Any]:
        """Process a document to detect and mask PII."""
        start_time = time.time()
        try:
            entities = self.detect_pii(text, language)
            masked_text = self.mask_pii(text, entities)
            elapsed = time.time() - start_time
            
            # Group entities by type and method for statistics
            entity_stats = {}
            entity_types = set()
            entity_methods = set()
            
            for entity in entities:
                entity_type = entity["type"]
                method = entity["method"]
                
                # Track unique entity types and methods
                entity_types.add(entity_type)
                entity_methods.add(method)
                
                # Create combined key for statistics
                key = f"{entity_type}_{method}"
                if key not in entity_stats:
                    entity_stats[key] = 0
                entity_stats[key] += 1
            
            return {
                'original_text': text,
                'masked_text': masked_text,
                'detected_entities': entities,
                'statistics': {
                    'total_entities': len(entities),
                    'entity_types': entity_stats,
                    'unique_types': list(entity_types),
                    'detection_methods': list(entity_methods)
                },
                'language': language,
                'processing_time': elapsed,
                'success': True
            }
        except Exception as e:
            logger.error(f"Processing failed: {e}")
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

    def mask_pii(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Mask detected PII entities using appropriate strategies."""
        if not entities:
            return text
        entities.sort(key=lambda x: x["start"], reverse=True)
        for entity in entities:
            strategy = self.config.MASKING_STRATEGIES.get(entity["type"], self.config.MASKING_STRATEGIES["default"])
            text = text[:entity["start"]] + strategy(entity["text"]) + text[entity["end"]:]
        return text
        
    def batch_process(self, texts: List[str], language: str, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel using ThreadPoolExecutor.
        
        Args:
            texts: List of text documents to process
            language: Language code for the documents
            max_workers: Maximum number of worker threads (defaults to auto-configuration)
        
        Returns:
            List of processing results for each document
        """
        # Determine optimal number of workers
        if max_workers is None:
            # Start with CPU-based calculation
            cpu_count = os.cpu_count() or 4
            
            # For GPU processing, we need to be more conservative to avoid CUDA memory issues
            if self.force_gpu and torch.cuda.is_available():
                # Get GPU memory info
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU memory: {gpu_memory_gb:.2f} GB")
                
                # Adjust workers based on GPU memory
                # Rule of thumb: 1GB per worker for small texts, less for larger texts
                if gpu_memory_gb < 4:  # Small GPUs (like yours with 4GB)
                    suggested_workers = min(4, cpu_count)
                elif gpu_memory_gb < 8:  # Mid-range GPUs
                    suggested_workers = min(6, cpu_count)
                else:  # High-end GPUs
                    suggested_workers = min(8, cpu_count)
                
                # Adjust based on text length
                avg_text_length = sum(len(text) for text in texts) / len(texts) if texts else 0
                if avg_text_length > 1000:
                    # Reduce workers for longer texts to avoid CUDA OOM
                    suggested_workers = max(2, suggested_workers - 2)
                
                logger.info(f"Using {suggested_workers} workers for GPU-accelerated batch processing (avg text length: {avg_text_length:.1f} chars)")
            else:
                # For CPU-only processing, we can use more workers
                suggested_workers = max(2, min(cpu_count * 2, 16))
                logger.info(f"Using {suggested_workers} workers for CPU-only batch processing")
            
            max_workers = min(suggested_workers, len(texts))
        
        # Process in parallel
        logger.info(f"Processing {len(texts)} documents with {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda text: self.process_document(text, language), texts))