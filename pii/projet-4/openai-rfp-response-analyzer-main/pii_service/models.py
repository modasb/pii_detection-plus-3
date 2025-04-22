"""Data models for PII detection and analysis."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from datetime import datetime

@dataclass
class Entity:
    """Represents a detected PII entity."""
    type: str
    text: str
    start: int
    end: int
    score: float
    context: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class DetectionResult:
    """Represents the result of PII detection on a text."""
    original_text: str
    anonymized_text: str
    entities: List[Entity]
    language: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating PII detection performance."""
    true_positives: Dict[str, int]
    false_positives: Dict[str, int]
    false_negatives: Dict[str, int]
    partial_matches: Dict[str, List[Dict]]
    scores: Dict[str, List[float]]
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, and F1 score for each entity type."""
        metrics = {}
        for entity_type in set(self.true_positives.keys()) | set(self.false_positives.keys()) | set(self.false_negatives.keys()):
            tp = self.true_positives.get(entity_type, 0)
            fp = self.false_positives.get(entity_type, 0)
            fn = self.false_negatives.get(entity_type, 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        return metrics

@dataclass
class BatchResult:
    """Represents results from batch processing."""
    results: List[DetectionResult]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: float
    errors: Optional[List[Dict]] = None

@dataclass
class ModelConfig:
    """Configuration for NLP models."""
    model_name: str
    language: str
    confidence_threshold: float
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = False
    cache_dir: Optional[str] = None

@dataclass
class AnonymizationConfig:
    """Configuration for PII anonymization."""
    masking_char: str = "*"
    preserve_length: bool = True
    preserve_format: bool = True
    custom_masks: Optional[Dict[str, str]] = None
    preserve_last_digits: Optional[Dict[str, int]] = None 