from typing import Dict, List, Optional
import logging
from .pii_detector import PIIDetector

logger = logging.getLogger(__name__)

class PIIService:
    """Service class for PII detection in the RFP application."""
    
    def __init__(self):
        self.detector = PIIDetector()
    
    def detect_pii(self, content: str, language: str = "en") -> Dict:
        """
        Detect PII in the given content.
        
        Args:
            content: The text content to analyze
            language: Language code (default: "en")
            
        Returns:
            Dict containing detected PII entities and anonymized text
        """
        try:
            return self.detector.detect_pii(content, language)
        except Exception as e:
            logger.error(f"Error detecting PII: {e}")
            return {
                'entities': [],
                'anonymized_text': content,
                'error': str(e)
            }
    
    def analyze_rfp_content(self, content: str, language: str = "en") -> Dict:
        """
        Analyze RFP content for PII and return both detected entities and anonymized text.
        
        Args:
            content: The RFP text content to analyze
            language: Language code (default: "en")
            
        Returns:
            Dict containing detected PII entities and anonymized text
        """
        try:
            result = self.detector.detect_and_anonymize(content, language)
            return {
                'has_pii': len(result['entities']) > 0,
                'pii_count': len(result['entities']),
                'entities': result['entities'],
                'anonymized_text': result['anonymized_text']
            }
        except Exception as e:
            logger.error(f"Error analyzing RFP content for PII: {e}")
            return {
                'has_pii': False,
                'pii_count': 0,
                'entities': [],
                'anonymized_text': content
            }
    
    def get_pii_summary(self, entities: List[Dict]) -> Dict:
        """
        Generate a summary of detected PII entities.
        
        Args:
            entities: List of detected PII entities
            
        Returns:
            Dict containing PII statistics and risk assessment
        """
        if not entities:
            return {
                'total_pii': 0,
                'types_found': [],
                'risk_level': 'LOW',
                'recommendations': ['No PII detected in the document.']
            }
        
        # Count PII by type
        pii_counts = {}
        for entity in entities:
            pii_type = entity['type']
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
        
        # Assess risk level
        risk_level = self._assess_risk_level(pii_counts)
        
        return {
            'total_pii': len(entities),
            'types_found': list(pii_counts.keys()),
            'type_counts': pii_counts,
            'risk_level': risk_level,
            'recommendations': self._get_recommendations(risk_level, pii_counts)
        }
    
    def _assess_risk_level(self, pii_counts: Dict[str, int]) -> str:
        """Assess the risk level based on PII types and counts."""
        high_risk_types = {'CREDIT_CARD', 'SSN', 'BANK_ACCOUNT'}
        medium_risk_types = {'EMAIL', 'PHONE_NUMBER'}
        
        total_pii = sum(pii_counts.values())
        has_high_risk = any(pii_type in high_risk_types for pii_type in pii_counts)
        has_medium_risk = any(pii_type in medium_risk_types for pii_type in pii_counts)
        
        if has_high_risk or total_pii > 5:
            return 'HIGH'
        elif has_medium_risk or total_pii > 2:
            return 'MEDIUM'
        return 'LOW'
    
    def _get_recommendations(self, risk_level: str, pii_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on risk level and PII types."""
        recommendations = []
        
        if risk_level == 'HIGH':
            recommendations.extend([
                'Consider removing or redacting sensitive information.',
                'Review data handling policies and procedures.',
                'Ensure secure transmission and storage of this document.'
            ])
        
        if 'CREDIT_CARD' in pii_counts:
            recommendations.append('Remove or mask credit card numbers.')
        if 'SSN' in pii_counts:
            recommendations.append('Remove or mask Social Security Numbers.')
        if 'EMAIL' in pii_counts:
            recommendations.append('Consider using generic contact information.')
        
        if not recommendations:
            recommendations.append('Continue monitoring for sensitive information.')
        
        return recommendations 