from typing import Dict, List, Optional
import logging
from .pii_detector import PIIDetector
from .pii_analyzer import PIIAnalyzer

logger = logging.getLogger(__name__)

class PIIService:
    """Main service class that coordinates PII detection and analysis."""
    
    def __init__(self):
        """Initialize the PII service."""
        self.detector = PIIDetector()
        self.analyzer = PIIAnalyzer()
    
    def detect_pii(self, content: str) -> dict:
        """Detect and analyze PII in the given content.
        
        Args:
            content: The text content to analyze.
            
        Returns:
            Dict containing detection and analysis results.
        """
        # Detect PII
        detection_result = self.detector.detect_pii(content)
        
        # Analyze PII
        analysis_result = self.analyzer.analyze_pii(detection_result)
        
        return {
            "detection": detection_result,
            "analysis": analysis_result
        }
    
    def analyze_rfp_content(self, content: str, language: str = 'en') -> Dict:
        """
        Analyze RFP content for PII and provide detailed analysis.
        
        Args:
            content (str): The RFP content to analyze
            language (str): Language code (default: 'en')
            
        Returns:
            Dict: Analysis results including PII entities and risk assessment
        """
        # Detect PII
        detection_result = self.detector.detect_pii(content, language)
        
        # Analyze the detected PII
        analysis_result = self.analyzer.analyze_pii(detection_result['entities'])
        
        return {
            'has_pii': len(detection_result['entities']) > 0,
            'pii_count': len(detection_result['entities']),
            'entities': detection_result['entities'],
            'anonymized_text': detection_result['anonymized_text'],
            **analysis_result
        }
    
    def get_pii_summary(self, entities: List[Dict]) -> Dict:
        """
        Generate a summary of PII findings.
        
        Args:
            entities (List[Dict]): List of detected PII entities
            
        Returns:
            Dict: Summary including risk level and recommendations
        """
        return self.analyzer.generate_summary(entities)
    
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