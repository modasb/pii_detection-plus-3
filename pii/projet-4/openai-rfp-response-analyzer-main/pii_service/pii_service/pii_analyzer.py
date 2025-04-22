from typing import Dict, List

class PIIAnalyzer:
    """Analyzes detected PII entities and provides risk assessment."""
    
    def __init__(self):
        self.risk_levels = {
            'EMAIL': 'MEDIUM',
            'PHONE_NUMBER': 'HIGH',
            'SSN': 'CRITICAL',
            'CREDIT_CARD': 'CRITICAL',
            'BANK_ACCOUNT': 'CRITICAL'
        }
    
    def analyze_pii(self, entities: List[Dict]) -> Dict:
        """
        Analyze detected PII entities.
        
        Args:
            entities: List of detected PII entities
            
        Returns:
            Dict containing analysis results
        """
        if not entities:
            return {
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
            'risk_level': risk_level,
            'recommendations': self._get_recommendations(risk_level, pii_counts)
        }
    
    def _assess_risk_level(self, pii_counts: Dict[str, int]) -> str:
        """Assess overall risk level based on PII types and counts."""
        max_risk = 'LOW'
        for pii_type, count in pii_counts.items():
            risk = self.risk_levels.get(pii_type, 'MEDIUM')
            if risk == 'CRITICAL':
                return 'CRITICAL'
            elif risk == 'HIGH' and max_risk != 'CRITICAL':
                max_risk = 'HIGH'
            elif risk == 'MEDIUM' and max_risk == 'LOW':
                max_risk = 'MEDIUM'
        return max_risk
    
    def _get_recommendations(self, risk_level: str, pii_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on risk level and PII types."""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.append('CRITICAL: Document contains sensitive financial or identity information.')
            recommendations.append('Immediate action required to remove or redact sensitive information.')
        elif risk_level == 'HIGH':
            recommendations.append('HIGH: Document contains personal contact information.')
            recommendations.append('Consider redacting or anonymizing personal information.')
        elif risk_level == 'MEDIUM':
            recommendations.append('MEDIUM: Document contains some personal information.')
            recommendations.append('Review and consider if information needs to be shared.')
        else:
            recommendations.append('LOW: No sensitive information detected.')
        
        # Add specific recommendations for each PII type
        for pii_type, count in pii_counts.items():
            recommendations.append(f'Found {count} {pii_type}(s).')
        
        return recommendations
    
    def generate_summary(self, entities: List[Dict]) -> Dict:
        """
        Generate a summary of PII findings.
        
        Args:
            entities: List of detected PII entities
            
        Returns:
            Dict containing summary information
        """
        analysis = self.analyze_pii(entities)
        return {
            'risk_level': analysis['risk_level'],
            'recommendations': analysis['recommendations']
        } 