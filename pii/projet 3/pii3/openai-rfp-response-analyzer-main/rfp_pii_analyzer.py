import requests
import json
import os
from pathlib import Path

class RFPPIIAnalyzer:
    """Analyzes RFP documents for PII using the PII Detection API."""
    
    def __init__(self, api_url="http://127.0.0.1:8000"):
        self.api_url = api_url
        self.verify_api_connection()
    
    def verify_api_connection(self):
        """Verify that the PII API is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/api/v1/health")
            if response.status_code == 200:
                print("✓ PII API is running and accessible")
                return True
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to PII API. "
                "Make sure the server is running at http://127.0.0.1:8000"
            )
    
    def analyze_text(self, text: str, language: str = "en") -> dict:
        """
        Analyze text for PII using the API.
        
        Args:
            text: The text to analyze
            language: Language code (default: "en")
            
        Returns:
            dict containing detected entities and anonymized text
        """
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/detect",
                json={
                    "text": text,
                    "language": language,
                    "anonymize": True
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error analyzing text: {e}")
            return None
    
    def analyze_file(self, file_path: str) -> dict:
        """
        Analyze a text file for PII.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            dict containing analysis results and statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = self.analyze_text(content)
            if result:
                return {
                    'filename': os.path.basename(file_path),
                    'total_entities': len(result['entities']),
                    'entities_by_type': self._count_entities_by_type(result['entities']),
                    'entities': result['entities'],
                    'anonymized_text': result['anonymized_text']
                }
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
        return None
    
    def _count_entities_by_type(self, entities):
        """Count entities by type."""
        counts = {}
        for entity in entities:
            counts[entity['type']] = counts.get(entity['type'], 0) + 1
        return counts
    
    def save_analysis(self, analysis: dict, output_dir: str):
        """
        Save analysis results to files.
        
        Args:
            analysis: Analysis results
            output_dir: Directory to save results
        """
        if not analysis:
            return
            
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save anonymized text
        anon_path = output_dir / f"{analysis['filename']}_anonymized.txt"
        with open(anon_path, 'w', encoding='utf-8') as f:
            f.write(analysis['anonymized_text'])
        
        # Save analysis report
        report_path = output_dir / f"{analysis['filename']}_pii_report.json"
        report = {
            'filename': analysis['filename'],
            'statistics': {
                'total_entities': analysis['total_entities'],
                'entities_by_type': analysis['entities_by_type']
            },
            'entities': analysis['entities']
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nAnalysis saved:")
        print(f"- Anonymized text: {anon_path}")
        print(f"- PII report: {report_path}")

def main():
    """Main function to demonstrate RFP PII analysis."""
    analyzer = RFPPIIAnalyzer()
    
    # Example RFP text for testing
    test_text = """
    REQUEST FOR PROPOSAL (RFP)
    
    Contact Information:
    Project Manager: John Smith
    Email: john.smith@company.com
    Phone: (555) 123-4567
    
    Company Details:
    ABC Corporation
    Tax ID: 123-45-6789
    Account: 4532-0158-7845-9623
    
    Project Requirements:
    1. Implementation timeline
    2. Cost breakdown
    3. Technical specifications
    """
    
    print("\nAnalyzing sample RFP text...")
    result = analyzer.analyze_text(test_text)
    
    if result:
        print("\nDetected PII entities:")
        for entity in result['entities']:
            print(f"- {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
        
        print("\nAnonymized text:")
        print(result['anonymized_text'])
    
    # Save test results
    analysis = {
        'filename': 'sample_rfp',
        'total_entities': len(result['entities']),
        'entities_by_type': analyzer._count_entities_by_type(result['entities']),
        'entities': result['entities'],
        'anonymized_text': result['anonymized_text']
    }
    analyzer.save_analysis(analysis, 'pii_analysis_results')

if __name__ == "__main__":
    main() 