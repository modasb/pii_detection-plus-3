import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pii_protection.pii import PIIProtectionLayer, PIIDetectionError


class TestPIIProtectionLayer(unittest.TestCase):
    """Test cases for the improved PII Protection Layer."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenAI API key for testing
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            self.pii_layer = PIIProtectionLayer()

    def test_initialization(self):
        """Test initialization of the PII Protection Layer."""
        self.assertIsNotNone(self.pii_layer.analyzer)
        self.assertIsNotNone(self.pii_layer.anonymizer)
        self.assertEqual(self.pii_layer.supported_languages, ['en', 'fr', 'ar'])
        self.assertEqual(self.pii_layer.redaction_strategy, 'mask')
        self.assertEqual(self.pii_layer.confidence_threshold, 0.5)

    def test_regex_detection(self):
        """Test regex-based PII detection."""
        # Test email detection
        text = "My email is test@example.com"
        entities = self.pii_layer.detect_pii(text)
        
        # Check if at least one entity was detected
        self.assertTrue(any(entity['type'] == 'email' for entity in entities))
        
        # Test phone number detection
        text = "My phone number is 555-123-4567"
        entities = self.pii_layer.detect_pii(text)
        self.assertTrue(any(entity['type'] == 'phone_number' for entity in entities))
        
        # Test credit card detection
        text = "My credit card is 4111-1111-1111-1111"
        entities = self.pii_layer.detect_pii(text)
        self.assertTrue(any(entity['type'] == 'credit_card' for entity in entities))

    def test_redaction_strategies(self):
        """Test different redaction strategies."""
        text = "My email is test@example.com"
        entities = [{'type': 'email', 'text': 'test@example.com', 'start': 12, 'end': 28, 'score': 1.0}]
        
        # Test mask strategy
        self.pii_layer.redaction_strategy = 'mask'
        redacted = self.pii_layer.redact_pii(text, entities, 'en')
        self.assertEqual(redacted, "My email is [REDACTED]")
        
        # Test type strategy
        self.pii_layer.redaction_strategy = 'type'
        redacted = self.pii_layer.redact_pii(text, entities, 'en')
        self.assertEqual(redacted, "My email is [EMAIL]")
        
        # Test hash strategy
        self.pii_layer.redaction_strategy = 'hash'
        redacted = self.pii_layer.redact_pii(text, entities, 'en')
        self.assertTrue(redacted.startswith("My email is [HASH:"))
        
        # Test partial strategy
        self.pii_layer.redaction_strategy = 'partial'
        redacted = self.pii_layer.redact_pii(text, entities, 'en')
        self.assertTrue("@example.com" in redacted)
        self.assertNotEqual(redacted, text)

    def test_analyze_text(self):
        """Test the analyze_text method."""
        text = "My name is John Doe and my email is john.doe@example.com"
        
        # Mock the detect_pii method to return predefined entities
        with patch.object(self.pii_layer, 'detect_pii') as mock_detect:
            mock_detect.return_value = [
                {'type': 'PERSON', 'text': 'John Doe', 'start': 11, 'end': 19, 'score': 0.9, 'method': 'presidio'},
                {'type': 'email', 'text': 'john.doe@example.com', 'start': 33, 'end': 54, 'score': 1.0, 'method': 'regex'}
            ]
            
            result = self.pii_layer.analyze_text(text)
            
            # Check the result structure
            self.assertIn('original_text', result)
            self.assertIn('redacted_text', result)
            self.assertIn('detected_entities', result)
            self.assertIn('pii_count', result)
            self.assertIn('pii_types', result)
            
            # Check values
            self.assertEqual(result['original_text'], text)
            self.assertEqual(result['pii_count'], 2)
            self.assertEqual(set(result['pii_types']), {'PERSON', 'email'})

    def test_batch_analyze(self):
        """Test batch analysis of multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock analyze_text to return a simple result
        with patch.object(self.pii_layer, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = {'redacted_text': 'Redacted', 'detected_entities': []}
            
            results = self.pii_layer.batch_analyze(texts)
            
            # Check that analyze_text was called for each text
            self.assertEqual(mock_analyze.call_count, 3)
            
            # Check results
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertEqual(result['redacted_text'], 'Redacted')

    def test_error_handling(self):
        """Test error handling in the analyze_text method."""
        text = "Test text"
        
        # Mock detect_pii to raise an exception
        with patch.object(self.pii_layer, 'detect_pii') as mock_detect:
            mock_detect.side_effect = Exception("Test error")
            
            result = self.pii_layer.analyze_text(text)
            
            # Check that the method handled the error gracefully
            self.assertEqual(result['redacted_text'], text)
            self.assertEqual(result['detected_entities'], [])
            self.assertIn('error', result)

    def test_overlapping_entities(self):
        """Test handling of overlapping entities."""
        # Create overlapping entities
        entities = [
            {'type': 'PERSON', 'text': 'John Doe', 'start': 0, 'end': 8, 'score': 0.8},
            {'type': 'NAME', 'text': 'John', 'start': 0, 'end': 4, 'score': 0.9},
            {'type': 'LAST_NAME', 'text': 'Doe', 'start': 5, 'end': 8, 'score': 0.7}
        ]
        
        # Remove overlapping entities
        filtered = self.pii_layer._remove_overlapping_entities(entities)
        
        # Should keep only the highest scoring entity for each position
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['type'], 'NAME')


if __name__ == '__main__':
    unittest.main() 