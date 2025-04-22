from django.test import TestCase, Client
from django.urls import reverse
from .pii import PIIProtectionLayer
import time
import json

class PIIProtectionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.pii_layer = PIIProtectionLayer()
        self.api_url = reverse('pii-detect')  # Make sure this matches your URL name

    def test_pii_detection(self):
        """Test different types of PII detection"""
        test_cases = [
            # Personal Information
            {
                "input": "My name is John Smith and I live in New York",
                "expected_entities": ["PERSON", "LOCATION"]
            },
            # Contact Information
            {
                "input": "Contact me at john.smith@example.com",
                "expected_entities": ["EMAIL_ADDRESS"]
            },
            # Financial Information - Using a valid credit card format
            {
                "input": "Credit card: 4111-1111-1111-1111",
                "expected_entities": ["CREDIT_CARD"]
            },
            # SSN Information
            {
                "input": "SSN: 123-45-6789",
                "expected_entities": ["US_SSN"]
            }
        ]

        for case in test_cases:
            redacted_text, entities = self.pii_layer.detect_and_redact(case["input"])
            detected_types = [e["entity_type"] for e in entities]
            
            # Check if all expected entities were found
            for expected in case["expected_entities"]:
                self.assertIn(expected, detected_types)
            
            # Check if PII is actually redacted
            self.assertNotIn("John Smith", redacted_text)
            self.assertNotIn("123-45-6789", redacted_text)

    def test_api_endpoints(self):
        """Test API functionality"""
        test_payloads = [
            {
                "text": "Contact me at john.smith@example.com",  # Guaranteed to detect email
                "service": "openai"
            },
            {
                "text": "SSN: 123-45-6789",  # Guaranteed to detect SSN
                "service": "openai"
            }
        ]

        for payload in test_payloads:
            response = self.client.post(
                self.api_url,
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('processed_text', data)
            self.assertIn('pii_detected', data)
            self.assertTrue(data['pii_detected'])

    def test_error_handling(self):
        """Test error cases"""
        error_cases = [
            {},  # Empty payload
            {"text": ""},  # Empty text
            {"service": "openai"},  # Missing text
            {"text": "Test", "service": "invalid_service"},  # Invalid service
        ]

        for payload in error_cases:
            response = self.client.post(
                self.api_url,
                data=json.dumps(payload),
                content_type='application/json'
            )
            self.assertNotEqual(response.status_code, 200)

    def test_performance(self):
        """Test performance with different text lengths"""
        texts = [
            "Short text with email test@example.com",  # Short
            "Medium text " * 100 + "with email test@example.com",  # Medium
            "Long text " * 1000 + "with email test@example.com"  # Long
        ]

        for text in texts:
            start_time = time.time()
            redacted_text, entities = self.pii_layer.detect_and_redact(text)
            duration = time.time() - start_time

            # Basic performance assertions
            self.assertLess(duration, 5.0)  # Should process in less than 5 seconds
            self.assertGreater(len(entities), 0)  # Should find at least one entity
