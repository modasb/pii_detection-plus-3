from app.core.pii_detector import PIIDetector

def test_pii_detection():
    # Initialize the detector
    detector = PIIDetector()
    
    # Test cases
    test_cases = [
        {
            "name": "Credit Card Test",
            "text": "My credit card number is 4532-0158-7845-9623",
            "expected_type": "CREDIT_CARD"
        },
        {
            "name": "Email Test",
            "text": "Contact me at john.doe@example.com",
            "expected_type": "EMAIL"
        },
        {
            "name": "Phone Number Test",
            "text": "Call me at (123) 456-7890",
            "expected_type": "PHONE_NUMBER"
        },
        {
            "name": "Multiple PII Test",
            "text": "My email is alice@example.com and my SSN is 123-45-6789",
            "expected_types": ["EMAIL", "SSN"]
        }
    ]
    
    # Run tests
    for test_case in test_cases:
        print(f"\nRunning test: {test_case['name']}")
        result = detector.detect_and_anonymize(test_case["text"])
        
        print(f"Input text: {test_case['text']}")
        print(f"Anonymized text: {result['anonymized_text']}")
        print("Detected entities:")
        for entity in result['entities']:
            print(f"  - Type: {entity['type']}")
            print(f"    Value: {entity['value']}")
            print(f"    Confidence: {entity['confidence']:.2f}")

if __name__ == "__main__":
    print("Starting PII Detection Tests...")
    test_pii_detection()
    print("\nTests completed!") 