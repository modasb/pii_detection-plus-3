import requests
import json

def test_api():
    """Test the PII Detection API endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    health_response = requests.get(f"{base_url}/api/v1/health")
    print(f"Health status: {health_response.json()}")
    
    # Test supported types endpoint
    print("\nTesting supported types endpoint...")
    types_response = requests.get(f"{base_url}/api/v1/supported-types")
    print("Supported PII types:", json.dumps(types_response.json(), indent=2))
    
    # Test PII detection
    print("\nTesting PII detection...")
    test_cases = [
        {
            "text": "My credit card is 4532-0158-7845-9623 and my email is john@example.com",
            "language": "en",
            "anonymize": True
        },
        {
            "text": "Call me at (123) 456-7890 or send mail to 123 Main St, City, 12345",
            "language": "en",
            "anonymize": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input text: {test_case['text']}")
        
        response = requests.post(
            f"{base_url}/api/v1/detect",
            json=test_case
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nDetected entities:")
            for entity in result['entities']:
                print(f"  - Type: {entity['type']}")
                print(f"    Value: {entity['value']}")
                print(f"    Confidence: {entity['confidence']:.2f}")
            print(f"\nAnonymized text: {result['anonymized_text']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    print("Testing PII Detection API...")
    try:
        test_api()
        print("\nAPI testing completed!")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Make sure the server is running at http://127.0.0.1:8000") 