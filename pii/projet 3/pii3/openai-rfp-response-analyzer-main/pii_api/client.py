import requests
import json

def test_pii_api():
    # API endpoint
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "text": "My credit card is 4532-0158-7845-9623 and my email is john@example.com",
            "language": "en"
        },
        {
            "text": "Call me at (123) 456-7890 or send mail to 123 Main St, City, 12345",
            "language": "en"
        }
    ]
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/api/v1/health")
        print(f"\nHealth check status: {health_response.json()['status']}")
    except requests.exceptions.ConnectionError:
        print("Error: API server is not running. Please start the server first.")
        return
    
    # Test PII detection
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input text: {test_case['text']}")
        
        try:
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
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")

if __name__ == "__main__":
    print("Testing PII Detection API...")
    test_pii_api()
    print("\nAPI testing completed!") 