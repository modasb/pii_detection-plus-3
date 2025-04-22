#!/usr/bin/env python
"""
Example script demonstrating the usage of the PII protection system.
"""

from pii_protection.pii import PIIProtectionLayer
import json

def print_detection_results(result, language):
    """Print PII detection results in a readable format."""
    print(f"\n=== Results for {language.upper()} ===")
    print(f"\nOriginal text:\n{result['original_text']}")
    print(f"\nMasked text:\n{result['masked_text']}")
    
    if result['detected_entities']:
        print("\nDetected PII entities:")
        for entity in result['detected_entities']:
            print(f"- {entity['type']} ({entity['method']}): {entity['text']} (confidence: {entity['score']:.2f})")
    
    print("\nStatistics:")
    print(f"Total entities found: {result['statistics']['total_entities']}")
    print("Entity types and methods:")
    for entity_type, count in result['statistics']['entity_types'].items():
        print(f"- {entity_type}: {count}")
    
    print("\n" + "="*80)

def main():
    # Initialize the PII protection layer
    pii_layer = PIIProtectionLayer(
        redaction_strategy="mask",
        confidence_threshold={
            'en': 0.6,
            'fr': 0.4,
            'ar': 0.7
        }
    )
    
    print("=== PII Protection Demo ===\n")
    
    # Example texts in different languages
    texts = {
        "en": """
        My name is John Smith and I work at Microsoft Corporation.
        You can reach me at john.smith@microsoft.com or call 555-123-4567.
        My credit card number is 4111-1111-1111-1111 and my SSN is 123-45-6789.
        I live at 123 Main Street, New York, NY 10001.
        """,
        
        "fr": """
        Je m'appelle Jean Dupont et je travaille chez Renault.
        Mon numéro de téléphone est +33 6 12 34 56 78.
        Mon adresse email est jean.dupont@example.fr.
        Mon numéro SIRET est 123 456 789 12345 et mon numéro de TVA est FR12 345 678 912.
        J'habite au 45 rue de la République, 75001 Paris.
        """,
        
        "ar": """
        اسمي محمد أحمد وأعمل في شركة الاتصالات السعودية.
        رقم هاتفي هو 0512345678
        بريدي الإلكتروني هو mohammed.ahmed@example.sa
        رقم جواز سفري هو A1234567
        أسكن في شارع الملك فهد، الرياض.
        """
    }
    
    # Process each language
    for language, text in texts.items():
        try:
            # Process the text
            result = pii_layer.process_document(text.strip(), language=language)
            print_detection_results(result, language)
            
        except Exception as e:
            print(f"\nError processing {language} text: {str(e)}")
    
    # Demonstrate batch processing
    print("\n=== Batch Processing Demo ===")
    
    batch_texts = [
        "John Smith works at Apple Inc.",
        "Sarah Johnson's phone is 555-987-6543",
        "Contact support@company.com for help"
    ]
    
    batch_results = pii_layer.batch_process(batch_texts, language="en")
    
    print("\nBatch processing results:")
    for i, result in enumerate(batch_results, 1):
        print(f"\nDocument {i}:")
        print(f"Original: {result['original_text']}")
        print(f"Masked: {result['masked_text']}")
        print(f"Entities found: {result['statistics']['total_entities']}")

if __name__ == "__main__":
    main() 