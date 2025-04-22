#!/usr/bin/env python
"""
Evaluation script for the PII Protection System.
Tests accuracy, performance, and robustness across different languages and scenarios.
"""

import time
import json
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from pii_protection.pii import PIIProtectionLayer

class PIIEvaluator:
    """Evaluator for PII detection system."""
    
    def __init__(self):
        self.pii_layer = PIIProtectionLayer(
            confidence_threshold={
                'en': 0.6,
                'fr': 0.4,
                'ar': 0.7
            }
        )
        
        # Test cases with ground truth annotations
        self.test_cases = {
            'en': [
                {
                    'text': """
                    My name is John Smith and I work at Microsoft Corporation.
                    You can reach me at john.smith@microsoft.com or call 555-123-4567.
                    My credit card number is 4111-1111-1111-1111 and my SSN is 123-45-6789.
                    I live at 123 Main Street, New York, NY 10001.
                    """,
                    'expected_entities': [
                        {'type': 'PERSON', 'text': 'John Smith'},
                        {'type': 'ORGANIZATION', 'text': 'Microsoft Corporation'},
                        {'type': 'email', 'text': 'john.smith@microsoft.com'},
                        {'type': 'phone_number', 'text': '555-123-4567'},
                        {'type': 'credit_card', 'text': '4111-1111-1111-1111'},
                        {'type': 'us_ssn', 'text': '123-45-6789'},
                        {'type': 'LOCATION', 'text': 'New York'},
                        {'type': 'ADDRESS', 'text': '123 Main Street'}
                    ]
                }
            ],
            'fr': [
                {
                    'text': """
                    Je m'appelle Jean Dupont et je travaille chez Renault.
                    Mon numéro de téléphone est +33 6 12 34 56 78.
                    Mon adresse email est jean.dupont@example.fr.
                    Mon numéro SIRET est 123 456 789 12345 et mon numéro de TVA est FR12 345 678 912.
                    J'habite au 45 rue de la République, 75001 Paris.
                    """,
                    'expected_entities': [
                        {'type': 'PERSON', 'text': 'Jean Dupont'},
                        {'type': 'ORGANIZATION', 'text': 'Renault'},
                        {'type': 'fr_phone', 'text': '+33 6 12 34 56 78'},
                        {'type': 'email', 'text': 'jean.dupont@example.fr'},
                        {'type': 'fr_siret', 'text': '123 456 789 12345'},
                        {'type': 'fr_tva', 'text': 'FR12 345 678 912'},
                        {'type': 'fr_postcode', 'text': '75001'},
                        {'type': 'LOCATION', 'text': 'Paris'}
                    ]
                }
            ],
            'ar': [
                {
                    'text': """
                    اسمي محمد أحمد وأعمل في شركة الاتصالات السعودية.
                    رقم هاتفي هو 0512345678
                    بريدي الإلكتروني هو mohammed.ahmed@example.sa
                    رقم جواز سفري هو A1234567
                    أسكن في شارع الملك فهد، الرياض.
                    """,
                    'expected_entities': [
                        {'type': 'PERSON', 'text': 'محمد أحمد'},
                        {'type': 'ORGANIZATION', 'text': 'شركة الاتصالات السعودية'},
                        {'type': 'ar_phone', 'text': '0512345678'},
                        {'type': 'email', 'text': 'mohammed.ahmed@example.sa'},
                        {'type': 'ar_passport', 'text': 'A1234567'},
                        {'type': 'LOCATION', 'text': 'الرياض'}
                    ]
                }
            ]
        }

    def evaluate_detection(self, language: str) -> Dict:
        """Evaluate PII detection for a specific language."""
        results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'processing_time': [],
            'detected_types': [],
            'missed_types': [],
            'false_positives': []
        }
        
        for test_case in self.test_cases[language]:
            start_time = time.time()
            detection_result = self.pii_layer.process_document(
                test_case['text'].strip(),
                language=language
            )
            processing_time = time.time() - start_time
            
            # Calculate metrics
            detected_entities = detection_result['detected_entities']
            expected_entities = test_case['expected_entities']
            
            # Convert to sets for comparison
            detected_set = {(e['type'], e['text']) for e in detected_entities}
            expected_set = {(e['type'], e['text']) for e in expected_entities}
            
            # Calculate true positives, false positives, and false negatives
            true_positives = detected_set & expected_set
            false_positives = detected_set - expected_set
            false_negatives = expected_set - detected_set
            
            # Calculate metrics
            precision = len(true_positives) / len(detected_set) if detected_set else 0
            recall = len(true_positives) / len(expected_set) if expected_set else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store results
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['processing_time'].append(processing_time)
            
            # Track detected and missed entity types
            detected_types = {entity['type'] for entity in detected_entities}
            expected_types = {entity['type'] for entity in expected_entities}
            results['detected_types'].extend(list(detected_types - set(results['detected_types'])))
            results['missed_types'].extend(list(expected_types - detected_types - set(results['missed_types'])))
            
            # Track false positives
            results['false_positives'].extend([
                {'type': type_, 'text': text} 
                for type_, text in false_positives
            ])
        
        # Calculate averages
        results['avg_precision'] = sum(results['precision']) / len(results['precision'])
        results['avg_recall'] = sum(results['recall']) / len(results['recall'])
        results['avg_f1'] = sum(results['f1']) / len(results['f1'])
        results['avg_processing_time'] = sum(results['processing_time']) / len(results['processing_time'])
        
        return results

    def evaluate_masking(self, language: str) -> Dict:
        """Evaluate PII masking effectiveness."""
        results = {
            'masked_texts': [],
            'leak_count': 0,
            'mask_patterns': []
        }
        
        for test_case in self.test_cases[language]:
            detection_result = self.pii_layer.process_document(
                test_case['text'].strip(),
                language=language
            )
            
            masked_text = detection_result['masked_text']
            results['masked_texts'].append(masked_text)
            
            # Check for PII leaks
            for entity in test_case['expected_entities']:
                if entity['text'] in masked_text:
                    results['leak_count'] += 1
            
            # Track masking patterns used
            for entity in detection_result['detected_entities']:
                mask_start = masked_text[entity['start']:entity['end']]
                if mask_start not in results['mask_patterns']:
                    results['mask_patterns'].append(mask_start)
        
        return results

    def evaluate_performance(self, language: str, num_iterations: int = 5) -> Dict:
        """Evaluate performance characteristics."""
        results = {
            'processing_times': [],
            'entity_counts': [],
            'memory_usage': []
        }
        
        for test_case in self.test_cases[language]:
            text = test_case['text'].strip()
            
            # Run multiple iterations for stable measurements
            for _ in range(num_iterations):
                start_time = time.time()
                detection_result = self.pii_layer.process_document(text, language=language)
                processing_time = time.time() - start_time
                
                results['processing_times'].append(processing_time)
                results['entity_counts'].append(len(detection_result['detected_entities']))
        
        # Calculate statistics
        results['avg_processing_time'] = sum(results['processing_times']) / len(results['processing_times'])
        results['max_processing_time'] = max(results['processing_times'])
        results['min_processing_time'] = min(results['processing_times'])
        results['avg_entity_count'] = sum(results['entity_counts']) / len(results['entity_counts'])
        
        return results

    def run_evaluation(self) -> Dict:
        """Run complete evaluation across all languages."""
        evaluation_results = {}
        
        for language in self.test_cases.keys():
            print(f"\nEvaluating {language.upper()} language...")
            
            # Run all evaluations
            detection_results = self.evaluate_detection(language)
            masking_results = self.evaluate_masking(language)
            performance_results = self.evaluate_performance(language)
            
            # Convert sets to lists for JSON serialization
            detection_results['detected_types'] = list(set(detection_results['detected_types']))
            detection_results['missed_types'] = list(set(detection_results['missed_types']))
            
            # Combine results
            evaluation_results[language] = {
                'detection': detection_results,
                'masking': masking_results,
                'performance': performance_results
            }
            
            # Print summary
            print(f"\nResults for {language.upper()}:")
            print(f"Average Precision: {detection_results['avg_precision']:.2f}")
            print(f"Average Recall: {detection_results['avg_recall']:.2f}")
            print(f"Average F1 Score: {detection_results['avg_f1']:.2f}")
            print(f"Average Processing Time: {detection_results['avg_processing_time']:.3f} seconds")
            print(f"PII Leaks Found: {masking_results['leak_count']}")
            print(f"Detected Entity Types: {', '.join(detection_results['detected_types'])}")
            if detection_results['missed_types']:
                print(f"Missed Entity Types: {', '.join(detection_results['missed_types'])}")
        
        return evaluation_results

def generate_evaluation_report(results: Dict) -> pd.DataFrame:
    """Generate a pandas DataFrame with evaluation results."""
    report_data = []
    
    for language, lang_results in results.items():
        detection = lang_results['detection']
        masking = lang_results['masking']
        performance = lang_results['performance']
        
        row = {
            'Language': language.upper(),
            'Precision': f"{detection['avg_precision']:.2f}",
            'Recall': f"{detection['avg_recall']:.2f}",
            'F1 Score': f"{detection['avg_f1']:.2f}",
            'Avg Processing Time (s)': f"{performance['avg_processing_time']:.3f}",
            'PII Leaks': masking['leak_count'],
            'Detected Types': len(detection['detected_types']),
            'Missed Types': len(detection['missed_types']),
            'False Positives': len(detection['false_positives'])
        }
        report_data.append(row)
    
    return pd.DataFrame(report_data)

def main():
    """Run evaluation and display results."""
    evaluator = PIIEvaluator()
    results = evaluator.run_evaluation()
    
    # Generate and display report
    report_df = generate_evaluation_report(results)
    print("\nEvaluation Report:")
    print(report_df.to_string(index=False))
    
    # Save detailed results to JSON
    with open('pii_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nDetailed results saved to 'pii_evaluation_results.json'")

if __name__ == "__main__":
    main() 