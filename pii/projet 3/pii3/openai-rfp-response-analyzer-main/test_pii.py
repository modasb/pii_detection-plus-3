import os
import sys
import json
import csv
import numpy as np
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from langdetect import detect
from tqdm import tqdm
from pii_service.pii_detector import MultilingualPIIDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pii_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Class for tracking and calculating evaluation metrics."""
    true_positives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_positives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_negatives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    score_differences: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def update(self, entity_type: str, is_true_positive: bool, predicted_score: float = None, true_score: float = None):
        """Update metrics for a single prediction."""
        if is_true_positive:
            self.true_positives[entity_type] += 1
            if predicted_score is not None and true_score is not None:
                self.score_differences[entity_type].append(abs(predicted_score - true_score))
        else:
            self.false_positives[entity_type] += 1
    
    def add_false_negative(self, entity_type: str):
        """Add a false negative for the given entity type."""
        self.false_negatives[entity_type] += 1
    
    def calculate_metrics(self) -> Dict:
        """Calculate precision, recall, F1, and MAE for all entity types."""
        metrics = {}
        for entity_type in set(list(self.true_positives.keys()) + 
                             list(self.false_positives.keys()) + 
                             list(self.false_negatives.keys())):
            tp = self.true_positives[entity_type]
            fp = self.false_positives[entity_type]
            fn = self.false_negatives[entity_type]
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            mae = np.mean(self.score_differences[entity_type]) if self.score_differences[entity_type] else 0.0
            
            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mae": mae,
                "support": tp + fn
            }
        
        # Calculate macro and micro averages
        metrics["macro_avg"] = {
            "precision": np.mean([m["precision"] for m in metrics.values()]),
            "recall": np.mean([m["recall"] for m in metrics.values()]),
            "f1": np.mean([m["f1"] for m in metrics.values()]),
            "mae": np.mean([m["mae"] for m in metrics.values()])
        }
        
        total_tp = sum(self.true_positives.values())
        total_fp = sum(self.false_positives.values())
        total_fn = sum(self.false_negatives.values())
        
        metrics["micro_avg"] = {
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_tp + total_fn, 1),
            "f1": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
            "mae": np.mean([x for v in self.score_differences.values() for x in v]) if any(self.score_differences.values()) else 0.0
        }
        
        return metrics

class PIITestRunner:
    """Class for running PII detection tests with enhanced features."""
    
    def __init__(self, detector, config_path: Optional[str] = None):
        self.detector = detector
        self.metrics = EvaluationMetrics()
        self.config = self.load_config(config_path) if config_path else {}
        self.verbose = self.config.get('verbose', False)
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def detect_language(self, text: str, provided_lang: str) -> str:
        """Auto-detect language if provided language seems incorrect."""
        try:
            detected_lang = detect(text)
            if detected_lang != provided_lang:
                logger.warning(f"Provided language '{provided_lang}' differs from detected '{detected_lang}'")
                return detected_lang if self.config.get('use_auto_language', False) else provided_lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        return provided_lang
    
    def process_single_test(self, test_case: Dict) -> Dict:
        """Process a single test case with enhanced error handling."""
        try:
            # Verify/detect language
            language = self.detect_language(test_case['text'], test_case['lang'])
            
            # Run detection
            results = self.detector.detect_and_anonymize(test_case['text'], language)
            
            # Compare with ground truth if available
            if 'ground_truth' in test_case:
                self._evaluate_results(results['entities'], test_case['ground_truth']['entities'])
            
            if self.verbose:
                logger.info(f"Processed test case: {test_case['name']}")
                logger.debug(f"Detected entities: {results['entities']}")
            
            return {
                'test_case_name': test_case['name'],
                'results': results,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error processing test case {test_case['name']}: {e}")
            return {
                'test_case_name': test_case['name'],
                'error': str(e)
            }
    
    def _evaluate_results(self, predicted_entities: List[Dict], true_entities: List[Dict]):
        """Evaluate predicted entities against ground truth."""
        # Track matched entities to avoid double-counting
        matched = set()
        
        # Find true positives and false positives
        for pred in predicted_entities:
            found_match = False
            for i, truth in enumerate(true_entities):
                if i in matched:
                    continue
                    
                # Check for matching span and type
                if (pred['start'] == truth['start'] and 
                    pred['end'] == truth['end'] and 
                    pred['type'] == truth['type']):
                    self.metrics.update(
                        truth['type'],
                        True,
                        pred.get('score'),
                        truth.get('score')
                    )
                    matched.add(i)
                    found_match = True
                    break
            
            if not found_match:
                self.metrics.update(pred['type'], False)
        
        # Find false negatives
        for i, truth in enumerate(true_entities):
            if i not in matched:
                self.metrics.add_false_negative(truth['type'])
    
    def run_tests(self, test_cases: List[Dict], max_workers: int = 4) -> Dict:
        """Run tests in parallel with progress tracking."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {
                executor.submit(self.process_single_test, test_case): test_case
                for test_case in test_cases
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_test),
                             total=len(test_cases),
                             desc="Processing test cases"):
                test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Test case failed: {test_case['name']}, Error: {e}")
        
        # Calculate final metrics
        metrics = self.metrics.calculate_metrics()
        
        return {
            'results': results,
            'metrics': metrics
        }

def test_pii():
    """Main test function with enhanced features."""
    logger.info("Initializing PII detector...")
    detector = MultilingualPIIDetector()
    
    # Initialize test runner with config
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    runner = PIITestRunner(detector, config_path)
    
    # Define test cases
    test_cases = [
        {
            "name": "Credit Card Variations",
            "text": """Payment Information:
                Visa: 4532-0158-7845-9623
                MasterCard: 5425 2334 5521 9012
                Amex: 3782 822463 10005
                Discover: 6011 0000 0000 0004
                JCB: 3530 1113 3330 0000
                Diners: 3056 9309 0259 04
                UnionPay: 6250 9470 0000 0014""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "CREDIT_CARD", "start": 43, "end": 62, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 85, "end": 104, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 121, "end": 137, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 158, "end": 177, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 193, "end": 212, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 231, "end": 248, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 269, "end": 288, "score": 0.95}
                ]
            }
        },
        {
            "name": "International Bank Accounts",
            "text": """Banking Details:
                UK: GB29 NWBK 6016 1331 9268 19
                FR: FR76 3000 6000 0112 3456 7890 189
                DE: DE89 3704 0044 0532 0130 00
                IT: IT60 X054 2811 1010 0000 0123 456
                ES: ES91 2100 0418 4502 0005 1332
                BE: BE68 5390 0754 7034
                NL: NL91 ABNA 0417 1643 00""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "IBAN", "start": 38, "end": 60, "score": 0.95},
                    {"type": "IBAN", "start": 81, "end": 109, "score": 0.95},
                    {"type": "IBAN", "start": 130, "end": 152, "score": 0.95},
                    {"type": "IBAN", "start": 173, "end": 201, "score": 0.95},
                    {"type": "IBAN", "start": 222, "end": 246, "score": 0.95},
                    {"type": "IBAN", "start": 267, "end": 287, "score": 0.95},
                    {"type": "IBAN", "start": 308, "end": 330, "score": 0.95}
                ]
            }
        },
        {
            "name": "Tax and Business Identifiers",
            "text": """Business Information:
                US EIN: 12-3456789
                UK VAT: GB123456789
                FR VAT: FR12345678901
                DE VAT: DE123456789
                IT VAT: IT12345678901
                AU ACN: 000 000 019
                AU ABN: 51 824 753 556""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "EIN", "start": 41, "end": 51, "score": 0.95},
                    {"type": "VAT_NUMBER", "start": 68, "end": 79, "score": 0.95},
                    {"type": "VAT_NUMBER", "start": 96, "end": 109, "score": 0.95},
                    {"type": "VAT_NUMBER", "start": 126, "end": 137, "score": 0.95},
                    {"type": "VAT_NUMBER", "start": 154, "end": 167, "score": 0.95},
                    {"type": "REGISTRATION_NUMBER", "start": 184, "end": 195, "score": 0.9},
                    {"type": "REGISTRATION_NUMBER", "start": 212, "end": 226, "score": 0.9}
                ]
            }
        },
        {
            "name": "Payment and Investment Identifiers",
            "text": """Transaction Details:
                Payment Token: tok_1234567890abcdef
                Stripe Charge: ch_1234567890abcdef
                PayPal Trans: PP-1234567890ABCD
                Investment ID: INV-2023-123456
                Portfolio: PTF-2023-789012
                Account: ACC-X123456789
                Reference: REF-20231231-001""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "PAYMENT_TOKEN", "start": 47, "end": 67, "score": 0.95},
                    {"type": "PAYMENT_TOKEN", "start": 92, "end": 112, "score": 0.95},
                    {"type": "PAYMENT_TOKEN", "start": 135, "end": 152, "score": 0.9},
                    {"type": "INVESTMENT_ID", "start": 177, "end": 193, "score": 0.9},
                    {"type": "INVESTMENT_ID", "start": 214, "end": 229, "score": 0.9},
                    {"type": "ACCOUNT_NUMBER", "start": 249, "end": 263, "score": 0.9},
                    {"type": "REFERENCE_NUMBER", "start": 284, "end": 302, "score": 0.85}
                ]
            }
        },
        {
            "name": "French Financial Documents",
            "text": """Informations financières:
                RIB: 30006 00001 12345678901 89
                SIRET: 123 456 789 00012
                SIREN: 123 456 789
                Code NAF/APE: 6202A
                N° TVA: FR12345678901
                N° de facture: FACT-2023-12345
                Référence: RF-2023-98765""",
            "lang": "fr",
            "ground_truth": {
                "entities": [
                    {"type": "BANK_ACCOUNT", "start": 47, "end": 73, "score": 0.95},
                    {"type": "REGISTRATION_NUMBER", "start": 91, "end": 109, "score": 0.95},
                    {"type": "REGISTRATION_NUMBER", "start": 127, "end": 139, "score": 0.95},
                    {"type": "REGISTRATION_NUMBER", "start": 155, "end": 160, "score": 0.9},
                    {"type": "VAT_NUMBER", "start": 179, "end": 192, "score": 0.95},
                    {"type": "INVOICE_NUMBER", "start": 217, "end": 233, "score": 0.9},
                    {"type": "REFERENCE_NUMBER", "start": 255, "end": 268, "score": 0.85}
                ]
            }
        },
        {
            "name": "Arabic Financial Documents",
            "text": """معلومات مالية:
                رقم الحساب: 12-3456-7890-1234-5678-9012
                رقم البطاقة: 4532 0158 7845 9623
                الرقم الضريبي: 310999999900003
                رقم السجل التجاري: 1234567890
                رقم الفاتورة: INV-2023-54321
                المرجع: REF-20231231-002
                رقم العميل: CUS-987654321""",
            "lang": "ar",
            "ground_truth": {
                "entities": [
                    {"type": "BANK_ACCOUNT", "start": 44, "end": 74, "score": 0.95},
                    {"type": "CREDIT_CARD", "start": 97, "end": 116, "score": 0.95},
                    {"type": "VAT_NUMBER", "start": 142, "end": 155, "score": 0.95},
                    {"type": "REGISTRATION_NUMBER", "start": 184, "end": 194, "score": 0.9},
                    {"type": "INVOICE_NUMBER", "start": 217, "end": 233, "score": 0.9},
                    {"type": "REFERENCE_NUMBER", "start": 253, "end": 271, "score": 0.85},
                    {"type": "CUSTOMER_ID", "start": 293, "end": 307, "score": 0.9}
                ]
            }
        },
        {
            "name": "Cryptocurrency Identifiers",
            "text": """Crypto Wallets:
                BTC: bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq
                ETH: 0x742d35Cc6634C0532925a3b844Bc454e4438f44e
                BNB: bnb1grpf0955h0ykzq3ar5nmum7y6gdfl6lxfn46h2
                XRP: rNqf5sHJ8UYsZSxuYhAZH4TcPGWDUjZgv1
                Token Contract: 0x2170ed0880ac9a755fd29b2688956bd959f933f8
                NFT ID: 0x06012c8cf97bead5deae237070f9587f8e7a266d
                Transaction: 0x5d141d6f6a16dc7359e6d00f0cb91bb7c7593106""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "CRYPTO_WALLET", "start": 41, "end": 83, "score": 0.95},
                    {"type": "CRYPTO_WALLET", "start": 100, "end": 142, "score": 0.95},
                    {"type": "CRYPTO_WALLET", "start": 159, "end": 201, "score": 0.95},
                    {"type": "CRYPTO_WALLET", "start": 218, "end": 251, "score": 0.95},
                    {"type": "CRYPTO_CONTRACT", "start": 279, "end": 321, "score": 0.9},
                    {"type": "NFT_ID", "start": 340, "end": 382, "score": 0.9},
                    {"type": "TRANSACTION_HASH", "start": 406, "end": 448, "score": 0.9}
                ]
            }
        },
        {
            "name": "Edge Cases and Ambiguous Numbers",
            "text": """Ambiguous Identifiers:
                Generic Number: 123456789012345
                Code: function(id_12345)
                URL: http://example.com/user?id=12345
                Mixed: ACC-123-456-789
                Reference: REF_2023_12345
                Invalid Card: 0000 0000 0000 0000
                Test IBAN: XX00 TEST 0000 0000
                Invalid Email: not.an.email
                Invalid BTC: 1234abcd""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "CODE_IDENTIFIER", "start": 65, "end": 73, "score": 0.7},
                    {"type": "URL_PARAM", "start": 116, "end": 121, "score": 0.7},
                    {"type": "REFERENCE_NUMBER", "start": 138, "end": 153, "score": 0.8},
                    {"type": "REFERENCE_NUMBER", "start": 175, "end": 189, "score": 0.8}
                ]
            }
        },
        {
            "name": "Context-Dependent Classification",
            "text": """Multiple Contexts:
                Account: 123456789012345
                Card Number: 123456789012345
                Tax ID: 123456789012345
                Reference: 123456789012345
                Serial: 123456789012345""",
            "lang": "en",
            "ground_truth": {
                "entities": [
                    {"type": "ACCOUNT_NUMBER", "start": 43, "end": 58, "score": 0.9},
                    {"type": "CREDIT_CARD", "start": 82, "end": 97, "score": 0.9},
                    {"type": "TAX_ID", "start": 117, "end": 132, "score": 0.9},
                    {"type": "REFERENCE_NUMBER", "start": 154, "end": 169, "score": 0.85},
                    {"type": "SERIAL_NUMBER", "start": 189, "end": 204, "score": 0.85}
                ]
            }
        }
    ]
    
    # Run tests and get results
    results = runner.run_tests(test_cases)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.path.expanduser("~/Desktop"))
    
    # Save detailed results
    results_path = output_dir / f"pii_test_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results['results'], f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    # Save metrics
    metrics_path = output_dir / f"pii_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results['metrics'], f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Generate CSV report
    csv_path = output_dir / f"pii_test_results_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Test Case', 'Language', 'Entity Type', 'Text', 'Score',
            'Ground Truth Available', 'Correctly Detected', 'Error Type',
            'Precision', 'Recall', 'F1', 'MAE'
        ])
        
        for result in results['results']:
            if 'error' in result:
                continue
                
            metrics = results['metrics'].get(result['test_case_name'], {})
            for entity in result['results']['entities']:
                writer.writerow([
                    result['test_case_name'],
                    result['language'],
                    entity['type'],
                    entity['text'],
                    entity['score'],
                    'Yes' if 'ground_truth' in result else 'No',
                    metrics.get('precision', 'N/A'),
                    metrics.get('recall', 'N/A'),
                    metrics.get('f1', 'N/A'),
                    metrics.get('mae', 'N/A')
                ])
    
    logger.info(f"CSV report saved to: {csv_path}")
    
    # Print summary metrics
    logger.info("\nTest Results Summary:")
    logger.info(f"Micro-average metrics: {results['metrics']['micro_avg']}")
    logger.info(f"Macro-average metrics: {results['metrics']['macro_avg']}")
    
    return results

if __name__ == "__main__":
    test_pii() 