from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult, AnalysisExplanation
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from transformers import pipeline
import spacy
from typing import List, Dict, Optional, Set, Tuple
import re
import os
from presidio_analyzer.nlp_engine import NlpEngine, SpacyNlpEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.recognizer_registry import RecognizerRegistry
import stdnum.iban
import stdnum.bic
import stdnum.luhn
import stdnum.vatin
import json
import numpy as np
from spacy.util import filter_spans
from collections import defaultdict
import base58
import logging
from langdetect import detect
import pycountry

logger = logging.getLogger(__name__)

# Our own implementation of EvaluationMetrics
class EvaluationMetrics:
    def __init__(self):
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.scores = defaultdict(list)

    def update(self, entity_type: str, is_true_positive: bool, score: float = None):
        if is_true_positive:
            self.true_positives[entity_type] += 1
            if score is not None:
                self.scores[entity_type].append(score)
        else:
            self.false_positives[entity_type] += 1

    def add_false_negative(self, entity_type: str):
        self.false_negatives[entity_type] += 1

    def get_metrics(self) -> Dict:
        metrics = {}
        for entity_type in set(list(self.true_positives.keys()) + 
                             list(self.false_positives.keys()) + 
                             list(self.false_negatives.keys())):
            tp = self.true_positives[entity_type]
            fp = self.false_positives[entity_type]
            fn = self.false_negatives[entity_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        return metrics

class FinancialValidators:
    """Validation utilities for financial data"""
    
    @staticmethod
    def validate_iban(iban: str) -> bool:
        """Validate IBAN number"""
        try:
            return stdnum.iban.validate(iban.replace(" ", ""))
        except Exception:
            return False
    
    @staticmethod
    def validate_bic(bic: str) -> bool:
        """Validate BIC/SWIFT code"""
        try:
            return stdnum.bic.validate(bic)
        except Exception:
            return False
    
    @staticmethod
    def validate_vat(vat_number: str) -> bool:
        """Validate VAT number with enhanced checks"""
        try:
            # Remove spaces and normalize
            clean_vat = vat_number.replace(" ", "")
            
            # Check for Arabic tax ID format (15 digits)
            if clean_vat.isdigit() and len(clean_vat) == 15:
                # Enhanced validation for Arabic tax IDs
                # Check if it looks like a credit card (Luhn algorithm)
                if stdnum.luhn.validate(clean_vat):
                    return False
                # Basic checksum validation for Arabic tax IDs
                odd_sum = sum(int(clean_vat[i]) for i in range(0, 15, 2))
                even_sum = sum(int(clean_vat[i]) for i in range(1, 15, 2))
                if odd_sum % 11 == 0 and even_sum % 7 == 0:
                    return True
            
            # Standard EU VAT format with enhanced validation
            if len(clean_vat) >= 8:
                country_code = clean_vat[:2].upper()
                if country_code.isalpha():
                    # Specific country format validation
                    vat_patterns = {
                        'FR': r'^FR[0-9A-Z]{2}[0-9]{9}$',  # French VAT
                        'GB': r'^GB(?:[0-9]{9}|[0-9]{12})$',  # UK VAT
                        'DE': r'^DE[0-9]{9}$',  # German VAT
                        'IT': r'^IT[0-9]{11}$',  # Italian VAT
                        'ES': r'^ES[A-Z0-9][0-9]{7}[A-Z0-9]$',  # Spanish VAT
                        'BE': r'^BE[0-9]{10}$',  # Belgian VAT
                        'NL': r'^NL[0-9]{9}B[0-9]{2}$',  # Dutch VAT
                        'LU': r'^LU[0-9]{8}$',  # Luxembourg VAT
                        'AT': r'^ATU[0-9]{8}$',  # Austrian VAT
                        'DK': r'^DK[0-9]{8}$',  # Danish VAT
                        'FI': r'^FI[0-9]{8}$',  # Finnish VAT
                        'GR': r'^EL[0-9]{9}$',  # Greek VAT
                        'IE': r'^IE[0-9A-Z\+\*][0-9]{7}[A-Z]$',  # Irish VAT
                        'PT': r'^PT[0-9]{9}$',  # Portuguese VAT
                        'SE': r'^SE[0-9]{12}$',  # Swedish VAT
                    }
                    
                    # Try country-specific validation first
                    try:
                        if country_code in vat_patterns:
                            if not re.match(vat_patterns[country_code], clean_vat):
                                return False
                            # Additional check: should not pass Luhn algorithm (credit card check)
                            if stdnum.luhn.validate(re.sub(r'[^0-9]', '', clean_vat)):
                                return False
                            return True
                    except Exception:
                        pass
                    
                    # Fallback to general format check
                    if re.match(r'^[A-Z]{2}[0-9A-Z]{8,12}$', clean_vat):
                        # Should not look like a credit card number
                        if stdnum.luhn.validate(re.sub(r'[^0-9]', '', clean_vat)):
                            return False
                        return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def validate_credit_card(number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        try:
            clean_number = re.sub(r'[\s-]', '', number)
            return stdnum.luhn.validate(clean_number)
        except Exception:
            return False
    
    @staticmethod
    def validate_rib(rib: str) -> bool:
        """Validate French RIB"""
        try:
            clean_rib = re.sub(r'\s+', '', rib)
            return stdnum.fr.rib.is_valid(clean_rib)
        except Exception:
            return False
    
    @staticmethod
    def validate_ssn(ssn: str) -> bool:
        """Validate US Social Security Number"""
        try:
            return stdnum.us.ssn.is_valid(ssn)
        except Exception:
            return False
    
    @staticmethod
    def validate_routing_number(routing: str) -> bool:
        """Validate US routing number"""
        try:
            clean_routing = re.sub(r'\s+', '', routing)
            # Check basic format
            if not re.match(r'^\d{9}$', clean_routing):
                return False
            # Validate checksum
            weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
            total = sum(int(d) * w for d, w in zip(clean_routing, weights))
            return total % 10 == 0
        except Exception:
            return False
    
    @staticmethod
    def validate_crypto_address(address: str, crypto_type: str = None) -> bool:
        """Validate cryptocurrency wallet address format"""
        patterns = {
            'BTC': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$',  # Bitcoin
            'ETH': r'^0x[a-fA-F0-9]{40}$',  # Ethereum
            'generic': r'^[a-zA-Z0-9]{26,35}$'  # Generic pattern
        }
        
        if crypto_type and crypto_type in patterns:
            return bool(re.match(patterns[crypto_type], address))
        return any(bool(re.match(pattern, address)) for pattern in patterns.values())
    
    @staticmethod
    def validate_payment_token(token: str) -> bool:
        """Validate payment token format (e.g., Stripe-like tokens)"""
        patterns = [
            r'^tok_[a-zA-Z0-9]{16,}$',  # Stripe-like
            r'^pi_[a-zA-Z0-9]{16,}$',   # Payment Intent
            r'^ch_[a-zA-Z0-9]{16,}$'    # Charge
        ]
        return any(bool(re.match(pattern, token)) for pattern in patterns)
    
    @staticmethod
    def validate_investment_id(inv_id: str) -> bool:
        """Validate investment account ID format"""
        patterns = [
            r'^INV-\d{9}$',             # Standard format
            r'^FUND-[A-Z0-9]{8,}$',     # Fund format
            r'^ACC-[A-Z0-9]{9,}$'       # Account format
        ]
        return any(bool(re.match(pattern, inv_id)) for pattern in patterns)
    
    @staticmethod
    def validate_loan_number(loan_number: str) -> bool:
        """Validate loan number format"""
        patterns = [
            r'^LN-\d{4}-\d{5,}$',       # Standard format
            r'^MORT-\d{4}-\d{5,}$',     # Mortgage format
            r'^LOAN-[A-Z0-9]{8,}$'      # Generic format
        ]
        return any(bool(re.match(pattern, loan_number)) for pattern in patterns)
    
    @staticmethod
    def validate_invoice_number(invoice_number: str) -> bool:
        """Validate invoice number format"""
        patterns = [
            r'^INV-\d{4}-\d{5,}$',      # Standard format
            r'^FINV-[A-Z0-9]{8,}$',     # Financial invoice
            r'^FAC-\d{4}-\d{5,}$'       # French format
        ]
        return any(bool(re.match(pattern, invoice_number)) for pattern in patterns)

class FinancialPIIRecognizer(PatternRecognizer):
    """Enhanced recognizer for financial entities with highest priority"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "VAT_Number",
                r"\b(?:(?:Numéro\s+(?:de\s+)?(?:TVA|tax|taxe)|VAT|Tax ID|رقم\s+(?:ضريبي|الضريبة|ضريبة القيمة المضافة))\s*:?\s*)?(?:FR|BE|DE|IT|LU|NL|GB|ES|PT|AT|BG|CY|CZ|DK|EE|FI|GR|HR|HU|IE|LT|LV|MT|PL|RO|SE|SI|SK)[0-9A-Z]{8,12}\b",
                0.95
            ),
            Pattern(
                "SSN",
                r"\b(?:SSN|Social Security)?\s*\d{3}-\d{2}-\d{4}\b",
                0.95
            ),
            Pattern(
                "EIN",
                r"\b(?:EIN|Tax ID)?\s*\d{2}-\d{7}\b",
                0.95
            ),
            Pattern(
                "Bank_Account",
                r"\b(?:رقم\s+الحساب|Account|Compte)\s*:?\s*\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                0.95
            ),
            Pattern(
                "Invoice_Number",
                r"\b(?:INV|FAC|FINV)-\d{4}[-/]?\d{4,}\b",
                0.95
            ),
            Pattern(
                "Loan_Number",
                r"\b(?:LN|LOAN|MORT)-\d{4}[-/]?\d{4,}\b",
                0.95
            ),
            Pattern(
                "Credit_Card",
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
                0.95
            )
        ]
        
        # Enhanced context words in multiple languages
        default_context = [
            # English
            "tax", "vat", "ssn", "ein", "invoice", "loan", "account", "credit",
            # French
            "tva", "taxe", "compte", "facture", "prêt", "crédit",
            # Arabic
            "ضريبي", "الضريبة", "حساب", "فاتورة", "قرض"
        ]
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = "all"
            
        super().__init__(
            supported_entity="FINANCIAL_INFO",
            patterns=patterns,
            **kwargs
        )
        
        self.validators = FinancialValidators()
        
        # Entity type mapping for validation
        self.entity_type_mapping = {
            'VAT_Number': ['VAT_NUMBER', self.validators.validate_vat],
            'SSN': ['SSN', self.validators.validate_ssn],
            'Bank_Account': ['BANK_ACCOUNT', None],  # Custom validation in analyze method
            'Invoice_Number': ['INVOICE_NUMBER', None],
            'Loan_Number': ['LOAN_NUMBER', None],
            'Credit_Card': ['CREDIT_CARD', self.validators.validate_credit_card]
        }

    def analyze(self, text, entities=None, nlp_artifacts=None):
        results = super().analyze(text, entities)
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            
            # Get context
            context_before = text[max(0, result.start - 50):result.start].lower()
            context_after = text[result.end:min(len(text), result.end + 50)].lower()
            full_context = context_before + " " + extracted_text.lower() + " " + context_after
            
            # Clean the text
            clean_text = re.sub(r'[\s-]', '', extracted_text)
            
            # Determine entity type and validate
            entity_type = None
            score = result.score
            validation_passed = False
            
            # Check for VAT number first (to avoid credit card confusion)
            if (re.match(r"^[A-Z]{2}", clean_text) and 
                any(term in full_context for term in ["tva", "vat", "tax", "ضريبي", "الضريبة"])):
                if self.validators.validate_vat(clean_text):
                    entity_type = "VAT_NUMBER"
                    score = 0.95
                    validation_passed = True
            
            # Check for SSN
            elif re.match(r"^\d{3}-\d{2}-\d{4}$", extracted_text):
                if self.validators.validate_ssn(extracted_text):
                    entity_type = "SSN"
                    score = 0.95
                    validation_passed = True
            
            # Check for bank account (including Arabic)
            elif (re.match(r"^\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$", extracted_text) or
                  any(term in full_context for term in ["account", "compte", "حساب"])):
                entity_type = "BANK_ACCOUNT"
                score = 0.95
                validation_passed = True
            
            # Check for invoice numbers
            elif re.match(r"^(?:INV|FAC|FINV)-\d{4}[-/]?\d{4,}$", extracted_text):
                entity_type = "INVOICE_NUMBER"
                score = 0.95
                validation_passed = True
            
            # Check for loan numbers
            elif re.match(r"^(?:LN|LOAN|MORT)-\d{4}[-/]?\d{4,}$", extracted_text):
                entity_type = "LOAN_NUMBER"
                score = 0.95
                validation_passed = True
            
            # Only check for credit card if no other type matched
            elif not validation_passed:
                if (len(clean_text) in [15, 16] and 
                    self.validators.validate_credit_card(clean_text) and
                    not any(term in full_context for term in ["ضريبي", "الضريبة", "ضريبة", "vat", "tax"])):
                    entity_type = "CREDIT_CARD"
                    score = 0.95
                    validation_passed = True
            
            if validation_passed and entity_type:
                cleaned_results.append(RecognizerResult(
                    entity_type=entity_type,
                    start=result.start,
                    end=result.end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="FinancialPIIRecognizer",
                        original_score=score,
                        pattern_name=f"{entity_type} Pattern",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Detected and validated {entity_type}: {extracted_text}"
                    )
                ))
        
        return self._remove_overlapping_results(cleaned_results)

    def _remove_overlapping_results(self, results):
        """Remove overlapping results keeping ones with highest confidence"""
        if not results:
            return results
            
        # Sort by score (descending) and position
        sorted_results = sorted(results, key=lambda x: (-x.score, x.start))
        
        filtered = []
        current = sorted_results[0]
        
        for result in sorted_results[1:]:
            if result.start >= current.end:
                filtered.append(current)
                current = result
            elif result.score > current.score + 0.1:  # Only replace if significantly more confident
                current = result
        
        filtered.append(current)
        return filtered

    def anonymize(self, text: str) -> str:
        """Custom anonymization for financial data"""
        # Keep last 4 digits for some types
        if re.match(r"\b\d{16}\b", text.replace(" ", "")):  # Credit card
            return "*" * (len(text) - 4) + text[-4:]
        elif re.match(r"\b[A-Z]{2}\d{2}", text):  # IBAN
            return text[:4] + "*" * (len(text) - 8) + text[-4:]
        return "*" * len(text)

class SpacyRecognizer(EntityRecognizer):
    """Custom recognizer using spaCy NER"""
    
    def __init__(self, supported_language: str):
        self.supported_language = supported_language
        
        # Initialize spaCy models
        try:
            if supported_language == "en":
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    print("Downloading English model...")
                    spacy.cli.download("en_core_web_md")
                    self.nlp = spacy.load("en_core_web_md")
            else:
                try:
                    self.nlp = spacy.load("fr_core_news_md")
                except OSError:
                    print("Downloading French model...")
                    spacy.cli.download("fr_core_news_md")
                    self.nlp = spacy.load("fr_core_news_md")
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            raise
        
        # Map spaCy entity labels to Presidio entity types
        self.mapping = {
            # English types
            "PERSON": "PERSON",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "FAC": "LOCATION",
            # French types (fr_core_news_md)
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "MISC": "ORGANIZATION"
        }
        
        # Common context words to exclude
        self.context_words = {
            "téléphone", "mobile", "email", "address", "information", "numéro",
            "phone", "contact", "tel", "tél", "mail", "adresse", "portable",
            "fixe", "fax", "address", "location", "lieu", "endroit"
        }
        
        # Add spacy.util import
        self.filter_spans = filter_spans
        
        supported_entities = set(self.mapping.values())
        super().__init__(supported_entities=supported_entities, supported_language=supported_language)

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        doc = self.nlp(text)
        
        print(f"\nDebugging SpacyRecognizer ({self.supported_language}):")
        print("Raw spaCy entities:")
        
        # Filter overlapping spans before processing
        filtered_ents = self.filter_spans([ent for ent in doc.ents])
        doc.ents = filtered_ents
        
        for ent in doc.ents:
            print(f"- Text: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")
        
        spacy_results = []
        for ent in doc.ents:
            # Skip email-like strings and context words
            if '@' in ent.text or ent.text.lower() in self.context_words:
                print(f"Skipping: {ent.text}")
                continue
                
            # Skip phone number-like strings
            if re.search(r"(?:\+\d{1,4}|00\d{2}|0)\s*[1-9][\d\s.-]{8,}", ent.text):
                print(f"Skipping phone number: {ent.text}")
                continue
                
            if ent.label_ in self.mapping and self.mapping[ent.label_] in entities:
                score = 0.85
                if self.supported_language == "fr":
                    if ent.label_ == "PER":
                        if any(title in text[max(0, ent.start_char-20):ent.start_char].lower() 
                              for title in ["m.", "mme.", "dr.", "monsieur", "madame"]):
                            score = 0.95
                    elif ent.label_ == "ORG" and len(ent.text.split()) > 1:
                        score = 0.9
                    elif ent.label_ == "LOC" and any(word in text[max(0, ent.start_char-20):ent.end_char+20].lower() 
                                                   for word in ["rue", "avenue", "boulevard", "place"]):
                        score = 0.95
                else:
                    if ent.label_ == "PERSON" and len(ent.text.split()) > 1:
                        score = 0.95
                    elif ent.label_ == "ORG" and len(ent.text.split()) > 1:
                        score = 0.9
                
                spacy_results.append(
                    RecognizerResult(
                        entity_type=self.mapping[ent.label_],
                        start=ent.start_char,
                        end=ent.end_char,
                        score=score,
                        analysis_explanation=f"spaCy NER: {ent.label_}"
                    )
                )
        
        print("\nProcessed results:")
        for result in spacy_results:
            print(f"- Type: {result.entity_type}, Text: {text[result.start:result.end]}, Score: {result.score}")
        
        return self._remove_overlapping_results(spacy_results)
    
    def _remove_overlapping_results(self, results: List[RecognizerResult]) -> List[RecognizerResult]:
        """Remove overlapping results, keeping the ones with higher scores"""
        if not results:
            return results
        
        # Sort by start position and score (higher score first for same position)
        sorted_results = sorted(results, key=lambda x: (x.start, -x.score))
        
        final_results = []
        current = sorted_results[0]
        
        for next_result in sorted_results[1:]:
            if current.start <= next_result.start < current.end:
                # Results overlap
                if next_result.score > current.score:
                    current = next_result
            else:
                final_results.append(current)
                current = next_result
        
        final_results.append(current)
        return final_results

class InternationalPhoneRecognizer(PatternRecognizer):
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Arabic Phone",
                r"(?:(?:جوال|تليفون|هاتف|موبايل|اتصل|Tel|Phone|رقم|نقال)[\s:]+)?(?:\+|00)?9665\d{8}|05\d{8}",
                0.95
            ),
            Pattern(
                "International Phone",
                r"(?:(?:Tel|Phone|Mobile|Contact|Phone Number|Tel No|T:|P:|هاتف|تليفون)[\s:]+)?(?:\+|00)(?:1|7|20|27|30|31|32|33|34|36|39|40|41|43|44|45|46|47|48|49|51|52|53|54|55|56|57|58|60|61|62|63|64|65|66|81|82|84|86|90|91|92|93|94|95|98|212|213|216|218|220|221|222|223|224|225|226|227|228|229|230|231|232|233|234|235|236|237|238|239|240|241|242|243|244|245|246|247|248|249|250|251|252|253|254|255|256|257|258|260|261|262|263|264|265|266|267|268|269|290|291|297|298|299|350|351|352|353|354|355|356|357|358|359|370|371|372|373|374|375|376|377|378|379|380|381|382|383|385|386|387|389|420|421|423|500|501|502|503|504|505|506|507|508|509|590|591|592|593|594|595|596|597|598|599|670|672|673|674|675|676|677|678|679|680|681|682|683|685|686|687|688|689|690|691|692|850|852|853|855|856|870|880|886|960|961|962|963|964|965|966|967|968|970|971|972|973|974|975|976|977|992|993|994|995|996|998)\s*[0-9\s.-]{8,}",
                0.95
            ),
            Pattern(
                "European Phone",
                r"(?:(?:Tel|Phone|Mobile|Contact|T:|P:|Téléphone|Portable)[\s:]+)?(?:\+|00)(?:30|31|32|33|34|36|39|40|41|43|44|45|46|47|48|49)\s*[0-9\s.-]{8,}",
                0.98
            ),
            Pattern(
                "French Phone",
                r"(?:(?:Tel|Phone|Mobile|Contact|T:|P:|Téléphone|Portable)[\s:]+)?(?:(?:\+33|0033|0)\s*[1-9](?:[\s.-]?\d{2}){4})",
                0.98
            )
        ]
        
        default_context = [
            "phone", "mobile", "tel", "telephone", "contact", "call",
            "téléphone", "portable", "fixe", "numéro",
            "هاتف", "جوال", "موبايل", "اتصال", "نقال", "رقم"
        ]
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = "all"
            
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        results = super().analyze(text, entities)
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            
            # Get context
            context_before = text[max(0, result.start - 30):result.start].lower()
            context_after = text[result.end:min(len(text), result.end + 30)].lower()
            full_context = context_before + " " + extracted_text.lower() + " " + context_after
            
            # Skip if looks like financial data
            if any(word in full_context for word in 
                  ["iban", "account", "routing", "swift", "bic", "rib", "compte", "acn", "abn",
                   "visa", "mastercard", "credit", "ضريبي", "الضريبة", "حساب"]):
                continue
            
            # Clean and normalize the number
            clean_number = re.sub(r'[\s.-]', '', extracted_text)
            if not clean_number:
                continue
            
            # Validate the phone number
            if self._validate_phone_number(clean_number, full_context):
                score = result.score
                
                # Increase score if strong phone context exists
                if any(word in full_context for word in 
                      ["phone:", "tel:", "mobile:", "هاتف:", "جوال:", "téléphone:"]):
                    score = min(1.0, score + 0.05)
                
                cleaned_results.append(RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=result.start,
                    end=result.end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="InternationalPhoneRecognizer",
                        original_score=score,
                        pattern_name="Phone Pattern",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Detected phone number: {clean_number}"
                    )
                ))
        
        return cleaned_results

    def _validate_phone_number(self, number: str, context: str = "") -> bool:
        """Enhanced validation for phone numbers with context awareness"""
        # Remove non-digits except + at start
        if number.startswith('+'):
            digits = '+' + re.sub(r'\D', '', number[1:])
        else:
            digits = re.sub(r'\D', '', number)
        
        # Basic length check
        if len(digits) < 8 or len(digits) > 15:
            return False
        
        # Check for repeating patterns that might indicate non-phone numbers
        if re.search(r'(\d)\1{4,}', digits):  # 5 or more repeated digits
            return False
        
        # Check for sequential numbers that might be part of other data
        if re.search(r'(?:0123|1234|2345|3456|4567|5678|6789|7890){2,}', digits):
            return False
        
        # Country-specific validation with context
        if digits.startswith('+') or digits.startswith('00'):
            # Remove international prefix
            if digits.startswith('00'):
                digits = digits[2:]
            elif digits.startswith('+'):
                digits = digits[1:]
            
            # Arabic numbers (Saudi Arabia, UAE, etc.)
            if digits.startswith('966') or digits.startswith('971'):
                return len(digits) == 12 and digits[3] in '5789'
            
            # French numbers
            elif digits.startswith('33'):
                return len(digits) == 11 and digits[2] in '123456789'
            
            # UK numbers
            elif digits.startswith('44'):
                return len(digits) in [11, 12] and digits[2] != '0'
            
            # US/Canada numbers
            elif digits.startswith('1'):
                return len(digits) == 11 and digits[1] in '23456789'
        else:
            # Local format validation
            # Saudi/UAE local format
            if digits.startswith('05'):
                return len(digits) == 10
            
            # French local format
            elif digits.startswith('0'):
                return len(digits) == 10 and digits[1] in '123456789'
        
        # General international format check
        return bool(re.match(r'^\+?(?:\d{1,3})?[1-9]\d{7,11}$', digits))

class FrenchPhoneRecognizer(PatternRecognizer):
    """Custom recognizer for French phone numbers"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern("French Mobile", r"(?:\+33|0033|0)\s*[67]\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2}", 0.98),
            Pattern("French Landline", r"(?:\+33|0033|0)\s*[1-5]\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2}", 0.98),
            Pattern("French Dot Format", r"(?:\+33|0033|0)\s*[1-9](?:\.\d{2}){4}", 0.98),
        ]
        default_context = ["téléphone", "mobile", "portable", "fixe", "tel"]
        default_language = "fr"
        
        # Set defaults only if not provided in kwargs
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging FrenchPhoneRecognizer: Text: {text}")
        results = super().analyze(text, entities)
        print("Raw results:", [(r.entity_type, text[r.start:r.end], r.score) for r in results])
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            # Clean up whitespace and normalize format
            cleaned_text = re.sub(r'\s+', '', extracted_text)
            if cleaned_text:
                analysis_explanation = AnalysisExplanation(
                    recognizer="FrenchPhoneRecognizer",
                    original_score=result.score,
                    pattern_name=result.analysis_explanation.pattern_name if hasattr(result, 'analysis_explanation') else "French Phone Pattern",
                    pattern=result.analysis_explanation.pattern if hasattr(result, 'analysis_explanation') else None,
                    validation_result=True,
                    textual_explanation=f"Detected French phone number: {cleaned_text}"
                )
                cleaned_results.append(RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    analysis_explanation=analysis_explanation
                ))
        
        print("Cleaned results:", [(r.entity_type, text[r.start:r.end], r.score) for r in cleaned_results])
        return cleaned_results

class FrenchNameRecognizer(PatternRecognizer):
    """Custom recognizer for French names"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "French Name",
                r"(?:Je\s+m[e']appelle|Je suis|Monsieur|Madame|Mme\.?|M\.?|Dr\.?)\s+([A-ZÉÈ][a-zéèêëîïôöûüç]+(?:[-'\s][A-ZÉÈ][a-zéèêëîïôöûüç]+)*)",
                0.95
            )
        ]
        default_context = ["nom", "prénom", "appelle", "monsieur", "madame"]
        default_language = "fr"
        
        # Set defaults only if not provided in kwargs
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="PERSON",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging FrenchNameRecognizer: Text: {text}")
        results = super().analyze(text, entities)
        print("Raw results:", [(r.entity_type, text[r.start:r.end], r.score) for r in results])

        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end]
            # Only match names that follow specific prefixes
            name_match = re.search(r"(?:Je\s+m[e']appelle|Je suis|Monsieur|Madame|Mme\.?|M\.?|Dr\.?)\s+([A-ZÉÈ][a-zéèêëîïôöûüç]+(?:[-'\s][A-ZÉÈ][a-zéèêëîïôöûüç]+)*)", extracted_text, re.I)
            if name_match:
                name = name_match.group(1)
                # Skip common false positives
                if name.lower() in ["address", "adresse", "email", "courriel", "téléphone", "tél", "mobile", "portable"]:
                    continue
                # Find the actual name in the text
                start = text.find(name, result.start)
                if start != -1:
                    analysis_explanation = AnalysisExplanation(
                        recognizer="FrenchNameRecognizer",
                        original_score=result.score,
                        pattern_name=result.analysis_explanation.pattern_name if hasattr(result, 'analysis_explanation') else "French Name Pattern",
                        pattern=result.analysis_explanation.pattern if hasattr(result, 'analysis_explanation') else None,
                        validation_result=True,
                        textual_explanation=f"Detected French name: {name}"
                    )
                    cleaned_results.append(RecognizerResult(
                        entity_type="PERSON",
                        start=start,
                        end=start + len(name),
                        score=result.score,
                        analysis_explanation=analysis_explanation
                    ))

        print("Cleaned results:", [(r.entity_type, text[r.start:r.end], r.score) for r in cleaned_results])
        return cleaned_results

class ArabicNameRecognizer(PatternRecognizer):
    """Custom recognizer for Arabic names using transformers model"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Arabic Name",
                r"(?:اسمي|انا|السيد|السيدة|الدكتور|د\.)\s+([ء-ي]+(?:\s+[ء-ي]+)*)",
                0.95
            )
        ]
        default_context = ["اسم", "السيد", "السيدة", "الدكتور", "د"]
        default_language = "ar"
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="PERSON",
            patterns=patterns,
            **kwargs
        )
        
        # Initialize Arabic NER model with improved configuration
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            self.model_name = "hatmimoha/arabic-ner"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
            
            # Define entity type mappings
            self.entity_mapping = {
                'LOC': 'LOCATION',
                'PER': 'PERSON',
                'ORG': 'ORGANIZATION'
            }
            
            # Location keywords to help disambiguation
            self.location_keywords = {
                'مدينة', 'شارع', 'حي', 'منطقة', 'الرياض', 'جدة', 'مكة', 'المدينة',
                'طريق', 'ميدان', 'ساحة', 'برج', 'مبنى', 'عمارة', 'فندق'
            }
            
        except Exception as e:
            print(f"Error loading Arabic NER model: {e}")
            self.model = None
            self.tokenizer = None
            self.ner_pipeline = None

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging ArabicNameRecognizer: Text: {text}")
        results = []
        
        # First try pattern-based detection
        pattern_results = super().analyze(text, entities)
        print("Pattern results:", [(r.entity_type, text[r.start:r.end], r.score) for r in pattern_results])
        results.extend(pattern_results)
        
        # Then try transformer-based detection if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                print("NER results:", ner_results)
                
                # Convert NER results to spaCy-like spans for filtering
                from spacy.tokens import Span
                doc = type('Doc', (), {'text': text})()  # Mock Doc object
                spans = []
                
                for ner_result in ner_results:
                    if 0 <= ner_result['start'] < ner_result['end'] <= len(text):
                        # Create a mock Span object with the necessary attributes
                        span = type('Span', (), {
                            'start_char': ner_result['start'],
                            'end_char': ner_result['end'],
                            'label_': ner_result['entity_group'],
                            'text': text[ner_result['start']:ner_result['end']],
                            'score': ner_result['score']
                        })
                        spans.append(span)
                
                # Filter overlapping spans
                filtered_spans = filter_spans(spans)
                
                # Convert filtered spans back to RecognizerResult objects
                for span in filtered_spans:
                    # Get the word and surrounding context
                    word = span.text
                    context_before = text[max(0, span.start_char - 30):span.start_char].lower()
                    
                    # Determine entity type with improved disambiguation
                    entity_type = self._determine_entity_type(word, span.label_, context_before)
                    
                    if entity_type:
                        results.append(RecognizerResult(
                            entity_type=entity_type,
                            start=span.start_char,
                            end=span.end_char,
                            score=getattr(span, 'score', 0.8),
                            analysis_explanation=AnalysisExplanation(
                                recognizer="ArabicNameRecognizer",
                                original_score=getattr(span, 'score', 0.8),
                                pattern_name=f"Arabic {entity_type}",
                                pattern=None,
                                validation_result=True,
                                textual_explanation=f"Detected Arabic {entity_type}: {word}"
                            )
                        ))
            except Exception as e:
                print(f"Error in transformer-based detection: {e}")
        
        # Remove any remaining overlaps using score-based filtering
        results = self._remove_overlapping_results(results)
        print("Final results:", [(r.entity_type, text[r.start:r.end], r.score) for r in results])
        return results
    
    def _determine_entity_type(self, word: str, predicted_type: str, context: str) -> Optional[str]:
        """Enhanced entity type determination with context awareness"""
        # Convert predicted type to standard format
        entity_type = self.entity_mapping.get(predicted_type, predicted_type)
        
        # Check if the word is in our location keywords
        if word in self.location_keywords:
            return 'LOCATION'
        
        # Check context for location indicators
        if any(keyword in context for keyword in self.location_keywords):
            return 'LOCATION'
        
        # If it's predicted as a person but looks like a location, correct it
        if entity_type == 'PERSON' and any(word.startswith(prefix) for prefix in ['ال', 'مدينة', 'حي']):
            return 'LOCATION'
        
        return entity_type
    
    def _remove_overlapping_results(self, results):
        """Remove overlapping results keeping ones with highest confidence"""
        if not results:
            return results
            
        # Sort by score (descending) and position
        sorted_results = sorted(results, key=lambda x: (-x.score, x.start))
        
        filtered = []
        current = sorted_results[0]
        
        for result in sorted_results[1:]:
            if result.start >= current.end:
                filtered.append(current)
                current = result
            elif result.score > current.score + 0.1:  # Only replace if significantly more confident
                current = result
        
        filtered.append(current)
        return filtered

class ArabicAddressRecognizer(PatternRecognizer):
    """Custom recognizer for Arabic addresses"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Arabic Address",
                r"(?:العنوان|المنزل|السكن|شارع|طريق)\s+[ء-ي\s,٠-٩]+",
                0.95
            ),
            Pattern(
                "Arabic Postal Code",
                r"\b\d{5,6}\s*(?:المملكة العربية السعودية|مصر|الإمارات|الكويت|قطر|عمان|البحرين|العراق|الأردن|لبنان)\b",
                0.85
            ),
            Pattern(
                "Arabic Building",
                r"\b(?:مبنى|عمارة|برج|فيلا)\s+[ء-ي\s,٠-٩]+",
                0.8
            )
        ]
        default_context = ["عنوان", "شارع", "طريق", "حي", "مدينة", "منطقة", "مبنى", "عمارة", "برج", "فيلا", "رقم", "ص.ب"]
        default_language = "ar"
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="LOCATION",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging ArabicAddressRecognizer: Text: {text}")
        results = super().analyze(text, entities)
        print("Raw results:", [(r.entity_type, text[r.start:r.end], r.score) for r in results])
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            if any(word in extracted_text.lower() for word in ["بريد", "هاتف", "جوال", "موبايل"]):
                continue
                
            analysis_explanation = AnalysisExplanation(
                recognizer="ArabicAddressRecognizer",
                original_score=result.score,
                pattern_name="Arabic Address Pattern",
                pattern=None,
                validation_result=True,
                textual_explanation=f"Detected Arabic address: {extracted_text}"
            )
            cleaned_results.append(RecognizerResult(
                entity_type="LOCATION",
                start=result.start,
                end=result.end,
                score=result.score,
                analysis_explanation=analysis_explanation
            ))
        
        print("Cleaned results:", [(r.entity_type, text[r.start:r.end], r.score) for r in cleaned_results])
        return cleaned_results

class ArabicFinancialRecognizer(PatternRecognizer):
    """Custom recognizer for Arabic financial information"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Arabic VAT",
                r"\b(?:(?:رقم\s+)?(?:ضريبة|الضريبة|ضريبي|الرقم الضريبي|VAT|ضريبة القيمة المضافة)\s*[:؛]?\s*)?\d{15}\b",
                0.95
            ),
            Pattern(
                "Arabic Credit Card",
                r"\b(?:(?:بطاقة|كارت|فيزا|ماستر)\s*[:؛]?\s*)?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                0.95
            ),
            Pattern(
                "Arabic Bank Account",
                r"\b\d{2}-\d{4}-\d{4}-\d{4}-\d{4}-\d{4}\b|\b\d{20}\b",  # IBAN-like format
                0.95
            ),
            Pattern(
                "Arabic Currency Amount",
                r"\b(?:\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ريال|درهم|دينار|جنيه|ر\.س|د\.إ|د\.ك|ج\.م)(?:\s+(?:سعودي|إماراتي|كويتي|مصري))?\b",
                0.85
            ),
            Pattern(
                "Arabic Invoice Number",
                r"\b(?:فاتورة|رقم الفاتورة)[-\s]?\d{4,}(?:[-/]\d{4,})?\b",
                0.85
            ),
            Pattern(
                "Arabic Payment Token",
                r"\b(?:tok|pi|ch)_[A-Za-z0-9]{16,}\b",
                0.95
            )
        ]
        default_context = [
            "حساب", "بنك", "مصرف", "بطاقة", "ائتمان", "فاتورة", "دفع", "تحويل",
            "رقم", "ضريبي", "الضريبة", "ريال", "درهم", "دينار", "جنيه", "سعودي",
            "إماراتي", "كويتي", "مصري"
        ]
        default_language = "ar"
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="FINANCIAL_INFO",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging ArabicFinancialRecognizer: Text: {text}")
        results = super().analyze(text, entities)
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            
            # Get context (expanded window)
            context_before = text[max(0, result.start - 50):result.start].lower()
            context_after = text[result.end:min(len(text), result.end + 50)].lower()
            
            # Clean the number for validation
            clean_number = re.sub(r'[\s-]', '', extracted_text)
            
            # Determine entity type and validate
            entity_type = None
            score = result.score
            validation_passed = False
            
            # Check for VAT number (15 digits with tax context)
            if (len(clean_number) == 15 and clean_number.isdigit() and
                (any(word in context_before for word in ["ضريبي", "الضريبة", "ضريبة", "vat", "tax"]) or
                 any(word in context_after for word in ["ضريبي", "الضريبة", "ضريبة", "vat", "tax"]))):
                entity_type = "VAT_NUMBER"
                score = 0.95
                validation_passed = True
                
                # Additional check: if it passes Luhn algorithm, it might be a credit card
                if stdnum.luhn.validate(clean_number):
                    # Look for stronger VAT context
                    strong_vat_context = ["ضريبة القيمة المضافة", "الرقم الضريبي", "tax identification"]
                    if not any(term in context_before + context_after for term in strong_vat_context):
                        validation_passed = False
            
            # Check for credit card
            elif (re.match(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", extracted_text) or
                  (len(clean_number) in [15, 16] and stdnum.luhn.validate(clean_number))):
                # Check for credit card context
                if any(word in context_before + context_after for word in 
                      ["بطاقة", "كارت", "فيزا", "ماستر", "visa", "mastercard", "credit", "card"]):
                    entity_type = "CREDIT_CARD"
                    score = 0.95
                    validation_passed = True
            
            # Handle other financial types
            elif result.analysis_explanation:
                if "Arabic Bank Account" in result.analysis_explanation.pattern_name:
                    entity_type = "BANK_ACCOUNT"
                    validation_passed = True
                elif "Arabic Currency Amount" in result.analysis_explanation.pattern_name:
                    entity_type = "CURRENCY_AMOUNT"
                    validation_passed = True
                elif "Arabic Payment Token" in result.analysis_explanation.pattern_name:
                    entity_type = "PAYMENT_TOKEN"
                    validation_passed = True
            
            if validation_passed and entity_type:
                cleaned_results.append(RecognizerResult(
                    entity_type=entity_type,
                    start=result.start,
                    end=result.end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="ArabicFinancialRecognizer",
                        original_score=score,
                        pattern_name=f"Arabic {entity_type} Pattern",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Detected Arabic {entity_type}: {extracted_text}"
                    )
                ))
        
        return self._remove_overlapping_results(cleaned_results)

class CreditCardRecognizer(PatternRecognizer):
    """Custom recognizer for credit card numbers"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Credit Card",
                r"\b(?:\d[ -]*?){13,16}\b",
                0.95
            ),
            Pattern(
                "Credit Card with Spaces",
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                0.95
            ),
            Pattern(
                "Credit Card Compact",
                r"\b\d{16}\b",
                0.85
            )
        ]
        default_context = ["credit", "card", "carte", "crédit", "visa", "mastercard", "american express", "amex", "paiement"]
        default_language = "all"
        
        # Set defaults only if not provided in kwargs
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="CREDIT_CARD",
            patterns=patterns,
            **kwargs
        )

class AddressRecognizer(PatternRecognizer):
    """Custom recognizer for addresses"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "French Address",
                r"\d{1,4}\s+(?:rue|avenue|boulevard|allée|impasse|chemin|place)\s+[A-Za-zÀ-ÿ '-]+,\s*\d{5}\s+[A-ZÉÈ][a-zéèêëîïôöûüç]+",
                score=0.95,
            ),
            Pattern(
                "French Postal Code",
                r"\b\d{5}\s+[A-ZÉÈ][a-zéèêëîïôöûüç\s-]{2,50}\b",
                score=0.85,
            ),
            Pattern(
                "French Street Only",
                r"\b(?:rue|avenue|boulevard|impasse|place|chemin|route|allée|cours|av\.|bd\.|rte\.)\s+[^,\n]{2,50}\b",
                score=0.7,
            ),
            Pattern(
                "French Address with Context",
                r"(?:adresse|habite|demeure|résidence|situé[e]?\s+(?:au|à))\s+(?:\d{1,4}[\s,]*)?(?:rue|avenue|boulevard|impasse|place|chemin|route|allée|cours|av\.|bd\.|rte\.)\s+[^,\n]{2,50}",
                score=0.9,
            ),
            Pattern(
                "French Building",
                r"\b(?:appartement|appt\.?|bâtiment|bat\.?|résidence)\s+[A-Z0-9][A-Za-z0-9\s-]{1,30}",
                score=0.8,
            )
        ]
        default_context = ["adresse", "habite", "demeure", "résidence", "appartement", "maison", "domicile", "situé", "code postal", "ville", "bâtiment", "bat.", "appt.", "étage"]
        default_language = "fr"
        
        # Set defaults only if not provided in kwargs
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = default_language
            
        super().__init__(
            supported_entity="LOCATION",
            patterns=patterns,
            **kwargs
        )

    def analyze(self, text, entities=None, nlp_artifacts=None):
        print(f"\nDebugging AddressRecognizer: Text: {text}")
        results = super().analyze(text, entities)
        print("Raw results:", [(r.entity_type, text[r.start:r.end], r.score) for r in results])
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            # Skip common false positives
            if any(word in extracted_text.lower() for word in ["email", "courriel", "téléphone", "tél", "mobile", "portable"]):
                continue
                
            # Find the full address if it exists
            full_address_match = re.search(r"\d{1,4}\s+(?:rue|avenue|boulevard|allée|impasse|chemin|place)\s+[A-Za-zÀ-ÿ '-]+,\s*\d{5}\s+[A-ZÉÈ][a-zéèêëîïôöûüç]+", extracted_text)
            if full_address_match:
                full_address = full_address_match.group(0)
                start = text.find(full_address, result.start)
                if start != -1:
                    analysis_explanation = AnalysisExplanation(
                        recognizer="AddressRecognizer",
                        original_score=result.score,
                        pattern_name=result.analysis_explanation.pattern_name if hasattr(result, 'analysis_explanation') else "French Address Pattern",
                        pattern=result.analysis_explanation.pattern if hasattr(result, 'analysis_explanation') else None,
                        validation_result=True,
                        textual_explanation=f"Detected French address: {full_address}"
                    )
                    cleaned_results.append(RecognizerResult(
                        entity_type="LOCATION",
                        start=start,
                        end=start + len(full_address),
                        score=result.score,
                        analysis_explanation=analysis_explanation
                    ))
            else:
                # If no full address found, use the original result
                cleaned_results.append(result)
        
        print("Cleaned results:", [(r.entity_type, text[r.start:r.end], r.score) for r in cleaned_results])
        return cleaned_results

class CustomEmailRecognizer(PatternRecognizer):
    """Custom recognizer for email addresses with highest priority and enhanced detection"""
    
    def __init__(self):
        email_patterns = [
            Pattern(
                name="standard_email",
                # Enhanced regex pattern for better coverage
                regex=r"(?i)(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
                score=1.0
            ),
            Pattern(
                name="quoted_email",
                regex=r'\b"[^"]+"\s*<[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}>\b',
                score=0.95
            ),
            Pattern(
                name="display_name_email",
                regex=r"\b[A-Za-z0-9\s]+\s*<[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}>\b",
                score=0.95
            ),
            Pattern(
                name="fallback_email",
                # Simple fallback pattern to catch basic emails that might be missed
                regex=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                score=0.85
            )
        ]
        
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            patterns=email_patterns,
            supported_language="all",
            context=["email", "mail", "contact", "support", "info", "@", "e-mail", "courriel", "البريد", 
                    "adresse", "mél", "courrier"]
        )

    def validate_email(self, email: str) -> bool:
        """Enhanced validation for email addresses"""
        # Remove display name if present
        if '<' in email and '>' in email:
            email = email[email.find('<')+1:email.find('>')]
        
        # Basic format validation
        if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$", email, re.I):
            return False
            
        # Check for common invalid patterns
        invalid_patterns = [
            r'example\.com$',
            r'test\.com$',
            r'domain\.com$',
            r'@.*@',
            r'\.{2,}',
            r'^[0-9]+@',  # Emails starting with numbers are often fake
            r'@.*_.*\.'   # Underscores in domain part
        ]
        
        if any(re.search(pattern, email, re.I) for pattern in invalid_patterns):
            return False
            
        # Additional validation rules
        parts = email.split('@')
        if len(parts) != 2:
            return False
            
        local_part, domain = parts
        
        # Local part checks
        if len(local_part) > 64:
            return False
        if local_part.startswith('.') or local_part.endswith('.'):
            return False
            
        # Domain part checks
        if len(domain) > 255:
            return False
        if domain.startswith('-') or domain.endswith('-'):
            return False
        if not re.match(r'^[A-Za-z0-9.-]+$', domain):
            return False
        
        return True

    def analyze(self, text: str, entities=None, nlp_artifacts=None) -> List[RecognizerResult]:
        results = super().analyze(text, entities)
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            
            # Extract actual email from display name format
            if '<' in extracted_text and '>' in extracted_text:
                email = extracted_text[extracted_text.find('<')+1:extracted_text.find('>')]
            else:
                email = extracted_text
            
            # Get context
            context_before = text[max(0, result.start - 30):result.start].lower()
            context_after = text[result.end:min(len(text), result.end + 30)].lower()
            
            # Skip if looks like part of a URL or file path
            if any(x in context_before + context_after for x in ['http://', 'https://', 'ftp://', '\\', '/']):
                continue
                
            if self.validate_email(email):
                # Increase score if email has strong context
                score = result.score
                if any(word in context_before + context_after for word in 
                      ['email:', 'mail:', 'contact:', 'e-mail:', 'courriel:', 'البريد:']):
                    score = min(1.0, score + 0.1)
                
                cleaned_results.append(RecognizerResult(
                    entity_type="EMAIL_ADDRESS",
                    start=result.start,
                    end=result.end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="CustomEmailRecognizer",
                        original_score=score,
                        pattern_name="Email Pattern",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Detected valid email address: {extracted_text}"
                    )
                ))
        
        return cleaned_results

class CryptocurrencyRecognizer(PatternRecognizer):
    """Enhanced recognizer for cryptocurrency addresses and tokens"""
    
    def __init__(self, **kwargs):
        patterns = [
            Pattern(
                "Bitcoin_Address",
                r"\b(?:(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b",
                0.99
            ),
            Pattern(
                "Ethereum_Address",
                r"\b0x[a-fA-F0-9]{40}\b",
                0.99
            ),
            Pattern(
                "Ethereum_Token",
                r"\b0x[a-fA-F0-9]{64}\b",
                0.95
            ),
            Pattern(
                "Binance_Address",
                r"\bbnb[a-zA-Z0-9]{39}\b",
                0.95
            ),
            Pattern(
                "Ripple_Address",
                r"\br[a-zA-Z0-9]{24,34}\b",
                0.95
            ),
            Pattern(
                "Crypto_Context",
                r"\b(?:bitcoin|btc|ethereum|eth|wallet|address|crypto|blockchain|token|nft)\b",
                0.7
            )
        ]
        
        default_context = [
            "crypto", "wallet", "bitcoin", "btc", "ethereum", "eth", "address", 
            "token", "blockchain", "nft", "binance", "ripple", "xrp", "transfer",
            "محفظة", "عملة", "بيتكوين", "إيثيريوم"  # Arabic context
        ]
        
        if "context" not in kwargs:
            kwargs["context"] = default_context
        if "supported_language" not in kwargs:
            kwargs["supported_language"] = "all"
            
        super().__init__(
            supported_entity="CRYPTO_WALLET",
            patterns=patterns,
            **kwargs
        )
        
        # Common crypto address prefixes
        self.address_prefixes = {
            'btc': ['1', '3', 'bc1'],
            'eth': ['0x'],
            'bnb': ['bnb'],
            'xrp': ['r']
        }

    def validate_address(self, address: str, crypto_type: str = None) -> bool:
        """Enhanced validation for cryptocurrency addresses"""
        # Remove spaces and normalize
        clean_address = address.strip()
        
        # Bitcoin address validation
        if crypto_type == 'btc' or any(clean_address.startswith(prefix) for prefix in self.address_prefixes['btc']):
            # Legacy address (P2PKH)
            if clean_address.startswith('1'):
                return (len(clean_address) == 34 and 
                       all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in clean_address[1:]))
            # P2SH address
            elif clean_address.startswith('3'):
                return (len(clean_address) == 34 and 
                       all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in clean_address[1:]))
            # Bech32 address
            elif clean_address.startswith('bc1'):
                return (len(clean_address) >= 14 and len(clean_address) <= 74 and 
                       all(c in '023456789acdefghjklmnpqrstuvwxyz' for c in clean_address[3:]))
        
        # Ethereum address validation
        elif crypto_type == 'eth' or clean_address.startswith('0x'):
            if len(clean_address) == 42 and clean_address.startswith('0x'):
                # Check if it's a valid hex string
                try:
                    int(clean_address[2:], 16)
                    # Additional check: should not be all zeros after 0x
                    return clean_address[2:] != '0' * 40
                except ValueError:
                    return False
        
        # Binance address validation
        elif crypto_type == 'bnb' or clean_address.startswith('bnb'):
            return (len(clean_address) == 42 and 
                   clean_address.startswith('bnb') and 
                   all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in clean_address[3:]))
        
        # Ripple address validation
        elif crypto_type == 'xrp' or clean_address.startswith('r'):
            return (len(clean_address) >= 25 and len(clean_address) <= 35 and
                   clean_address.startswith('r') and
                   all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in clean_address[1:]))
        
        # Generic validation if type not specified
        return bool(re.match(r'^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]{25,42}$', clean_address))

    def analyze(self, text: str, entities=None, nlp_artifacts=None) -> List[RecognizerResult]:
        results = super().analyze(text, entities)
        
        cleaned_results = []
        for result in results:
            extracted_text = text[result.start:result.end].strip()
            
            # Get context
            context_before = text[max(0, result.start - 30):result.start].lower()
            context_after = text[result.end:min(len(text), result.end + 30)].lower()
            full_context = context_before + " " + extracted_text.lower() + " " + context_after
            
            # Skip if looks like part of a URL or file path
            if any(x in full_context for x in ['http://', 'https://', 'ftp://', '\\', '/']):
                continue
            
            # Determine crypto type from context or pattern
            crypto_type = None
            if 'bitcoin' in full_context or 'btc' in full_context:
                crypto_type = 'btc'
            elif 'ethereum' in full_context or 'eth' in full_context:
                crypto_type = 'eth'
            elif 'binance' in full_context or 'bnb' in full_context:
                crypto_type = 'bnb'
            elif 'ripple' in full_context or 'xrp' in full_context:
                crypto_type = 'xrp'
            
            # Validate the address
            if self.validate_address(extracted_text, crypto_type):
                score = result.score
                
                # Increase score if strong context exists
                if any(word in full_context for word in ['wallet:', 'address:', 'محفظة:']):
                    score = min(1.0, score + 0.05)
                
                cleaned_results.append(RecognizerResult(
                    entity_type="CRYPTO_WALLET",
                    start=result.start,
                    end=result.end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="CryptocurrencyRecognizer",
                        original_score=score,
                        pattern_name=f"Crypto Address Pattern ({crypto_type or 'unknown'})",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Detected valid cryptocurrency address: {extracted_text}"
                    )
                ))
        
        return self._remove_overlapping_results(cleaned_results)
    
    def _remove_overlapping_results(self, results):
        """Remove overlapping results keeping ones with highest confidence"""
        if not results:
            return results
            
        # Sort by score (descending) and position
        sorted_results = sorted(results, key=lambda x: (-x.score, x.start))
        
        filtered = []
        current = sorted_results[0]
        
        for result in sorted_results[1:]:
            if result.start >= current.end:
                filtered.append(current)
                current = result
            elif result.score > current.score + 0.1:  # Only replace if significantly more confident
                current = result
        
        filtered.append(current)
        return filtered

class MultilingualPIIDetector:
    """PII detector supporting English, French, and Arabic"""
    
    def __init__(self):
        """Initialize the multilingual PII detector with English, French, and Arabic support."""
        # Initialize recognizer instances first
        self.email_recognizer = CustomEmailRecognizer()
        self.financial_pii_recognizer = FinancialPIIRecognizer()
        self.credit_card_recognizer = CreditCardRecognizer()
        self.crypto_recognizer = CryptocurrencyRecognizer()
        self.international_phone_recognizer = InternationalPhoneRecognizer()
        self.french_phone_recognizer = FrenchPhoneRecognizer()
        self.french_name_recognizer = FrenchNameRecognizer()
        self.address_recognizer = AddressRecognizer()
        self.arabic_name_recognizer = ArabicNameRecognizer()
        self.arabic_address_recognizer = ArabicAddressRecognizer()
        self.arabic_financial_recognizer = ArabicFinancialRecognizer()
        
        # Initialize spaCy recognizers
        self.spacy_recognizer_en = SpacyRecognizer(supported_language="en")
        self.spacy_recognizer_fr = SpacyRecognizer(supported_language="fr")

        # Define operators with fixed integer values for chars_to_mask
        self.operators = {
            "EMAIL_ADDRESS": OperatorConfig("mask", {
                "chars_to_mask": 8,
                "masking_char": "*",
                "from_end": False
            }),
            "CREDIT_CARD": OperatorConfig("mask", {
                "chars_to_mask": 12,
                "masking_char": "*",
                "from_end": True
            }),
            "BANK_ACCOUNT": OperatorConfig("mask", {
                "chars_to_mask": 8,
                "masking_char": "*",
                "from_end": True
            }),
            "PHONE_NUMBER": OperatorConfig("mask", {
                "chars_to_mask": 6,
                "masking_char": "*",
                "from_end": True
            }),
            "PERSON": OperatorConfig("mask", {
                "chars_to_mask": -1,  # Mask entire value
                "masking_char": "*",
                "from_end": False
            }),
            "LOCATION": OperatorConfig("mask", {
                "chars_to_mask": -1,  # Mask entire value
                "masking_char": "*",
                "from_end": False
            }),
            "ORGANIZATION": OperatorConfig("mask", {
                "chars_to_mask": -1,  # Mask entire value
                "masking_char": "*",
                "from_end": False
            }),
            "FINANCIAL_INFO": OperatorConfig("mask", {
                "chars_to_mask": 10,
                "masking_char": "*",
                "from_end": True
            }),
            "CRYPTO_WALLET": OperatorConfig("mask", {
                "chars_to_mask": 30,
                "masking_char": "*",
                "from_end": False
            }),
            "IBAN": OperatorConfig("mask", {
                "chars_to_mask": 16,
                "masking_char": "*",
                "from_end": True
            }),
            "SWIFT_BIC": OperatorConfig("mask", {
                "chars_to_mask": 6,
                "masking_char": "*",
                "from_end": False
            }),
            "SSN": OperatorConfig("mask", {
                "chars_to_mask": 5,
                "masking_char": "*",
                "from_end": True
            }),
            "VAT_NUMBER": OperatorConfig("mask", {
                "chars_to_mask": 8,
                "masking_char": "*",
                "from_end": True
            }),
            "PAYMENT_TOKEN": OperatorConfig("mask", {
                "chars_to_mask": 8,
                "masking_char": "*",
                "from_end": True
            }),
            "INVOICE_NUMBER": OperatorConfig("mask", {
                "chars_to_mask": 6,
                "masking_char": "*",
                "from_end": True
            }),
            "DEFAULT": OperatorConfig("mask", {
                "chars_to_mask": -1,  # Mask entire value
                "masking_char": "*",
                "from_end": False
            })
        }

        # Define minimum confidence thresholds for each entity type
        self.confidence_thresholds = {
            "EMAIL_ADDRESS": 0.7,
            "CREDIT_CARD": 0.8,  # Higher threshold for sensitive financial data
            "BANK_ACCOUNT": 0.8,
            "PHONE_NUMBER": 0.7,
            "PERSON": 0.7,
            "LOCATION": 0.7,
            "ORGANIZATION": 0.8,  # Higher threshold to prevent weak matches
            "FINANCIAL_INFO": 0.8,
            "CRYPTO_WALLET": 0.9,  # Very high threshold for crypto addresses
            "IBAN": 0.9,
            "SWIFT_BIC": 0.9,
            "SSN": 0.9,
            "VAT_NUMBER": 0.8,
            "INVOICE_NUMBER": 0.7,
            "DEFAULT": 0.7
        }

        # Initialize NLP engines
        try:
            # Initialize spaCy models
            try:
                spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading English model...")
                spacy.cli.download("en_core_web_sm")
            
            try:
                spacy.load("fr_core_news_sm")
            except OSError:
                print("Downloading French model...")
                spacy.cli.download("fr_core_news_sm")
            
            try:
                spacy.load("xx_ent_wiki_sm")
            except OSError:
                print("Downloading universal model...")
                spacy.cli.download("xx_ent_wiki_sm")
            
            # Initialize Arabic NLP engine using transformers
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                self.ar_model_name = "hatmimoha/arabic-ner"
                self.ar_tokenizer = AutoTokenizer.from_pretrained(self.ar_model_name)
                self.ar_model = AutoModelForTokenClassification.from_pretrained(self.ar_model_name)
                self.ar_ner = pipeline("ner", model=self.ar_model, tokenizer=self.ar_tokenizer)
            except Exception as e:
                print(f"Error initializing Arabic NER model: {e}")
                self.ar_model = None
                self.ar_tokenizer = None
                self.ar_ner = None
            
            # Initialize NLP engines using NlpEngineProvider
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {
                        "lang_code": "en",
                        "model_name": "en_core_web_sm"
                    },
                    {
                        "lang_code": "fr",
                        "model_name": "fr_core_news_sm"
                    }
                ]
            }
            
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            
            # Split the engine for English and French
            self.nlp_engine_en = nlp_engine
            self.nlp_engine_fr = nlp_engine
            
            # Initialize transformers NLP engine for Arabic
            if self.ar_model and self.ar_tokenizer:
                from presidio_analyzer.nlp_engine import TransformersNlpEngine
                self.nlp_engine_ar = TransformersNlpEngine(
                    models={
                        "ar": {
                            "spacy": "xx_ent_wiki_sm",  # Lightweight universal model
                            "transformers": self.ar_model_name
                        }
                    }
                )
            else:
                self.nlp_engine_ar = None
        except Exception as e:
            print(f"Error initializing NLP engines: {e}")
            raise
        
        # Initialize analyzers with empty registries
        self.analyzer_en = AnalyzerEngine(
            nlp_engine=self.nlp_engine_en,
            registry=RecognizerRegistry(recognizers=[])
        )
        self.analyzer_fr = AnalyzerEngine(
            nlp_engine=self.nlp_engine_fr,
            registry=RecognizerRegistry(recognizers=[])
        )
        if self.nlp_engine_ar:
            self.analyzer_ar = AnalyzerEngine(
                nlp_engine=self.nlp_engine_ar,
                registry=RecognizerRegistry(recognizers=[])
            )
        else:
            self.analyzer_ar = None
        
        # Initialize anonymizer
        self.anonymizer = AnonymizerEngine()
        
        # Initialize custom recognizers
        print("\nInitializing and configuring recognizers...")
        
        # Create recognizer instances with proper configuration
        recognizers = {
            'all': [
                (self.email_recognizer, 1.0),  # Highest priority
                (self.financial_pii_recognizer, 0.95),
                (self.credit_card_recognizer, 0.95),
                (self.crypto_recognizer, 0.9),
                (self.international_phone_recognizer, 0.85)
            ],
            'fr': [
                (self.french_phone_recognizer, 0.9),
                (self.french_name_recognizer, 0.85),
                (self.address_recognizer, 0.85),
                (self.spacy_recognizer_fr, 0.8)
            ],
            'en': [
                (self.spacy_recognizer_en, 0.8)
            ],
            'ar': [
                (self.arabic_name_recognizer, 0.9),
                (self.arabic_address_recognizer, 0.85),
                (self.arabic_financial_recognizer, 0.85)
            ]
        }
        
        # Add recognizers to each analyzer with proper priority
        print("\nConfiguring English analyzer...")
        for recognizer, priority in recognizers['all'] + recognizers['en']:
            try:
                self.analyzer_en.registry.add_recognizer(recognizer)
                print(f"Added {recognizer.__class__.__name__} with priority {priority}")
            except Exception as e:
                print(f"Error adding {recognizer.__class__.__name__}: {str(e)}")
        
        print("\nConfiguring French analyzer...")
        for recognizer, priority in recognizers['all'] + recognizers['fr']:
            try:
                self.analyzer_fr.registry.add_recognizer(recognizer)
                print(f"Added {recognizer.__class__.__name__} with priority {priority}")
            except Exception as e:
                print(f"Error adding {recognizer.__class__.__name__}: {str(e)}")
        
        if self.analyzer_ar:
            print("\nConfiguring Arabic analyzer...")
            for recognizer, priority in recognizers['all'] + recognizers['ar']:
                try:
                    self.analyzer_ar.registry.add_recognizer(recognizer)
                    print(f"Added {recognizer.__class__.__name__} with priority {priority}")
                except Exception as e:
                    print(f"Error adding {recognizer.__class__.__name__}: {str(e)}")
        
        # Verify configuration
        print("\nVerifying recognizer configuration...")
        print(f"English recognizers: {len(self.analyzer_en.registry.recognizers)}")
        print(f"French recognizers: {len(self.analyzer_fr.registry.recognizers)}")
        if self.analyzer_ar:
            print(f"Arabic recognizers: {len(self.analyzer_ar.registry.recognizers)}")
        
        print("\nRecognizer initialization complete")
    
    def _get_analyzer_results(self, text: str, language: str) -> List[RecognizerResult]:
        """Get analyzer results based on language with improved entity handling"""
        print(f"\nProcessing text in language: {language}")
        
        # Define comprehensive entity lists per language
        entities = {
            'en': [
                "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", 
                "EMAIL_ADDRESS", "CREDIT_CARD", "FINANCIAL_INFO", "CRYPTO_WALLET",
                "IBAN", "SWIFT_BIC", "SSN", "VAT_NUMBER", "INVOICE_NUMBER",
                "BANK_ACCOUNT"
            ],
            'fr': [
                "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", 
                "EMAIL_ADDRESS", "CREDIT_CARD", "FINANCIAL_INFO", "CRYPTO_WALLET",
                "IBAN", "SWIFT_BIC", "VAT_NUMBER", "INVOICE_NUMBER"
            ],
            'ar': [
                "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", 
                "EMAIL_ADDRESS", "CREDIT_CARD", "FINANCIAL_INFO", "CRYPTO_WALLET",
                "BANK_ACCOUNT", "VAT_NUMBER", "CURRENCY_AMOUNT"
            ]
        }
        
        # Get appropriate analyzer and entities
        if language == 'fr':
            analyzer = self.analyzer_fr
            entity_list = entities['fr']
        elif language == 'ar':
            analyzer = self.analyzer_ar
            entity_list = entities['ar']
        else:  # default to English
            analyzer = self.analyzer_en
            entity_list = entities['en']
        
        if not analyzer:
            print(f"Warning: No analyzer available for language {language}")
            return []
        
        try:
            # Debug print active recognizers
            print("\nActive recognizers:")
            for recognizer in analyzer.registry.recognizers:
                print(f"- {recognizer.__class__.__name__}")
                if hasattr(recognizer, 'patterns'):
                    print(f"  Patterns: {[p.name for p in recognizer.patterns]}")
            
            # First pass: Pattern recognizers with higher priority
            pattern_results = []
            for recognizer in analyzer.registry.recognizers:
                if isinstance(recognizer, PatternRecognizer):
                    try:
                        results = recognizer.analyze(text=text, entities=entity_list)
                        if results:
                            print(f"\nResults from {recognizer.__class__.__name__}:")
                            for r in results:
                                if r.score >= self.confidence_thresholds.get(r.entity_type, 0.5):
                                    print(f"- {r.entity_type}: {text[r.start:r.end]} (score: {r.score})")
                                    pattern_results.append(r)
                    except Exception as e:
                        print(f"Error in recognizer {recognizer.__class__.__name__}: {str(e)}")
            
            # Second pass: NLP-based recognizers
            nlp_results = []
            try:
                nlp_results = analyzer.analyze(text=text, entities=entity_list, language=language)
                print("\nResults from NLP analysis:")
                for r in nlp_results:
                    if r.score >= self.confidence_thresholds.get(r.entity_type, 0.5):
                        print(f"- {r.entity_type}: {text[r.start:r.end]} (score: {r.score})")
            except Exception as e:
                print(f"Error in NLP analysis: {str(e)}")
            
            # Combine and filter results
            all_results = pattern_results + nlp_results
            if not all_results:
                print("\nWarning: No entities detected!")
                return []
            
            # Remove duplicates and overlaps, prioritizing higher confidence results
            filtered_results = self._remove_overlapping_entities(all_results)
            
            print(f"\nFinal results count: {len(filtered_results)}")
            return filtered_results
            
        except Exception as e:
            print(f"Error in analyzer for {language}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before pattern matching"""
        # Normalize newlines and whitespace
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove zero-width spaces and other invisible characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        return text
    
    def _clean_entity_boundaries(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """Clean up entity boundaries to avoid partial matches"""
        # Don't include trailing/leading whitespace or punctuation
        while start < end and text[start] in ' \t\n.,;:':
            start += 1
        while end > start and text[end - 1] in ' \t\n.,;:':
            end -= 1
            
        # Don't break in the middle of a word
        if start > 0 and text[start - 1].isalnum() and text[start].isalnum():
            return None, None
        if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
            return None, None
            
        return start, end
    
    def _post_process_entities(self, results: List[RecognizerResult], text: str) -> List[RecognizerResult]:
        """Enhanced post-processing with better boundary handling and type validation"""
        if not results:
            return results
            
        # Known payment token prefixes
        payment_token_prefixes = {
            'tok_', 'pi_', 'ch_', 'pm_', 'cus_', 'sub_', 'in_', 'si_',
            'seti_', 'src_', 'prv_', 'acct_'
        }
        
        # Known organization names (card brands, banks, etc.)
        known_organizations = {
            'visa', 'mastercard', 'amex', 'american express', 'discover',
            'diners club', 'jcb', 'unionpay', 'maestro', 'cirrus',
            'hsbc', 'barclays', 'citibank', 'chase', 'wells fargo',
            'bank of america', 'santander', 'bbva', 'bnp', 'paribas',
            'société générale', 'crédit agricole', 'deutsche bank'
        }
        
        # Location indicators
        location_indicators = {
            'street', 'avenue', 'road', 'boulevard', 'lane', 'drive',
            'court', 'circle', 'square', 'place', 'rue', 'avenue', 'boulevard',
            'المملكة', 'مدينة', 'شارع', 'طريق', 'حي', 'منطقة'
        }
        
        # Sort by position and score (higher score first)
        sorted_results = sorted(results, key=lambda x: (x.start, -x.score))
        processed_ranges = []
        corrected_results = []
        
        for result in sorted_results:
            # Clean entity boundaries
            start, end = self._clean_entity_boundaries(text, result.start, result.end)
            if start is None or end is None:
                continue
                
            entity_text = text[start:end].strip()
            entity_type = result.entity_type
            score = result.score
            
            # Skip if this range overlaps with a higher confidence result
            if any(s <= start < e or s < end <= e or
                  (start <= s and end >= e)
                  for s, e in processed_ranges):
                continue
            
            # Get surrounding context
            context_before = text[max(0, start - 50):start].lower()
            context_after = text[end:min(len(text), end + 50)].lower()
            full_context = context_before + " " + entity_text.lower() + " " + context_after
            
            # Fix common misclassifications
            
            # Fix AU_ACN misclassification
            if entity_type == "AU_ACN" and re.match(r'^\+?\d{1,4}[-\s]?\d{6,}$', entity_text):
                entity_type = "PHONE_NUMBER"
                score = 0.95
            
            # Fix organization vs location confusion
            elif entity_type == "ORGANIZATION":
                if any(indicator in entity_text.lower() for indicator in location_indicators):
                    entity_type = "LOCATION"
                    score = 0.9
                elif re.match(r'^\d+.*$', entity_text):  # Starts with numbers
                    continue  # Skip likely false positives
            
            # Fix person vs location confusion for Arabic
            elif entity_type == "PERSON" and any(word in entity_text for word in ['المملكة', 'مدينة', 'شارع']):
                entity_type = "LOCATION"
                score = 0.9
            
            # Fix payment token detection
            elif any(entity_text.startswith(prefix) for prefix in payment_token_prefixes):
                entity_type = "PAYMENT_TOKEN"
                score = 0.99
            
            # Fix organization names
            elif entity_type == "PERSON" and entity_text.lower() in known_organizations:
                entity_type = "ORGANIZATION"
                score = 0.95
            
            # Handle line breaks
            if '\n' in entity_text:
                parts = entity_text.split('\n')
                # Only keep the relevant part
                clean_part = next((part.strip() for part in parts if part.strip() and 
                                 not any(word in part.lower() for word in ['email', 'phone', 'fax', 'address'])), None)
                if clean_part:
                    entity_text = clean_part
                    end = start + len(clean_part)
            
            # Add to processed ranges and corrected results
            processed_ranges.append((start, end))
            corrected_results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer=result.analysis_explanation.recognizer if result.analysis_explanation else "PostProcessor",
                        original_score=score,
                        pattern_name=f"{entity_type} Pattern",
                        pattern=None,
                        validation_result=True,
                        textual_explanation=f"Post-processed {entity_type}: {entity_text}"
                    )
                )
            )
        
        return corrected_results
    
    def detect_and_anonymize(self, text: str, language: str = 'en') -> Dict:
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Get analyzer results
            analyzer_results = self._get_analyzer_results(text, language)
            
            # Post-process results to fix misclassifications
            analyzer_results = self._post_process_entities(analyzer_results, text)
            
            # Sort by position and filter by confidence
            analyzer_results = [
                result for result in sorted(analyzer_results, key=lambda x: x.start)
                if result.score >= self.confidence_thresholds.get(result.entity_type, 0.5)
            ]
            
            # Remove overlapping entities
            analyzer_results = self._remove_overlapping_entities(analyzer_results)
            
            # Create operator configurations for anonymization
            anonymization_config = {}
            for result in analyzer_results:
                entity_type = result.entity_type
                operator = self.operators.get(entity_type, self.operators['DEFAULT'])
                anonymization_config[result] = operator
            
            # Perform anonymization with validated operators
            try:
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=anonymization_config
                )
                
                # Replace XML-style tags with simpler format
                cleaned_text = anonymized_result.text
                for entity_type in set(result.entity_type for result in analyzer_results):
                    cleaned_text = re.sub(
                        f'<{entity_type}>[^<]+</{entity_type}>',
                        f'<{entity_type}>',
                        cleaned_text
                    )
                
            except Exception as e:
                print(f"Anonymization error: {e}")
                # Fallback manual anonymization
                cleaned_text = text
                sorted_results = sorted(analyzer_results, key=lambda x: -x.start)
                for result in sorted_results:
                    cleaned_text = (
                        cleaned_text[:result.start] +
                        f"<{result.entity_type}>" +
                        cleaned_text[result.end:]
                    )
            
            return {
                'original_text': text,
                'anonymized_text': cleaned_text,
                'entities': [
                    {
                        'type': result.entity_type,
                        'text': text[result.start:result.end],
                        'start': result.start,
                        'end': result.end,
                        'score': float(result.score)
                    }
                    for result in analyzer_results
                ],
                'language': language
            }
        except Exception as e:
            print(f"Error in detect_and_anonymize: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'original_text': text,
                'anonymized_text': text,
                'entities': [],
                'language': language
            }

    def _remove_overlapping_entities(self, entities: List[RecognizerResult]):
        if not entities:
            return entities
        
        # Sort by position and score
        entities.sort(key=lambda e: (e.start, -e.score))
        filtered = []
        current = entities[0]
        
        for entity in entities[1:]:
            if entity.start >= current.end:
                filtered.append(current)
                current = entity
            elif entity.score > current.score:  # Fix the variable name from result to entity
                current = entity
        
        filtered.append(current)
        return filtered
    
    def batch_detect_and_anonymize(self, texts: List[str], language: str = 'en') -> List[Dict]:
        """Process multiple texts in batch"""
        return [self.detect_and_anonymize(text, language) for text in texts]

def convert_to_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj

class RecognizerFactory:
    @staticmethod
    def create_recognizer(entity_type: str, language: str) -> PatternRecognizer:
        recognizer_map = {
            'en': {
                'SWIFT_BIC': EnglishSwiftBicRecognizer,
                'VAT_NUMBER': EnglishVatRecognizer,
                'INVOICE_NUMBER': EnglishInvoiceRecognizer,
                # ... etc
            },
            'fr': {
                'SWIFT_BIC': FrenchSwiftBicRecognizer,
                'VAT_NUMBER': FrenchVatRecognizer,
                'INVOICE_NUMBER': FrenchInvoiceRecognizer,
            },
            'ar': {
                'SWIFT_BIC': ArabicSwiftBicRecognizer,
                'VAT_NUMBER': ArabicVatRecognizer,
                'INVOICE_NUMBER': ArabicInvoiceRecognizer,
            }
        }
        return recognizer_map[language][entity_type]()

class EnhancedCreditCardRecognizer(PatternRecognizer):
    def __init__(self):
        self.context_patterns = {
            'CREDIT_CARD': [
                r'(?i)(?:credit|debit|card|visa|mastercard|amex)',
                r'(?i)(?:expir|cvv|security code)'
            ],
            'TAX_ID': [
                r'(?i)(?:tax|fiscal|vat|tva)',
                r'(?i)(?:number|id|identifier)'
            ],
            'REFERENCE': [
                r'(?i)(?:ref|reference|invoice|document)',
                r'(?i)(?:number|id|#)'
            ]
        }
        
    def analyze_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        context = text[max(0, start - window):min(len(text), end + window)]
        scores = defaultdict(int)
        
        for entity_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context):
                    scores[entity_type] += 1
                    
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'UNKNOWN'

class EnhancedCryptoRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__()
        self.xrp_pattern = Pattern(
            "Ripple_Address",
            r"\br[a-zA-Z0-9]{24,34}\b",
            0.99
        )
        self.crypto_context = [
            r'(?i)(?:wallet|address|crypto|bitcoin|eth|xrp|ripple)',
            r'(?i)(?:transaction|block|hash|token)'
        ]
    
    def validate_ripple_address(self, address: str) -> bool:
        # Add base58 check and checksum validation for XRP
        try:
            if not address.startswith('r'):
                return False
            decoded = base58.b58decode_check(address)
            return len(decoded) == 20
        except:
            return False

    def override_spacy_classification(self, text: str, entities: List[Dict]) -> List[Dict]:
        for entity in entities:
            if (entity['type'] == 'PERSON' and 
                self.validate_ripple_address(entity['text'])):
                entity['type'] = 'CRYPTO_WALLET'
                entity['subtype'] = 'XRP'
        return entities

class EnhancedEvaluationMetrics(EvaluationMetrics):
    def __init__(self):
        super().__init__()
        self.partial_matches = defaultdict(list)
        
    def calculate_overlap(self, pred_start: int, pred_end: int, 
                         true_start: int, true_end: int) -> float:
        overlap = min(pred_end, true_end) - max(pred_start, true_start)
        union = max(pred_end, true_end) - min(pred_start, true_start)
        return overlap / union if union > 0 else 0
    
    def update_with_partial_matches(self, prediction: Dict, truth: Dict, 
                                  overlap_threshold: float = 0.5):
        overlap = self.calculate_overlap(
            prediction['start'], prediction['end'],
            truth['start'], truth['end']
        )
        
        if overlap >= overlap_threshold:
            self.partial_matches[prediction['type']].append({
                'overlap': overlap,
                'prediction': prediction,
                'truth': truth
            })

class PIIDetector:
    """Simple PII detector using regular expressions."""
    
    def __init__(self):
        """Initialize the PII detector with regex patterns."""
        self.patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE_NUMBER": r"(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "BANK_ACCOUNT": r"\b\d{8,17}\b"  # Basic pattern for bank account numbers
        }
        
        # Initialize country codes for IBAN validation
        self.country_codes = {country.alpha_2: country for country in pycountry.countries}
    
    def detect_pii(self, content: str) -> Dict:
        """Detect PII in the given content.
        
        Args:
            content: The text content to analyze.
            
        Returns:
            Dict containing detection results and masked text.
        """
        entities = []
        masked_text = content
        
        # Detect each type of PII
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                value = match.group()
                start, end = match.span()
                
                # Additional validation for specific types
                if pii_type == "BANK_ACCOUNT" and self._is_valid_iban(value):
                    pii_type = "IBAN"
                
                entities.append({
                    "type": pii_type,
                    "value": value,
                    "start": start,
                    "end": end
                })
                
                # Mask the detected PII
                masked_text = masked_text[:start] + f"[{pii_type}]" + masked_text[end:]
        
        return {
            "has_pii": len(entities) > 0,
            "entities": entities,
            "masked_text": masked_text
        }
    
    def _is_valid_iban(self, account_number: str) -> bool:
        """Check if a bank account number is a valid IBAN.
        
        Args:
            account_number: The account number to validate.
            
        Returns:
            bool indicating if the number is a valid IBAN.
        """
        # Remove spaces and convert to uppercase
        account_number = account_number.replace(" ", "").upper()
        
        # Check if it starts with a valid country code
        country_code = account_number[:2]
        if country_code not in self.country_codes:
            return False
            
        # Basic IBAN validation (length and format)
        if not re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$", account_number):
            return False
            
        return True