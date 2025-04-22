"""Constants for PII detection and analysis."""

# Entity Types
ENTITY_EMAIL = "EMAIL_ADDRESS"
ENTITY_PHONE = "PHONE_NUMBER"
ENTITY_SSN = "SSN"
ENTITY_CREDIT_CARD = "CREDIT_CARD"
ENTITY_BANK_ACCOUNT = "BANK_ACCOUNT"
ENTITY_IBAN = "IBAN"
ENTITY_SWIFT_BIC = "SWIFT_BIC"
ENTITY_VAT = "VAT_NUMBER"
ENTITY_INVOICE = "INVOICE_NUMBER"
ENTITY_PERSON = "PERSON"
ENTITY_LOCATION = "LOCATION"
ENTITY_ORGANIZATION = "ORGANIZATION"
ENTITY_CRYPTO = "CRYPTO_WALLET"
ENTITY_PAYMENT_TOKEN = "PAYMENT_TOKEN"
ENTITY_CURRENCY = "CURRENCY_AMOUNT"

# Entity Type Mappings (for normalization)
ENTITY_TYPE_MAPPING = {
    "EMAIL": ENTITY_EMAIL,
    "email": ENTITY_EMAIL,
    "PHONE": ENTITY_PHONE,
    "phone": ENTITY_PHONE,
    "SSN": ENTITY_SSN,
    "ssn": ENTITY_SSN,
    "CREDIT_CARD": ENTITY_CREDIT_CARD,
    "credit_card": ENTITY_CREDIT_CARD,
    "BANK_ACCOUNT": ENTITY_BANK_ACCOUNT,
    "bank_account": ENTITY_BANK_ACCOUNT,
    "IBAN": ENTITY_IBAN,
    "iban": ENTITY_IBAN,
    "SWIFT_BIC": ENTITY_SWIFT_BIC,
    "swift_bic": ENTITY_SWIFT_BIC,
    "VAT_NUMBER": ENTITY_VAT,
    "vat_number": ENTITY_VAT,
    "INVOICE_NUMBER": ENTITY_INVOICE,
    "invoice_number": ENTITY_INVOICE,
    "PERSON": ENTITY_PERSON,
    "person": ENTITY_PERSON,
    "LOCATION": ENTITY_LOCATION,
    "location": ENTITY_LOCATION,
    "ORGANIZATION": ENTITY_ORGANIZATION,
    "organization": ENTITY_ORGANIZATION,
    "CRYPTO_WALLET": ENTITY_CRYPTO,
    "crypto_wallet": ENTITY_CRYPTO,
    "PAYMENT_TOKEN": ENTITY_PAYMENT_TOKEN,
    "payment_token": ENTITY_PAYMENT_TOKEN,
    "CURRENCY_AMOUNT": ENTITY_CURRENCY,
    "currency_amount": ENTITY_CURRENCY,
}

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    ENTITY_EMAIL: 0.7,
    ENTITY_CREDIT_CARD: 0.8,
    ENTITY_BANK_ACCOUNT: 0.8,
    ENTITY_PHONE: 0.7,
    ENTITY_PERSON: 0.7,
    ENTITY_LOCATION: 0.7,
    ENTITY_ORGANIZATION: 0.8,
    ENTITY_CRYPTO: 0.9,
    ENTITY_IBAN: 0.9,
    ENTITY_SWIFT_BIC: 0.9,
    ENTITY_SSN: 0.9,
    ENTITY_VAT: 0.8,
    ENTITY_INVOICE: 0.7,
    ENTITY_PAYMENT_TOKEN: 0.9,
    ENTITY_CURRENCY: 0.7,
}

# Pre-compiled Regex Patterns
import re

EMAIL_REGEX_SIMPLE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
EMAIL_REGEX_COMPLEX = re.compile(
    r"(?i)(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
)

PHONE_REGEX = re.compile(r"(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
IBAN_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$")

# Context Words
CONTEXT_WORDS = {
    ENTITY_EMAIL: ["email", "mail", "contact", "support", "info", "@", "e-mail", "courriel", "البريد"],
    ENTITY_PHONE: ["phone", "mobile", "tel", "telephone", "contact", "call", "téléphone", "portable", "fixe", "numéro"],
    ENTITY_CREDIT_CARD: ["credit", "card", "carte", "crédit", "visa", "mastercard", "american express", "amex", "paiement"],
    ENTITY_BANK_ACCOUNT: ["account", "compte", "bank", "banque", "iban", "swift", "bic", "حساب", "بنك"],
    ENTITY_VAT: ["vat", "tax", "tva", "taxe", "ضريبة", "الضريبة", "ضريبي"],
    ENTITY_INVOICE: ["invoice", "facture", "bill", "فاتورة", "رقم الفاتورة"],
    ENTITY_PERSON: ["name", "nom", "person", "personne", "اسم", "السيد", "السيدة"],
    ENTITY_LOCATION: ["address", "adresse", "location", "lieu", "عنوان", "شارع", "طريق"],
    ENTITY_ORGANIZATION: ["company", "société", "organization", "organisation", "شركة", "مؤسسة"],
    ENTITY_CRYPTO: ["crypto", "wallet", "bitcoin", "ethereum", "blockchain", "محفظة", "عملة"],
    ENTITY_PAYMENT_TOKEN: ["token", "payment", "paiement", "دفع", "رمز"],
    ENTITY_CURRENCY: ["amount", "montant", "price", "prix", "cost", "coût", "مبلغ", "سعر"]
}

# Language Codes
LANG_EN = "en"
LANG_FR = "fr"
LANG_AR = "ar"

# Model Names
ARABIC_MODEL_NAME = "hatmimoha/arabic-ner"
ENGLISH_MODEL_NAME = "en_core_web_sm"
FRENCH_MODEL_NAME = "fr_core_news_sm"
UNIVERSAL_MODEL_NAME = "xx_ent_wiki_sm" 