import pytest
from pii_detector import MultilingualPIIDetector

@pytest.fixture
def pii_detector():
    return MultilingualPIIDetector()

def test_english_person_detection(pii_detector):
    text = "My name is John Smith and I work at Microsoft."
    result = pii_detector.detect_and_anonymize(text, "en")
    assert "<PERSONNE>" in result["anonymized_text"]
    assert any(entity["type"] == "PERSON" for entity in result["entities"])

def test_french_person_detection(pii_detector):
    text = "Je m'appelle Jean Dupont et je travaille chez Renault."
    result = pii_detector.detect_and_anonymize(text, "fr")
    assert "<PERSONNE>" in result["anonymized_text"]
    assert any(entity["type"] == "PERSON" for entity in result["entities"])

def test_email_detection(pii_detector):
    text = "Contact me at john.doe@example.com"
    result = pii_detector.detect_and_anonymize(text, "en")
    assert "@" in result["anonymized_text"]
    assert any(entity["type"] == "EMAIL_ADDRESS" for entity in result["entities"])

def test_phone_number_detection(pii_detector):
    text = "Call me at +1 (555) 123-4567"
    result = pii_detector.detect_and_anonymize(text, "en")
    assert "*" in result["anonymized_text"]
    assert any(entity["type"] == "PHONE_NUMBER" for entity in result["entities"])

def test_credit_card_detection(pii_detector):
    text = "My credit card number is 4532 0159 6784 1234"
    result = pii_detector.detect_and_anonymize(text, "en")
    assert "#" in result["anonymized_text"]
    assert any(entity["type"] == "CREDIT_CARD" for entity in result["entities"])

def test_batch_processing(pii_detector):
    texts = [
        "John Smith lives in New York",
        "Marie Dubois habite à Paris"
    ]
    results = pii_detector.batch_detect_and_anonymize(texts, "en")
    assert len(results) == 2
    assert all("<PERSONNE>" in result["anonymized_text"] for result in results)

def test_empty_text(pii_detector):
    result = pii_detector.detect_and_anonymize("", "en")
    assert result["original_text"] == ""
    assert result["anonymized_text"] == ""
    assert result["entities"] == []

def test_no_pii_text(pii_detector):
    text = "This text contains no personal information."
    result = pii_detector.detect_and_anonymize(text, "en")
    assert result["original_text"] == text
    assert result["anonymized_text"] == text
    assert result["entities"] == [] 