# Multilingual PII Detection System

A comprehensive system for detecting and masking Personally Identifiable Information (PII) in multilingual text, with enhanced support for Arabic text processing.

## Features

- **Multilingual Support**: Detects PII in English, French, and Arabic text
- **Enhanced Arabic Processing**: Specialized detection for Arabic names, addresses, and country-specific identifiers
- **Multiple Detection Methods**:
  - NER models for each supported language
  - Regex pattern matching
  - Presidio integration
  - Specialized Arabic text processing with bidirectional support
- **Memory-Optimized Version**: Lightweight implementation for resource-constrained environments
- **Flexible Masking Strategies**: Configurable redaction approaches for different entity types
- **Parallel Processing**: Thread-based parallel execution for improved performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pii-detection.git
cd pii-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required SpaCy models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download xx_ent_wiki_sm
```

## Usage

### Basic Usage

```python
from pii_protection.pii import PIIProtectionLayer

# Initialize the PII protection layer
pii_layer = PIIProtectionLayer()

# Process English text
result = pii_layer.process_document("My name is John Smith and my email is john.smith@example.com", language="en")
print(result['masked_text'])

# Process French text
result = pii_layer.process_document("Je m'appelle Pierre Dupont et mon numéro est 01 23 45 67 89", language="fr")
print(result['masked_text'])

# Process Arabic text
result = pii_layer.process_document("اسمي محمد أحمد وبريدي الإلكتروني هو mohammed.ahmed@example.sa", language="ar")
print(result['masked_text'])
```

### Batch Processing

```python
from pii_protection.pii import PIIProtectionLayer

# Initialize the PII protection layer
pii_layer = PIIProtectionLayer()

# Process multiple documents in batch
texts = [
    "My email is john@example.com",
    "My phone number is 555-123-4567",
    "My credit card is 4111-1111-1111-1111"
]
results = pii_layer.batch_process(texts, language="en")

# Print masked texts
for result in results:
    print(result['masked_text'])
```

### Memory-Optimized Version

```python
from pii_protection.pii_light import PIIProtectionLayerLight

# Initialize the lightweight PII protection layer
pii_layer = PIIProtectionLayerLight()

# Process text
result = pii_layer.process_document("My name is John Smith and my email is john.smith@example.com", language="en")
print(result['masked_text'])
```

## Evaluation Scripts

The repository includes several evaluation scripts:

- `test_arabic_pii.py`: Tests the enhanced Arabic PII detection capabilities
- `test_arabic_pii_light.py`: Tests the memory-optimized version with Arabic text
- `pii_evaluation_light.py`: Evaluates PII detection across multiple languages with memory optimization
- `pii_evaluation_notebook.ipynb`: Jupyter notebook for interactive evaluation and visualization

Run the evaluation scripts to see the system in action:

```bash
python test_arabic_pii.py
```

## Arabic PII Detection

The system includes specialized support for Arabic PII detection:

- Country-specific phone number formats (Saudi Arabia, UAE, Egypt, Qatar)
- National ID formats for different Arab countries
- Passport number detection
- Arabic name recognition
- Address detection with Arabic street/location indicators
- Arabic email domain detection

## Configuration

The system is highly configurable:

- Confidence thresholds can be adjusted per language
- Regex patterns can be customized
- Masking strategies can be modified for different entity types

## Memory Optimization

If you encounter memory issues, use the lightweight version:

```python
from pii_protection.pii_light import PIIProtectionLayerLight

pii_layer = PIIProtectionLayerLight()
```

The lightweight version uses only regex-based detection without loading large NLP models.

## Performance Optimization

The system uses parallel processing with ThreadPoolExecutor to run detection methods concurrently:

1. Regex pattern matching
2. NER model detection
3. Presidio analyzer detection

This approach provides better performance while avoiding the complexities of asyncio.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Presidio](https://github.com/microsoft/presidio) for PII analysis components
- [SpaCy](https://spacy.io/) for NLP capabilities
- [Hugging Face Transformers](https://huggingface.co/transformers/) for NER models
- [Arabic Reshaper](https://github.com/mpcabd/python-arabic-reshaper) for Arabic text processing
