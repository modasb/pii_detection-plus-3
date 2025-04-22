#!/usr/bin/env python
"""
Script to test the PII detection system on the augmented_data_partial.csv file.
Includes support for French, English, and Arabic languages.
"""

import pandas as pd
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pii_protection.pii import PIIProtectionLayer

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_csv_data(file_path):
    """Load data from CSV file with proper encoding."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Successfully loaded CSV with {len(df)} rows")
    except UnicodeDecodeError:
        # Try different encodings if utf-8 fails
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            print(f"Successfully loaded CSV with {len(df)} rows using latin1 encoding")
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return None
    
    # Display the first few rows to verify the data
    print("\nSample data:")
    print(df.head())
    
    # Check available languages
    available_languages = df['language'].unique()
    print(f"\nLanguages in dataset: {available_languages}")
    
    return df

def process_text_with_pii_detection(text, language, pii_layer):
    """Process a single text with PII detection."""
    if pd.isna(text):
        return None
    
    try:
        start_time = time.time()
        result = pii_layer.process_document(text, language=language)
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        return result
    except Exception as e:
        print(f"Error processing text in {language}: {str(e)}")
        print(f"Text: {text[:100]}...")
        return None

def process_language_data(df, language, pii_layer, max_samples=20):
    """Process data for a specific language."""
    language_df = df[df['language'] == language]
    
    if len(language_df) == 0:
        print(f"No data found for language: {language}")
        return []
    
    # Limit the number of samples to process
    language_df = language_df.head(max_samples)
    
    print(f"\nProcessing {len(language_df)} {language.upper()} texts...")
    
    results = []
    for _, row in language_df.iterrows():
        text = row['augmented_text']
        result = process_text_with_pii_detection(text, language, pii_layer)
        if result:
            # Add metadata from the row
            result['text_id'] = row.get('text_id', None)
            result['augmentation_method'] = row.get('augmentation_method', None)
            results.append(result)
    
    return results

def analyze_results(results_by_language):
    """Analyze PII detection results across languages."""
    all_results = []
    for language, results in results_by_language.items():
        for result in results:
            all_results.append({
                'language': language,
                'entity_count': result['statistics']['total_entities'],
                'unique_types': len(result['statistics']['unique_types']),
                'detection_methods': result['statistics']['detection_methods'],
                'processing_time': result.get('processing_time', 0),
                'text_length': len(result['original_text']),
                'entity_density': result['statistics']['total_entities'] / len(result['original_text']) if len(result['original_text']) > 0 else 0
            })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    summary = results_df.groupby('language').agg({
        'entity_count': ['mean', 'sum', 'std'],
        'unique_types': ['mean', 'std'],
        'processing_time': ['mean', 'min', 'max'],
        'text_length': ['mean', 'std'],
        'entity_density': ['mean', 'std']
    }).reset_index()
    
    print("\nSummary Statistics:")
    print(summary)
    
    return results_df

def visualize_results(results_df):
    """Create visualizations of PII detection results."""
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Entity count by language
    sns.barplot(x='language', y='entity_count', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title('Average PII Entities Detected by Language')
    axes[0, 0].set_ylabel('Number of PII Entities')
    
    # Plot 2: Entity density by language
    sns.boxplot(x='language', y='entity_density', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title('PII Entity Density by Language')
    axes[0, 1].set_ylabel('Entity Density (entities per character)')
    
    # Plot 3: Processing time by language
    sns.boxplot(x='language', y='processing_time', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('Processing Time by Language')
    axes[1, 0].set_ylabel('Processing Time (seconds)')
    
    # Plot 4: Unique entity types by language
    sns.barplot(x='language', y='unique_types', data=results_df, ax=axes[1, 1])
    axes[1, 1].set_title('Average Unique Entity Types by Language')
    axes[1, 1].set_ylabel('Number of Unique Entity Types')
    
    plt.tight_layout()
    plt.savefig('pii_analysis_by_language.png')
    print("\nVisualization saved to pii_analysis_by_language.png")
    
    # Create additional visualizations for detection methods
    # Extract detection methods and count them
    method_counts = defaultdict(lambda: defaultdict(int))
    for _, row in results_df.iterrows():
        language = row['language']
        methods = row['detection_methods']
        for method in methods:
            method_counts[language][method] += 1
    
    # Convert to DataFrame for plotting
    method_data = []
    for language, methods in method_counts.items():
        for method, count in methods.items():
            method_data.append({
                'language': language,
                'method': method,
                'count': count
            })
    
    method_df = pd.DataFrame(method_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='language', y='count', hue='method', data=method_df)
    plt.title('Detection Methods Used by Language')
    plt.ylabel('Count')
    plt.savefig('detection_methods_by_language.png')
    print("Visualization saved to detection_methods_by_language.png")

def main():
    # Initialize the PII protection layer with support for all three languages
    pii_layer = PIIProtectionLayer(
        redaction_strategy="mask",
        confidence_threshold={
            'en': 0.6,
            'fr': 0.5,  # Lower threshold for French
            'ar': 0.5   # Lower threshold for Arabic
        }
    )
    
    # Load data from CSV
    df = load_csv_data("augmented_data_partial.csv")
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Process data for each language
    results_by_language = {}
    for language in ['en', 'fr', 'ar']:
        results = process_language_data(df, language, pii_layer)
        results_by_language[language] = results
        print(f"Processed {len(results)} {language.upper()} texts")
    
    # Analyze results
    results_df = analyze_results(results_by_language)
    
    # Visualize results
    visualize_results(results_df)
    
    # Save detailed results for further analysis
    all_results = []
    for language, results in results_by_language.items():
        all_results.extend(results)
    
    with open('pii_csv_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print("\nDetailed results saved to pii_csv_results.json")
    
    # Display sample results for each language
    print("\nSample Results:")
    for language, results in results_by_language.items():
        if results:
            print(f"\n{language.upper()} Example:")
            sample = results[0]
            print(f"Original: {sample['original_text'][:100]}...")
            print(f"Masked: {sample['masked_text'][:100]}...")
            print(f"Entities Detected: {sample['statistics']['total_entities']}")
            print(f"Entity Types: {', '.join(sample['statistics']['unique_types'])}")
            print(f"Detection Methods: {', '.join(sample['statistics']['detection_methods'])}")
            if sample['detected_entities']:
                print("Sample Entities:")
                for entity in sample['detected_entities'][:3]:  # Show first 3 entities
                    print(f"  - {entity['type']} ({entity['method']}): {entity['text']}")

if __name__ == "__main__":
    main() 