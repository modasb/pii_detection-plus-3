#!/usr/bin/env python

"""
GPU-optimized evaluation script for PII detection.
This script tests the performance of PII detection on GPU vs CPU.
"""

import os
import time
import json
import logging
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
import gc

from pii_protection.pii import PIIProtectionLayer, PIIConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def print_gpu_info():
    """Print detailed information about available GPUs."""
    print("\n=== GPU Information ===")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = gpu_props.total_memory / (1024 ** 3)  # Convert to GB
            
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  - Total memory: {gpu_memory:.2f} GB")
            print(f"  - CUDA capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  - Multi-processor count: {gpu_props.multi_processor_count}")
            
        # Current device info
        current_device = torch.cuda.current_device()
        print(f"\nCurrent device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        print(f"Memory reserved: {memory_reserved:.2f} MB")
    else:
        print("No GPU available. Using CPU for processing.")
        print("To enable GPU, ensure PyTorch is installed with CUDA support")
    
    print("\n" + "="*50)

def compare_gpu_cpu_performance(texts, languages):
    """Compare PII detection performance between GPU and CPU."""
    results = []
    
    # Force GPU usage
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize PII layer with GPU
        gpu_pii_layer = PIIProtectionLayer()
        
        print("\n=== Testing GPU Performance ===")
        for lang, text_list in zip(languages, texts):
            for i, text in enumerate(text_list):
                print(f"\nProcessing {lang.upper()} text {i+1} on GPU...")
                
                # Measure GPU processing time
                start_time = time.time()
                result = gpu_pii_layer.process_document(text.strip(), language=lang)
                gpu_time = time.time() - start_time
                
                print(f"GPU processing time: {gpu_time:.3f} seconds")
                print(f"Entities found: {result['statistics']['total_entities']}")
                
                # Record result
                results.append({
                    'language': lang,
                    'text_id': i+1,
                    'text_length': len(text),
                    'gpu_time': gpu_time,
                    'entities_found': result['statistics']['total_entities'],
                    'device': 'GPU'
                })
                
                # Clear GPU cache after each text
                torch.cuda.empty_cache()
                gc.collect()
    
    # Test on CPU
    # Force CPU usage by setting CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Initialize PII layer for CPU
    cpu_pii_layer = PIIProtectionLayer()
    
    print("\n=== Testing CPU Performance ===")
    for lang, text_list in zip(languages, texts):
        for i, text in enumerate(text_list):
            print(f"\nProcessing {lang.upper()} text {i+1} on CPU...")
            
            # Measure CPU processing time
            start_time = time.time()
            result = cpu_pii_layer.process_document(text.strip(), language=lang)
            cpu_time = time.time() - start_time
            
            print(f"CPU processing time: {cpu_time:.3f} seconds")
            print(f"Entities found: {result['statistics']['total_entities']}")
            
            # Record result
            results.append({
                'language': lang,
                'text_id': i+1,
                'text_length': len(text),
                'cpu_time': cpu_time,
                'entities_found': result['statistics']['total_entities'],
                'device': 'CPU'
            })
            
            # Force garbage collection
            gc.collect()
    
    # Restore GPU visibility
    if torch.cuda.is_available():
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    return pd.DataFrame(results)

def print_performance_comparison(df):
    """Print performance comparison between GPU and CPU."""
    print("\n=== Performance Comparison: GPU vs CPU ===")
    
    # Reshape data for comparison
    gpu_data = df[df['device'] == 'GPU'].copy()
    cpu_data = df[df['device'] == 'CPU'].copy()
    
    # Merge data
    comparison = pd.merge(
        gpu_data, 
        cpu_data,
        on=['language', 'text_id', 'text_length', 'entities_found'],
        suffixes=('_gpu', '_cpu')
    )
    
    # Calculate speedup
    if 'gpu_time' in comparison.columns and 'cpu_time' in comparison.columns:
        comparison['speedup'] = comparison['cpu_time'] / comparison['gpu_time']
    
    # Print comparison table
    table_data = []
    for _, row in comparison.iterrows():
        table_data.append([
            row['language'],
            row['text_id'],
            row['text_length'],
            row['entities_found'],
            f"{row.get('gpu_time', 0):.3f}",
            f"{row.get('cpu_time', 0):.3f}",
            f"{row.get('speedup', 0):.2f}x"
        ])
    
    print(tabulate(
        table_data,
        headers=["Language", "Text ID", "Length", "Entities", "GPU Time (s)", "CPU Time (s)", "Speedup"],
        tablefmt="grid"
    ))
    
    # Print summary
    if len(comparison) > 0 and 'speedup' in comparison.columns:
        avg_speedup = comparison['speedup'].mean()
        print(f"\nAverage speedup with GPU: {avg_speedup:.2f}x")
    
    print("\n" + "="*50)

def main():
    """Main function to run GPU vs CPU performance comparison."""
    # Force garbage collection before starting
    gc.collect()
    
    print("=== GPU-Optimized PII Detection Evaluation ===\n")
    
    # Print GPU information
    print_gpu_info()
    
    # Example texts for different languages (varying lengths)
    texts = [
        # English texts
        [
            "My name is John Smith and my email is john.smith@example.com. My phone number is 555-123-4567.",
            """
            Please contact Sarah Johnson at sarah.j@company.org or call her at (123) 456-7890.
            She lives at 123 Main Street, New York, NY 10001. Her social security number is 123-45-6789.
            The meeting will be held on January 15, 2023 at 3:00 PM. Please bring your ID card.
            """
        ],
        # French texts
        [
            "Je m'appelle Pierre Dupont et mon numéro de téléphone est 01 23 45 67 89.",
            """
            Contactez Marie Bernard à marie.bernard@exemple.fr ou au 06 12 34 56 78.
            Elle habite au 15 rue de la Paix, 75001 Paris. Son numéro de sécurité sociale est 1 23 45 67 890 123 45.
            La réunion aura lieu le 15 janvier 2023 à 15h00. Veuillez apporter votre carte d'identité.
            """
        ],
        # Arabic texts
        [
            """
            اسمي محمد أحمد وأعمل في شركة الاتصالات السعودية.
            رقم هاتفي هو 0512345678
            بريدي الإلكتروني هو mohammed.ahmed@example.sa
            """,
            """
            رقم الهاتف السعودي: +966512345678
            رقم الهوية السعودية: 1234567890
            رقم الهوية المصرية: 12345678901234
            رقم الهوية الإماراتية: 784123456789012
            رقم جواز السفر السعودي: Z1234567
            رقم جواز السفر المصري: A12345678
            المدير التنفيذي هو محمد عبد الله العتيبي
            المهندس المسؤول أحمد محمد علي حسن
            الدكتورة فاطمة عبد الرحمن الزهراني
            """
        ]
    ]
    
    languages = ["en", "fr", "ar"]
    
    # Compare performance
    results_df = compare_gpu_cpu_performance(texts, languages)
    
    # Print comparison
    print_performance_comparison(results_df)
    
    # Save results
    results_df.to_csv("gpu_cpu_performance_comparison.csv", index=False)
    print("\nResults saved to gpu_cpu_performance_comparison.csv")

if __name__ == "__main__":
    main() 