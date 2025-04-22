#!/usr/bin/env python

"""
Simple script to test if GPU is working with PyTorch.
"""

import torch
import time
import sys

def test_gpu():
    """Test if GPU is available and working with PyTorch."""
    print("=== PyTorch GPU Test ===\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        # Set current device
        torch.cuda.set_device(0)
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Test GPU performance
        print("\nTesting GPU performance...")
        
        # Matrix multiplication on GPU
        try:
            # Start with smaller matrices to avoid CUDA out of memory
            for size in [1000, 2000, 5000, 10000]:
                print(f"\nTesting with {size}x{size} matrices:")
                
                # GPU computation
                start_time = time.time()
                x_gpu = torch.randn(size, size, device='cuda')
                y_gpu = torch.randn(size, size, device='cuda')
                z_gpu = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()  # Wait for GPU to finish
                gpu_time = time.time() - start_time
                
                # CPU computation
                start_time = time.time()
                x_cpu = torch.randn(size, size)
                y_cpu = torch.randn(size, size)
                z_cpu = torch.matmul(x_cpu, y_cpu)
                cpu_time = time.time() - start_time
                
                # Print results
                print(f"  GPU time: {gpu_time:.4f} seconds")
                print(f"  CPU time: {cpu_time:.4f} seconds")
                print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
                
                # Free memory
                del x_gpu, y_gpu, z_gpu, x_cpu, y_cpu, z_cpu
                torch.cuda.empty_cache()
            
            print("\nGPU test completed successfully!")
            return True
            
        except Exception as e:
            print(f"\nError during GPU test: {str(e)}")
            return False
    else:
        print("CUDA is not available. Using CPU only.")
        print("\nTo enable GPU support, install PyTorch with CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1) 