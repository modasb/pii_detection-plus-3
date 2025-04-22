#!/usr/bin/env python

"""
Script to check CUDA availability and install the correct PyTorch version.
"""

import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return its output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def check_cuda():
    """Check if CUDA is installed and available."""
    print("Checking CUDA availability...")
    
    # Check if NVIDIA drivers are installed
    if platform.system() == "Windows":
        nvidia_smi = run_command("nvidia-smi")
        if not nvidia_smi:
            print("NVIDIA drivers not found or not working properly.")
            return False
    else:
        nvidia_smi = run_command("nvidia-smi")
        if not nvidia_smi:
            print("NVIDIA drivers not found or not working properly.")
            return False
    
    # Try to import torch and check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA is available! Version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("PyTorch is installed but CUDA is not available.")
            print(f"PyTorch version: {torch.__version__}")
            return False
    except ImportError:
        print("PyTorch is not installed.")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\nInstalling PyTorch with CUDA support...")
    
    # Determine the correct installation command based on the platform
    if platform.system() == "Windows":
        # For Windows, use pip with the PyTorch website command
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # For Linux/Mac, use pip with the PyTorch website command
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    success = run_command(cmd)
    if success:
        print("PyTorch with CUDA support installed successfully!")
    else:
        print("Failed to install PyTorch with CUDA support.")
    
    return success

def main():
    """Main function to check and install CUDA-enabled PyTorch."""
    print("=== CUDA and PyTorch Setup ===\n")
    
    # Check if CUDA is available
    cuda_available = check_cuda()
    
    if cuda_available:
        print("\nCUDA is already available with PyTorch. No action needed.")
        return
    
    # If CUDA is not available, install PyTorch with CUDA support
    print("\nCUDA is not available with the current PyTorch installation.")
    user_input = input("Do you want to install PyTorch with CUDA support? (y/n): ")
    
    if user_input.lower() == 'y':
        install_pytorch_cuda()
        
        # Verify installation
        print("\nVerifying installation...")
        if check_cuda():
            print("\nSuccess! PyTorch with CUDA support is now installed.")
            print("You can now run your PII detection with GPU acceleration.")
        else:
            print("\nPyTorch installation completed, but CUDA is still not available.")
            print("This might be due to incompatible NVIDIA drivers or missing CUDA toolkit.")
            print("Please ensure your NVIDIA drivers are up to date and CUDA toolkit is installed.")
    else:
        print("Installation cancelled.")

if __name__ == "__main__":
    main() 