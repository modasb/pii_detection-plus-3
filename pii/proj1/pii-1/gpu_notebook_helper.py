#!/usr/bin/env python

"""
Helper script to add GPU support code to Jupyter notebooks.
"""

import json
import os
import sys

def create_gpu_check_cell():
    """Create a cell to check GPU availability in a notebook."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Check if GPU is available\n",
            "import torch\n",
            "import time\n",
            "\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"CUDA is available! Version: {torch.version.cuda}\")\n",
            "    print(f\"GPU count: {torch.cuda.device_count()}\")\n",
            "    for i in range(torch.cuda.device_count()):\n",
            "        print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")\n",
            "        \n",
            "    # Test GPU performance\n",
            "    print(\"\\nTesting GPU performance...\")\n",
            "    # Create a large tensor on GPU\n",
            "    start_time = time.time()\n",
            "    x = torch.randn(10000, 10000, device='cuda')\n",
            "    y = torch.randn(10000, 10000, device='cuda')\n",
            "    z = torch.matmul(x, y)\n",
            "    torch.cuda.synchronize()  # Wait for GPU to finish\n",
            "    gpu_time = time.time() - start_time\n",
            "    print(f\"GPU computation time: {gpu_time:.4f} seconds\")\n",
            "    \n",
            "    # Test CPU performance for comparison\n",
            "    start_time = time.time()\n",
            "    x_cpu = torch.randn(10000, 10000)\n",
            "    y_cpu = torch.randn(10000, 10000)\n",
            "    z_cpu = torch.matmul(x_cpu, y_cpu)\n",
            "    cpu_time = time.time() - start_time\n",
            "    print(f\"CPU computation time: {cpu_time:.4f} seconds\")\n",
            "    \n",
            "    print(f\"Speedup: {cpu_time / gpu_time:.2f}x\")\n",
            "else:\n",
            "    print(\"CUDA is not available. Using CPU only.\")\n",
            "    print(\"To enable GPU support, install PyTorch with CUDA:\")\n",
            "    print(\"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\")"
        ],
        "outputs": []
    }

def create_pii_gpu_cell():
    """Create a cell to initialize PII with GPU support."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Initialize PII Protection Layer with GPU support\n",
            "from pii_protection.pii import PIIProtectionLayer\n",
            "\n",
            "# Force GPU usage if available\n",
            "force_gpu = torch.cuda.is_available()\n",
            "pii_layer = PIIProtectionLayer(force_gpu=force_gpu)\n",
            "\n",
            "# Test with a sample text\n",
            "sample_text = \"My name is John Smith and my email is john.smith@example.com. My phone number is 555-123-4567.\"\n",
            "\n",
            "# Process with GPU\n",
            "start_time = time.time()\n",
            "result = pii_layer.process_document(sample_text, language=\"en\")\n",
            "processing_time = time.time() - start_time\n",
            "\n",
            "print(f\"Processing time: {processing_time:.4f} seconds\")\n",
            "print(f\"Detected entities: {result['statistics']['total_entities']}\")\n",
            "print(f\"Masked text: {result['masked_text']}\")"
        ],
        "outputs": []
    }

def add_gpu_cells_to_notebook(notebook_path):
    """Add GPU check and PII GPU cells to a Jupyter notebook."""
    try:
        # Check if file exists
        if not os.path.exists(notebook_path):
            print(f"Error: Notebook file '{notebook_path}' not found.")
            return False
        
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Create backup
        backup_path = f"{notebook_path}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"Created backup at {backup_path}")
        
        # Add GPU check cell at the beginning
        notebook['cells'].insert(0, create_gpu_check_cell())
        
        # Add PII GPU cell after the imports
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and any('import' in line for line in cell['source']):
                notebook['cells'].insert(i + 1, create_pii_gpu_cell())
                break
        else:
            # If no import cell found, add after the GPU check cell
            notebook['cells'].insert(1, create_pii_gpu_cell())
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Successfully added GPU support cells to {notebook_path}")
        return True
    
    except Exception as e:
        print(f"Error modifying notebook: {str(e)}")
        return False

def main():
    """Main function to add GPU support to a notebook."""
    if len(sys.argv) < 2:
        print("Usage: python gpu_notebook_helper.py <notebook_path>")
        print("Example: python gpu_notebook_helper.py pii_evaluation_notebook.ipynb")
        return
    
    notebook_path = sys.argv[1]
    add_gpu_cells_to_notebook(notebook_path)

if __name__ == "__main__":
    main() 