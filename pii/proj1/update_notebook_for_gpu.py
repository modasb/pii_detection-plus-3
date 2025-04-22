#!/usr/bin/env python

"""
Script to update the pii_evaluation.ipynb notebook to use GPU.
"""

import json
import os
import sys
import glob

def update_notebook_for_gpu(notebook_path):
    """Update a Jupyter notebook to use GPU for PII detection."""
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
        gpu_check_cell = {
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
                "    print(\"\\nGPU will be used for PII detection!\")\n",
                "else:\n",
                "    print(\"CUDA is not available. Using CPU only.\")\n",
                "    print(\"To enable GPU support, install PyTorch with CUDA:\")\n",
                "    print(\"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\")"
            ],
            "outputs": []
        }
        
        notebook['cells'].insert(0, gpu_check_cell)
        
        # Find and update PII initialization cells
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Check if this cell initializes PIIProtectionLayer
                if 'PIIProtectionLayer' in source and '=' in source:
                    # Update to use force_gpu parameter
                    new_source = []
                    for line in cell['source']:
                        if 'PIIProtectionLayer' in line and '=' in line:
                            # Extract indentation
                            indent = ''
                            for char in line:
                                if char.isspace():
                                    indent += char
                                else:
                                    break
                            
                            # Add force_gpu parameter
                            if 'force_gpu' not in line:
                                if line.strip().endswith(')'):
                                    # Replace the closing parenthesis
                                    new_line = line.replace(')', ', force_gpu=torch.cuda.is_available())')
                                else:
                                    # Add parameter at the end
                                    new_line = line.rstrip() + ', force_gpu=torch.cuda.is_available()\n'
                                new_source.append(new_line)
                            else:
                                new_source.append(line)
                        else:
                            new_source.append(line)
                    
                    # Update the cell source
                    cell['source'] = new_source
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Successfully updated {notebook_path} to use GPU")
        return True
    
    except Exception as e:
        print(f"Error updating notebook: {str(e)}")
        return False

def main():
    """Main function to update a notebook for GPU usage."""
    # Find all .ipynb files in the current directory
    notebook_files = glob.glob("*.ipynb")
    
    if not notebook_files:
        print("No Jupyter notebook files found in the current directory.")
        return
    
    print(f"Found {len(notebook_files)} notebook files:")
    for i, file in enumerate(notebook_files, 1):
        print(f"{i}. {file}")
    
    # If only one notebook, use it automatically
    if len(notebook_files) == 1:
        notebook_path = notebook_files[0]
        print(f"\nAutomatically selecting the only notebook: {notebook_path}")
    else:
        # Let user choose which notebook to update
        try:
            choice = int(input("\nEnter the number of the notebook to update: "))
            if 1 <= choice <= len(notebook_files):
                notebook_path = notebook_files[choice - 1]
            else:
                print("Invalid choice. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return
    
    # Update the selected notebook
    update_notebook_for_gpu(notebook_path)

if __name__ == "__main__":
    main() 