#!/bin/bash
# Helper script to run commands with the virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run the command passed as arguments
"$@"

                                                                                                      
  1. Training: python train.py --config configs/config.yaml                                           
  2. Evaluation: python evaluate.py --checkpoint checkpoints/best.pth                               
  3. Inference: python demo.py --video sample.mp4  
  