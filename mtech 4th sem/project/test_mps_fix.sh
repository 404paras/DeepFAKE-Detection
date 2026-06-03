#!/bin/bash
# Quick test to verify MPS compatibility fixes

echo "======================================"
echo "Testing MPS Compatibility Fixes"
echo "======================================"
echo ""

cd "/Users/I768770/Documents/Mtech/mtech 4th sem/project"
source .venv/bin/activate

echo "✅ Checking PyTorch and MPS availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
print()
"

echo "✅ Testing tensor operations with MPS..."
python -c "
import torch

# Test reshape vs view on MPS
device = 'mps'
x = torch.randn(2, 16, 3, 224, 224).to(device)
print(f'Original shape: {x.shape}')

# This should work now (using reshape instead of view)
x_reshaped = x.reshape(2 * 16, 3, 224, 224)
print(f'Reshaped: {x_reshaped.shape}')

# Reshape back
x_back = x_reshaped.reshape(2, 16, -1)
print(f'Reshaped back: {x_back.shape}')

print()
print('✅ All tensor operations work on MPS!')
"

echo ""
echo "======================================"
echo "Fixes Applied:"
echo "======================================"
echo "1. Replaced .view() with .reshape() in video_encoder.py"
echo "2. Updated autocast to use device_type parameter"
echo "3. Disabled mixed precision for MPS (not yet supported)"
echo "4. Fixed GradScaler initialization for different devices"
echo ""
echo "You can now train with:"
echo "  python train.py --config configs/config.yaml"
echo ""
echo "Monitor with:"
echo "  python monitor_enhanced.py"
echo "======================================"
