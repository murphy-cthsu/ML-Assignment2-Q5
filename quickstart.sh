#!/bin/bash

# Quick Start Script for Go Style Detection
# This script helps you get started with training and inference

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Go Playing Style Detection - Quick Start"
echo "========================================================================"

# Check if conda environment exists
if ! conda env list | grep -q "ml_hw2"; then
    echo -e "${RED}Error: Conda environment 'ml_hw2' not found${NC}"
    echo "Please run: conda create -n ml_hw2 python=3.10"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found conda environment 'ml_hw2'"

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ml_hw2

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC} PyTorch not found in environment"
    echo "Installing PyTorch..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install tqdm
fi

echo -e "${GREEN}✓${NC} PyTorch is available"

# Run system test
echo ""
echo "========================================================================"
echo "Running System Test"
echo "========================================================================"
python test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Setup Complete! You can now:"
    echo "========================================================================"
    echo ""
    echo "1. Run a quick training test (5 epochs, small batch):"
    echo "   python train_style_model.py \\"
    echo "       --model_type triplet \\"
    echo "       --epochs 5 \\"
    echo "       --batch_size 16 \\"
    echo "       --hidden_channels 128 \\"
    echo "       --num_blocks 5 \\"
    echo "       --output_dir models/test_run"
    echo ""
    echo "2. Run full training (recommended settings):"
    echo "   python train_style_model.py \\"
    echo "       --model_type triplet \\"
    echo "       --epochs 30 \\"
    echo "       --batch_size 32 \\"
    echo "       --hidden_channels 256 \\"
    echo "       --num_blocks 10 \\"
    echo "       --embedding_dim 512 \\"
    echo "       --output_dir models/triplet_model"
    echo ""
    echo "3. Run inference (after training):"
    echo "   python infer_style.py \\"
    echo "       --model_path models/triplet_model/best_model.pth \\"
    echo "       --query_dir test_set/query_set \\"
    echo "       --candidate_dir test_set/cand_set \\"
    echo "       --output submission.csv"
    echo ""
    echo "4. Read the documentation:"
    echo "   cat README_STYLE_DETECTION.md"
    echo ""
    echo "========================================================================"
else
    echo -e "${RED}✗ System test failed${NC}"
    exit 1
fi
