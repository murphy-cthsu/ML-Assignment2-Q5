#!/bin/bash

# Go Style Detection - Correct Training Script
# This script trains on train_set/ (NOT test_set/query_set!)

echo "======================================================================="
echo "Go Style Detection - Training with Validation"
echo "======================================================================="
echo ""
echo "CRITICAL: Training on train_set/ (600 training players)"
echo "          NOT on test_set/query_set (test players)!"
echo ""
echo "Validation will show if model is learning meaningful style features"
echo "======================================================================="
echo ""

# Check if train_set exists
if [ ! -d "train_set" ]; then
    echo "ERROR: train_set/ directory not found!"
    echo "Expected structure:"
    echo "  train_set/"
    echo "    ├── 1.sgf"
    echo "    ├── 2.sgf"
    echo "    └── ..."
    exit 1
fi

# Count SGF files
NUM_TRAIN=$(ls train_set/*.sgf 2>/dev/null | wc -l)
echo "Found $NUM_TRAIN training SGF files"

if [ "$NUM_TRAIN" -eq 0 ]; then
    echo "ERROR: No SGF files found in train_set/"
    exit 1
fi

echo ""
echo "Starting training with validation..."
echo "======================================================================="
echo ""

# Default: medium preset with validation
PRESET="${1:-medium}"

case "$PRESET" in
    test)
        echo "Preset: TEST (quick verification)"
        conda run -n ml_hw2 python train_with_validation.py \
            --train_dir train_set \
            --output_dir models/test_with_val \
            --epochs 5 \
            --batch_size 4 \
            --num_samples 100 \
            --max_moves 30 \
            --hidden_channels 64 \
            --num_blocks 2 \
            --move_embed_dim 64 \
            --num_heads 4 \
            --num_transformer_layers 2 \
            --style_embed_dim 128 \
            --lr 0.001 \
            --val_interval 2 \
            --num_val_players 20 \
            --num_workers 0
        ;;
    
    small)
        echo "Preset: SMALL (~3 hours)"
        conda run -n ml_hw2 python train_with_validation.py \
            --train_dir train_set \
            --output_dir models/small_with_val \
            --epochs 30 \
            --batch_size 16 \
            --num_samples 500 \
            --max_moves 50 \
            --hidden_channels 96 \
            --num_blocks 3 \
            --move_embed_dim 96 \
            --num_heads 6 \
            --num_transformer_layers 2 \
            --style_embed_dim 192 \
            --lr 0.0005 \
            --val_interval 5 \
            --num_val_players 50 \
            --num_workers 4
        ;;
    
    medium)
        echo "Preset: MEDIUM (~10 hours)"
        conda run -n ml_hw2 python train_with_validation.py \
            --train_dir train_set \
            --output_dir models/medium_with_val \
            --epochs 50 \
            --batch_size 16 \
            --num_samples 1000 \
            --max_moves 75 \
            --hidden_channels 128 \
            --num_blocks 3 \
            --move_embed_dim 128 \
            --num_heads 8 \
            --num_transformer_layers 3 \
            --style_embed_dim 256 \
            --lr 0.0003 \
            --val_interval 5 \
            --num_val_players 50 \
            --num_workers 4
        ;;
    
    large)
        echo "Preset: LARGE (~40 hours)"
        conda run -n ml_hw2 python train_with_validation.py \
            --train_dir train_set \
            --output_dir models/large_with_val \
            --epochs 100 \
            --batch_size 12 \
            --num_samples 2000 \
            --max_moves 100 \
            --hidden_channels 192 \
            --num_blocks 5 \
            --move_embed_dim 192 \
            --num_heads 12 \
            --num_transformer_layers 4 \
            --style_embed_dim 384 \
            --lr 0.0002 \
            --val_interval 5 \
            --num_val_players 50 \
            --num_workers 4
        ;;
    
    *)
        echo "Unknown preset: $PRESET"
        echo "Usage: $0 [test|small|medium|large]"
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "Training completed!"
echo "Check validation metrics in: models/*/training_log.txt"
echo "======================================================================="
