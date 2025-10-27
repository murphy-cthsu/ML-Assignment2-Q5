#!/bin/bash

# Quick training script - uses subset of data to avoid memory issues
# Since the C++ DataLoader has memory limits, we train on fewer players

cd /home/cthsu/Workspace/ML_2025/HW2/Q5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_hw2

echo "======================================================================="
echo "Go Style Detection - Training with Subset"
echo "======================================================================="
echo ""
echo "WORKAROUND: Using 100 players instead of 200 to avoid C++ memory limits"
echo "This should still give good results for style detection"
echo "======================================================================="
echo ""

# Create subset directory with 100 players
if [ ! -d "train_set_subset" ]; then
    echo "Creating train_set_subset with 100 players..."
    mkdir -p train_set_subset
    ls train_set/*.sgf | head -100 | while read f; do
        ln -s "../$f" "train_set_subset/$(basename $f)"
    done
    echo "✓ Created subset with 100 players"
    echo ""
fi

NUM_FILES=$(ls train_set_subset/*.sgf 2>/dev/null | wc -l)
echo "Training on $NUM_FILES players"
echo ""

PRESET="${1:-medium}"

case "$PRESET" in
    test)
        python train_with_validation.py \
            --train_dir train_set_subset \
            --output_dir models/test_subset \
            --epochs 5 \
            --batch_size 8 \
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
    
    medium)
        python train_with_validation.py \
            --train_dir train_set_subset \
            --output_dir models/medium_subset \
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
            --num_val_players 30 \
            --num_workers 0
        ;;
    
    large)
        python train_with_validation.py \
            --train_dir train_set_subset \
            --output_dir models/large_subset \
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
            --num_val_players 30 \
            --num_workers 0
        ;;
    
    *)
        echo "Unknown preset: $PRESET"
        echo "Usage: $0 [test|medium|large]"
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "Training completed!"
echo "======================================================================="
