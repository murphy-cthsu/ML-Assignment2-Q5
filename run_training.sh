#!/bin/bash

# Simple training launcher - run this in tmux
# Usage: ./run_training.sh [test|small|medium|large]

cd /home/cthsu/Workspace/ML_2025/HW2/Q5

PRESET="${1:-medium}"

# Activate conda and run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_hw2

echo "======================================================================="
echo "Go Style Detection - Training with Validation"
echo "======================================================================="
echo ""
echo "Training on: train_set/ ($(ls train_set/*.sgf 2>/dev/null | wc -l) SGF files)"
echo "Preset: $PRESET"
echo "======================================================================="
echo ""

case "$PRESET" in
    test)
        python train_lazy.py \
            --train_dir train_set \
            --output_dir models/test_lazy \
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
            --num_val_players 15 \
            --batch_load_size 30
        ;;
    
    small)
        python train_lazy.py \
            --train_dir train_set \
            --output_dir models/small_lazy \
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
            --num_val_players 30 \
            --batch_load_size 50
        ;;
    
    medium)
        python train_gradual.py \
            --train_dir train_set \
            --output_dir models/medium_gradual \
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
            --load_batch_size 20 \
            --load_delay 1.0
        ;;
    
    large)
        python train_lazy.py \
            --train_dir train_set \
            --output_dir models/large_lazy \
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
            --batch_load_size 50
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
echo "======================================================================="
