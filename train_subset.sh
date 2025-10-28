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
        # Quick test run using GE2E training script
        python train_with_validation_new.py \
            --train_dir train_set_subset \
            --config_file conf.cfg \
            --output_dir models/test_subset \
            --epochs 3 \
            --steps_per_epoch 50 \
            --N 8 \
            --M 4 \
            --d_model 256 \
            --num_heads 4 \
            --num_transformer_layers 2 \
            --style_embed_dim 128 \
            --max_moves 32 \
            --num_samples 500 \
            --lr 0.001 \
            --eval_ref 2 \
            --eval_qry 2 \
            --eval_max_players 20
        ;;

    medium)
        # Medium preset aligned to GE2E script defaults
        python train_with_validation_new.py \
            --train_dir train_set_subset \
            --config_file conf.cfg \
            --output_dir models/medium_subset \
            --epochs 20 \
            --steps_per_epoch 200 \
            --N 40 \
            --M 10 \
            --d_model 1024 \
            --num_heads 8 \
            --num_transformer_layers 4 \
            --style_embed_dim 256 \
            --max_moves 32 \
            --num_samples 10000 \
            --lr 0.001 \
            --eval_ref 5 \
            --eval_qry 5 \
            --eval_max_players 50
        ;;

    large)
        # Larger run (more players per batch / more steps)
        python train_with_validation_new.py \
            --train_dir train_set_subset \
            --config_file conf.cfg \
            --output_dir models/large_subset \
            --epochs 50 \
            --steps_per_epoch 400 \
            --N 64 \
            --M 10 \
            --d_model 1024 \
            --num_heads 12 \
            --num_transformer_layers 6 \
            --style_embed_dim 512 \
            --max_moves 64 \
            --num_samples 20000 \
            --lr 0.0005 \
            --eval_ref 5 \
            --eval_qry 5 \
            --eval_max_players 60
        ;;

    paper)
        # Paper-aligned GE2E training with SGD + step LR (memory optimized)
        python train_with_validation_new.py \
            --train_dir train_set_subset \
            --config_file conf.cfg \
            --output_dir models/paper_subset \
            --epochs 30 \
            --steps_per_epoch 200 \
            --N 16 \
            --M 8 \
            --d_model 512 \
            --num_heads 8 \
            --num_transformer_layers 6 \
            --style_embed_dim 256 \
            --max_moves 32 \
            --num_samples 10000 \
            --use_sgd \
            --eval_ref 5 \
            --eval_qry 5 \
            --eval_max_players 50
        ;;

    paper_lite)
        # Lighter paper-aligned preset (smallest memory footprint)
        python train_with_validation_new.py \
            --train_dir train_set_subset \
            --config_file conf.cfg \
            --output_dir models/paper_lite_subset \
            --epochs 30 \
            --steps_per_epoch 200 \
            --N 12 \
            --M 6 \
            --d_model 256 \
            --num_heads 4 \
            --num_transformer_layers 4 \
            --style_embed_dim 256 \
            --max_moves 32 \
            --num_samples 10000 \
            --use_sgd \
            --eval_ref 5 \
            --eval_qry 5 \
            --eval_max_players 50
        ;;

    *)
        echo "Unknown preset: $PRESET"
        echo "Usage: $0 [test|medium|large|paper|paper_lite]"
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "Training completed!"
echo "======================================================================="
