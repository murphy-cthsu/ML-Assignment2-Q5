#!/bin/bash
# Full-scale training script for Go Style Detection
# Usage: ./train_full.sh [small|medium|large|custom]

set -e  # Exit on error

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_hw2

# Change to script directory
cd "$(dirname "$0")"

# Default configuration
CONFIG=${1:-medium}
TRAIN_DIR="test_set/query_set"
OUTPUT_DIR="models/full_training_${CONFIG}"
EPOCHS=50
BATCH_SIZE=16
NUM_SAMPLES=1000
MAX_MOVES=100
NUM_WORKERS=4
SAVE_INTERVAL=5

echo "======================================"
echo "Go Style Detection - Full Training"
echo "Configuration: $CONFIG"
echo "======================================"

case $CONFIG in
    small)
        echo "Using SMALL configuration (fast training, lower accuracy)"
        HIDDEN_CHANNELS=64
        NUM_BLOCKS=2
        MOVE_EMBED_DIM=64
        NUM_HEADS=4
        NUM_TRANSFORMER_LAYERS=2
        STYLE_EMBED_DIM=128
        BATCH_SIZE=32
        MAX_MOVES=50
        ;;
    
    medium)
        echo "Using MEDIUM configuration (balanced)"
        HIDDEN_CHANNELS=128
        NUM_BLOCKS=3
        MOVE_EMBED_DIM=128
        NUM_HEADS=8
        NUM_TRANSFORMER_LAYERS=3
        STYLE_EMBED_DIM=256
        BATCH_SIZE=16
        MAX_MOVES=75
        ;;
    
    large)
        echo "Using LARGE configuration (paper specification - highest accuracy)"
        HIDDEN_CHANNELS=256
        NUM_BLOCKS=5
        MOVE_EMBED_DIM=256
        NUM_HEADS=8
        NUM_TRANSFORMER_LAYERS=4
        STYLE_EMBED_DIM=512
        BATCH_SIZE=8
        MAX_MOVES=100
        EPOCHS=100
        ;;
    
    test)
        echo "Using TEST configuration (quick test)"
        HIDDEN_CHANNELS=64
        NUM_BLOCKS=2
        MOVE_EMBED_DIM=64
        NUM_HEADS=4
        NUM_TRANSFORMER_LAYERS=2
        STYLE_EMBED_DIM=128
        EPOCHS=5
        BATCH_SIZE=4
        NUM_SAMPLES=50
        MAX_MOVES=30
        OUTPUT_DIR="models/test_run"
        ;;
    
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Available: small, medium, large, test"
        exit 1
        ;;
esac

# Print configuration
echo ""
echo "Training Parameters:"
echo "  Train directory: $TRAIN_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Samples per epoch: $NUM_SAMPLES"
echo "  Max moves: $MAX_MOVES"
echo ""
echo "Model Architecture:"
echo "  Hidden channels: $HIDDEN_CHANNELS"
echo "  Residual blocks: $NUM_BLOCKS"
echo "  Move embedding dim: $MOVE_EMBED_DIM"
echo "  Transformer heads: $NUM_HEADS"
echo "  Transformer layers: $NUM_TRANSFORMER_LAYERS"
echo "  Style embedding dim: $STYLE_EMBED_DIM"
echo ""
echo "System:"
echo "  Workers: $NUM_WORKERS"
echo "  Save interval: $SAVE_INTERVAL epochs"
echo "======================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
echo ""

python train_style_model.py \
    --train_dir "$TRAIN_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --hidden_channels $HIDDEN_CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --move_embed_dim $MOVE_EMBED_DIM \
    --num_heads $NUM_HEADS \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --style_embed_dim $STYLE_EMBED_DIM \
    --max_moves $MAX_MOVES \
    --output_dir "$OUTPUT_DIR" \
    --num_workers $NUM_WORKERS \
    --save_interval $SAVE_INTERVAL

echo ""
echo "======================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/best_model.pth"
echo "======================================"
