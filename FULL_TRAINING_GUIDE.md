# Full-Scale Training Guide

## Quick Start

### Option 1: Using Bash Script (Simplest)

```bash
# Activate environment
conda activate ml_hw2
cd /home/cthsu/Workspace/ML_2025/HW2/Q5

# Run with default medium configuration
./train_full.sh medium

# Or choose different presets:
./train_full.sh test     # Quick test (5 epochs)
./train_full.sh small    # Fast training (50 epochs)
./train_full.sh medium   # Balanced (50 epochs) - RECOMMENDED
./train_full.sh large    # Best accuracy (100 epochs, slow)
```

### Option 2: Using Python Script (More Control)

```bash
# Basic usage with preset
python train_full_scale.py --preset medium

# With custom settings
python train_full_scale.py \
    --preset large \
    --epochs 200 \
    --batch_size 16 \
    --output_dir models/my_training

# Resume from checkpoint
python train_full_scale.py \
    --preset medium \
    --resume models/full_training_medium/checkpoint_epoch_25.pth
```

### Option 3: Direct Python Call (Maximum Control)

```bash
python train_style_model.py \
    --train_dir test_set/query_set \
    --epochs 100 \
    --batch_size 16 \
    --num_samples 2000 \
    --hidden_channels 256 \
    --num_blocks 5 \
    --move_embed_dim 256 \
    --num_heads 8 \
    --num_transformer_layers 4 \
    --style_embed_dim 512 \
    --max_moves 100 \
    --output_dir models/custom_training \
    --num_workers 4 \
    --save_interval 10
```

## Configuration Presets

### Test Configuration
**Use for**: Quick testing, debugging
- **Time**: ~5 minutes
- **Accuracy**: Low
- Parameters: 279K
- Epochs: 5
- Batch size: 4
- Max moves: 30

### Small Configuration
**Use for**: Fast iteration, initial experiments
- **Time**: ~2-3 hours
- **Accuracy**: Good
- Parameters: ~500K
- Epochs: 50
- Batch size: 32
- Max moves: 50
- Hidden channels: 64
- Transformer layers: 2

### Medium Configuration (RECOMMENDED)
**Use for**: Production training, balanced speed/accuracy
- **Time**: ~8-12 hours
- **Accuracy**: Very good
- Parameters: ~2M
- Epochs: 50
- Batch size: 16
- Max moves: 75
- Hidden channels: 128
- Transformer layers: 3

### Large Configuration
**Use for**: Maximum accuracy, paper replication
- **Time**: ~24-48 hours
- **Accuracy**: Best
- Parameters: ~9M
- Epochs: 100
- Batch size: 8
- Max moves: 100
- Hidden channels: 256
- Transformer layers: 4
- Style embedding: 512-dim

## Training Data

### Current Setup
- **Training set**: `test_set/query_set` (600 players)
- Each player has multiple games
- Features: 18 channels × 19×19 board

### Using Different Data
```bash
# Use candidate set for training
./train_full.sh medium --train_dir test_set/cand_set

# Use actual training set (if available)
./train_full.sh medium --train_dir train_set
```

## Monitoring Training

### Real-time Monitoring
Training will display:
```
Epoch 15/50: 100%|████████| 125/125 [00:45<00:00, 2.75it/s, loss=0.523]

Epoch 15/50 - Average Loss: 0.5234
✓ Saved best model (loss: 0.5234)
```

### Checkpoints
Saved in `models/full_training_<preset>/`:
- `best_model.pth` - Best model so far (lowest loss)
- `checkpoint_epoch_N.pth` - Checkpoints every N epochs
- `training_config.json` - Configuration used
- `full_training_config.json` - Extended configuration

### Loss Values
- **Initial**: ~1.0-1.5 (random initialization)
- **Good training**: ~0.3-0.6 after 50 epochs
- **Well-trained**: ~0.1-0.3 after 100 epochs
- Loss should **decrease** over time

## GPU Memory Requirements

| Configuration | Batch Size | GPU Memory | Recommended GPU |
|--------------|------------|------------|-----------------|
| Test         | 4          | ~2 GB      | Any modern GPU  |
| Small        | 32         | ~6 GB      | GTX 1060+       |
| Medium       | 16         | ~10 GB     | RTX 3060+       |
| Large        | 8          | ~14 GB     | RTX 3090, 4090  |

**If OOM (Out of Memory)**:
```bash
# Reduce batch size
python train_full_scale.py --preset large --batch_size 4

# Or reduce max_moves
python train_full_scale.py --preset large --max_moves 50
```

## Resuming Training

If training is interrupted:

```bash
# Find latest checkpoint
ls -lh models/full_training_medium/checkpoint_*.pth

# Resume from checkpoint
python train_full_scale.py \
    --preset medium \
    --resume models/full_training_medium/checkpoint_epoch_30.pth
```

## After Training

### 1. Check Training Results
```bash
# View final model
ls -lh models/full_training_medium/

# Check configuration
cat models/full_training_medium/training_config.json
```

### 2. Run Inference
```bash
# Use the trained model for style matching
python infer_style.py \
    --model_path models/full_training_medium/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output_file predictions.csv \
    --max_moves 75
```

## Troubleshooting

### Training is slow
- Reduce `--max_moves` (e.g., 50 instead of 100)
- Increase `--num_workers` (e.g., 8 instead of 4)
- Use smaller preset (medium instead of large)

### Loss not decreasing
- Check if data is loading correctly (should see 600 players)
- Try lower learning rate (modify train_style_model.py)
- Increase `--num_samples` (more triplets per epoch)
- Train for more epochs

### Out of memory
- Reduce `--batch_size` (e.g., 8 instead of 16)
- Reduce `--max_moves` (e.g., 50 instead of 100)
- Use smaller model preset

### Training crashes
- Check GPU memory: `nvidia-smi`
- Verify conda environment: `conda activate ml_hw2`
- Check data directory exists and has .sgf files
- Review error messages in terminal

## Example Training Runs

### Quick Test (verify everything works)
```bash
./train_full.sh test
# Expected time: 5 minutes
# Expected final loss: ~0.6-0.8
```

### Development Training
```bash
python train_full_scale.py \
    --preset medium \
    --epochs 30 \
    --output_dir models/dev_run
# Expected time: 4-6 hours
# Expected final loss: ~0.4-0.6
```

### Production Training
```bash
python train_full_scale.py \
    --preset large \
    --epochs 100 \
    --num_samples 2000 \
    --output_dir models/production
# Expected time: 36-48 hours
# Expected final loss: ~0.2-0.4
```

### Overnight Training
```bash
nohup ./train_full.sh medium > training.log 2>&1 &
# Check progress: tail -f training.log
# Job will continue even if you logout
```

## Best Practices

1. **Start small**: Run `test` preset first to verify setup
2. **Monitor GPU**: Use `watch -n 1 nvidia-smi` in another terminal
3. **Save logs**: Redirect output to file for later analysis
4. **Validate checkpoints**: Test inference after training completes
5. **Use screen/tmux**: For long training sessions on remote servers

## Next Steps After Training

1. **Evaluate**: Test on validation set
2. **Inference**: Generate style predictions
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Ensemble**: Train multiple models and combine predictions
