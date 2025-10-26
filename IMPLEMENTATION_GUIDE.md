# Complete Implementation Guide: Go Playing Style Detection

## 📋 Table of Contents

1. [Overview](#overview)
2. [What Was Implemented](#what-was-implemented)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Understanding the Approach](#understanding-the-approach)
6. [Troubleshooting](#troubleshooting)

---

## Overview

I've implemented a **complete deep learning system** for Go playing style detection and matching. The system uses convolutional neural networks (CNNs) to learn distinctive features of each player's style from their game records.

### Key Components

1. **`style_model.py`**: Neural network architectures
   - ResNet-based style encoder
   - Triplet loss model for metric learning
   - Supervised classification model
   
2. **`train_style_model.py`**: Training pipeline
   - Loads SGF files using the built C++ module
   - Trains the model to recognize playing styles
   - Supports multiple training strategies
   
3. **`infer_style.py`**: Inference pipeline
   - Extracts style embeddings from player games
   - Matches query players to similar candidates
   - Generates submission file

---

## What Was Implemented

### 🎯 Neural Network Architecture

**Style Encoder (ResNet-based CNN)**
```
Input: Go board position (18 channels × 19×19)
  ↓
Initial Conv Layer (18 → 256 channels)
  ↓
Residual Blocks × N (default: 10)
  ├─ Conv 3×3 + BatchNorm + ReLU
  ├─ Conv 3×3 + BatchNorm
  └─ Skip Connection + ReLU
  ↓
Global Average Pooling (19×19 → 1×1)
  ↓
Fully Connected (256 → 512)
  ↓
L2 Normalization
  ↓
Output: Style embedding (512-dim vector)
```

### 🎓 Training Strategies

#### 1. **Triplet Loss** (Recommended)
- **Concept**: Learn by comparing
- **How it works**: 
  - Take 3 positions: Anchor (Player A), Positive (also Player A), Negative (Player B)
  - Pull anchor closer to positive
  - Push anchor away from negative
- **Best for**: Style matching (our task)

#### 2. **Supervised Classification**
- **Concept**: Direct player identification
- **How it works**: Predict which player played given position
- **Best for**: When you have labeled data and want direct classification

### 📊 Data Flow

```
SGF Files
  ↓ (C++ DataLoader)
Board Positions (18×19×19 tensors)
  ↓ (Style Encoder)
Style Embeddings (512-dim vectors)
  ↓ (Similarity Computation)
Player Matches (Top-5 rankings)
```

---

## Quick Start

### Step 1: Verify Setup

```bash
# Make sure you're in the right directory
cd /home/cthsu/Workspace/ML_2025/HW2/Q5

# Run system test
python test_system.py
```

You should see:
```
✓ All critical tests passed!
```

### Step 2: Train a Model (Quick Test)

```bash
# Quick training run (5 epochs, ~5-10 minutes on GPU)
python train_style_model.py \
    --model_type triplet \
    --epochs 5 \
    --batch_size 16 \
    --hidden_channels 128 \
    --num_blocks 5 \
    --embedding_dim 256 \
    --output_dir models/quick_test
```

### Step 3: Run Inference

```bash
# Generate predictions
python infer_style.py \
    --model_path models/quick_test/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output submission.csv
```

### Step 4: Check Results

```bash
# View first few lines of submission
head -10 submission.csv
```

Expected format:
```csv
query_id,rank1,rank2,rank3,rank4,rank5
1,145,298,67,412,89
2,234,567,12,345,678
...
```

---

## Detailed Usage

### Training Options Explained

#### Basic Options
```bash
python train_style_model.py \
    --model_type triplet \          # or 'supervised'
    --train_dir train_set \          # SGF training data
    --epochs 30 \                    # number of training epochs
    --batch_size 32 \                # samples per batch
    --learning_rate 0.001            # initial learning rate
```

#### Model Architecture
```bash
    --hidden_channels 256 \          # CNN channel width
    --num_blocks 10 \                # number of residual blocks
    --embedding_dim 512              # output embedding size
```

More blocks & channels = more powerful but slower

#### Data Sampling
```bash
    --n_frames 10 \                  # positions sampled per game
    --move_step 4                    # step between samples
```

#### Training Behavior
```bash
    --margin 1.0 \                   # triplet loss margin
    --weight_decay 0.0001 \          # L2 regularization
    --num_workers 4 \                # parallel data loading
    --save_interval 10               # checkpoint frequency
```

### Recommended Settings

#### For Fast Experimentation
```bash
python train_style_model.py \
    --model_type triplet \
    --epochs 10 \
    --batch_size 16 \
    --hidden_channels 128 \
    --num_blocks 5 \
    --embedding_dim 256 \
    --output_dir models/fast_experiment
```
- Training time: ~10-15 minutes
- Good for: Testing ideas quickly

#### For Best Performance
```bash
python train_style_model.py \
    --model_type triplet \
    --epochs 50 \
    --batch_size 32 \
    --hidden_channels 256 \
    --num_blocks 10 \
    --embedding_dim 512 \
    --learning_rate 0.001 \
    --margin 1.0 \
    --output_dir models/best_model
```
- Training time: ~1-2 hours
- Good for: Final submission

### Inference Options

```bash
python infer_style.py \
    --model_path models/best_model/best_model.pth \  # trained model
    --config conf.cfg \                               # configuration
    --query_dir test_set/query_set \                  # query players
    --candidate_dir test_set/cand_set \               # candidates
    --n_frames 10 \                                   # positions per game
    --top_k 5 \                                       # number of matches
    --output submission.csv \                         # output file
    --device cuda                                     # use GPU
```

---

## Understanding the Approach

### Why This Architecture Works

1. **Convolutional Layers**: Capture local patterns (joseki, shapes)
2. **Residual Connections**: Enable deep networks without degradation
3. **Global Pooling**: Position-invariant features
4. **L2 Normalization**: Fair similarity comparison

### What the Model Learns

The model learns to recognize:
- **Opening preferences**: Fuseki choices
- **Fighting style**: Aggressive vs. territorial
- **Reading patterns**: How players calculate
- **Endgame technique**: Efficiency in final moves

### How Triplet Loss Works

```python
# For each training step:
anchor = position_from_player_A
positive = different_position_from_player_A
negative = position_from_player_B

# Compute embeddings
emb_a = model(anchor)
emb_p = model(positive)
emb_n = model(negative)

# Loss encourages:
# distance(emb_a, emb_p) < distance(emb_a, emb_n) + margin
```

This teaches the model:
- Same player's positions should be close in embedding space
- Different players' positions should be far apart

### Similarity Computation

```python
# Extract embeddings
query_emb = model(query_positions).mean(dim=0)  # average over positions
cand_emb = model(cand_positions).mean(dim=0)

# Compute cosine similarity
similarity = dot(query_emb, cand_emb)  # higher = more similar
```

---

## Troubleshooting

### Issue: Out of Memory

**Symptoms**: `CUDA out of memory` error

**Solutions**:
```bash
# Reduce batch size
--batch_size 8

# Reduce model size
--hidden_channels 128
--num_blocks 5

# Use CPU
--device cpu
```

### Issue: Training Too Slow

**Solutions**:
```bash
# Reduce data loading bottleneck
--num_workers 8

# Sample fewer positions
--n_frames 5

# Use smaller model for testing
--num_blocks 3
```

### Issue: Poor Matching Performance

**Possible Causes & Solutions**:

1. **Not enough training**
   - Increase `--epochs 50` or more
   
2. **Model too small**
   - Increase `--num_blocks 10`
   - Increase `--hidden_channels 256`
   
3. **Bad data sampling**
   - Increase `--n_frames` to sample more positions
   - Ensure SGF files are valid

4. **Wrong model type**
   - Use `triplet` for style matching
   - `supervised` is for classification

### Issue: DataLoader Errors

**Check**:
1. SGF files are in correct format
2. Directory structure is correct:
   ```
   train_set/
     ├── 1.sgf
     ├── 2.sgf
     └── ...
   test_set/
     ├── query_set/
     │   ├── player001.sgf
     │   └── ...
     └── cand_set/
         ├── player001.sgf
         └── ...
   ```

3. Configuration file `conf.cfg` exists

---

## Advanced Usage

### Custom Training Loss

Edit `train_style_model.py` to add custom loss:

```python
# Add to training loop
def custom_loss(anchor, positive, negative):
    # Your loss implementation
    ...
    return loss
```

### Ensemble Models

Train multiple models and average predictions:

```python
# Train different models
python train_style_model.py --output_dir models/model1 --num_blocks 5
python train_style_model.py --output_dir models/model2 --num_blocks 10
python train_style_model.py --output_dir models/model3 --hidden_channels 128
```

Then modify `infer_style.py` to load and average all three.

### Hyperparameter Tuning

Key parameters to tune:
1. `--learning_rate`: 0.0001 to 0.01
2. `--margin`: 0.5 to 2.0  
3. `--embedding_dim`: 256 to 1024
4. `--num_blocks`: 5 to 20

---

## Monitoring Training

### Watch Training Progress

```bash
# Training will show:
Epoch 1/30 [Train]: 100%|██████| 150/150 [00:45<00:00,  3.31it/s, loss=0.523]
Epoch 1/30 - Average Loss: 0.5234
✓ Saved best model (loss: 0.5234)
```

### Good Signs
- Loss decreasing over epochs
- Validation accuracy increasing (supervised)
- Model saves checkpoint

### Bad Signs
- Loss not decreasing after 10+ epochs
- Loss = NaN or inf
- No checkpoint saved

---

## Files Reference

### Created Files

| File | Purpose |
|------|---------|
| `style_model.py` | Neural network models |
| `train_style_model.py` | Training script |
| `infer_style.py` | Inference script |
| `test_system.py` | System verification |
| `README_STYLE_DETECTION.md` | Detailed documentation |
| `IMPLEMENTATION_GUIDE.md` | This file |
| `quickstart.sh` | Quick start script |

### Generated Files

| File | Created By | Purpose |
|------|-----------|---------|
| `models/*/best_model.pth` | Training | Best model checkpoint |
| `models/*/checkpoint_epoch_*.pth` | Training | Periodic checkpoints |
| `models/*/training_args.json` | Training | Training configuration |
| `submission.csv` | Inference | Final predictions |

---

## Next Steps

1. **Run Quick Test**: Verify everything works with 5 epochs
2. **Full Training**: Train for 30-50 epochs for best results
3. **Experiment**: Try different hyperparameters
4. **Submit**: Generate `submission.csv` and submit

---

## Support

If you encounter issues:
1. Check this guide
2. Read `README_STYLE_DETECTION.md`
3. Run `python test_system.py`
4. Check error messages carefully

---

**Good luck with your style detection project!** 🎯🎮

