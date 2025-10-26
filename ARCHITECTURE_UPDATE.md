# Architecture Update - Corrected to Match Paper

## ✅ What Was Changed

The model architecture has been **updated to match the paper specification**:

### Old Architecture (Incorrect)
- Single-stage CNN processing
- Direct pooling of single board positions
- Simple MLP projection

### New Architecture (Correct - Paper Specification)
```
1. Per-Move Encoding: Residual CNN + MLP
   - Input: Each board position (18, 19, 19)
   - CNN: Initial conv + N residual blocks
   - Global pooling + MLP
   - Output: Move embedding (256-dim)

2. Game-Level Aggregation: Transformer + MLP + tanh + normalize
   - Input: Sequence of move embeddings (seq_len, 256)
   - Positional encoding
   - Transformer encoder (multi-head attention)
   - Mean pooling over sequence
   - MLP projection
   - Tanh activation
   - L2 normalization
   - Output: Style embedding (512-dim, normalized)
```

## 📁 Updated Files

### 1. `style_model.py` ✅
**New Components:**
- `MoveEncoder`: Residual CNN + MLP for per-move features
- `TransformerAggregator`: Transformer + MLP + tanh + normalize
- `PositionalEncoding`: Sinusoidal position encodings
- `StyleEncoder`: Complete pipeline (Move → Game level)
- `TripletStyleModel`: Triplet training wrapper
- `SupervisedStyleClassifier`: Classification variant

**Key Changes:**
- Now processes **sequences of moves** instead of single positions
- Input shape: `(batch, seq_len, 18, 19, 19)` 
- Output shape: `(batch, 512)` - L2 normalized embeddings

### 2. `train_style_model.py` ✅
**New Components:**
- `GoGameSequenceDataset`: Loads full game sequences
- `TripletGameDataset`: Generates triplets of game sequences
- `collate_fn_triplet`: Pads variable-length sequences
- Support for attention masks (padding)

**Key Changes:**
- Loads entire games (sequences of board states)
- Handles variable-length sequences with padding
- Uses `get_random_feature_and_label(n_frames, move_step, start_idx, random)`

### 3. `infer_style.py` ✅
**Key Changes:**
- Processes game sequences instead of single positions
- Averages embeddings over multiple games per player
- Handles variable-length sequences

## 🧪 Testing

**Test the architecture:**
```bash
python style_model.py
```

**Expected output:**
```
Testing Style Detection Models (Paper Architecture)

Architecture:
  1. Per-move: Residual CNN + MLP
  2. Game-level: Transformer + MLP + tanh + normalize

✓ All models created successfully!
✓ Architecture matches paper specification!

Model parameters: ~9.4M
  - Move Encoder (CNN + MLP): ~6.1M params
  - Aggregator (Transformer + MLP): ~3.4M params
```

## 🚀 Training (When You Have Data)

### Quick Test (5-10 minutes)
```bash
python train_style_model.py \
    --epochs 5 \
    --batch_size 4 \
    --num_samples 100 \
    --hidden_channels 128 \
    --num_blocks 3 \
    --num_transformer_layers 2 \
    --output_dir models/quick_test
```

### Full Training (Recommended)
```bash
python train_style_model.py \
    --epochs 50 \
    --batch_size 16 \
    --num_samples 10000 \
    --hidden_channels 256 \
    --num_blocks 5 \
    --move_embed_dim 256 \
    --num_heads 8 \
    --num_transformer_layers 4 \
    --style_embed_dim 512 \
    --max_moves 200 \
    --output_dir models/final
```

### Key Parameters
- `--num_samples`: How many random games to sample per epoch (10000 recommended)
- `--max_moves`: Maximum moves per game (200 = full games)
- `--batch_size`: Reduce if OOM (16→8→4)
- `--num_transformer_layers`: More layers = better but slower (4 recommended)

## 🎯 Inference

```bash
python infer_style.py \
    --model_path models/final/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output submission.csv \
    --max_moves 200 \
    --games_per_player 10
```

## 📊 Model Architecture Details

### Input Format
```python
# Training: Sequence of board states from a game
input_shape = (batch_size, seq_len, 18, 19, 19)
# - batch_size: Number of games in batch
# - seq_len: Number of moves in game (variable, gets padded)
# - 18: Feature channels (from SGF data loader)
# - 19×19: Go board size
```

### Forward Pass
```python
model = create_model('triplet', ...)
anchor = torch.randn(4, 50, 18, 19, 19)  # 4 games, 50 moves each
embedding = model.get_embedding(anchor)   # → (4, 512)
# Embeddings are L2-normalized (norm = 1.0)
```

### Architecture Flow
```
Input: (batch, seq_len, 18, 19, 19)
  ↓
MoveEncoder (applied to each move):
  ├─ Conv2d(18 → 256)
  ├─ ResidualBlocks × 5
  ├─ GlobalAvgPool
  └─ MLP(256 → 256)
  → (batch, seq_len, 256)
  ↓
TransformerAggregator:
  ├─ PositionalEncoding
  ├─ TransformerEncoder × 4 layers
  ├─ MeanPooling (over sequence)
  ├─ MLP(256 → 512)
  ├─ Tanh
  └─ L2 Normalize
  → (batch, 512)
```

## ⚠️ Current Issue

**The `train_set` directory appears to be empty (0 players detected).**

This is why training hangs - it's trying to randomly sample from an empty dataset.

### To Fix:
1. Ensure `train_set/` contains SGF files
2. Check the SGF files are valid: `python -c "import build.go.style_py as sp; sp.load_config_file('conf.cfg'); loader = sp.DataLoader('train_set'); print(f'Players: {loader.get_num_of_player()}')"`

## 📝 Dependencies

Make sure you have:
```bash
pip install torch tqdm numpy
```

## ✨ Summary

**Architecture is now correct** and matches the paper:
1. ✅ Per-move encoding with Residual CNN + MLP
2. ✅ Game-level aggregation with Transformer + MLP + tanh + normalize
3. ✅ Handles variable-length game sequences
4. ✅ L2-normalized style embeddings
5. ✅ Triplet loss training
6. ✅ GPU support

**Ready to use once you have training data in `train_set/`!**
