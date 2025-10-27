# Go Style Detection - Complete Pipeline Explanation

## 🎯 CRITICAL ISSUE: Why You Got Score 0

Your score is likely **0** because of one or more of these issues:

### Issue 1: **Training on QUERY set, Testing on QUERY vs CANDIDATE**
- **WRONG**: You trained on `test_set/query_set` 
- **CORRECT**: You should train on `train_set/` (your training data with 600 different players)
- **Problem**: The model learned embeddings for query players, but query players and candidate players are DIFFERENT people!

### Issue 2: **No Validation During Training**
- You can't tell if the model is learning meaningful style representations
- Loss going down doesn't mean it can match similar players

### Issue 3: **Random Embeddings**
- If trained only on query set, the model outputs random embeddings for candidate set (unseen players)
- Cosine similarity between random vectors is meaningless

---

## 📋 Complete Pipeline Explanation

### **Phase 1: Data Preparation**

```
train_set/           # 600 SGF files for training (player001-player600.sgf)
  ├── 1.sgf ... 600.sgf    # Different games/players for training

test_set/
  ├── query_set/     # 600 SGF files (player001-player600.sgf) - Players to identify
  └── cand_set/      # 600 SGF files (player001-player600.sgf) - Candidate matches
```

**Key Point**: Query and candidate players are THE SAME 600 people (ground truth matching)!
But your TRAINING should use train_set/, NOT test_set/!

---

### **Phase 2: Model Architecture**

```python
# Input: Game sequence (N moves × 18 channels × 19 × 19)
# Output: Style embedding (256-dim vector)

1. PER-MOVE ENCODER (Residual CNN + MLP)
   Input: (batch, seq_len, 18, 19, 19)
   ├── Conv2d(18 → 128 channels)
   ├── Residual Blocks (×3)
   ├── Global Average Pool → (batch×seq_len, 128)
   └── MLP → (batch, seq_len, 128)  # Per-move embeddings

2. GAME-LEVEL AGGREGATOR (Transformer + MLP)
   Input: (batch, seq_len, 128)  # Sequence of move embeddings
   ├── Positional Encoding
   ├── Transformer Encoder (×3 layers, 8 heads)
   ├── Mean pooling over sequence → (batch, 128)
   ├── MLP → (batch, 256)
   ├── Tanh activation
   └── L2 Normalize → (batch, 256)  # Final style embedding
```

**Why this architecture?**
- **CNN**: Captures spatial patterns on the board (local tactics)
- **Transformer**: Captures temporal patterns (move sequences, strategy evolution)
- **Normalization**: Makes cosine similarity meaningful (all embeddings on unit sphere)

---

### **Phase 3: Training with Triplet Loss**

```python
# For each batch:
1. Sample triplet (Anchor, Positive, Negative)
   - Anchor: Game from Player A
   - Positive: Another game from Player A (same style)
   - Negative: Game from Player B (different style)

2. Compute embeddings
   embed_a = model(anchor_game)    # (batch, 256)
   embed_p = model(positive_game)  # (batch, 256)
   embed_n = model(negative_game)  # (batch, 256)

3. Compute triplet loss
   dist_pos = ||embed_a - embed_p||²  # Should be SMALL
   dist_neg = ||embed_a - embed_n||²  # Should be LARGE
   
   loss = max(0, dist_pos - dist_neg + margin)
   
   # Goal: dist_pos + margin < dist_neg
   # i.e., same player's games are closer than different players
```

**What the model learns:**
- Games from the same player cluster together in embedding space
- Games from different players are pushed apart
- Style is captured as a point in 256-dimensional space

---

### **Phase 4: Inference**

```python
# For each player (query or candidate):
1. Load 10 random games
2. Extract embedding for each game → (10, 256)
3. Average embeddings → (1, 256)  # Player's style embedding

# Matching:
For each query player i:
    For each candidate player j:
        similarity[i,j] = cosine_similarity(query_embed[i], cand_embed[j])
    
    best_match[i] = argmax(similarity[i, :])  # Candidate with highest similarity
```

**Why average 10 games?**
- Reduces noise from single game
- More stable style representation
- Player's true style = consistent patterns across games

---

## 🔧 What's Wrong With Your Current Setup

### Current Training:
```bash
--train_dir test_set/query_set  # ❌ WRONG!
```

**Problem Chain:**
1. Model learns embeddings for 600 query players
2. During inference:
   - Query players → Model recognizes them (good embeddings)
   - Candidate players → Model has NEVER seen them (random embeddings)
3. Matching random query embeddings vs random candidate embeddings = **random matches**
4. Score = 0

### Correct Training:
```bash
--train_dir train_set/  # ✅ CORRECT!
```

**What should happen:**
1. Model learns to extract **general style features** from 600 training players
2. During inference:
   - Query players → Model extracts style (never seen, but uses learned features)
   - Candidate players → Model extracts style (never seen, but uses learned features)
3. Matching meaningful embeddings → **correct matches**
4. Score > 0

---

## 🎯 Why Validation Matters

Without validation, you can't tell:
- ✗ Is the model learning meaningful styles?
- ✗ Can it generalize to new players?
- ✗ Is it just memorizing training player IDs?

**Validation should test:**
1. Can it match the same player across different games?
2. Does it give higher similarity to same player than different players?
3. Top-1/Top-5 accuracy on held-out players

---

## 📊 Expected Behavior

### During Training (with validation):
```
Epoch 1: Loss=0.87, Val Acc@1=15%, Val Acc@5=35%
Epoch 10: Loss=0.45, Val Acc@1=45%, Val Acc@5=78%
Epoch 30: Loss=0.15, Val Acc@1=72%, Val Acc@5=92%
Epoch 50: Loss=0.06, Val Acc@1=85%, Val Acc@5=97%
```

### During Inference:
```
Extracting query embeddings: 600 players × 10 games
Extracting candidate embeddings: 600 players × 10 games
Computing similarities: 600 × 600 matrix
Matching: Each query → Best candidate
```

**If working correctly:**
- Many queries should match to themselves (same player in both sets)
- Similar style players should match even if not exact
- Score should be > 0 (at least some correct matches)

---

## 🔍 Debugging Checklist

- [ ] Training on `train_set/` NOT `test_set/query_set`
- [ ] Validation shows increasing accuracy
- [ ] Model outputs normalized embeddings (||embed|| ≈ 1)
- [ ] Same player's games have high similarity (> 0.8)
- [ ] Different players' games have lower similarity (< 0.5)
- [ ] Submission.csv has 600 rows (excluding header)
- [ ] All IDs in submission are 1-600 (1-indexed)

---

## 🚀 Next Steps

1. **Fix training data**: Use `train_set/` instead of `test_set/query_set`
2. **Add validation**: Split train_set into train/val (e.g., 500/100)
3. **Monitor metrics**: Track Top-1 and Top-5 accuracy during training
4. **Verify embeddings**: Check that same player's games cluster together
5. **Re-run inference**: With properly trained model

