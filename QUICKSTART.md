# Quick Start Guide - Go Style Detection

## 🚀 Get Started in 3 Steps

### Step 1: Verify Setup (30 seconds)
```bash
python test_system.py
```
✓ Should see: "All critical tests passed!"

### Step 2: Train Model (Choose one)

#### Option A: Quick Test (5 minutes)
```bash
python train_style_model.py \
    --model_type triplet \
    --epochs 5 \
    --batch_size 16 \
    --output_dir models/quick_test
```

#### Option B: Full Training (1-2 hours, recommended for submission)
```bash
python train_style_model.py \
    --model_type triplet \
    --epochs 50 \
    --batch_size 32 \
    --hidden_channels 256 \
    --num_blocks 10 \
    --embedding_dim 512 \
    --output_dir models/final
```

### Step 3: Generate Submission
```bash
python infer_style.py \
    --model_path models/final/best_model.pth \
    --output submission.csv
```

---

## 📊 What You'll See

### During Training
```
Epoch 1/50 [Train]: 100%|██████| 150/150 [00:45<00:00, loss=0.523]
Epoch 1/50 - Average Loss: 0.5234
✓ Saved best model (loss: 0.5234)
```

### During Inference
```
Extracting Query Player Embeddings
Query players: 100%|██████| 600/600 [01:23<00:00]
✓ Extracted 600 query embeddings

Extracting Candidate Player Embeddings  
Candidate players: 100%|██████| 600/600 [01:25<00:00]
✓ Extracted 600 candidate embeddings

✓ Computed top-5 matches for 600 query players
✓ Saved results to submission.csv
```

### Output File (submission.csv)
```csv
query_id,rank1,rank2,rank3,rank4,rank5
1,145,298,67,412,89
2,234,567,12,345,678
...
```

---

## 🎛️ Key Parameters

### For Better Accuracy (slower)
```bash
--epochs 100             # More training
--hidden_channels 512    # Bigger model
--num_blocks 15          # Deeper model
--batch_size 64          # More samples per batch
```

### For Faster Training (less accurate)
```bash
--epochs 10              # Less training  
--hidden_channels 128    # Smaller model
--num_blocks 5           # Shallower model
--batch_size 8           # Fewer samples per batch
```

---

## 💡 Tips

1. **Use GPU**: Automatically detected, 20-30x faster than CPU
2. **Monitor Loss**: Should decrease over epochs
3. **Save Often**: Use `--save_interval 5` for frequent checkpoints
4. **Try Different Models**: Experiment with hyperparameters

---

## ❓ Common Issues

### Out of Memory
```bash
# Reduce batch size
--batch_size 8
```

### Slow Training
```bash
# Use more workers
--num_workers 8
```

### Poor Results
```bash
# Train longer
--epochs 100
# Or use bigger model
--num_blocks 15 --hidden_channels 512
```

---

## 📚 More Information

- **Full Documentation**: `README_STYLE_DETECTION.md`
- **Step-by-Step Guide**: `IMPLEMENTATION_GUIDE.md`
- **Implementation Details**: `SUMMARY.md`

---

## 🎯 Typical Workflow

```bash
# 1. Verify everything works
python test_system.py

# 2. Quick test (optional but recommended)
python train_style_model.py --epochs 5 --output_dir models/test

# 3. Full training
python train_style_model.py --epochs 50 --output_dir models/final

# 4. Generate submission
python infer_style.py --model_path models/final/best_model.pth

# 5. Submit!
cat submission.csv
```

---

**That's it! You're ready to train and generate your submission. Good luck! 🎉**

