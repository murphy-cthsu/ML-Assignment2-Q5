# Inference Guide - Go Style Detection

## Quick Start

After training is complete, run inference to match query players with candidate players:

### Basic Inference Command

```bash
cd /home/cthsu/Workspace/ML_2025/HW2/Q5

# Using the trained model
conda run -n ml_hw2 python infer_style.py \
    --model_path models/full_training_medium/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output submission.csv \
    --max_moves 75 \
    --top_k 5
```

## Command Options

### Required Parameters

- `--model_path`: Path to your trained model (e.g., `models/full_training_medium/best_model.pth`)
- `--query_dir`: Directory with query players (default: `test_set/query_set`)
- `--candidate_dir`: Directory with candidate players (default: `test_set/cand_set`)

### Optional Parameters

- `--output`: Output CSV filename (default: `submission.csv`)
- `--max_moves`: Maximum moves per game (default: 200)
  - Use same value as training for best results
  - Medium model: 75
  - Large model: 100
- `--top_k`: Number of top matches to return (default: 5)
- `--config_file`: Config file path (default: `conf.cfg`)
- `--device`: Device to use (default: `cuda`)
- `--batch_size`: Batch size for inference (default: 8)

## Examples

### Using Medium Model
```bash
conda run -n ml_hw2 python infer_style.py \
    --model_path models/full_training_medium/best_model.pth \
    --max_moves 75 \
    --output predictions_medium.csv
```

### Using Large Model
```bash
conda run -n ml_hw2 python infer_style.py \
    --model_path models/full_training_large/best_model.pth \
    --max_moves 100 \
    --output predictions_large.csv
```

### Using Test Model (Quick Test)
```bash
conda run -n ml_hw2 python infer_style.py \
    --model_path models/test_run/best_model.pth \
    --max_moves 30 \
    --output test_predictions.csv
```

### Get Top 10 Matches
```bash
conda run -n ml_hw2 python infer_style.py \
    --model_path models/full_training_medium/best_model.pth \
    --top_k 10 \
    --output predictions_top10.csv
```

## What Happens During Inference

1. **Load Model**: Loads your trained model checkpoint
2. **Extract Query Embeddings**: Processes each query player
   - Samples 10 games per player
   - Extracts style embeddings
   - Averages embeddings for each player
3. **Extract Candidate Embeddings**: Processes each candidate player
   - Same process as query players
4. **Compute Similarities**: Calculates cosine similarity between all pairs
5. **Match Players**: Finds top-k most similar candidates for each query
6. **Save Results**: Writes results to CSV file

## Output Format

The output CSV file has this format:

```csv
query_id,rank1,rank2,rank3,rank4,rank5
0,342,156,89,23,501
1,78,234,445,12,389
...
```

Where:
- `query_id`: The query player ID (0-599)
- `rank1`: Most similar candidate player
- `rank2`: Second most similar candidate
- etc.

## Monitoring Progress

During inference, you'll see progress like:

```
======================================================================
Go Style Detection - Inference
======================================================================

Loading model from: models/full_training_medium/best_model.pth
✓ Model loaded successfully
✓ Using device: cuda

======================================================================
EXTRACTING QUERY PLAYER EMBEDDINGS
======================================================================

Extracting embeddings from: test_set/query_set
Loading SGF files: 100%|████████████| 600/600 [00:10<00:00, 58.43it/s]
Found 600 players
Will sample 10 games per player

Extracting player embeddings: 100%|████████| 600/600 [02:30<00:00, 4.0it/s]
✓ Extracted embeddings: shape (600, 256)

======================================================================
EXTRACTING CANDIDATE PLAYER EMBEDDINGS
======================================================================

Extracting embeddings from: test_set/cand_set
Loading SGF files: 100%|████████████| 600/600 [00:10<00:00, 58.43it/s]
Found 600 players
Will sample 10 games per player

Extracting player embeddings: 100%|████████| 600/600 [02:30<00:00, 4.0it/s]
✓ Extracted embeddings: shape (600, 256)

======================================================================
MATCHING STYLES
======================================================================

Computing style similarities...
Query players: 600
Candidate players: 600
✓ Computed top-5 matches for 600 query players

Example matches:
Query player 0:
  Rank 1: Candidate 342 (similarity: 0.9234)
  Rank 2: Candidate 156 (similarity: 0.9102)
  ...

======================================================================
SAVING RESULTS
======================================================================

Saving results to: submission.csv
✓ Saved 600 query results

======================================================================
✓ INFERENCE COMPLETED SUCCESSFULLY!
======================================================================

Results saved to: submission.csv
```

## Inference Time Estimates

| Dataset Size | Model Size | Time |
|--------------|------------|------|
| 600 query + 600 candidates | Small | ~5 min |
| 600 query + 600 candidates | Medium | ~10 min |
| 600 query + 600 candidates | Large | ~15 min |

## Troubleshooting

### "Model file not found"
```bash
# Check if model exists
ls -lh models/full_training_medium/best_model.pth

# Use correct path
--model_path models/full_training_medium/best_model.pth
```

### "Out of memory"
```bash
# Reduce batch size
--batch_size 4

# Or reduce max_moves
--max_moves 50
```

### "No games found for player"
- This warning is normal if a player has very few games
- The script will use a zero embedding as fallback

### Inference is slow
- This is normal - processing 600 players takes time
- Use GPU for faster inference (automatic if available)
- Reduce `games_per_player` in code if needed

## Advanced Usage

### Run in Background
```bash
nohup conda run -n ml_hw2 python infer_style.py \
    --model_path models/full_training_medium/best_model.pth \
    --output submission.csv \
    > inference.log 2>&1 &

# Monitor progress
tail -f inference.log
```

### Multiple Predictions
```bash
# Generate predictions with different models
for model in models/*/best_model.pth; do
    model_name=$(basename $(dirname $model))
    conda run -n ml_hw2 python infer_style.py \
        --model_path $model \
        --output predictions_${model_name}.csv
done
```

## Verify Results

After inference:

```bash
# Check output file
head submission.csv

# Count lines (should be 601: header + 600 query players)
wc -l submission.csv

# Check file size
ls -lh submission.csv
```

## Next Steps

1. Run inference with your trained model
2. Check the output CSV file
3. Optionally: Ensemble multiple models for better results
4. Submit your predictions!
