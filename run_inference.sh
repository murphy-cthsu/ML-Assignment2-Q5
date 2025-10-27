#!/bin/bash

# Run inference with the trained model
# Usage: ./run_inference.sh

cd /home/cthsu/Workspace/ML_2025/HW2/Q5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_hw2

echo "======================================================================="
echo "Running Inference - Go Style Detection"
echo "======================================================================="
echo ""
echo "Model: models/medium_subset/best_model.pth"
echo "Query players: test_set/query_set (600 players)"
echo "Candidate players: test_set/cand_set (600 players)"
echo "Output: submission.csv"
echo ""
echo "======================================================================="
echo ""

python infer_style.py \
    --model_path models/medium_subset/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output submission.csv \
    --games_per_player 10 \
    --max_moves 75 \
    --batch_size 8

echo ""
echo "======================================================================="
echo "Inference completed!"
echo ""
echo "Results saved to: submission.csv"
echo ""
echo "Verify the output:"
echo "  head -20 submission.csv"
echo "  wc -l submission.csv  # Should be 601 (header + 600 predictions)"
echo ""
echo "Compare with sample:"
echo "  diff <(head -10 submission.csv | cut -d',' -f1) <(head -10 submission_sample.csv | cut -d',' -f1)"
echo "======================================================================="
