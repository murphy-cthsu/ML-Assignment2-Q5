# Go Playing Style Detection

This project implements a deep learning system for detecting and matching Go playing styles using convolutional neural networks.

## Overview

The system learns to:
1. Extract style embeddings from Go game positions
2. Match players based on playing style similarity
3. Identify similar players from a candidate pool

## Architecture

### Models
- **Style Encoder**: ResNet-based CNN for extracting style features
- **Triplet Loss Model**: Metric learning for style similarity
- **Supervised Classifier**: Direct player identification

### Key Features
- Multi-residual block architecture for deep feature extraction
- L2-normalized embeddings for robust similarity computation
- Supports both triplet and supervised learning strategies

## Installation

### Prerequisites
```bash
# Activate conda environment
conda activate ml_hw2

# Install required packages
pip install torch torchvision tqdm
```

### Build C++ Module
```bash
cd Q5
bash scripts/build.sh go
```

## Usage

### 1. Training

#### Train with Triplet Loss (Recommended for Style Matching)
```bash
python train_style_model.py \
    --model_type triplet \
    --train_dir train_set \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --hidden_channels 256 \
    --num_blocks 10 \
    --embedding_dim 512 \
    --output_dir models/triplet
```

#### Train with Supervised Learning
```bash
python train_style_model.py \
    --model_type supervised \
    --train_dir train_set \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --hidden_channels 256 \
    --num_blocks 10 \
    --output_dir models/supervised
```

#### Training Options
- `--model_type`: Type of model (`triplet` or `supervised`)
- `--train_dir`: Directory with training SGF files
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--hidden_channels`: Number of CNN channels
- `--num_blocks`: Number of residual blocks
- `--embedding_dim`: Dimension of style embedding vector
- `--n_frames`: Number of positions sampled per game
- `--move_step`: Step size between sampled positions
- `--margin`: Margin for triplet loss
- `--output_dir`: Directory to save models

### 2. Inference

```bash
python infer_style.py \
    --model_path models/triplet/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output submission.csv \
    --top_k 5 \
    --n_frames 10
```

#### Inference Options
- `--model_path`: Path to trained model checkpoint
- `--query_dir`: Directory with query player SGF files
- `--candidate_dir`: Directory with candidate player SGF files
- `--output`: Output CSV file
- `--top_k`: Number of top matches to return (default: 5)
- `--n_frames`: Number of positions to sample per game
- `--device`: Device to use (`cuda` or `cpu`)

### 3. Output Format

The inference script generates a CSV file with the following format:
```
query_id,rank1,rank2,rank3,rank4,rank5
1,145,298,67,412,89
2,234,567,12,345,678
...
```

Where:
- `query_id`: Query player ID (1-indexed)
- `rank1-rank5`: Top 5 matched candidate player IDs (1-indexed)

## Model Architecture

### Style Encoder
```
Input (18 x 19 x 19)
  ↓
Conv2D (18 → 256) + BatchNorm + ReLU
  ↓
Residual Block × 10
  ↓
Global Average Pooling (256 → 256)
  ↓
Fully Connected (256 → 512)
  ↓
L2 Normalization
  ↓
Output Embedding (512-dim)
```

### Residual Block
```
Input
  ↓
Conv2D (3x3) + BatchNorm + ReLU
  ↓
Conv2D (3x3) + BatchNorm
  ↓
Add Input (Skip Connection)
  ↓
ReLU
  ↓
Output
```

## Data Format

### Input SGF Files
- Standard Go SGF format
- 19x19 board size
- Organized by player directories or player*.sgf files

### Feature Representation
- **Input channels**: 18 (board state encoding)
  - Channels 0-7: Current player's stones (8 time steps)
  - Channels 8-15: Opponent's stones (8 time steps)
  - Channel 16: Color to play
  - Channel 17: Legal move mask

## Training Strategy

### Triplet Loss
- **Anchor**: Position from player A
- **Positive**: Another position from player A
- **Negative**: Position from different player B
- **Objective**: Minimize distance between anchor-positive, maximize anchor-negative

### Supervised Classification
- **Input**: Board position
- **Output**: Player ID (600 classes)
- **Objective**: Maximize classification accuracy

## Performance Tips

1. **Batch Size**: Use larger batches (32-64) for stable training
2. **Learning Rate**: Start with 0.001, reduce on plateau
3. **Data Augmentation**: Enabled via random rotation
4. **Sampling**: Sample diverse positions across game stages
5. **GPU**: Use CUDA for faster training (20-30x speedup)

## Configuration

Edit `conf.cfg` to adjust:
- Network architecture (blocks, channels)
- Training hyperparameters (batch size, learning rate)
- Data sampling (n_frames, move_step_to_choose)

Key configuration values:
```properties
learner_batch_size=1024
learner_learning_rate=0.01
nn_num_blocks=1
nn_num_hidden_channels=256
n_frames=10
move_step_to_choose=4
```

## File Structure

```
Q5/
├── style_model.py          # Neural network model definitions
├── train_style_model.py    # Training script
├── infer_style.py          # Inference script
├── conf.cfg                # Configuration file
├── train_set/              # Training SGF files
│   ├── 1.sgf
│   ├── 2.sgf
│   └── ...
├── test_set/
│   ├── query_set/          # Query player SGF files
│   │   ├── player001.sgf
│   │   └── ...
│   └── cand_set/           # Candidate player SGF files
│       ├── player001.sgf
│       └── ...
├── models/                 # Saved model checkpoints
│   ├── best_model.pth
│   └── training_args.json
└── submission.csv          # Inference results
```

## Example Workflow

```bash
# 1. Test model creation
python style_model.py

# 2. Train model with triplet loss
python train_style_model.py \
    --model_type triplet \
    --train_dir train_set \
    --epochs 30 \
    --batch_size 32 \
    --output_dir models/my_model

# 3. Run inference
python infer_style.py \
    --model_path models/my_model/best_model.pth \
    --query_dir test_set/query_set \
    --candidate_dir test_set/cand_set \
    --output my_submission.csv

# 4. Check results
head my_submission.csv
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--hidden_channels` (try 128)
- Reduce `--num_blocks` (try 5)

### Slow Training
- Increase `--num_workers` for data loading
- Use smaller `--n_frames` (try 5-7)
- Enable CUDA if available

### Poor Matching Performance
- Train for more epochs (50-100)
- Increase model capacity (more blocks/channels)
- Adjust triplet margin (try 0.5-2.0)
- Use more training data

## Advanced Usage

### Custom Model Architecture
Edit `style_model.py` to modify:
- Number of residual blocks
- Channel dimensions
- Embedding dimension
- Activation functions

### Custom Training Loss
Implement new loss functions in `train_style_model.py`:
- Contrastive loss
- Center loss
- Angular margin loss

### Ensemble Methods
Combine multiple models:
```python
# Load multiple models
model1 = StyleInference('models/model1/best_model.pth', 'conf.cfg')
model2 = StyleInference('models/model2/best_model.pth', 'conf.cfg')

# Average embeddings
embedding = (model1.extract_player_embedding(...) + 
             model2.extract_player_embedding(...)) / 2
```

## References

- FaceNet: A Unified Embedding for Face Recognition and Clustering
- Deep Residual Learning for Image Recognition
- Go game SGF format specification

## License

This project is for educational purposes (ML Assignment 2, Question 5).

## Authors

Machine Learning Course, 2025
