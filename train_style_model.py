"""
Training script for Go playing style detection model.

Architecture from paper:
1. Per-move encoding: Residual CNN + MLP
2. Game-level aggregation: Transformer + MLP + tanh + normalize

Trains using triplet loss for metric learning.
#
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
from datetime import datetime
import random

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model


class GoGameSequenceDataset(Dataset):
    """
    Dataset for loading sequences of Go board states from games.
    Each sample is a full game (sequence of moves).
    Uses get_random_feature_and_label for efficient random sampling.
    """
    
    def __init__(self, sgf_directory, config_file, max_moves=200, num_samples=10000):
        """
        Args:
            sgf_directory: Directory containing SGF files
            config_file: Path to configuration file
            max_moves: Maximum number of moves to extract per game
            num_samples: Number of random samples to generate per epoch
        """
        self.sgf_directory = sgf_directory
        self.max_moves = max_moves
        self.num_samples = num_samples
        
        # Load configuration
        style_py.load_config_file(config_file)
        
        # Create data loader
        self.data_loader = style_py.DataLoader(sgf_directory)
        
        # Load all SGF files in the directory
        print(f"Loading SGF files from {sgf_directory}...")
        sgf_files = sorted([f for f in os.listdir(sgf_directory) if f.endswith('.sgf')])
        print(f"  Found {len(sgf_files)} SGF files")
        
        for sgf_file in tqdm(sgf_files, desc="Loading SGF files"):
            filepath = os.path.join(sgf_directory, sgf_file)
            self.data_loader.load_data_from_file(filepath)
        
        # Get dataset statistics
        self.num_players = self.data_loader.get_num_of_player()
        
        print(f"Loaded dataset:")
        print(f"  - Players: {self.num_players}")
        print(f"  - Will generate {num_samples} random samples per epoch")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a random game from a random player.
        Returns (features, label) where label is the player ID.
        """
        # Get random player
        player_id = np.random.randint(0, self.num_players)
        
        # Get random game from that player (game_id will be randomized by C++ if is_train=True)
        game_id = 0  # Will be randomized by C++ when is_train=True
        start = 0
        is_train = True
        
        # Returns flat list of floats representing the game features
        # API: get_random_feature_and_label(player_num, game_id, start, is_train)
        flat_features = self.data_loader.get_random_feature_and_label(
            player_id, game_id, start, is_train
        )
        
        # The C++ code uses getNFrames() to determine how many moves
        # Result is n_frames * 18 * 19 * 19 floats
        feature_size = 18 * 19 * 19  # 6498
        actual_n_frames = len(flat_features) // feature_size
        
        # Reshape flat features to (actual_n_frames, 18, 19, 19)
        features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
        
        # Truncate or pad to max_moves
        if actual_n_frames > self.max_moves:
            features = features[:self.max_moves]
        elif actual_n_frames < self.max_moves:
            pad_size = self.max_moves - actual_n_frames
            padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        # Label is the player_id
        label = player_id
        
        return torch.from_numpy(features), label


class TripletGameDataset(Dataset):
    """
    Dataset that generates triplets of games for triplet loss training.
    Each sample contains:
    - Anchor: a game from player A
    - Positive: another game from player A
    - Negative: a game from player B (B != A)
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: GoGameSequenceDataset instance
        """
        self.base_dataset = base_dataset
        self.num_players = base_dataset.num_players
        self.data_loader = base_dataset.data_loader
        self.max_moves = base_dataset.max_moves
        
        print(f"  - Total players available: {self.num_players}")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _get_game_for_player(self, target_player):
        """
        Helper to get a game from a specific player.
        Uses the C++ API correctly: get_random_feature_and_label(player_num, game_id, start, is_train)
        """
        game_id = 0  # Will be randomized by C++ when is_train=True
        start = 0
        is_train = True
        
        # Get features for this specific player
        flat_features = self.data_loader.get_random_feature_and_label(
            target_player, game_id, start, is_train
        )
        
        # Calculate actual number of frames
        feature_size = 18 * 19 * 19
        actual_n_frames = len(flat_features) // feature_size
        
        # Reshape
        features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
        
        # Truncate or pad to max_moves
        if actual_n_frames > self.max_moves:
            features = features[:self.max_moves]
        elif actual_n_frames < self.max_moves:
            pad_size = self.max_moves - actual_n_frames
            padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        return torch.from_numpy(features), target_player
    
    def __getitem__(self, idx):
        """
        Returns a triplet of game sequences.
        
        Returns:
            anchor, positive, negative: each is (features, label)
        """
        # Get anchor game (random)
        anchor_features, anchor_label = self.base_dataset[idx]
        
        # anchor_label is already an integer, not a tensor
        if isinstance(anchor_label, torch.Tensor):
            anchor_player = anchor_label.item()
        else:
            anchor_player = anchor_label
        
        # Get positive (another game from same player)
        positive_features, positive_label = self._get_game_for_player(anchor_player)
        
        # Get negative (game from different player)
        # Choose a random player different from anchor
        all_players = list(range(self.num_players))
        all_players.remove(anchor_player)
        negative_player = np.random.choice(all_players)
        negative_features, negative_label = self._get_game_for_player(negative_player)
        
        return (anchor_features, anchor_label), (positive_features, positive_label), (negative_features, negative_label)


def collate_fn_triplet(batch):
    """
    Collate function for triplet batches with variable-length sequences.
    
    Args:
        batch: List of ((anchor, a_label), (pos, p_label), (neg, n_label))
    
    Returns:
        Padded and batched triplets with masks
    """
    anchors, positives, negatives = [], [], []
    anchor_labels, pos_labels, neg_labels = [], [], []
    
    for (a, a_label), (p, p_label), (n, n_label) in batch:
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
        anchor_labels.append(a_label)
        pos_labels.append(p_label)
        neg_labels.append(n_label)
    
    # Pad sequences (pad on sequence dimension)
    # Each sequence is (seq_len, channels, H, W)
    anchor_padded = pad_sequence(anchors, batch_first=True, padding_value=0)
    positive_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    negative_padded = pad_sequence(negatives, batch_first=True, padding_value=0)
    
    # Create attention masks (True for padding positions)
    anchor_mask = torch.zeros(len(anchors), anchor_padded.size(1), dtype=torch.bool)
    pos_mask = torch.zeros(len(positives), positive_padded.size(1), dtype=torch.bool)
    neg_mask = torch.zeros(len(negatives), negative_padded.size(1), dtype=torch.bool)
    
    for i, seq in enumerate(anchors):
        anchor_mask[i, len(seq):] = True
    for i, seq in enumerate(positives):
        pos_mask[i, len(seq):] = True
    for i, seq in enumerate(negatives):
        neg_mask[i, len(seq):] = True
    
    # Convert labels to tensors
    anchor_labels_tensor = torch.tensor(anchor_labels, dtype=torch.long)
    pos_labels_tensor = torch.tensor(pos_labels, dtype=torch.long)
    neg_labels_tensor = torch.tensor(neg_labels, dtype=torch.long)
    
    return (anchor_padded, anchor_mask, anchor_labels_tensor), \
           (positive_padded, pos_mask, pos_labels_tensor), \
           (negative_padded, neg_mask, neg_labels_tensor)


def train_triplet_model(args):
    """Train model using triplet loss"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataset
    print("\nLoading training data...")
    base_dataset = GoGameSequenceDataset(
        args.train_dir,
        args.config_file,
        max_moves=args.max_moves,
        num_samples=args.num_samples
    )
    
    triplet_dataset = TripletGameDataset(base_dataset)
    
    # Create data loader
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_triplet
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type='triplet',
        input_channels=18,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        move_embed_dim=args.move_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_transformer_layers,
        style_embed_dim=args.style_embed_dim
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    
    # Triplet loss
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (anchor, anchor_mask, _), (positive, pos_mask, _), (negative, neg_mask, _) in pbar:
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_mask = anchor_mask.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)
            
            # Forward pass
            anchor_embed, positive_embed, negative_embed = model(
                anchor, positive, negative,
                anchor_mask, pos_mask, neg_mask
            )
            
            # Compute loss
            loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Learning rate step
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    print(f"\n✓ Training completed! Best loss: {best_loss:.4f}")
    print(f"✓ Model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")


def main():
    parser = argparse.ArgumentParser(description='Train Go playing style detection model')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='train_set',
                        help='Directory containing training SGF files')
    parser.add_argument('--config_file', type=str, default='conf.cfg',
                        help='Path to configuration file')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Maximum number of moves per game')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples per epoch')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Number of channels in CNN')
    parser.add_argument('--num_blocks', type=int, default=5,
                        help='Number of residual blocks in CNN')
    parser.add_argument('--move_embed_dim', type=int, default=256,
                        help='Dimension of per-move embeddings')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of transformer attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--style_embed_dim', type=int, default=512,
                        help='Dimension of final style embeddings')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=10,
                        help='Learning rate decay step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                        help='Learning rate decay gamma')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Start training
    train_triplet_model(args)


if __name__ == '__main__':
    main()
