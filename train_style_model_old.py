"""
Training script for Go playing style detection model.

Supports multiple training strategies:
1. Triplet loss for metric learning
2. Contrastive loss for self-supervised learning
3. Supervised classification
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model


class GoStyleDataset(Dataset):
    """
    Dataset for loading Go game positions with player labels.
    
    Loads SGF files and extracts board positions for style learning.
    """
    
    def __init__(self, sgf_directory, config_file, n_frames=10, move_step=4):
        """
        Args:
            sgf_directory: Directory containing SGF files
            config_file: Path to configuration file
            n_frames: Number of frames to extract per game
            move_step: Step size between frames
        """
        self.sgf_directory = sgf_directory
        self.n_frames = n_frames
        self.move_step = move_step
        
        # Load configuration
        style_py.load_config_file(config_file)
        
        # Create data loader
        self.data_loader = style_py.DataLoader(sgf_directory)
        
        # Get dataset statistics
        self.num_players = self.data_loader.get_num_of_player()
        self.num_games = self.data_loader.get_num_of_game()
        
        print(f"Loaded dataset:")
        print(f"  - Players: {self.num_players}")
        print(f"  - Games: {self.num_games}")
        
        # Pre-compute dataset size
        self.positions = []
        self.labels = []
        self._load_all_positions()
    
    def _load_all_positions(self):
        """Pre-load all positions and labels."""
        print("Loading all positions...")
        
        for player_id in tqdm(range(self.num_players), desc="Loading players"):
            num_games_player = self.data_loader.get_num_of_game_of_player(player_id)
            
            for game_idx in range(num_games_player):
                # Get features and labels for this game
                features, label = self.data_loader.get_feature_and_label(player_id, game_idx)
                
                # Convert to numpy array if needed
                if not isinstance(features, np.ndarray):
                    features = np.array(features)
                
                # Features shape should be (num_positions, channels, height, width)
                # Sample positions at regular intervals
                num_positions = features.shape[0]
                
                if num_positions >= self.n_frames:
                    # Sample evenly spaced positions
                    indices = np.linspace(0, num_positions-1, self.n_frames, dtype=int)
                    sampled_features = features[indices]
                else:
                    # If not enough positions, use all available
                    sampled_features = features
                
                # Add each position with its player label
                for pos in sampled_features:
                    self.positions.append(pos)
                    self.labels.append(player_id)
        
        print(f"Total positions loaded: {len(self.positions)}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        """
        Returns:
            Tuple of (position_tensor, player_label)
        """
        position = torch.FloatTensor(self.positions[idx])
        label = self.labels[idx]
        
        return position, label


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.
    
    Returns (anchor, positive, negative) triplets.
    """
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        # Group positions by player
        self.player_positions = {}
        for idx, label in enumerate(base_dataset.labels):
            if label not in self.player_positions:
                self.player_positions[label] = []
            self.player_positions[label].append(idx)
        
        self.players = list(self.player_positions.keys())
        print(f"Triplet dataset: {len(self.players)} unique players")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get anchor
        anchor_pos, anchor_label = self.base_dataset[idx]
        
        # Get positive (same player, different position)
        positive_idx = np.random.choice(self.player_positions[anchor_label])
        positive_pos, _ = self.base_dataset[positive_idx]
        
        # Get negative (different player)
        negative_label = np.random.choice([p for p in self.players if p != anchor_label])
        negative_idx = np.random.choice(self.player_positions[negative_label])
        negative_pos, _ = self.base_dataset[negative_idx]
        
        return anchor_pos, positive_pos, negative_pos, anchor_label


def train_triplet_model(args):
    """Train model using triplet loss."""
    
    print("\n" + "="*60)
    print("Training Style Detection Model with Triplet Loss")
    print("="*60 + "\n")
    
    # Create dataset
    base_dataset = GoStyleDataset(
        sgf_directory=args.train_dir,
        config_file=args.config,
        n_frames=args.n_frames,
        move_step=args.move_step
    )
    
    triplet_dataset = TripletDataset(base_dataset)
    
    dataloader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        'triplet',
        input_channels=18,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        embedding_dim=args.embedding_dim
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (loss: {avg_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n✓ Training completed!")
    return model


def train_supervised_model(args):
    """Train model using supervised classification."""
    
    print("\n" + "="*60)
    print("Training Style Detection Model with Supervised Learning")
    print("="*60 + "\n")
    
    # Create dataset
    dataset = GoStyleDataset(
        sgf_directory=args.train_dir,
        config_file=args.config,
        n_frames=args.n_frames,
        move_step=args.move_step
    )
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        'supervised',
        input_channels=18,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_players=dataset.num_players
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for positions, labels in pbar:
            positions = positions.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(positions)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * train_correct / train_total
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for positions, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                positions = positions.to(device)
                labels = labels.to(device)
                
                logits = model(positions)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("\n✓ Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Go playing style detection model')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default='train_set',
                        help='Directory containing training SGF files')
    parser.add_argument('--config', type=str, default='conf.cfg',
                        help='Configuration file')
    parser.add_argument('--n_frames', type=int, default=10,
                        help='Number of frames to extract per game')
    parser.add_argument('--move_step', type=int, default=4,
                        help='Step size between frames')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='triplet',
                        choices=['triplet', 'supervised', 'contrastive'],
                        help='Type of model to train')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Number of hidden channels')
    parser.add_argument('--num_blocks', type=int, default=10,
                        help='Number of residual blocks')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Dimension of embedding vector')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train model based on type
    if args.model_type == 'triplet':
        model = train_triplet_model(args)
    elif args.model_type == 'supervised':
        model = train_supervised_model(args)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented yet")
    
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
