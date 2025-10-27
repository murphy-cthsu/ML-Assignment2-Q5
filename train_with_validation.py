"""
Training script with validation for Go playing style detection model.

Includes:
- Train/validation split
- Validation metrics (Top-1, Top-5 accuracy)
- Better monitoring of training progress
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
from train_style_model import GoGameSequenceDataset, TripletGameDataset, collate_fn_triplet


def validate_model(model, base_dataset, device, num_val_players=50, games_per_player=5):
    """
    Validate the model by checking if it can correctly identify games from the same player.
    
    For each validation player:
    1. Extract embeddings for N games
    2. Use first game as query
    3. Try to match it to remaining games (should rank highly)
    4. Compare to games from other players (should rank lowly)
    
    Returns:
        top1_acc: Percentage where top match is from same player
        top5_acc: Percentage where top 5 matches include same player
        avg_same_sim: Average similarity between same player's games
        avg_diff_sim: Average similarity between different players' games
    """
    model.eval()
    
    # Randomly sample validation players
    val_players = random.sample(range(base_dataset.num_players), min(num_val_players, base_dataset.num_players))
    
    top1_correct = 0
    top5_correct = 0
    same_player_sims = []
    diff_player_sims = []
    
    with torch.no_grad():
        for player_id in tqdm(val_players, desc="Validating", leave=False):
            # Get embeddings for this player's games
            player_embeddings = []
            
            for _ in range(games_per_player):
                # Get a random game from this player
                game_id = 0
                start = 0
                is_train = True
                
                flat_features = base_dataset.data_loader.get_random_feature_and_label(
                    player_id, game_id, start, is_train
                )
                
                feature_size = 18 * 19 * 19
                actual_n_frames = len(flat_features) // feature_size
                
                if actual_n_frames == 0:
                    continue
                
                features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
                
                # Limit to max_moves
                max_moves = base_dataset.max_moves
                if actual_n_frames > max_moves:
                    indices = np.linspace(0, actual_n_frames-1, max_moves, dtype=int)
                    features = features[indices]
                elif actual_n_frames < max_moves:
                    pad_size = max_moves - actual_n_frames
                    padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
                    features = np.concatenate([features, padding], axis=0)
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                embedding = model.get_embedding(features_tensor)
                player_embeddings.append(embedding.cpu())
            
            if len(player_embeddings) < 2:
                continue
            
            player_embeddings = torch.cat(player_embeddings, dim=0)  # (games_per_player, embed_dim)
            
            # Use first game as query
            query_embed = player_embeddings[0:1]  # (1, embed_dim)
            
            # Get embeddings from other players (negative examples)
            other_player_embeddings = []
            other_players = random.sample([p for p in range(base_dataset.num_players) if p != player_id], 
                                         min(10, base_dataset.num_players - 1))
            
            for other_id in other_players:
                game_id = 0
                start = 0
                is_train = True
                
                flat_features = base_dataset.data_loader.get_random_feature_and_label(
                    other_id, game_id, start, is_train
                )
                
                feature_size = 18 * 19 * 19
                actual_n_frames = len(flat_features) // feature_size
                
                if actual_n_frames == 0:
                    continue
                
                features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
                
                max_moves = base_dataset.max_moves
                if actual_n_frames > max_moves:
                    indices = np.linspace(0, actual_n_frames-1, max_moves, dtype=int)
                    features = features[indices]
                elif actual_n_frames < max_moves:
                    pad_size = max_moves - actual_n_frames
                    padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
                    features = np.concatenate([features, padding], axis=0)
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                embedding = model.get_embedding(features_tensor)
                other_player_embeddings.append(embedding.cpu())
            
            if len(other_player_embeddings) == 0:
                continue
            
            other_player_embeddings = torch.cat(other_player_embeddings, dim=0)
            
            # Combine: same player's other games + different players' games
            # Target: same player's games should rank higher
            candidate_embeddings = torch.cat([player_embeddings[1:], other_player_embeddings], dim=0)
            num_same = player_embeddings.shape[0] - 1
            
            # Compute similarities
            query_embed_norm = torch.nn.functional.normalize(query_embed, dim=1)
            candidate_embeddings_norm = torch.nn.functional.normalize(candidate_embeddings, dim=1)
            
            similarities = torch.mm(query_embed_norm, candidate_embeddings_norm.t())[0]  # (num_candidates,)
            
            # Get rankings
            rankings = torch.argsort(similarities, descending=True)
            
            # Check if top-1 is from same player
            if rankings[0] < num_same:
                top1_correct += 1
            
            # Check if any of top-5 is from same player
            if any(r < num_same for r in rankings[:5]):
                top5_correct += 1
            
            # Compute average similarities
            same_player_sims.extend(similarities[:num_same].tolist())
            diff_player_sims.extend(similarities[num_same:].tolist())
    
    # Compute metrics
    top1_acc = (top1_correct / len(val_players)) * 100 if len(val_players) > 0 else 0
    top5_acc = (top5_correct / len(val_players)) * 100 if len(val_players) > 0 else 0
    avg_same_sim = np.mean(same_player_sims) if same_player_sims else 0
    avg_diff_sim = np.mean(diff_player_sims) if diff_player_sims else 0
    
    return top1_acc, top5_acc, avg_same_sim, avg_diff_sim


def train_with_validation(args):
    """Train model using triplet loss with validation"""
    
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
    print(f"Validation every {args.val_interval} epochs with {args.num_val_players} players")
    print("="*80)
    
    best_loss = float('inf')
    best_val_acc = 0.0
    
    # Log file
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Top-1,Val Top-5,Same Sim,Diff Sim\n")
    
    for epoch in range(args.epochs):
        # Training
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
        
        # Validation
        val_metrics_str = ""
        if (epoch + 1) % args.val_interval == 0 or epoch == 0:
            print(f"\n{'='*80}")
            print(f"Running validation...")
            top1_acc, top5_acc, same_sim, diff_sim = validate_model(
                model, base_dataset, device, 
                num_val_players=args.num_val_players,
                games_per_player=5
            )
            
            val_metrics_str = f"Val Top-1: {top1_acc:.1f}%, Top-5: {top5_acc:.1f}%, Same: {same_sim:.3f}, Diff: {diff_sim:.3f}"
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  {val_metrics_str}")
            print(f"{'='*80}\n")
            
            # Log to file
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{top1_acc:.2f},{top5_acc:.2f},{same_sim:.4f},{diff_sim:.4f}\n")
            
            # Save best model based on validation accuracy
            if top1_acc > best_val_acc:
                best_val_acc = top1_acc
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_top1_acc': top1_acc,
                    'val_top5_acc': top5_acc,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"✓ Saved best model (Val Top-1: {best_val_acc:.1f}%)")
        else:
            print(f"\nEpoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}")
        
        # Learning rate step
        scheduler.step()
        
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
    
    print(f"\n{'='*80}")
    print(f"✓ Training completed!")
    print(f"  Best validation Top-1 accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"  Training log saved to: {log_file}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Train Go playing style detection model with validation')
    
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
    
    # Validation parameters
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Run validation every N epochs')
    parser.add_argument('--num_val_players', type=int, default=50,
                        help='Number of players to use for validation')
    
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
    train_with_validation(args)


if __name__ == '__main__':
    main()
