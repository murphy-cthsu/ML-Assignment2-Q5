"""
Training script with GRADUAL file loading to avoid memory issues.

Loads files in small batches at startup, then keeps them all in memory.
This avoids both:
1. Getting killed during initial load (by loading gradually)
2. Slow training from constant reloading (by keeping all in memory)
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
import random
import gc
import time

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model
from train_style_model import collate_fn_triplet


class GradualLoadDataset(Dataset):
    """
    Dataset that loads ALL files gradually to avoid memory spike.
    Once loaded, keeps everything in memory for fast training.
    """
    
    def __init__(self, sgf_directory, config_file, max_moves=200, num_samples=10000, 
                 load_batch_size=20, load_delay=0.5):
        """
        Args:
            sgf_directory: Directory containing SGF files
            config_file: Path to configuration file
            max_moves: Maximum number of moves per game
            num_samples: Number of samples per epoch
            load_batch_size: Load this many files at a time (to avoid memory spike)
            load_delay: Seconds to wait between batches (let GC work)
        """
        self.sgf_directory = sgf_directory
        self.max_moves = max_moves
        self.num_samples = num_samples
        
        # Load configuration
        style_py.load_config_file(config_file)
        
        # Get all SGF files
        self.sgf_files = sorted([f for f in os.listdir(sgf_directory) if f.endswith('.sgf')])
        self.num_players = len(self.sgf_files)
        
        print(f"\nFound {self.num_players} SGF files in {sgf_directory}")
        print(f"Loading gradually in batches of {load_batch_size} to avoid memory issues...")
        
        # Create data loader and load all files gradually
        self.data_loader = style_py.DataLoader(sgf_directory)
        
        # Load files in small batches with delays
        for i in range(0, len(self.sgf_files), load_batch_size):
            batch_files = self.sgf_files[i:i+load_batch_size]
            
            for sgf_file in tqdm(batch_files, desc=f"Loading batch {i//load_batch_size + 1}", leave=False):
                filepath = os.path.join(sgf_directory, sgf_file)
                self.data_loader.load_data_from_file(filepath)
            
            # Force garbage collection and brief pause
            gc.collect()
            time.sleep(load_delay)
            
            # Progress update
            loaded = min(i + load_batch_size, len(self.sgf_files))
            print(f"  Loaded {loaded}/{len(self.sgf_files)} files ({loaded/len(self.sgf_files)*100:.0f}%)")
        
        print(f"✓ All {self.num_players} players loaded into memory")
        print(f"✓ Will generate {num_samples} random samples per epoch")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a random game from a random player"""
        player_id = np.random.randint(0, self.num_players)
        
        # Get random game
        game_id = 0
        start = 0
        is_train = True
        
        flat_features = self.data_loader.get_random_feature_and_label(
            player_id, game_id, start, is_train
        )
        
        # Process features
        feature_size = 18 * 19 * 19
        actual_n_frames = len(flat_features) // feature_size
        
        features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
        
        # Truncate or pad
        if actual_n_frames > self.max_moves:
            features = features[:self.max_moves]
        elif actual_n_frames < self.max_moves:
            pad_size = self.max_moves - actual_n_frames
            padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        return torch.from_numpy(features), player_id


class TripletGameDataset(Dataset):
    """Triplet dataset wrapper"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.num_players = base_dataset.num_players
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _get_game_for_player(self, target_player):
        """Get a game from a specific player"""
        flat_features = self.base_dataset.data_loader.get_random_feature_and_label(
            target_player, 0, 0, True
        )
        
        feature_size = 18 * 19 * 19
        actual_n_frames = len(flat_features) // feature_size
        
        features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
        
        max_moves = self.base_dataset.max_moves
        if actual_n_frames > max_moves:
            features = features[:max_moves]
        elif actual_n_frames < max_moves:
            pad_size = max_moves - actual_n_frames
            padding = np.zeros((pad_size, 18, 19, 19), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        return torch.from_numpy(features), target_player
    
    def __getitem__(self, idx):
        """Generate a triplet"""
        anchor_player = np.random.randint(0, self.num_players)
        negative_player = np.random.randint(0, self.num_players)
        while negative_player == anchor_player:
            negative_player = np.random.randint(0, self.num_players)
        
        anchor, anchor_label = self._get_game_for_player(anchor_player)
        positive, pos_label = self._get_game_for_player(anchor_player)
        negative, neg_label = self._get_game_for_player(negative_player)
        
        return (anchor, anchor_label), (positive, pos_label), (negative, neg_label)


def validate_model(model, base_dataset, device, num_val_players=30, games_per_player=3):
    """Lightweight validation"""
    model.eval()
    
    val_players = random.sample(range(base_dataset.num_players), min(num_val_players, base_dataset.num_players))
    
    top1_correct = 0
    top5_correct = 0
    same_player_sims = []
    diff_player_sims = []
    
    with torch.no_grad():
        for player_id in tqdm(val_players, desc="Validating", leave=False):
            player_embeddings = []
            
            for _ in range(games_per_player):
                flat_features = base_dataset.data_loader.get_random_feature_and_label(
                    player_id, 0, 0, True
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
                embedding = model.encoder.get_embedding(features_tensor)
                player_embeddings.append(embedding.cpu())
            
            if len(player_embeddings) < 2:
                continue
            
            player_embeddings = torch.cat(player_embeddings, dim=0)
            query_embed = player_embeddings[0:1]
            
            # Get other players
            other_player_embeddings = []
            other_players = random.sample([p for p in range(base_dataset.num_players) if p != player_id], 
                                         min(5, base_dataset.num_players - 1))
            
            for other_id in other_players:
                flat_features = base_dataset.data_loader.get_random_feature_and_label(
                    other_id, 0, 0, True
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
                embedding = model.encoder.get_embedding(features_tensor)
                other_player_embeddings.append(embedding.cpu())
            
            if len(other_player_embeddings) == 0:
                continue
            
            other_player_embeddings = torch.cat(other_player_embeddings, dim=0)
            candidate_embeddings = torch.cat([player_embeddings[1:], other_player_embeddings], dim=0)
            num_same = player_embeddings.shape[0] - 1
            
            # Compute similarities
            query_embed_norm = torch.nn.functional.normalize(query_embed, dim=1)
            candidate_embeddings_norm = torch.nn.functional.normalize(candidate_embeddings, dim=1)
            
            similarities = torch.mm(query_embed_norm, candidate_embeddings_norm.t())[0]
            rankings = torch.argsort(similarities, descending=True)
            
            if rankings[0] < num_same:
                top1_correct += 1
            
            if any(r < num_same for r in rankings[:5]):
                top5_correct += 1
            
            same_player_sims.extend(similarities[:num_same].tolist())
            diff_player_sims.extend(similarities[num_same:].tolist())
    
    top1_acc = (top1_correct / len(val_players)) * 100 if len(val_players) > 0 else 0
    top5_acc = (top5_correct / len(val_players)) * 100 if len(val_players) > 0 else 0
    avg_same_sim = np.mean(same_player_sims) if same_player_sims else 0
    avg_diff_sim = np.mean(diff_player_sims) if diff_player_sims else 0
    
    return top1_acc, top5_acc, avg_same_sim, avg_diff_sim


def train_with_validation(args):
    """Train model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataset with gradual loading
    print("\nInitializing dataset with gradual loading...")
    base_dataset = GradualLoadDataset(
        args.train_dir,
        args.config_file,
        max_moves=args.max_moves,
        num_samples=args.num_samples,
        load_batch_size=args.load_batch_size,
        load_delay=args.load_delay
    )
    
    triplet_dataset = TripletGameDataset(base_dataset)
    
    # Create data loader
    train_loader = DataLoader(
        triplet_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Validation every {args.val_interval} epochs")
    print("="*80)
    
    best_val_acc = 0.0
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Top-1,Val Top-5,Same Sim,Diff Sim\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (anchor, anchor_mask, _), (positive, pos_mask, _), (negative, neg_mask, _) in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_mask = anchor_mask.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)
            
            anchor_embed, positive_embed, negative_embed = model(
                anchor, positive, negative,
                anchor_mask, pos_mask, neg_mask
            )
            
            loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        if (epoch + 1) % args.val_interval == 0 or epoch == 0:
            print(f"\n{'='*80}")
            print(f"Running validation...")
            top1_acc, top5_acc, same_sim, diff_sim = validate_model(
                model, base_dataset, device, 
                num_val_players=args.num_val_players,
                games_per_player=3
            )
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Top-1: {top1_acc:.1f}%, Top-5: {top5_acc:.1f}%, Same: {same_sim:.3f}, Diff: {diff_sim:.3f}")
            print(f"{'='*80}\n")
            
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{top1_acc:.2f},{top5_acc:.2f},{same_sim:.4f},{diff_sim:.4f}\n")
            
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
        
        scheduler.step()
        
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
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_dir', type=str, default='train_set')
    parser.add_argument('--config_file', type=str, default='conf.cfg')
    parser.add_argument('--max_moves', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--load_batch_size', type=int, default=20,
                        help='Load this many files at once during startup')
    parser.add_argument('--load_delay', type=float, default=0.5,
                        help='Seconds to wait between load batches')
    
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--move_embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_transformer_layers', type=int, default=4)
    parser.add_argument('--style_embed_dim', type=int, default=512)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=1.0)
    
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--num_val_players', type=int, default=30)
    
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--save_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    train_with_validation(args)


if __name__ == '__main__':
    main()
