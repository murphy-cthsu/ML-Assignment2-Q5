"""
Inference script for Go playing style matching.

Loads a trained model and matches query players to candidate players
based on style similarity.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model
from train_style_model import GoGameSequenceDataset


class StyleInference:
    """
    Inference class for Go playing style matching.
    """
    
    def __init__(self, model_path, config_file, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_file: Path to configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config_file = config_file
        
        # Load configuration
        style_py.load_config_file(config_file)
        
        # Load model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model arguments from checkpoint
        if 'args' in checkpoint:
            model_args = checkpoint['args']
        else:
            # Use default arguments if not saved
            print("Warning: Model arguments not found in checkpoint, using defaults")
            model_args = {
                'hidden_channels': 256,
                'num_blocks': 5,
                'move_embed_dim': 256,
                'num_heads': 8,
                'num_transformer_layers': 4,
                'style_embed_dim': 512
            }
        
        # Create model
        self.model = create_model(
            model_type='triplet',
            input_channels=18,
            hidden_channels=model_args.get('hidden_channels', 256),
            num_blocks=model_args.get('num_blocks', 5),
            move_embed_dim=model_args.get('move_embed_dim', 256),
            num_heads=model_args.get('num_heads', 8),
            num_layers=model_args.get('num_transformer_layers', 4),
            style_embed_dim=model_args.get('style_embed_dim', 512)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Using device: {self.device}")
    
    def extract_player_embedding(self, sgf_directory, max_moves=200, batch_size=8, games_per_player=10):
        """
        Extract style embeddings for all players in a directory.
        
        Args:
            sgf_directory: Directory containing SGF files for players
            max_moves: Maximum number of moves per game
            batch_size: Batch size for processing
            games_per_player: Number of games to sample per player
            
        Returns:
            embeddings: numpy array of shape (num_players, style_embed_dim)
            player_ids: list of player IDs
        """
        print(f"\nExtracting embeddings from: {sgf_directory}")
        
        # Create dataset
        dataset = GoGameSequenceDataset(
            sgf_directory,
            self.config_file,
            max_moves=max_moves,
            num_samples=1000  # Dummy value, we'll use manual sampling
        )
        
        player_ids = list(range(dataset.num_players))
        print(f"Found {len(player_ids)} players")
        print(f"Will sample {games_per_player} games per player")
        
        # Extract embeddings for each player (average over their games)
        player_embeddings = []
        
        with torch.no_grad():
            for player_id in tqdm(player_ids, desc="Extracting player embeddings"):
                game_embeddings = []
                
                # Sample multiple games for this player using corrected API
                for _ in range(games_per_player):
                    # Correct API: get_random_feature_and_label(player_num, game_id, start, is_train)
                    game_id = 0  # Will be randomized by C++ when is_train=True
                    start = 0
                    is_train = True
                    
                    # Get flat features from C++ API
                    flat_features = dataset.data_loader.get_random_feature_and_label(
                        player_id, game_id, start, is_train
                    )
                    
                    # Calculate actual number of frames and reshape
                    feature_size = 18 * 19 * 19  # 6498
                    actual_n_frames = len(flat_features) // feature_size
                    
                    if actual_n_frames == 0:
                        continue
                    
                    # Reshape to (n_frames, 18, 19, 19)
                    features = np.array(flat_features, dtype=np.float32).reshape(actual_n_frames, 18, 19, 19)
                    
                    # Limit to max_moves
                    if actual_n_frames > max_moves:
                        # Sample evenly distributed moves
                        indices = np.linspace(0, actual_n_frames-1, max_moves, dtype=int)
                        features = features[indices]
                    
                    # Convert to tensor (add batch dimension)
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    
                    # Get embedding
                    embedding = self.model.get_embedding(features_tensor)
                    game_embeddings.append(embedding.cpu())
                
                if len(game_embeddings) == 0:
                    # Fallback: create zero embedding
                    print(f"Warning: No games found for player {player_id}, using zero embedding")
                    game_embeddings = [torch.zeros(1, self.model.encoder.aggregator.mlp[-1].out_features)]
                
                # Average embeddings over all games for this player
                game_embeddings = torch.cat(game_embeddings, dim=0)
                player_embedding = game_embeddings.mean(dim=0)
                player_embeddings.append(player_embedding)
        
        # Stack into array
        player_embeddings = torch.stack(player_embeddings).numpy()
        
        print(f"✓ Extracted embeddings: shape {player_embeddings.shape}")
        
        return player_embeddings, player_ids
    
    def match_styles(self, query_embeddings, candidate_embeddings, top_k=5):
        """
        Match query players to candidate players based on style similarity.
        
        Args:
            query_embeddings: numpy array of shape (num_query, embed_dim)
            candidate_embeddings: numpy array of shape (num_candidates, embed_dim)
            top_k: Number of top matches to return
            
        Returns:
            matches: numpy array of shape (num_query, top_k) with candidate indices
        """
        print(f"\nComputing style similarities...")
        print(f"Query players: {len(query_embeddings)}")
        print(f"Candidate players: {len(candidate_embeddings)}")
        
        # Convert to tensors
        query_tensor = torch.FloatTensor(query_embeddings)
        candidate_tensor = torch.FloatTensor(candidate_embeddings)
        
        # Compute cosine similarity matrix
        # similarity[i, j] = cosine similarity between query i and candidate j
        similarity = F.cosine_similarity(
            query_tensor.unsqueeze(1),  # (num_query, 1, embed_dim)
            candidate_tensor.unsqueeze(0),  # (1, num_candidates, embed_dim)
            dim=2
        )
        
        # Get top-k matches for each query
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
        
        print(f"✓ Computed top-{top_k} matches for {len(query_embeddings)} query players")
        
        return top_k_indices.numpy(), top_k_values.numpy()
    
    def save_results(self, matches, query_ids, candidate_ids, output_file):
        """
        Save matching results in submission format.
        Format: id,label where both are 1-indexed player IDs
        Sorted lexicographically by id to match submission_sample.csv format
        """
        # Collect all predictions
        predictions = []
        for query_idx, query_id in enumerate(query_ids):
            # Get best match (rank 0) for this query
            candidate_idx = matches[query_idx, 0]
            candidate_id = candidate_ids[candidate_idx]
            
            # Convert to 1-indexed IDs (query_id+1, candidate_id+1)
            # Assuming playerXXX.sgf files are numbered starting from 1
            query_id_1indexed = query_id + 1
            candidate_id_1indexed = candidate_id + 1
            
            predictions.append([query_id_1indexed, candidate_id_1indexed])
        
        # Sort lexicographically by query_id (as string) to match submission_sample.csv
        predictions_sorted = sorted(predictions, key=lambda x: str(x[0]))
        
        # Write to file
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header for submission format
            writer.writerow(['id', 'label'])
            
            # Write sorted predictions
            writer.writerows(predictions_sorted)
        
        print(f"✓ Saved {len(predictions_sorted)} predictions in submission format")
        print(f"✓ Format: id (query player 1-indexed), label (best match candidate 1-indexed)")
        print(f"✓ Sorted lexicographically to match submission_sample.csv")


def main():
    parser = argparse.ArgumentParser(description='Inference for Go playing style matching')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_file', type=str, default='conf.cfg',
                        help='Path to configuration file')
    
    # Data parameters
    parser.add_argument('--query_dir', type=str, default='test_set/query_set',
                        help='Directory containing query player SGF files')
    parser.add_argument('--candidate_dir', type=str, default='test_set/cand_set',
                        help='Directory containing candidate player SGF files')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Maximum number of moves per game')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--games_per_player', type=int, default=10,
                        help='Number of games to sample per player for embedding')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top matches to return (for debugging)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = StyleInference(args.model_path, args.config_file, args.device)
    
    # Extract query embeddings
    print("\n" + "="*60)
    print("EXTRACTING QUERY PLAYER EMBEDDINGS")
    print("="*60)
    query_embeddings, query_ids = inference.extract_player_embedding(
        args.query_dir,
        max_moves=args.max_moves,
        batch_size=args.batch_size,
        games_per_player=args.games_per_player
    )
    
    # Extract candidate embeddings
    print("\n" + "="*60)
    print("EXTRACTING CANDIDATE PLAYER EMBEDDINGS")
    print("="*60)
    candidate_embeddings, candidate_ids = inference.extract_player_embedding(
        args.candidate_dir,
        max_moves=args.max_moves,
        batch_size=args.batch_size,
        games_per_player=args.games_per_player
    )
    
    # Match styles
    print("\n" + "="*60)
    print("MATCHING STYLES")
    print("="*60)
    matches, similarities = inference.match_styles(
        query_embeddings,
        candidate_embeddings,
        top_k=args.top_k
    )
    
    # Print some example matches
    print("\nExample matches:")
    for i in range(min(5, len(query_ids))):
        print(f"Query player {query_ids[i]}:")
        for rank in range(args.top_k):
            candidate_idx = matches[i, rank]
            candidate_id = candidate_ids[candidate_idx]
            similarity = similarities[i, rank]
            print(f"  Rank {rank+1}: Candidate {candidate_id} (similarity: {similarity:.4f})")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    inference.save_results(matches, query_ids, candidate_ids, args.output)
    
    print("\n" + "="*60)
    print("✓ INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
