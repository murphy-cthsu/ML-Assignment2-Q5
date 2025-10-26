"""
Inference script for Go playing style detection and matching.

This script:
1. Loads a trained style detection model
2. Extracts embeddings from query and candidate player games
3. Performs style matching to identify similar playing styles
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import csv

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model


class StyleInference:
    """
    Inference engine for style detection and matching.
    """
    
    def __init__(self, model_path, config_file, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_file: Path to configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        style_py.load_config_file(config_file)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_args = checkpoint.get('args', {})
        
        # Determine model type
        model_type = self.model_args.get('model_type', 'triplet')
        
        # Create model
        if model_type == 'triplet':
            self.model = create_model(
                'triplet',
                input_channels=18,
                hidden_channels=self.model_args.get('hidden_channels', 256),
                num_blocks=self.model_args.get('num_blocks', 10),
                embedding_dim=self.model_args.get('embedding_dim', 512)
            )
        elif model_type == 'supervised':
            self.model = create_model(
                'supervised',
                input_channels=18,
                hidden_channels=self.model_args.get('hidden_channels', 256),
                num_blocks=self.model_args.get('num_blocks', 10),
                num_players=600  # This should match training
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Model type: {model_type}")
        print(f"  Embedding dim: {self.model_args.get('embedding_dim', 512)}")
    
    def extract_player_embedding(self, sgf_directory, player_id=None, n_frames=10):
        """
        Extract embedding for a player from their games.
        
        Args:
            sgf_directory: Directory containing player's SGF files
            player_id: Player ID (optional, for multi-player directories)
            n_frames: Number of positions to sample per game
        
        Returns:
            Averaged embedding vector for the player
        """
        # Create data loader for this directory
        data_loader = style_py.DataLoader(sgf_directory)
        
        num_players = data_loader.get_num_of_player()
        
        if player_id is None:
            player_id = 0
        
        # Collect all positions for this player
        all_embeddings = []
        
        num_games = data_loader.get_num_of_game_of_player(player_id)
        
        for game_idx in range(num_games):
            # Get features for this game
            features, label = data_loader.get_feature_and_label(player_id, game_idx)
            
            # Convert to numpy if needed
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Sample positions
            num_positions = features.shape[0]
            
            if num_positions > 0:
                # Sample evenly spaced positions
                if num_positions >= n_frames:
                    indices = np.linspace(0, num_positions-1, n_frames, dtype=int)
                    sampled_features = features[indices]
                else:
                    sampled_features = features
                
                # Convert to tensor and get embeddings
                positions = torch.FloatTensor(sampled_features).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.model.get_embedding(positions)
                    all_embeddings.append(embeddings.cpu().numpy())
        
        if len(all_embeddings) == 0:
            raise ValueError(f"No valid positions found for player {player_id}")
        
        # Average all embeddings
        all_embeddings = np.vstack(all_embeddings)
        player_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize
        player_embedding = player_embedding / (np.linalg.norm(player_embedding) + 1e-8)
        
        return player_embedding
    
    def extract_single_sgf_embedding(self, sgf_path, n_frames=10):
        """
        Extract embedding from a single SGF file.
        
        Args:
            sgf_path: Path to SGF file
            n_frames: Number of positions to sample
        
        Returns:
            Embedding vector
        """
        # Create a temporary directory with just this SGF
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy SGF to temp directory
            basename = os.path.basename(sgf_path)
            shutil.copy(sgf_path, os.path.join(tmpdir, basename))
            
            # Extract embedding
            embedding = self.extract_player_embedding(tmpdir, player_id=0, n_frames=n_frames)
        
        return embedding
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1, embedding2: Embedding vectors
            metric: Similarity metric ('cosine' or 'euclidean')
        
        Returns:
            Similarity score (higher = more similar for cosine)
        """
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return similarity
        elif metric == 'euclidean':
            # Negative euclidean distance (so higher is more similar)
            distance = np.linalg.norm(embedding1 - embedding2)
            return -distance
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def match_styles(self, query_dir, candidate_dir, top_k=5, n_frames=10):
        """
        Match query players to candidate players based on style similarity.
        
        Args:
            query_dir: Directory containing query player SGF files
            candidate_dir: Directory containing candidate player SGF files
            top_k: Number of top matches to return per query
            n_frames: Number of positions to sample per game
        
        Returns:
            Dictionary mapping query player IDs to ranked list of candidate matches
        """
        # Load query players
        print("\n" + "="*60)
        print("Extracting Query Player Embeddings")
        print("="*60)
        
        query_loader = style_py.DataLoader(query_dir)
        num_query_players = query_loader.get_num_of_player()
        
        query_embeddings = []
        for player_id in tqdm(range(num_query_players), desc="Query players"):
            embedding = self.extract_player_embedding(query_dir, player_id, n_frames)
            query_embeddings.append(embedding)
        
        query_embeddings = np.array(query_embeddings)
        print(f"✓ Extracted {len(query_embeddings)} query embeddings")
        
        # Load candidate players
        print("\n" + "="*60)
        print("Extracting Candidate Player Embeddings")
        print("="*60)
        
        cand_loader = style_py.DataLoader(candidate_dir)
        num_cand_players = cand_loader.get_num_of_player()
        
        cand_embeddings = []
        for player_id in tqdm(range(num_cand_players), desc="Candidate players"):
            embedding = self.extract_player_embedding(candidate_dir, player_id, n_frames)
            cand_embeddings.append(embedding)
        
        cand_embeddings = np.array(cand_embeddings)
        print(f"✓ Extracted {len(cand_embeddings)} candidate embeddings")
        
        # Compute similarity matrix
        print("\n" + "="*60)
        print("Computing Similarity Matrix")
        print("="*60)
        
        # Cosine similarity: query x candidate
        similarity_matrix = np.dot(query_embeddings, cand_embeddings.T)
        
        # Get top-k matches for each query
        matches = {}
        for query_id in range(num_query_players):
            similarities = similarity_matrix[query_id]
            
            # Get top-k candidates
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_scores = similarities[top_k_indices]
            
            matches[query_id] = [
                (int(cand_id), float(score))
                for cand_id, score in zip(top_k_indices, top_k_scores)
            ]
        
        print(f"✓ Computed top-{top_k} matches for {num_query_players} query players")
        
        return matches
    
    def save_results(self, matches, output_file):
        """
        Save matching results to CSV file.
        
        Args:
            matches: Dictionary of matches from match_styles()
            output_file: Path to output CSV file
        """
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query_id', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'])
            
            for query_id in sorted(matches.keys()):
                match_list = matches[query_id]
                # Get candidate IDs (1-indexed for output)
                top_5_ids = [cand_id + 1 for cand_id, score in match_list[:5]]
                
                # Pad with -1 if less than 5 matches
                while len(top_5_ids) < 5:
                    top_5_ids.append(-1)
                
                writer.writerow([query_id + 1] + top_5_ids)
        
        print(f"✓ Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Go playing style inference and matching')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='conf.cfg',
                        help='Configuration file')
    
    # Data arguments
    parser.add_argument('--query_dir', type=str, default='test_set/query_set',
                        help='Directory containing query player SGF files')
    parser.add_argument('--candidate_dir', type=str, default='test_set/cand_set',
                        help='Directory containing candidate player SGF files')
    parser.add_argument('--n_frames', type=int, default=10,
                        help='Number of frames to extract per game')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top matches to return')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file for results')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = StyleInference(
        model_path=args.model_path,
        config_file=args.config,
        device=args.device
    )
    
    # Perform style matching
    matches = inference.match_styles(
        query_dir=args.query_dir,
        candidate_dir=args.candidate_dir,
        top_k=args.top_k,
        n_frames=args.n_frames
    )
    
    # Save results
    inference.save_results(matches, args.output)
    
    print("\n" + "="*60)
    print("Inference Completed Successfully!")
    print("="*60)
    print(f"\nResults saved to: {args.output}")
    
    # Show sample results
    print("\nSample results (first 5 queries):")
    print("-" * 60)
    for query_id in range(min(5, len(matches))):
        print(f"Query {query_id+1:3d}: ", end="")
        match_list = matches[query_id]
        for rank, (cand_id, score) in enumerate(match_list[:5], 1):
            print(f"#{rank}=Player{cand_id+1:03d}({score:.3f}) ", end="")
        print()


if __name__ == "__main__":
    main()
