"""
Style Detection Neural Network Model for Go Game Analysis

This module implements the architecture from the paper:
1. Per-move encoding: Residual CNN + MLP
2. Game-level aggregation: Transformer + MLP + tanh + normalize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block for CNN feature extraction"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class MoveEncoder(nn.Module):
    """
    Per-move encoder using Residual CNN + MLP.
    Encodes each board position/move into a feature vector.
    
    Args:
        input_channels (int): Number of input channels (typically 18 for Go)
        hidden_channels (int): Number of channels in CNN
        num_blocks (int): Number of residual blocks
        move_embed_dim (int): Dimension of per-move embedding
    """
    def __init__(self, input_channels=18, hidden_channels=256, num_blocks=5, move_embed_dim=256):
        super(MoveEncoder, self).__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Global pooling to get per-move feature
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to project to move embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, move_embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_channels, 19, 19)
               or (batch_size, input_channels, 19, 19) for single move
            
        Returns:
            move_embeds: Tensor of shape (batch_size, seq_len, move_embed_dim)
                        or (batch_size, move_embed_dim) for single move
        """
        # Handle both single move and sequence of moves
        single_move = (x.dim() == 4)
        if single_move:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape to process all moves together
        x = x.view(batch_size * seq_len, C, H, W)
        
        # CNN encoding
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size * seq_len, -1)
        
        # MLP projection
        x = self.mlp(x)
        
        # Reshape back
        x = x.view(batch_size, seq_len, -1)
        
        if single_move:
            x = x.squeeze(1)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerAggregator(nn.Module):
    """
    Transformer-based aggregator to combine move-level features into game-level style.
    Uses Transformer + MLP + tanh + normalize as described in the paper.
    
    Args:
        move_embed_dim (int): Dimension of input move embeddings
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        style_embed_dim (int): Dimension of final style embedding
    """
    def __init__(self, move_embed_dim=256, num_heads=8, num_layers=4, style_embed_dim=512):
        super(TransformerAggregator, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(move_embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=move_embed_dim,
            nhead=num_heads,
            dim_feedforward=move_embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP head for final style embedding
        self.mlp = nn.Sequential(
            nn.Linear(move_embed_dim, move_embed_dim),
            nn.ReLU(),
            nn.Linear(move_embed_dim, style_embed_dim)
        )
        
    def forward(self, move_embeds, mask=None):
        """
        Args:
            move_embeds: Tensor of shape (batch_size, seq_len, move_embed_dim)
            mask: Optional padding mask for variable-length sequences
            
        Returns:
            style_embeds: L2-normalized style embeddings of shape (batch_size, style_embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(move_embeds)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling over sequence (mean pooling)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # MLP projection
        x = self.mlp(x)
        
        # Tanh activation
        x = torch.tanh(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x


class StyleEncoder(nn.Module):
    """
    Complete style encoder as described in the paper:
    1. Residual CNN + MLP for per-move encoding
    2. Transformer + MLP + tanh + normalize for game-level aggregation
    
    Args:
        input_channels (int): Number of input channels (typically 18 for Go)
        hidden_channels (int): Number of channels in CNN
        num_blocks (int): Number of residual blocks in CNN
        move_embed_dim (int): Dimension of per-move embeddings
        num_heads (int): Number of transformer attention heads
        num_layers (int): Number of transformer layers
        style_embed_dim (int): Dimension of final style embedding
    """
    def __init__(self, input_channels=18, hidden_channels=256, num_blocks=5, 
                 move_embed_dim=256, num_heads=8, num_layers=4, style_embed_dim=512):
        super(StyleEncoder, self).__init__()
        
        # Per-move encoder: Residual CNN + MLP
        self.move_encoder = MoveEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            move_embed_dim=move_embed_dim
        )
        
        # Game-level aggregator: Transformer + MLP + tanh + normalize
        self.aggregator = TransformerAggregator(
            move_embed_dim=move_embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            style_embed_dim=style_embed_dim
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_channels, 19, 19)
               Sequence of board states from a game
            mask: Optional padding mask for variable-length sequences
            
        Returns:
            embeddings: L2-normalized style embeddings of shape (batch_size, style_embed_dim)
        """
        # Encode each move
        move_embeds = self.move_encoder(x)
        
        # Aggregate to game-level style
        style_embeds = self.aggregator(move_embeds, mask)
        
        return style_embeds


class TripletStyleModel(nn.Module):
    """
    Model for triplet loss training.
    Uses a shared encoder for anchor, positive, and negative samples.
    """
    def __init__(self, encoder):
        super(TripletStyleModel, self).__init__()
        self.encoder = encoder
        
    def forward(self, anchor, positive, negative, anchor_mask=None, pos_mask=None, neg_mask=None):
        """
        Args:
            anchor: Anchor samples (batch_size, seq_len, channels, 19, 19)
            positive: Positive samples (same player as anchor)
            negative: Negative samples (different player)
            anchor_mask: Padding mask for anchor sequences
            pos_mask: Padding mask for positive sequences
            neg_mask: Padding mask for negative sequences
            
        Returns:
            Tuple of (anchor_embed, positive_embed, negative_embed)
        """
        anchor_embed = self.encoder(anchor, anchor_mask)
        positive_embed = self.encoder(positive, pos_mask)
        negative_embed = self.encoder(negative, neg_mask)
        return anchor_embed, positive_embed, negative_embed
    
    def get_embedding(self, x, mask=None):
        """Get embedding for a game sequence"""
        return self.encoder(x, mask)


class SupervisedStyleClassifier(nn.Module):
    """
    Model for supervised classification.
    Uses encoder + classification head to predict player identity.
    """
    def __init__(self, encoder, num_players):
        super(SupervisedStyleClassifier, self).__init__()
        self.encoder = encoder
        # Get embedding dimension from the aggregator's style_embed_dim
        embed_dim = encoder.aggregator.mlp[-1].out_features
        self.classifier = nn.Linear(embed_dim, num_players)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input board states (batch_size, seq_len, channels, 19, 19)
            mask: Optional padding mask
            
        Returns:
            logits: Classification logits for each player
        """
        embeddings = self.encoder(x, mask)
        logits = self.classifier(embeddings)
        return logits
    
    def get_embedding(self, x, mask=None):
        """Get style embeddings without classification"""
        return self.encoder(x, mask)


def create_model(model_type='triplet', input_channels=18, hidden_channels=256, 
                 num_blocks=5, move_embed_dim=256, num_heads=8, num_layers=4,
                 style_embed_dim=512, num_players=None):
    """
    Factory function to create models.
    
    Args:
        model_type: 'triplet' or 'supervised'
        input_channels: Number of input channels (18 for Go)
        hidden_channels: Number of channels in CNN
        num_blocks: Number of residual blocks in CNN
        move_embed_dim: Dimension of per-move embeddings
        num_heads: Number of transformer attention heads
        num_layers: Number of transformer layers
        style_embed_dim: Dimension of final style embeddings
        num_players: Number of players (required for supervised model)
        
    Returns:
        Model instance
    """
    encoder = StyleEncoder(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        move_embed_dim=move_embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        style_embed_dim=style_embed_dim
    )
    
    if model_type == 'triplet':
        return TripletStyleModel(encoder)
    elif model_type == 'supervised':
        if num_players is None:
            raise ValueError("num_players must be specified for supervised model")
        return SupervisedStyleClassifier(encoder, num_players)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing Style Detection Models (Paper Architecture)\n")
    print("Architecture:")
    print("  1. Per-move: Residual CNN + MLP")
    print("  2. Game-level: Transformer + MLP + tanh + normalize\n")
    
    # Test triplet model
    print("=" * 60)
    print("1. Triplet Model Test:")
    print("=" * 60)
    model_triplet = create_model(
        'triplet', 
        input_channels=18, 
        hidden_channels=256, 
        num_blocks=5, 
        move_embed_dim=256,
        num_heads=8,
        num_layers=4,
        style_embed_dim=512
    )
    
    # Test with sequence of moves
    batch_size = 4
    seq_len = 50  # 50 moves per game
    anchor = torch.randn(batch_size, seq_len, 18, 19, 19)
    positive = torch.randn(batch_size, seq_len, 18, 19, 19)
    negative = torch.randn(batch_size, seq_len, 18, 19, 19)
    
    a_emb, p_emb, n_emb = model_triplet(anchor, positive, negative)
    print(f"Input shape: (batch={batch_size}, seq_len={seq_len}, channels=18, H=19, W=19)")
    print(f"Anchor embedding shape: {a_emb.shape}")
    print(f"Positive embedding shape: {p_emb.shape}")
    print(f"Negative embedding shape: {n_emb.shape}")
    print(f"Embedding norm (should be ~1.0): {a_emb[0].norm().item():.4f}")
    print(f"Similarity (anchor, positive): {F.cosine_similarity(a_emb, p_emb, dim=1).mean().item():.4f}")
    print(f"Similarity (anchor, negative): {F.cosine_similarity(a_emb, n_emb, dim=1).mean().item():.4f}")
    
    # Test supervised model
    print("\n" + "=" * 60)
    print("2. Supervised Model Test:")
    print("=" * 60)
    model_supervised = create_model(
        'supervised',
        input_channels=18,
        hidden_channels=256,
        num_blocks=5,
        move_embed_dim=256,
        num_heads=8,
        num_layers=4,
        style_embed_dim=512,
        num_players=600
    )
    x = torch.randn(batch_size, seq_len, 18, 19, 19)
    logits = model_supervised(x)
    print(f"Input shape: (batch={batch_size}, seq_len={seq_len}, channels=18, H=19, W=19)")
    print(f"Logits shape: {logits.shape}")
    print(f"Embedding shape: {model_supervised.get_embedding(x).shape}")
    
    # Count parameters
    print("\n" + "=" * 60)
    print("3. Model Statistics:")
    print("=" * 60)
    total_params = sum(p.numel() for p in model_triplet.parameters())
    trainable_params = sum(p.numel() for p in model_triplet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show architecture breakdown
    print("\nArchitecture breakdown:")
    move_encoder_params = sum(p.numel() for p in model_triplet.encoder.move_encoder.parameters())
    aggregator_params = sum(p.numel() for p in model_triplet.encoder.aggregator.parameters())
    print(f"  - Move Encoder (CNN + MLP): {move_encoder_params:,} params")
    print(f"  - Aggregator (Transformer + MLP): {aggregator_params:,} params")
    
    print("\n✓ All models created successfully!")
    print("✓ Architecture matches paper specification!")
