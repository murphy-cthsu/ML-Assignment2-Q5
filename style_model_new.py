
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Positional Encoding (Sinusoidal)
# ----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ----------------------------
# Residual CNN for per-move encoding
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out

class MoveEncoder(nn.Module):
    # Per-move encoder: small ResNet-ish tower -> global avg pool -> linear
    def __init__(self, in_channels: int = 18, channels: int = 64, num_blocks: int = 6, out_dim: int = 320):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels, 3) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B*T, C, H, W]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x  # [B*T, out_dim]

# ----------------------------
# StyleModel (Transformer aggregator)
# ----------------------------
class StyleModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        move_embed_dim: int = 320,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        out_dim: int = 512,
        max_len: int = 512,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.move_encoder = MoveEncoder(in_channels=in_channels, out_dim=move_embed_dim)
        self.up_proj = nn.Linear(move_embed_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    @staticmethod
    def masked_mean(x, mask, eps: float = 1e-6):
        # x: [B, T, D], mask: [B, T] -> [B, D]
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float()
        s = (x * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return s / (denom.unsqueeze(-1) + eps)

    def forward(self, x, mask=None):
        """Compute normalized style embeddings.
        Args:
            x: FloatTensor [B, T, C, H, W]
            mask: BoolTensor [B, T] (True for valid moves). If None, all valid.
        Returns:
            emb: FloatTensor [B, out_dim] (tanh + L2 normalized)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        per_move = self.move_encoder(x)  # [B*T, E]
        E = per_move.size(-1)
        per_move = per_move.view(B, T, E)

        tokens = self.up_proj(per_move)  # [B, T, d_model]
        tokens = self.pos_encoding(tokens)

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=tokens.device)

        # PyTorch expects True=to be ignored for src_key_padding_mask
        src_key_padding_mask = ~mask  # [B, T]
        z = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)  # [B, T, d_model]

        pooled = self.masked_mean(z, mask)  # [B, d_model]
        out = self.mlp(pooled)              # [B, out_dim]
        out = torch.tanh(out)
        out = F.normalize(out, p=2, dim=-1)
        return out



    # Backwards-compatible helper used by the user's pipeline
    def get_embedding(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """x: [B, T, C, H, W], mask: [B, T] (optional) -> [B, out_dim]"""
        return self.forward(x, mask)
    
def create_model(**kwargs):
    return StyleModel(**kwargs)
