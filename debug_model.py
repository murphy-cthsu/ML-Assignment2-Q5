#!/usr/bin/env python3
"""Quick diagnostic for model embeddings"""
import torch
import numpy as np
from style_model_new import create_model

# Load model
model_path = "models/paper_lite_subset/best_ge2e.pt"
device = torch.device("cuda")

print(f"Loading {model_path}...")
ckpt = torch.load(model_path, map_location=device)
state = ckpt.get('model', ckpt.get('model_state_dict'))
args = ckpt.get('args', {})

d_model = args.get('d_model', 256)
nhead = args.get('num_heads', 4)
num_layers = args.get('num_transformer_layers', 4)
out_dim = args.get('style_embed_dim', 256)

print(f"Model config: d_model={d_model}, heads={nhead}, layers={num_layers}, out_dim={out_dim}")

model = create_model(
    in_channels=18,
    move_embed_dim=320,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    out_dim=out_dim,
).to(device)
model.load_state_dict(state, strict=True)
model.eval()

# Test with random input
batch_size = 3
seq_len = 32
x = torch.randn(batch_size, seq_len, 18, 19, 19, device=device)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

with torch.no_grad():
    emb = model(x, mask)

print(f"\nEmbedding shape: {emb.shape}")
print(f"Embedding norms: {emb.norm(dim=1)}")
print(f"First embedding (first 10 dims): {emb[0, :10]}")
print(f"Second embedding (first 10 dims): {emb[1, :10]}")
print(f"Third embedding (first 10 dims): {emb[2, :10]}")

# Check if all embeddings are the same
print(f"\nPairwise cosine similarities:")
sims = emb @ emb.T
print(sims)

# Check variance
print(f"\nPer-dimension variance across batch: {emb.var(dim=0).mean():.6f}")
print(f"Expected for random normalized vectors: ~0.33")

if emb.var(dim=0).mean() < 0.01:
    print("\n⚠ WARNING: Very low variance - model might be outputting constant embeddings!")
else:
    print("\n✓ Model appears to generate diverse embeddings")
