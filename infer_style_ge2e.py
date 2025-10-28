
"""
Paper-aligned inference for Go stylistic matching (centroid voting).
- Loads a GE2E-trained (or compatible) StyleModel.
- Builds per-player centroids by averaging multiple game embeddings.
- Matches query player centroids to candidate centroids via cosine similarity.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model_new import create_model
from train_style_model import GoGameSequenceDataset


def load_model(model_path: str, device: torch.device):
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    # Try both keys (new trainer saves under 'model', old under 'model_state_dict')
    state = ckpt.get('model', ckpt.get('model_state_dict', None))
    if state is None:
        raise ValueError("Checkpoint missing 'model' or 'model_state_dict' keys")
    # Pull optional model hyperparams
    args = ckpt.get('args', {})
    d_model = args.get('d_model', 1024)
    nhead = args.get('num_heads', 8)
    num_layers = args.get('num_transformer_layers', 12)
    out_dim = args.get('style_embed_dim', 512)

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
    print("✓ Model loaded")
    return model, out_dim


def fetch_one_game(data_loader, player_id: int, max_moves: int) -> np.ndarray:
    """Return [T,18,19,19] padded to max_moves (zeros)."""
    game_id = 0
    start = 0
    is_train = False  # inference time; still returns a random game if game_id=0 in this project
    flat = data_loader.get_random_feature_and_label(player_id, game_id, start, is_train)
    feat = 18 * 19 * 19
    T = len(flat) // feat
    if T == 0:
        arr = np.zeros((0, 18, 19, 19), dtype=np.float32)
    else:
        arr = np.asarray(flat, dtype=np.float32).reshape(T, 18, 19, 19)
    if T > max_moves:
        # sample evenly to avoid bias to openings
        idx = np.linspace(0, T - 1, max_moves, dtype=int)
        arr = arr[idx]
    elif T < max_moves:
        pad = np.zeros((max_moves - T, 18, 19, 19), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


@torch.no_grad()
def player_centroids(model, sgf_dir: str, config_file: str, device: torch.device,
                     max_moves: int = 32, games_per_player: int = 5,
                     ignore_first_k_moves: int = 0):
    """
    Build a centroid per player by averaging several game embeddings.
    Returns:
      centroids: FloatTensor [P, D] (L2-normalized)
      player_ids: list[int]
    """
    # init data loader the same way as training
    ds = GoGameSequenceDataset(sgf_dir, config_file, max_moves=max_moves, num_samples=1000)
    P = ds.num_players
    print(f"Found {P} players in {sgf_dir}")
    C_list = []
    ids = []
    for pid in tqdm(range(P), desc=f"Embedding players from {os.path.basename(sgf_dir)}"):
        embs = []
        for _ in range(games_per_player):
            arr = fetch_one_game(ds.data_loader, pid, max_moves)     # [T,18,19,19]
            mask = np.any(arr != 0, axis=(1, 2, 3))
            if ignore_first_k_moves > 0:
                mask[:ignore_first_k_moves] = False
            x = torch.from_numpy(arr).unsqueeze(0).to(device)        # [1,T,18,19,19]
            m = torch.from_numpy(mask.astype(np.bool_)).unsqueeze(0).to(device)  # [1,T]
            e = model.get_embedding(x, m)                             # [1,D], already L2-normalized
            embs.append(e)
        if len(embs) == 0:
            # fall back to zero (unlikely) — keep dimensionality
            D = next(model.parameters()).shape[0] if hasattr(model, "parameters") else 512
            C = torch.zeros(1, D, device=device)
        else:
            E = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)  # [1,D]
            C = F.normalize(E, dim=-1)
        C_list.append(C)
        ids.append(pid)
    C = torch.cat(C_list, dim=0)   # [P,D]
    return C, ids


def cosine_rank(query_C: torch.Tensor, cand_C: torch.Tensor, top_k: int = 5):
    """Return top-k indices and scores for each query against candidates."""
    sims = query_C @ cand_C.T   # [Q, P]
    vals, idx = torch.topk(sims, k=min(top_k, cand_C.shape[0]), dim=1)
    return idx.cpu().numpy(), vals.cpu().numpy()


def save_submission(matches, query_ids, cand_ids, out_csv: str):
    """Save id,label with 1-indexed IDs sorted lexicographically by id."""
    rows = []
    for qi, qpid in enumerate(query_ids):
        cid = cand_ids[matches[qi, 0]]
        rows.append([qpid + 1, cid + 1])
    rows = sorted(rows, key=lambda x: str(x[0]))
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(rows)
    print(f"✓ Wrote submission to {out_csv} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Paper-aligned inference: centroid voting for Go stylistic matching")
    ap.add_argument("--model_path", required=True, help="Path to checkpoint (best_ge2e.pt or compatible)")
    ap.add_argument("--config_file", default="conf.cfg")
    ap.add_argument("--query_dir", default="test_set/query_set")
    ap.add_argument("--candidate_dir", default="test_set/cand_set")
    ap.add_argument("--max_moves", type=int, default=32, help="Trim/pad per game to this many moves (paper uses 32)")
    ap.add_argument("--games_per_player", type=int, default=5, help="Games per player to average for centroid")
    ap.add_argument("--ignore_first_k_moves", type=int, default=0, help="Down-weight openings by ignoring first k moves")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="submission.csv")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.model_path, device)

    # Build centroids
    print("\n=== QUERY CENTROIDS ===")
    query_C, query_ids = player_centroids(
        model, args.query_dir, args.config_file, device,
        max_moves=args.max_moves, games_per_player=args.games_per_player,
        ignore_first_k_moves=args.ignore_first_k_moves,
    )
    print("\n=== CANDIDATE CENTROIDS ===")
    cand_C, cand_ids = player_centroids(
        model, args.candidate_dir, args.config_file, device,
        max_moves=args.max_moves, games_per_player=args.games_per_player,
        ignore_first_k_moves=args.ignore_first_k_moves,
    )

    # Rank
    print("\n=== MATCHING ===")
    matches, scores = cosine_rank(query_C, cand_C, top_k=args.top_k)

    # Show a few
    for i in range(min(5, len(query_ids))):
        print(f"Query {query_ids[i]} best -> Candidate {cand_ids[matches[i,0]]} (score={scores[i,0]:.4f})")

    # Save submission
    save_submission(matches, query_ids, cand_ids, args.output)


if __name__ == "__main__":
    main()
