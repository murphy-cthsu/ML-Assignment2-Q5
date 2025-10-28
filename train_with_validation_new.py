
"""
GE2E training script with few-shot validation for Go playing style detection.
- Uses user's C++ backend (style_py) via GoGameSequenceDataset for data access.
- Trains with N×M batches (players × games per player) using GE2E.
- Evaluates with centroid voting (Top-1 / Top-5), optional k-move truncation.
"""

import os
import argparse
import numpy as np
import random
from typing import Tuple, List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# Import C++ bridge and dataset helpers from the user's project
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model_new import create_model
from train_style_model import GoGameSequenceDataset

# ----------------------------
# GE2E Loss
# ----------------------------
class GE2ELoss(nn.Module):
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float))

    @staticmethod
    def _cosine(a, b, eps=1e-6):
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return a @ b.T

    def forward(self, emb: torch.Tensor, N: int, M: int) -> torch.Tensor:
        """
        emb: [N*M, D] embeddings, grouped per player (players are contiguous)
        returns: scalar GE2E loss
        """
        D = emb.size(-1)
        assert emb.size(0) == N * M, "emb must be [N*M, D]"
        G = emb.view(N, M, D)                  # [N, M, D]
        centroids = G.mean(dim=1)              # [N, D]
        sum_per_player = G.sum(dim=1, keepdim=True)  # [N,1,D]
        leave_out = (sum_per_player - G) / max(M - 1, 1)  # [N,M,D]

        cos_all = self._cosine(G.reshape(N*M, D), centroids)        # [N*M, N]
        true_cos = self._cosine(G.reshape(N*M, D), leave_out.reshape(N*M, D)).diagonal().reshape(N*M, 1)
        S = cos_all.clone()
        correct_cols = torch.arange(N, device=emb.device).repeat_interleave(M)  # [N*M]
        S[torch.arange(N*M, device=emb.device), correct_cols] = true_cos.squeeze(1)
        S = self.w * S + self.b

        logsumexp = torch.logsumexp(S, dim=1)
        correct_scores = S[torch.arange(N*M, device=emb.device), correct_cols]
        loss = (logsumexp - correct_scores).mean()
        return loss

# ----------------------------
# Helpers: fetch one game for a specific player via style_py
# ----------------------------
def fetch_game_for_player(data_loader, player_id: int, max_moves: int) -> np.ndarray:
    """Returns an array of shape [T, 18, 19, 19], T<=max_moves, zero-padded to max_moves."""
    game_id = 0
    start = 0
    is_train = True
    flat = data_loader.get_random_feature_and_label(player_id, game_id, start, is_train)
    feature_size = 18 * 19 * 19
    T = len(flat) // feature_size
    if T == 0:
        arr = np.zeros((0, 18, 19, 19), dtype=np.float32)
    else:
        arr = np.asarray(flat, dtype=np.float32).reshape(T, 18, 19, 19)
    if T > max_moves:
        arr = arr[:max_moves]
    elif T < max_moves:
        pad = np.zeros((max_moves - T, 18, 19, 19), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr  # [max_moves, 18, 19, 19]

# ----------------------------
# GE2E N×M batch from dataset
# ----------------------------
def sample_ge2e_batch(dataset: GoGameSequenceDataset, N: int, M: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Returns (x, mask, N, M) suitable for StyleModel.forward.
       x: [N*M, T, C, H, W], mask: [N*M, T]"""
    players = random.sample(range(dataset.num_players), N)
    xs = []
    ms = []
    for pid in players:
        for _ in range(M):
            arr = fetch_game_for_player(dataset.data_loader, pid, dataset.max_moves)  # [T,18,19,19]
            T = arr.shape[0]
            mask = np.any(arr != 0, axis=(1,2,3))  # True where some non-zero plane exists
            xs.append(torch.from_numpy(arr).unsqueeze(0))   # [1,T,18,19,19]
            ms.append(torch.from_numpy(mask.astype(bool)).unsqueeze(0)) # [1,T]
    x = torch.cat(xs, dim=0).to(device)
    mask = torch.cat(ms, dim=0).to(device)
    return x.float(), mask.bool(), N, M

# ----------------------------
# Few-shot evaluation (centroid voting)
# ----------------------------
def fewshot_eval(model: nn.Module, dataset: GoGameSequenceDataset, device: torch.device,
                 players_subset: Optional[List[int]] = None,
                 ref_games_per_player: int = 5,
                 qry_games_per_player: int = 5,
                 ignore_first_k_moves: int = 0,
                 max_players: int = 50) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        all_players = list(range(dataset.num_players))
        if players_subset is not None:
            players = [p for p in players_subset if 0 <= p < dataset.num_players]
        else:
            players = all_players
        if len(players) > max_players:
            players = random.sample(players, max_players)

        centroids = []
        centroid_pids = []
        qry_embeddings = []
        qry_labels = []

        # Progress bar for validation
        for pid in tqdm(players, desc="Validating", ncols=80, leave=False):
            # Refs
            ref_embs = []
            for _ in range(ref_games_per_player):
                arr = fetch_game_for_player(dataset.data_loader, pid, dataset.max_moves)  # [T,18,19,19]
                mask = np.any(arr != 0, axis=(1,2,3))
                if ignore_first_k_moves > 0:
                    mask[:ignore_first_k_moves] = False
                x = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1,T,18,19,19]
                m = torch.from_numpy(mask.astype(bool)).unsqueeze(0).to(device)
                e = model.get_embedding(x, m)  # [1,D]
                ref_embs.append(e)
            if len(ref_embs) == 0:
                continue
            ref = torch.cat(ref_embs, dim=0).mean(dim=0, keepdim=True)  # [1,D]
            ref = torch.nn.functional.normalize(ref, dim=-1)
            centroids.append(ref)
            centroid_pids.append(pid)

            # Queries
            for _ in range(qry_games_per_player):
                arr = fetch_game_for_player(dataset.data_loader, pid, dataset.max_moves)
                mask = np.any(arr != 0, axis=(1,2,3))
                if ignore_first_k_moves > 0:
                    mask[:ignore_first_k_moves] = False
                x = torch.from_numpy(arr).unsqueeze(0).to(device)
                m = torch.from_numpy(mask.astype(bool)).unsqueeze(0).to(device)
                e = model.get_embedding(x, m)  # [1,D]
                qry_embeddings.append(e)
                qry_labels.append(pid)

        if len(qry_embeddings) == 0 or len(centroids) == 0:
            return 0.0, 0.0

        C = torch.cat(centroids, dim=0)        # [P,D]
        Q = torch.cat(qry_embeddings, dim=0)   # [Q,D]
        sims = Q @ C.T                         # cosine since embeddings are normalized

        top1 = 0
        top5 = 0
        for i, pid in enumerate(qry_labels):
            ranks = torch.argsort(sims[i], descending=True)
            pred = [centroid_pids[j] for j in ranks.tolist()]
            if pred[0] == pid:
                top1 += 1
            if pid in pred[:5]:
                top5 += 1
        n = len(qry_labels)
        return top1 / n, top5 / n

# ----------------------------
# Training
# ----------------------------
def train_with_validation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset built over C++ backend
    base_dataset = GoGameSequenceDataset(
        args.train_dir,
        args.config_file,
        max_moves=args.max_moves,
        num_samples=args.num_samples
    )

    # Model
    model = create_model(
        in_channels=18,
        move_embed_dim=320,
        d_model=args.d_model,
        nhead=args.num_heads,
        num_layers=args.num_transformer_layers,
        out_dim=args.style_embed_dim,
        max_len=args.max_moves,
    ).to(device)

    # Optimizer
    if args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.5)
        print("Optimizer: SGD(lr=0.01, momentum=0.9) w/ StepLR(40k, 0.5)")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
        print(f"Optimizer: Adam(lr={args.lr})")

    ge2e = GE2ELoss().to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        
        # Progress bar for training steps
        pbar = tqdm(range(1, args.steps_per_epoch + 1), 
                    desc=f"Epoch {epoch}/{args.epochs}",
                    ncols=100)
        
        for step in pbar:
            x, mask, N, M = sample_ge2e_batch(base_dataset, args.N, args.M, device)
            emb = model(x, mask)                    # [N*M, D]
            loss = ge2e(emb, N, M)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            
            # Update progress bar
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{running/10:.4f}'})
                running = 0.0

        if scheduler is not None:
            scheduler.step()

        # Few-shot validation each epoch
        top1, top5 = fewshot_eval(
            model, base_dataset, device,
            ref_games_per_player=args.eval_ref,
            qry_games_per_player=args.eval_qry,
            ignore_first_k_moves=args.ignore_first_k_moves_eval,
            max_players=args.eval_max_players
        )
        print(f"[Eval] Epoch {epoch}: Top1={top1:.3f}, Top5={top5:.3f}")

        # Save best
        if top1 > best_top1:
            best_top1 = top1
            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch,
                'top1': top1,
                'top5': top5,
                'args': vars(args)
            }
            path = os.path.join(args.output_dir, 'best_ge2e.pt')
            torch.save(ckpt, path)
            print(f"✓ Saved new best checkpoint: {path} (Top1={top1:.3f})")

    # Save last
    torch.save({'model': model.state_dict(), 'top1': best_top1}, os.path.join(args.output_dir, 'last_ge2e.pt'))
    print("Training complete.")

def build_argparser():
    p = argparse.ArgumentParser(description="GE2E training for Go stylistic embeddings")
    # Data
    p.add_argument('--train_dir', type=str, default='train_set')
    p.add_argument('--config_file', type=str, default='conf.cfg')
    p.add_argument('--max_moves', type=int, default=32)        # paper-aligned
    p.add_argument('--num_samples', type=int, default=10000)
    # Model
    p.add_argument('--d_model', type=int, default=1024)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_transformer_layers', type=int, default=12)
    p.add_argument('--style_embed_dim', type=int, default=512)
    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps_per_epoch', type=int, default=200)
    p.add_argument('--N', type=int, default=40)   # players per batch
    p.add_argument('--M', type=int, default=20)   # games per player
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--use_sgd', action='store_true', help='Use SGD(0.01,0.9) + step LR (paper)')
    p.add_argument('--output_dir', type=str, default='models_ge2e')
    # Eval
    p.add_argument('--eval_ref', type=int, default=5)
    p.add_argument('--eval_qry', type=int, default=5)
    p.add_argument('--ignore_first_k_moves_eval', type=int, default=0)
    p.add_argument('--eval_max_players', type=int, default=50)
    return p

if __name__ == '__main__':
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_with_validation(args)
