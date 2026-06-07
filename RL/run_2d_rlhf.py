import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from td3_partial_obs import train_with_modification, plot_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardModel(nn.Module):
    """Scalar reward model r̂_φ(o, a) → ℝ."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))

class PreferenceDataset:
    """Stores (σ¹, σ²) segment pairs with binary preference labels."""
    def __init__(self):
        self.pairs: List[Tuple] = []

    def add(self, obs1, act1, obs2, act2, gt_return1: float, gt_return2: float):
        label = 1.0 if gt_return1 > gt_return2 else 0.0
        self.pairs.append((obs1, act1, obs2, act2, label))

    def sample(self, batch_size: int):
        idx = random.sample(range(len(self.pairs)), min(batch_size, len(self.pairs)))
        batch = [self.pairs[i] for i in idx]
        obs1 = torch.FloatTensor(np.stack([b[0] for b in batch])).to(device)
        act1 = torch.FloatTensor(np.stack([b[1] for b in batch])).to(device)
        obs2 = torch.FloatTensor(np.stack([b[2] for b in batch])).to(device)
        act2 = torch.FloatTensor(np.stack([b[3] for b in batch])).to(device)
        labels = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(device)
        return obs1, act1, obs2, act2, labels

def main(seeds, total_steps, save_dir="results_2d"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for at in ("mlp", "transformer"):
        window = 8 if at == "transformer" else 0
        for seed in seeds:
            print(f"\n--- Training {at.upper()} with Ground-Truth Reward (Seed {seed}) ---")
            r_gt = train_with_modification(
                modification="none", actor_type=at, window=window,
                total_steps=total_steps, seed=seed, save_dir=save_dir
            )
            r_gt["tag"] = r_gt["tag"].replace("full_obs", "gt_reward")
            
            print(f"\n--- Training {at.upper()} with RLHF Proxy Reward (Seed {seed}) ---")
            r_rlhf = train_with_modification(
                modification="rlhf", actor_type=at, window=window,
                total_steps=total_steps, seed=seed, save_dir=save_dir
            )
            results.extend([r_gt, r_rlhf])
            
    plot_comparison(results, title="RLHF vs Ground-Truth Reward: MLP vs Transformer",
                    filename="2d_rlhf_comparison.png", save_dir=save_dir)
    print(f"\n✓ Part 2(d) Complete. Results saved in {save_dir}/2d_rlhf_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    main(args.seeds, args.steps)