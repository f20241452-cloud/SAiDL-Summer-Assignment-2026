import os
import argparse
import gymnasium as gym
import numpy as np
import torch

from td3_causal_transformer import TD3Config, TD3Agent, RunningNormaliser, make_env, evaluate
from td3_partial_obs import train_with_modification, plot_comparison

class ObservationNoiseWrapper(gym.ObservationWrapper):
    """(b) Add i.i.d. Gaussian noise to every observation."""
    def __init__(self, env: gym.Env, sigma: float = 0.1):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.sigma, size=obs.shape)
        return (obs + noise).astype(np.float32)

def main(sigmas, seeds, total_steps, save_dir="results_2b"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for sigma in sigmas:
        for at in ("mlp", "transformer"):
            window = 8 if at == "transformer" else 0
            for seed in seeds:
                print(f"\n--- Training {at.upper()} with Noise σ={sigma} (Seed {seed}) ---")
                res = train_with_modification(
                    modification="obs_noise", actor_type=at, window=window,
                    sigma=sigma, total_steps=total_steps, seed=seed, save_dir=save_dir
                )
                results.append(res)
                
    plot_comparison(results, title="Observation Noise σ∈{0.1,0.3}: MLP vs Transformer",
                    filename="2b_obs_noise_comparison.png", save_dir=save_dir)
    print(f"\n✓ Part 2(b) Complete. Results saved in {save_dir}/2b_obs_noise_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    main(args.sigmas, args.seeds, args.steps)