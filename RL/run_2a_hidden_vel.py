import os
import pickle
import argparse
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# Assume td3_causal_transformer.py provides these
from td3_causal_transformer import TD3Config, TD3Agent, RunningNormaliser, make_env, evaluate
from td3_partial_obs import train_with_modification, plot_comparison

HOPPER_VEL_INDICES = list(range(5, 11))

class HiddenVelocityWrapper(gym.ObservationWrapper):
    """
    (a) Remove all velocity components.
    Velocity indices are set to 0.0; the obs space shape is UNCHANGED.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[HOPPER_VEL_INDICES] = 0.0
        return obs

def make_hidden_vel_env(render=False):
    return HiddenVelocityWrapper(make_env(render=render))

def main(seeds, total_steps, save_dir="results_2a"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for at in ("mlp", "transformer"):
        window = 8 if at == "transformer" else 0
        for seed in seeds:
            print(f"\n--- Training {at.upper()} with Hidden Velocities (Seed {seed}) ---")
            # Using the modular train loop adapted for this specific env
            res = train_with_modification(
                modification="hidden_vel", actor_type=at, window=window,
                total_steps=total_steps, seed=seed, save_dir=save_dir
            )
            results.append(res)
            
    plot_comparison(results, title="Hidden Velocities: MLP-TD3 vs Transformer-TD3",
                    filename="2a_hidden_vel_comparison.png", save_dir=save_dir)
    print(f"\n✓ Part 2(a) Complete. Results saved in {save_dir}/2a_hidden_vel_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    main(args.seeds, args.steps)