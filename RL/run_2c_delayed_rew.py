import os
import argparse
import gymnasium as gym

from td3_causal_transformer import make_env
from td3_partial_obs import train_with_modification, plot_comparison

class DelayedRewardWrapper(gym.Wrapper):
    """(c) Accumulates rewards internally; only releases the sum every K steps."""
    def __init__(self, env: gym.Env, K: int = 10):
        super().__init__(env)
        self.K = K
        self._step_count = 0
        self._reward_accum = 0.0

    def reset(self, **kwargs):
        self._step_count = 0
        self._reward_accum = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._reward_accum += reward
        
        if self._step_count % self.K == 0:
            delayed = self._reward_accum
            self._reward_accum = 0.0
        else:
            delayed = 0.0
            
        return obs, delayed, terminated, truncated, info

def main(Ks, seeds, total_steps, save_dir="results_2c"):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for K in Ks:
        for at in ("mlp", "transformer"):
            window = 8 if at == "transformer" else 0
            for seed in seeds:
                print(f"\n--- Training {at.upper()} with Delayed Reward K={K} (Seed {seed}) ---")
                res = train_with_modification(
                    modification="delayed_rew", actor_type=at, window=window,
                    K=K, total_steps=total_steps, seed=seed, save_dir=save_dir
                )
                results.append(res)
                
    plot_comparison(results, title="Delayed Rewards K∈{10,30}: MLP vs Transformer",
                    filename="2c_delayed_rew_comparison.png", save_dir=save_dir)
    print(f"\n✓ Part 2(c) Complete. Results saved in {save_dir}/2c_delayed_rew_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--Ks", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    main(args.Ks, args.seeds, args.steps)