"""
TD3 with Causal Transformer Actor on Hopper-v5
Variant: Causal Transformer actor (window L ∈ {4, 8, 16, 32}) + MLP critic

Preferred config: 2 layers, 4 heads, embed dim 128, L=8, pre-LN,
running mean/std normalisation; TD3: lr 3e-4, buffer 1e6, batch 256,
tau=0.005, policy delay 2.
"""

import os
import random
import time
import pickle
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════
# RUNNING NORMALISER  (mean/std over a running window)
# ═══════════════════════════════════════════════════════════════════
class RunningNormaliser:
    def __init__(self, shape: int, clip: float = 5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: np.ndarray):
        batch_mean = x.mean(0)
        batch_var  = x.var(0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean  + delta * batch_count / total
        self.var   = (self.var * self.count + batch_var * batch_count
                      + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -self.clip, self.clip
        )

    def normalise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.FloatTensor(self.mean).to(x.device)
        std  = torch.FloatTensor(np.sqrt(self.var) + 1e-8).to(x.device)
        return torch.clamp((x - mean) / std, -self.clip, self.clip)


# ═══════════════════════════════════════════════════════════════════
# WINDOW REPLAY BUFFER  (for Transformer actor — stores trajectories)
# ═══════════════════════════════════════════════════════════════════
class WindowReplayBuffer:
    """
    Efficient numpy-backed replay buffer supporting window sampling for the
    Transformer actor. Each sample returns the last L obs-act pairs
    ending at the sampled timestep (zero-padded at episode boundaries).

    We track episode ids so we do not cross episode boundaries.
    """

    def __init__(self, obs_dim: int, act_dim: int, window: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.window   = window
        self.act_dim  = act_dim
        self.obs_dim  = obs_dim

        self.obs      = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.acts     = np.zeros((max_size, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((max_size, 1),        dtype=np.float32)
        self.dones    = np.zeros((max_size, 1),        dtype=np.float32)
        self.ep_ids   = np.zeros(max_size,             dtype=np.int64)
        self.ep_count = 0

    def add(self, obs, act, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.acts[self.ptr]     = act
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ep_ids[self.ptr]   = self.ep_count
        
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if done:
            self.ep_count += 1

    def sample_windows(self, batch_size: int):
            """
            Returns:
                obs_win:      (B, L, obs_dim) ending at t
                act_win:      (B, L, act_dim) ending at t-1 (prevents action leakage)
                next_obs_win: (B, L, obs_dim) ending at t+1 (for target critic)
                next_act_win: (B, L, act_dim) ending at t   (for target critic)
                base:         standard (obs, act, rew, next_obs, done) tuple
            """
            idx = np.random.randint(0, self.size, size=batch_size)

            obs_win = np.zeros((batch_size, self.window, self.obs_dim), dtype=np.float32)
            act_win = np.zeros((batch_size, self.window, self.act_dim), dtype=np.float32)
            next_obs_win = np.zeros((batch_size, self.window, self.obs_dim), dtype=np.float32)
            next_act_win = np.zeros((batch_size, self.window, self.act_dim), dtype=np.float32)

            for b, i in enumerate(idx):
                ep = self.ep_ids[i]
                
                for w in range(self.window):
                    # j is the index of the current window step
                    j = i - (self.window - 1 - w)
                    j_mod = j % self.max_size
                    
                    # 1. Ensure we don't read before the buffer started filling
                    # 2. Ensure the step belongs to the same episode
                    valid_j = (not (j < 0 and self.size < self.max_size)) and (self.ep_ids[j_mod] == ep)
                    
                    if valid_j:
                        obs_win[b, w] = self.obs[j_mod]
                        next_act_win[b, w] = self.acts[j_mod]
                        
                        if w == self.window - 1:
                            # At the last window step, 'next_obs' is explicitly the stored next_obs
                            next_obs_win[b, w] = self.next_obs[i]
                        else:
                            # Otherwise, 'next_obs' is just the observation at j+1
                            next_obs_win[b, w] = self.obs[(j + 1) % self.max_size]

                    # Shift action back by 1 to prevent data leakage
                    j_prev = j - 1
                    j_prev_mod = j_prev % self.max_size
                    valid_prev = (not (j_prev < 0 and self.size < self.max_size)) and (self.ep_ids[j_prev_mod] == ep)
                    
                    if valid_prev:
                        act_win[b, w] = self.acts[j_prev_mod]

            # Convert to device tensors
            obs_win = torch.FloatTensor(obs_win).to(device)
            act_win = torch.FloatTensor(act_win).to(device)
            next_obs_win = torch.FloatTensor(next_obs_win).to(device)
            next_act_win = torch.FloatTensor(next_act_win).to(device)

            base = (
                torch.FloatTensor(self.obs[idx]).to(device),
                torch.FloatTensor(self.acts[idx]).to(device),
                torch.FloatTensor(self.rewards[idx]).to(device),
                torch.FloatTensor(self.next_obs[idx]).to(device),
                torch.FloatTensor(self.dones[idx]).to(device),
            )
            
            return obs_win, act_win, next_obs_win, next_act_win, base

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════
# CAUSAL TRANSFORMER ACTOR
# ═══════════════════════════════════════════════════════════════════
class CausalTransformerActor(nn.Module):
    """
    Takes a sliding window of (obs, act) pairs → outputs a_t.

    Architecture (preferred config):
        - 2 transformer layers, 4 heads, embed dim 128
        - Pre-LayerNorm  (LayerNorm before attention / FFN)
        - Causal mask (each position can only attend to itself + past)
        - Input: concat(obs_t, act_{t-1}) projected to embed_dim
          The last token's output → action head
    """

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        action_limit: float,
        window: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window       = window
        self.action_limit = action_limit
        self.embed_dim    = embed_dim

        # Input projection: obs + prev_act → embed
        self.input_proj = nn.Linear(obs_size + act_size, embed_dim)

        # Learnable positional embedding
        self.pos_embed = nn.Embedding(window, embed_dim)

        # Transformer encoder layers with pre-LN
        self.layers = nn.ModuleList([
            _PreLNTransformerLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Action head
        self.action_head = nn.Linear(embed_dim, act_size)

        # Causal mask (registered as buffer so it moves with .to(device))
        causal_mask = torch.triu(
            torch.ones(window, window, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        obs_win: torch.Tensor,   # (B, L, obs_dim)
        act_win: torch.Tensor,   # (B, L, act_dim)
    ) -> torch.Tensor:           # (B, act_dim)

        B, L, _ = obs_win.shape
        x = torch.cat([obs_win, act_win], dim=-1)   # (B, L, obs+act)
        x = self.input_proj(x)                       # (B, L, E)

        pos = torch.arange(L, device=x.device)
        x   = x + self.pos_embed(pos).unsqueeze(0)  # broadcast over batch

        for layer in self.layers:
            x = layer(x, self.causal_mask)

        x = self.norm(x)                               # (B, L, E)
        last = x[:, -1, :]                            # (B, E)  last token
        return self.action_limit * torch.tanh(self.action_head(last))


class _PreLNTransformerLayer(nn.Module):
    """Single pre-LayerNorm transformer encoder layer."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=True)
        x = x + self.drop(attn_out)
        # Pre-LN FFN
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x


# ═══════════════════════════════════════════════════════════════════
# TWIN CRITIC
# ═══════════════════════════════════════════════════════════════════
class Critic(nn.Module):

    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        inp = obs_size + act_size

        self.q1 = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ):
        sa = torch.cat([obs, act], dim=1)
        return self.q1(sa), self.q2(sa)


# ═══════════════════════════════════════════════════════════════════
# TD3 AGENT  (Task 2: Causal Transformer Actor Variant)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TD3Config:
    gamma:        float = 0.99
    tau:          float = 0.005
    actor_lr:     float = 3e-4
    critic_lr:    float = 3e-4
    batch_size:   int   = 256
    buffer_size:  int   = 1_000_000
    policy_delay: int   = 2
    noise_std:    float = 0.2       # target policy smoothing noise
    noise_clip:   float = 0.5
    expl_noise:   float = 0.1       # exploration noise


class TD3Agent:

    def __init__(
        self,
        obs_size:     int,
        act_size:     int,
        action_limit: float,
        cfg:          TD3Config = TD3Config(),
        window:       int       = 8,
        normaliser:   Optional[RunningNormaliser] = None,
    ):
        self.cfg          = cfg
        self.act_size     = act_size
        self.action_limit = action_limit
        self.window       = window
        self.normaliser   = normaliser
        self.update_count = 0

        # ── Actor ──────────────────────────────────────────────────
        self.actor = CausalTransformerActor(
            obs_size, act_size, action_limit, window=window
        ).to(device)
        self.actor_target = CausalTransformerActor(
            obs_size, act_size, action_limit, window=window
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # ── Critic ─────────────────────────────────────────────────
        self.critic        = Critic(obs_size, act_size).to(device)
        self.critic_target = Critic(obs_size, act_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ─────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # ── Replay buffer ──────────────────────────────────────────
        self.buffer = WindowReplayBuffer(
            obs_size, act_size, window, max_size=cfg.buffer_size
        )

        # ── Context window for online inference ────────────────────
        self._obs_ctx = deque(maxlen=window)
        self._act_ctx = deque(maxlen=window)

    def reset_context(self, obs_size: int, act_size: int):
        """Call at episode start to clear the context window."""
        self._obs_ctx.clear()
        self._act_ctx.clear()

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        if self.normaliser is not None:
            obs = self.normaliser.normalise(obs)

        # Build context window
        self._obs_ctx.append(obs)
        if len(self._act_ctx) == 0:
            prev_act = np.zeros(self.act_size, dtype=np.float32)
        else:
            prev_act = self._act_ctx[-1]
        self._act_ctx.append(prev_act)   # placeholder; updated after

        L = self.window
        obs_win = np.zeros((1, L, obs.shape[0]), dtype=np.float32)
        act_win = np.zeros((1, L, self.act_size), dtype=np.float32)
        for i, (o, a) in enumerate(zip(self._obs_ctx, self._act_ctx)):
            obs_win[0, i] = o
            act_win[0, i] = a

        t_obs = torch.FloatTensor(obs_win).to(device)
        t_act = torch.FloatTensor(act_win).to(device)
        with torch.no_grad():
            act = self.actor(t_obs, t_act).cpu().numpy().reshape(-1)

        # Update the last slot with the chosen action
        self._act_ctx[-1] = act

        if explore:
            noise = np.random.normal(
                0, self.cfg.expl_noise * self.action_limit, size=act.shape
            )
            act = np.clip(act + noise, -self.action_limit, self.action_limit)

        return act

    def store(self, obs, act, reward, next_obs, done):
        if self.normaliser is not None:
            obs      = self.normaliser.normalise(obs)
            next_obs = self.normaliser.normalise(next_obs)
        self.buffer.add(obs, act, reward, next_obs, done)

    def train(self):
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return

        # 1. Unpack the newly structured windows
        obs_win, act_win, next_obs_win, next_act_win, (obs, acts, rewards, next_obs, dones) = \
            self.buffer.sample_windows(cfg.batch_size)

        # ── Critic update ──────────────────────────────────────────
        with torch.no_grad():
            # 2. Pass the correct chronological next-windows to the target actor
            next_act = self.actor_target(next_obs_win, next_act_win)

            noise = (
                torch.randn_like(next_act) * cfg.noise_std * self.action_limit
            ).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_act = (next_act + noise).clamp(-self.action_limit, self.action_limit)

            q1_t, q2_t = self.critic_target(next_obs, next_act)
            target_q = rewards + (1 - dones) * cfg.gamma * torch.min(q1_t, q2_t)
            

        q1, q2 = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Delayed actor update ───────────────────────────────────
        if self.update_count % cfg.policy_delay == 0:
            actor_act = self.actor(obs_win, act_win)
            actor_loss = -self.critic(obs, actor_act)[0].mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.update_count += 1

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        tau = self.cfg.tau
        for s, t in zip(src.parameters(), tgt.parameters()):
            t.data.copy_(tau * s.data + (1 - tau) * t.data)

    def save(self, path: str):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
def make_env(render: bool = False):
    return gym.make(
        "Hopper-v5",
        render_mode="human" if render else None,
        forward_reward_weight=10.0,
        ctrl_cost_weight=0.0069,
        healthy_reward=23.0,
        terminate_when_unhealthy=False,
    )


def train_agent(
    window:      int   = 8,
    total_steps: int   = 1_000_000,
    seed:        int   = 0,
    eval_every:  int   = 5_000,
    eval_eps:    int   = 5,
    save_dir:    str   = "checkpoints",
    render:      bool  = False,
) -> dict:

    os.makedirs(save_dir, exist_ok=True)

    # ── Seeding ────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env      = make_env(render=render)
    eval_env = make_env(render=False)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)

    obs_size     = env.observation_space.shape[0]
    act_size     = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    cfg = TD3Config(
        batch_size=256,
        buffer_size=1_000_000,
        tau=0.005,
        policy_delay=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )

    normaliser = RunningNormaliser(obs_size)
    agent = TD3Agent(
        obs_size, act_size, action_limit,
        cfg=cfg,
        window=window,
        normaliser=normaliser,
    )

    tag = f"transformer_L{window}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  Training: {tag}  |  Steps: {total_steps:,}")
    print(f"{'='*60}")

    rewards_log  = []
    eval_log     = []   # [(step, mean_return)]
    step         = 0
    ep_reward    = 0.0
    ep_num       = 0

    obs, _ = env.reset()
    agent.reset_context(obs_size, act_size)

    pbar = tqdm(total=total_steps, desc=tag, unit="step")

    while step < total_steps:
        if step < 10_000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, explore=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        normaliser.update(obs.reshape(1, -1))

        agent.store(obs, action, reward, next_obs, done)
        agent.train()

        obs        = next_obs
        ep_reward += reward
        step      += 1
        pbar.update(1)

        if done:
            rewards_log.append(ep_reward)
            ep_reward = 0.0
            ep_num   += 1
            obs, _   = env.reset()
            agent.reset_context(obs_size, act_size)

        # ── Evaluation ────────────────────────────────────────────
        if step % eval_every == 0:
            mean_ret = evaluate(agent, eval_env, obs_size, act_size, eval_eps)
            eval_log.append((step, mean_ret))
            pbar.set_postfix({"eval_ret": f"{mean_ret:.1f}", "ep": ep_num})
            agent.save(os.path.join(save_dir, f"{tag}_step{step}.pt"))

    pbar.close()
    env.close()
    eval_env.close()

    result = {
        "tag":        tag,
        "actor_type": "transformer",
        "window":     window,
        "seed":       seed,
        "eval_log":   eval_log,
        "ep_rewards": rewards_log,
    }
    with open(os.path.join(save_dir, f"{tag}_log.pkl"), "wb") as f:
        pickle.dump(result, f)

    print(f"\n✓ Finished {tag}. Final eval: {eval_log[-1][1]:.1f}")
    return result


def evaluate(
    agent:    TD3Agent,
    env:      gym.Env,
    obs_size: int,
    act_size: int,
    n_eps:    int = 5,
) -> float:
    returns = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        agent.reset_context(obs_size, act_size)
        ep_ret = 0.0
        done   = False
        while not done:
            act  = agent.select_action(obs, explore=False)
            obs, r, term, trunc, _ = env.step(act)
            ep_ret += r
            done    = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


# ═══════════════════════════════════════════════════════════════════
# SWEEP  (3 seeds × {L=4,8,16,32})
# ═══════════════════════════════════════════════════════════════════
def run_sweep(
    seeds:       List[int] = [0, 1, 2],
    windows:     List[int] = [4, 8, 16, 32],
    total_steps: int       = 1_000_000,
    save_dir:    str       = "results",
):
    all_results = []

    for L in windows:
        for seed in seeds:
            r = train_agent(
                window=L,
                total_steps=total_steps,
                seed=seed,
                save_dir=save_dir,
            )
            all_results.append(r)

    plot_results(all_results, save_dir)
    return all_results


def plot_results(results: list, save_dir: str = "results"):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    groups: dict = {}
    for r in results:
        key = f"Transformer L={r['window']}"
        groups.setdefault(key, []).append(r["eval_log"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10.colors

    for idx, (label, logs_list) in enumerate(sorted(groups.items())):
        all_steps   = [x for x, _ in logs_list[0]]
        all_returns = []
        for log in logs_list:
            all_returns.append([ret for _, ret in log])

        arr  = np.array(all_returns)
        mean = arr.mean(0)
        std  = arr.std(0)

        c = colours[idx % len(colours)]
        ax.plot(all_steps, mean, label=label, color=c, linewidth=2)
        ax.fill_between(all_steps, mean - std, mean + std, alpha=0.2, color=c)

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("Average Return (5 eval episodes)", fontsize=13)
    ax.set_title("TD3 Causal Transformer Actor Performance on Hopper-v5", fontsize=14)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "sweep_results.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved to {path}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["transformer", "sweep"],
        default="sweep",
        help="Run mode (transformer or sweep)",
    )
    parser.add_argument("--window",   type=int,   default=8,   help="Context window L")
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--steps",    type=int,   default=1_000_000)
    parser.add_argument("--save_dir", type=str,   default="results")
    parser.add_argument("--render",   action="store_true")
    args = parser.parse_args()

    if args.mode == "transformer":
        train_agent(
            window=args.window,
            total_steps=args.steps,
            seed=args.seed,
            save_dir=args.save_dir,
            render=args.render,
        )
    else:  # sweep
        run_sweep(
            seeds=[0, 1, 2],
            windows=[4, 8, 16, 32],
            total_steps=args.steps,
            save_dir=args.save_dir,
        )