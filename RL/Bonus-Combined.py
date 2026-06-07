"""
Unified RL Training Pipeline for Transformer-TD3 and Variants.
Includes: MLP Baseline, Causal Transformer (with Positional Encoding Ablation),
xLSTM Backbone, POMDP Wrappers, and Algorithm Distillation.
"""

import os
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
from typing import Optional, Tuple, Dict
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# 1. ENVIRONMENT WRAPPERS (POMDP & BONUS B)
# ============================================================================

class HiddenVelocityWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pos_dim = 5 # Hopper-v5 specific: first 5 are positions

    def observation(self, obs):
        masked_obs = obs.copy()
        masked_obs[self.pos_dim:] = 0.0
        return masked_obs

class GaussianNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, sigma=0.1):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs):
        noise = np.random.normal(0, self.sigma, size=obs.shape)
        return (obs + noise).astype(np.float32)

class DelayedRewardWrapper(gym.Wrapper):
    def __init__(self, env, k=10):
        super().__init__(env)
        self.k = k
        self.step_count = 0
        self.acc_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.acc_reward += reward
        
        if self.step_count % self.k == 0 or terminated or truncated:
            yield_reward = self.acc_reward
            self.acc_reward = 0.0
        else:
            yield_reward = 0.0
            
        return obs, yield_reward, terminated, truncated, info

def make_env(task_type="fully_observable"):
    env = gym.make("Hopper-v5")
    if task_type in ["hidden_vel", "combined"]:
        env = HiddenVelocityWrapper(env)
    if task_type == "combined":
        env = GaussianNoiseWrapper(env, sigma=0.1)
        env = DelayedRewardWrapper(env, k=10)
    return env

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
        self.mean = self.mean + delta * batch_count / total
        self.var  = (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)

# ============================================================================
# 2. REPLAY BUFFERS (FIXED CAUSALITY)
# ============================================================================

class WindowReplayBuffer:
    def __init__(self, obs_dim, act_dim, window, max_size=1_000_000):
        self.max_size, self.window = max_size, window
        self.ptr, self.size, self.ep_count = 0, 0, 0
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.ep_ids = np.zeros(max_size, dtype=np.int64)

    def add(self, obs, act, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ep_ids[self.ptr] = self.ep_count
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if done: self.ep_count += 1

    def sample_windows(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs_win = np.zeros((batch_size, self.window, self.obs_dim), dtype=np.float32)
        act_win = np.zeros((batch_size, self.window, self.act_dim), dtype=np.float32)
        n_obs_win = np.zeros((batch_size, self.window, self.obs_dim), dtype=np.float32)
        n_act_win = np.zeros((batch_size, self.window, self.act_dim), dtype=np.float32)

        for b, i in enumerate(idx):
            ep = self.ep_ids[i]
            for w in range(self.window):
                j = i - (self.window - 1 - w)
                j_mod = j % self.max_size
                
                valid_j = (not (j < 0 and self.size < self.max_size)) and (self.ep_ids[j_mod] == ep)
                if valid_j:
                    obs_win[b, w] = self.obs[j_mod]
                    n_act_win[b, w] = self.acts[j_mod]
                    n_obs_win[b, w] = self.next_obs[i] if w == self.window - 1 else self.obs[(j + 1) % self.max_size]

                # Prevent Action Leakage: Shift actor context back by 1
                j_prev = (j - 1) % self.max_size
                valid_prev = (not (j - 1 < 0 and self.size < self.max_size)) and (self.ep_ids[j_prev] == ep)
                if valid_prev:
                    act_win[b, w] = self.acts[j_prev]

        base = (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.acts[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )
        return (torch.FloatTensor(obs_win).to(device), torch.FloatTensor(act_win).to(device),
                torch.FloatTensor(n_obs_win).to(device), torch.FloatTensor(n_act_win).to(device), base)

# ============================================================================
# 3. ARCHITECTURES
# ============================================================================


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_limit):
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )
    def forward(self, obs_win, act_win=None):
        return self.action_limit * self.net(obs_win[:, -1, :])

class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_size + act_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_size + act_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=1)
        return self.q1(sa), self.q2(sa)

# --- RoPE Helpers ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # Broadcast over B and Heads
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CustomAttentionBlock(nn.Module):
    """Custom Transformer Block to support RoPE inside Q/K projections."""
    def __init__(self, embed_dim, nhead, use_rope=False):
        super().__init__()
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, mask, freqs_cis=None):
        B, L, D = x.shape
        norm_x = self.norm1(x)
        
        qkv = self.qkv(norm_x).reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, L, Heads, Head_Dim)
        
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # (B, Heads, L, Head_Dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores + mask.unsqueeze(0).unsqueeze(0) # Apply causal mask
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x

class CausalTransformerActor(nn.Module):
    def __init__(self, obs_size, act_size, action_limit, window=8, embed_dim=128, pe_type="learned"):
        super().__init__()
        self.action_limit, self.window = action_limit, window
        self.pe_type = pe_type
        self.input_proj = nn.Linear(obs_size + act_size, embed_dim)
        
        # Positional Encodings
        self.freqs_cis = None
        if pe_type == "learned":
            self.pos_embed = nn.Embedding(window, embed_dim)
        elif pe_type == "sinusoidal":
            pe = torch.zeros(window, embed_dim)
            position = torch.arange(0, window, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_embed', pe.unsqueeze(0))
        elif pe_type == "rope":
            self.freqs_cis = precompute_freqs_cis(embed_dim // 4, window).to(device) # embed_dim // nhead
            
        use_rope = (pe_type == "rope")
        self.layers = nn.ModuleList([CustomAttentionBlock(embed_dim, nhead=4, use_rope=use_rope) for _ in range(2)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.action_head = nn.Linear(embed_dim, act_size)
        
        # Causal mask: -inf for future tokens, 0 for past/current
        mask = torch.triu(torch.ones(window, window) * float('-inf'), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, obs_win, act_win):
        x = self.input_proj(torch.cat([obs_win, act_win], dim=-1))
        
        if self.pe_type == "learned":
            x = x + self.pos_embed(torch.arange(self.window, device=x.device)).unsqueeze(0)
        elif self.pe_type == "sinusoidal":
            x = x + self.pos_embed
            
        for layer in self.layers:
            x = layer(x, self.causal_mask, self.freqs_cis)
            
        last = self.norm(x[:, -1, :])
        return self.action_limit * torch.tanh(self.action_head(last))

# ============================================================================
# 4. BONUS D: xLSTM BACKBONE
# ============================================================================

class AdaptiveLSTMCell(nn.Module): # sLSTM
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.W_i = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_f = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_c = nn.Linear(in_dim + hid_dim, hid_dim)
        self.W_o = nn.Linear(in_dim + hid_dim, hid_dim)

    def forward(self, x, h, c, m):
        xh = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.W_i(xh))
        f = torch.sigmoid(self.W_f(xh))
        c_tilde = torch.tanh(self.W_c(xh))
        o = torch.sigmoid(self.W_o(xh))
        
        # Exponential scalar gating (numerically stable)
        log_f, log_i = torch.log(f + 1e-8), torch.log(i + 1e-8)
        m_new = torch.max(log_f + m, log_i)
        
        c_new = torch.exp(log_f + m - m_new) * c + torch.exp(log_i - m_new) * c_tilde
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, m_new

class mLSTMCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.W_q = nn.Linear(in_dim, hid_dim)
        self.W_k = nn.Linear(in_dim, hid_dim)
        self.W_v = nn.Linear(in_dim, hid_dim)
        self.W_i = nn.Linear(in_dim, 1)
        self.W_f = nn.Linear(in_dim, 1)
        self.W_o = nn.Linear(in_dim, hid_dim)

    def forward(self, x, C, n):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        i, f, o = torch.exp(self.W_i(x)), torch.sigmoid(self.W_f(x)), torch.sigmoid(self.W_o(x))
        
        # Matrix memory covariance update
        v_k_T = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        C_new = f.unsqueeze(-1) * C + i.unsqueeze(-1) * v_k_T
        n_new = f * n + i * k
        
        h_tilde = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2) / (torch.sum(n_new * q, dim=1, keepdim=True) + 1e-8)
        h_new = o * h_tilde
        return h_new, C_new, n_new

class xLSTMActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_limit, window, hid_dim=128):
        super().__init__()
        self.action_limit = action_limit
        self.embed = nn.Linear(obs_dim + act_dim, hid_dim)
        self.slstm = AdaptiveLSTMCell(hid_dim, hid_dim)
        self.mlstm = mLSTMCell(hid_dim, hid_dim)
        self.head = nn.Sequential(nn.Linear(hid_dim * 2, hid_dim), nn.GELU(), nn.Linear(hid_dim, act_dim))

    def forward(self, obs_win, act_win):
        B, L, _ = obs_win.shape
        x = self.embed(torch.cat([obs_win, act_win], dim=-1))
        
        h_s, c_s, m_s = [torch.zeros(B, 128, device=x.device) for _ in range(3)]
        C_m = torch.zeros(B, 128, 128, device=x.device)
        n_m = torch.zeros(B, 128, device=x.device)
        
        for t in range(L):
            h_s, c_s, m_s = self.slstm(x[:, t, :], h_s, c_s, m_s)
            h_m, C_m, n_m = self.mlstm(x[:, t, :], C_m, n_m)
            
        return self.action_limit * torch.tanh(self.head(torch.cat([h_s, h_m], dim=-1)))

# ============================================================================
# 5. TD3 AGENT
# ============================================================================

class TD3Agent:
    def __init__(self, obs_dim, act_dim, action_limit, actor_type="transformer", window=8, pe_type="learned"):
        self.act_dim, self.action_limit, self.window = act_dim, action_limit, window
        self.actor_type = actor_type
        
        if actor_type == "mlp":
            self.actor = MLPActor(obs_dim, act_dim, action_limit).to(device)
            self.actor_target = MLPActor(obs_dim, act_dim, action_limit).to(device)
        elif actor_type == "xlstm":
            self.actor = xLSTMActor(obs_dim, act_dim, action_limit, window).to(device)
            self.actor_target = xLSTMActor(obs_dim, act_dim, action_limit, window).to(device)
        else:
            self.actor = CausalTransformerActor(obs_dim, act_dim, action_limit, window, pe_type=pe_type).to(device)
            self.actor_target = CausalTransformerActor(obs_dim, act_dim, action_limit, window, pe_type=pe_type).to(device)
            
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.buffer = WindowReplayBuffer(obs_dim, act_dim, window)
        
        self._obs_ctx, self._act_ctx = deque(maxlen=window), deque(maxlen=window)
        self.update_count = 0

    def select_action(self, obs, explore=True):
        self._obs_ctx.append(obs)
        self._act_ctx.append(self._act_ctx[-1] if len(self._act_ctx) > 0 else np.zeros(self.act_dim, dtype=np.float32))
        
        obs_win = np.zeros((1, self.window, obs.shape[0]), dtype=np.float32)
        act_win = np.zeros((1, self.window, self.act_dim), dtype=np.float32)
        for i, (o, a) in enumerate(zip(self._obs_ctx, self._act_ctx)):
            obs_win[0, i], act_win[0, i] = o, a
            
        with torch.no_grad():
            act = self.actor(torch.FloatTensor(obs_win).to(device), torch.FloatTensor(act_win).to(device)).cpu().numpy().reshape(-1)
            
        self._act_ctx[-1] = act
        if explore:
            act = np.clip(act + np.random.normal(0, 0.1 * self.action_limit, size=act.shape), -self.action_limit, self.action_limit)
        return act

    def train(self):
        if self.buffer.size < 256: return
        
        obs_win, act_win, n_obs_win, n_act_win, (obs, acts, rewards, next_obs, dones) = self.buffer.sample_windows(256)

        with torch.no_grad():
            next_act = self.actor_target(n_obs_win, n_act_win)
            noise = (torch.randn_like(next_act) * 0.2 * self.action_limit).clamp(-0.5, 0.5)
            next_act = (next_act + noise).clamp(-self.action_limit, self.action_limit)
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            target_q = rewards + (1 - dones) * 0.99 * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, acts)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.update_count % 2 == 0:
            actor_loss = -self.critic(obs, self.actor(obs_win, act_win))[0].mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for s, t in zip(self.actor.parameters(), self.actor_target.parameters()): t.data.copy_(0.005 * s.data + 0.995 * t.data)
            for s, t in zip(self.critic.parameters(), self.critic_target.parameters()): t.data.copy_(0.005 * s.data + 0.995 * t.data)
            
        self.update_count += 1

# ============================================================================
# 6. RUNNERS
# ============================================================================

def run_experiment(args):
    env = make_env(args.env_type)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])
    
    agent = TD3Agent(obs_dim, act_dim, action_limit, actor_type=args.actor_type, window=args.window, pe_type=args.pe_type)
    norm = RunningNormaliser(obs_dim)
    
    returns = []
    obs, _ = env.reset()
    ep_ret = 0
    
    pbar = tqdm(range(args.steps), desc=f"{args.task} ({args.actor_type})")
    for step in pbar:
        if step < 10000:
            act = env.action_space.sample()
        else:
            act = agent.select_action(norm.normalise(obs), explore=True)
            
        next_obs, reward, term, trunc, _ = env.step(act)
        norm.update(obs.reshape(1, -1))
        
        agent.buffer.add(norm.normalise(obs), act, reward, norm.normalise(next_obs), term or trunc)
        agent.train()
        
        obs = next_obs
        ep_ret += reward
        
        if term or trunc:
            returns.append(ep_ret)
            ep_ret = 0
            obs, _ = env.reset()
            agent._obs_ctx.clear()
            agent._act_ctx.clear()
            
            if len(returns) % 10 == 0:
                pbar.set_postfix({"Avg Ret (10ep)": f"{np.mean(returns[-10:]):.1f}"})

    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.task}_results.pkl", "wb") as f: pickle.dump(returns, f)
    print(f"Finished {args.task}. Final Average Return: {np.mean(returns[-10:]):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["baseline", "task_a_learned", "task_a_sinusoidal", "task_b", "task_d"], required=True)
    parser.add_argument("--steps", type=int, default=500_000)
    args = parser.parse_args()

    # Configure parameters based on task
    if args.task == "baseline":
        args.env_type = "fully_observable"
        args.actor_type = "mlp"
        args.window, args.pe_type = 1, "learned"
        
    elif args.task.startswith("task_a"):
        args.env_type = "hidden_vel"
        args.actor_type = "transformer"
        args.window = 8
        args.pe_type = args.task.split("_")[-1]
        
    elif args.task == "task_b":
        args.env_type = "combined"
        args.actor_type = "transformer"
        args.window = 32
        args.pe_type = "learned"
        
    elif args.task == "task_d":
        args.env_type = "hidden_vel" # or combined based on your eval preference
        args.actor_type = "xlstm"
        args.window = 32
        args.pe_type = "none"

    run_experiment(args)