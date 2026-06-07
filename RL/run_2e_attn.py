import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from td3_causal_transformer import CausalTransformerActor, _PreLNTransformerLayer
from td3_partial_obs import run_attention_analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _InstrumentedPreLNLayer(_PreLNTransformerLayer):
    """Returns attention weights alongside output."""
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, attn_w = self.attn(
            h, h, h, attn_mask=causal_mask, is_causal=True,
            need_weights=True, average_attn_weights=False,
        )
        x = x + self.drop(attn_out)
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x, attn_w

class InstrumentedTransformerActor(CausalTransformerActor):
    """Subclass that caches raw attention weights during forward passes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = []
        embed_dim, num_heads, num_layers, dropout = (
            kwargs.get("embed_dim", 128), kwargs.get("num_heads", 4),
            kwargs.get("num_layers", 2), kwargs.get("dropout", 0.0)
        )
        self.layers = nn.ModuleList([
            _InstrumentedPreLNLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, obs_win: torch.Tensor, act_win: torch.Tensor) -> torch.Tensor:
        self.attn_weights = []
        B, L, _ = obs_win.shape
        x = torch.cat([obs_win, act_win], dim=-1)
        x = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        x = x + self.pos_embed(pos).unsqueeze(0)

        for layer in self.layers:
            x, w = layer(x, self.causal_mask)
            self.attn_weights.append(w)

        x = self.norm(x)
        return self.action_limit * torch.tanh(self.action_head(x[:, -1, :]))

def chefer_relevancy(actor: InstrumentedTransformerActor, obs_win: torch.Tensor, act_win: torch.Tensor) -> np.ndarray:
    """R_t = Σ_l  Ā_l[:, -1, t]  where Ā_l = attention + grad_attention ⊙ I"""
    actor.train()
    obs_win, act_win = obs_win.requires_grad_(False), act_win.requires_grad_(False)

    act_out = actor(obs_win, act_win)
    scalar = act_out.sum()
    relevancies = []
    
    for w in actor.attn_weights:
        w_mean = w.mean(dim=1)
        grad = torch.autograd.grad(scalar, w, retain_graph=True)[0]
        g_mean = grad.mean(dim=1)
        
        A_bar = (w_mean + g_mean * w_mean).clamp(min=0)
        A_bar = A_bar + torch.eye(A_bar.shape[-1], device=A_bar.device).unsqueeze(0)
        A_bar = A_bar / A_bar.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        relevancies.append(A_bar[0, -1, :].detach().cpu().numpy())

    actor.eval()
    rel = np.stack(relevancies, axis=0).mean(axis=0)
    return rel / (rel.sum() + 1e-8)

def main(ckpt_full, ckpt_hidden, window, steps, seed, save_dir="results_2e"):
    print("\n--- Running Section 2e: Attention Attribution Analysis ---")
    run_attention_analysis(
        ckpt_full_obs=ckpt_full, ckpt_hidden_vel=ckpt_hidden,
        window=window, n_episodes=5, total_steps=steps, seed=seed, save_dir=save_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_full", type=str, default=None)
    parser.add_argument("--ckpt_hidden", type=str, default=None)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.ckpt_full, args.ckpt_hidden, args.window, args.steps, args.seed)