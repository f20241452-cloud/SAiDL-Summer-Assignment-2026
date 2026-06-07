import os
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast

os.makedirs("/kaggle/working/saidl", exist_ok=True)

config = {
    "vocab_size": 50257,
    "n_layers": 4,
    "n_heads": 4,
    "embed_dim": 256,
    "context_len": 1024,
    "dropout": 0.1,
    "batch_size": 8,
    "lr": 3e-4,
    "max_steps": 2500,
    "eval_interval": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def tokenize(text_list):
    all_tokens = []
    for text in text_list:
        if text.strip():
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
    return torch.tensor(all_tokens, dtype=torch.long)

train_data = tokenize(dataset["train"]["text"])
val_data   = tokenize(dataset["validation"]["text"])

def get_batch(data, context_len, batch_size, device):
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x  = torch.stack([data[i:i+context_len] for i in ix])
    y  = torch.stack([data[i+1:i+context_len+1] for i in ix])
    return x.to(device), y.to(device)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len))
                             .view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len, window_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        positions = torch.arange(T, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        mask = torch.where(
            (dist >= -self.window_size) & (dist <= 0),
            torch.zeros(T, T, device=x.device),
            torch.full((T, T), float('-inf'), device=x.device)
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.proj   = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len))
                             .view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class LinearAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        k_cum = k.cumsum(dim=2)
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        num = torch.einsum('bhnd,bhdm->bhnm', q, kv)
        den = torch.einsum('bhnd,bhnd->bhn', q, k_cum).unsqueeze(-1) + 1e-6
        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        theta = 10000 ** (-2 * torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("theta", theta)

    def rotate(self, x):
        B, H, T, D = x.shape
        positions = torch.arange(T, device=x.device).float()
        angles = torch.outer(positions, self.theta)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rotate(q)
        k = self.rotate(k)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class ALiBiAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        slopes = 2 ** (-8 * torch.arange(1, n_heads + 1).float() / n_heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        positions = torch.arange(T, device=x.device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() * -1
        alibi = self.slopes.view(self.n_heads, 1, 1) * dist.unsqueeze(0)
        attn = attn + alibi
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class RelativeAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_emb = nn.Embedding(2 * context_len - 1, self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        positions = torch.arange(T, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1) + (self.rel_emb.num_embeddings // 2)
        rel_bias = self.rel_emb(rel_pos)
        rel_attn = torch.einsum('bhnd,tsd->bhts', q, rel_bias) * scale
        attn = attn + rel_attn
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class ALiBiSlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len, window_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        slopes = 2 ** (-8 * torch.arange(1, n_heads + 1).float() / n_heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        positions = torch.arange(T, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        alibi = self.slopes.view(self.n_heads, 1, 1) * dist.abs() * -1
        attn = attn + alibi
        sw_mask = torch.where(
            (dist >= -self.window_size) & (dist <= 0),
            torch.zeros(T, T, device=x.device),
            torch.full((T, T), float('-inf'), device=x.device)
        )
        attn = attn + sw_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class RoPESlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len, window_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        theta = 10000 ** (-2 * torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("theta", theta)

    def rotate(self, x):
        B, H, T, D = x.shape
        positions = torch.arange(T, device=x.device).float()
        angles = torch.outer(positions, self.theta)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.rotate(q)
        k = self.rotate(k)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        positions = torch.arange(T, device=x.device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        mask = torch.where(
            (dist >= -self.window_size) & (dist <= 0),
            torch.zeros(T, T, device=x.device),
            torch.full((T, T), float('-inf'), device=x.device)
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class AFTSimple(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj   = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        exp_k = torch.exp(k)
        num = torch.cumsum(exp_k * v, dim=1)
        den = torch.cumsum(exp_k, dim=1) + 1e-6
        out = torch.sigmoid(q) * (num / den)
        return self.proj(out)


class AFTFull(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.context_len = context_len
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj   = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Parameter(torch.zeros(context_len, context_len))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        w = self.w[:T, :T]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        w = w.masked_fill(causal_mask == 0, float('-inf'))
        logits = k.unsqueeze(1) + w.unsqueeze(0).unsqueeze(-1)
        logits = logits - logits.max(dim=2, keepdim=True).values
        exp_logits = torch.exp(logits)
        v_exp = (v.unsqueeze(1) * exp_logits).sum(dim=2)
        norm  = exp_logits.sum(dim=2) + 1e-6
        out = torch.sigmoid(q) * (v_exp / norm)
        return self.proj(out)


class AFTLocal(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len, local_window=64):
        super().__init__()
        self.local_window = local_window
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj   = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.w_local = nn.Parameter(torch.zeros(local_window))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        w = torch.zeros(T, T, device=x.device)
        for i in range(T):
            for j in range(max(0, i - self.local_window + 1), i + 1):
                dist = i - j
                if dist < self.local_window:
                    w[i, j] = self.w_local[dist]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        w = w.masked_fill(causal_mask == 0, float('-inf'))
        logits = k.unsqueeze(1) + w.unsqueeze(0).unsqueeze(-1)
        logits = logits - logits.max(dim=2, keepdim=True).values
        exp_logits = torch.exp(logits)
        v_exp = (v.unsqueeze(1) * exp_logits).sum(dim=2)
        norm  = exp_logits.sum(dim=2) + 1e-6
        out = torch.sigmoid(q) * (v_exp / norm)
        return self.proj(out)


class AFTConv(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len, kernel_size=3):
        super().__init__()
        self.aft_simple = AFTSimple(embed_dim, n_heads, dropout, context_len)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=0, groups=embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        aft_out = self.aft_simple(x)
        xc = self.ln(x).transpose(1, 2)
        xc = F.pad(xc, (2, 0))
        xc = self.conv(xc).transpose(1, 2)
        return self.dropout(aft_out + xc)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def make_block(attn, embed_dim, dropout):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = attn
            self.ff   = FeedForward(embed_dim, dropout)
            self.ln1  = nn.LayerNorm(embed_dim)
            self.ln2  = nn.LayerNorm(embed_dim)
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x
    return Block()


def build_gpt(config, attn_fn, use_pos_emb=True):
    embed_dim   = config["embed_dim"]
    n_heads     = config["n_heads"]
    dropout     = config["dropout"]
    context_len = config["context_len"]
    n_layers    = config["n_layers"]
    vocab_size  = config["vocab_size"]

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, embed_dim)
            self.pos_emb   = nn.Embedding(context_len, embed_dim) if use_pos_emb else None
            self.dropout   = nn.Dropout(dropout)
            self.blocks    = nn.Sequential(*[
                make_block(attn_fn(embed_dim, n_heads, dropout, context_len), embed_dim, dropout)
                for _ in range(n_layers)
            ])
            self.ln_final  = nn.LayerNorm(embed_dim)
            self.lm_head   = nn.Linear(embed_dim, vocab_size)

        def forward(self, x, targets=None):
            B, T = x.shape
            tok = self.token_emb(x)
            if self.pos_emb is not None:
                pos = self.pos_emb(torch.arange(T, device=x.device))
                tok = tok + pos
            x = self.dropout(tok)
            x = self.blocks(x)
            x = self.ln_final(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    return GPT()


class ConvAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, context_len):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=0, groups=embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout, context_len)
        self.ff   = FeedForward(embed_dim, dropout)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ln3  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        xc = self.ln1(x).transpose(1, 2)
        xc = F.pad(xc, (2, 0))
        xc = self.conv(xc).transpose(1, 2)
        x = x + xc
        x = x + self.attn(self.ln2(x))
        x = x + self.ff(self.ln3(x))
        return x


def build_conv_gpt(config, use_pos_emb=True):
    embed_dim   = config["embed_dim"]
    n_heads     = config["n_heads"]
    dropout     = config["dropout"]
    context_len = config["context_len"]
    n_layers    = config["n_layers"]
    vocab_size  = config["vocab_size"]

    class ConvGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, embed_dim)
            self.pos_emb   = nn.Embedding(context_len, embed_dim) if use_pos_emb else None
            self.dropout   = nn.Dropout(dropout)
            self.blocks    = nn.Sequential(*[
                ConvAttentionBlock(embed_dim, n_heads, dropout, context_len)
                for _ in range(n_layers)
            ])
            self.ln_final  = nn.LayerNorm(embed_dim)
            self.lm_head   = nn.Linear(embed_dim, vocab_size)

        def forward(self, x, targets=None):
            B, T = x.shape
            tok = self.token_emb(x)
            if self.pos_emb is not None:
                pos = self.pos_emb(torch.arange(T, device=x.device))
                tok = tok + pos
            x = self.dropout(tok)
            x = self.blocks(x)
            x = self.ln_final(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    return ConvGPT()


class InterleavedGPT(nn.Module):
    def __init__(self, config, attn_fn=None, use_pos_emb=True):
        super().__init__()
        embed_dim   = config["embed_dim"]
        n_heads     = config["n_heads"]
        dropout     = config["dropout"]
        context_len = config["context_len"]
        n_layers    = config["n_layers"]
        vocab_size  = config["vocab_size"]

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_len, embed_dim) if use_pos_emb else None
        self.dropout   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()

        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(nn.ModuleDict({
                    'conv': nn.Conv1d(embed_dim, embed_dim, 3, padding=0, groups=embed_dim),
                    'ff':   FeedForward(embed_dim, dropout),
                    'ln1':  nn.LayerNorm(embed_dim),
                    'ln2':  nn.LayerNorm(embed_dim),
                }))
            else:
                attn = attn_fn(embed_dim, n_heads, dropout, context_len) if attn_fn else \
                       MultiHeadSelfAttention(embed_dim, n_heads, dropout, context_len)
                self.blocks.append(nn.ModuleDict({
                    'attn': attn,
                    'ff':   FeedForward(embed_dim, dropout),
                    'ln1':  nn.LayerNorm(embed_dim),
                    'ln2':  nn.LayerNorm(embed_dim),
                }))

        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head  = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.token_emb(x)
        if self.pos_emb is not None:
            tok = tok + self.pos_emb(torch.arange(T, device=x.device))
        x = self.dropout(tok)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                xc = block['ln1'](x).transpose(1, 2)
                xc = F.pad(xc, (2, 0))
                xc = block['conv'](xc).transpose(1, 2)
                x = x + xc
                x = x + block['ff'](block['ln2'](x))
            else:
                x = x + block['attn'](block['ln1'](x))
                x = x + block['ff'](block['ln2'](x))
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def build_aft_gpt(config, attn_class, **attn_kwargs):
    embed_dim   = config["embed_dim"]
    n_heads     = config["n_heads"]
    dropout     = config["dropout"]
    context_len = config["context_len"]
    n_layers    = config["n_layers"]
    vocab_size  = config["vocab_size"]

    class AFTGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, embed_dim)
            self.dropout   = nn.Dropout(dropout)
            self.blocks    = nn.Sequential(*[
                make_block(
                    attn_class(embed_dim, n_heads, dropout, context_len, **attn_kwargs),
                    embed_dim, dropout
                )
                for _ in range(n_layers)
            ])
            self.ln_final  = nn.LayerNorm(embed_dim)
            self.lm_head   = nn.Linear(embed_dim, vocab_size)

        def forward(self, x, targets=None):
            B, T = x.shape
            x    = self.dropout(self.token_emb(x))
            x    = self.blocks(x)
            x    = self.ln_final(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    return AFTGPT()





@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, eval_steps=100):
    model.eval()
    out = {}
    device = config["device"]
    for split, data in [("train", train_data), ("val", val_data)]:
        batch_perplexities = []
        losses = []
        for v_step in range(eval_steps):
            x, y = get_batch(data, config["context_len"], config["batch_size"], device)
            _, loss = model(x, y)
            losses.append(loss.item())
            batch_perplexities.append(math.exp(loss.item()))
        out[split] = {
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "ppl_mean": float(np.mean(batch_perplexities)),
            "ppl_std": float(np.std(batch_perplexities))
        }
    model.train()
    return out

def train(model, train_data, val_data, config, save_path=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"]
    )

    start_time = time.time()
    total_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()

    for step in range(config["max_steps"]):

        x, y = get_batch(
            train_data,
            config["context_len"],
            config["batch_size"],
            config["device"]
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_tokens += (
            config["batch_size"]
            * config["context_len"]
        )

        if step % config["eval_interval"] == 0:

            stats = estimate_loss(
                model,
                train_data,
                val_data,
                config
            )

            elapsed = time.time() - start_time

            throughput = (
                total_tokens / elapsed
                if elapsed > 0 else 0
            )

            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
                if torch.cuda.is_available()
                else 0
            )

            print(
                f"step {step:5d} | "
                f"train loss {stats['train']['loss_mean']:.4f} | "
                f"val loss {stats['val']['loss_mean']:.4f} | "
                f"val ppl {stats['val']['ppl_mean']:.2f} "
                f"(±{stats['val']['ppl_std']:.2f}) | "
                f"{throughput:,.0f} tok/s | "
                f"{peak_mb:.0f} MB"
            )

            if save_path:
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_perplexity": stats["val"]["ppl_mean"]
                    },
                    f"{save_path}_step{step}.pt"
                )

    total_time = time.time() - start_time

    final_stats = estimate_loss(
        model,
        train_data,
        val_data,
        config
    )

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0
    )

    throughput = (
        total_tokens / total_time
        if total_time > 0 else 0
    )

    print(
        f"\nFINAL | "
        f"val ppl {final_stats['val']['ppl_mean']:.2f} "
        f"(±{final_stats['val']['ppl_std']:.2f}) | "
        f"time {total_time:.1f}s | "
        f"gpu {peak_mb:.0f} MB | "
        f"{throughput:,.0f} tok/s"
    )

    return {
        "perplexity": final_stats["val"]["ppl_mean"],
        "ppl_std": final_stats["val"]["ppl_std"],
        "loss": final_stats["val"]["loss_mean"],
        "loss_std": final_stats["val"]["loss_std"],
        "time_seconds": total_time,
        "peak_gpu_mb": peak_mb,
        "throughput": throughput
    }

baseline = build_gpt(config, MultiHeadSelfAttention, use_pos_emb=True)
baseline_stats = run_experiment("Baseline", baseline.to(config["device"]), config, "baseline")
del baseline; gc.collect(); torch.cuda.empty_cache()

sw = build_gpt(config, SlidingWindowAttention, use_pos_emb=True)
sw_stats = run_experiment("Sliding Window", sw.to(config["device"]), config, "sliding_window")
del sw; gc.collect(); torch.cuda.empty_cache()

mqa = build_gpt(config, MultiQueryAttention, use_pos_emb=True)
mqa_stats = run_experiment("MQA", mqa.to(config["device"]), config, "mqa")
del mqa; gc.collect(); torch.cuda.empty_cache()

linear = build_gpt(config, LinearAttention, use_pos_emb=True)
linear_stats = run_experiment("Linear Attention", linear.to(config["device"]), config, "linear_attention")
del linear; gc.collect(); torch.cuda.empty_cache()

rope = build_gpt(config, RoPEAttention, use_pos_emb=False)
rope_stats = run_experiment("RoPE", rope.to(config["device"]), config, "rope")
del rope; gc.collect(); torch.cuda.empty_cache()

alibi = build_gpt(config, ALiBiAttention, use_pos_emb=False)
alibi_stats = run_experiment("ALiBi", alibi.to(config["device"]), config, "alibi")
del alibi; gc.collect(); torch.cuda.empty_cache()

rel = build_gpt(config, RelativeAttention, use_pos_emb=False)
rel_stats = run_experiment("Relative PE", rel.to(config["device"]), config, "relative_pe")
del rel; gc.collect(); torch.cuda.empty_cache()

conv_attn = build_conv_gpt(config, use_pos_emb=True)
conv_stats = run_experiment("Conv+Attention", conv_attn.to(config["device"]), config, "conv_attention")
del conv_attn; gc.collect(); torch.cuda.empty_cache()

interleaved = InterleavedGPT(config)
interleaved_stats = run_experiment("Interleaved", interleaved.to(config["device"]), config, "interleaved")
del interleaved; gc.collect(); torch.cuda.empty_cache()

conv_alibi = InterleavedGPT(config, attn_fn=ALiBiSlidingWindowAttention, use_pos_emb=False)
conv_alibi_stats = run_experiment("Conv+ALiBi+SW", conv_alibi.to(config["device"]), config, "conv_alibi_sw")
del conv_alibi; gc.collect(); torch.cuda.empty_cache()

conv_rope = InterleavedGPT(config, attn_fn=RoPESlidingWindowAttention, use_pos_emb=False)
conv_rope_stats = run_experiment("Conv+RoPE+SW", conv_rope.to(config["device"]), config, "conv_rope_sw")
del conv_rope; gc.collect(); torch.cuda.empty_cache()

extrap_config = config.copy()
extrap_config["context_len"] = 512

rope_ext = build_gpt(extrap_config, RoPEAttention, use_pos_emb=False).to(config["device"])
train(rope_ext, train_data, val_data, extrap_config)

alibi_ext = build_gpt(extrap_config, ALiBiAttention, use_pos_emb=False).to(config["device"])
train(alibi_ext, train_data, val_data, extrap_config)

rel_ext = build_gpt(extrap_config, RelativeAttention, use_pos_emb=False).to(config["device"])
train(rel_ext, train_data, val_data, extrap_config)

print(f"\n{'Model':<15} {'512':>10} {'1024':>10} {'2048':>10}")
for name, model in [("RoPE", rope_ext), ("ALiBi", alibi_ext), ("Relative PE", rel_ext)]:
    p512  = test_extrapolation(model, val_data, 512,  config["device"])
    p1024 = test_extrapolation(model, val_data, 1024, config["device"])
    p2048 = test_extrapolation(model, val_data, 2048, config["device"])
    print(f"{name:<15} {p512:>10.2f} {p1024:>10.2f} {p2048:>10.2f}")

del rope_ext, alibi_ext, rel_ext; gc.collect(); torch.cuda.empty_cache()

context_lengths = [512, 1024, 2048]
multi_ctx_results = {}

for model_name, attn_fn, use_pos in [
    ("Baseline",         MultiHeadSelfAttention, True),
    ("Sliding Window",   SlidingWindowAttention, True),
    ("MQA",              MultiQueryAttention,    True),
    ("Linear Attention", LinearAttention,        True),
]:
    multi_ctx_results[model_name] = {}
    for ctx_len in context_lengths:
        gc.collect(); torch.cuda.empty_cache()
        cfg = config.copy()
        cfg["context_len"] = ctx_len
        cfg["batch_size"]  = 4 if ctx_len >= 2048 else config["batch_size"]
        m = build_gpt(cfg, attn_fn, use_pos_emb=use_pos).to(config["device"])
        stats = train(m, train_data, val_data, cfg)
        multi_ctx_results[model_name][ctx_len] = stats
        del m; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'Model':<20} {'CTX':>6} {'PPL':>10} {'GPU MB':>10} {'tok/s':>12}")
for name, ctx_dict in multi_ctx_results.items():
    for ctx, s in ctx_dict.items():
        print(f"{name:<20} {ctx:>6} {s['perplexity']:>10.2f} {s['peak_gpu_mb']:>10.0f} {s['throughput']:>12,.0f}")

aft_variants = [
    ("AFT-Simple", AFTSimple, {}),
    ("AFT-Full",   AFTFull,   {}),
    ("AFT-Local",  AFTLocal,  {"local_window": 64}),
    ("AFT-Conv",   AFTConv,   {}),
]

aft_results = {}
for name, attn_class, kwargs in aft_variants:
    gc.collect(); torch.cuda.empty_cache()
    m = build_aft_gpt(config, attn_class, **kwargs).to(config["device"])
    stats = run_experiment(name, m, config, f"aft_{name.lower().replace('-', '_')}")
    aft_results[name] = stats
    del m; gc.collect(); torch.cuda.empty_cache()

print(f"\n{'Model':<25} {'PPL':>10} {'GPU MB':>10} {'tok/s':>12}")
for name, s in aft_results.items():
    print(f"{name:<25} {s['perplexity']:>10.2f} {s['peak_gpu_mb']:>10.0f} {s['throughput']:>12,.0f}")
