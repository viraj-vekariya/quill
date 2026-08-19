import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention, written out at the
    tensor-op level (matmul -> mask -> softmax -> matmul) instead of calling
    nn.MultiheadAttention or F.scaled_dot_product_attention. Autograd still
    handles the backward pass — reimplementing backprop by hand wouldn't
    teach anything a from-scratch forward pass doesn't already — but nothing
    about *how attention computes its output* is hidden behind a library call.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precomputed causal mask: position i may attend to positions <= i.
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        """past_kv, if given, is a (k, v) tuple of cached keys/values from
        previous positions, each shaped (B, n_head, T_past, head_dim). x only
        needs to carry the *new* positions — their k/v get computed here and
        concatenated onto the cache, so attention scores only need computing
        for the new queries against the full (past + new) key/value history,
        instead of recomputing every past position's attention from scratch.

        When use_cache is False (the default), behaves exactly as before and
        returns just the output tensor — existing callers are unaffected.
        When use_cache is True, returns (out, (k, v)) so the caller can carry
        the updated cache into the next step.
        """
        B, T, C = x.shape

        q, k, v = self.qkv_proj(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        past_len = k.size(2) - T
        total_len = k.size(2)

        # scaled dot-product attention scores: (B, n_head, T, total_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Query row i here is absolute position past_len + i; it may attend
        # to key column j (absolute position j) iff j <= past_len + i. With
        # no cache (past_len=0, total_len=T) this is exactly the original
        # square causal mask sliced [:T, :T].
        mask = self.causal_mask[:, :, past_len:past_len + T, :total_len]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        if use_cache:
            return out, (k, v)
        return out
