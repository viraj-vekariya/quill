import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .layers import LayerNorm


class MLP(nn.Module):
    """Position-wise feedforward: expand 4x, GELU, project back down."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(nn.functional.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> attention -> residual, LN -> MLP ->
    residual. Pre-LN (norm before the sub-layer, not after) is what GPT-2
    onward uses because it keeps gradients well-behaved at depth without
    a learning-rate warmup being load-bearing for stability.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        if use_cache:
            attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, present_kv

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
