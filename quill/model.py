import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import GPTConfig
from .layers import LayerNorm


class GPT(nn.Module):
    """A decoder-only transformer language model (GPT-style), built from the
    from-scratch attention/LayerNorm/block pieces in this package.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the same matrix maps tokens -> embeddings and
        # final-hidden-state -> next-token logits. Halves the embedding
        # parameter count and is standard practice since GPT-2.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual output projections: without
        # this, activations grow with sqrt(n_layer) as blocks stack. Scaling
        # by 1/sqrt(2 * n_layer) keeps the residual stream's variance roughly
        # constant regardless of depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list | None = None,
        use_cache: bool = False,
    ):
        """past_kv, if given, is a list of one (k, v) tuple per block, as
        returned by a previous use_cache=True call — idx then only needs to
        carry the *new* positions (e.g. a single new token during
        generation), since attention for the already-seen positions doesn't
        need recomputing. Position embeddings are offset by the cached
        length so absolute positions stay correct.

        With use_cache=False (the default) this is the original method
        exactly — existing callers (training, loss computation, tests) are
        unaffected and still get back a plain (logits, loss) pair.
        """
        B, T = idx.shape
        past_len = past_kv[0][0].size(2) if past_kv is not None else 0
        assert past_len + T <= self.config.block_size, (
            f"sequence length {past_len + T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.dropout(self.token_emb(idx) + self.pos_emb(pos))

        if use_cache:
            new_past_kv = []
            for i, block in enumerate(self.blocks):
                layer_past = past_kv[i] if past_kv is not None else None
                x, present_kv = block(x, past_kv=layer_past, use_cache=True)
                new_past_kv.append(present_kv)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss, new_past_kv

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _next_token(self, logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens continuations of idx.

        use_cache=True is the fast path: past keys/values are cached per
        layer so each new token only costs attention over its one new query
        against the cached history, instead of recomputing attention over
        the whole growing sequence at every step. Unlike the default path,
        it has no sliding-window eviction — len(idx) + max_new_tokens must
        fit within block_size (the default path instead silently slides its
        window over the last block_size tokens once the sequence outgrows
        it, which is why use_cache defaults to False: it's the strictly more
        permissive behavior). For any generation that fits in one context
        window — the common case — pass use_cache=True for a real speedup.
        """
        self.eval()

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                next_id = self._next_token(logits, temperature, top_k)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

        assert idx.size(1) + max_new_tokens <= self.config.block_size, (
            "cached generate() has no sliding-window eviction: prompt + "
            f"max_new_tokens ({idx.size(1) + max_new_tokens}) must fit within "
            f"block_size ({self.config.block_size})"
        )

        logits, _, past_kv = self(idx, use_cache=True)
        next_id = self._next_token(logits, temperature, top_k)
        idx = torch.cat((idx, next_id), dim=1)

        for _ in range(max_new_tokens - 1):
            logits, _, past_kv = self(next_id, past_kv=past_kv, use_cache=True)
            next_id = self._next_token(logits, temperature, top_k)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
