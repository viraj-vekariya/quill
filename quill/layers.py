import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm written out by hand instead of using nn.LayerNorm, so the
    normalization math isn't hidden behind a library call either. Normalizes
    each token's feature vector to zero mean / unit variance, then applies a
    learned per-feature scale and shift.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias
