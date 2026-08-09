import torch
import torch.nn as nn

from quill.layers import LayerNorm


def test_matches_pytorch_layernorm_numerically():
    """The whole point of writing LayerNorm by hand is that it computes the
    same thing nn.LayerNorm does. Prove it, don't just assert it runs.
    """
    torch.manual_seed(0)
    dim = 32
    x = torch.randn(4, 10, dim)

    ours = LayerNorm(dim)
    reference = nn.LayerNorm(dim)
    # Match the learned affine parameters so this is purely a math check.
    with torch.no_grad():
        reference.weight.copy_(ours.weight)
        reference.bias.copy_(ours.bias)

    out_ours = ours(x)
    out_ref = reference(x)
    assert torch.allclose(out_ours, out_ref, atol=1e-5)


def test_output_has_zero_mean_unit_variance_before_affine():
    torch.manual_seed(0)
    dim = 16
    x = torch.randn(3, 5, dim) * 10 + 100  # arbitrary scale/shift

    ln = LayerNorm(dim)
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)
    out = ln(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(3, 5), atol=1e-3)
