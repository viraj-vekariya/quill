import torch

from quill.attention import CausalSelfAttention
from quill.config import GPTConfig


def make_attn(dropout=0.0):
    config = GPTConfig(vocab_size=10, block_size=16, n_layer=1, n_head=2, n_embd=8, dropout=dropout)
    attn = CausalSelfAttention(config)
    attn.eval()  # dropout=0.0 makes this redundant, but be explicit
    return attn, config


def test_output_shape_matches_input():
    attn, config = make_attn()
    x = torch.randn(3, 6, config.n_embd)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_mask_blocks_future_positions():
    """The defining property of causal self-attention: the output at
    position t must depend only on inputs at positions <= t. We verify this
    directly rather than trusting the mask math: perturb only the LAST
    token's input and check that every earlier position's output is
    completely unchanged.
    """
    attn, config = make_attn()
    torch.manual_seed(0)
    x = torch.randn(2, config.block_size, config.n_embd)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 100.0  # large perturbation to the last token only

    with torch.no_grad():
        out_original = attn(x)
        out_perturbed = attn(x_perturbed)

    # Every position except the last must be identical.
    assert torch.allclose(out_original[:, :-1, :], out_perturbed[:, :-1, :], atol=1e-6)
    # The last position (which attends to itself) must actually have changed
    # — otherwise this test would pass trivially with a broken attention
    # that ignores its input entirely.
    assert not torch.allclose(out_original[:, -1, :], out_perturbed[:, -1, :], atol=1e-4)


def test_attention_weights_are_causal_and_normalized():
    """Directly inspect the attention weight matrix: for query position t,
    weight mass on keys > t must be exactly zero, and each row must sum to 1.
    """
    config = GPTConfig(vocab_size=10, block_size=5, n_layer=1, n_head=1, n_embd=4, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(1, 5, 4)
    q, k, v = attn.qkv_proj(x).split(4, dim=2)
    import math
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
    scores = scores.masked_fill(attn.causal_mask[:, :, :5, :5].squeeze(1) == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)[0]  # (5, 5)

    for t in range(5):
        future_mass = weights[t, t + 1:].sum().item()
        assert future_mass == 0.0
        row_sum = weights[t].sum().item()
        assert abs(row_sum - 1.0) < 1e-5


def test_gradients_flow_through_attention():
    attn, config = make_attn()
    x = torch.randn(2, 6, config.n_embd, requires_grad=True)
    out = attn(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
