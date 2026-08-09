import torch

from quill.config import GPTConfig
from quill.model import GPT


def make_model(vocab_size=20, block_size=16):
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    return GPT(config), config


def test_forward_logits_shape():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (3, 10))
    targets = torch.randint(0, config.vocab_size, (3, 10))
    logits, loss = model(idx, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_lm_head_is_tied_to_token_embedding():
    model, _ = make_model()
    assert model.lm_head.weight is model.token_emb.weight


def test_generate_appends_exactly_max_new_tokens():
    model, config = make_model()
    context = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(context, max_new_tokens=15)
    assert out.shape == (2, 3 + 15)
    assert out[:, :3].equal(context)
    assert out.min() >= 0 and out.max() < config.vocab_size


def test_model_can_overfit_a_tiny_repeated_sequence():
    """Classic ML sanity check: a model that can't memorize 8 tokens
    repeated a few times has a bug, no amount of real-data training will
    fix it. This is the single most useful test for catching a broken
    forward/backward pass — much more useful than checking shapes alone.
    """
    torch.manual_seed(0)
    vocab_size, block_size = 8, 8
    model, _ = make_model(vocab_size=vocab_size, block_size=block_size)

    pattern = torch.arange(vocab_size).unsqueeze(0)  # (1, 8): 0,1,2,...,7
    x = pattern[:, :-1]
    y = pattern[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    initial_loss = None
    for step in range(200):
        _, loss = model(x, y)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, (
        f"expected the model to memorize an 8-token pattern; loss went {initial_loss:.3f} -> {final_loss:.3f}"
    )
