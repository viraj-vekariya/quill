from pathlib import Path

import pytest
import torch

from quill.config import GPTConfig
from quill.model import GPT

ROOT = Path(__file__).resolve().parent.parent


def make_model(vocab_size=20, block_size=32):
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    model.eval()
    return model, config


def test_cached_generation_matches_uncached_token_ids():
    """The critical correctness property: caching must not change WHAT gets
    generated, only how fast. Greedy decoding (top_k=1) removes sampling
    randomness so token IDs must match exactly, not just approximately.
    """
    torch.manual_seed(0)
    model, config = make_model()
    prompt = torch.randint(0, config.vocab_size, (2, 5))

    torch.manual_seed(1)
    out_uncached = model.generate(prompt.clone(), max_new_tokens=10, top_k=1, use_cache=False)

    torch.manual_seed(1)
    out_cached = model.generate(prompt.clone(), max_new_tokens=10, top_k=1, use_cache=True)

    assert torch.equal(out_uncached, out_cached)


def test_cached_logits_match_uncached_logits_at_each_step():
    """Stronger check than final token IDs alone: verify the actual logits
    the cached path computes at an interior generation step match a full
    from-scratch forward pass over the same growing sequence, within float
    tolerance (the two paths sum floating point terms in a different order,
    so exact equality isn't guaranteed — closeness is what matters).
    """
    torch.manual_seed(0)
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (1, 4))

    logits_full, _ = model(idx)
    logits_cached, _, past_kv = model(idx, use_cache=True)
    assert torch.allclose(logits_full[:, -1, :], logits_cached[:, -1, :], atol=1e-5)

    next_tok = torch.randint(0, config.vocab_size, (1, 1))
    idx_ext = torch.cat([idx, next_tok], dim=1)
    logits_full_ext, _ = model(idx_ext)
    logits_step, _, _ = model(next_tok, past_kv=past_kv, use_cache=True)

    assert torch.allclose(logits_full_ext[:, -1, :], logits_step[:, -1, :], atol=1e-5)


def test_cached_generation_matches_uncached_on_trained_checkpoint():
    """Same equivalence check against the real trained checkpoint committed
    in outputs/ — the actual model this project produced, not just a
    randomly-initialized toy."""
    ckpt_path = ROOT / "outputs" / "checkpoint.pt"
    if not ckpt_path.exists():
        pytest.skip("no trained checkpoint present")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = torch.zeros((1, 3), dtype=torch.long)

    torch.manual_seed(42)
    out_uncached = model.generate(prompt.clone(), max_new_tokens=40, top_k=1, use_cache=False)
    torch.manual_seed(42)
    out_cached = model.generate(prompt.clone(), max_new_tokens=40, top_k=1, use_cache=True)

    assert torch.equal(out_uncached, out_cached)


def test_cache_grows_by_exactly_one_position_per_step():
    torch.manual_seed(0)
    model, config = make_model()
    idx = torch.randint(0, config.vocab_size, (1, 4))

    _, _, past_kv = model(idx, use_cache=True)
    for k, v in past_kv:
        assert k.shape[2] == 4
        assert v.shape[2] == 4

    next_tok = torch.randint(0, config.vocab_size, (1, 1))
    _, _, past_kv2 = model(next_tok, past_kv=past_kv, use_cache=True)
    for k, v in past_kv2:
        assert k.shape[2] == 5
        assert v.shape[2] == 5


def test_generate_with_cache_rejects_overflow_past_block_size():
    model, config = make_model(block_size=8)
    prompt = torch.zeros((1, 5), dtype=torch.long)
    with pytest.raises(AssertionError):
        model.generate(prompt, max_new_tokens=10, use_cache=True)
