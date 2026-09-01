import torch

from quill.config import GPTConfig
from quill.model import GPT


def make_model(vocab_size=20, block_size=32):
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    model.eval()
    return model, config


def test_top_p_on_a_sharply_peaked_distribution_keeps_only_the_dominant_token():
    """One token near-certain, low top_p -- the nucleus should collapse to
    just that token, and it must never crash or produce an empty set."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    filtered = GPT._top_p_filter(logits.clone(), top_p=0.1)
    kept = torch.isfinite(filtered)
    assert kept.sum().item() == 1
    assert kept[0, 0].item()


def test_higher_top_p_keeps_at_least_as_many_candidates_as_lower_top_p():
    """On the same flatter distribution, a larger nucleus should never be
    smaller than a tighter one."""
    torch.manual_seed(0)
    logits = torch.randn(1, 50)
    kept_low = torch.isfinite(GPT._top_p_filter(logits.clone(), top_p=0.5)).sum().item()
    kept_high = torch.isfinite(GPT._top_p_filter(logits.clone(), top_p=0.9)).sum().item()
    assert kept_high >= kept_low


def test_top_p_never_produces_an_empty_candidate_set():
    torch.manual_seed(1)
    for p in (0.0, 0.01, 0.3, 0.7, 0.99, 1.0):
        logits = torch.randn(4, 30)
        filtered = GPT._top_p_filter(logits.clone(), top_p=p)
        kept_per_row = torch.isfinite(filtered).sum(dim=-1)
        assert (kept_per_row >= 1).all(), f"empty nucleus at top_p={p}"


def test_top_p_close_to_one_keeps_effectively_everything():
    torch.manual_seed(2)
    logits = torch.randn(1, 40)
    filtered = GPT._top_p_filter(logits.clone(), top_p=1.0)
    assert torch.isfinite(filtered).all()


def test_combining_top_k_and_top_p_yields_a_valid_nonempty_sample():
    """_next_token applies top_k first, then narrows with top_p -- confirm
    the composition never breaks (empty softmax / NaN) and always returns
    a legal token id."""
    torch.manual_seed(3)
    model, config = make_model()
    logits = torch.randn(1, 1, config.vocab_size)

    for _ in range(20):
        next_id = model._next_token(logits.clone(), temperature=1.0, top_k=5, top_p=0.3)
        assert next_id.shape == (1, 1)
        assert 0 <= next_id.item() < config.vocab_size


def test_generate_with_top_p_runs_end_to_end_without_cache_and_with_cache():
    torch.manual_seed(4)
    model, config = make_model()
    prompt = torch.randint(0, config.vocab_size, (1, 5))

    out_no_cache = model.generate(prompt.clone(), max_new_tokens=8, top_p=0.9, use_cache=False)
    out_cache = model.generate(prompt.clone(), max_new_tokens=8, top_p=0.9, use_cache=True)

    assert out_no_cache.shape == (1, 13)
    assert out_cache.shape == (1, 13)


def test_low_top_p_with_greedy_equivalent_seed_still_matches_argmax_choice():
    """A degenerate but useful sanity check: with a very low top_p the
    nucleus is just the single most likely token, so sampling from it must
    equal a plain argmax over the unfiltered distribution."""
    torch.manual_seed(5)
    model, config = make_model()
    logits = torch.randn(1, 1, config.vocab_size)
    expected = logits[0, -1, :].argmax().item()

    next_id = model._next_token(logits.clone(), temperature=1.0, top_k=None, top_p=1e-6)
    assert next_id.item() == expected
