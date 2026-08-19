"""Measure the real wall-clock speedup from KV-cached generation.

Loads the trained checkpoint in outputs/ (falls back to a freshly
initialized model of the same size if no checkpoint is present) and times
generating the same number of tokens with use_cache=False vs use_cache=True.

Usage:
    python benchmark_cache.py --max_new_tokens 300 --repeats 3
"""
import argparse
import time
from pathlib import Path

import torch

from quill.config import GPTConfig
from quill.model import GPT

ROOT = Path(__file__).resolve().parent


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str) -> GPT:
    ckpt_path = ROOT / "outputs" / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"loaded trained checkpoint: {config}")
    else:
        config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0)
        model = GPT(config).to(device)
        print(f"no checkpoint found, using a freshly initialized model: {config}")
    model.eval()
    return model, config


def timed_generate(model, idx, max_new_tokens, use_cache, device) -> float:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(idx, max_new_tokens=max_new_tokens, top_k=1, use_cache=use_cache)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    device = pick_device()
    model, config = load_model(device)
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    assert 1 + args.max_new_tokens <= config.block_size, (
        "reduce --max_new_tokens or this won't fit in one context window"
    )

    # One untimed warmup call each, so device/kernel warmup doesn't skew the
    # first timed measurement.
    model.generate(prompt, max_new_tokens=5, top_k=1, use_cache=False)
    model.generate(prompt, max_new_tokens=5, top_k=1, use_cache=True)

    uncached_times = [timed_generate(model, prompt, args.max_new_tokens, False, device) for _ in range(args.repeats)]
    cached_times = [timed_generate(model, prompt, args.max_new_tokens, True, device) for _ in range(args.repeats)]

    uncached_best = min(uncached_times)
    cached_best = min(cached_times)

    print(f"\ndevice: {device}, max_new_tokens: {args.max_new_tokens}, repeats: {args.repeats}")
    print(f"uncached: best={uncached_best:.3f}s  all={[f'{t:.3f}' for t in uncached_times]}")
    print(f"cached:   best={cached_best:.3f}s  all={[f'{t:.3f}' for t in cached_times]}")
    print(f"speedup (best-of-{args.repeats}): {uncached_best / cached_best:.2f}x")


if __name__ == "__main__":
    main()
