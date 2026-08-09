"""Sample text from a trained Quill checkpoint.

Usage:
    python generate.py --prompt "ROMEO:" --max_new_tokens 300
"""
import argparse
from pathlib import Path

import torch

from quill.config import GPTConfig
from quill.model import GPT
from quill.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(ROOT / "outputs"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    device = pick_device()

    tokenizer = CharTokenizer.load(str(out_dir / "tokenizer.json"))
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out_ids = model.generate(
        context, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
    )[0].tolist()
    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
