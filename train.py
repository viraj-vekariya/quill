"""Train Quill on a character corpus.

Usage:
    python train.py                       # defaults: data/tinyshakespeare.txt
    python train.py --max_iters 500       # quick smoke run
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from quill.config import GPTConfig
from quill.dataset import CharDataset
from quill.model import GPT
from quill.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lr_at(step: int, max_iters: int, base_lr: float, warmup: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_iters - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


@torch.no_grad()
def estimate_loss(model, splits: dict, batch_size: int, device: str, eval_iters: int) -> dict:
    model.eval()
    out = {}
    for name, dataset in splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = dataset.get_batch(batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "tinyshakespeare.txt"))
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", default=str(ROOT / "outputs"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")

    text = Path(args.data).read_text()
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    print(f"corpus: {len(text):,} chars, vocab: {tokenizer.vocab_size} unique characters")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_point = int(0.9 * len(ids))
    train_ds = CharDataset(ids[:split_point], args.block_size)
    val_ds = CharDataset(ids[split_point:], args.block_size)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f)

    model = GPT(config).to(device)
    print(f"model: {model.num_parameters():,} parameters (non-embedding)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)

    history = {"iter": [], "train_loss": [], "val_loss": []}
    start = time.time()

    for step in range(args.max_iters):
        lr = lr_at(step, args.max_iters, args.lr, args.warmup_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_ds.get_batch(args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, {"train": train_ds, "val": val_ds}, args.batch_size, device, args.eval_iters)
            elapsed = time.time() - start
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            history["iter"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

    torch.save({"model_state": model.state_dict(), "config": config.__dict__}, out_dir / "checkpoint.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(history["iter"], history["train_loss"], label="train")
        plt.plot(history["iter"], history["val_loss"], label="val")
        plt.xlabel("iteration")
        plt.ylabel("cross-entropy loss")
        plt.title("Quill training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=120)
        print(f"loss curve -> {out_dir / 'loss_curve.png'}")
    except ImportError:
        print("matplotlib not installed, skipping loss curve plot")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()
    sample_text = tokenizer.decode(sample_ids)
    (out_dir / "sample.txt").write_text(sample_text)
    print("\n=== sample generation ===")
    print(sample_text)


if __name__ == "__main__":
    main()
