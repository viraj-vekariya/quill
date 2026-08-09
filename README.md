# quill — a GPT-style transformer built from scratch

A decoder-only transformer language model where the attention mechanism and
LayerNorm are written out at the tensor-op level — not `nn.MultiheadAttention`,
not `F.scaled_dot_product_attention` — trained end-to-end on character-level
Shakespeare. PyTorch still handles autograd (reimplementing backprop by hand
wouldn't teach anything the forward pass doesn't already), but nothing about
*how attention computes its output* is hidden behind a library call.

## Why from scratch

It's easy to get a transformer training by calling `nn.TransformerEncoderLayer`
without ever understanding what happens inside it. This project exists to
prove the opposite: every matmul in `softmax(QK^T / sqrt(d)) V`, the causal
mask, multi-head reshaping, and LayerNorm's mean/variance normalization are
all written by hand in `quill/`, and tested against the properties that make
them correct — not just "it runs."

## Architecture

```
tokens ─▶ token_emb + pos_emb ─▶ [ pre-LN block ] x n_layer ─▶ LayerNorm ─▶ lm_head ─▶ logits
                                       │
                         x + attn(LN(x))   — manual multi-head causal self-attention
                         x + mlp(LN(x))    — 4x-expansion GELU feedforward
```

- **`quill/attention.py`** — `CausalSelfAttention`: manual QKV projection, per-head
  reshape, `(q @ k.T) / sqrt(head_dim)`, causal mask, softmax, `@ v`. No fused kernel.
- **`quill/layers.py`** — `LayerNorm`: mean/variance normalize + learned affine, by hand.
- **`quill/block.py`** — pre-LN transformer block (GPT-2-onward convention: normalize
  *before* the sub-layer, not after — keeps gradients well-behaved at depth).
- **`quill/model.py`** — `GPT`: embeddings, block stack, weight-tied output head,
  GPT-2-style scaled residual-projection init, `generate()` with temperature/top-k sampling.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q      # 11 tests: causality, LayerNorm correctness, overfit sanity check
python train.py                 # ~14 min on Apple Silicon MPS, 3000 iterations
python generate.py --prompt "ROMEO:"
```

(`./run.sh` / `run.bat` do venv + deps + tests + training in one step.)

## Tests (`make` not required — `python -m pytest tests/`)

The tests prove the "from scratch" claims rather than just checking the code runs:

| Test | What it proves |
|---|---|
| `test_matches_pytorch_layernorm_numerically` | The hand-written LayerNorm produces numerically identical output to `nn.LayerNorm` given the same weights — not just "runs," but *correct*. |
| `test_causal_mask_blocks_future_positions` | Perturbing only the **last** token's input leaves every earlier position's attention output byte-for-byte unchanged — the defining property of causal attention, checked directly rather than trusted from the mask math. |
| `test_attention_weights_are_causal_and_normalized` | Inspects the actual attention weight matrix: zero mass on future positions, each row sums to 1. |
| `test_model_can_overfit_a_tiny_repeated_sequence` | The single most useful sanity check in ML: a model that can't memorize an 8-token pattern in 200 steps has a bug in forward/backward, no amount of real data will fix it. |
| `test_lm_head_is_tied_to_token_embedding` | Weight tying is actually wired up (`is`, not just equal-valued). |
| `test_generate_appends_exactly_max_new_tokens` | Autoregressive generation produces exactly the requested length, tokens in-vocab. |
| + gradient-flow and shape tests | Every parameter in attention receives a gradient; output shapes match input shapes throughout. |

## Training run (this repo's actual numbers)

3000 iterations, batch size 64, block size 128, 4 layers / 4 heads / 128 embedding
dim (~800K non-embedding parameters), AdamW with cosine LR decay + warmup, on
Apple Silicon MPS:

```
step     0 | train 4.2052 | val 4.2036 | lr 1.50e-06
step   250 | train 2.4879 | val 2.4878 | lr 3.00e-04
step  1000 | train 1.9506 | val 2.0291 | lr 2.49e-04
step  2000 | train 1.7016 | val 1.8518 | lr 1.06e-04
step  2999 | train 1.6342 | val 1.7986 | lr 3.00e-05
```

![training loss curve](outputs/loss_curve.png)

Sample generation at the end of this run (temperature 0.8, top-k 40):

```
CHINRY GRITHARD II:
Comest in of love he croince the the art,
Have she ince of the have of the, sir, and that yet
mreas?

PALINA:
See to may couselve you wilt the duke is on be come.

HASTINIUS Edward; that I do desce, mo them, the put this cous,
Or that with he may hreart stand feear they stand;
And and yet but serving of the quin's the king hoave fair,
```

**Honest read of this output:** it has correctly learned Shakespeare's *surface
structure* — character-name headers in caps followed by a colon, verse-like
line breaks, archaic diction ("thou," "hath"-adjacent forms), blank lines
between speaker turns — but the actual words are frequently non-words or
garbled ("croince," "hreart"). That's expected and consistent with published
char-level results at this scale: this is an ~800K-parameter model trained for
3000 steps on 1MB of text, not a production language model. nanoGPT's own
char-Shakespeare configs use a larger model and 5000+ iterations to bring val
loss down toward ~1.4-1.5, where output starts forming mostly-real words.
Val loss of 1.80 here is in the right range for this much smaller/shorter
training budget, and the loss curve (linked above) is still trending down at
step 2999 — this is an undertrained-by-design demo, not a converged model.

## Honest caveats

- **Character-level, not subword.** No BPE/tokenizer training — the vocabulary
  is just the corpus's unique characters (65 for Shakespeare). This keeps the
  whole tokenizer reconstructible from one text file, at the cost of needing
  more tokens (and more training) to reach fluent output than a subword model
  would.
- **Small by design.** 4 layers, 128-dim embeddings, ~800K parameters — sized to
  train on a laptop in minutes, not to produce publication-quality text. The
  architecture (attention, LayerNorm, blocks) is what's "from scratch," not
  the scale.
- **No KV-cache in `generate()`.** Each new token reprocesses the full context
  window through every layer — fine at `block_size=128`, would need a KV-cache
  to be usable at production context lengths.
- **Dataset:** `data/tinyshakespeare.txt` is the standard "Tiny Shakespeare"
  corpus (public-domain text, commonly redistributed for exactly this kind of
  char-level LM tutorial/benchmark — e.g. Karpathy's char-rnn and nanoGPT both
  bundle the same file).

## Layout

```
quill/
  config.py      GPTConfig dataclass
  tokenizer.py   character-level encode/decode, vocab saved to json
  layers.py      hand-written LayerNorm
  attention.py   hand-written multi-head causal self-attention
  block.py       pre-LN transformer block (attention + MLP + residuals)
  model.py       GPT: embeddings, block stack, generate()
  dataset.py     random contiguous-window batching
tests/           causality, LayerNorm-equivalence, overfit sanity check, generation
data/            tinyshakespeare.txt
train.py         full training loop, loss curve, checkpoint
generate.py      sample from a saved checkpoint
```
