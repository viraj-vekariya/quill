from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
