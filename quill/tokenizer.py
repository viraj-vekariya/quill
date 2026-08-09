import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer: vocab is just the unique characters seen
    in the training corpus. No subword merges, no external vocab file to
    download — the whole tokenizer is reconstructible from one text file.
    """

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}))

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(data["chars"])
