import torch


class CharDataset:
    """Wraps a 1D tensor of token ids and hands out random contiguous
    windows for next-token-prediction training: x = tokens[i:i+block_size],
    y = tokens[i+1:i+block_size+1] (y is x shifted one position right).
    """

    def __init__(self, data_ids: torch.Tensor, block_size: int):
        assert len(data_ids) > block_size, "dataset shorter than one block"
        self.data = data_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)
