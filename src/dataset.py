"""PyTorch dataset and dataloader utilities for autoregressive language-model training."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class TokenBlockDataset(Dataset):
    """Dataset of fixed-length token sequences for next-token prediction.

    Each row in ``sequences`` must have shape ``[block_size + 1]`` so that we can
    split it into input tokens (all but last) and target tokens (all but first).
    """

    def __init__(self, sequences: torch.Tensor):
        """Validate and store token sequence tensor.

        Args:
            sequences: 2D tensor with shape [num_examples, block_size + 1].
        """
        if sequences.ndim != 2:
            raise ValueError("Expected 2D tensor [num_examples, block_size+1].")
        self.sequences = sequences

    def __len__(self) -> int:
        """Return total number of training examples."""
        return self.sequences.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (input_ids, target_ids) pair for autoregressive training."""
        full_sequence = self.sequences[idx]
        input_ids = full_sequence[:-1]
        target_ids = full_sequence[1:]
        return input_ids, target_ids


def create_dataloader(
    sequences: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a standard PyTorch DataLoader from token sequences."""
    token_dataset = TokenBlockDataset(sequences)
    return DataLoader(
        token_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
