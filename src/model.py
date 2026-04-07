"""Transformer-based Mini-GPT model using built-in PyTorch Transformer modules."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    """Decoder-style language model built from PyTorch TransformerEncoder blocks.

    This model uses a causal attention mask so each token can only attend to
    previous tokens, matching next-token prediction requirements.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
    ) -> None:
        """Initialize embeddings, Transformer stack, and output LM head.

        Args:
            vocab_size: Token vocabulary size.
            block_size: Maximum sequence length.
            embed_dim: Hidden size / embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers.
            dropout: Dropout probability.
            mlp_ratio: Expansion ratio for feed-forward hidden dimension.
        """
        super().__init__()

        self.block_size = block_size

        # Token and position embeddings.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # PyTorch built-in Transformer layer and stack.
        ff_hidden_dim = embed_dim * mlp_ratio
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive attention.

        The mask is True where attention should be blocked (future positions).
        """
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optional cross-entropy loss for language modeling."""
        _, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}.")

        position_ids = torch.arange(seq_len, device=input_ids.device)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)[None, :, :]
        hidden_states = self.embedding_dropout(hidden_states)

        causal_mask = self._build_causal_mask(seq_len=seq_len, device=input_ids.device)
        hidden_states = self.transformer(hidden_states, mask=causal_mask)

        logits = self.lm_head(hidden_states)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        return logits, loss
