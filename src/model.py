"""Transformer-based Mini-GPT model implemented with PyTorch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention layer with a causal mask."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        """Initialize projection layers and head geometry for attention."""
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # One linear layer produces query, key, and value tensors.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention to hidden states.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, embed_dim].
        Returns:
            Tensor of same shape after attention and output projection.
        """
        batch_size, seq_len, embed_dim = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask blocks access to future positions.
        attention_scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.output_proj(context)


class TransformerBlock(nn.Module):
    """Single GPT-style decoder block: attention + feed-forward network."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        """Create normalization, attention, and MLP sub-layers."""
        super().__init__()
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attention = CausalSelfAttention(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)

        self.pre_mlp_norm = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = embed_dim * mlp_ratio
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply residual attention and MLP updates."""
        hidden_states = hidden_states + self.self_attention(self.pre_attn_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(self.pre_mlp_norm(hidden_states))
        return hidden_states


class MiniGPT(nn.Module):
    """Decoder-only transformer language model for next-token prediction."""

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
        """Create token/position embeddings, transformer blocks, and LM head."""
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute token logits and optional cross-entropy loss."""
        batch_size, seq_len = input_ids.shape
        del batch_size  # Name kept for readability in tensor-shape documentation.

        position_ids = torch.arange(seq_len, device=input_ids.device)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)[None, :, :]
        hidden_states = self.embedding_dropout(hidden_states)

        for transformer_block in self.blocks:
            hidden_states = transformer_block(hidden_states)

        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        return logits, loss
