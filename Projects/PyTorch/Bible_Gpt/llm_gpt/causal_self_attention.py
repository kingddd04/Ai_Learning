import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Ensure embedding dimension is divisible across attention heads
        assert config.n_embd % config.n_head == 0

        # Linear layer produces Q, K, V concatenated: shape (..., 3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection after attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropouts for attention weights and residual output
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Store configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Check if PyTorch Flash Attention is available
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # If Flash Attention is not available, create a causal mask manually
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time, Channels

        # Compute Q, K, V and split into separate tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # --- Flash Attention path (fast, optimized) ---
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )

        # --- Manual attention path (fallback) ---
        else:
            # Compute raw attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Apply causal mask: prevent attending to future tokens
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            # Softmax over last dimension (attention weights)
            att = F.softmax(att, dim=-1)

            # Dropout on attention weights
            att = self.attn_dropout(att)

            # Weighted sum of values
            y = att @ v

        # Recombine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection + dropout
        y = self.resid_dropout(self.c_proj(y))

        return y
