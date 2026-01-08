import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # First linear layer: expands embedding dimension by a factor of 4
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # Non‑linear activation (GELU is standard in GPT architectures)
        self.gelu = nn.GELU()

        # Projection layer: brings the dimension back to the original embedding size
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Fully connected expansion
        x = self.c_fc(x)

        # Non-linear activation
        x = self.gelu(x)

        # Projection back to embedding dimension
        x = self.c_proj(x)

        # Apply dropout
        x = self.dropout(x)

        return x
