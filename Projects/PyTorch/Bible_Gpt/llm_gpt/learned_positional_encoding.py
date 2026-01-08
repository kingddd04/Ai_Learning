import torch
from torch import nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Token embedding: maps token IDs to embedding vectors
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embedding: one learned vector per position (0 ... block_size-1)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Dropout applied after adding token + positional embeddings
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        # idx shape: (Batch, Time)
        device = idx.device
        _, t = idx.size()

        # Create positional indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Compute embeddings
        tok_emb = self.wte(idx)   # Token embeddings
        pos_emb = self.wpe(pos)   # Positional embeddings

        # Add token + positional embeddings, then apply dropout
        return self.drop(tok_emb + pos_emb)
