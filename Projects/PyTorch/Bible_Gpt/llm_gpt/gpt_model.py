import torch
import torch.nn as nn
from torch.nn import functional as F
from .learned_positional_encoding import LearnedPositionalEncoding
from .transformer_block import TransformerBlock


class GptModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Core Transformer components:
        # - embeddings: token + positional embeddings
        # - h: list of Transformer blocks
        # - ln_f: final LayerNorm
        self.transformer = nn.ModuleDict(dict(
            embeddings = LearnedPositionalEncoding(config),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language modeling head: projects embeddings to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding weights = output projection weights
        self.transformer.embeddings.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Standard GPT-style initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx shape: (Batch, Time)
        # 1. Token + positional embeddings
        x = self.transformer.embeddings(idx)
        
        # 2. Pass through all Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 3. Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # 4. If training: compute logits for all positions and compute loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        else:
            # If generating: only compute logits for the last time step
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive text generation:
        - idx: initial token sequence
        - max_new_tokens: how many tokens to generate
        - temperature: sampling temperature
        - top_k: restrict sampling to top-k most likely tokens
        """
        for _ in range(max_new_tokens):

            # Crop context to block_size if needed
            if idx.size(1) <= self.config.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.block_size:]

            # Forward pass to get logits for the last position
            logits, _ = self(idx_cond)

            # Select logits of the last time step and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
