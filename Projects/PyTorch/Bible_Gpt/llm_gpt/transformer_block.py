import torch.nn as nn
from .causal_self_attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # LayerNorm before attention (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(config.n_embd)

        # Causal self-attention module (masked attention)
        self.attn = CausalSelfAttention(config)

        # LayerNorm before MLP
        self.ln2 = nn.LayerNorm(config.n_embd)

        # Feed-forward network (MLP block)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connection: x + Attention(LN(x))
        x = x + self.attn(self.ln1(x))

        # Residual connection: x + MLP(LN(x))
        x = x + self.mlp(self.ln2(x))

        return x
