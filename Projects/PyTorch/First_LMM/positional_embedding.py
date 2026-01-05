import torch
import torch.nn as nn

class GPTInputEmbeddings(nn.Module):
    def init(self, config):
        super().init()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout) # dropout needed to limit overfitting

    def forward(self, idx):
        device = idx.device
        B, T = idx.shape # B=Batch size, T=seq_len
        tok_emb = self.token_embedding(idx) 
        pos_idxs = torch.arange(T, device=device)
        pos_emb = self.position_embedding(pos_idxs) # obtining positional vectors
        x = tok_emb + pos_emb
        x = self.dropout(x) # Dropout

        return x