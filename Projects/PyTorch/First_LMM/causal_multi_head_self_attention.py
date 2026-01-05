import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadSelfAttention(nn.Module):
    def init(self, config):
        super().init()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # FLASH ATTENTION CHECK
        # Controlliamo se PyTorch supporta la Flash Attention (versione 2.0+)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash:
            print("WARNING: Usando Attention lenta (Manuale). Aggiorna PyTorch per Flash Attention.")
            # Fallback: Maschera manuale (bias)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # 1. Calcolo Q, K, V
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape per le teste: (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 2. CALCOLO ATTENZIONE (Il bivio)
        if self.flash:
            # --- STRADA VELOCE (Flash Attention) ---
            # is_causal=True fa automaticamente la maschera triangolare!
            # Non serve creare buffer o matrici enormi di -inf.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                               dropout_p=self.dropout if self.training else 0, 
                                               is_causal=True)
        else:
            # --- STRADA LENTA (Manuale, didattica) ---
            # (Q @ K) / sqrt(dim)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Applica maschera
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # Softmax
            att = F.softmax(att, dim=-1)
            # Dropout
            att = F.dropout(att, p=self.dropout, training=self.training)
            # (Att @ V)
            y = att @ v

        # 3. Ricomposizione
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 4. Proiezione finale
        y = self.c_proj(y)
        return y