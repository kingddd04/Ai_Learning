class GPTConfig:
    """
    Config class of gpt model
    """
    def __init__(self, block_size=256, vocab_size=50257, n_layer=6, 
                 n_head=6, n_embd=384, dropout=0.2, bias=True):
        
        # Attributes of the model
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        ### Multi Head Self attention safety check
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"Error: n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_head})")

    def repr(self):
        """
        prints the attributes of our model
        """
        return (f"GPTConfig(block_size={self.block_size}, vocab_size={self.vocab_size}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, "
                f"n_embd={self.n_embd}, dropout={self.dropout})")