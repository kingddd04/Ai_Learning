import torch

class Batcher:
    """
    Creates random training batches from a 1D tensor of token IDs.
    Parameters
    data_tensor : torch.Tensor
        Full tokenized dataset (1D tensor).
    device : torch.device or str
        Device where batches will be placed.

    Methods
    get_batch(batch_size=16, block_size=256)
        Returns (X, Y) where:
        - X is a sequence of length `block_size`
        - Y is X shifted by one token
    """

    def __init__(self, data_tensor, device):
        self.data = data_tensor
        self.device = device
        self.n_tokens = len(data_tensor)

    def get_batch(self, batch_size=16, block_size=256):
        """Return a random batch of input/target sequences."""
        ix = torch.randint(self.n_tokens - block_size, (batch_size,))
        x = torch.stack([self.data[i:i+block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)
