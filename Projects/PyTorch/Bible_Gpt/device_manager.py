import torch

class DeviceManager:
    """
    Manages device selection (CPU/GPU) and sets the random seed.
    Parameters
    seed : int
        Seed for reproducibility.
    Methods
    get_device()
        Returns the selected device ("cpu" or "cuda").
    """

    def __init__(self, seed=1337):
        torch.manual_seed(seed)
        self.device = self._detect_device()
        print(f"You are using: {self.device}")

    def _detect_device(self):
        """Return 'cuda' if available, otherwise 'cpu'."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul_allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return "cuda"
        return "cpu"

    def get_device(self):
        """Return the selected device."""
        return self.device
