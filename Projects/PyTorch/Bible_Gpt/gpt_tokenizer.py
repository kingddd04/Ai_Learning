import json
import tiktoken
import torch


class GptTokenizer:
    def __init__(self, encoding_name="gpt2"):
        print(f"Tokenizer in use: ({encoding_name})")
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text):
        """String -> List of integers (token IDs)."""
        return self.enc.encode(text)

    def decode(self, token_ids):
        """List of integers (or tensor) -> String."""
        # Convert tensor to list if necessary
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.enc.decode(token_ids)

    def to_tensor(self, text, dtype=torch.long):
        """Utility: String -> PyTorch tensor of token IDs."""
        encoded = self.encode(text)
        return torch.tensor(encoded, dtype=dtype)

    @staticmethod
    def save_tokenization(tokens: list[int], filepath: str):
        """
        Save a list of token IDs into a JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)
        return tokens
