import json
import tiktoken

class Tokenizer:
    """
    Industrial-quality tokenizer using OpenAI's tiktoken.
    Provides tokenization and detokenization compatible with GPT models.
    """
    def __init__(self, model_name: str = "gpt-2"):
        # Load the encoding for the chosen model
        self.enc = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str) -> list[int]:
        """
        Convert a text string into a list of integer tokens.
        """
        return self.enc.encode(text)

    def detokenize(self, token_list: list[int]) -> str:
        """
        Convert a list of tokens back into a text string.
        """
        return self.enc.decode(token_list)

    @staticmethod
    def save_tokenization(tokens:list[int], filepath: str):
        """
        Takes tokenized tokens an writes them
        into a JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)
        return tokens

    @staticmethod
    def load_tokenization( filename: str) -> list[int]:
        """ Load a list of tokens from a JSON file. """ 
        with open(filename, "r", encoding="utf-8") as f: 
            tokens = json.load(f) 
        
        return tokens