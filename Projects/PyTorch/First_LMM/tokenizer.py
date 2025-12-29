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
