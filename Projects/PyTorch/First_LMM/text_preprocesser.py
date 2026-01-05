class Text_Preprocesser:
    """
    A utility class for preprocessing raw text before tokenization.
    """

    @staticmethod
    def text_preprocess(text):
        """
        Preprocess a raw text string into a list of cleaned tokens.

        Steps performed:
        - Convert all characters to lowercase.
        - Keep only alphabetic characters and selected punctuation.
        - Surround punctuation with spaces to ensure separation.
        - Split the polished text into a list of tokens
        """
        text = text.lower()
        polished_text = ""
        for char in text:
            if char.isalpha():
                polished_text += char
            if char in " ;,.!?":
                if char == " ":
                    polished_text += char
                    continue
                if char == ",":
                    polished_text += " , "
                    continue
                temp_string = " " + char + " "
                polished_text += temp_string
        polished_text = polished_text.replace("  ", " ")
        return polished_text

    @staticmethod
    def save_preprocessed_text(filepath, text):
        """
        Save preprocessed text to a file.
        This function writes the given text string into a file at the specified path.
        """
        with open(filepath, "w", encoding="utf8") as txt:
            txt.write(text)
