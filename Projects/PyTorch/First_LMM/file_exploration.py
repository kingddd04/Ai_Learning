import os
import tiktoken

print("Current working directory:", os.getcwd())

with open("divina_commedia.txt", "r", encoding="utf-8") as file:
    content = file.read()

print("Number of characters in the file:", len(content))

# Choose encoding for your model (e.g., "cl100k_base" for GPT-4/3.5)
encoding = tiktoken.get_encoding("cl100k_base")

text = "Hello Davide, this is a test sentence."
tokens = encoding.encode(content)

print("Number of tokens:", len(tokens))