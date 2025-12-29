from text_reader import Text_Reader
from text_preprocesser import Text_Preprocesser
from tokenizer import Tokenizer

text = Text_Reader.extract_text("C:/Users/david_bbnm/OneDrive/Documents/GitHub/Ai_Learning/Projects/PyTorch/First_LMM/promessi_sposi-cap1.txt")
polished_text = Text_Preprocesser.text_preprocess(text)
tokenizer = Tokenizer()
tokenizer.tokenize_training_text(polished_text)
tokens = tokenizer.tokenize(polished_text)
print(tokens[:20])
text = tokenizer.detokenize(tokens[:20])
print(text)

