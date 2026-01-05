from text_reader import Text_Reader
from text_preprocesser import Text_Preprocesser
from tokenizer import Tokenizer

"""
### Preprocessing Testo
text = Text_Reader.extract_text("C:\\Users\\david_bbnm\\Downloads\\bible clean.txt")
polished_text = Text_Preprocesser.text_preprocess(text)
Text_Preprocesser.save_preprocessed_text("C:\\Users\\david_bbnm\\Downloads\\bible cleaned.txt", polished_text)
"""
"""
### Tokenizzazione e salvataggio 
text = Text_Reader.extract_text("C:\\Users\\david_bbnm\\Downloads\\bible cleaned preprocessed.txt")
tokenizer = Tokenizer()
tokenized_txt = tokenizer.tokenize(text)
Tokenizer.save_tokenization(tokenized_txt, "C:\\Users\\david_bbnm\\Downloads\\bible tokenized.json")
"""

