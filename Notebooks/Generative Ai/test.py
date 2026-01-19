from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

prompt = "siege guns were bombing the city walls"

input_ids = tokenizer.encode(prompt, return_tensors="pt")
# Generate completion
output = model.generate(
    input_ids,              # Tokenized input for the model
    max_length=90,         # max lenght in token of the total text
    do_sample=True,         # the next token is chosen randomly not deterministically, but weighted by probability
    temperature=0.9,        # Controls the creativity of the generation 0-1
    top_p=0.90,              # Instead of considering all tokens, keep only the smallest set of tokens whose cumulative probability reaches p
    repetition_penalty= 1.2 # more than 1 discourages repetition 
)

# Decode the completion
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# Print the generated text
print(output_text)
model.save_pretrained("./my_model")
