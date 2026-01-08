import torch
from gpt_tokenizer import GptTokenizer
from llm_gpt import GptModel, GPTConfig
from gpt_tokenizer import GptTokenizer   

def load_model(checkpoint_path):
    config = GPTConfig(
        vocab_size=50257, 
        block_size=256,    
        n_layer=4,         
        n_head=4, 
        n_embd=256, 
        dropout=0.1
    )
    
    # Model Creation
    model = GptModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
):
    model.eval()

    # Encode prompt
    x = tokenizer.to_tensor(prompt).unsqueeze(0)

    for _ in range(max_new_tokens):

        # Respect block size limit
        if x.size(1) > model.config.block_size:
            x_cond = x[:, -model.config.block_size:]
        else:
            x_cond = x



        # Forward pass
        logits, _ = model(x_cond)

        # Take logits for last token
        logits = logits[:, -1, :]

        # Apply repetition penalty 
        for token in set(x[0].tolist()):
            logits[0, token] /= repetition_penalty

        # Temperature scaling
        logits = logits / max(temperature, 1e-6)

        # --- Top-k filtering ---
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")

        # --- Top-p (nucleus) filtering ---
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_logits[mask] = -float("inf")
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0])


model = load_model("bible_gpt.pt")
tokenizer = GptTokenizer("gpt2")

prompt = "holy spirit as fire"
tokens_to_predict= 33
output = generate(model, tokenizer, prompt, tokens_to_predict , 0.7 , 30, 0.9)
print("Predicted next : "+str(tokens_to_predict)+" Tokens\n\n"+output+"\n\n------------\n\n")
