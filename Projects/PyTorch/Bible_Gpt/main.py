from device_manager import DeviceManager
from file_reader import File_Reader
from gpt_tokenizer import GptTokenizer
from batcher import Batcher
from llm_gpt import GPTConfig, GptModel
from trainer import Trainer


if __name__ == "__main__":
    # 1. Hardware Setup
    device_mgr = DeviceManager()
    device = device_mgr.get_device()

    # 2. Data Pipeline: Loader -> Tokenizer -> Batcher
    file_reader = File_Reader()
    raw_text = file_reader.extract_text("Bible.txt")
    
    tokenizer = GptTokenizer()
    # Create the large tensor only once
    data_tensor = tokenizer.to_tensor(raw_text)
    
    batcher = Batcher(data_tensor, device)

    # 3. Model Configuration
    config = GPTConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1
    )
    
    # 4. Model Creation
    model = GptModel(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params / 1_000_000:.2f} Million parameters")

    # 5. Training
    # The Trainer now receives the Batcher instead of a generic DataManager
    trainer = Trainer(model, batcher, learning_rate=3e-4)
    
    trainer.train(
        max_iters=2026,
        batch_size=8,
        block_size=config.block_size,
        grad_accum_steps=4
    )
    
    trainer.save_checkpoint()

    # 6. Text Generation (Test)
    print("\n--- Text Generation ---")
    model.eval()
    
    start_str = "jesus is"
    # Use the tokenizer to prepare the input
    x_input = tokenizer.to_tensor(start_str).unsqueeze(0).to(device)

    # Generate text
    y_output = model.generate(
        x_input,
        max_new_tokens=50,
        temperature=0.8
    )
    
    # Decode the output tokens
    print(tokenizer.decode(y_output[0]))
