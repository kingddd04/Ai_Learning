import torch
import tiktoken
import time
import os
import json
from gpt import GPT, GPTConfig 

# ==========================================
# 1. GESTIONE HARDWARE
# ==========================================
class DeviceManager:
    def __init__(self, seed=1337):
        torch.manual_seed(seed)
        self.device = self._detect_device()
        print(f"Sto usando il device: {self.device}")

    def _detect_device(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul_allow_tf32 = True 
            torch.backends.cudnn.allow_tf32 = True
            return 'cuda'
        else:
            return 'cpu'

    def get_device(self):
        return self.device


# ==========================================
# 2. GESTIONE FILE (INPUT/OUTPUT)
# ==========================================
class TextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_text(self):
        """Legge il file di testo. Se non esiste, ne crea uno dummy."""
        if not os.path.exists(self.file_path):
            print(f"ATTENZIONE: File {self.file_path} non trovato. Creo un file dummy.")
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("In principio Dio creò il cielo e la terra. " * 500)
        
        print(f"Lettura file '{self.file_path}'...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()


# ==========================================
# 3. GESTIONE TOKENIZZAZIONE
# ==========================================
class GPTTokenizer:
    def __init__(self, encoding_name="gpt2"):
        print(f"Inizializzazione Tokenizer ({encoding_name})...")
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text):
        """Stringa -> Lista di Interi"""
        return self.enc.encode(text)

    def decode(self, token_ids):
        """Lista di Interi -> Stringa"""
        # Se riceve un tensore, lo converte in lista
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.enc.decode(token_ids)
    
    def to_tensor(self, text, dtype=torch.long):
        """Utility: Stringa -> Tensore PyTorch"""
        encoded = self.encode(text)
        return torch.tensor(encoded, dtype=dtype)
    
    @staticmethod
    def save_tokenization(tokens: list[int], filepath: str):
        """
        Takes tokenized tokens and writes them into a JSON file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2)
        return tokens

    @staticmethod
    def load_tokenization(filename: str) -> list[int]:
        """ Load a list of tokens from a JSON file. """ 
        with open(filename, "r", encoding="utf-8") as f: 
            tokens = json.load(f) 
        return tokens


# ==========================================
# 4. GESTIONE BATCHING (DATASET)
# ==========================================
class Batcher:
    def __init__(self, data_tensor, device):
        self.data = data_tensor
        self.device = device
        self.n_tokens = len(data_tensor)
        print(f"Batcher inizializzato su {self.n_tokens} token.")

    def get_batch(self, batch_size=16, block_size=256):
        """Estrae un mini-batch casuale (X, Y)"""
        # Generiamo indici casuali validi
        ix = torch.randint(self.n_tokens - block_size, (batch_size,))
        
        # Creiamo le sequenze (X) e i target shiftati di 1 (Y)
        x = torch.stack([self.data[i : i+block_size] for i in ix])
        y = torch.stack([self.data[i+1 : i+block_size+1] for i in ix])
        
        # Spostiamo su GPU/CPU
        return x.to(self.device), y.to(self.device)


# ==========================================
# 5. GESTIONE TRAINING LOOP
# ==========================================
class Trainer:
    def __init__(self, model, batcher, learning_rate=3e-4):
        self.model = model
        self.batcher = batcher
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train(self, steps=1000, batch_size=8, block_size=256, grad_accum_steps=4):
        self.model.train()
        start_time = time.time()
        
        print(f"\n--- INIZIO TRAINING ({steps} steps) ---")
        
        for step in range(steps):
            self.optimizer.zero_grad()
            loss_accum = 0.0
            
            # Gradient Accumulation Loop
            for _ in range(grad_accum_steps):
                # Chiediamo il batch alla classe Batcher
                X, Y = self.batcher.get_batch(batch_size, block_size)
                
                logits, loss = self.model(X, Y) # Le parentesi Chiamano Forward
                
                loss = loss / grad_accum_steps
                loss_accum += loss.item()
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if step % 50 == 0:
                dt = time.time() - start_time
                print(f"Step {step} | Loss: {loss_accum:.4f} | Tempo: {dt:.2f}s")
                start_time = time.time()
                
        print("Training finito.")

    def save_checkpoint(self, filename="modello_bibbia.pt"):
        torch.save(self.model.state_dict(), filename)
        print(f"Modello salvato in: {filename}")


# ==========================================
# 6. MAIN (ASSEMBLAGGIO)
# ==========================================
if __name__ == "__main__":
    # 1. Setup Hardware
    device_mgr = DeviceManager()
    device = device_mgr.get_device()

    # 2. Pipeline Dati: Loader -> Tokenizer -> Batcher
    loader = TextLoader('input.txt')
    raw_text = loader.load_text()
    
    tokenizer = GPTTokenizer()
    # Creiamo il tensore gigante una volta sola
    data_tensor = tokenizer.to_tensor(raw_text)
    
    batcher = Batcher(data_tensor, device)

    # 3. Configurazione Modello
    config = GPTConfig(
        vocab_size=50257, 
        block_size=256,    
        n_layer=4,         
        n_head=4, 
        n_embd=256, 
        dropout=0.1
    )
    
    # 4. Creazione Modello
    model = GPT(config)
    model.to(device)
    print(f"Modello creato con {sum(p.numel() for p in model.parameters())/1e6:.2f}M parametri")

    # 5. Training
    # Il Trainer ora riceve il Batcher, non più un generico 'DataManager'
    trainer = Trainer(model, batcher, learning_rate=3e-4)
    
    trainer.train(
        steps=200, 
        batch_size=8, 
        block_size=config.block_size, 
        grad_accum_steps=4
    )
    
    trainer.save_checkpoint()

    # 6. Generazione (Test)
    print("\n--- GENERAZIONE TESTO ---")
    model.eval()
    
    start_str = "In principio"
    # Usiamo il Tokenizer per preparare l'input
    x_input = tokenizer.to_tensor(start_str).unsqueeze(0).to(device)

    # Generiamo
    y_output = model.generate(x_input, max_new_tokens=50, temperature=0.8)
    
    # Usiamo il Tokenizer per leggere l'output
    print(tokenizer.decode(y_output[0]))
