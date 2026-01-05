import torch

class DeviceManager:
    def init(self, seed: int = 42):
        self.seed = seed
        self.device = None
        self.set_seed()
        self.detect_hardware()
        
    def set_seed(self):
        """Set the seed for reproducibility"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            
    def detect_hardware(self):
        """Detects GPU Nvidia, Mac Apple Silicon or CPU"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.enable_tf32() # Attiva TF32 se siamo su Nvidia
            print(f"Device: CUDA (Nvidia) - TF32 Enabled: {torch.backends.cudnn.allow_tf32}")
            
        elif torch.backends.mps.is_available():
            # Supporto per Mac M1/M2/M3
            self.device = torch.device("mps")
            print("Device: MPS (Apple Silicon)")
            
        else:
            self.device = torch.device("cpu")
            print("Device: CPU")

        return self.device

    def enable_tf32(self):
        """Boost performance su GPU Ampere+ (RTX 30xx/40xx)"""
        # Matmul precision 'high' o 'medium' sacrificano pochissima precisione per molta velocità
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True

    def getdevice(self):
        return self.device
