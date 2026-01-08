import torch
import math
import time

class Trainer:
    def __init__(self, model, batcher, learning_rate=6e-4):
        self.model = model
        self.batcher = batcher
        # We use a higher base LR because we will decay it later
        self.learning_rate = learning_rate 
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
        self.best_loss = float("inf")

    # --- IMPROVEMENT 1: LR Scheduler ---
    def get_lr(self, it, max_iters, warmup_iters=100, min_lr=6e-5):
        # 1. Linear Warmup: during the first 'warmup_iters' steps, increase LR linearly
        if it < warmup_iters:
            return self.learning_rate * it / warmup_iters
        
        # 2. If we exceed max_iters, stay at the minimum LR
        if it > max_iters:
            return min_lr
        
        # 3. Cosine Decay: decrease LR following a cosine curve
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coefficient between 0 and 1
        
        return min_lr + coeff * (self.learning_rate - min_lr)

    # --- IMPROVEMENT 2: More Reliable Loss Estimation ---
    @torch.no_grad()
    def estimate_loss(self, eval_iters=50, batch_size=8, block_size=256):
        """
        Computes the average loss over multiple batches without backprop.
        Much more reliable than the loss from a single training step.
        """
        self.model.eval()  # Switch to evaluation mode (disables dropout)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = self.batcher.get_batch(batch_size, block_size)
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        self.model.train()  # Switch back to training mode
        return losses.mean()

    def train(self, max_iters=5000, batch_size=16, block_size=256, grad_accum_steps=4):
        self.model.train()
        print(f"\n--- TRAINING STARTED for {max_iters} steps ---")
        start_time = time.time()
        
        for step in range(max_iters):
            
            # 1. Update the Learning Rate for this step
            lr = self.get_lr(step, max_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # 2. Forward & Backward (with Gradient Accumulation)
            # Note: we zero gradients BEFORE accumulation
            self.optimizer.zero_grad(set_to_none=True) 
            
            for _ in range(grad_accum_steps):
                X, Y = self.batcher.get_batch(batch_size, block_size)
                _, loss = self.model(X, Y)
                # Normalize loss because gradients will be accumulated
                loss = loss / grad_accum_steps 
                loss.backward()
            
            # 3. Gradient Clipping (stability) and Optimizer Step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 4. Logging and Checkpointing (every 100 steps)
            if step % 100 == 0:
                # Estimate loss in a robust way
                val_loss = self.estimate_loss(eval_iters=10, batch_size=batch_size, block_size=block_size)
                dt = time.time() - start_time
                print(f"Step {step}/{max_iters} | LR: {lr:.5f} | Val Loss: {val_loss:.4f} | Time: {dt:.2f}s")
                start_time = time.time()  # Reset timer

                # Save only if this is the best loss so far
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model_bible.pt")

        print(f"Training Completed. Best Loss: {self.best_loss:.4f}")

    def save_checkpoint(self, filename):
        # Save not only the weights, but also the config if possible
        torch.save(self.model.state_dict(), filename)
