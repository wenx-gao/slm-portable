import torch
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self, model, optimizer, config, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scaler = GradScaler() # For Mixed Precision
        self.grad_accum_steps = config['training'].get('grad_accum_steps', 1)

    def train_step(self, x, y, step_idx):
        x, y = x.to(self.device), y.to(self.device)

        # Mixed Precision context
        with autocast():
            logits, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps

        # Backward pass with Scaler
        self.scaler.scale(loss).backward()

        # Only update weights every 'grad_accum_steps'
        if (step_idx + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True) # Saves memory

        return loss.item() * self.grad_accum_steps
