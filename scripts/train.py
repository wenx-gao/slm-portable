import os
import sys
import torch

# Add 'src' to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SmallLM
from src.data_loader import get_dataloader
from src.trainer import Trainer
from src.utils import load_config, set_seed, save_checkpoint

def main():
    # 1. Setup
    # To use RTX 500 Ada specific config, pass it as an argument
    config = load_config('configs/base_config.yaml')
    # Optional: Merge with rtx_500_small.yaml if needed
    
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    print(f"Starting training on {device} (RTX 500 Ada mode)")

    # 2. Data
    train_loader = get_dataloader(config['paths']['train_data'], config)
    print(f"Total steps per epoch: {len(train_loader)}")

    # 3. Model & Optimizer
    model = SmallLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        fused=True if torch.cuda.is_available() else False # Fused is faster on Ada
    )

    # 4. Trainer
    trainer = Trainer(model, optimizer, config, device)

    # 5. Training Loop
    model.train()
    step = 0
    for epoch in range(config['training']['epochs']):
        for x, y in train_loader:
            loss = trainer.train_step(x, y, step)
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}")
            
            if step > 0 and step % config['training']['save_interval'] == 0:
                ckpt_path = os.path.join(config['paths']['checkpoint_dir'], f"model_step_{step}.pt")
                save_checkpoint(model, optimizer, step, ckpt_path)
            
            step += 1

if __name__ == "__main__":
    main()
