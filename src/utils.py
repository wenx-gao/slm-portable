import yaml
import torch
import random
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, step, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
