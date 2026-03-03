import torch
import numpy as np
import os

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, context_window):
        # Load pre-tokenized data as a memory-mapped file
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.context_window = context_window

    def __len__(self):
        return len(self.data) - self.context_window - 1

    def __getitem__(self, idx):
        # Grab a chunk of data
        chunk = self.data[idx : idx + self.context_window + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def get_dataloader(data_path, config, shuffle=True):
    dataset = TokenDataset(data_path, config['model']['context_window'])
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=shuffle,
        pin_memory=True # Faster transfer to GPU
    )
