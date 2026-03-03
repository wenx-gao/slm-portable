import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['model']['num_heads']
        self.n_embd = config['model']['hidden_size']
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = config['model']['dropout']

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Use Flash Attention (Efficient)
        y = F.scaled_dot_product_attention(
            q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            is_causal=True, dropout_p=self.dropout if self.training else 0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['model']['hidden_size'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['model']['hidden_size'])
        self.mlp = nn.Sequential(
            nn.Linear(config['model']['hidden_size'], 4 * config['model']['hidden_size']),
            nn.GELU(),
            nn.Linear(4 * config['model']['hidden_size'], config['model']['hidden_size']),
            nn.Dropout(config['model']['dropout']),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SmallLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['model']['vocab_size'], config['model']['hidden_size']),
            wpe = nn.Embedding(config['model']['context_window'], config['model']['hidden_size']),
            h = nn.ModuleList([Block(config) for _ in range(config['model']['num_layers'])]),
            ln_f = nn.LayerNorm(config['model']['hidden_size']),
        ))
        self.lm_head = nn.Linear(config['model']['hidden_size'], config['model']['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
