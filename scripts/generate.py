import os
import sys
import torch
from transformers import GPT2Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SmallLM
from src.utils import load_config

def generate(prompt, max_new_tokens=50, temperature=0.8):
    config = load_config('configs/base_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = SmallLM(config).to(device)
    # Load latest checkpoint if available
    ckpt_path = os.path.join(config['paths']['checkpoint_dir'], "latest.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Encode prompt
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generation Loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the window
            x_cond = x if x.size(1) <= config['model']['context_window'] else x[:, -config['model']['context_window']:]
            
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    return tokenizer.decode(x[0].tolist())

if __name__ == "__main__":
    user_prompt = "The future of AI is"
    print(f"Prompt: {user_prompt}")
    output = generate(user_prompt)
    print(f"Generated: {output}")
