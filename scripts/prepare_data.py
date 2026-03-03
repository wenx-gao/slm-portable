import os
import numpy as np
from transformers import GPT2Tokenizer
import argparse

def tokenize_data(input_file, output_file):
    # Load the standard GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please place your text file in data/raw/")
        return

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    print(f"Tokenizing {len(data)} characters...")
    
    # In Hugging Face, we use .encode()
    # add_special_tokens=False keeps it "ordinary"
    ids = tokenizer.encode(data, add_special_tokens=False)
    
    # Convert to uint16 to save disk space (GPT2 vocab is ~50k, fits in 65k)
    ids = np.array(ids, dtype=np.uint16)
    
    print(f"Saving {len(ids)} tokens to {output_file}...")
    ids.tofile(output_file)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.txt")
    parser.add_argument("--output", default="data/processed/train.bin")
    args = parser.parse_args()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    tokenize_data(args.input, args.output)
