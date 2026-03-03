***

# Portable-SLM: Small Language Model for Low-VRAM GPUs

A highly optimized PyTorch implementation of a GPT-style (decoder-only) transformer, specifically designed to be trained on hardware with limited VRAM, such as the **NVIDIA RTX 500 Ada (4GB)**.

This project demonstrates that you don't need a data-center GPU to experiment with LLM architectures. By utilizing modern techniques like **Flash Attention**, **Automatic Mixed Precision (AMP)**, and **Memory Mapping**, this model achieves efficient throughput on mobile workstations.

## 🚀 Features

- **Hardware Optimized:** Specifically tuned for the Ada Lovelace architecture (RTX 500/1000/2000 series).
- **Memory Efficiency:** 
    - **Flash Attention:** Uses `scaled_dot_product_attention` for reduced memory footprint.
    - **Gradient Accumulation:** Simulates larger batch sizes on small VRAM.
    - **Numpy Memmap:** Streams training data from disk to keep RAM usage near zero.
- **Config-Driven:** Fully decoupled architecture using YAML configuration files.
- **Production-Ready Structure:** Professional file architecture suitable for GitHub and collaborative research.

---

## 📂 Project Architecture

```text
small-lm-project/
├── configs/                # Configuration overrides for different hardware
├── data/                   # Dataset storage (Ignored by Git)
├── src/                    # Core library
│   ├── model.py            # Transformer Architecture (Flash Attention)
│   ├── data_loader.py      # Memory-mapped data streaming
│   ├── trainer.py          # AMP & Gradient Accumulation logic
│   └── utils.py            # Helper functions
├── scripts/                # Entry-point execution scripts
│   ├── prepare_data.py     # Tokenization pipeline
│   ├── train.py            # Training orchestration
│   └── generate.py         # Inference/Text generation
└── requirements.txt        # Minimal dependencies
```

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/slm-portable.git
   cd slm-portable
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch numpy pyyaml transformers
   ```

---

## 📖 Usage Guide

### 1. Prepare Data
Place your raw text data in `data/raw/train.txt`. Then, run the tokenizer to create the binary memory-mapped files:
```bash
python scripts/prepare_data.py
```

### 2. Training
The training script is configured to use the RTX 500 Ada profile. It utilizes FP16 mixed precision to maximize the performance of your Tensor Cores.
```bash
python scripts/train.py
```
*Note: You can modify `configs/base_config.yaml` to change the number of layers or hidden dimensions.*

### 3. Generate Text
Once you have a checkpoint in the `checkpoints/` folder, test your model:
```bash
python scripts/generate.py
```

---

## 📉 Hardware Specifics (RTX 500 Ada)

Training a language model on **4GB VRAM** requires careful tuning. This project employs:
1. **Batch Size Control:** Defaulting to a small micro-batch (e.g., 2-4) with high gradient accumulation (e.g., 16-32).
2. **Fused Optimizers:** Using `AdamW(fused=True)` to reduce overhead.
3. **Context Windowing:** Defaulting to 256–512 tokens to prevent OOM (Out of Memory) errors during the attention mechanism.

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! If you find a way to squeeze even more performance out of low-end GPUs, feel free to open a Pull Request.

***
