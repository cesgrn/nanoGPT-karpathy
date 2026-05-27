# nanoGPT

A minimal implementation of a character-level GPT, built by following Andrej Karpathy's YouTube video [**Let's build GPT: from scratch, in code, spelled out**](https://www.youtube.com/watch?v=kCc8FmEb1nY).

## What is this?

This is a small Transformer-based language model trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. Given a sequence of characters, it learns to predict the next character — and after training, it can generate Shakespeare-like text.

The architecture follows the original **Attention Is All You Need** paper (Vaswani et al., 2017), scaled down to run on a personal machine.

## Architecture

- **Token + positional embeddings**
- **Multi-head self-attention** with causal masking
- **Feed-forward layers** with ReLU activation
- **Layer normalization** (pre-norm formulation)
- **Residual connections**

## Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Block size (context) | 256 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Layers | 6 |
| Dropout | 0.2 |
| Max iterations | 5000 |

## Usage

**1. Download the data**
```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**2. Train the model**
```bash
python train.py
```

The model will print train/val loss every 500 steps and generate a sample at the end.

## Device support

The code automatically selects the best available device:
```python
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
```

Works on NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU.

## Requirements

```bash
pip install torch
```

## Credits

Fully inspired by [Andrej Karpathy](https://github.com/karpathy) and his video [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). Original nanoGPT repository: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).