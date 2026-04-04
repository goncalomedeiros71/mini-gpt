# MiniGPT — Character-Level Language Model

A GPT-style Transformer language model built from scratch with PyTorch, trained on **Os Lusíadas** by Luís de Camões — one of the greatest works of Portuguese literature.

## Overview

This project implements a miniature GPT (Generative Pre-trained Transformer) that learns to generate text at the character level. Given a prompt, the model continues the text in the style of Camões' epic poem.

## Architecture

The model follows the decoder-only Transformer architecture introduced in *Attention Is All You Need*:

| Component | Details |
|---|---|
| Embedding dimension | 192 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Context length | 128 tokens |
| Vocabulary | Character-level (derived from input) |
| Tokenization | Character-to-index mapping |

Key components implemented in `mini_gpt.py`:

- **`Head`** — Single causal self-attention head with scaled dot-product attention and a causal mask
- **`MultiHeadAttention`** — Runs `n_head` attention heads in parallel and projects the concatenated output
- **`FeedForward`** — Two-layer MLP with ReLU activation (4× expansion factor)
- **`Block`** — Transformer block combining multi-head attention and feed-forward layers with pre-norm (`LayerNorm`) and residual connections
- **`MiniGPT`** — Full model with token + positional embeddings, stacked Transformer blocks, and a linear language model head

## Training

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max iterations | 8000 |
| Learning rate | 4e-4 |
| Optimizer | AdamW |
| Dropout | 0.1 |
| Train/val split | 90% / 10% |

The model is trained with cross-entropy loss over next-character prediction. Loss is evaluated every 500 steps on both the training and validation splits.

## Dataset

**Os Lusíadas** (1572) — Luís de Camões' Portuguese epic poem, sourced from [Project Gutenberg](https://www.gutenberg.org/ebooks/3333). The full text is loaded from `input.txt` and encoded at the character level.

## Files

```
LLM/
├── mini_gpt.py   # Model definition, training loop, and text generation
├── test.py       # CUDA/GPU availability check
└── input.txt     # Training data (Os Lusíadas, plain text)
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support recommended)

Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your platform.

## Usage

**Check GPU availability:**
```bash
python test.py
```

**Train the model and generate text:**
```bash
python mini_gpt.py
```

Training will print the loss every 500 steps. After training completes, the model generates two samples:

1. A continuation of the prompt `"Ó mar salgado, quanto do teu sal"` (500 tokens)
2. Text generated from an empty context (400 tokens)

**Example output prompt:**
```
Ó mar salgado, quanto do teu sal
```
The model will attempt to continue in Camões' style, drawing on patterns learned from the full poem.

## Text Generation

Generation uses temperature sampling (`temperature=0.85` by default). Lower temperature produces more conservative, repetitive output; higher temperature increases creativity and randomness.

```python
generate_text("your prompt here", max_new_tokens=500, temperature=0.85)
```

## Notes

- Training on CPU is possible but slow. A CUDA-capable GPU significantly reduces training time.
- The character-level vocabulary is built entirely from the input text — no external tokenizer is needed.
- This is an educational implementation inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
