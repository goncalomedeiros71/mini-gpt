# MiniGPT — Character-Level Language Model

A GPT-style Transformer language model built from scratch with PyTorch, trained on **Os Lusíadas** by Luís de Camões — one of the greatest works of Portuguese literature.

## Overview

This project implements a miniature GPT (Generative Pre-trained Transformer) that learns to generate text at the character level. Given a prompt, the model continues the text in the style of Camões' epic poem. It also includes a reproducible **evaluation suite** (perplexity, bits-per-character, top-k accuracy) and **unit tests**.

## Architecture

The model follows the decoder-only Transformer architecture introduced in *Attention Is All You Need*:

| Component | Details |
|---|---|
| Embedding dimension | 192 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Context length | 256 tokens |
| Parameters | ~2.73M |
| Vocabulary | Character-level (derived from input) |
| Tokenization | Character-to-index mapping |

Key components:

- **`MultiHeadAttention`** (`model/attention.py`) — Causal multi-head self-attention, vectorized with `F.scaled_dot_product_attention` (fused QKV projection, causal mask handled internally)
- **`FeedForward`** (`model/feedforward.py`) — Two-layer MLP with GELU activation (4× expansion factor)
- **`Block`** (`model/block.py`) — Transformer block combining attention and feed-forward with pre-norm (`LayerNorm`) and residual connections
- **`MiniGPT`** (`model/gpt.py`) — Full model with token + positional embeddings, stacked blocks, and a language-model head. Uses **weight tying** (shared embedding / output projection) and **GPT-2 scaled init** on residual projections

## Training

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Max iterations | 3500 |
| Learning rate | 4e-4 (warmup + cosine decay) |
| Optimizer | AdamW (weight decay 0.1, in param groups) |
| Dropout | 0.2 |
| Train/val/test split | 90% / 5% / 5% |

The model is trained with cross-entropy loss over next-character prediction. Every 250 steps the loss is evaluated on the train and validation splits, and the checkpoint with the **lowest validation loss** is saved to `mini_gpt.pt` (implicit early stopping — avoids keeping an overfit model).

## Evaluation

`eval.py` loads the best checkpoint and reports, on the held-out **validation and test** splits (seed-fixed for reproducibility):

| Metric | Meaning |
|---|---|
| **Loss** | Cross-entropy of next-character prediction |
| **Perplexity** (`exp(loss)`) | How many options the model "hesitates" between — lower is better |
| **BPC** (`loss / ln 2`) | Bits-per-character — the standard metric for char-level LMs |
| **Top-1 / Top-5 accuracy** | How often the correct character is the 1st / among the top 5 predictions (relevant for autocomplete) |

Latest results (clean corpus, regularized model — see [RESULTADOS.md](RESULTADOS.md)):

| Split | Perplexity | BPC | Top-1 | Top-5 |
|-------|-----------:|----:|------:|------:|
| val   | 4.94 | 2.30 | 52.0% | 83.8% |
| test  | 4.81 | 2.27 | 53.5% | 84.1% |

## Dataset

**Os Lusíadas** (1572) — Luís de Camões' Portuguese epic poem, sourced from [Project Gutenberg](https://www.gutenberg.org/ebooks/3333). The text is loaded from `input.txt`; `data/corpus.py` strips the Project Gutenberg header and license boilerplate so the model trains only on the poem (~318k characters), then it is encoded at the character level.

## Files

```
LLM/
├── config.py            # Hyperparameters
├── main_train.py        # Training entry point
├── main.py              # Interactive inference (long generation / autocomplete)
├── eval.py              # Evaluation: perplexity, BPC, top-k -> metrics.json
├── test.py              # CUDA/GPU availability check
├── input.txt            # Raw training data (Os Lusíadas + Gutenberg boilerplate)
├── data/
│   ├── corpus.py        # Loads input.txt, strips Gutenberg boilerplate
│   ├── tokenizer.py     # Character-level tokenizer
│   └── dataset.py       # Train/val/test split + batching
├── model/
│   ├── attention.py     # Vectorized causal multi-head attention
│   ├── feedforward.py   # GELU MLP
│   ├── block.py         # Transformer block
│   └── gpt.py           # Full MiniGPT model
├── train/
│   └── trainer.py       # Training loop, LR schedule, best-val checkpointing
├── inference/
│   └── modes.py         # generate_long() and autocomplete()
└── tests/
    └── test_model.py    # Unit tests (shapes, causality, encode/decode, generate)
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support recommended)
- `pytest` (for the test suite)

Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) for your platform.

## Usage

**Check GPU availability:**
```bash
python test.py
```

**Train the model** (saves the best checkpoint to `mini_gpt.pt`):
```bash
python main_train.py
```

**Evaluate** (prints metrics and writes `metrics.json`):
```bash
python eval.py
```

**Run the tests:**
```bash
python -m pytest -q
```

**Generate text interactively:**
```bash
python main.py
```
Then choose a mode:
1. **Os Lusíadas** — long generation from an optional prompt
2. **Autocomplete** — top character suggestions for the next position

## Text Generation

Generation uses temperature + top-k sampling (`temperature=0.85`, `top_k=50` by default). Lower temperature produces more conservative, repetitive output; higher temperature increases creativity and randomness.

```python
from inference.modes import generate_long
generate_long(model, tokenizer, "your prompt here", max_new_tokens=300, temperature=0.85)
```

## Notes

- Training on CPU is possible but slow. A CUDA-capable GPU significantly reduces training time.
- The character-level vocabulary is built entirely from the input text — no external tokenizer is needed.
- This is an educational implementation inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
