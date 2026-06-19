import json
import math
import os

import torch

from config import Config
from data.corpus import load_text
from data.tokenizer import CharTokenizer
from data.dataset import TextDataset
from model.gpt import MiniGPT


# Quantos batches usar para estimar cada métrica. Mais batches = estimativa
# mais estável (menos variância), mas mais lento.
EVAL_BATCHES = 200
SEED = 1337
CHECKPOINT_PATH = "mini_gpt.pt"
METRICS_PATH = "metrics.json"


@torch.no_grad()
def evaluate_split(model, dataset, split):
    """Loss média, perplexity, bits-per-char e top-1/top-5 accuracy."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0

    for _ in range(EVAL_BATCHES):
        x, y = dataset.get_batch(split)
        logits, loss = model(x, y)

        total_loss += loss.item()

        # top-k accuracy do próximo caráter
        top5 = logits.topk(5, dim=-1).indices          # (B, T, 5)
        target = y.unsqueeze(-1)                         # (B, T, 1)
        correct_top1 += (top5[..., :1] == target).sum().item()
        correct_top5 += (top5 == target).any(dim=-1).sum().item()
        total_tokens += y.numel()

    avg_loss = total_loss / EVAL_BATCHES
    return {
        "loss":        avg_loss,
        "perplexity":  math.exp(avg_loss),
        "bpc":         avg_loss / math.log(2),
        "top1_acc":    correct_top1 / total_tokens,
        "top5_acc":    correct_top5 / total_tokens,
    }


def main():
    torch.manual_seed(SEED)

    text = load_text("input.txt")

    tokenizer = CharTokenizer(text)
    dataset = TextDataset(text, tokenizer)

    model = MiniGPT(tokenizer.vocab_size).to(Config.device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint '{CHECKPOINT_PATH}' não encontrado. Treina primeiro."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=Config.device)
    model.load_state_dict(checkpoint["model"])
    trained_iter = checkpoint.get("iter", "?")
    print(f"Checkpoint carregado (iter {trained_iter}) — device: {Config.device}")
    print(f"Batches por split: {EVAL_BATCHES} | seed: {SEED}\n")

    results = {}
    for split in ["val", "test"]:
        results[split] = evaluate_split(model, dataset, split)

    # ---- print ----
    header = f"{'split':<6} {'loss':>8} {'ppl':>9} {'bpc':>7} {'top1':>8} {'top5':>8}"
    print(header)
    print("-" * len(header))
    for split, m in results.items():
        print(
            f"{split:<6} {m['loss']:>8.4f} {m['perplexity']:>9.3f} "
            f"{m['bpc']:>7.4f} {m['top1_acc']:>7.2%} {m['top5_acc']:>7.2%}"
        )

    # ---- ficheiro ----
    payload = {
        "checkpoint": CHECKPOINT_PATH,
        "trained_iter": trained_iter,
        "eval_batches": EVAL_BATCHES,
        "seed": SEED,
        "vocab_size": tokenizer.vocab_size,
        "metrics": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas guardadas em {METRICS_PATH}")


if __name__ == "__main__":
    main()
