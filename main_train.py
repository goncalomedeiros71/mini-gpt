import torch
from config import Config
from data.corpus import load_text
from data.tokenizer import CharTokenizer
from data.dataset import TextDataset
from model.gpt import MiniGPT
from train.trainer import Trainer

# ---------------- LOAD DATA ----------------
text = load_text("input.txt")

print(f"Usando dispositivo: {Config.device}")
print(f"Tamanho do texto: {len(text):,} caracteres")

# ---------------- TOKENIZER ----------------
tokenizer = CharTokenizer(text)
dataset = TextDataset(text, tokenizer)

# ---------------- MODEL ----------------
model = MiniGPT(tokenizer.vocab_size)

print(f"Parâmetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# ---------------- TRAIN ----------------
trainer = Trainer(model, dataset)
trainer.train()

# O melhor modelo (menor val loss) já foi guardado em mini_gpt.pt durante o treino.
print(f"\nTreino terminado. Melhor val loss: {trainer.best_val:.4f} -> mini_gpt.pt")