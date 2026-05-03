import torch
from config import Config
from data.tokenizer import CharTokenizer
from data.dataset import TextDataset
from model.gpt import MiniGPT
from train.trainer import Trainer

# ---------------- LOAD DATA ----------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

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

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "model.pt")
print("Modelo guardado em model.pt")