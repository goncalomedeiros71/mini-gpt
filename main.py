import torch
import os
from config import Config
from model.gpt import MiniGPT
from data.tokenizer import CharTokenizer
from inference.modes import generate_long, autocomplete


# LOAD DATA SÓ PARA TOKENIZER
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)


# MODEL
model = MiniGPT(tokenizer.vocab_size).to(Config.device)


# LOAD CHECKPOINT
checkpoint_path = "mini_gpt.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    model.load_state_dict(checkpoint["model"])
    print("Modelo carregado")
else:
    print("Sem modelo treinado")


# MENU
print("\n=== MODOS ===")
print("1 - Os Lusíadas (geração longa)")
print("2 - Autocomplete")

mode = input("Escolhe: ")


if mode == "1":
    prompt = input("Prompt (ENTER para vazio): ")
    print("\n--- LUSÍADAS ---\n")
    print(generate_long(model, tokenizer, prompt))


elif mode == "2":
    prompt = input("Frase: ")
    print("\n--- AUTOCOMPLETE ---\n")

    suggestions = autocomplete(model, tokenizer, prompt)

    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s}")


else:
    print("Modo inválido")