import torch
from config import Config


def generate_long(model, tokenizer, prompt="", max_new_tokens=300, temperature=0.85, top_k=50):
    model.eval()

    if prompt.strip() == "":
        prompt = " "

    idx = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=Config.device
    ).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    return tokenizer.decode(out[0].tolist())


def autocomplete(model, tokenizer, prompt, temperature=1.0, top_k=5):
    model.eval()

    if prompt.strip() == "":
        prompt = " "

    idx = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=Config.device
    ).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(idx)

    logits = logits[:, -1, :] / temperature
    probs = torch.softmax(logits, dim=-1)

    values, indices = torch.topk(probs, top_k)

    suggestions = []

    for i in range(top_k):
        token_id = indices[0, i].item()
        suggestions.append(tokenizer.decode([token_id]))

    return suggestions