import torch
from collections import Counter
from config import Config


# Caracteres que terminam uma palavra.
SEPARATORS = set(" \n\t\r.,;:!?…«»\"'()[]—-")


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


def _sample_one_word(model, tokenizer, idx, temperature, top_k, max_chars, skip_leading):
    """Gera caracteres até completar uma palavra. Devolve a string gerada."""
    chars = []
    started = not skip_leading

    # Em modo 'próxima palavra' damos folga para saltar separadores iniciais.
    budget = max_chars + (5 if skip_leading else 0)

    for _ in range(budget):
        idx_cond = idx[:, -Config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ch = tokenizer.decode([nxt.item()])

        if ch in SEPARATORS:
            if started:
                break          # fim da palavra
            idx = torch.cat((idx, nxt), dim=1)
            continue           # ainda a saltar separadores iniciais

        started = True
        chars.append(ch)
        idx = torch.cat((idx, nxt), dim=1)
        if len(chars) >= max_chars:
            break

    return "".join(chars)


def autocomplete_word(model, tokenizer, prompt, n_suggestions=5,
                      temperature=0.8, top_k=20, n_samples=24, max_chars=20):
    """Sugere palavras completas.

    Modo esperto: se a frase acaba a meio de uma palavra, completa-a;
    se acaba num separador, sugere a próxima palavra inteira.
    Várias sugestões via sampling (as candidatas mais frequentes).
    """
    model.eval()

    if prompt == "":
        prompt = " "

    # Parte da palavra já escrita (caracteres finais até ao último separador).
    partial = ""
    i = len(prompt)
    while i > 0 and prompt[i - 1] not in SEPARATORS:
        partial = prompt[i - 1] + partial
        i -= 1

    completing = partial != ""          # True = acabar palavra atual
    base = tokenizer.encode(prompt)

    def sample_words(skip_leading):
        counts = Counter()
        empty = 0
        for _ in range(n_samples):
            idx = torch.tensor(base, dtype=torch.long, device=Config.device).unsqueeze(0)
            with torch.no_grad():
                cont = _sample_one_word(
                    model, tokenizer, idx,
                    temperature, top_k, max_chars,
                    skip_leading=skip_leading,
                )
            if cont == "":
                empty += 1
            if cont.strip():
                counts[cont] += 1
        return counts, empty

    if completing:
        counts, empty = sample_words(skip_leading=False)
        # Fallback: se a palavra atual já está completa (a maioria das amostras
        # não gera continuação), sugere antes a próxima palavra.
        if empty >= n_samples // 2:
            counts, _ = sample_words(skip_leading=True)
        else:
            counts = Counter({partial + c: n for c, n in counts.items()})
    else:
        counts, _ = sample_words(skip_leading=True)

    return [w for w, _ in counts.most_common(n_suggestions)]