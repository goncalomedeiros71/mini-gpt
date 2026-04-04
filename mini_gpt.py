import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparâmetros
batch_size    = 32
block_size    = 128
max_iters     = 8000
learning_rate = 4e-4
n_embd        = 192
n_head        = 6
n_layer       = 6
dropout       = 0.1
eval_interval = 500
eval_iters    = 20
warmup_iters  = 200
grad_clip     = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Usando dispositivo: {device}")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Tamanho do texto: {len(text):,} caracteres")

# Vocabulário character-level
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Dados
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size]   for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# Componentes do modelo

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),                          # GELU é padrão nos GPTs (vs ReLU)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self._init_weights()

    def _init_weights(self):
        """Inicialização estilo GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.85, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling: descarta tokens fora dos k mais prováveis
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Instanciar modelo
model = MiniGPT().to(device)
print(f"Parâmetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Carregar checkpoint se existir
import os
checkpoint_exists = os.path.exists('mini_gpt.pt')
if checkpoint_exists:
    model.load_state_dict(torch.load('mini_gpt.pt', map_location=device))
    print("Checkpoint carregado — a continuar treino.")
else:
    print("Sem checkpoint — a treinar do zero.")


def get_lr(it):
    """Cosine decay com linear warmup. Salta warmup se já há checkpoint."""
    if not checkpoint_exists and it < warmup_iters:
        return learning_rate * it / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Loop de treino
for iter in range(max_iters):
    # Atualizar learning rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient clipping
    optimizer.step()

# Guardar modelo
torch.save(model.state_dict(), 'mini_gpt.pt')
print("Modelo guardado em mini_gpt.pt")


# Geração de texto
def generate_text(prompt="", max_new_tokens=600, temperature=0.85, top_k=50):
    model.eval()
    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(generated[0].tolist())


print("\n" + "=" * 60)
print("Texto gerado com prompt:")
print(generate_text("Ó mar salgado, quanto do teu sal", max_new_tokens=500))
print("\n" + "=" * 60)
print("Texto gerado do zero:")
print(generate_text("", max_new_tokens=400))
