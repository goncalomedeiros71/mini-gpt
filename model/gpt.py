import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from .block import Block


class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=Config.device))

        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        if idx is None or idx.size(1) == 0:
            idx = torch.zeros((1, 1), dtype=torch.long, device=next(self.parameters()).device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                cutoff = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < cutoff,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits
                )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx