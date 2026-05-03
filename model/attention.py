import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)
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
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(Config.n_head)])
        self.proj  = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))