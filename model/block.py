import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from config import Config

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa   = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(Config.n_embd)
        self.ln2  = nn.LayerNorm(Config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x