import torch.nn as nn
from config import Config

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.n_embd, 4 * Config.n_embd),
            nn.GELU(),
            nn.Linear(4 * Config.n_embd, Config.n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)