import torch.nn as nn
from config import Config

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        proj = nn.Linear(4 * Config.n_embd, Config.n_embd)
        proj.RESIDUAL_SCALE = True  # init escalado (estilo GPT-2)
        self.net = nn.Sequential(
            nn.Linear(Config.n_embd, 4 * Config.n_embd),
            nn.GELU(),
            proj,
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)