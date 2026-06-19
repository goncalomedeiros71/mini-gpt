import torch.nn as nn
import torch.nn.functional as F
from config import Config


class MultiHeadAttention(nn.Module):
    """Self-attention causal multi-head, vetorizada com scaled_dot_product_attention."""

    def __init__(self):
        super().__init__()
        assert Config.n_embd % Config.n_head == 0
        self.n_head = Config.n_head
        self.head_size = Config.n_embd // Config.n_head

        # Projeção única para Q, K, V (mais eficiente que 3 lineares separados).
        self.qkv = nn.Linear(Config.n_embd, 3 * Config.n_embd, bias=False)
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.proj.RESIDUAL_SCALE = True  # init escalado (estilo GPT-2)
        self.dropout = Config.dropout

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Máscara causal e escala tratadas internamente pelo PyTorch.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
