import torch

class Config:
    batch_size    = 32
    block_size    = 256
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