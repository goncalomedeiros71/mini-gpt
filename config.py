import torch

class Config:
    batch_size    = 32
    block_size    = 256
    max_iters     = 3500
    learning_rate = 4e-4
    weight_decay  = 0.1
    n_embd        = 192
    n_head        = 6
    n_layer       = 6
    dropout       = 0.2

    eval_interval = 250
    eval_iters    = 40
    warmup_iters  = 200
    grad_clip     = 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'