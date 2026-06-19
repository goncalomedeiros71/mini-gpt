import torch
from config import Config

class TextDataset:
    def __init__(self, text, tokenizer):
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # Treino fica nos mesmos 90% (checkpoints antigos continuam válidos).
        # Os 10% reservados são divididos em val (5%) e test (5%) — o test
        # nunca foi visto durante o treino.
        n_train = int(0.90 * len(data))
        n_val   = int(0.95 * len(data))
        self.train_data = data[:n_train]
        self.val_data   = data[n_train:n_val]
        self.test_data  = data[n_val:]

    def get_batch(self, split):
        d = {
            'train': self.train_data,
            'val':   self.val_data,
            'test':  self.test_data,
        }[split]
        ix = torch.randint(len(d) - Config.block_size, (Config.batch_size,))
        x = torch.stack([d[i:i+Config.block_size] for i in ix])
        y = torch.stack([d[i+1:i+Config.block_size+1] for i in ix])
        return x.to(Config.device), y.to(Config.device)