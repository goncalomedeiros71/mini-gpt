import torch
from config import Config

class TextDataset:
    def __init__(self, text, tokenizer):
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data   = data[n:]

    def get_batch(self, split):
        d = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(d) - Config.block_size, (Config.batch_size,))
        x = torch.stack([d[i:i+Config.block_size] for i in ix])
        y = torch.stack([d[i+1:i+Config.block_size+1] for i in ix])
        return x.to(Config.device), y.to(Config.device)