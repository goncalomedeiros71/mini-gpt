import torch
import math
import os
from config import Config

class Trainer:
    def __init__(self, model, dataset):
        self.model = model.to(Config.device)
        self.dataset = dataset

        self.optimizer = self._build_optimizer()

        self.checkpoint_path = "mini_gpt.pt"
        self.start_iter = 0
        self.best_val = float("inf")

        self._load_checkpoint()

    # ---------------- OPTIMIZER ----------------
    def _build_optimizer(self):
        # Weight decay só em tensores 2D (matrizes de pesos); biases e LayerNorm
        # ficam de fora — convenção GPT-2.
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": Config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=Config.learning_rate)

    # ---------------- CHECKPOINT ----------------
    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=Config.device)

            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_iter = checkpoint.get("iter", 0)
            self.best_val = checkpoint.get("best_val", float("inf"))

            print(f"Checkpoint carregado — a continuar do iter {self.start_iter}")
        else:
            print("Sem checkpoint — treino do zero.")

    def _save_checkpoint(self, iter, val_loss):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter": iter,
            "best_val": val_loss,
        }, self.checkpoint_path)

    # ---------------- LR SCHEDULE ----------------
    def get_lr(self, it):
        if it < Config.warmup_iters:
            return Config.learning_rate * it / Config.warmup_iters
        progress = (it - Config.warmup_iters) / max(1, Config.max_iters - Config.warmup_iters)
        return Config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---------------- LOSS ----------------
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(Config.eval_iters)
            for k in range(Config.eval_iters):
                x, y = self.dataset.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.model.train()
        return out

    # ---------------- TRAIN LOOP ----------------
    def train(self):
        for iter in range(self.start_iter, Config.max_iters):

            lr = self.get_lr(iter)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            if iter % Config.eval_interval == 0:
                losses = self.estimate_loss()
                val = losses['val'].item()

                # Guarda só quando a val loss melhora (early-stopping implícito):
                # ficamos com o melhor modelo, não com o último (que faz overfit).
                marker = ""
                if val < self.best_val:
                    self.best_val = val
                    self._save_checkpoint(iter, val)
                    marker = "  <- guardado (melhor val)"

                print(f"step {iter}: train {losses['train']:.4f} | val {val:.4f} | lr {lr:.2e}{marker}")

            xb, yb = self.dataset.get_batch('train')
            _, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            self.optimizer.step()