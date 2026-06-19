import torch
import pytest

from config import Config
from data.tokenizer import CharTokenizer
from data.dataset import TextDataset
from model.gpt import MiniGPT


# Grande o suficiente para que cada split (incl. os 5% de test) caiba > block_size.
SAMPLE = "abcdefg hijklmn opqrst. uvwxyz!\n" * 400


@pytest.fixture
def tokenizer():
    return CharTokenizer(SAMPLE)


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return MiniGPT(tokenizer.vocab_size).to(Config.device)


# ---------------- TOKENIZER ----------------
def test_encode_decode_is_inverse(tokenizer):
    s = "abc def!\n"
    assert tokenizer.decode(tokenizer.encode(s)) == s


def test_vocab_size_matches_unique_chars(tokenizer):
    assert tokenizer.vocab_size == len(set(SAMPLE))


# ---------------- DATASET ----------------
def test_splits_are_disjoint_and_cover_data(tokenizer):
    ds = TextDataset(SAMPLE, tokenizer)
    total = len(ds.train_data) + len(ds.val_data) + len(ds.test_data)
    full = len(tokenizer.encode(SAMPLE))
    assert total == full
    assert len(ds.test_data) > Config.block_size  # test set utilizável


def test_get_batch_shapes_and_shift(tokenizer):
    ds = TextDataset(SAMPLE, tokenizer)
    x, y = ds.get_batch('train')
    assert x.shape == (Config.batch_size, Config.block_size)
    assert y.shape == (Config.batch_size, Config.block_size)
    # y é x deslocado um caráter: x[:, 1:] == y[:, :-1]
    assert torch.equal(x[:, 1:], y[:, :-1])


# ---------------- MODEL ----------------
def test_forward_logits_shape(model, tokenizer):
    x = torch.randint(0, tokenizer.vocab_size, (2, 16), device=Config.device)
    logits, loss = model(x)
    assert logits.shape == (2, 16, tokenizer.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(model, tokenizer):
    x = torch.randint(0, tokenizer.vocab_size, (2, 16), device=Config.device)
    y = torch.randint(0, tokenizer.vocab_size, (2, 16), device=Config.device)
    logits, loss = model(x, y)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_attention_is_causal(model, tokenizer):
    """Mudar um token futuro não pode alterar a previsão de posições anteriores."""
    model.eval()
    x = torch.randint(0, tokenizer.vocab_size, (1, 12), device=Config.device)
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % tokenizer.vocab_size  # altera só o último
        changed, _ = model(x2)
    # Todas as posições menos a última têm de ficar idênticas.
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-5)


def test_generate_respects_block_size(model, tokenizer):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    out = model.generate(idx, max_new_tokens=Config.block_size + 10)
    assert out.shape[1] == 1 + Config.block_size + 10
    assert out.max().item() < tokenizer.vocab_size  # tokens válidos
