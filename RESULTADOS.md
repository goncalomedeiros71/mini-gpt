# Resultados da Avaliação — MiniGPT

Modelo char-level (GPT) treinado em *Os Lusíadas*. Documento gerado a partir de
`eval.py` (seed `1337`, 200 batches por split) e dos logs de treino.

## Métricas finais (corpus limpo, modelo regularizado)

Checkpoint: `mini_gpt.pt` (step 3000, melhor val loss). Avaliação em conjuntos
**nunca vistos no treino** (split 90/5/5).

| Split | Loss | Perplexity | BPC | Top-1 acc | Top-5 acc |
|-------|------|-----------|-----|-----------|-----------|
| val   | 1.5972 | 4.939 | 2.3043 | 52.04% | 83.77% |
| test  | 1.5704 | 4.808 | 2.2655 | 53.46% | 84.14% |

> **val e test concordam** (4.94 vs 4.81) → a avaliação é fidedigna e o modelo
> generaliza. O **top-5 de 84%** significa que, no modo autocomplete, o caráter
> correto está nas 5 sugestões 84% das vezes.

### O que medem as métricas

- **Perplexity** = `exp(loss)` — entre quantas opções o modelo "hesita". Menor = melhor.
- **BPC** (bits-per-character) = `loss / ln(2)` — métrica standard para LMs char-level. Menor = melhor.
- **Top-1 / Top-5 accuracy** — % de vezes que o caráter certo está na 1ª / nas 5 primeiras previsões. Relevante para o autocomplete.

---

## Evolução (perplexity no test set — menor é melhor)

| Estado | val ppl | test ppl | test top-1 | test top-5 |
|--------|--------:|---------:|-----------:|-----------:|
| Original (test contaminado com licença Gutenberg) | 9.0 | **77.6** ⚠️ | 22% | 52% |
| Corpus limpo, sem regularizar (último checkpoint) | 8.0 | 7.6 | 52% | 82% |
| **+ regularização + best-checkpoint + arquitetura** | **4.9** | **4.8** ✅ | **53%** | **84%** |

---

## Problemas encontrados e corrigidos

### 1. Contaminação do corpus
Os últimos ~5% do `input.txt` eram a **licença do Project Gutenberg em inglês**,
não o poema. O test set calhava inteiro nesse texto → perplexity 77.6 (sem
significado). Corrigido em [`data/corpus.py`](data/corpus.py) com `clean_gutenberg()`,
que remove cabeçalho e licença (17 caracteres de lixo + ~19k caracteres fora).

### 2. Overfitting + guardar o pior checkpoint
O treino original (8000 iters) fazia overfitting e o `Trainer` guardava o
**último** checkpoint (o pior em val), não o melhor.

| | train loss | val loss | gap |
|---|---:|---:|---:|
| Antes (8000 iters) | 0.50 | 2.08 | 1.58 (overfit grave) |
| Depois (regularizado) | 1.22 | 1.59 | 0.37 (saudável) |

Correções:
- Guardar o checkpoint com **menor val loss** (early-stopping implícito) — [`train/trainer.py`](train/trainer.py).
- `dropout` 0.1 → 0.2 e **weight decay** 0.1 (em param-groups) — [`config.py`](config.py).
- `max_iters` 8000 → 3500.

---

## Melhorias de arquitetura

- **Weight tying** — embedding partilhado com a projeção final ([`model/gpt.py`](model/gpt.py)).
- **Atenção vetorizada** com `F.scaled_dot_product_attention` ([`model/attention.py`](model/attention.py)).
- **Init escalado GPT-2** nas projeções residuais.
- FeedForward já usava **GELU**.

---

## Como reproduzir

```bash
venv\Scripts\python.exe main_train.py   # treina -> mini_gpt.pt (melhor val)
venv\Scripts\python.exe eval.py         # métricas -> consola + metrics.json
venv\Scripts\python.exe -m pytest -q    # 8 testes (shapes, causalidade, generate...)
```

Configuração: char-level, 2.73M parâmetros, `block_size` 256, `n_embd` 192,
`n_head` 6, `n_layer` 6. Corpus: 318,363 caracteres (Os Lusíadas, limpo).
