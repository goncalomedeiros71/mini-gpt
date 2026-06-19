# Resultados e Hiperparâmetros — MiniGPT

> Para a avaliação completa (perplexity, BPC, top-k) e o histórico de melhorias,
> ver [RESULTADOS.md](RESULTADOS.md).

## Resultados do treino

Treinado em **318,363 caracteres** (Os Lusíadas, sem boilerplate do Gutenberg),
com **2.73M de parâmetros** numa GPU CUDA. Guarda-se o checkpoint com **menor val
loss** (early-stopping implícito), não o último.

| Step | Train Loss | Val Loss | LR |
|------|-----------|----------|----|
| 0    | 4.6134    | 4.6135   | 0.00e+00 |
| 250  | 2.3132    | 2.3614   | 4.00e-04 |
| 500  | 2.0886    | 2.1492   | 3.92e-04 |
| 750  | 1.8901    | 1.9800   | 3.73e-04 |
| 1000 | 1.7153    | 1.8479   | 3.45e-04 |
| 1250 | 1.5845    | 1.7614   | 3.08e-04 |
| 1500 | 1.4879    | 1.7013   | 2.65e-04 |
| 1750 | 1.4115    | 1.6532   | 2.19e-04 |
| 2000 | 1.3508    | 1.6321   | 1.72e-04 |
| 2250 | 1.3113    | 1.6172   | 1.26e-04 |
| 2500 | 1.2685    | 1.6111   | 8.40e-05 |
| 2750 | 1.2396    | 1.6058   | 4.89e-05 |
| **3000** | **1.2243** | **1.5936** | 2.22e-05 | ← melhor val (guardado) |
| 3250 | 1.2176    | 1.6000   | 5.64e-06 |

> Com regularização (`dropout` 0.2 + weight decay) o **gap train/val é saudável**
> (1.22 vs 1.59), ao contrário da versão anterior (0.50 vs 2.08, overfitting grave).
> O val loss estabiliza em ~1.59 e o melhor checkpoint é o do step 3000.

### Métricas finais (test set, nunca visto)

| Métrica | val | test |
|---|---|---|
| Perplexity | 4.94 | **4.81** |
| BPC | 2.30 | 2.27 |
| Top-1 accuracy | 52.0% | 53.5% |
| Top-5 accuracy | 83.8% | 84.1% |

---

## Hiperparâmetros

### Dados e Treino

| Parâmetro | Valor | Explicação |
|---|---|---|
| `batch_size` | `32` | Sequências processadas em paralelo por step. Mais = treino mais estável, mas mais VRAM. |
| `block_size` | `256` | Contexto máximo em caracteres. O modelo só "vê" os últimos 256 chars ao gerar. Aumentar melhora coerência a longo prazo mas tem custo quadrático. |
| `max_iters` | `3500` | Steps totais de treino. Com ~318k chars, o val loss estabiliza por volta do step 3000 — treinar mais só aumentaria o overfitting. |
| `learning_rate` | `4e-4` | Taxa de aprendizagem de pico, ajustada pelo scheduler ao longo do treino. |
| `weight_decay` | `0.1` | Regularização L2, aplicada só a matrizes de peso (2D), não a biases/LayerNorm. Combate o overfitting. |
| `grad_clip` | `1.0` | Se a norma dos gradientes exceder 1.0, são escalados. Previne instabilidade no início do treino. |
| `eval_interval` | `250` | De quantos em quantos steps se avalia e se tenta guardar o melhor checkpoint. |
| `eval_iters` | `40` | Batches usados para estimar o loss em cada avaliação. Mais = estimativa mais precisa mas mais lenta. |

### Arquitetura

| Parâmetro | Valor | Explicação |
|---|---|---|
| `n_embd` | `192` | Dimensão dos embeddings — tamanho da representação interna de cada token. GPT-2 small usa 768. |
| `n_head` | `6` | Cabeças de atenção em paralelo. Cada uma aprende padrões diferentes (rima, sintaxe, etc). `head_size = 192 / 6 = 32`. |
| `n_layer` | `6` | Blocos Transformer empilhados. Mais camadas = mais capacidade, mas mais lento e mais overfitting. |
| `dropout` | `0.2` | Desativa 20% dos neurónios aleatoriamente durante o treino para reduzir overfitting. Desativado na geração. |

Extras de arquitetura: **weight tying** (embedding partilhado com a projeção
final), **atenção vetorizada** (`F.scaled_dot_product_attention`), **init escalado
GPT-2** nas projeções residuais e **GELU** no FeedForward.

### Learning Rate Scheduler

| Parâmetro | Valor | Explicação |
|---|---|---|
| `warmup_iters` | `200` | Nos primeiros 200 steps o LR sobe linearmente de 0 até `4e-4`. Evita updates grandes quando os pesos ainda são aleatórios. |
| — | cosine decay | Após o warmup, o LR segue uma curva cosseno até quase 0 no step final. |

### Geração de Texto

| Parâmetro | Valor | Explicação |
|---|---|---|
| `temperature` | `0.85` | Controla aleatoriedade. `< 1` → texto mais focado. `> 1` → mais criativo mas incoerente. `0.85` é um bom equilíbrio. |
| `top_k` | `50` | Só os 50 tokens mais prováveis são considerados em cada passo. Elimina escolhas absurdas sem perder variedade. |
