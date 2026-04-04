# Resultados e Hiperparâmetros — MiniGPT

## Resultados do treino

Treinado em **337,640 caracteres** (Os Lusíadas), com **2.73M de parâmetros** numa GPU CUDA.

| Step | Train Loss | Val Loss | LR |
|------|-----------|----------|----|
| 0    | 4.7907    | 4.7795   | 0.00e+00 |
| 500  | 2.0574    | 3.1228   | 3.99e-04 |
| 1000 | 1.7573    | 2.8694   | 3.90e-04 |
| 1500 | 1.5728    | 2.7936   | 3.73e-04 |
| 2000 | 1.4358    | 2.8161   | 3.50e-04 |
| 2500 | 1.3358    | 2.7516   | 3.20e-04 |
| 3000 | 1.2636    | 2.6759   | 2.86e-04 |
| 3500 | 1.1800    | 2.7281   | 2.48e-04 |
| 4000 | 1.1074    | 2.8086   | 2.08e-04 |
| 4500 | 1.0449    | 2.8244   | 1.68e-04 |
| 5000 | 0.9860    | 2.7267   | 1.29e-04 |
| 5500 | 0.9359    | 2.8494   | 9.31e-05 |
| 6000 | 0.8965    | 2.7350   | 6.15e-05 |
| 6500 | 0.8642    | 2.9502   | 3.54e-05 |
| 7000 | 0.8550    | 2.8671   | 1.60e-05 |
| 7500 | 0.8405    | 2.8389   | 4.04e-06 |
| 7999 | 0.8476    | 2.9594   | 1.62e-11 |

> O **train loss** desce consistentemente até ~0.84. O **val loss** estabiliza em ~2.7–2.9 — indica algum overfitting, esperado num dataset pequeno. Para melhorar, aumenta `dropout` ou usa um corpus maior.

---

## Hiperparâmetros

### Dados e Treino

| Parâmetro | Valor | Explicação |
|---|---|---|
| `batch_size` | `32` | Sequências processadas em paralelo por step. Mais = treino mais estável, mas mais VRAM. |
| `block_size` | `128` | Contexto máximo em caracteres. O modelo só "vê" os últimos 128 chars ao gerar. Aumentar melhora coerência a longo prazo mas tem custo quadrático. |
| `max_iters` | `8000` | Steps totais de treino. Com ~337k chars, 8000 é suficiente — mais iters aumentaria o overfitting. |
| `learning_rate` | `4e-4` | Taxa de aprendizagem de pico, ajustada pelo scheduler ao longo do treino. |
| `grad_clip` | `1.0` | Se a norma dos gradientes exceder 1.0, são escalados. Previne instabilidade no início do treino. |
| `eval_interval` | `500` | De quantos em quantos steps se imprime o loss. Não afeta o treino. |
| `eval_iters` | `20` | Batches usados para estimar o loss em cada avaliação. Mais = estimativa mais precisa mas mais lenta. |

### Arquitetura

| Parâmetro | Valor | Explicação |
|---|---|---|
| `n_embd` | `192` | Dimensão dos embeddings — tamanho da representação interna de cada token. GPT-2 small usa 768. |
| `n_head` | `6` | Cabeças de atenção em paralelo. Cada uma aprende padrões diferentes (rima, sintaxe, etc). `head_size = 192 / 6 = 32`. |
| `n_layer` | `6` | Blocos Transformer empilhados. Mais camadas = mais capacidade, mas mais lento e mais overfitting. |
| `dropout` | `0.1` | Desativa 10% dos neurónios aleatoriamente durante o treino para reduzir overfitting. Desativado na geração. |

### Learning Rate Scheduler

| Parâmetro | Valor | Explicação |
|---|---|---|
| `warmup_iters` | `200` | Nos primeiros 200 steps o LR sobe linearmente de 0 até `4e-4`. Evita updates grandes quando os pesos ainda são aleatórios. |
| — | cosine decay | Após o warmup, o LR segue uma curva cosseno até quase 0 no step final (`1.62e-11`). O modelo convergiu completamente. |

### Geração de Texto

| Parâmetro | Valor | Explicação |
|---|---|---|
| `temperature` | `0.85` | Controla aleatoriedade. `< 1` → texto mais focado. `> 1` → mais criativo mas incoerente. `0.85` é um bom equilíbrio. |
| `top_k` | `50` | Só os 50 tokens mais prováveis são considerados em cada passo. Elimina escolhas absurdas sem perder variedade. |

---

## Exemplo de output

**Com prompt** `"Ó mar salgado, quanto do teu sal"`:
```
Ó mar salgado, quanto do teu salgado
Onde vê a vila de ventura mandado
Os primeiros ânimos deleitos
Piloto, fazendo, brando, afeiçoado:
Já não vencerá nas bombardantes...
```

**Sem prompt** (geração livre):
```
Os povos inimigos do Oriente,
Que eu foi vosso arvor poderoso.

"E vereis ao Rei manda que chegado
Muito estava já será na constante?...
```
