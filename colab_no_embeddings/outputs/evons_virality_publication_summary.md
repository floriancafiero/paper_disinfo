# Synthèse des résultats

- Nombre de modèles comparés: **5**
- Meilleur modèle (selon F1): **gating_mistral**
- F1 du meilleur modèle: **0.3120**

## Tableau comparatif (moyenne CV)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9503 | 0.312  |      0.5247 |   0.2292 | 0.8812 |
| mlp_avg_eng    |     0.9506 | 0.0866 |      0.6305 |   0.0485 | 0.8677 |
| mlp_source     |     0.9506 | 0.084  |      0.6665 |   0.0468 | 0.8703 |
| gating_bert    |     0.9501 | 0.0063 |      0.0633 |   0.0033 | 0.8685 |
| mlp_text       |     0.95   | 0.0059 |      0.313  |   0.003  | 0.8281 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9503 | 0.312  |      0.5247 |   0.2292 | 0.8812 |
| mlp_avg_eng    |     0.9506 | 0.0866 |      0.6305 |   0.0485 | 0.8677 |
| mlp_source     |     0.9506 | 0.084  |      0.6665 |   0.0468 | 0.8703 |