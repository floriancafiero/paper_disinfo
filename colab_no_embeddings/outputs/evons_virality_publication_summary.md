# Synthèse des résultats

- Nombre de modèles comparés: **5**
- Meilleur modèle (selon F1): **gating_mistral**
- F1 du meilleur modèle: **0.9739**

## Tableau comparatif (moyenne CV)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9974 | 0.9739 |      0.9781 |   0.97   | 0.9999 |
| gating_bert    |     0.9965 | 0.9658 |      0.9542 |   0.9799 | 0.9999 |
| mlp_avg_eng    |     0.9961 | 0.9615 |      0.9635 |   0.9624 | 0.9999 |
| mlp_source     |     0.9506 | 0.0754 |      0.6144 |   0.0414 | 0.8701 |
| mlp_text       |     0.9501 | 0.0221 |      0.4264 |   0.0116 | 0.828  |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9974 | 0.9739 |      0.9781 |   0.97   | 0.9999 |
| gating_bert    |     0.9965 | 0.9658 |      0.9542 |   0.9799 | 0.9999 |
| mlp_avg_eng    |     0.9961 | 0.9615 |      0.9635 |   0.9624 | 0.9999 |