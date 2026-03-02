# Synthèse des résultats

- Nombre de modèles comparés: **5**
- Meilleur modèle (selon F1): **gating_mistral**
- F1 du meilleur modèle: **0.9721**

## Tableau comparatif (moyenne CV)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9972 | 0.9721 |      0.9758 |   0.9692 | 0.9999 |
| gating_bert    |     0.9968 | 0.9685 |      0.9571 |   0.9813 | 0.9999 |
| mlp_avg_eng    |     0.996  | 0.9615 |      0.9377 |   0.9882 | 0.9999 |
| mlp_source     |     0.9502 | 0.1137 |      0.6336 |   0.0686 | 0.8714 |
| mlp_text       |     0.95   | 0.0047 |      0.4059 |   0.0024 | 0.8312 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| gating_mistral |     0.9972 | 0.9721 |      0.9758 |   0.9692 | 0.9999 |
| gating_bert    |     0.9968 | 0.9685 |      0.9571 |   0.9813 | 0.9999 |
| mlp_avg_eng    |     0.996  | 0.9615 |      0.9377 |   0.9882 | 0.9999 |