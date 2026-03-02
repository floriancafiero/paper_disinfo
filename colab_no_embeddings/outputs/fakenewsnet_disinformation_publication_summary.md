# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **rf_bert**
- F1 du meilleur modèle: **0.9071**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert         |     0.9463 | 0.9071 |      0.9381 |   0.8793 | 0.9644 |
| mlp_bert        |     0.9395 | 0.8985 |      0.913  |   0.8852 | 0.9705 |
| mlp_mistral     |     0.9429 | 0.8967 |      0.9637 |   0.8452 | 0.9813 |
| xgboost_bert    |     0.9377 | 0.8939 |      0.9163 |   0.8736 | 0.9695 |
| rf_mistral      |     0.9308 | 0.8767 |      0.9463 |   0.8217 | 0.9564 |
| xgboost_mistral |     0.9256 | 0.8688 |      0.9094 |   0.8385 | 0.9644 |
| logreg_bert     |     0.9273 | 0.8654 |      0.9713 |   0.7817 | 0.9684 |
| logreg_mistral  |     0.8599 | 0.6919 |      1      |   0.5345 | 0.9824 |

## Top 3 (F1)

| model       |   accuracy |     f1 |   precision |   recall |    auc |
|:------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert     |     0.9463 | 0.9071 |      0.9381 |   0.8793 | 0.9644 |
| mlp_bert    |     0.9395 | 0.8985 |      0.913  |   0.8852 | 0.9705 |
| mlp_mistral |     0.9429 | 0.8967 |      0.9637 |   0.8452 | 0.9813 |