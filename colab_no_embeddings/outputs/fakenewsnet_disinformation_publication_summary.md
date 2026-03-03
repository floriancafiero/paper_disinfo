# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **rf_bert**
- F1 du meilleur modèle: **0.9061**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert         |     0.9457 | 0.9061 |      0.9421 |   0.8743 | 0.9704 |
| mlp_mistral     |     0.9448 | 0.9026 |      0.9591 |   0.8571 | 0.9805 |
| mlp_bert        |     0.9397 | 0.8948 |      0.9409 |   0.8571 | 0.9747 |
| xgboost_bert    |     0.9302 | 0.8805 |      0.9057 |   0.86   | 0.9755 |
| rf_mistral      |     0.9293 | 0.872  |      0.9552 |   0.8057 | 0.9562 |
| logreg_bert     |     0.9276 | 0.8655 |      0.975  |   0.78   | 0.9702 |
| xgboost_mistral |     0.9216 | 0.863  |      0.9078 |   0.8257 | 0.9601 |
| logreg_mistral  |     0.869  | 0.7206 |      1      |   0.5657 | 0.982  |

## Top 3 (F1)

| model       |   accuracy |     f1 |   precision |   recall |    auc |
|:------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert     |     0.9457 | 0.9061 |      0.9421 |   0.8743 | 0.9704 |
| mlp_mistral |     0.9448 | 0.9026 |      0.9591 |   0.8571 | 0.9805 |
| mlp_bert    |     0.9397 | 0.8948 |      0.9409 |   0.8571 | 0.9747 |