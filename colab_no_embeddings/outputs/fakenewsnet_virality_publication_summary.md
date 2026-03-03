# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **xgboost_bert**
- F1 du meilleur modèle: **0.7774**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| xgboost_bert    |     0.7698 | 0.7774 |      0.7542 |   0.8034 | 0.8614 |
| mlp_bert        |     0.7483 | 0.7722 |      0.704  |   0.8586 | 0.8065 |
| logreg_mistral  |     0.7552 | 0.7719 |      0.7234 |   0.8276 | 0.8082 |
| logreg_bert     |     0.7491 | 0.7657 |      0.7197 |   0.819  | 0.8086 |
| rf_bert         |     0.7612 | 0.7645 |      0.7549 |   0.7759 | 0.8615 |
| mlp_mistral     |     0.75   | 0.7601 |      0.7346 |   0.7966 | 0.8094 |
| xgboost_mistral |     0.7362 | 0.7401 |      0.7311 |   0.75   | 0.831  |
| rf_mistral      |     0.7491 | 0.74   |      0.768  |   0.7155 | 0.8492 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| xgboost_bert   |     0.7698 | 0.7774 |      0.7542 |   0.8034 | 0.8614 |
| mlp_bert       |     0.7483 | 0.7722 |      0.704  |   0.8586 | 0.8065 |
| logreg_mistral |     0.7552 | 0.7719 |      0.7234 |   0.8276 | 0.8082 |