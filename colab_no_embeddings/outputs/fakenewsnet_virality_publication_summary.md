# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **rf_bert**
- F1 du meilleur modèle: **0.7697**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert         |     0.7647 | 0.7697 |      0.7551 |   0.7855 | 0.8549 |
| logreg_mistral  |     0.7387 | 0.7567 |      0.7103 |   0.8129 | 0.7823 |
| logreg_bert     |     0.7336 | 0.7549 |      0.7    |   0.82   | 0.7872 |
| xgboost_mistral |     0.7457 | 0.7531 |      0.7347 |   0.7751 | 0.8241 |
| xgboost_bert    |     0.7422 | 0.7494 |      0.7287 |   0.7716 | 0.8518 |
| mlp_bert        |     0.7266 | 0.7437 |      0.7005 |   0.8062 | 0.7882 |
| mlp_mistral     |     0.725  | 0.7395 |      0.7019 |   0.7927 | 0.7856 |
| rf_mistral      |     0.7474 | 0.7387 |      0.7684 |   0.7164 | 0.8416 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| rf_bert        |     0.7647 | 0.7697 |      0.7551 |   0.7855 | 0.8549 |
| logreg_mistral |     0.7387 | 0.7567 |      0.7103 |   0.8129 | 0.7823 |
| logreg_bert    |     0.7336 | 0.7549 |      0.7    |   0.82   | 0.7872 |