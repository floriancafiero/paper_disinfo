# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **mlp_mistral**
- F1 du meilleur modèle: **0.9885**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| mlp_mistral     |     0.9894 | 0.9885 |      0.9883 |   0.9887 | 0.9995 |
| mlp_bert        |     0.9823 | 0.9807 |      0.9838 |   0.9777 | 0.9986 |
| logreg_mistral  |     0.9748 | 0.9725 |      0.9777 |   0.9674 | 0.9972 |
| xgboost_bert    |     0.9722 | 0.9697 |      0.9749 |   0.9645 | 0.9967 |
| logreg_bert     |     0.9711 | 0.9684 |      0.9738 |   0.9631 | 0.996  |
| xgboost_mistral |     0.9591 | 0.955  |      0.9666 |   0.9437 | 0.9933 |
| rf_bert         |     0.9489 | 0.9439 |      0.9529 |   0.9351 | 0.9893 |
| rf_mistral      |     0.9311 | 0.922  |      0.9627 |   0.8846 | 0.9843 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| mlp_mistral    |     0.9894 | 0.9885 |      0.9883 |   0.9887 | 0.9995 |
| mlp_bert       |     0.9823 | 0.9807 |      0.9838 |   0.9777 | 0.9986 |
| logreg_mistral |     0.9748 | 0.9725 |      0.9777 |   0.9674 | 0.9972 |