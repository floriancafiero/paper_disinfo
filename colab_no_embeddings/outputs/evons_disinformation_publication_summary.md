# Synthèse des résultats

- Nombre de modèles comparés: **8**
- Meilleur modèle (selon F1): **mlp_mistral**
- F1 du meilleur modèle: **0.9872**

## Tableau comparatif (moyenne CV)

| model           |   accuracy |     f1 |   precision |   recall |    auc |
|:----------------|-----------:|-------:|------------:|---------:|-------:|
| mlp_mistral     |     0.9883 | 0.9872 |      0.9901 |   0.9845 | 0.9994 |
| mlp_bert        |     0.9842 | 0.9828 |      0.9857 |   0.9799 | 0.9987 |
| logreg_mistral  |     0.975  | 0.9727 |      0.9779 |   0.9675 | 0.9972 |
| xgboost_bert    |     0.972  | 0.9694 |      0.9748 |   0.9642 | 0.9966 |
| logreg_bert     |     0.9716 | 0.969  |      0.9744 |   0.9636 | 0.996  |
| xgboost_mistral |     0.959  | 0.9548 |      0.9671 |   0.9429 | 0.9934 |
| rf_bert         |     0.9487 | 0.9437 |      0.953  |   0.9345 | 0.9891 |
| rf_mistral      |     0.9306 | 0.9214 |      0.9624 |   0.8838 | 0.9843 |

## Top 3 (F1)

| model          |   accuracy |     f1 |   precision |   recall |    auc |
|:---------------|-----------:|-------:|------------:|---------:|-------:|
| mlp_mistral    |     0.9883 | 0.9872 |      0.9901 |   0.9845 | 0.9994 |
| mlp_bert       |     0.9842 | 0.9828 |      0.9857 |   0.9799 | 0.9987 |
| logreg_mistral |     0.975  | 0.9727 |      0.9779 |   0.9675 | 0.9972 |