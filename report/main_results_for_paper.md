# Main results to report (current codebase)

## 1) EVONS — Disinformation detection
- Best model by F1: **mlp_mistral**.
- Mean CV scores: **F1 = 0.9885**, Accuracy = 0.9894, AUC = 0.9995.
- Interpretation: disinformation detection on EVONS is near-ceiling with text embeddings + light MLP.

## 2) EVONS — Virality prediction
- Best model by F1: **gating_mistral**.
- Mean CV scores: **F1 = 0.3120**, Accuracy = 0.9503, AUC = 0.8812.
- Interpretation: virality remains hard with the q95 label; accuracy is inflated by class imbalance, so F1/recall are more informative.

## 3) FakeNewsNet — Disinformation detection
- Best model by F1: **rf_bert**.
- Mean CV scores: **F1 = 0.9061**, Accuracy = 0.9457, AUC = 0.9704.
- Interpretation: strong but not saturated performance; multiple models are close, suggesting robust separability.

## 4) FakeNewsNet — Virality prediction
- Best model by F1: **xgboost_bert**.
- Mean CV scores: **F1 = 0.7774**, Accuracy = 0.7698, AUC = 0.8614.
- Interpretation: substantially better than EVONS virality because the median-based label is more balanced and easier to learn.

## 5) Cross-task takeaway to highlight in the paper
- **Detection > virality difficulty gap** is consistent across datasets.
- **Label construction matters strongly**: EVONS q95 virality is much harder than FakeNewsNet median virality.
- **Model ranking depends on task/dataset**, but practical conclusions remain stable: high detection performance, moderate virality performance, and strong dependence on operational label definitions.
