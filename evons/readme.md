# Evons – Disinformation detection & Virality prediction

This repository contains notebook experiments on the Evons news dataset, centered on two supervised tasks:

1. Disinformation Detection: Classify news items as fake vs real.
2. Virality Prediction: Predict whether an item will become highly engaged (viral) based on a threshold over Facebook engagement statistics.

## Repository Structure

```
.
├── data/                                 # raw data and (once created) saved embeddings (.pt)
│
├── data_preprocessing/                   # script to build derived datasets with mistral
│   └── create_embeddings_evons.py        # generates Mistral embeddings for title & description (BERT embeddings can be created on-the-fly in notebooks)
│
├── disinformation_detection/             # notebooks for training models on disinformation detection
│   ├── MLP.ipynb                         # MLP + classic ML baselines
│   ├── MLP_mistral.ipynb                 # Same architecture on Mistral embeddings
│   └── readme.md
│
└── virality_prediction/                  # notebooks for training models on virality prediction
    ├── MLP.ipynb                         # MLP + classic ML baselines
    ├── source_embedding_model.ipynb      # Adds learned media-source embedding
    ├── average_engagement_model.ipynb    # Adds numeric per-source average engagement feature
    ├── gating_model.ipynb                # Gated fusion of engagement + text features
    ├── gating_model_mistral.ipynb        # Gated fusion variant on Mistral embeddings
    └── readme.md
```

Instructions on how to download data are provided in the [data](./data) readme.

## Data Representation
Each item in the Evons dataset includes a title and a caption. Source information is also included.


## Model families

All experiments use lightweight feed‑forward (MLP) classifiers over frozen (precomputed) text embeddings (e.g., RoBERTa CLS vectors or alternative large-model embeddings). Variants differ only in what auxiliary source/context signal they add (none, per‑source statistics, gating fusion, or learned source embeddings).
