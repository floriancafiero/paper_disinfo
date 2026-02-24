## FakeNewsNet – Disinformation detection & Virality prediction

This repository explores how news propagates on Twitter through the FakeNewsNet dataset (*Politifact* subfolder), centered on two supervised tasks:

1. Disinformation Detection: Detect disinformation in news propagations.

2. Virality Prediction: Predict whether a propagation will become viral.

It combines per-tweet text embeddings with lightweight metadata to build short temporal sequences per propagation and compares neural and classical models across these tasks. Models are evaluated with stratified cross‑validation. 

## Repository Structure

```
data/                      # Raw / intermediate data (notebooks expect precomputed embeddings & sequences)
data_preprocessing/        # Scripts to build standardized sequence datasets
  path_creation.py         # Creates tweet sequences from raw data
  ordering_data.py         # Orders tweets chronologically & groups into propagations
  create_embeddings.py     # Generates BERT text embeddings tensors
  create_embeddings_mistral.py # Generates Mistral embeddings through the Mistral API
disinformation_detection/  # Notebooks for fake vs real news classification
virality_prediction/       # Notebooks for viral vs non‑viral propagation classification
```

Both task folders mirror the same modeling approaches; only the target label differs (fake/real vs above/below virality threshold, e.g. median total likes).

## Data Representation

Each propagation = short sequence of tweets.
Per tweet features:
- Text embedding (BERT; some variants use Mistral)
- Scalar metadata: user verification, follower/following counts, favorites, elapsed time, etc.

Instructions on how to download data are in the [data](./data) folder.

## Model Families (Used in Both Tasks)

- CNN (1D temporal convolutions + adaptive pooling)
- RNN / GRU / LSTM (bidirectional recurrent encoders)
- Transformer encoder (self‑attention with positional embeddings + masked pooling)
- GRU ablations isolating text vs numerical features
- Classical baselines (dummy, logistic regression, random forest) on aggregated statistics
---
- GRU with alternative text encoder (Mistral)

Core comparisons: convolution vs recurrence vs self‑attention; gating depth; feature group contribution; embedding source impact.