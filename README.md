# M2 Mémoire – Disinformation Detection & Virality Prediction

This repository contains the codebase for the master thesis (M2 « Humanités Numériques », École nationale des chartes) studying automatic classification of (a) fake vs. real news and (b) viral vs. non‑viral items / propagations across two different news datasets. 

[**Click here to read the thesis!**](https://savaij.github.io/memoire_disinfo/report/memoire.pdf)

## 1. Tasks

| Task | Definition | Label Source |
|------|------------|--------------|
| Disinformation Detection | Binary classification: fake vs real news item / propagation | Dataset ground‑truth labels (Evons & FakeNewsNet-Politifact) |
| Virality Prediction | Binary classification: will an item / propagation be “viral” | Evons: ≥ 95th percentile Facebook engagement. FakeNewsNet: ≥ median total likes across propagations |

## 2. Datasets

### Evons Dataset
Static news articles with metadata and engagement statistics (e.g. Facebook). Modeling treats each article independently (non‑sequential). Text = title + caption/description. Virality label derived from high‑end engagement threshold (95th percentile). See `evons/data/readme.md` for download & embedding links.

### FakeNewsNet (Politifact subset)
Twitter propagation trees. Each propagation becomes a sequence of per‑tweet features: text embedding + scalar metadata (verification flag, follower/following counts, favorites, elapsed time, etc.). Virality defined via median total likes threshold. Raw data access subject to Twitter/X restrictions — see `FakeNewsNet/data/readme.md`.

## 3. Repository Overview

```
memoire_disinfo/
├── evons/
│   ├── data/                        # Evons raw data & precomputed embeddings (external download)
│   ├── disinformation_detection/    # MLP variants (RoBERTa & Mistral)
│   └── virality_prediction/         # MLP baseline + source / engagement feature variants
│
├── FakeNewsNet/
│   ├── data/                        # Politifact sequences & embeddings (external download)
│   ├── data_preprocessing/          # Scripts: ordering, path creation, embedding generation
│   ├── disinformation_detection/    # Sequence model notebooks (CNN/RNN/GRU/LSTM/Transformer)
│   └── virality_prediction/         # Same architectures for virality label
│
└── report/                          # LaTeX sources and compiled PDF of the report
```

Both dataset folders intentionally mirror a two‑task layout for clarity and comparability.


## 4. Data Access & Privacy

Data are **not** committed because of size & licensing:
* Evons: follow upstream instructions; precomputed embeddings via provided Drive folder.
* FakeNewsNet Politifact: raw data requires permission / contact; processed sequences & embeddings via protected Drive link. Respect Twitter/X terms for any redistribution.

See each dataset’s `data/readme.md` for authoritative links and any required credentials or requests.


## 5. How To Reproduce

1. Obtain and place data as described in `evons/data/readme.md` and `FakeNewsNet/data/readme.md`.
2. (Optional) For FakeNewsNet: regenerate data using scripts in `FakeNewsNet/data_preprocessing/` (`path_creation.py`, `ordering_data.py`, `create_embeddings.py`, `create_embeddings_mistral.py`). <br> Evons notebooks already provide code for embedding texts on the fly if they are not available in the `evons/data` folder. If you download preprocessed data, you can skip this step.
3. Open the relevant notebook (e.g., `evons/disinformation_detection/MLP.ipynb`) and execute cells top‑to‑bottom. Notebooks are self‑contained (data paths assume relative placement inside each dataset’s `data/`).
4. Compare output metrics across variants.

### Suggested Python Environment 
The following packages are required:
`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `seaborn`, `wandb`. 

`mistralai` is required for Mistral embedding generation.

## 6. Folder Quick Reference

| Path | Purpose |
|------|---------|
| `evons/disinformation_detection/` | Article fake vs real (MLP variants) |
| `evons/virality_prediction/` | Article virality (MLP + feature fusion variants) |
| `FakeNewsNet/data_preprocessing/` | Build tweet sequences & embeddings |
| `FakeNewsNet/disinformation_detection/` | Propagation fake vs real (sequence models) |
| `FakeNewsNet/virality_prediction/` | Propagation virality (sequence models) |


---
**Important**: the code does not save model weights after training. To save them, modify the notebooks, specifically the 
_`train_single_fold`_ functions.

---
For questions or access issues (e.g., processed data links), contact the author.


## 7. Statistical significance add-on (for rebuttal / paper revision)

To support model-comparison claims with inferential statistics (instead of only mean/std across folds), use:

```bash
python tools/statistical_evaluation.py --input <metrics.csv> --output <stats.csv>
```

The script expects per-fold metrics and outputs, for each pair of models (within each dataset/task/metric):
- paired mean difference
- bootstrap 95% CI of the mean difference
- exact sign-flip p-value (paired test; Monte-Carlo approximation for many folds)
- Holm-Bonferroni corrected p-values
- Cliff's delta effect size

Supported input formats:
- **long**: `dataset,task,metric,fold,model,value`
- **wide** (one column per model) with `--wide --id-cols dataset task metric fold`

This is compatible with already-computed embeddings and preprocessed data; no embedding recomputation is required.


## 8. EVONS source-confounding audit (no embedding recomputation)

To directly test source leakage risk (review concern), run:

```bash
python tools/evons_source_confounding_audit.py \
  --input evons/data/evons.csv \
  --metrics-out report/evons_source_audit_folds.csv \
  --report-out report/evons_source_audit_report.json \
  --export-group-folds report/evons_group_folds.csv
```

This script computes:
- source-label association diagnostics (Chi² and Cramér's V)
- a **source-only baseline** under random K-fold vs group K-fold by source
- per-fold metrics (`accuracy`, `macro_f1`, `unseen_source_ratio`)
- optional row-to-fold assignment for source-grouped CV reuse in notebooks

Interpretation: if source-only performance is strong on random CV but drops on source-grouped CV, the task is likely source-confounded.


## 9. FakeNewsNet virality sensitivity audit (no embedding recomputation)

To justify the virality definition and evaluate an early-detection framing with existing preprocessed propagations:

```bash
python tools/fakenewsnet_virality_sensitivity.py \
  --input-jsonl FakeNewsNet/data/ordered_real_propagation_paths.jsonl \
  --thresholds-out report/fnn_virality_thresholds_real.csv \
  --early-out report/fnn_virality_early_real.csv \
  --summary-out report/fnn_virality_summary_real.json
```

Run it on both real and fake propagation files, then merge rows for a global view if needed.

What it reports:
- threshold sensitivity for q50/q75/q90/q95 (or custom quantiles): threshold value and class balance
- early-signal diagnostics using first `k` tweets (configurable):
  - correlation between prefix likes and final total likes
  - AUC of prefix-likes as a simple score for each threshold-defined label
  - average prefix/total engagement ratio

This enables reframing q50 as high/low engagement and testing stricter virality (q90/q95) without retraining embeddings.


## 10. Lightweight reliability checks

A minimal smoke-test suite validates that each utility script in `tools/` runs end-to-end on tiny synthetic inputs and produces expected outputs:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The same command is executed automatically in CI via `.github/workflows/tools-smoke-tests.yml` when tool-related files change.

