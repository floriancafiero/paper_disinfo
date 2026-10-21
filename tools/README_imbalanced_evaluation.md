# PLOS ONE major revision — imbalanced classification (editor point 2)

## Implemented

- `tools/imbalanced_evaluation.py`: evaluation with **ROC-AUC**, **Average
  Precision (AP)**, positive prevalence (the expected AP of a random ranking),
  positive-class F1, accuracy, and **majority-class** accuracy/F1.
- Thresholds are chosen exclusively on a **calibration subset** of outer
  training data, optimizing F1 (not on the outer test fold).
- `colab_no_embeddings/evons_virality_prediction.py`: integrated in each of
  the five EVONS virality networks, including Mistral gating. Existing
  `accuracy`, `f1`, `precision`, `recall`, and `auc` retain their
  **threshold-0.5** meanings, while `f1_tuned`, `precision_tuned`,
  `recall_tuned`, `threshold_tuned`, `pr_auc`, `pr_baseline`,
  `majority_accuracy` and `majority_f1` are appended.

## Protocol and fair comparisons

For each of 10 80/20 *stratified shuffle* outer splits, the 80% outer
training portion is further split into 75% model fit and 25% calibration
(effective split **60% fit / 20% calibration / 20% test**). The model is
trained **only** on fit examples; calibration scores and labels choose
the threshold; scores on the independent outer test split evaluate both
operating points. Different models use the same deterministic split indices
for each outer fold. Class imbalance may cause threshold ties; the larger
cutoff wins. Both classes must occur in the calibration sample.

**Critical:** this refits the classifier on 60% rather than the old 80%
of the dataset. Its new .5-threshold F1 and AUC are NOT directly comparable
to previously reported values obtained with 80% fitting samples. All
models and historical baselines must be rerun under the same protocol before
claiming an improvement due solely to threshold calibration.

Evaluation is based on the **positive class** as defined in the pipeline
(top-5%-engagement viral examples). The majority baseline almost always
predicts no positives, hence high accuracy but F1 = 0. AP = prevalence
is a random-score baseline, not a majority-class classifier's AP.

The pre-existing EVONS source-average-engagement feature uses the model-fit
articles' engagement in its own per-source mean when constructing training
features, which can create train-only target contamination. That issue
requires a separate leave-one-out/cross-fitting correction and is **not**
resolved by choosing a threshold on a separate calibration set.

## Run after acquiring data

```bash
python colab_no_embeddings/evons_virality_prediction.py
```

Results appear under `colab_no_embeddings/outputs/` when raw CSVs and
precomputed embeddings are available. Results in the repository are old
and have **not** been replaced.

```bash
python -m unittest discover -s tests -p 'test_imbalanced_evaluation.py' -v
```

## Expanded coverage (follow-up commit)

The same calibration protocol has now been integrated into the EVONS
**disinformation** standalone Python script and notebook and both
FakeNewsNet **disinformation** and **virality** notebooks. All four
notebooks use the same shared metric helper. The EVONS virality notebook
is synchronized with its revised Python script.

The EVONS virality training source-engagement feature now uses a
**leave-one-out** per-source engagement mean for model-fit items, avoiding
the direct inclusion of an article's own engagement in its own feature.
Calibration and test items use *fit-only* source means. Note that the
training-feature standardization still uses statistics of the whole fit set;
this is not sample-wise independent normalization. The source-level mean
may also proxy editorial characteristics; group-split evaluation remains
needed to characterize source confounding.

No output CSV or published result is automatically replaced: all four
models suites must be rerun with original corpus/embeddings. Raw data and
embeddings remain outside GitHub. Historical results were generated under
a different training regime and **must not** be compared unqualified to
these new results.

Validation includes source-feature unit tests and AST syntax checks of
all modified Python scripts and code cells of all four notebooks. These
checks do not constitute a real-data training run.
