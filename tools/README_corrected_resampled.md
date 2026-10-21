# PLOS ONE major revision — statistical inference on overlapping splits

## What changed

The legacy `report/stats_current.csv` and LaTeX significance table use
fold-level sign flips / naive bootstrap confidence intervals, even though
the original experimental protocol is **10 repeated stratified 80/20
shuffle splits**, NOT 10 independent disjoint folds. Their inferential
interpretation must be revised. Their effect-size column is Cliff's delta,
which is included as requested but treats the two collections of ten fold
scores as unpaired samples. Interpret it descriptively, alongside paired
mean differences; it is NOT a dependence correction.

The new `tools/corrected_resampled_tests.py` implements the **Nadeau &
Bengio corrected resampled t-test** for *paired* model scores from shared
repeated holdout splits:

```
difference_i = score_A_i - score_B_i
SE_corrected = sd(differences) * sqrt(1/n_splits + test_size/train_size)
t = mean(differences) / SE_corrected
df = n_splits - 1
```

For 10 holdouts with 80% train and 20% test, the correction factor under
the square root is `0.1 + 0.2/0.8 = 0.35`, compared with `0.1` in the
usual paired t-test. The new script uses SciPy's Student-t survival and
quantile functions to provide a corrected two-sided p-value and interval,
then applies Holm correction **within each dataset/task/metric family**.

The method is an approximation, not exact inferential independence.
It is only valid for comparisons with the **same examples in each matched
train/test split**. Identical numeric fold identifiers are not sufficient
evidence by themselves: confirm dataset ordering, preprocessing and
alignment of model predictions before interpreting pairwise p-values.
Repeated splits still do not establish generalization across sources,
platforms or time periods.

The archived 80/20 fold scores can be analyzed with:

```bash
python -m pip install scipy
python tools/corrected_resampled_tests.py \
  --input combined_cv.csv \
  --output report/stats_corrected_resampled_f1.csv \
  --metric f1 --train-fraction 0.8
```

A separate GitHub Actions workflow generates the historical corrected
F1 CSV as a downloadable build artifact; it **does not overwrite** the
legacy report. New 60/20/20 evaluation scores need separate calculation:
here the model-fit fraction is 0.6, calibration fraction 0.2, test fraction
0.2. The standard two-set corrected-resampled formula has no universally
agreed plug-in correction for a nested three-part procedure: treat any
application of the 80/20 correction to these NEW scores as approximate and
justify the estimand and split design before reporting significance.

## What remains to do before final paper tables

- Confirm that paired fold IDs correspond to *exactly* the same dataset
  records across compared models and that no embedding cohort truncation
  changed the analyzed rows.
- Generate the corrected F1 CSV; **replace paper significance claims** if
  Holm-adjusted p-values change, including any claims of gating_mistral's
  statistically significant advantage.
- Rerun all four tasks on original data under the revised common protocol.
- Report actual dataset sizes, embedding coverage, thresholds and AP.
- No raw tweets, embeddings or private data are included in this commit.
