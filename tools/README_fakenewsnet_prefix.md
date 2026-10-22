# FakeNewsNet text prefix experiments

Both FakeNewsNet notebooks now train on the mean of the **first k recovered
tweet embeddings**, for k = 1, 3, 5, 10. A separately labelled `full` window
uses all recovered tweets as a retrospective benchmark. The shared
implementation is `tools/fakenewsnet_prefix.py`.

## Inputs and checks

Reuse the existing ordered PolitiFact JSONLs and cached RoBERTa/Mistral
**per-tweet** embeddings. No embedding generation or API calls are needed.
Already pooled article-level vectors are rejected. Checks cover story counts,
tweet counts per story, embedding dimensions/finiteness, and chronological
`created_at` ordering; no silent truncation or automatic reordering is allowed.
Confirm with the data owner that embeddings were produced from these exact
JSONLs in the same order. Matching dimensions cannot prove that provenance.

All windows and both encoders use the same fake-then-real cohort: stories
with **at least 11 recovered tweets**, even if only k=1 is requested. Counts
before/after selection and exclusions are written to the CSVs. Cohort and
split hashes allow comparison without exporting text or individual tweet IDs.
The cohort selection uses final recovered sequence length, so this experiment
does not establish deployability on all stories at an arbitrary early time.

## Outcomes and evaluation

- Virality: **total** likes over the recovered propagation, strictly greater
  than the cohort quantile. The target includes prefix likes. Common quantiles
  remain .50/.75/.90/.95; the target and cutoff are fixed across windows at
  each quantile. Missing/null favorite counts are zero; invalid/nonfinite or
  negative counts are rejected. Future engagement remains a separate audit
  in `fakenewsnet_future_virality.py`, not a second trained-classifier target.
- Detection: the existing fake/real labels.
- Inputs: text embeddings only. Snapshot `favorite_count` values are never
  model features and are not asserted to have been available at tweet k.
- Ten repeated stratified splits: 60% fit, 20% threshold calibration, 20%
  test. Exact split arrays are reused across windows, models and encoders.
  Model seeds are explicit. Insufficient class counts stop the run rather
  than changing the quantile or protocol.
- Metrics: ROC-AUC, AP (`pr_auc`), prevalence baseline, majority baseline,
  conventional and calibration-tuned operating metrics. Summaries group by
  **model and observation_window**. `n_positive` counts test positives;
  `n_positive_cohort` and `positive_rate_cohort` describe the selected cohort.

## Running and outputs

Run notebooks with the repository root as working directory. By default all
five windows are trained, increasing FakeNewsNet training work approximately
fivefold relative to one full-propagation run.

```bash
python tools/run_virality_quantile_grid.py --dataset fakenewsnet
python tools/run_virality_quantile_grid.py --dataset fakenewsnet --quantiles 0.5 --fnn-windows 1,10,full
python tools/colleague_revision_run.py --stage train --fnn-windows 1,3,5,10,full
```

For direct notebook execution set `FNN_WINDOWS` (default `1,3,5,10,full`).
Allowed values are a nonempty, duplicate-free subset of those windows.
The runner's `--fnn-windows` explicitly overrides the notebook environment.

New results use these directories below `colab_no_embeddings/outputs/`:

- `fakenewsnet_virality_prefix_q50` (and q75/q90/q95).
- `fakenewsnet_disinformation_prefix`.

Existing CSV basenames are preserved. Each per-fold record identifies its
window (`k1`, `k3`, `k5`, `k10`, `full`), target, cohort statistics and hashes;
virality records also report quantile and engagement cutoff. Notebook outputs
are cleared in source control and executed notebooks are not saved by runners.
The colleague archive includes only approved aggregate filenames, including
the new directories; historical FakeNewsNet output directories are excluded.

Neither the new `full` benchmark nor the prefix results are directly
comparable to historical tables: cohort selection and training protocol differ.
Report coverage and describe results as using the first k **recovered** tweets,
not as a fixed elapsed-time forecast. The source-label limitations for detection
still apply. Real-data reruns are required before updating paper conclusions.
