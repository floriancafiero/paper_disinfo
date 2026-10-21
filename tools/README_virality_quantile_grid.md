# PLOS ONE point 1 — cross-dataset quantile-grid reruns

The EVONS and FakeNewsNet viral tasks now take the same
`VIRALITY_QUANTILE` setting (default EVONS 0.95, FakeNewsNet 0.50,
preserving each dataset's original operationalization).

Quantiles `0.50, 0.75, 0.90, 0.95` share the **strict greater-than**
definition, with actual positive rates potentially below 1-q because of
tied engagement values. Results are written into separate dataset/q
directories to prevent overwriting the original reports.

After obtaining the EVONS CSV, PolitiFact ordered propagations, and
their respective precomputed embeddings:

```bash
python tools/run_virality_quantile_grid.py --dataset both
```

Or run a single task and quantile:

```bash
VIRALITY_QUANTILE=0.9 python colab_no_embeddings/evons_virality_prediction.py
```

The runner executes the FakeNewsNet notebook via `nbclient` using
the repo root as its working directory. The notebook environment needs
`nbclient`, `nbformat`, `ipykernel` and the repository's ML requirements,
plus correct tensor paths. No encoder/API calls are made.

**Critical caveats:**
- The experiments now use 60% fit / 20% calibration / 20% test, unlike
  the original 80/20 runs. Results across quantiles are comparable
  *within the revised protocol*, not directly to historical tables.
- Matching event quantiles does not make Facebook article interactions
  and Twitter tweet favorites, or their time horizons, equivalent.
- Before publishing, report per-q **actual cutoff value, number of
  positives, actual positive rate, AP, calibrated F1 and majority-class
  baseline**, plus the population used in each experiment.
- The available FakeNewsNet text pipeline pools **all** recovered tweet
  embeddings: this is a full-propagation text benchmark, not an early
  real-world deployable predictor.
- Older FakeNewsNet sensitivity CSVs used `>=`; regenerated ones now
  use the classifiers' strict `>`, so archived files must not be
  silently presented as newly harmonized results.
- Quantile thresholds are defined globally over the fixed modeling
  cohort (as in the historical implementation); this is a retrospective
  relative-outcome definition, not an absolute threshold that could
  necessarily be applied to a live stream of new articles.
- The runner was syntax-checked without access to raw tensors. Outputs,
  absolute cohort matching, and numerical model results still require
  empirical execution and manual validation.

The runner does not call any embedding generator or require Mistral
API credentials.
