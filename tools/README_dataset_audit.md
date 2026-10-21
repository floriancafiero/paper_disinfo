# PLOS ONE major revision — dataset audit

This script addresses the editor's requests to document sample sizes, class balance,
propagation coverage, and viral engagement thresholds. It generates aggregated
statistics only; it does not change the training scripts or regenerate experiment results.

Run from the project root after restoring the non-redistributable raw data:

```bash
python tools/dataset_audit.py --out report/dataset_audit.json
```

To audit just one corpus:

```bash
python tools/dataset_audit.py --dataset evons
python tools/dataset_audit.py --dataset fakenewsnet
```

To count the original PolitiFact story directories as a separate denominator:

```bash
python tools/dataset_audit.py --dataset fakenewsnet \
  --raw-politifact-dir FakeNewsNet/data/raw_data/politifact
```

The output reports EVONS veracity balance and Facebook engagement coverage,
FakeNewsNet–PolitiFact fake/real counts and tweet sequence coverage, and the
median, 75th, 90th, and 95th percentile engagement thresholds and resulting
positive rates. For FakeNewsNet, thresholds are given for the pooled main
training cohort and separately for fake and real subsets used by the sensitivity
audit. Sequence coverage is measured for k = 1, 3, 5, 10 tweets.

**Important interpretation cautions:** The audit matches the main classifiers'
strict `engagement > threshold` convention; the existing sensitivity script
uses `>=`. Ties mean prevalence is not necessarily `1 - quantile`.
The processed JSONL corpus omits stories without recovered propagation in the
normal preprocessing route. Its counts alone cannot establish coverage of the
original FakeNewsNet release or API rehydration success. The optional source
directory counts are not themselves an API recovery rate. EVONS cutoffs exclude
records whose engagement is missing; check that this is the exact modeling
cohort before reporting numbers.

The script requires only Python's standard library. Run its unit tests with:

```bash
python -m unittest discover -s tests -p 'test_dataset_audit.py' -v
```

No raw data, labels, or model outputs have been committed here.
