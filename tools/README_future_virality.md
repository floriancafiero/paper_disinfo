# Prospective engagement audit for PLOS ONE (editor point 3)

The original Fig. 2 CSVs in `report/fnn_virality_early_{fake,real}.csv`
must be retained as **total-engagement** analyses: early engagement is part
of the outcome. `tools/fakenewsnet_future_virality.py` provides a separate
**future-only** target, requiring no embeddings or model training.

For every story and observation count `k`:
- **Score:** sum of `favorite_count` for the first `k` tweets.
- **Prospective outcome:** sum of `favorite_count` strictly after tweet `k`.
- **Binary target:** outcome strictly greater than the within-cohort q-th
  percentile, q = .50/.75/.90/.95. Tied thresholds can lower the positive rate.
- **ROC-AUC:** rank-based AUC for the raw early-engagement score; no classifier
  and no model selection. Also report average precision (AP), positive-class
  prevalence, cutoff, and class counts.

Run from the project root, once for each PolitiFact veracity subset:

```bash
python tools/fakenewsnet_future_virality.py \
  --input-jsonl FakeNewsNet/data/ordered_real_propagation_paths.jsonl \
  --out report/fnn_future_real.csv \
  --summary-out report/fnn_future_summary_real.json

python tools/fakenewsnet_future_virality.py \
  --input-jsonl FakeNewsNet/data/ordered_fake_propagation_paths.jsonl \
  --out report/fnn_future_fake.csv \
  --summary-out report/fnn_future_summary_fake.json
```

Main specification: `--cohort fixed` (default). Use the **same stories**
for all k by selecting sequences with length > max(k): for the usual
`1,3,5,10`, stories must have at least 11 tweets, so future engagement is
observable at k=10. The CSV reports `n_sequences` and `n_positive` for
each row; the JSON summary reports attrition from the processed corpus.
No longer sequence is silently padded or counted as zero future activity.

Sensitivity analysis: `--cohort eligible` retains stories with length > k
separately at each k; here cohort composition varies with k, so trends
may reflect selection effects.

**Interpretation and limitations:**
- The endpoint is observed *future favorite counts*, not retweet spread or
  belief change. Its observation horizon ends with the last recovered tweet.
- Thresholds are calculated separately for each k and cohort, and separately
  for real and fake subsets. This tests prospective ranking for k-specific
  outcomes, not whether a fixed absolute number of likes is reached.
- Unlike the earlier full-dataset analysis, this excludes very short cascades
  from the fixed-cohort evaluation. Do not directly compare its AUC with
  original Fig. 2 as if only the target changed; the population changes too.
  For strict comparisons, run the original total-engagement calculation on
  the identical fixed cohort as a paired companion analysis.
- An AUC of NaN signals a degenerate binary target, not poor ranking.
- Scores are calculated without fit, so in-sample rank AUC is not subject to
  fitted-model leakage. These results are descriptive and not causal.

Test with `python -m unittest discover -s tests -p 'test_future_virality.py' -v`.
Data are not checked into this repository. Real results and the revised
figure must be generated when the raw ordered JSONL files are available.
