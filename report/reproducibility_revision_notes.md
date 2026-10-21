# PLOS ONE revision — reproducibility details and claim limitations

This records what can be verified from the repository **without access to
data or rerunning experiments**. All reported numbers still need to be
computed and checked against the actual article revision.

## Preprocessing and textual evidence

| Dataset | Input used by deployed text pipelines | Encoder(s) | Pooling |
|---|---|---|---|
| EVONS | News title and description (encoded separately) | `FacebookAI/roberta-base` or Mistral API `mistral-embed` | RoBERTa first-token final hidden state; title and description vectors concatenated for MLP; gating blends them with a learned gate |
| FakeNewsNet PolitiFact | Text of each recovered propagation tweet | `FacebookAI/roberta-base` or Mistral API `mistral-embed` | Per-tweet embedding first; news-level text classifier receives mean of **all** recovered tweet embeddings |

The Mistral API model string is precisely `mistral-embed` (rather than a
chat/instruction model). Embedding scripts use batches of 256 for Mistral;
the FakeNewsNet RoBERTa preprocessing script defaults to batch size 32.
The classifiers expect 768-dimensional RoBERTa features or 1024-dimensional
Mistral features, subject to checking actual stored tensor shapes.

**Crucial:** The FakeNewsNet mean embedding uses the **entire** recovered
propagation sequence, rather than a prefix available at first posting.
Therefore the full-sequence classifier experiment should not be advertised
as deployable **early virality prediction**. The separate first-k-tweet
audits do measure an early observational signal. This distinction matters
for Fig. 2 and the comparison with EVONS article-level content.

## Models and evaluation

- EVONS virality: five classifiers: text MLP; source-embedding MLP;
  source-average-engagement MLP; gated RoBERTa; gated Mistral.
- EVONS veracity and FakeNewsNet tasks: MLP plus logistic regression,
  random forest and (when XGBoost is installed) gradient-boosted trees.
- EVONS MLP: two hidden dense layers, widths 256 and 128, ReLU and
  dropout 0.3; binary cross-entropy from logits, Adam learning rate
  0.001, batch size 64, typically 10 epochs for virality and 12 for
  disinformation. Gating network uses a separate 128-dimensional hidden
  layer. Classical RF uses 250 trees. XGBoost uses 250 estimators, max
  depth 6, learning rate 0.05, row/column fractions 0.8.
- Historical evaluation: ten *stratified shuffle* 80/20 holdouts seeded
  with 42. The revised runs separate each outer split into 60% fit,
  20% calibration, 20% external evaluation. **Not** ten disjoint folds.
- Historical threshold: EVONS 95th percentile of `fb_engagements`;
  FakeNewsNet virality median of summed `favorite_count` over recovered
  tweets. Both training tasks use **strict greater-than** the cutoff.
- Threshold selection in revised scripts uses **calibration F1** only.
  Positive-class Average Precision and empirical positive rate are now
  available. High majority-class accuracy is not evidence of useful
  detection at approximately 5% prevalence.

## Critical comparability limits

1. Quantile-matched labels can equalize the event base rate but **cannot
   eliminate the difference between Facebook engagement counts and
   Twitter/X favorites, article-level text versus propagation-tweet text,
   or temporal observation windows**.
2. Separate real/fake FakeNewsNet analyses are deliberate sensitivity
   checks; they do not represent a single pooled virality prevalence.
3. Report counts at each processing stage: original news items, retrieved
   stories, those with recovered tweets, those with embeddings, those used
   in actual model training. Rehydration is often incomplete, so the last
   cohort must not be described as the original dataset.
4. Dataset-provided veracity labels can encode source/selection biases.
   The existing EVONS source-confounding audit compares random and
   source-grouped splits; results are needed to judge whether the classifier
   identifies sources rather than more general falsehood cues.
5. The EVONS source-mean training feature has been revised to leave the
   record's engagement out of its own source mean; older results should
   not be compared as if this feature engineering had been unchanged.
6. The late-stage revised runs must avoid future information when selecting
   thresholds, fitting scaling, and computing source statistics.

## Still needed for a factual paper rewrite

- Exact valid article counts, fake/real class fractions and propagation
  coverage: run `tools/dataset_audit.py` with the licensed local datasets.
- Observed EVONS engagement cutoffs (not only the .95 quantile).
- Recomputed results under common quantile grid and split protocol.
- Prospective Fig. 2 CSVs from
  `tools/fakenewsnet_future_virality.py` and fair same-cohort total labels.
- Clarification from dataset documentation of label provenance and source
  annotation methodology, not just assumptions from the scripts.
- Version identifiers for API models or archived embeddings **as used
  historically**: the repo records the API model string, not a
  reproducible immutable server-side model snapshot.

This audit is a source-code description, NOT a successful empirical
replication. Do not fill missing manuscript numbers by estimation.
