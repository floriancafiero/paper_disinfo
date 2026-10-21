# Draft response to PLOS ONE editor — major revision

**Working draft, not ready for submission.** This lists verified code changes
and uses placeholders wherever reruns or source documents are needed.
Do not claim the empirical checks below are complete until their outputs
have been generated and inspected.

## 1. Virality operationalization and comparable percentile grids

Thank you for highlighting the ambiguity in our threshold definition.
The existing EVONS training implementation uses the strict
`fb_engagements > empirical 95th percentile` rule, whereas
FakeNewsNet–PolitiFact uses a strict `total favorite_count > median`
rule. We have added a dataset audit reporting percentile cutoff values,
observed positive counts, actual class prevalence, and ties at the
threshold. We have also parameterized both modeling pipelines for common
q = 0.50, 0.75, 0.90, and 0.95 thresholds, retaining the same strict
comparison. The original platform-specific engagement measures remain
different, and we will not equate these signals solely on the basis of
matching positive rates.

**Fill after reruns:** EVONS actual 95th-percentile cutoff [TBD],
positive count [TBD] / N [TBD], and matched-quantile result table [TBD].

## 2. Class imbalance and operating thresholds

We agree that default cutoff 0.5 F1 is insufficient to characterize an
imbalanced classification problem. The revised evaluation pipeline
reports Average Precision (area summarized by AP), the prevalence
baseline for random ranking, ROC-AUC, majority-class accuracy/F1,
and F1 at both the conventional 0.5 and an internally calibrated cutoff.
For each stratified 80/20 outer holdout we create a disjoint training
and calibration partition inside the 80% portion (60% fit, 20%
calibration, 20% external test). The calibration label information is
never used when fitting model parameters, and test labels do not affect
the selected cutoff.

**Fill after reruns:** Revised per-model scores and confidence
intervals [TBD]. The different model fitting size means historical
80/20 estimates are not directly comparable with this new protocol.

## 3. Early observation window and prospective target

We agree that AUC on the original total-engagement target is partly
mechanical because the early engagement scores are included in that
target. The original figure uses a direct rank-based AUC of the summed
likes in the first k observed tweets, not the predictions of a trained
classifier. A new prospective audit uses `sum(likes after tweet k)`
as its outcome. We also compute the original total-engagement AUC on
the **same propagation cohort**, compare both at k = 1, 3, 5, and 10,
and report thresholds, counts, Average Precision, and the availability
of future observations. The principal comparison keeps only
propagations with at least 11 recovered tweets, so all k values have
an observable future window; a second analysis allows the eligible
cohort to vary by k.

**Fill after reruns:** Revised figure and prospective-vs-total curves
[TBD]. The measure is descriptive and does not establish causality or
time-based forecasting beyond the recovered tweet sequence.

## 4. Samples, data provenance and scope

We have prepared a dedicated audit of EVONS and the FakeNewsNet
PolitiFact subset, including veracity class counts, engagement data
availability, tweet propagation length/coverage, and the fraction of
processed stories eligible for each early-observation window. The
comparison between the datasets is explicitly qualified: EVONS embeds
news title and description, while FakeNewsNet embeds recovered
propagation tweets; the source platform, observation units, feature
availability and outcome definitions also differ. Incomplete
propagation rehydration means the processed sample must not be
described as the original dataset's complete coverage. To assess
source-related veracity confounding, the repository additionally
contains a source-only versus group-by-source validation audit.

**Fill after inspection of data and original dataset documentation:**
Counts at each filtering stage [TBD], source-level veracity label
provenance [TBD], source-grouped audit results [TBD], and explicit
selection criteria [TBD].

## 5. Embeddings, pooling, hyperparameters and significance

The Mistral embeddings are obtained using the API model identifier
`mistral-embed`; RoBERTa is `FacebookAI/roberta-base`.
EVONS text representations are based on title/description embeddings,
whereas FakeNewsNet first embeds each tweet and then averages tweet
embeddings over the recovered propagation. In particular, **the
full-propagation FakeNewsNet classifier has access to late text and
should not be interpreted as an early predictor**. A code-grounded
description of model architecture, batching, hidden units, optimizer,
dropout, epoch counts and evaluation partitions is now included in
`report/reproducibility_revision_notes.md`.

Because the original ten 80/20 stratified shuffle splits have
overlapping training/test partitions, treating fold scores as
independent gives overly optimistic inferential summaries. We have
added a paired corrected-resampled t-test (Nadeau–Bengio) using the
variance factor `1/10 + 0.2/0.8 = 0.35`, with Holm-adjusted
p-values and corrected confidence intervals. Cliff's delta is
included as a **descriptive**, not overlap-adjusted, effect size;
the paired standardized difference is also reported.

**Fill after verification:** Corrected statistical table from GitHub
Actions artifact [TBD], revised inferential wording [TBD], final
hyperparameter table [TBD]. Confirm that matching fold IDs refer to
identical input records before interpreting the corrected paired
tests, and verify dataset-specific source label provenance.

## Outstanding blockers

- Original EVONS records, PolitiFact propagation JSONLs, and precomputed
  text embedding tensors, all in the paths expected by the scripts.
- Execution of the quantile grid, internal calibration and prospective
  future-engagement audit on those data.
- Confirmation of matched training/test records across model variants.
- Expert reading of dataset annotation provenance and updated paper text.
- Replace all [TBD] placeholders with measured results before submission.

The branch implementing these preparations is
`floriancafiero/paper_disinfo:plos-one-major-revision`.
