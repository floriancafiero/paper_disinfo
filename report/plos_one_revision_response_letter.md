# Response to the PLOS ONE editor — revised experiments and analyses

**Working cover response; 23 October 2026.** This is organized around the five editorial issues in the working revision checklist, not a verbatim reproduction of the editor's letter. Insert the editor's original quotations, final manuscript section/page numbers, figure IDs and stable code archive identifier before submission. The findings below are based on files at `floriancafiero/paper_disinfo:plos-one-major-revision`. Final scientific and author review is required.

Dear Editor,

Thank you for the opportunity to revise “From veracity to diffusion: Operationalizing misinformation prediction beyond fake-news detection.” We have revisited the operationalization of diffusion, corrected important ambiguities in our evaluation protocol, recomputed the models using disjoint threshold-calibration data, and extended our early-window analyses. The revised files and scripts document the analyses and enable reproduction using authorized access to the underlying datasets and cached embeddings. We summarize the substantive changes below.

## 1. Operationalization, thresholds and dataset comparability

We agree that a single unqualified binary “virality” outcome cannot support claims across different platforms and base rates. We now describe the operational quantity used in each resource and define the label as `1[E > Q_q(E)]` with a *strict* greater-than threshold. We expanded the classifier grid to q = 0.50, 0.75, 0.90 and 0.95 for both datasets and report actual thresholds and positive counts, including cases where ties may change prevalence.

EVONS contains 92,969 articles in our processed export; at q95 its Facebook engagement cutoff is 18,886.2 (4,649 positives). The processed FakeNewsNet–PolitiFact files contain 578 stories. Of these, 434 stories have at least eleven recovered tweets and are used as the fixed cohort in the matched text-prefix experiments. Within this restricted cohort the pooled total-favorite thresholds at q50, q75, q90 and q95 are 140, 1,942.25, 14,313.9 and 58,985.65, respectively (217, 109, 44 and 22 positive stories). The source-specific real and fake subset thresholds used for the descriptive early-likes audit are separately documented. We explicitly distinguish Facebook article engagement from the summed favorite counts of Twitter/X posts; matching empirical quantiles does not equate these outcomes substantively.

## 2. Imbalanced metrics, baselines and threshold tuning

We have replaced reliance on classification accuracy and default-threshold F1 alone with **average precision (AP)**, ROC-AUC, positive-class prevalence (as a random-ranking AP baseline), majority-class accuracy/F1, and both standard and threshold-selected F1. The threshold is chosen to maximize F1 **only on a calibration subset**; it is not optimized using the held-out test labels. Each of ten repeated stratified holdouts uses approximately 60% of the instances to fit model parameters, 20% for threshold selection and 20% for independent evaluation.

The q95 EVONS results illustrate why the change matters: the Mistral gated model achieves ROC-AUC = 0.876 and AP = 0.345 (positive prevalence approximately 0.050), while mean F1 increases from 0.262 at decision threshold 0.5 to 0.378 with a calibration-selected threshold. Other EVONS models also benefit from calibration, reducing the apparent gap created by their previously near-zero default-threshold F1. New results were computed under the changed training fraction; we do not treat them as direct controlled comparisons to earlier 80/20 estimates.

We additionally corrected the source-average-engagement feature to remove an article's own engagement from its *fitting* feature (leave-one-out calculation). For calibration and testing, the averages use the fitting subset only. This removes direct self-target contamination but cannot by itself eliminate publisher confounding.

## 3. Early prediction and inclusion of observed engagement in the outcome

We acknowledge that the original descriptive prefix-engagement AUC predicted total engagement, which necessarily includes the early scores used as predictors. We therefore added a separate **future-only outcome** defined as the sum of favorite counts of tweets after the first k tweets. For every k, the new analysis calculates both full-outcome and future-only ROC-AUC on exactly the same selected stories, with no fitted classifier. The principal evaluation uses the 434 cascades with at least eleven recovered tweets and reports results separately for 291 real-labelled and 143 fake-labelled stories.

In the real-labelled subset, future-only median-label ROC-AUC increases from 0.544 after the first tweet to 0.670 after the tenth; at q95, it increases from 0.674 to 0.933. In the fake-labelled subset, the corresponding q95 AUC decreases from 0.622 to 0.512. Crucially, the q95 real and fake subset tests contain only 15 and 8 positives, respectively. These results therefore indicate exploratory, retrospective associations rather than a reliable, deployed early warning score. `favorite_count` is a collected snapshot; we cannot verify that the recorded engagement of tweets 1–k was observed in real time by tweet k.

We have also implemented genuinely *prefix-restricted text features* for FakeNewsNet classifiers (mean-pooled embeddings of the first 1, 3, 5 or 10 recovered tweets), evaluated against a distinct retrospective full-sequence benchmark using identical model evaluation partitions. Unlike the future-only audit, **these classifiers still target total engagement**, and this distinction is explicit throughout. The fixed >=11-tweet cohort is retrospective and not a representative arbitrary first-tweet stream.

## 4. Data provenance, cohort definition and source confounding

The revised data audit documents the actual analyzed counts: EVONS, 92,969 items (42,776 labelled fake, 50,193 real); processed FakeNewsNet PolitiFact, 578 stories (174 fake, 404 real), containing 401,689 recovered tweets; and the common matched prefix cohort, 434 stories (143 fake, 291 real). The shorter cascades excluded from the latter are listed as sample attrition, and the reuse of fixed cohort and split hashes is recorded. These are **processed-data counts**, not coverage of every original FakeNewsNet item or an API rehydration rate.

We also clarify an essential difference in label provenance: the EVONS `is_fake` field describes the associated **media source category** (fake-news source vs real-news source), rather than an independent verification of the truth of each individual article. We have qualified the EVONS “fake-news detection” result accordingly as a **source-associated** classification task. The near-ceiling EVONS scores should not be described as article-by-article fact checking, and source-grouped validation remains necessary to assess transfer to unseen publishers. FakeNewsNet uses the PolitiFact veracity source, a different collection and annotation regime. We do not treat these labels as interchangeable.

## 5. Models, reproducibility and appropriate statistical inference

The revision records the encoder identities (RoBERTa `FacebookAI/roberta-base` and Mistral `mistral-embed`), embedding pooling rules, model families, hyperparameters, split strategy, source-feature computations, metric definitions, and optional baseline-computation backend. The revised retraining included six configurations per generic model suite (MLP, logistic regression and random forest using RoBERTa/Mistral); **XGBoost was unavailable in the new run**, and older XGBoost rankings are removed from the revised main results. The optional cuML GPU implementations for logistic regression and random forest are documented as approximate algorithmic counterparts to the CPU scikit-learn versions, not exact numerical substitutes.

We have corrected the mistaken description of the former evaluation as ordinary ten-fold cross-validation: the partitions are ten **overlapping stratified random holdouts**. We now compute paired model-score differences under an **approximate** corrected-resampled t procedure and apply Holm multiplicity adjustment within declared families. The new statistical outputs contain 430 within-window model comparisons and 120 matched prefix-vs-full contrasts, based on actual revised fold metrics. For EVONS q95, the mean calibrated-F1 advantage of gated Mistral over gated RoBERTa is approximately 0.030 (Holm-adjusted p = 0.0137) and over the source-average MLP approximately 0.029 (Holm p = 0.0085). No FakeNewsNet within-window pairwise model contrast survives the adjusted testing families. At q75, several shorter-text versus full-propagation comparisons are distinguishable under the approximate test; at q50, none is. We emphasize that additional calibration partitioning is not fully accounted for by the textbook corrected-resampled variance derivation, and repeated observations and post hoc model ranking limit confirmatory interpretation.

We believe these changes sharpen the contribution of our paper. Our conclusions are now more appropriately about **how target definition, measurement, label provenance, and observational availability shape the apparent prediction task** than about a universal ability to predict true beliefs or online diffusion in real time.

Sincerely,

Francesco Paolo Savatteri, Chahan Vidal-Gorène, and Florian Cafiero

---

## Checks to complete before sending this letter

- Match each paragraph to the **original wording of the editor's and reviewers' comments**, not just the five-item working checklist.
- Replace filenames in the final response with the final revised manuscript's figure/table/section/page identifiers.
- Verify the EVONS publisher/source label mechanism with dataset authors and, if feasible, report a **source-grouped generalization** audit.
- Confirm FakeNewsNet embedding provenance (tweet identity/order) and annotate that the processed corpus is smaller than the original release.
- Inspect the final figure illustrating *future-only* prefix-outcome comparisons and ensure its caption mentions small q95 positive counts.
- Confirm final backend and all results with the colleague, then freeze the GitHub revision branch at a stable release/commit for Data Availability.
- Keep funding, disclosures, authorship and PLOS ONE reporting requirements consistent with the actual version submitted.
