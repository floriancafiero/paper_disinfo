# Final checks before resubmitting to PLOS ONE

**Current code/data audit date: 23 October 2026.** Use the version-controlled revised analyses, not the archived tables from the May manuscript. The new scripts and article-response drafts are on `plos-one-major-revision`.

## Confirmed from committed numerical artifacts

- EVONS: 92,969 observations (42,776 fake-*source* / 50,193 real-*source* labels). Percentile thresholds and positive counts: q50 230 / 46,483; q75 2,041 / 23,242; q90 8,344.4 / 9,297; q95 18,886.2 / 4,649.
- FakeNewsNet processed PolitiFact: 578 news stories, 174 fake / 404 real, 401,689 recovered tweets. Fixed >=11-tweet cohort used in all window-model comparisons: 434 stories (143 fake / 291 real). Positive story counts q50/q75/q90/q95 in that cohort: 217 / 109 / 44 / 22.
- The colleague's `training_gpu.log` ends in **All training completed**. There is also an earlier stale `training.log` / `revision_results/RUN_FAILED.txt`; those correspond to a different, failed attempt and should be marked as archived diagnostic files, not fresh training status.
- The trained current model suites have 6 classifiers, because **XGBoost was unavailable** in the run. Do not reuse old XGBoost rows for the new paper.
- Figure-generation CI and revised paired-statistics CI have both completed successfully; model/split-integrity checks are included. The numerical results already exist under `report/revised_inference`, and publication figure drafts are downloadable from GitHub Actions.

## Remaining scientific review items

1. **EVONS label provenance / source confounding — important.** The official EVONS README defines `is_fake` by media *source* rather than independent article checks. To obtain a source-association diagnostic, the colleague can run this data-only command locally:

   ```bash
   mkdir -p revision_results
   python tools/evons_source_confounding_audit.py \
      --input evons/data/evons.csv --folds 5 \
      --metrics-out revision_results/evons_source_only_fold_metrics.csv \
      --report-out revision_results/evons_source_only_report.json
   ```

   Review and share only those aggregate outputs after inspecting them. The random/grouped source-only audit **does not substitute** for training and testing text classifiers on genuinely unseen media sources. Since label is source-level, group folds may have a single class and can make ROC-AUC undefined; report those limitations rather than coercing scores. Source-first classification should be named honestly in abstract and discussion.

2. **FakeNewsNet alignment / provenance.** Have the colleague confirm that each per-tweet embedding sequence corresponds to the same news story and tweet order in the paired JSONL. Code checks structure, time ordering and hashes, but cannot prove text-to-embedding identity without provenance. Verify the original PolitiFact story denominator if describing rehydration coverage; otherwise report only **processed** sample counts.

3. **Early prediction semantics.** Distinguish a trained *textual prefix → total-likes label* benchmark from the independent descriptive *prefix favorites → later-tweet-favorites label* audit. Neither uses verified real-time favorite-count snapshots; the selected cohort additionally depends on its recovered final length. The corrected Fig. 2 shows a retrospective future-only outcome with small numbers at q95.

4. **Inferential scope.** The 60/20/20 repeated split statistics use an approximate Nadeau–Bengio correction for the model fitting/test ratio (10 overlapping splits) plus Holm testing. They do not fully account for the distinct calibration partition, hyperparameter choice or model selection. Report effect sizes and intervals as exploratory comparisons; avoid a definitive “proven best model” statement.

5. **Computation/backends.** The colleague's rerun used GPU cuML logistic regression and random forests, which are not bit-for-bit the same estimators as historical sklearn CPU baselines. Single-split benchmark discrepancies are small but insufficient for asserting identical results across all folds. Some cuML logistic-regression runs emitted line-search warnings; preserve and report their diagnostics if claiming precise convergence.

6. **Paper packaging.** Use **the PLOS ONE manuscript** (`PLOS_One_S_VG_C.pdf`, May 2026), not the distinct `report/previous_paper.tex` LREC draft. Update all May tables and Figure 2, abstract, labels, methods and data availability; PLOS ONE text and target journal metadata need final author review. Freeze a commit/release and ensure code/results are accessible from the declared public repository. Remove redundant binary caches or machine-specific logs from public distribution if not essential.

## Prepared outputs

- Replacement PLOS ONE passages: `report/plos_one_revision_manuscript_sections.md`
- Draft editor response: `report/plos_one_revision_response_letter.md`
- Revised model inference: `report/revised_inference/f1_tuned_model_pairs.csv`
- Revised matched prefix-vs-full inference: `report/revised_inference/f1_tuned_prefix_vs_full.csv`
- PLOS ONE LaTeX tables: `report/revised_figures/tables_revision.tex`
- Figures (PNG/PDF) as GitHub Actions artifact: https://github.com/floriancafiero/paper_disinfo/actions/runs/39984406815/artifacts/12515795066

**Not yet done:** an authoritative unseen-source robustness training run or raw embedding alignment verification; a final typeset PLOS ONE source manuscript with all figures/captions/references. Avoid claiming these are complete in the editor's letter.
