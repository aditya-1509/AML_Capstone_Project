# Stage 4 exploratory experiment results

Status: complete. Measured experiment time: 340.927 seconds (environment setup/tests excluded).

Selected family: **Random forest**, selected on S by mean nested-CV F1 at cutoff 0.50. Test rankings did not change this choice.

## Retrospective evaluation

Retrospective evaluation on the previously reported and inspected historical holdout. The holdout is not a newly untouched or external test.

| Model | Nested CV F1 | C cutoff | Accuracy | Recall | F1 | ROC-AUC | Brier | Search/refit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8231 | 0.51 | 0.8315 | 0.8824 | 0.8531 | 0.9125 | 0.1227 | 2.964 |
| Random forest | 0.8358 | 0.38 | 0.8043 | 0.9608 | 0.8448 | 0.9029 | 0.1308 | 35.755 |
| Histogram gradient boosting | 0.8104 | 0.37 | 0.8152 | 0.9510 | 0.8509 | 0.8945 | 0.1318 | 21.472 |
| XGBoost | 0.8308 | 0.34 | 0.8043 | 0.9804 | 0.8475 | 0.9039 | 0.1273 | 6.837 |
| Bayesian network | 0.8179 | 0.54 | 0.7880 | 0.8137 | 0.8098 | 0.8765 | 0.1544 | 0.851 |

Rows: 920 source, 918 after unchanged deduplication. Historical split: D=734, T=184. D is split into S=550 and C=184. Models fit only S; five outer/three inner CV folds select families/configurations. C chooses cutoffs over 0.00–1.00 by F1. No S+C refit.

Both default-0.50 and C-selected-cutoff metrics are in pooled/metrics.json. Point estimates above use C-selected cutoffs. Five-bin reliability summaries and 2000 paired row-bootstrap percentile intervals are saved; no recalibration was fitted. Intervals condition on fitted models and omit training-selection and unknown patient-cluster uncertainty.

## Independent hospital checks

Every hospital check uses only D. Each develops all five families on the other hospitals, selects locally, saves/fixes cutoffs, then evaluates its held-out D rows. No pooled winner, learned bins, fitted model or cutoff is imported.

| Held-out D hospital | S_H | C_H | Held-out rows | Local selected family | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Cleveland | 363 | 122 | 249 | Logistic regression | 0.7547 | 0.8656 |
| Hungary | 379 | 127 | 228 | Histogram gradient boosting | 0.6494 | 0.8416 |
| Switzerland | 478 | 160 | 96 | Random forest | 0.8712 | 0.7472 |
| VA Long Beach | 429 | 144 | 161 | Logistic regression | 0.8561 | 0.7079 |

All five family results for every hospital are saved in its metrics.json; the table reports each locally selected family. These are D-only denominators, not the historical full-dataset hospital experiments.

## Verification and limitations

The original 31-test suite passed before edits; the complete 33-test suite passed after adding explicit-guard and bootstrap checks. Post-run verification passed for 25 serialized models and 2940 completed fits reconstructed from search records. Saved-model probabilities, metrics, cutoff selection and partition memberships were checked independently.

The BN remains provisional and unapproved. All BN edges remain explicit statistical assumptions, not medical causal evidence. The only BN setting was the already documented alpha=1 default with existing training-only tercile bins. The run-specific effective_protocol.json records this user-authorized exploratory override; original protocol/graph JSON files were not changed. Search budgets differ between families. Faculty questions and source provenance remain unresolved. No new clinical-validation claim is made.

The initial XGBoost import failed to find OpenMP. The process used scikit-learn’s existing libomp.dylib from the disposable environment via DYLD_LIBRARY_PATH, without a system installation. A macOS physical-core-detection warning fell back to logical-core counting; it is saved in manifest.json. All model fits completed successfully.

Environment: see environment.json and requirements-freeze.txt. Exact executed source is in code/; code/data hashes, times, warnings and failures are in manifest.json. Partition/fold manifests were written before fitting and freeze.json files before evaluation. Historical models/results and academic deliverables were not overwritten.
