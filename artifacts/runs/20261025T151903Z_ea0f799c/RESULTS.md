# Phase 1 ML results

Generated from this run’s saved predictions. Historical holdout results are retrospective: these rows were previously inspected in the source project.

Selected family: **Random forest**, chosen by mean nested outer-fold F1 at cutoff 0.50 before cutoff/evaluation scoring.

| Model | Nested CV F1 | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8231 | 0.51 | 0.8315 | 0.8257 | 0.8824 | 0.8531 | 0.9125 |
| Random forest | 0.8358 | 0.38 | 0.8043 | 0.7538 | 0.9608 | 0.8448 | 0.9029 |
| Histogram gradient boosting | 0.8104 | 0.37 | 0.8152 | 0.7698 | 0.9510 | 0.8509 | 0.8945 |
| XGBoost | 0.8308 | 0.34 | 0.8043 | 0.7463 | 0.9804 | 0.8475 | 0.9039 |
| Bayesian network | 0.8179 | 0.54 | 0.7880 | 0.8058 | 0.8137 | 0.8098 | 0.8765 |

## D-only hospital holdouts

| Hospital | Locally selected model | N | F1 | Always-positive F1 | Specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | Logistic regression | 248 | 0.7951 | 0.6260 | 0.7481 | 0.8612 |
| Hungary | Random forest | 229 | 0.6948 | 0.5367 | 0.6207 | 0.8712 |
| Switzerland | Random forest | 96 | 0.8712 | 0.9565 | 0.5000 | 0.7472 |
| VA Long Beach | Logistic regression | 161 | 0.8561 | 0.8582 | 0.0500 | 0.7079 |

The BN graph is provisional and statistical; its edges and independence assumptions are not medically validated. Neural-network training remains Phase 2. Hospital checks exclude all historical holdout rows and repeat selection independently. Poor specificity or performance below the constant reference constrains transfer claims. Bootstrap intervals are conditional on saved fitted models; these results do not establish clinical utility.

See `verification.json`, `effective_protocol.json`, per-partition metrics/predictions, search traces and model files for the evidence.
