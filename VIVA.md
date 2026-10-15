# Phase 1 individual viva guide

Every team member should run the project, inspect the code and artifacts, and be able to explain these answers without reading them aloud. Record actual member contributions separately before submission.

## Problem and data

1. **What is the prediction target?** `num > 0` is positive; `num = 0` is negative. This collapses the original 0–4 scale and loses severity information.
2. **Where did the data come from?** The UCI Heart Disease compilation combines Cleveland, Hungary, Switzerland, and VA Long Beach. The supplied CSV has 920 rows; 2 repeated records, ignoring ID, are removed before splitting.
3. **Why remove duplicates before the split?** An identical source record appeared on both sides of the original split. That can make test performance look better because the model has already seen the same features and label.
4. **Why exclude `ca` and `thal`?** They have 609/918 and 484/918 missing values after deduplication, respectively, with strong site dependence. `id` is an identifier, and `dataset` is held back for a transport check.
5. **What does EDA show?** The analysis data contain 508 positive and 410 negative records. The positive-label share ranges from 36.3% in Hungary to 93.5% in Switzerland. Cholesterol has 201 missing values after zero cleaning, and ST slope has 307.

## Feature engineering and validation

6. **How are missing values handled?** Zero pressure/cholesterol readings become missing. Training-fold medians fill numerical values, with missingness indicators. Training-fold modes fill categorical values. Categories are one-hot encoded.
7. **What is leakage?** Information from validation/test records influencing model fitting or preprocessing. The pipeline fits imputers, scaler, encoder, and classifier within each training fold; duplicates are removed before splitting.
8. **Why use stratification?** It keeps the binary class proportions similar in the 734-row train and 184-row test partitions.
9. **How were models chosen?** Four model families were tuned by five-fold stratified CV on training records. Mean F1 was the selection rule; ROC-AUC broke ties. Histogram gradient boosting led random forest by only 0.0004 F1.
10. **Why is this difference uncertain?** It is far smaller than the fold standard deviations (about 0.029 and 0.032). The current experiment cannot establish meaningful superiority of one of these two model families.
11. **How was the cutoff chosen?** The selected model generated out-of-fold probabilities on training records. Among cutoffs 0.30–0.70, 0.47 gave the highest training F1. No test label was used to select it.
12. **What is the limitation of cutoff tuning?** F1 is a dataset-specific tradeoff. It does not encode clinical costs of false negatives and false positives, and the probability has not been clinically calibrated.

## Results and interpretation

13. **What does the confusion matrix mean?** Of 82 negative test records, 60 were correctly negative and 22 false positive. Of 102 positive records, 89 were correctly positive and 13 false negative.
14. **What are the main test metrics?** F1 0.836, recall 0.873, ROC-AUC 0.901, precision 0.802. Recall is `TP/(TP+FN)`; precision is `TP/(TP+FP)`; F1 is their harmonic mean.
15. **Why report ROC-AUC too?** It measures ranking discrimination over all thresholds. A single F1 value depends on the chosen threshold.
16. **What does permutation importance mean?** Shuffling a feature on the test set and measuring ROC-AUC change estimates the fitted model's reliance on it. It is affected by feature correlation and is not a causal or clinical importance ranking.
17. **What does leave-one-hospital-out test?** A separate model is tuned using three hospitals and evaluated on the fourth at cutoff 0.50. ROC-AUC ranges from 0.702 to 0.864, showing sensitivity to site shift. These runs are separate from the random-split model.
18. **Can this diagnose someone?** No. The data are historical and selected, external prospective validation is absent, and calibration and clinical utility have not been established.

## Demonstration sequence

1. Run `python eda.py`; compare `artifacts/eda.json` with the report's cohort counts.
2. Run `python train.py`; explain the selected model, threshold, confusion matrix, and site checks in `artifacts/metrics.json`.
3. Run `streamlit run app.py`; show the input validation, one educational prediction, the Data story, and Model studio.
4. Open the source and point to `modeling.py` for the fold-safe preprocessing, `train.py` for model comparison, and `app.py` for inference.

## Team record to fill with actual work

The submission requires exactly five students. Add their names and specific work here before submission; do not assign contributions based on assumptions.

| Student | Actual implementation or analysis | Evidence (commit, section, or experiment) | Viva topic practiced |
| --- | --- | --- | --- |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
