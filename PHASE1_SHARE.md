# Phase 1 submission and WhatsApp share pack

This repository contains the **Phase 1 machine learning** work for HeartLens. Deep learning and a hybrid approach are Phase 2 topics and are not part of this submission.

## Send this message to your team

> **HeartLens — CSAI Phase 1 project**
>
> Live Streamlit demo: https://mk-sem5-project.streamlit.app/
>
> GitHub source and reproducibility: https://github.com/aditya-1509/mk_sem5_project
>
> One-file download (report + slides + viva guide): https://github.com/aditya-1509/mk_sem5_project/blob/main/submission/phase1_share_bundle.zip
>
> 9-slide presentation (PDF, easy to view on phone): https://github.com/aditya-1509/mk_sem5_project/blob/main/presentation/phase1_presentation.pdf
>
> Editable presentation (PPTX): https://github.com/aditya-1509/mk_sem5_project/blob/main/presentation/phase1_presentation.pptx
>
> IEEE-format Phase 1 report (PDF): https://github.com/aditya-1509/mk_sem5_project/blob/main/report/phase1_report.pdf
>
> Report LaTeX source: https://github.com/aditya-1509/mk_sem5_project/blob/main/report/phase1_report.tex
>
> Beginner-friendly viva teaching PDF (read this first): https://github.com/aditya-1509/mk_sem5_project/blob/main/output/pdf/phase1_viva_teaching_guide.pdf
>
> Quick viva notes: https://github.com/aditya-1509/mk_sem5_project/blob/main/VIVA.md
>
> **Before submitting:** all five of us should review the PDF, run the app once, practise explaining EDA, feature engineering, the four-model comparison, test metrics, and limitations. We must add our real five names and truthful individual contributions to the report/presentation/contribution record. Please check the faculty's actual upload or form instructions; this message shares materials but does not submit them to a portal.

## What fulfils each Phase 1 requirement

| Faculty requirement | Evidence to share or demonstrate |
| --- | --- |
| Literature review | `report/phase1_report.pdf`, Section I and references; teaching PDF, Section 1 |
| Dataset quality and EDA | `eda.py`, `artifacts/eda.json`, Streamlit **Data story**, report Section II |
| Feature engineering | `modeling.py`, report Section III, teaching PDF Sections 3–4 |
| Theoretical rigor | Report Methods/Results, teaching PDF Sections 5–9 and worked metric formulas |
| Model application | `train.py`, `artifacts/metrics.json`, `artifacts/heart_model.joblib`, Streamlit **Prediction** and **Model studio** |
| GitHub repository and code quality | Public repository, pinned `requirements.txt`, reproducible commands, `tests/` |
| Project presentation | `presentation/phase1_presentation.pptx` or viewable PDF |
| Individual viva | `VIVA.md`, teaching PDF Sections 10–14; each student must understand the whole project |
| IEEE/LaTeX report | `report/phase1_report.tex` and compiled `report/phase1_report.pdf` |

## Facts everyone should be able to say

- The original 920 rows become 918 after removing two duplicate records while ignoring row ID.
- The binary target is `num > 0`; there are 508 positive and 410 negative records.
- Eleven predictors are used; `id` is an identifier, hospital is reserved for a shift check, and `ca`/`thal` have large missingness.
- Numeric imputation, missing indicators, scaling, categorical imputation, and one-hot encoding are fitted inside the training folds.
- A stratified split gives 734 train and 184 held-out test rows. Five-fold CV compares logistic regression, random forest, histogram gradient boosting, and XGBoost.
- Histogram gradient boosting is deployed by the predefined highest-CV-F1 rule. Its 0.0004 F1 lead over random forest is too small to claim convincing superiority.
- The 0.47 threshold was selected on training out-of-fold predictions. Held-out test: F1 0.836, recall 0.873, ROC-AUC 0.901; TN 60, FP 22, FN 13, TP 89.
- Separate leave-one-hospital-out checks show performance variation. The app is an educational demonstration, not a clinically validated diagnostic tool.

## Final team action before academic submission

1. Add exactly five real student names in the IEEE report and presentation. The current generic author label is a placeholder.
2. Fill the five-person contribution table in `VIVA.md` with actual work and evidence. Do not invent contributions.
3. Confirm that each member can explain and demonstrate the full pipeline, not only their own section.
4. Use the faculty's actual submission channel and any required form; the project instructions shared here do not specify a Phase 1 upload link or evaluation date.
