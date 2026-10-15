# HeartLens

HeartLens is a Phase 1 machine learning project and Streamlit demonstration that classifies **heart disease presence** in the historical UCI Heart Disease dataset. The source CSV has 920 records from four hospitals; two duplicate records are removed before modeling, leaving 918.

- **Live app:** [mk-sem5-project.streamlit.app](https://mk-sem5-project.streamlit.app/)
- **Phase 1 sharing checklist:** [PHASE1_SHARE.md](PHASE1_SHARE.md)
- **One-file team bundle:** [phase1_share_bundle.zip](submission/phase1_share_bundle.zip)
- **Beginner-friendly viva PDF:** [phase1_viva_teaching_guide.pdf](output/pdf/phase1_viva_teaching_guide.pdf)

## What's inside

- **Prediction**: enter a patient profile and get the model's estimate.
- **Data story**: descriptive charts about the source cohort, clearly separated from individual risk claims.
- **Model studio**: model comparison, held-out test results (F1, recall, ROC-AUC), ROC curve and confusion matrix.

## About the data

The original target `num` (0–4) is converted to `0 = no disease`, `1 = disease present` for every value above zero. The 918 distinct records include 410 negative and 508 positive labels. The source and target definition are described by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Run locally

Use Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The trained artifact is included in `artifacts/heart_model.joblib`, so the app starts without retraining. To reproduce the experiment:

```bash
python eda.py
python train.py
```

EDA writes `artifacts/eda.json`; training writes `artifacts/heart_model.joblib` and `artifacts/metrics.json`.

## How the model works

1. Read `heart_disease_uci.csv`, drop the two duplicate records after excluding the row ID, and define the target as `num > 0`.
2. Use 11 predictors. Exclude `id`, `dataset`, `ca`, and `thal`; the latter two have severe site-dependent missingness. The hospital name is used only for evaluation.
3. Convert impossible zero values in resting blood pressure and cholesterol to missing. Median imputation and scaling for numeric inputs, and mode imputation and one-hot encoding for categorical inputs, are fitted **inside** each training fold.
4. Compare logistic regression, random forest, histogram gradient boosting, and XGBoost with five-fold cross-validation on a stratified 80% training split. Select the best mean CV F1 score, breaking ties with ROC-AUC. The selected booster leads random forest by only 0.0004 F1, so this is a rule-based selection, not strong evidence of superiority.
5. Choose the classification cutoff using out-of-fold predictions from the training split. Evaluate the chosen model once on the untouched 20% test split.
6. Run a leave-one-hospital-out check to expose dataset shift. Hyperparameters are tuned only on the other three hospitals in each fold. F1 and recall use the standard 0.50 cutoff, so the held-out hospital influences neither fitting nor cutoff choice.

The corrected held-out test has 184 records: F1 0.836, recall 0.873, and ROC-AUC 0.901 at a training-selected threshold of 0.47. There are 13 false negatives and 22 false positives. The exact comparisons and site results are in `artifacts/metrics.json` and the report. Repeated tuning on one small training split and hospital differences limit how far these results can be generalized.

The metrics shown in the app are read from the generated artifact. The model is a demonstration trained on a small historical dataset and has not been clinically validated. Its output probability estimates the dataset label, **not a person's clinical risk**.

## Phase 1 submission materials

| Requirement | File or evidence |
| --- | --- |
| Literature review and theoretical rigor | `report/phase1_report.tex` |
| Dataset quality and EDA | `eda.py`, `artifacts/eda.json`, app Data story |
| Feature engineering | `modeling.py`, report Methods section |
| Model application and validation | `train.py`, `artifacts/metrics.json`, app Model studio |
| GitHub code quality | This repository, pinned dependencies, reproducible commands, `tests/` |
| Presentation | `presentation/phase1_presentation.pptx` |
| Individual viva preparation | `VIVA.md`, `output/pdf/phase1_viva_teaching_guide.pdf` |
| IEEE format project report | `report/phase1_report.tex`, `report/phase1_report.pdf` |
| Presentation handout PDF | `presentation/phase1_presentation.pdf` |

Run `python -m unittest discover -s tests -v` to check the feature contract. The report's author line is generic because the five student names were not supplied; the team should add its actual members and accurately describe each person's contribution before academic submission.

## Inputs

| Column | Meaning | Input type |
| --- | --- | --- |
| `age` | Age in years | Number, required |
| `sex` | Sex as recorded in the source dataset | `Female`, `Male` |
| `cp` | Chest pain type | `typical angina`, `atypical angina`, `non-anginal`, `asymptomatic` |
| `trestbps` | Resting blood pressure, mm Hg | Number; blank allowed |
| `chol` | Serum cholesterol, mg/dL | Number; blank allowed |
| `fbs` | Fasting blood sugar above 120 mg/dL | `True`, `False`; blank allowed |
| `restecg` | Resting ECG | `normal`, `st-t abnormality`, `lv hypertrophy`; blank allowed |
| `thalch` | Maximum heart rate achieved, bpm | Number; blank allowed |
| `exang` | Exercise-induced angina | `True`, `False`; blank allowed |
| `oldpeak` | ST depression during exercise | Number; blank allowed |
| `slope` | Peak exercise ST segment slope | `upsloping`, `flat`, `downsloping`; blank allowed |

Blank optional fields are imputed by the fitted pipeline.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository. Include `app.py`, `modeling.py`, `requirements.txt`, `.streamlit/config.toml`, and the `artifacts` folder.
2. In Streamlit Community Cloud, create a new app from the repository, select `app.py` as the entry point, and select Python 3.13 if the deployment form offers a version choice.
3. Deploy. No API keys or secrets are needed.

The app does not store anything you enter. Avoid entering real identifiable patient information in a public demonstration.

## Project files

- `app.py`: Streamlit interface and prediction workflow.
- `modeling.py`: shared feature schema, cleaning, preprocessing, validation.
- `train.py`: reproducible training and evaluation.
- `eda.py`: reproducible dataset audit.
- `report/phase1_report.tex`: IEEE style Phase 1 report with references, methods, results, and limitations.
- `presentation/phase1_presentation.pptx`: Phase 1 presentation.
- `presentation/phase1_presentation.pdf`: viewable copy of the nine-slide deck.
- `output/pdf/phase1_viva_teaching_guide.pdf`: beginner-friendly teaching guide, worked metrics, slide notes, and viva Q&A.
- `scripts/build_phase1_viva_pdf.py`: regenerate the teaching PDF with ReportLab from the saved metrics and EDA artifacts.
- `PHASE1_SHARE.md`: ready-to-send WhatsApp message and submission checklist.
- `submission/phase1_share_bundle.zip`: report, slides, viva guide, and sharing notes in one download.
- `VIVA.md`: individual oral exam guide.
- `artifacts/eda.json`: source quality and missing-data summary.
- `artifacts/metrics.json`: model comparisons, held-out test metrics, ROC curve, permutation importance, and hospital checks.
- `heart_disease_uci.csv`: the UCI Heart Disease dataset.
- `tests/`: unit tests for the feature contract (`python -m unittest discover -s tests`).
