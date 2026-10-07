# HeartLens

HeartLens is a Streamlit app that predicts **heart disease presence** from the UCI Heart Disease dataset (920 patients, 4 hospitals) and explains the data in plain language.

## What's inside

- **Prediction**: enter a patient profile and get the model's estimate.
- **Data story**: nine short chapters, each with one chart and a plain-language takeaway, from age and sex to the exercise-test warning signs. It ends with how risk climbs from 16% with no warning signs to 94% with all four.
- **Model studio**: model comparison, held-out test results (F1, recall, ROC-AUC), ROC curve and confusion matrix.

## About the data

The original target `num` (0–4) is converted to `0 = no disease`, `1 = disease present` for every value above zero. The original target `num` (0–4) is converted to `0 = no disease`, `1 = disease present` for every value above zero.

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
python train.py
```

Training writes `artifacts/heart_model.joblib` and `artifacts/metrics.json`.

## How the model works

1. Read `heart_disease_uci.csv` and define the target as `num > 0`.
2. Use the notebook's 11 predictors. Exclude `id`, `dataset`, `ca`, and `thal`; the latter two have severe site-dependent missingness. The hospital name is used only for evaluation.
3. Convert impossible zero values in resting blood pressure and cholesterol to missing. Median imputation and scaling for numeric inputs, and mode imputation and one-hot encoding for categorical inputs, are fitted **inside** each training fold.
4. Compare logistic regression, random forest, histogram gradient boosting, and XGBoost with five-fold cross-validation on a stratified 80% training split. Select the best mean CV F1 score, breaking ties with ROC-AUC.
5. Choose the classification cutoff using out-of-fold predictions from the training split. Evaluate the chosen model once on the untouched 20% test split.
6. Run a leave-one-hospital-out check to expose dataset shift. Hyperparameters are tuned only on the other three hospitals in each fold. F1 and recall use the standard 0.50 cutoff, so the held-out hospital influences neither fitting nor cutoff choice.

The exact cross-validation comparison is written to `artifacts/metrics.json` and displayed in the app. The selected model is the winner under this project's F1 rule; ROC-AUC, interpretability, and cross-hospital results remain important tradeoffs.

The metrics shown in the app are read from the generated artifact. The model is a demonstration trained on a small historical dataset and has not been clinically validated. Its output probability estimates the dataset label, **not a person's clinical risk**.

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
- `artifacts/metrics.json`: model comparisons, held-out test metrics, ROC curve, permutation importance, and hospital checks.
- `heart_disease_uci.csv`: the UCI Heart Disease dataset.
- `tests/`: unit tests for the feature contract (`python -m unittest discover -s tests`).
