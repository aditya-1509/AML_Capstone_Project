# AML Capstone Project: Phase 1 — Data, EDA and Machine Learning

This repository contains the Phase 1 implementation for the Advanced Machine Learning (AML) Capstone Project, focusing on **rigorous data ingestion, clinical schema validation, and reproducible exploratory data analysis (EDA)** on the multi-center UCI Heart Disease dataset.


## Completed ML workflow

[Verified results, model comparison and hospital checks](artifacts/runs/20261025T151903Z_ea0f799c/RESULTS.md).
The recorded run passed 48 tests and its 25 frozen models are checked with `verify_run.py`.

The original `modeling.py`, `train_baselines.py` and baseline tests remain intact.
The remaining Phase 1 ML implementation is adapted from the team's existing
HeartLens project, with separate `advanced_modeling.py` helpers so the original
baseline API remains available. No individual authorship of the reused code is implied.

- `train.py`: five-family nested CV (five outer / three inner folds), deterministic hyperparameter selection, final S-only fitting, cutoff selection and frozen evaluation.
- `bayesian_network.py`: discrete smoothed BN with training-only bins/imputation and missing-evidence inference. The graph is provisional, not faculty approved or causal.
- `experiment.py`: pooled and four independent D-only hospital experiments; exact fold memberships, search traces, frozen models, thresholds, predictions, bootstrap intervals, reliability bins and subgroup summaries.
- `verify_run.py`: reload saved models, reproduce probabilities and metrics, reconstruct cutoff/CV decisions and verify partition isolation.
- `summarize_run.py`: generate the result table, ROC curves and selected-model confusion matrix from saved evidence.

The original 918 retained rows and split rules are preserved: development D=734,
retrospective T=184; D gives S=550 for selection/fitting and C=184 for cutoffs.
All learned preprocessing is fitted inside its permitted training fold. The family
is chosen on S, before C or T scoring; models are never refitted on S+C.
The previously inspected historical holdout is retrospective, not a new untouched test.
Hospital experiments exclude T and repeat the full selection procedure independently.
Neural-network implementation remains Phase 2.

Run the complete workflow explicitly (about 2,940 fits; one worker):

```bash
python train.py --run-experiment --allow-provisional-bn
python verify_run.py artifacts/runs/<run_id>
python summarize_run.py artifacts/runs/<run_id>
```

On macOS, XGBoost requires an OpenMP runtime. If its import reports a missing
`libomp.dylib`, install the system OpenMP dependency or supply an existing
compatible runtime through `DYLD_LIBRARY_PATH`. The recorded run used the runtime
bundled with its scikit-learn wheel; see its `environment.json`.

Existing baseline-only execution remains available:

```bash
python train_baselines.py --output artifacts/my_baseline_run
python -m pytest tests -q
```

Protocol and graph assumptions are recorded in `config/` and `docs/`. The imported
protocol retains its historical draft/faculty questions; each executed run stores
an effective protocol declaring the provisional exploratory settings. Results
support an academic disease-presence comparison, not a clinical screening claim.

---

## 1. Project Overview & Clinical Context

Cardiovascular diseases are the leading cause of global mortality. The objective of this project is to build an auditable machine learning foundation to support coronary artery disease screening. 

The dataset aggregates clinical observations across **four international medical institutions**:
1. **Cleveland Clinic Foundation** (Cleveland, OH, USA)
2. **Hungarian Institute of Cardiology** (Budapest, Hungary)
3. **University Hospital** (Zurich, Switzerland)
4. **V.A. Medical Center** (Long Beach, CA, USA)

### Key Clinical Definitions
- **Target (`num`)**: Integer severity scale ranging from `0` (angiographic disease < 50% diameter stenosis / negative) to `1`, `2`, `3`, `4` (significant stenosis / positive).
- **Binary Outcome (`positive`)**: Defined as `num > 0` (presence of coronary heart disease) vs `num == 0` (absence of heart disease).

---

## 2. Data Ingestion & Clinical Quality Assurance

Data cleaning and validation policies are enforced deterministically in `data_loader.py` before any modeling:

### A. Deduplication
- Across all four clinical sites, **2 exact duplicate patient records** (excluding arbitrary row `id`) were detected and removed, leaving **918 unique patients** from the original 920 records.

### B. Physiological Sanity & Zero-as-Missing Cleaning
- In the raw dataset, several records contain `trestbps = 0` (resting blood pressure) or `chol = 0` (serum cholesterol). 
- Because blood pressure or serum cholesterol of 0 mg/dL is physiologically incompatible with life, these values represent unrecorded clinical tests.
- `data_loader.clean_features()` automatically converts these zero entries to `NaN` (missing) to prevent corrupting feature distributions.

### C. Predictor Scope & Exclusions
- **Retained Continuous Features (`5`)**: `age`, `trestbps`, `chol`, `thalch`, `oldpeak`
- **Retained Categorical Features (`6`)**: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`
- **Excluded Variables**:
  - `id`: Arbitrary row identifier.
  - `dataset`: Medical center identifier (retained exclusively for cross-site shift checks).
  - `ca` and `thal`: Excluded from primary predictive features because they are missing in >60% of patients across European and VA centers.

---

## 3. Key Exploratory Data Analysis (EDA) Insights

The EDA pipeline (`eda.py`) computes descriptive metrics and exports structured vector visualizations into `artifacts/`:

### A. Target Distribution & Balance
- **Total Validated Records**: 918 patients
- **Class Balance**: 
  - **Negative (`num == 0`)**: 410 patients (44.7%)
  - **Positive (`num > 0`)**: 508 patients (55.3%)
- The binary target exhibits balanced prevalence across the combined multi-center cohort, avoiding acute class imbalance.

### B. Multi-Center Site Heterogeneity
Substantial clinical differences exist across collection sites:
| Site | Total Records | Positive Count | Positive Prevalence (%) |
| :--- | :---: | :---: | :---: |
| **Cleveland** | 303 | 139 | 45.9% |
| **Hungary** | 293 | 106 | 36.2% |
| **VA Long Beach** | 199 | 151 | 75.9% |
| **Switzerland** | 123 | 112 | 91.1% |

*Takeaway*: The Swiss and VA Long Beach cohorts consist predominantly of high-risk patients with advanced coronary symptoms, whereas Cleveland and Hungarian cohorts show lower positive rates.

### C. Clinical Risk Correlates
- **Maximum Heart Rate (`thalch`)**: Shows a strong negative correlation with heart disease diagnosis ($r \approx -0.40$). Patients with heart disease achieve significantly lower peak heart rates during stress testing.
- **ST Depression (`oldpeak`)**: Positively correlated with coronary disease ($r \approx +0.40$), serving as a strong electrocardiographic biomarker.
- **Chest Pain Type (`cp`)**: Patients presenting with asymptomatic ischemia have significantly higher rates of coronary stenosis compared to non-anginal pain.

---

## 4. Repository Structure

```text
├── heart_disease_uci.csv       # Raw UCI multi-center heart disease dataset (920 records)
├── data_loader.py              # Ingestion, clinical cleaning, and schema validation
├── eda.py                      # Exploratory data analysis pipeline & plot generator
├── requirements.txt            # Dependency specification
├── .gitignore                  # Git ignore rules
├── artifacts/                  # Generated audit files and vector visual charts
│   ├── data_audit.json         # Complete structured audit metrics in JSON
│   ├── target_distribution.svg # Class balance visualization
│   ├── missing_values.svg      # Feature missingness breakdown
│   ├── positive_rate_by_site.svg # Site prevalence comparison
│   └── correlation_heatmap.svg # Inter-feature correlation heatmap
├── notebooks/                  # Interactive notebooks
│   └── eda_walkthrough.ipynb   # Guided step-by-step EDA walkthrough
└── tests/                      # Automated quality assurance tests
    └── test_data_loader.py     # Schema and validation unit tests
```

---

## 5. Getting Started & Reproducibility

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/nishtha-09-gupta/AML_Capstone_Project.git
cd AML_Capstone_Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Data Loading & Validation Tests
Execute the automated test suite to verify data integrity and cleaning rules:
```bash
pytest tests/ -v
```

### 3. Run EDA Pipeline & Recreate Artifacts
Execute `eda.py` to re-generate the full structured data audit and vector visual plots:
```bash
python eda.py
```
Outputs are saved directly into the `artifacts/` folder.
