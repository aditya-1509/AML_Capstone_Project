# HeartLens - Project 15, Team T008, Section C

**Assigned title:** Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network.

**Implemented Phase 1 scope:** historical disease-presence classification using four conventional classifiers and a **provisional Bayesian Network**. Neural-network implementation belongs to Phase 2 and is not a missing Phase 1 experiment. Faculty still needs to review whether this provisional statistical graph meets the domain-structure requirement.

Research question: how does a small probabilistic dependency model compare with Logistic Regression, Random Forest, Histogram Gradient Boosting and XGBoost under a common, leakage-controlled protocol?

## Team

Newton School of Technology, Rishihood University.

- Nishtha Gupta - 2401010303
- Aditya Srivastava - 2401010039
- Aanya Mehrotra - 2401010006
- Shrijan Sanidhya - 2401010449
- Abhay Mallik - 2401020002

Individual contributions and supporting evidence must be supplied truthfully in the viva team table.

## Verified evidence and app

The source of truth is `artifacts/runs/20261020T193544Z_2a8a1491/`, especially `RESULTS.md`, `manifest.json`, `verification.json`, `pooled/metrics.json`, saved per-row predictions and frozen models. Legacy `artifacts/metrics.json` and `artifacts/heart_model.joblib` remain historical records and do not supply the current app's results.

- **Prediction:** saved Random Forest with cutoff **0.38**; expandable provisional BN comparison with cutoff **0.54**.
- **Data story:** descriptive cohort patterns, not individual risk or medical causes. This run did not calculate permutation importance.
- **Model studio:** saved five-family nested-CV comparison and retrospective scores, ROC and confusion matrix. It does not train models or select by test ranking.

The existing [hosted app link](https://mk-sem5-project.streamlit.app/) may show an older deployment. These local changes have not been committed, pushed or deployed. Enter fictional, non-identifiable profiles only; inputs remain in the Streamlit session and are not intentionally written to disk by the app.

## Dataset and features

The supplied CSV has 920 rows and 16 columns. Removing the later two duplicate records while ignoring row ID leaves **918 records: 508 positive and 410 negative**. Only validated targets in {0,1,2,3,4} are accepted; `num > 0` defines disease presence and discards severity. The recorded hospital names are Cleveland, Hungary, Switzerland and VA Long Beach. Age median is 54 years, range 28-77.

Source CSV SHA-256: `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`.

### Provenance check (21 October 2026)

The CSV is byte-identical to its first repository appearance, commit `f739c2bba2dcdb6703891c4de1bc1e360cd1d975` (7 October 2026). Git contains no acquisition or merge script. A read-only comparison against the [official UCI archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip), retrieved on 21 October 2026, mapped the 14 measurement/target fields back to numeric UCI codes and compared row multisets, retaining duplicates. All 920 measurement/target tuples match the combined four processed files. **One hospital assignment differs: local ID 304 is labelled Cleveland but matches the first row of `processed.hungarian.data`.** It is in D, not T. This discrepancy limits the fidelity of the saved hospital grouping. The dataset, partitions, models and results have been preserved; correction and reevaluation require a separate documented decision, not a silent relabel or a fresh-test claim.

| Processed UCI file | Official rows | Local hospital rows | Same-site matching tuples |
| --- | ---: | ---: | ---: |
| `processed.cleveland.data` | 303 | 304 | 303 |
| `processed.hungarian.data` | 294 | 293 | 293 |
| `processed.switzerland.data` | 123 | 123 | 123 |
| `processed.va.data` | 200 | 200 | 200 |

The comparison excludes local `id`/`dataset`, treats blank/`?` as missing and normalizes numeric formatting. Sex is mapped Female/Male to 0/1; cp typical/atypical/non-anginal/asymptomatic to 1/2/3/4; fbs/exang FALSE/TRUE to 0/1; restecg normal/ST-T/LV-hypertrophy to 0/1/2; slope up/flat/down to 1/2/3; thal normal/fixed/reversable defect to 3/6/7. `thalch` corresponds to UCI `thalach`. These are observed correspondences, not proof of who produced the merge. Archive SHA-256: `b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc`.

UCI credits Andras Janosi, William Steinbrunn, Matthias Pfisterer and Robert Detrano, cites the dataset as 1989, DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X), and currently lists CC BY 4.0. That supports attribution to the underlying UCI data; the exact supplier/download date, merge/category-renaming process, redistribution/license chain and explanation for ID 304 still need team evidence. The generated IDs do not establish patient linkage. Do not substitute the official archive for the frozen CSV during this study.


Eleven predictors are eligible. `id`, hospital (`dataset`), and target `num` cannot enter the feature matrix; `ca` and `thal` are excluded for severe site-dependent missingness (609/918 and 484/918). Zero pressure/cholesterol readings become missing. Missing counts after cleaning include cholesterol 201, slope 307, fbs 90 and pressure 60. Conventional models use training-fitted median imputation with numeric indicators/scaling, and mode imputation with one-hot encoding. BN preprocessing differs as specified below but uses the same rows, predictors and evaluation partitions.

## Validation and selection

The historical stratified split (seed 42) gives D=734 development records and T=184 retrospective evaluation records. T was already reported and inspected. D is split (seed 43) into S=550 selection records and C=184 cutoff records. Five outer folds (seed 44) evaluate each family, with three inner folds (seed 45) selecting hyperparameters inside each outer training fold. A final three-fold search (seed 46) fits each family on S only. Every learned transformation stays inside its permitted training partition. C selects a separate cutoff for each frozen S-fitted model by F1 on the 101-point grid 0.00 to 1.00. Threshold ties choose the point nearest 0.50, then the smaller cutoff. Predictions use probability >= cutoff. There is no S+C refit. Neither C nor T can replace the selected family.

Random Forest was selected by the highest unweighted mean outer-fold F1 at cutoff 0.50: 0.8358, versus XGBoost 0.8308. ROC-AUC breaks exact F1 ties, followed by fixed order BN, LR, RF, HGB, XGB. Full-precision scores determine ties, not rounded display values. The selected outer-CV score still has family-selection optimism. Inner-search ties use mean F1, then mean AUC, then lexicographically smallest canonical JSON parameter object. Seeds, exact partitions and folds are recorded in `config/phase1_protocol.json` and the run's `partitions.json` files.

## Five-model comparison

**Retrospective evaluation on T, 184 previously reported and inspected records (102 positive, 82 negative). This is not a newly untouched or external test.** CV F1 below is the unweighted mean of five outer-fold F1 scores at 0.50. Remaining metrics use each model's C-selected cutoff. Displayed metrics round to four decimal places, with exact halfway values rounded up; saved metrics retain full precision.

| Model | Nested CV F1 | Cutoff | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8231 | 0.51 | 0.8315 | 0.8257 | 0.8824 | 0.8531 | 0.9125 | 0.1227 |
| Random forest | 0.8358 | 0.38 | 0.8043 | 0.7538 | 0.9608 | 0.8448 | 0.9029 | 0.1308 |
| Histogram gradient boosting | 0.8104 | 0.37 | 0.8152 | 0.7698 | 0.9510 | 0.8509 | 0.8945 | 0.1318 |
| XGBoost | 0.8308 | 0.34 | 0.8043 | 0.7463 | 0.9804 | 0.8475 | 0.9039 | 0.1273 |
| Bayesian network | 0.8179 | 0.54 | 0.7880 | 0.8058 | 0.8137 | 0.8098 | 0.8765 | 0.1544 |

Random Forest's confusion matrix is **TN 50, FP 32, FN 4, TP 98**. Its precision is 98/130, recall 98/102 and F1 196/232. Specificity is 50/82=0.6098 and balanced accuracy is 0.7853. Cutoff 0.38 is a development-selected statistical operating point, not a medically established threshold. Logistic Regression has a slightly higher retrospective F1/AUC, which does not change the predefined selection. No result establishes universal superiority. Brier is mean squared probability error, not proof of calibration.

## Separate D-only hospital checks

Each check uses **D only**. Exclude hospital H, split the other D records into S_H/C_H, and repeat the complete selection, fitting and cutoff procedure locally. No pooled winner, fitted preprocessing, BN bins/CPTs, model or cutoff transfers into these runs. The held-out hospital and T influence no local decisions. Within the permitted source hospitals, CV is stratified over rows, not additionally hospital-disjoint.

| Held-out D hospital | S_H | C_H | Evaluation N | Local selected family | Cutoff | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cleveland | 363 | 122 | 249 | Logistic regression | 0.50 | 0.7547 | 0.8656 |
| Hungary | 379 | 127 | 228 | Histogram gradient boosting | 0.28 | 0.6494 | 0.8416 |
| Switzerland | 478 | 160 | 96 | Random forest | 0.44 | 0.8712 | 0.7472 |
| VA Long Beach | 429 | 144 | 161 | Logistic regression | 0.30 | 0.8561 | 0.7079 |

These rows report each locally selected family, not each hospital's highest scoring model after evaluation. All five families per hospital are saved in the run. Switzerland has only 8 negative evaluation rows. These D-only counts differ from historical full-dataset site experiments and are not prospective external validation.

**Weak transport results matter.** Switzerland's local RF has F1 0.8712 and accuracy 0.7813, below always-positive F1 0.9565 and accuracy 0.9167 on the same 96 rows (88 positive, 8 negative). VA Long Beach's local LR has F1 0.8561 versus reference 0.8582; both have accuracy 0.7516. It rejects only 2/40 negative records (specificity 0.0500; balanced accuracy 0.5167), despite AUC 0.7079. These are concerns about negative-class decisions and transfer, not proof of zero ranking discrimination. No model or threshold is changed in response.

The following complementary metrics are recalculated from the saved selected-model predictions, at each existing C_H-selected cutoff. Balanced accuracy is (sensitivity + specificity)/2 and is added only as a descriptive summary after evaluation. Confusion matrices use [[TN, FP], [FN, TP]]. All five families have identical evaluation IDs/labels within each experiment; the always-positive decisions are constructed on those exact same rows. Reference confusion matrices are [[0, N-negative], [0, N-positive]]. Their sensitivity is 1, specificity 0, balanced accuracy 0.5 and constant-probability AUC 0.5. Saved reference probabilities use S_H prevalence; always-positive decisions are explicitly fixed and are not obtained by thresholding that constant probability at 0.5.

| Hospital | Positive / negative | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | 113 / 136 | 117 | 19 | 33 | 80 |
| Hungary | 84 / 144 | 72 | 72 | 9 | 75 |
| Switzerland | 88 / 8 | 4 | 4 | 17 | 71 |
| VA Long Beach | 121 / 40 | 2 | 38 | 2 | 119 |

| Hospital | Precision | Sensitivity | Specificity | Balanced accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cleveland | 0.8081 | 0.7080 | 0.8603 | 0.7841 | 0.7912 |
| Hungary | 0.5102 | 0.8929 | 0.5000 | 0.6964 | 0.6447 |
| Switzerland | 0.9467 | 0.8068 | 0.5000 | 0.6534 | 0.7813 |
| VA Long Beach | 0.7580 | 0.9835 | 0.0500 | 0.5167 | 0.7516 |

| Hospital | Selected F1 | Always-positive F1 | Selected accuracy | Always-positive accuracy |
| --- | ---: | ---: | ---: | ---: |
| Cleveland | 0.7547 | 0.6243 | 0.7912 | 0.4538 |
| Hungary | 0.6494 | 0.5385 | 0.6447 | 0.3684 |
| Switzerland | 0.8712 | 0.9565 | 0.7813 | 0.9167 |
| VA Long Beach | 0.8561 | 0.8582 | 0.7516 | 0.7516 |

For positive prevalence p, always-positive precision is p and F1 is 2p/(1+p), so high positive-class F1 can reward a classifier that never recognizes a negative. AUC summarizes ranking over thresholds; it cannot ensure useful specificity at the chosen cutoff. Small negative counts make specificity unstable. These are selected historical cohorts with differing prevalence, missingness and potentially measurements/labels; patient linkage is unavailable. ID 304's hospital discrepancy adds a concrete grouping limitation. These secondary D-only experiments overlap pooled development and are not independent prospective external validation.

## Provisional Bayesian Network

The provisional discrete DAG contains disease -> each of the 11 features, plus oldpeak -> slope. These directions specify a class-conditional statistical factorization, not medical causation. All feature nodes except the oldpeak/slope pair are mutually independent given disease; slope is independent of the other features given disease and oldpeak. The extra edge permits residual dependence between two recorded exercise-ST descriptors, but its orientation and the omitted dependencies are unvalidated assumptions. No edge has established medical or faculty approval.

Training uses numeric medians and categorical modes from the permitted training rows. Numeric features use training-only one-third/two-thirds quantiles, with duplicate/extreme cut points removed and unbounded exterior bins. Categories use a fixed vocabulary. CPT estimates use (count + alpha)/(parent-count + K*alpha), with alpha=1 pseudocount per child state in this exploratory run. At inference, exact sum-product marginalization computes P(disease | observed evidence), explicitly summing the dependent oldpeak/slope branch. Missing observations are omitted from evidence; no observations returns the fitted disease prior. Valid unseen states have smoothed probabilities; invalid categories raise an error. Imputed training counts are an approximation and do not eliminate missingness bias.

The factorization is `P(disease) * product(P(feature | disease))` over features other than slope, multiplied by `P(slope | disease, oldpeak)`. The maximum parent count is two. `bn_graph.json` and `pooled/bn_state.json` preserve the actual graph, bins, states and CPTs; the joblib artifact performs inference without fitting. The evidence register in [docs/bn_design.md](docs/bn_design.md) covers all 12 included edges and all 54 omitted feature-pair connections. The implemented graph is a manual class-conditional candidate with one extra dependency, not a learned TAN. Domain structure is only partly established; no topology or saved BN results were changed by this review.

## Run locally and test

Use Python 3.13 with the pinned existing dependencies in an isolated environment:

```bash
python3.13 -m venv /tmp/heartlens-app
/tmp/heartlens-app/bin/python -m pip install -r requirements.txt
/tmp/heartlens-app/bin/python -m streamlit run app.py
/tmp/heartlens-app/bin/python -m unittest discover -s tests -v
```

The app reads `artifacts/runs/20261020T193544Z_2a8a1491/` and the source CSV. Keep the full run, `bayesian_network.py` and `modeling.py` when copying the project. Stage 5 verification passed **38 tests (33 existing + 5 app tests)**, including historical split reproduction, isolation, BN probability/inference, artifact loading, missing inputs, saved cutoffs and page/result consistency. The final remediation reran all 38 tests successfully in an external disposable environment with the pinned versions; all 25 saved models reproduced their evaluation probabilities without fitting.

Training is deliberately guarded. `python train.py` refuses an experiment by default. For a future separately authorized exploratory reproduction, the implemented command is `python train.py --run-experiment --allow-provisional-bn`; it creates a new unique run. Do not run it to launch the app or regenerate documents. `python eda.py` overwrites the legacy EDA output and is also unnecessary for viewing verified results. On the recorded Mac run, XGBoost needed a process-local `DYLD_LIBRARY_PATH` pointing to scikit-learn's bundled `.dylibs` directory for OpenMP. No system library was installed.

## Reproducibility and limitations

The run completed all five families across pooled plus four hospital experiments, with **25 frozen final models and 2940 fits**, in **340.927 seconds** excluding setup/tests. Environment: Python 3.13.6, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, Joblib 1.6.0 and XGBoost 3.4.1. The app requirements additionally pin Streamlit 1.65.0 and Plotly 7.1.0. Code snapshots, data/code hashes, environment, complete searches, folds, OOF/C/T probabilities, thresholds, graph/CPTs and verification are in the run. The initial XGBoost OpenMP issue and physical-core fallback warning are recorded; all final fits succeeded.

Saved uncertainty uses 2000 paired row-bootstrap resamples (seed 47) of fixed predictions, with percentile 95% intervals. Five fixed probability bins describe reliability. No calibration model was fitted. Intervals condition on fitted models and omit training-selection and unknown patient-cluster uncertainty; they are not superiority tests. Stage 6 rechecked all 25 default/tuned metric records, nested-CV means and cutoffs from saved predictions without fitting.

This is retrospective disease-presence classification in selected historical hospital records. It is not a clinically validated diagnosis or future-risk predictor. The supplied merged CSV is described as UCI-derived, but its exact download/transformation history, license chain and patient linkage remain unverified. The read-only provenance comparison above establishes measurement/target correspondence to four processed UCI files but identifies one inconsistent site label; it does not recover the acquisition history. Row deduplication cannot establish patient independence. The BN graph remains provisional, unapproved and statistically restrictive. Search budgets differ (LR 8, RF 6, HGB 8, XGB 8, BN 1 configurations). Neural-network implementation is Phase 2; no such experiment is claimed for Phase 1. No prospective or independent external clinical validation, clinical utility analysis, or recalibration has been completed.

## Existing deliverables and code map

- `report/phase1_report.tex` and `.pdf`: IEEE report.
- `presentation/phase1_presentation.pptx` and `.pdf`: nine-slide presentation and handout.
- `VIVA.md` and `output/pdf/phase1_viva_teaching_guide.pdf`: technical questions, worked arithmetic and team contribution blanks.
- `scripts/build_phase1_viva_pdf.py`: builds the teaching PDF from the verified run and `VIVA.md`, without training.
- `PHASE1_SHARE.md`: local sharing/submission checklist. `submission/phase1_share_bundle.zip` contains these deliverables and compact verified evidence.
- `modeling.py`: validation and shared feature handling. `bayesian_network.py`: CPT estimation/inference. `train.py`: nested selection and isolated scoring. `experiment.py`: guarded run outputs. `app.py`: saved-model inference.

The five confirmed names/enrolments are recorded above. Actual contribution evidence, acquisition/transformation history and the ID 304 discrepancy remain team actions. Faculty review is needed for graph/independence acceptance, measurement stage, retrospective/D-only evaluation and submission instructions; approval is not presumed.

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



For the selected Random Forest, blank optional inputs use fitted imputation. For the BN, missing observations are marginalized. Malformed inputs remain errors.

## Verified methodological references

- UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease (variable definitions and original databases, not certification of this exact merge).
- Detrano et al. (1989), *International application of a new probability algorithm for the diagnosis of coronary artery disease*. DOI: [10.1016/0002-9149(89)90524-9](https://pubmed.ncbi.nlm.nih.gov/2756873/). Historical transport motivation; cohorts differ from this CSV.
- Breiman (2001), *Random Forests*. DOI: [10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324).
- Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*. DOI: [10.1145/2939672.2939785](https://arxiv.org/abs/1603.02754).
- Collins et al. (2024), *TRIPOD+AI statement*. DOI: [10.1136/bmj-2023-078378](https://pubmed.ncbi.nlm.nih.gov/38626948/). Reporting guidance, not a claim of compliance or clinical validity.
- Heckerman (1995), *A Tutorial on Learning With Bayesian Networks*, MSR-TR-95-06: https://www.microsoft.com/en-us/research/publication/a-tutorial-on-learning-with-bayesian-networks/ (general BN methods, no support for our medical edges).
- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) (methodological separation of tuning and evaluation).

- Friedman, Geiger and Goldszmidt (1997), *Bayesian Network Classifiers*, Machine Learning 29, 131-163. DOI: [10.1023/A:1007465528199](https://link.springer.com/article/10.1023/A:1007465528199); [author abstract](https://ai.stanford.edu/~nir/Abstracts/FrGG1.html). Supports classifier factorization and relaxing naive-Bayes independence; does not validate our medical edges or make our manual one-edge graph a learned TAN.
- Esmaeili, Roshanravan, Ghaffari, Mesri Alamdari and Asghari-Jafarabadi (2024), *Unraveling atherosclerotic cardiovascular disease risk factors through conditional probability analysis with Bayesian networks: insights from the AZAR cohort study*, Scientific Reports 14, 4361. DOI: [10.1038/s41598-024-55141-2](https://www.nature.com/articles/s41598-024-55141-2.pdf). Cross-sectional Iranian healthcare-provider study; different ASCVD endpoint, predictors and leave-one-out validation. Neither its graph nor its performance transfers to our angiographic label.
- Van Calster, McLernon, van Smeden, Wynants and Steyerberg (2019), *Calibration: the Achilles heel of predictive analytics*, BMC Medicine 17, 230. DOI: [10.1186/s12916-019-1466-7](https://link.springer.com/article/10.1186/s12916-019-1466-7). Distinguishes discrimination from reliable probabilities; our five descriptive bins do not establish clinical calibration.
- scikit-learn developers, [Ensembles: histogram-based gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), documentation checked 21 October 2026. Explains the implemented HGB family; our shared imputation pipeline is retained even though HGB supports missing values natively.

## GitHub publication check (21 October 2026)

Read-only `git ls-remote --heads origin main` returned `1f953e5e4281cb355e7f205062ca666e2cd36779`, also local HEAD on `main`. The working tree contains modified tracked files and untracked BN/protocol/test/run files. Therefore the current verified implementation and these remediations are **not published in that remote main commit**. No commit, push or deployment was performed. The hosted app's current contents were not verified.

Before publication: review the complete diff and untracked files; fill truthful contributions and disclose the data/graph limits; verify PDFs and ZIP; include the inference run files required by the app; check repository upload limits and data attribution; run the pinned tests; obtain team review; then commit/push only after explicit authorization. Verify the remote commit and hosted app after any later deployment. No CI or release framework is required.
