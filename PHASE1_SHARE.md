# Phase 1 sharing and submission checklist

**Project 15: Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network**
**Team T008, Section C.** Current work compares four conventional classifiers with a provisional statistical BN. Neural-network training is assigned to Phase 2, not a missing Phase 1 ML experiment. Graph/domain acceptance still needs faculty review.

## Share the current local bundle

Send `submission/phase1_share_bundle.zip` directly after checking its contents. It includes the report source/PDF, editable nine-slide deck/PDF, README, viva material, product notes, the teaching-PDF builder and compact verified results. It excludes trained models, the dataset and runtime dependencies; use the full repository to run the app.

Existing repository: https://github.com/aditya-1509/mk_sem5_project
Existing hosted demo: https://mk-sem5-project.streamlit.app/

These links may still show older versions. The verified implementation and final remediation are local and have not been committed, pushed, uploaded or deployed. Do not tell the team that the remote ZIP or hosted app already contains these updates.

## Message to accompany the local ZIP

> Project 15, Team T008, Section C: please review the attached updated Phase 1 materials. We compared LR, RF, HGB, XGBoost and a provisional BN. Random Forest remains selected by nested-CV F1, with cutoff 0.38. Its retrospective 184-row scores are F1 0.8448, recall 0.9608 and AUC 0.9029 (TN 50, FP 32, FN 4, TP 98). This holdout was already inspected. The BN's graph is statistical, unapproved and not medically validated. No neural-network comparison or clinical validation is complete. The confirmed five-member roster is included. Fill truthful contribution evidence and resolve the data/graph questions and submission instructions before submission.

## Honest readiness against all nine criteria

| Criterion | Status | Strongest evidence / remaining weakness |
| --- | --- | --- |
| Literature Review | Partially satisfied | Verified critical comparison and complete BN edge/omission register; exact medical graph remains unsupported |
| Dataset Quality & EDA | Partially satisfied | Reproducible counts/missingness, official-file correspondence; ID 304 site mismatch, missing acquisition history/patient linkage |
| Feature Engineering | Satisfied with evidence | Eleven-feature contract and fold-local learned preprocessing; validation/isolation tests pass |
| Theoretical Rigor | Partially satisfied | Actual factorization, CPT smoothing and marginalization; domain independences remain unvalidated |
| Model Application | Satisfied with evidence | Five completed families and 25 artifact round trips; weak hospital specificity/unequal grids constrain conclusions |
| GitHub Repository & Code Quality | Partially satisfied | 38 passing tests and frozen traces; working implementation remains uncommitted/unpublished |
| Project Presentation | Satisfied with evidence | Updated editable nine-slide deck and rendered PDF; live delivery still to be demonstrated |
| Individual Viva Voce | Cannot verify | Implementation-based guide and confirmed roster; individual understanding and contribution evidence not demonstrated |
| Project Report in LaTeX/IEEE format | Satisfied with evidence | Updated source and compiled/rendered PDF; final faculty template/approval remains external |

This is a local readiness assessment, not an official grade or a claim of clinical validation.

## Facts to defend

- 920 source rows become 918 after duplicate removal; 508 positive / 410 negative; target `num > 0`.
- D=734, historical T=184; D gives S=550 and C=184. T is retrospective, never newly untouched.
- Five outer / three inner stratified folds select on S, with learned preprocessing inside fitting folds. C selects cutoffs on frozen S-fitted models; no S+C refit.
- RF nested-CV F1=0.8358, XGB=0.8308. A higher retrospective LR score does not replace RF.
- BN graph: disease to all 11 features plus oldpeak to slope. Smoothed discrete CPTs (alpha=1) and exact inference support missing evidence. Edges/independences are provisional statistical assumptions.
- Hospital checks use D only, with independent local selection and cutoffs; no pooled learned state or winner is imported.
- All 25 final model evaluations completed, with 2940 fits in 340.927 seconds. Search budgets differ and BN used one alpha setting.
- Saved paired row-bootstrap intervals condition on fitted models. No prospective external validation, clinical utility or neural-network comparison is complete.

## Team and faculty actions still required

1. Check the five confirmed names/enrolments in the report, slides and viva table; fill accurate contributions with evidence. A roster or prepared guide does not establish who did each part.
2. Supply exact merged-CSV provenance, transformation/license history and any patient-linkage evidence.
3. Obtain faculty decisions about BN domain justification/independences, eligible measurement stage, and acceptance of retrospective and D-only hospital evaluation.
4. Confirm final report/slide template, authorship rules, upload channel, deadline and any required form. No submission or faculty approval is implied.
5. Practise the local app using fictional data and the saved run, without rerunning EDA or training.

## Findings that must stay visible

Switzerland: selected RF F1 0.8712 versus same-row always-positive 0.9565; accuracy 0.7813 versus 0.9167. VA Long Beach: LR F1 0.8561 versus 0.8582; specificity 0.05 (2/40 negatives), balanced accuracy 0.5167. Pooled RF specificity is 0.6098 at its development cutoff. These results constrain transfer claims and do not select a replacement model. The official-file comparison also identifies ID 304's local Cleveland/official Hungary discrepancy; no dataset or old result was changed.

## Publication checklist

Read-only remote main and local HEAD both resolve to `1f953e5e4281cb355e7f205062ca666e2cd36779` on 21 October 2026. Modified/untracked working files, including BN, protocol, tests, run and updated documents, are not in that remote main commit. No commit or push was performed.

Before a separately authorized publication: review tracked and untracked changes; include required inference artifacts and attribution; complete truthful contributions and disclose unresolved data/graph questions; verify tests, PDFs and ZIP; obtain team review; commit/push only after approval; then verify remote files and hosted demo. Do not assume the older public app already contains the current implementation.

## Final remediation verification record (21 October 2026)

- External disposable Python 3.13.6 environment, existing requirement pins: NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, Joblib 1.6.0, XGBoost 3.4.1, Streamlit 1.65.0, Plotly 7.1.0. Normal Python and dependency files were not changed.
- `python -B -m unittest discover -s tests -v`: 38 passed, no errors/skips; final run 9.630 seconds. XGBoost used only a process-local OpenMP library path. Tests fit synthetic fixtures; no real-data training occurred.
- Reloaded all 25 frozen models and reproduced evaluation probabilities; reconstructed default/tuned metrics, CV means, threshold choices and partition boundaries. The run's verification helper was copied to a temporary directory with its output redirected there, because its original form writes `verification.json`. The immutable run was not rewritten.
- Recomputed all five same-row always-positive references, all hospital class supports/confusion matrices, and descriptive balanced accuracy. Verified numerical tables/chart caches against saved predictions. Display rounding does not alter full-precision metrics.
- Report compiled in the existing editor and exported as five pages; slides exported as nine PDF pages; teaching guide regenerated as eight pages. Rendered every report/guide page and slide. Report exporter emitted font-substitution/underfull warnings, with no overfull boxes or unresolved citations; rendered text/math remain readable. Slides were checked by import/render and LibreOffice PDF export, not native PowerPoint execution.
- The submission bundle includes current materials, the two design/protocol explanations, original JSON specifications and compact frozen evidence; it excludes models, dataset and temporary environments. CRC and byte-for-byte content checks are required after every rebuild.
