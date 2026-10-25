# Phase 1 ML validation protocol

The continuation preserves the baseline's source data, cleaning rules and partitions.
It adapts the team's local HeartLens ML implementation; the source project's
historical holdout has already been inspected and is evaluated retrospectively.

## Partition rules

Validate and deduplicate source records before splitting. Exclude `id`, `dataset`,
`num`, `ca` and `thal` from the eleven predictors. Convert zero pressure/cholesterol
to missing. No fitted transformation may see its validation/evaluation rows.

- Historical stratified 80/20 split, seed 42: D=734, T=184.
- Stratified D split, seed 43: S=550, C=184.
- Family selection on S: five outer folds (seed 44), three inner folds (seed 45).
- Final search on S: three folds (seed 46), followed by fitting on S only.
- C selects each frozen model's cutoff. No refitting on S+C.
- T scores frozen models only. T cannot change families, parameters or cutoffs.

## Model comparison

Compare LR, RF, HGB, XGBoost and the provisional discrete BN on common eligible
rows/folds. Each inner search ranks configurations by mean F1 at cutoff 0.50,
then mean ROC-AUC, then canonical parameter JSON. Select the family by mean
outer-fold F1, then AUC, then the fixed family order recorded in the protocol.
All comparisons use full precision. Family selection makes the winning CV score
optimistic; it is not an independent performance estimate of the selection.

Cutoffs maximize C's F1 over the 101-point grid 0.00 through 1.00. Ties choose the
point nearest 0.50, then the smaller point. Predictions use probability >= cutoff.
The selected family remains fixed regardless of C or T ranking.

BN alpha=1 is a single-setting exploratory choice, not a tuned optimum. Binning,
imputation and CPT fitting use training rows only. The graph and its omitted
independences remain provisional: see `bn_design.md` and `config/bn_graph.json`.
Search budgets differ across families and do not establish universal superiority.

## Hospital checks

Each hospital check uses D only. Exclude the hospital, split the remaining D rows
into S_H/C_H, repeat nested family selection/fitting/cutoff selection, then score
D_H. Use the fixed site-specific seeds in the protocol. Neither pooled learned
state nor T rows may enter these experiments. Within-source CV is row-stratified,
not hospital-disjoint. Compare each local winner with a same-row always-positive
reference and report specificity as well as F1.

## Evidence and limitations

Persist exact partitions/inner and outer folds before fitting, full search traces,
OOF predictions, cutoff curves/predictions, fitted model hashes, retrospective
predictions, metrics, environment and source/code hashes. `verify_run.py` checks
model round trips, metric reconstruction and isolation on a completed run.

Metrics include accuracy, precision, recall, specificity, F1, ROC-AUC, average
precision, Brier, log loss and confusion matrices. Reliability bins and sex/age
subgroups are descriptive. The 2,000 paired row-bootstrap intervals condition on
the fitted models and do not capture training or selection uncertainty. Row IDs
are not verified patient identifiers. No result proves clinical utility, causal
relationships or generalization to a prospective population. Neural-network
training is Phase 2.
