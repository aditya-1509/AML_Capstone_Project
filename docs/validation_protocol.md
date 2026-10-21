# Phase 1 validation protocol - implemented exploratory run; preserved Stage 0 rules

Project 15; team T008; Section C. Date: 2026-10-21.

## Current implementation and execution status

This explanatory document accompanies the preserved Stage 0 JSON specification. The authorized exploratory run `artifacts/runs/20261020T193544Z_2a8a1491/` completed pooled plus four D-only hospital experiments, all five families, 25 saved final models and 2940 fits. Its `effective_protocol.json` records the explicit provisional-BN alpha=1 override, fixed subgroup definitions and execution settings. The root JSON remains the original draft; it is not silently rewritten as faculty approval.

`train.py` implements selection/cutoff/scoring, `experiment.py` writes unique run artifacts under explicit opt-in, and the app reads frozen results. The default training guard remains. Final remediation passed all 38 lightweight tests and reloaded all 25 models to reproduce their saved predictions, metrics, cutoffs and partition isolation without fitting. Synthetic tests trace fit boundaries; saved traces support the executed protocol, not independent clinical validation. No source or scientific selection rule was changed by final remediation.

The Stage 1 pandas-3 missing-boolean defect and historical ordered-index split check were resolved in Stage 3. Earlier 22-pass/two-error and 31-test counts are historical milestones, not current failures. BN graph/domain approval remains unresolved despite successful execution. Neural-network implementation belongs to Phase 2.

The provenance investigation now matches all 920 measurement/target tuples to four official UCI processed files, but local ID 304 is labelled Cleveland while its official match is Hungarian. It lies in D. The frozen dataset/site grouping and reported run are unchanged; hospital conclusions are qualified accordingly. See README for the mapping and source/archive identity. This is a data-quality finding, not permission to silently alter partitions or reuse T as untouched validation.

The remaining sections retain the Stage 0 partition rules and their rationale. Statements about proposed gates refer to that original design; current execution facts are recorded above and in the run's effective protocol.

## Historical test exposure

The existing 184-row holdout has already been reported and inspected. All future results on it must be labelled **retrospective evaluation on the previously reported and inspected historical holdout**, not newly untouched validation. Neither new code nor a new random split can undo this exposure. Obtaining a genuinely fresh test requires independent data and is outside the current scope.

The source CSV stays unchanged. Preserve the existing source-order deduplication, excluding only `id` from duplicate comparison. Validate the target before converting `num > 0`. Exact deduplication is not proof of patient independence; patient linkage is unavailable.

## Partition boundaries

1. Reconstruct the existing shuffled, binary-target-stratified 80/20 split with seed 42. Name the 734 development rows `D` and 184 historical holdout rows `T`.
2. Split `D`, preserving its reconstructed split order as the splitter input, into 75% selection rows `S` and 25% threshold rows `C`, stratified with seed 43. Expected sizes are 550 and 184.
3. Persist exact source IDs, row order, removed-duplicate lineage and all fold membership before fitting. These counts were expected assertions in Stage 0 and are now verified against saved partition manifests.

| Partition | Allowed operations | Forbidden operations |
| --- | --- | --- |
| S | Nested CV, family selection, hyperparameter search, fitted preprocessing and estimator fitting | Importing fitted state from C or T |
| C | Predict using S-fitted models; choose each family's cutoff | Fit model/preprocessing/bins/CPTs/calibrator; choose deployment family |
| T | Score frozen candidates after the complete freeze | Any development decision, including hospital development experiments |

All families use the same eleven raw predictors and eligible rows. Retain `dataset` only for partitioning/reporting, never as a predictor. No learned imputation, encoding, scaling or discretization precedes its permitted training partition.

## Nested OOF predictions and family selection

On S, create five shuffled stratified outer folds with seed 44. For each family independently, and inside each outer-training partition:

1. Run three-fold shuffled stratified inner CV with seed 45 over its declared grid.
2. Rank configurations by highest unweighted mean inner-fold F1 at probability >= 0.50, then highest mean ROC-AUC, then lexicographically smallest canonical parameter JSON.
3. Refit the winner on that outer-training partition only and predict its outer-validation rows.

The scoring function must explicitly use class-1 probabilities and `>= 0.50`; relying on estimator-specific `predict()` tie behavior is insufficient. Canonical parameter JSON means `json.dumps(params, sort_keys=True, separators=(',', ':'), allow_nan=False)` with JSON-native values.

Every S row receives one nested OOF prediction per family. Its label was excluded from preprocessing, parameter selection and fitting for that prediction. Do not tune on all S first and then call `cross_val_predict` with the resulting winner.

Choose the deployment family using highest **unweighted mean outer-fold F1 at 0.50**, then highest unweighted mean outer-fold ROC-AUC. Exact score ties use this fixed order: Bayesian network, logistic regression, random forest, histogram gradient boosting, XGBoost. Compare full-precision finite values without display rounding or tolerance groups. This order is reproducibility machinery, not a preference supported by expected performance.

Save fold scores and pooled OOF scores, but do not substitute pooled F1 for the specified mean-fold selection statistic. Fold SD is descriptive, not a confidence interval. Selecting a family from outer-CV scores introduces selection optimism in the winning score; do not present that selected score as an independent estimate of the family-selection procedure.

## Final fits and cutoff development

After selecting the family on S, run a final three-fold shuffled stratified search on all S with seed 46 for every family. Use the same configuration-ranking rule and refit each winner on S. All preprocessing remains inside its search pipeline; the BN's learned bins, imputation and CPTs obey the same boundary.

Freeze each fitted model before predicting C. For every family independently, evaluate the 101 thresholds `k / 100`, k = 0,...,100, using class-1 probability >= threshold and F1 with `zero_division=0`. Select highest full-precision F1; break ties by smallest integer distance from k=50, then smallest k. Save all scores, tied maxima and selected index.

The deployment family cannot change based on C results. Maximized C F1 is a development objective, not a generalization estimate. Do not refit on S+C: that would change the probability function associated with the selected cutoff. If later approved, a full-D deployment refit must be a distinct model and cannot inherit performance claims from this protocol without further justification.

The initial protocol assesses raw probabilities and fits no recalibrator. Adding calibration is an amendment requiring another leakage-safe training boundary and tests; it is not an automatic response to evaluation plots.

## Baselines and unresolved BN settings

The JSON preserves the four existing baseline grids. Seeds for stochastic baseline estimators are 42. Set estimator threads to one, CV workers to one by default, and disable histogram boosting's internal early stopping explicitly. This avoids hidden validation behavior and nested parallelism. Record fixed settings and installed versions; the old environment's API must not be assumed.

The BN graph remains unapproved. The original JSON grid is null; the authorized exploratory run explicitly records alpha=1 and provisional-graph opt-in in its effective protocol. That override permits an exploratory comparison, not medical or faculty approval. State/bin and missing-evidence behavior are documented in docs/bn_design.md. Do not expand grids after seeing retrospective or hospital results. Different declared grid sizes are acknowledged; equal raw inputs do not imply identical representations or search budgets.

## Hospital-wise isolation

Use D only. For each hospital H, set aside all its development rows as D_H. Split the remaining D rows into S_H/C_H at 75/25, stratified. Repeat the complete family comparison, final fitting and threshold procedure locally, using five outer and three inner/final folds.

| Held-out hospital | S_H/C_H split | Outer CV | Inner CV | Final search |
| --- | --- | --- | --- | --- |
| Cleveland | 143 | 144 | 145 | 146 |
| Hungary | 243 | 244 | 245 | 246 |
| Switzerland | 343 | 344 | 345 | 346 |
| VA Long Beach | 443 | 444 | 445 | 446 |

Stochastic estimator seed remains 42. Share only the frozen protocol, reviewed externally justified graph, fixed category vocabulary/grids and deterministic cleaning rules. Relearn imputation, encoding, scaling, numerical bins, parameters and CPTs within each permitted fit. Do not import a pooled family, fitted model, learned state or cutoff.

Score all frozen families on D_H at both 0.50 and their C_H-selected cutoffs. Identify the locally selected family from S_H. D_H labels cannot affect family/parameter/threshold selection. Inner and outer selection CV mix rows from permitted source hospitals: they are not additional hospital-disjoint folds.

These are secondary D-only assessments, not the old full-dataset hospital checks. Report actual counts and prevalence; do not reuse the old hospital denominators. Their labels overlap pooled development data, so they do not independently confirm the pooled procedure. Their results cannot change any pooled choices, graph or grids. Freeze the protocol before these results are viewed.

Require sufficient class support for every fixed CV split and both classes in scoring partitions. If that fails, record the experiment as not estimable and stop it. Do not silently reduce folds, resplit, drop a family or substitute a metric. Unknown repeat patients and informative site missingness remain limitations.

## Retrospective evaluation and reporting

Before scoring T, freeze the exact protocol/graph snapshots, code/data identity, partitions, grids, learned transformations, all fitted candidates, selected family, cutoffs, subgroup definitions and artifact schema. Score T once per frozen experiment using both cutoffs. Test rankings cannot choose the deployed model or revise the study.

Report accuracy, precision, recall, specificity, F1, ROC-AUC, average precision, Brier score, log loss and confusion matrices with rows actual and columns predicted: [[TN, FP], [FN, TP]]. Save original finite probabilities in [0,1]. Use float64 machine-epsilon clipping only to calculate binary log loss; never overwrite saved probabilities with clipped values.

Use five fixed equal-width reliability bins, reporting counts, mean predicted probabilities, observed positive fractions and empty bins. Brier score and reliability plots do not certify clinical calibration. Sex/age subgroup results remain descriptive; exact age strata must be frozen before any experiments. The exploratory effective protocol froze its subgroup definitions before scoring; the original JSON retains the Stage 0 unresolved field as history.

For evaluation uncertainty, use 2000 ordinary row-bootstrap resamples of saved predictions, seed 47, paired across families within each evaluation population. Use percentile 95% intervals; do not refit or reselect thresholds. Count undefined metric resamples; for ROC-AUC, exclude single-class resamples and report null intervals if fewer than 1000 valid resamples remain. These intervals condition on the fitted models/split and do not account for training selection or unknown patient clustering. Do not use them as a formal superiority test.

Include an always-positive decision reference and a constant probability estimated only from S (S_H for a hospital run). No new T permutation-importance calculations are part of this protocol. Historical results remain historical, not replacement measurements or target performance levels.

## Reproducibility and execution gates

Future runs write only to distinct `artifacts/runs/<run_id>/` directories, preserving legacy artifacts. Save configuration/graph snapshots and hashes, source and code identity, environment/dependency versions, partitions/folds, all searches/failures, predictions, thresholds, reconstructable metrics, fitted models and complete BN state. Pending/failed runs cannot be displayed as verified results.

Before a run: resolve material open questions; validate graph/specification; verify isolation tests; inspect the exact fit-count/resource dry run; obtain authorization for dependencies and expensive work. These were future requirements in Stage 0; explicit opt-in and unique run directories now implement execution/artifact guards. Faculty approval is still not inferred. No historical holdout outcomes may decide whether to proceed with a model change.

## Resolved inconsistencies from the earlier plan

- **Leakage fix:** separate S/C, nested configuration selection within S, no globally selected configuration reused to produce purportedly selection-independent OOF predictions.
- **Selection objective:** retain F1 at 0.50 for family/configuration ranking; tuned-cutoff performance is evaluated separately and cannot rerank families. This is not a claim to select the best tuned-cutoff family.
- **Final model size:** models fit S, not all 734 D rows. C is reserved; there is no silent full-D refit.
- **Hospital population:** D-only holdouts preserve T's exclusion; old full-site counts and metrics are not comparable measurements.
- **Cutoff range:** the proposed full [0,1] grid replaces the old restricted range; the change is explicit and occurs before new evaluation.
- **Early stopping:** disable histogram boosting's internal early stopping; baseline settings are explicit and no extra hidden partition is introduced.
- **Calibration:** assessment only, no implicitly added recalibration stage.
- **BN graph (historical Stage 0):** no audit illustration was adopted then; its empty topology/null grid initially blocked execution. The later explicit provisional graph/alpha=1 override is recorded in the exploratory run; it remains unapproved.
- **Test terminology:** historical holdout use is retrospective, never newly untouched validation.

## Open decisions and resolved implementation questions

The root JSON's F01-F06, D01, B01 and E01 are historical questions, not a claim that nothing has subsequently been implemented. Current user-supplied official scope assigns neural-network training to Phase 2. The custom BN needs no new graph library; the recorded run completed under its explicit resource budget. State/bins, alpha semantics and subgroups are recorded in the effective protocol. Historical split equivalence passes.

Actual remaining decisions: faculty acceptance of the provisional graph/independences, eligible measurement stage, F1 and retrospective/D-only evaluation as evidence, and submission details. Team evidence is still needed for acquisition/merge/license history, ID 304's site discrepancy, patient linkage if available, and individual contributions. Do not use the inspected hospital or T results to revise scientific choices.

## Method references

These are methodological sources, not evidence for any BN medical edge:

- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- scikit-learn, [Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html).
- scikit-learn, [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

## Historical stage boundary

Stage 0 originally created four specification files only. Stages 1-5 were subsequently authorized separately and completed validation, provisional BN, leakage-safe selection, one exploratory run and app integration. Stage 6/final remediation update academic materials and verify saved evidence; they do not authorize a new training run, graph change, dataset correction, commit or push.
