# HeartLens - Project 15, Team T008, Section C

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing Streamlit app with NumPy, pandas, scikit-learn, Joblib and Plotly. No new architecture or dependencies.

## Users

Students, faculty and teammates exploring the historical disease-presence experiment and its limitations.

## Product Purpose

Assigned title: **Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network**. Implemented Phase 1 scope: four conventional classifiers and a **provisional Bayesian Network**. Neural-network implementation is assigned to Phase 2 and is not claimed as completed in Phase 1.

The app demonstrates historical UCI-style disease-presence classification. It provides neither future disease-risk prediction nor clinically validated diagnosis.

## Operating Context

The supplied CSV has 920 rows; deduplication ignoring ID leaves 918, with 508 positive and 410 negative labels. Validate `num` before defining `num > 0`. Use 11 eligible predictors; exclude identifier, hospital, target, `ca` and `thal`. The exact merged-CSV provenance/patient linkage remain unresolved.

## Capabilities and Constraints

Preserve Prediction, Data story and Model studio. Load pooled Random Forest from `artifacts/runs/20261020T193544Z_2a8a1491/`, with matching cutoff 0.38. RF was selected on S by nested-CV F1, not retrospective test ranking. Show provisional BN inference (cutoff 0.54) in an expander and saved five-model metrics. RF imputes missing inputs using fitted preprocessing; BN marginalizes missing observations. Reject malformed inputs. No fitting runs in the app.

D=734 splits into S=550 and C=184. Nested selection and all learned state use S only; C chooses cutoffs. The 184-row historical T was already reported/inspected, so scores are retrospective. Separate D-only hospital experiments select locally and import no pooled winner or fitted state. No test permutation-importance results exist for this verified run.

## Evidence on Hand

Run manifest, default/tuned metrics, per-row predictions, frozen decisions/models, BN graph/states/CPTs, folds, environment and verification. All five families completed. Stage 5 recorded 38 passing tests including app interactions and RF/BN saved-prediction consistency. These local updates have not been pushed or deployed; the existing hosted app may be older.

## Product Principles

- Keep the existing UI and code simple and use matching saved model, threshold and results.
- Label the BN provisional and unapproved. Its class-conditional graph and oldpeak/slope edge do not establish medical causation.
- Explain historical selection/exposure, site shift, imputation approximation and unequal search budgets honestly.
- Use fictional profiles. Session inputs are not intentionally persisted to disk by the app.
- Keep clinical claims, prospective performance claims and completed neural-network claims out of the demo.

## Information still needed

The confirmed five-member roster is in README and VIVA; truthful contribution evidence remains missing. The official-file comparison matches all 920 measurement/target tuples but identifies ID 304 as a Cleveland/Hungary grouping discrepancy; supplier/merge/license history remains incomplete. Faculty review is needed for BN assumptions, measurement stage and submission requirements. Neural-network training belongs to Phase 2.

## Research limitations to preserve

RF retrospective specificity is 0.6098 (32/82 negatives falsely flagged); cutoff 0.38 is statistical, not clinical. Switzerland RF loses to the same-row always-positive reference on F1 and accuracy. VA LR specificity is 0.05, balanced accuracy 0.5167. These findings and the ID 304 site discrepancy limit transport claims. The real BN has a provisional manual graph; evidence for its domain structure remains incomplete. Detailed edge register: docs/bn_design.md. No model, threshold, UI or graph was changed by final remediation.
