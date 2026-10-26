# HeartLens

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Streamlit, explicitly requested for implementation and hosting. Python inference uses the existing frozen scikit-learn, XGBoost and Bayesian Network models.

## Users

The capstone team and visitors reviewing its Phase 1 machine learning project.

## Product Purpose

Let visitors enter a fictional profile and explicitly click Predict to run the model comparisons. Make the project's data, results and limitations understandable.

## Operating Context

The app accompanies Team T008's IEEE report and presentation. It is an academic demonstration of historical disease-presence classification, not clinical diagnosis or future-event prediction.

## Capabilities and Constraints

- Live profile inputs and inference from verified saved models.
- Interactive model comparisons, threshold exploration and hospital evaluation results.
- Preserve original models, source data, selection rules and recorded metrics.
- Default Random Forest and cutoff 0.38 follow development selection. Retrospective rankings do not change that choice.
- No model training during app execution. Profile values remain in the session and are not intentionally written to disk.
- Keep source in the shared GitHub repository; deploy its synchronized fork under the user’s account because Community Cloud requires repository admin access.
- Keep README unchanged in this work.

## Brand Commitments

The user requested warm colors. Continue the cream, terracotta and warm brown identity of the presentation. Follow the supplied dashboard screenshots: left sidebar, centered banner and two-column profile inputs.

## Evidence on Hand

Verified five-family experiment: artifacts/runs/20261025T151903Z_ea0f799c. The report and presentation use the same results. BN graph remains provisional. Historical evaluation rows were previously inspected.

## Product Principles

- Only Predict submits the current inputs and runs inference; typing does not require Enter before submission.
- Distinguish model scores from established clinical risks.
- Compare strengths and error trade-offs with actual evidence.
- Keep common controls readable and usable on mobile.

## Current scope
The user restricted the hosted app to two sections only: Prediction and Model studio. No Data story, Project page, or report/presentation download section. Keep explicit Predict submission and verified model evidence.
