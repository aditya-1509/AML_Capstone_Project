# HeartLens

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Streamlit, explicitly requested for implementation and hosting. Python inference uses the existing frozen scikit-learn, XGBoost and Bayesian Network models.

## Users

The capstone team and visitors reviewing its Phase 1 machine learning project.

## Product Purpose

Let visitors change a fictional profile and see the prediction and model comparisons update automatically, without pressing a Predict button. Make the project's data, results and limitations understandable.

## Operating Context

The app accompanies Team T008's IEEE report and presentation. It is an academic demonstration of historical disease-presence classification, not clinical diagnosis or future-event prediction.

## Capabilities and Constraints

- Live profile inputs and inference from verified saved models.
- Interactive model comparisons, threshold exploration and cohort filters.
- Project report and presentation downloads.
- Preserve original models, source data, selection rules and recorded metrics.
- Default Random Forest and cutoff 0.38 follow development selection. Retrospective rankings do not change that choice.
- No model training during app execution. Profile values remain in the session and are not intentionally written to disk.
- Deploy on Streamlit Community Cloud using the shared GitHub repository.
- Keep README unchanged in this work.

## Brand Commitments

The user requested warm colors. Continue the cream, terracotta and warm brown identity of the presentation. Use the existing generated terracotta-heart illustration.

## Evidence on Hand

Verified five-family experiment: artifacts/runs/20261025T151903Z_ea0f799c. The report and presentation use the same results. BN graph remains provisional. Historical evaluation rows were previously inspected.

## Product Principles

- Input changes immediately update visible results.
- Distinguish model scores from established clinical risks.
- Compare strengths and error trade-offs with actual evidence.
- Keep common controls readable and usable on mobile.
