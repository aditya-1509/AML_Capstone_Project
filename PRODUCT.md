# HeartLens

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Streamlit, delegated by the user's explicit hosting request.

## Users

People exploring heart disease prediction and the model behind it.

## Product Purpose

Predict binary heart disease presence from the supplied UCI dataset and explain the data, model selection, and limitations in an interactive app.

## Operating Context

The project includes `heart_disease_uci.csv` and `uci-heart-disease.ipynb`. The notebook defines target `num > 0`, removes site, ID, ca, and thal from predictors, and uses 11 patient features.

## Capabilities and Constraints

Train and compare models without preprocessing leakage. Provide single-profile predictions, model evidence, and a plain-language data story. Avoid emoji in the interface. The result is educational, not a clinical diagnosis.

## Evidence on Hand

The 920-row CSV and notebook are the supplied source material. No visual inspiration file was supplied; the user specified the notebook as the reference.

## Product Principles

- Show predictions and evaluation from an actually trained model.
- Make input meanings and missing values understandable.
- Show generalization limits across the four source hospitals.
