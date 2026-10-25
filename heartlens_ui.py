"""Heartlens: an interactive, read-only companion to the Phase 1 study."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from advanced_modeling import CATEGORY_OPTIONS, FEATURES
from study import (ROOT, RUN_ID, MODEL_NAMES, cohort, evidence, evaluation_rows,
                   hospital_table, model_table, predict_profile, threshold_scores)


COLORS = ["#A84E32", "#765238", "#BC783B", "#526955", "#9A6170"]
LABELS = {"age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol",
          "thalch": "Maximum heart rate", "oldpeak": "ST depression", "sex": "Recorded sex",
          "cp": "Chest pain type", "fbs": "Fasting blood sugar >120 mg/dL",
          "restecg": "Resting ECG", "exang": "Exercise-induced angina", "slope": "ST segment slope"}
PRESETS = {
    "Profile A": dict(age=45, trestbps=120, chol=220, thalch=170, oldpeak=0.0, sex="Female", cp="atypical angina", fbs=False, restecg="normal", exang=False, slope="upsloping"),
    "Profile B": dict(age=65, trestbps=160, chol=280, thalch=110, oldpeak=2.5, sex="Male", cp="asymptomatic", fbs=True, restecg="lv hypertrophy", exang=True, slope="flat"),
    "Missing values": dict(age=55, trestbps=None, chol=None, thalch=None, oldpeak=None, sex="Male", cp="non-anginal", fbs=None, restecg=None, exang=None, slope=None),
}


def chart(fig, height=330):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans, sans-serif", color="#3B261E", size=13),
                      margin=dict(l=8, r=12, t=30, b=24), colorway=COLORS,
                      legend=dict(orientation="h", y=-0.25))
    fig.update_xaxes(gridcolor="#E5D7C7", zeroline=False)
    fig.update_yaxes(gridcolor="#E5D7C7", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def load_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state["input_" + key] = value


def live_demo():
    hero, art = st.columns([4, 1], vertical_alignment="center")
    with hero:
        st.title("A closer look at the heart.")
        st.markdown("Explore how a heart disease classifier responds to a **fictional profile**. Change a value; the prediction updates with it.")
    with art:
        with st.container(key="hero-art"):
            st.image(str(ROOT / "assets/heart.webp"))
    st.caption("Educational model demo • Predicts historical disease presence, not future risk or a medical diagnosis.")
    st.divider()
    if "input_age" not in st.session_state:
        for key, value in PRESETS["Profile A"].items():
            st.session_state["input_" + key] = value
    inputs, output = st.columns([1.35, 1], gap="large")
    with inputs:
        st.subheader("Build a fictional profile")
        st.selectbox("Start with an example", list(PRESETS), key="preset", on_change=load_preset,
                     help="Examples are invented to demonstrate the model; they are not patient records.")
        profile = {}
        a, b = st.columns(2)
        with a:
            profile["age"] = st.slider("Age (years)", 18, 100, key="input_age")
            profile["sex"] = st.selectbox("Recorded sex", CATEGORY_OPTIONS["sex"], key="input_sex")
        with b:
            profile["cp"] = st.selectbox("Chest pain type", CATEGORY_OPTIONS["cp"], key="input_cp",
                                       help="‘Asymptomatic’ is a source dataset category.")
            profile["exang"] = st.selectbox("Exercise-induced angina", [False, True, None], key="input_exang",
                                           format_func=lambda x: "Not recorded" if x is None else "Yes" if x else "No")
        st.markdown("**Measurements**")
        st.caption("Clear an optional measurement to explore missing data. Blood pressure or cholesterol of zero is treated as missing.")
        a, b = st.columns(2)
        with a:
            profile["trestbps"] = st.number_input("Resting blood pressure (mm Hg)", 0, 260, key="input_trestbps", value=None)
            profile["thalch"] = st.number_input("Maximum heart rate (bpm)", 40, 240, key="input_thalch", value=None)
        with b:
            profile["chol"] = st.number_input("Cholesterol (mg/dL)", 0, 700, key="input_chol", value=None)
            profile["oldpeak"] = st.number_input("ST depression (oldpeak)", -3.0, 10.0, step=0.1, key="input_oldpeak", value=None)
        with st.expander("ECG & additional observations", expanded=True):
            profile["restecg"] = st.selectbox("Resting ECG", [*CATEGORY_OPTIONS["restecg"], None], key="input_restecg", format_func=lambda x: "Not recorded" if x is None else x.capitalize())
            profile["fbs"] = st.selectbox("Fasting blood sugar >120 mg/dL", [False, True, None], key="input_fbs", format_func=lambda x: "Not recorded" if x is None else "Yes" if x else "No")
            profile["slope"] = st.selectbox("ST segment slope", [*CATEGORY_OPTIONS["slope"], None], key="input_slope", format_func=lambda x: "Not recorded" if x is None else x.capitalize())
    with output:
        compare = st.toggle("Compare all five models", value=True)
        try:
            predictions = predict_profile(profile, compare=compare)
        except ValueError as error:
            st.error(str(error))
            return
        row = predictions.iloc[0]
        above = row["Score"] >= row["Study cutoff"]
        st.html(f'<div class="mobile-result"><span>Random forest · live score</span><strong>{row["Score"]:.1%}</strong><small>{"Above" if above else "Below"} 38% study cutoff</small></div>')
        with st.container(border=True):
            st.markdown("**Random forest** · Study-selected model")
            st.html(f'<div class="score">{row["Score"]:.1%}</div><div class="score-label">Model score for disease presence</div>')
            fig = go.Figure(go.Bar(x=[row["Score"]], y=[""], orientation="h", marker_color=COLORS[0], hovertemplate="Score %{x:.1%}<extra></extra>"))
            fig.add_vline(x=row["Study cutoff"], line_color="#3B261E", line_dash="dash", annotation_text="Study cutoff · 38%", annotation_position="top right")
            fig.update_xaxes(range=[0, 1], tickformat=".0%", fixedrange=True)
            fig.update_yaxes(visible=False, fixedrange=True)
            fig.update_layout(bargap=.55)
            chart(fig, 125)
            st.markdown(f'**{"Above" if above else "Below"} the study cutoff** · Predicted class {"1 (disease present)" if above else "0 (disease absent)"}.')
            st.caption("This score is a model estimate, not a validated personal probability. The saved decision cutoff is 0.38.")
        missing = sum(pd.isna(v) or (k in {"trestbps", "chol"} and v == 0) for k, v in profile.items())
        st.caption(f"{missing} of 11 inputs missing. " + ("The frozen pipelines handle missing values; uncertainty is not quantified." if missing else "All inputs supplied. Values are not deliberately saved by this app."))
        if compare:
            st.markdown("#### How the models respond")
            fig = px.bar(predictions.iloc[::-1], x="Score", y="Model", orientation="h", text="Score", color="Model", color_discrete_sequence=COLORS[::-1])
            fig.update_traces(texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_range=[0, 1.15], xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
            chart(fig, 270)
            st.dataframe(predictions[["Model", "Study cutoff", "Decision"]], hide_index=True, width="stretch")
            st.caption("Each model uses its own frozen cutoff. Agreement is not evidence of clinical reliability.")
        with st.expander("Why was random forest selected?"):
            st.write("It had the highest mean nested cross-validation F1 (0.836). At its chosen cutoff, it detected 98 of 102 positives in the retrospective test set, while producing 32 false positives. Logistic regression scored better on test F1 and ROC-AUC; model selection used development results.")


def models():
    st.title("Every model has a trade-off.")
    st.write("Compare saved study results, then move the decision cutoff to see what changes.")
    table = model_table()
    a, b = st.columns([1.5, 1])
    with a:
        metric = st.selectbox("Compare models by", ["Nested CV F1", "f1", "roc_auc", "recall", "specificity", "accuracy"], format_func=lambda x: {"f1":"Test F1", "roc_auc":"Test ROC-AUC", "recall":"Test recall", "specificity":"Test specificity", "accuracy":"Test accuracy"}.get(x,x))
        fig = px.bar(table.sort_values(metric), y="Model", x=metric, orientation="h", color="Model", color_discrete_map=dict(zip(MODEL_NAMES,COLORS)), text_auto=".3f")
        fig.update_layout(showlegend=False, xaxis_range=[0,1.08], yaxis_title=None)
        chart(fig)
    with b:
        st.subheader("Why the forest?")
        st.write("Random forest leads on development CV F1 (0.836), capturing nonlinear patterns and interactions. Its 0.38 cutoff favors recall: 96.1% on the saved test split.")
        st.write("The cost is lower specificity (61.0%). Logistic regression has higher test F1 (0.853) and ROC-AUC (0.912), with a simpler, more interpretable model.")
        st.caption("The 184-record test split has already been inspected. These are retrospective results, not a fresh external validation.")
    with st.expander("Full numerical comparison"):
        st.dataframe(table, hide_index=True, width="stretch", column_config={c:st.column_config.NumberColumn(format="%.3f") for c in table.columns if c != "Model"})
    st.divider()
    st.subheader("Explore the decision boundary")
    a, b = st.columns([1, 1.4], gap="large")
    with a:
        name = st.selectbox("Model", MODEL_NAMES, key="boundary_model")
        official = evidence()["freeze"]["thresholds"][name]
        cutoff = st.slider("Exploration cutoff", 0.0, 1.0, official, .01, key="cutoff_" + name)
        st.caption(f"Saved study cutoff: {official:.2f}. This control explores saved test predictions; it does not change the study or the live demo.")
        score = threshold_scores(name, cutoff)
        x, y = st.columns(2)
        x.metric("Recall", f'{score["recall"]:.1%}')
        y.metric("Specificity", f'{score["specificity"]:.1%}')
        x.metric("False negatives", score["fn"])
        y.metric("False positives", score["fp"])
    with b:
        fig = px.imshow([[score["tn"], score["fp"]], [score["fn"], score["tp"]]], x=["Predicted absent", "Predicted present"], y=["Recorded absent", "Recorded present"], text_auto=True, color_continuous_scale=["#F1E5D7", "#A84E32"], aspect="auto")
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
        chart(fig, 300)
    with st.expander("ROC curves · all five models"):
        fig = go.Figure()
        for name, color in zip(MODEL_NAMES,COLORS):
            rows = evaluation_rows().query("model == @name")
            fpr, tpr, _ = roc_curve(rows.target, rows.probability)
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=name,line_color=color))
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="#765238"))
        fig.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate")
        chart(fig,450)
    st.subheader("Does it travel across hospitals?")
    st.write("Four separate development-only, held-hospital-out evaluations reveal distribution shift. Each hospital experiment selected its own model and cutoff; these are separate from the pooled test above.")
    st.dataframe(hospital_table(), hide_index=True, width="stretch", column_config={c:st.column_config.NumberColumn(format="%.3f") for c in ["F1","Always positive F1","Specificity","ROC-AUC"]})
    st.caption("Switzerland has only 8 negative records in this evaluation. VA Long Beach specificity is 5%; high F1 alone obscures that failure. Always-positive F1 is a reference classifier, not a proposed solution.")


def data_explorer():
    st.title("Meet the cohort.")
    st.write("918 historical records after removing two exact duplicates. Explore how the sample and missing data differ across sites.")
    data = cohort().copy()
    a,b,c = st.columns([1.5,1,1.5])
    sites = a.multiselect("Hospitals", sorted(data.dataset.unique()), default=sorted(data.dataset.unique()))
    sexes = b.multiselect("Recorded sex", ["Female","Male"],default=["Female","Male"])
    age = c.slider("Age range (years)",int(data.age.min()),int(data.age.max()),(int(data.age.min()),int(data.age.max())))
    selected = data[data.dataset.isin(sites) & data.sex.isin(sexes) & data.age.between(*age)]
    if selected.empty:
        st.info("No records match these filters. Add a hospital or recorded sex, or widen the age range.")
        return
    a,b,c = st.columns(3)
    a.metric("Records in view",len(selected))
    b.metric("Recorded positive", f'{(selected.num>0).sum()} / {len(selected)}')
    c.metric("Positive proportion",f'{(selected.num>0).mean():.1%}')
    st.caption("Sample proportions describe this historical dataset; they are not population prevalence estimates.")
    a,b=st.columns(2)
    with a:
        st.subheader("Age distribution")
        fig=px.histogram(selected,x="age",color="Disease presence",nbins=20,barmode="overlay",opacity=.75,color_discrete_map={"Recorded positive":COLORS[0],"Recorded negative":"#526955"},labels={"age":"Age (years)"})
        chart(fig)
    with b:
        st.subheader("Where measurements are missing")
        missing=selected[FEATURES].isna().sum().sort_values()
        frame=pd.DataFrame({"Feature":[LABELS[x] for x in missing.index],"Missing records":missing.values})
        fig=px.bar(frame,x="Missing records",y="Feature",orientation="h",color_discrete_sequence=[COLORS[0]])
        fig.update_layout(yaxis_title=None)
        chart(fig,390)
    counts=selected.groupby(["dataset","Disease presence"]).size().reset_index(name="Records")
    st.subheader("A different mix at every site")
    fig=px.bar(counts,x="dataset",y="Records",color="Disease presence",color_discrete_map={"Recorded positive":COLORS[0],"Recorded negative":"#526955"},labels={"dataset":"Hospital"})
    chart(fig)
    with st.expander("Explore the filtered records"):
        safe=selected[["dataset",*FEATURES,"Disease presence"]].rename(columns={"dataset":"Hospital",**LABELS})
        st.dataframe(safe,hide_index=True,width="stretch")
        st.download_button("Download filtered cohort CSV",safe.to_csv(index=False),file_name="heartlens_filtered_cohort.csv",mime="text/csv")
    st.caption("ca and thal were excluded for extensive missingness; hospital, row ID and outcome are not model predictors. Zero blood pressure and zero cholesterol are interpreted as missing.")


def project():
    st.title("From records to evidence.")
    st.write("A reproducible comparison of classical machine learning methods for binary heart disease classification using the four-site UCI-derived dataset.")
    a,b=st.columns([1.5,1],gap="large")
    with a:
        st.subheader("The study, step by step")
        st.markdown("1. **Clean the source:** 920 records become 918 after exact deduplication; label disease presence as num > 0.\n2. **Separate the data:** 734 development records and 184 test records. Split development into 550 fitting and 184 cutoff-selection records.\n3. **Compare fairly:** nested 5 × 3 cross-validation with preprocessing inside each fitted pipeline.\n4. **Freeze decisions:** select the model using mean CV F1 and cutoffs using development data.\n5. **Evaluate:** inspect saved test predictions, calibration and development-only held-hospital-out checks.")
        st.subheader("Challenges & limits")
        st.markdown("- **Missing measurements:** severe missingness excludes ca and thal; remaining missingness is handled by the saved models.\n- **Hospital shift:** disease labels and feature distributions differ substantially by site.\n- **Small subgroups:** some specificity estimates rely on very few negatives.\n- **Retrospective validation:** the test split has already informed discussion; a new external cohort is needed.\n- **Interpretation:** the Bayesian network is provisional and statistical. Its graph does not establish causation.")
    with b:
        with st.container(border=True):
            st.subheader("Take the study with you")
            for filename,label in [("phase1_report.pdf","Download IEEE report"),("phase1_presentation.pdf","Download presentation")]:
                path=ROOT/"assets/documents"/filename
                if path.exists():
                    st.download_button(label,path.read_bytes(),file_name=filename,mime="application/pdf",width="stretch")
            st.link_button("Explore the GitHub repository","https://github.com/nishtha-09-gupta/AML_Capstone_Project",width="stretch")
        st.markdown("#### The team")
        st.write("Nishtha Gupta · Aditya Srivastava · Aanya Mehrotra · Shrijan Sanidhya · Abhay Mallik")
        st.caption("CSAI capstone · Phase 1: Machine Learning")
    with st.expander("Reproducibility & provenance"):
        st.write(f"Verified artifact run: `{RUN_ID}`. The app verifies source and saved-model checksums before inference. It uses frozen models and never trains on visitor inputs.")
        st.write("Dataset: the repository’s heart_disease_uci.csv, derived from the UCI Heart Disease collection. Positive labels represent recorded historical disease presence. Record identity linkage beyond exact duplication is unverified.")
        st.link_button("UCI Heart Disease dataset","https://archive.ics.uci.edu/dataset/45/heart+disease")
