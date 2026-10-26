"""Heartlens: an interactive, read-only companion to the Phase 1 study."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from advanced_modeling import CATEGORY_OPTIONS, FEATURES
from study import (MODEL_NAMES, evidence, evaluation_rows,
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


def chart(fig, height=330, *, key=None):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans, sans-serif", color="#3B261E", size=13),
                      margin=dict(l=8, r=12, t=30, b=24), colorway=COLORS,
                      legend=dict(orientation="h", y=-0.25))
    fig.update_xaxes(gridcolor="#E5D7C7", zeroline=False)
    fig.update_yaxes(gridcolor="#E5D7C7", zeroline=False)
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def load_preset():
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state["input_" + key] = value


def hero(title, description):
    with st.container(key="page-hero"):
        st.title(title)
        st.write(description)


def yes_no(value):
    return "Not recorded" if value is None else "Yes" if value else "No"


def live_demo():
    hero("Understand the signal behind a prediction.",
         "Explore how our frozen models classify historical heart disease presence. Change a fictional profile and the prediction updates automatically.")
    st.info("For education and demonstration only. This model is not a diagnosis. A low score cannot rule out disease.", icon=None)
    if "input_age" not in st.session_state:
        for key, value in PRESETS["Profile A"].items():
            st.session_state["input_" + key] = value
    profile = {}
    with st.container(border=True, key="profile-panel"):
        st.subheader("Profile inputs")
        st.caption("Use fictional values. Leave optional measurements blank if unavailable; the saved model handles missing values.")
        st.selectbox("Start with an example", list(PRESETS), key="preset", on_change=load_preset,
                     help="Examples are invented to demonstrate the model; they are not patient records.")
        a, b = st.columns(2, gap="large")
        with a:
            profile["age"] = st.number_input("Age (years)", 18, 100, key="input_age")
            profile["sex"] = st.selectbox("Recorded sex", CATEGORY_OPTIONS["sex"], key="input_sex")
            profile["cp"] = st.selectbox("Chest pain type", CATEGORY_OPTIONS["cp"], key="input_cp",
                                       help="‘Asymptomatic’ is a source dataset category.")
            profile["trestbps"] = st.number_input("Resting blood pressure (mm Hg)", 0, 260, key="input_trestbps", value=None,
                                                 help="Zero is treated as a missing measurement.")
            profile["chol"] = st.number_input("Cholesterol (mg/dL)", 0, 700, key="input_chol", value=None,
                                              help="Zero is treated as a missing measurement.")
            profile["fbs"] = st.selectbox("Fasting blood sugar >120 mg/dL", [False, True, None], key="input_fbs", format_func=yes_no)
        with b:
            profile["restecg"] = st.selectbox("Resting ECG", [*CATEGORY_OPTIONS["restecg"], None], key="input_restecg", format_func=lambda x: "Not recorded" if x is None else x.capitalize())
            profile["thalch"] = st.number_input("Maximum heart rate (bpm)", 40, 240, key="input_thalch", value=None)
            profile["exang"] = st.selectbox("Exercise-induced angina", [False, True, None], key="input_exang", format_func=yes_no)
            profile["oldpeak"] = st.number_input("ST depression (oldpeak)", -3.0, 10.0, step=0.1, key="input_oldpeak", value=None)
            profile["slope"] = st.selectbox("ST segment slope", [*CATEGORY_OPTIONS["slope"], None], key="input_slope", format_func=lambda x: "Not recorded" if x is None else x.capitalize())
        st.caption("Prediction updates automatically after a selection, a stepper change, or when you finish editing a number. No submission needed.")
    compare = st.toggle("Compare all five models", value=True)
    try:
        predictions = predict_profile(profile, compare=compare)
    except ValueError as error:
        st.error(str(error))
        return
    row = predictions.iloc[0]
    above = row["Score"] >= row["Study cutoff"]
    st.html(f'<div class="mobile-result"><span>Random forest · live score</span><strong>{row["Score"]:.1%}</strong><small>{"Above" if above else "Below"} 38% study cutoff</small></div>')
    st.subheader("Live prediction")
    with st.container(border=True, key="result-panel"):
        a, b = st.columns([1, 2], vertical_alignment="center")
        with a:
            st.html(f'<div class="score">{row["Score"]:.1%}</div><div class="score-label">Random forest model score</div>')
        with b:
            st.markdown(f'**{"Above" if above else "Below"} the study cutoff** · Predicted class {"1 (disease present)" if above else "0 (disease absent)"}.')
            st.caption("The saved cutoff is 0.38. This score is a model estimate, not a validated personal probability.")
            fig = go.Figure(go.Bar(x=[row["Score"]], y=[""], orientation="h", marker_color=COLORS[0], hovertemplate="Score %{x:.1%}<extra></extra>"))
            fig.add_vline(x=.38, line_color="#3B261E", line_dash="dash")
            fig.update_xaxes(range=[0, 1], tickformat=".0%", fixedrange=True)
            fig.update_yaxes(visible=False, fixedrange=True)
            chart(fig, 110)
    missing = sum(pd.isna(v) or (k in {"trestbps", "chol"} and v == 0) for k, v in profile.items())
    st.caption(f"{missing} of 11 inputs missing. " + ("The frozen pipelines handle missing values; uncertainty is not quantified." if missing else "All inputs supplied. Values are not deliberately saved by this app."))
    if compare:
        st.subheader("How the models respond")
        fig = px.bar(predictions.iloc[::-1], x="Score", y="Model", orientation="h", text="Score", color="Model", color_discrete_map=dict(zip(MODEL_NAMES,COLORS)))
        fig.update_traces(texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_range=[0, 1.12], xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
        chart(fig, 310)
        st.dataframe(predictions[["Model", "Study cutoff", "Decision"]], hide_index=True, width="stretch")
        st.caption("Each model uses its own frozen cutoff. Agreement is not evidence of clinical reliability.")
    with st.expander("Why was random forest selected?"):
        st.write("It had the highest mean nested cross-validation F1 (0.836). At its chosen cutoff, it detected 98 of 102 positives in the retrospective test set, while producing 32 false positives. Logistic regression scored better on test F1 and ROC-AUC; model selection used development results.")


def roc_plot(names):
    fig = go.Figure()
    for name in names:
        rows = evaluation_rows().query("model == @name")
        fpr, tpr, _ = roc_curve(rows.target, rows.probability)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                line_color=COLORS[MODEL_NAMES.index(name)]))
    fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="#765238"))
    fig.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate",showlegend=len(names)>1)
    chart(fig, 330 if len(names)==1 else 450)


def confusion_plot(score, *, key):
    fig = px.imshow([[score["tn"], score["fp"]], [score["fn"], score["tp"]]],
                    x=["Predicted absent", "Predicted present"], y=["Recorded absent", "Recorded present"],
                    text_auto=True, color_continuous_scale=["#F1E5D7", "#A84E32"], aspect="auto")
    fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
    chart(fig, 330, key=key)


def models():
    hero("How well does the model hold up?",
         "Models were compared using nested cross-validation on development data, frozen, then evaluated on the saved retrospective test split.")
    table = model_table()
    winner = table.loc[table.Model == "Random forest"].iloc[0]
    lead = winner["Nested CV F1"] - table.loc[table.Model != "Random forest", "Nested CV F1"].max()
    st.subheader("Model selection")
    st.write("Nested 5 × 3 cross-validation compares model families using mean F1. Random forest achieved the highest score under the study’s selection rule. Higher is better.")
    a, b = st.columns(2)
    a.metric("Random forest · mean CV F1", f'{winner["Nested CV F1"]:.3f}', border=True)
    b.metric("Lead over the runner-up", f"+{lead:.3f}", border=True)
    metric = st.selectbox("Compare models by", ["Nested CV F1", "f1", "roc_auc", "recall", "specificity", "accuracy"], format_func=lambda x: {"f1":"Test F1", "roc_auc":"Test ROC-AUC", "recall":"Test recall", "specificity":"Test specificity", "accuracy":"Test accuracy"}.get(x,x))
    ranking=table.sort_values(metric)
    fig=px.bar(ranking,y="Model",x=metric,orientation="h",text_auto=".3f")
    fig.update_traces(marker_color=["#A84E32" if name == "Random forest" else "#C8A68B" for name in ranking.Model],textposition="outside",cliponaxis=False)
    fig.update_layout(showlegend=False,xaxis_range=[0,1.06],yaxis_title=None)
    chart(fig,340)
    st.info(f"Why random forest was selected: mean validation F1 was {winner['Nested CV F1']:.3f}, ahead of XGBoost by {lead:.3f}. This small lead supports the selection rule; it does not establish universal superiority.",icon=None)
    with st.expander("Full numerical comparison"):
        st.dataframe(table, hide_index=True, width="stretch", column_config={c:st.column_config.NumberColumn(format="%.3f") for c in table.columns if c != "Model"})
    st.caption("The random forest cutoff of 0.38 was selected on development data. Models and cutoffs were frozen before the retrospective test evaluation.")
    st.divider()
    st.subheader("Retrospective test performance")
    a,b,c=st.columns(3)
    a.metric("F1 score",f'{winner.f1:.3f}',border=True)
    b.metric("Recall",f'{winner.recall:.3f}',border=True)
    c.metric("ROC-AUC",f'{winner.roc_auc:.3f}',border=True)
    st.caption("184 test records. This split has already been inspected. These are retrospective results, not fresh external validation.")
    a,b=st.columns(2)
    with a:
        st.subheader("ROC curve")
        roc_plot(["Random forest"])
    with b:
        st.subheader("Confusion matrix")
        confusion_plot(threshold_scores("Random forest", .38), key="study_confusion")
    st.info("A random split measures performance on records from the same four hospitals. It does not establish clinical reliability in new hospitals or populations.",icon=None)
    st.divider()
    st.subheader("Explore the decision boundary")
    name = st.selectbox("Model", MODEL_NAMES, key="boundary_model")
    official=evidence()["freeze"]["thresholds"][name]
    cutoff=st.slider("Exploration cutoff",0.0,1.0,official,.01,key="cutoff_"+name)
    st.caption(f"Saved study cutoff: {official:.2f}. Explore saved test predictions here; this does not change the study or the live demo.")
    score=threshold_scores(name,cutoff)
    a,b,c,d=st.columns(4)
    a.metric("Recall",f'{score["recall"]:.1%}')
    b.metric("Specificity",f'{score["specificity"]:.1%}')
    c.metric("False negatives",score["fn"])
    d.metric("False positives",score["fp"])
    confusion_plot(score, key="exploration_confusion")
    st.write("Lowering the cutoff usually captures more positives but creates more false positives. At its study cutoff, random forest has 96.1% recall and 61.0% specificity. Logistic regression leads on test F1 (0.853) and ROC-AUC (0.912), with a simpler, more interpretable model.")
    with st.expander("ROC curves · all five models"):
        roc_plot(MODEL_NAMES)
    st.subheader("Does it travel across hospitals?")
    st.write("Four development-only held-hospital-out evaluations reveal distribution shift. Each experiment selected its own model and cutoff; these are separate from the pooled test above.")
    st.dataframe(hospital_table(),hide_index=True,width="stretch",column_config={c:st.column_config.NumberColumn(format="%.3f") for c in ["F1","Always positive F1","Specificity","ROC-AUC"]})
    st.caption("Switzerland has only 8 negative records in this evaluation. VA Long Beach specificity is 5%; high F1 alone obscures that failure. Always-positive F1 is a reference classifier, not a proposed solution.")
