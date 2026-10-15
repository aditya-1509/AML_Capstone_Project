"""HeartLens: an interactive UCI heart-disease prediction app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modeling import clean_features, deduplicate_records, validate_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "heart_model.joblib"
COLORS = {"navy": "#102b3f", "teal": "#087f78", "muted": "#5d7180", "coral": "#bd5547"}

st.set_page_config(page_title="HeartLens | Heart disease prediction", page_icon=":material/favorite:", layout="wide",
                   initial_sidebar_state="auto")


@st.cache_resource
def load_artifact(model_mtime_ns: int):
    """Reload the model when a new trained artifact replaces the file."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    frame = pd.read_csv(ROOT / "heart_disease_uci.csv")
    frame = deduplicate_records(frame)
    frame["Disease status"] = np.where(frame["num"] > 0, "Disease present", "No disease")
    return frame


def style():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { color-scheme: light; }
      html, body, [class*="css"], [data-testid="stApp"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #f5f8f7; color: #102b3f; }
      [data-testid="stSidebar"] { background: #102b3f; }
      [data-testid="stSidebar"] * { color: #f3f7f6 !important; }
      [data-testid="stSidebar"] .stRadio label { padding: .22rem .15rem; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { line-height: 1.55; }
      [data-testid="stHeader"] { background: #f5f8f7; }
      .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 5rem; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #102b3f; letter-spacing: -.025em; }
      h1 { font-size: clamp(2.25rem, 4vw, 3.65rem) !important; line-height: 1.11; font-weight: 800 !important; }
      h2 { font-size: 1.65rem !important; font-weight: 750 !important; }
      h3 { font-size: 1.16rem !important; font-weight: 700 !important; }
      p, li, label { color: #253e4c; }
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #f3f7f6 !important; }
      .brand { color: #f5fbf9; font: 800 1.45rem 'Manrope', sans-serif; letter-spacing: -.04em; padding: .4rem 0 .05rem; }
      .brand-mark { display:inline-block; width: 18px; height:18px; border: 4px solid #78d7c6; border-radius: 50%; margin-right: 9px; vertical-align:-1px; }
      .sidebar-copy { color: #c8d9dc; font-size: .86rem; margin-bottom: 1.7rem; }
      .hero { background: #dfeeea; border-radius: 18px; padding: clamp(1.55rem, 3vw, 2.6rem); margin-bottom: 1.5rem; position: relative; overflow: hidden; }
      .hero:after { content:''; width: 240px; height: 240px; border: 24px solid rgba(8,127,120,.13); border-radius: 50%; position:absolute; right:-55px; top:-73px; }
      .hero p { max-width: 63ch; font-size: 1.04rem; margin-bottom: 0; }
      .hero h1 { max-width: 850px; margin: .15rem 0 .7rem; }
      .section-intro { color:#5d7180; max-width:72ch; margin-bottom:1.2rem; }
      .thin-rule { border-top: 1px solid #c7d8d5; margin: 1.4rem 0 1.8rem; }
      .result { padding: 1.65rem 1.8rem; border-radius: 16px; background:#e0f0eb; color:#123e3d; margin-top: 1.5rem; }
      .result.positive { background:#faeae6; color:#733c35; }
      .result-label { font-size:.78rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:inherit; }
      .result-number { font: 800 clamp(2.5rem, 5vw, 4.2rem) 'Manrope',sans-serif; letter-spacing:-.04em; line-height:1.1; margin:.3rem 0; }
      .result-title { font:700 1.25rem 'Manrope',sans-serif; margin:.2rem 0 .45rem; }
      .result p { color:inherit; margin:0; max-width:75ch; }
      .note { background:#edf3f3; padding:1rem 1.15rem; border-radius:12px; font-size:.9rem; color:#365460; margin:1.1rem 0; }
      .small-note { font-size:.84rem; color:#5d7180; }
      .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button { background:#087f78; border:0; color:white; border-radius:10px; padding:.62rem 1.05rem; font-weight:700; }
      .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover { background:#086b66; color:white; }
      button:focus-visible, input:focus-visible, [role="radio"]:focus-visible { outline:3px solid #31aa9e !important; outline-offset:2px; }
      ::selection { background:#99d9cf; color:#102b3f; }
      [data-testid="stMetric"] { background:#fff; padding:1rem 1.1rem; border-radius:12px; }
      [data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; color:#102b3f; font-weight:800; }
      [data-testid="stExpander"] { background:#fff; border-radius:12px; border:0; }
      .chapter { font-size:.74rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:#087f78; margin-bottom:-.4rem; }
      .takeaway, .summary { background:#fff; border-left:4px solid #087f78; border-radius:12px; padding:1rem 1.25rem; margin:.4rem 0 .6rem; }
      .takeaway ul { margin:.35rem 0 0; padding-left:1.1rem; } .takeaway li { margin-bottom:.35rem; line-height:1.55; }
      .takeaway-label { font-size:.74rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#087f78; }
      .summary { background:#dfeeea; } .summary p { margin:.35rem 0 0; font-size:1.04rem; line-height:1.6; }
      @media(max-width:700px) { .block-container { padding: 1.2rem 1rem 3rem; } .hero { padding:1.35rem; } .hero:after { width:140px; height:140px; right:-70px; top:-65px; } }
    </style>
    """, unsafe_allow_html=True)


def hero(title, description):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def figure_style(fig, height=365):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=COLORS["navy"]),
        margin=dict(l=15, r=20, t=25, b=20),
        legend=dict(orientation="h", y=1.13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#dbe6e4", zeroline=False)
    return fig


def prediction_page(bundle):
    hero("Understand the signal behind a prediction.",
         "Enter a patient profile to explore how our trained model classifies heart disease presence in the UCI dataset. Every input matches the features used during training.")
    st.markdown('<div class="note">For education and demonstration only. This model is not a diagnosis or a substitute for a clinician. A low score cannot rule out disease.</div>', unsafe_allow_html=True)

    with st.form("patient_profile"):
        st.subheader("Patient profile")
        st.caption("If a measurement is unavailable, leave its optional field blank. The model uses a training-set imputation value.")
        left, right = st.columns(2, gap="large")
        with left:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=54, step=1)
            sex = st.selectbox("Sex recorded in the dataset", ["Female", "Male"], index=1)
            cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"], index=3,
                              help="Dataset terminology. Asymptomatic means no typical chest pain was reported.")
            trestbps = st.text_input("Resting blood pressure (mm Hg)", value="130", help="Leave blank if unavailable.")
            chol = st.text_input("Serum cholesterol (mg/dL)", value="245", help="Leave blank if unavailable; 0 is treated as missing.")
            fbs = st.selectbox("Fasting blood sugar above 120 mg/dL", ["No", "Yes", "Not available"])
        with right:
            restecg = st.selectbox("Resting ECG result", ["normal", "st-t abnormality", "lv hypertrophy", "Not available"])
            thalch = st.text_input("Maximum heart rate achieved (bpm)", value="150", help="Leave blank if unavailable.")
            exang = st.selectbox("Exercise-induced angina", ["No", "Yes", "Not available"])
            oldpeak = st.text_input("ST depression (oldpeak)", value="1.0", help="Leave blank if unavailable. Negative values occur in the source data.")
            slope = st.selectbox("Peak exercise ST slope", ["upsloping", "flat", "downsloping", "Not available"])
        submitted = st.form_submit_button("Run prediction", type="primary", width="stretch")

    if submitted:
        def optional_number(value):
            return np.nan if not value.strip() else value.strip()

        def optional_bool(value):
            return np.nan if value == "Not available" else (value == "Yes")

        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp,
            "trestbps": optional_number(trestbps), "chol": optional_number(chol),
            "fbs": optional_bool(fbs), "restecg": np.nan if restecg == "Not available" else restecg,
            "thalch": optional_number(thalch), "exang": optional_bool(exang),
            "oldpeak": optional_number(oldpeak), "slope": np.nan if slope == "Not available" else slope,
        }])
        errors = validate_features(row)
        if errors:
            st.error("Please correct the profile: " + "; ".join(errors))
        else:
            p = float(bundle["model"].predict_proba(clean_features(row))[0, 1])
            st.session_state["prediction"] = {"probability": p, "inputs": row.to_dict("records")[0]}

    if "prediction" in st.session_state:
        probability = st.session_state["prediction"]["probability"]
        threshold = bundle["metadata"]["threshold"]
        positive = probability >= threshold
        title = "Disease pattern detected" if positive else "No disease pattern detected"
        detail = ("The model classified this profile as disease present. A clinical assessment is still needed." if positive
                  else "The model classified this profile as no disease. This does not rule out disease.")
        st.markdown(f'<div class="result {"positive" if positive else ""}"><div class="result-label">Model output</div><div class="result-number">{probability:.1%}</div><div class="result-title">{title}</div><p>{detail}</p></div>', unsafe_allow_html=True)
        st.caption(f"Estimated probability of the dataset label ‘disease present’. Classification cutoff: {threshold:.0%}. The score is not an individual’s clinical risk.")
        with st.expander("Review the submitted inputs"):
            details = pd.DataFrame([{"Feature": key, "Value": str(value)} for key, value in st.session_state["prediction"]["inputs"].items()])
            st.dataframe(details, hide_index=True, width="stretch")


def model_page(meta):
    hero("How well does the model hold up?", "The model was selected by five-fold cross-validation on the training split, then evaluated once on a separate test split.")
    comparison = pd.DataFrame(meta["comparison"])
    comparison = comparison[["model", "cv_f1", "cv_auc"]].sort_values("cv_f1", ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]
    runner_up = comparison.iloc[1]
    lead = winner["cv_f1"] - runner_up["cv_f1"]
    st.subheader("Model selection")
    st.markdown('<p class="section-intro">The training data was divided into five validation folds. We selected the model with the highest mean F1 score, which rewards catching true heart-disease cases while keeping false alarms low. Higher is better.</p>', unsafe_allow_html=True)
    summary_left, summary_right = st.columns(2)
    summary_left.metric(f"{meta['selected_model']} · mean CV F1", f"{winner['cv_f1']:.3f}")
    summary_right.metric("Lead over the runner-up", f"+{lead:.3f}")
    fig = go.Figure(go.Bar(
        x=comparison["cv_f1"], y=comparison["model"], orientation="h",
        marker_color=[COLORS["teal"] if name == meta["selected_model"] else "#b8cfca" for name in comparison["model"]],
        text=[f"{score:.3f}" for score in comparison["cv_f1"]], textposition="outside",
        hovertemplate="%{y}<br>Mean CV F1: %{x:.4f}<extra></extra>",
    ))
    fig.update_yaxes(categoryorder="array", categoryarray=comparison["model"].iloc[::-1].tolist(), title="")
    fig.update_xaxes(range=[0, 0.95], title="Mean five-fold CV F1")
    st.plotly_chart(figure_style(fig, 315), width="stretch")
    st.markdown(f'<div class="note"><strong>Why {meta["selected_model"]} was selected:</strong> it reached the highest mean validation F1 ({winner["cv_f1"]:.3f}), ahead of {runner_up["model"]} by {lead:.3f}. This is a small lead, so it supports the project’s selection rule without proving the model is universally superior.</div>', unsafe_allow_html=True)
    with st.expander("See all validation metrics"):
        display = comparison.rename(columns={"model": "Model", "cv_f1": "CV F1", "cv_auc": "CV ROC-AUC"})
        st.dataframe(display.style.format({"CV F1": "{:.3f}", "CV ROC-AUC": "{:.3f}"}), hide_index=True, width="stretch")
        auc_leader = max(meta["comparison"], key=lambda row: row["cv_auc"])["model"]
        st.caption(f"{auc_leader} had the highest CV ROC-AUC. {meta['selected_model']} was selected because F1 was the predefined selection metric.")
    st.caption(f"The classification cutoff ({meta['threshold']:.2f}) was chosen from training-only out-of-fold predictions to maximize F1.")
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    st.subheader("Held-out test performance")
    cols = st.columns(3)
    for col, (label, key) in zip(cols, [("F1 score", "f1"), ("Recall", "recall"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{meta['test'][key]:.3f}")
    st.caption(f"{meta['test_rows']} test records. A positive label means original UCI target > 0. Recall matters because false negatives can occur.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### ROC curve")
        curve = meta["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(color=COLORS["teal"], width=3), name="Selected model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9fb5b6", dash="dash"), name="Chance"))
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1])
        st.plotly_chart(figure_style(fig), width="stretch")
    with right:
        st.markdown("#### Confusion matrix")
        cm = meta["test"]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#e0f0eb"], [1, COLORS["teal"]]],
                        x=["No disease", "Disease"], y=["No disease", "Disease"], labels={"x": "Predicted", "y": "Actual"})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure_style(fig), width="stretch")
    st.markdown('<div class="note">A random test split measures performance on records from the same four source hospitals. It does not establish clinical reliability in new hospitals or populations.</div>', unsafe_allow_html=True)


CHEST_PAIN_LABELS = {
    "asymptomatic": "No chest pain (silent)",
    "typical angina": "Classic heart-related chest pain",
    "non-anginal": "Chest pain not linked to the heart",
    "atypical angina": "Unusual chest pain",
}
FEATURE_LABELS = {
    "cp": "Type of chest pain", "exang": "Chest pain during exercise", "oldpeak": "ECG dip during exercise",
    "chol": "Cholesterol", "sex": "Sex", "thalch": "Highest heart rate reached", "fbs": "High blood sugar",
    "age": "Age", "trestbps": "Resting blood pressure", "restecg": "Resting ECG", "slope": "ECG shape during exercise",
}


def disease_rates(frame, column, order):
    """Share of patients with heart disease in each group, in the given display order."""
    grouped = frame.dropna(subset=[column]).groupby(column, observed=True)["has_disease"].agg(rate="mean", patients="size")
    return grouped.reindex(order).dropna().reset_index()


def rate_chart(rates, column, highlight, height=320, horizontal=False):
    """One-color bar chart of disease rates; the bar the story is about is drawn darker."""
    colors = [COLORS["teal"] if value in highlight else "#a9d3cc" for value in rates[column]]
    labels = rates["rate"].map(lambda value: f"{value:.0%}")
    bar = dict(marker_color=colors, marker_cornerradius=4, text=labels, textposition="outside",
               textfont=dict(color=COLORS["navy"], size=14), customdata=rates["patients"])
    if horizontal:
        fig = go.Figure(go.Bar(x=rates["rate"], y=rates[column], orientation="h",
                               hovertemplate="<b>%{y}</b><br>%{x:.0%} had heart disease<br>%{customdata} patients<extra></extra>", **bar))
        fig.update_xaxes(tickformat=".0%", range=[0, 1.12], title="Patients with heart disease", showgrid=True, gridcolor="#dbe6e4")
        fig.update_yaxes(title="", autorange="reversed", gridcolor="rgba(0,0,0,0)")
    else:
        fig = go.Figure(go.Bar(x=rates[column].astype(str), y=rates["rate"],
                               hovertemplate="<b>%{x}</b><br>%{y:.0%} had heart disease<br>%{customdata} patients<extra></extra>", **bar))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Patients with heart disease")
        fig.update_xaxes(title="")
    fig.update_layout(bargap=0.35, showlegend=False)
    return figure_style(fig, height)


def chapter(number, title, intro):
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chapter">Chapter {number}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<p class="section-intro">{intro}</p>', unsafe_allow_html=True)


def takeaway(points):
    items = "".join(f"<li>{point}</li>" for point in points)
    st.markdown(f'<div class="takeaway"><div class="takeaway-label">What this tells us</div><ul>{items}</ul></div>', unsafe_allow_html=True)


def data_page(meta):
    hero(f"The story in {meta['dataset_rows']} distinct records.",
         "We looked at real patient records from four hospitals to find out which signs show up most often in people with heart disease. Here is what we found, in plain language.")
    frame = load_data()
    frame["has_disease"] = (frame["num"] > 0).astype(int)
    overall = frame["has_disease"].mean()

    a, b, c = st.columns(3)
    a.metric("Distinct records analyzed", f"{len(frame):,}")
    b.metric("Had heart disease", f"{int(frame['has_disease'].sum()):,} ({overall:.0%})")
    c.metric("Hospitals", f"{frame['dataset'].nunique()}")
    st.markdown(f'<p class="section-intro">More than half of these records have a positive disease label. In every chart below, keep the dataset average of <strong>{overall:.0%}</strong> in mind. Group differences describe this dataset and do not establish individual risk.</p>', unsafe_allow_html=True)

    # Chapter 1: age
    chapter(1, "Older patients are more likely to have heart disease",
            "Patients are grouped by age. Each bar shows how many people in that age group had heart disease.")
    frame["Age group"] = pd.cut(frame["age"], [0, 40, 50, 60, 70, 120], right=False,
                                labels=["Under 40", "40 to 49", "50 to 59", "60 to 69", "70 and over"])
    age = disease_rates(frame, "Age group", ["Under 40", "40 to 49", "50 to 59", "60 to 69", "70 and over"])
    st.plotly_chart(rate_chart(age, "Age group", {"60 to 69", "70 and over"}), width="stretch")
    age_lookup = age.set_index("Age group")["rate"]
    takeaway([
        f"Only about <strong>1 in 3</strong> patients under 40 had heart disease ({age_lookup['Under 40']:.0%}). Among patients in their 60s, it was nearly <strong>3 in 4</strong> ({age_lookup['60 to 69']:.0%}).",
        "This is an association in the dataset; age alone is not enough to classify an individual.",
    ])

    # Chapter 2: sex
    chapter(2, "Men in this group were more than twice as likely to have heart disease",
            "The same comparison for men and women.")
    sex = disease_rates(frame, "sex", ["Female", "Male"])
    sex["sex"] = sex["sex"].map({"Female": "Women", "Male": "Men"})
    st.plotly_chart(rate_chart(sex, "sex", {"Men"}, height=290), width="stretch")
    sex_lookup = sex.set_index("sex")["rate"]
    takeaway([
        f"<strong>{sex_lookup['Men']:.0%}</strong> of men had heart disease, compared with <strong>{sex_lookup['Women']:.0%}</strong> of women.",
        f"Most of the patients were men ({(frame['sex'] == 'Male').mean():.0%}), so the results describe men more reliably than women.",
    ])

    # Chapter 3: chest pain
    chapter(3, "The asymptomatic category had the highest disease rate",
            "Patients described the kind of chest pain they had, if any. You might expect people with the worst chest pain to be the sickest. The data shows the opposite.")
    frame["Chest pain"] = frame["cp"].map(CHEST_PAIN_LABELS)
    cp = disease_rates(frame, "Chest pain", list(CHEST_PAIN_LABELS.values()))
    st.plotly_chart(rate_chart(cp, "Chest pain", {CHEST_PAIN_LABELS["asymptomatic"]}, height=300, horizontal=True), width="stretch")
    cp_lookup = cp.set_index("Chest pain")["rate"]
    takeaway([
        f"<strong>{cp_lookup[CHEST_PAIN_LABELS['asymptomatic']]:.0%}</strong> of patients with no chest pain had heart disease, the highest of any group. Only <strong>{cp_lookup[CHEST_PAIN_LABELS['atypical angina']]:.0%}</strong> of patients with unusual chest pain did.",
        "The source category describes how chest pain was recorded. The group comparison does not establish a screening rule.",
    ])

    # Chapter 4: max heart rate
    chapter(4, "Lower achieved heart rate aligned with more positive labels",
            "The dataset records the maximum heart rate achieved during an exercise test.")
    frame["Highest heart rate"] = pd.cut(frame["thalch"], [0, 120, 140, 160, 300], right=False,
                                         labels=["Below 120", "120 to 139", "140 to 159", "160 or more"])
    hr = disease_rates(frame, "Highest heart rate", ["Below 120", "120 to 139", "140 to 159", "160 or more"])
    fig = rate_chart(hr, "Highest heart rate", {"Below 120"})
    fig.update_xaxes(title="Highest heart rate reached during exercise (beats per minute)")
    st.plotly_chart(fig, width="stretch")
    hr_lookup = hr.set_index("Highest heart rate")["rate"]
    takeaway([
        f"Of patients whose heart rate stayed <strong>below 120</strong>, {hr_lookup['Below 120']:.0%} had heart disease. Of those who reached <strong>160 or more</strong>, only {hr_lookup['160 or more']:.0%} did.",
        "This comparison does not account for age, medication, or differences in testing conditions.",
    ])

    # Chapter 5: exercise angina
    chapter(5, "Chest pain during exercise is a strong red flag",
            "This shows whether patients felt chest pain while exercising during the test.")
    ex = frame.dropna(subset=["exang"]).copy()
    ex["Pain during exercise"] = ex["exang"].map({True: "Yes", False: "No"})
    ex_rates = disease_rates(ex, "Pain during exercise", ["No", "Yes"])
    st.plotly_chart(rate_chart(ex_rates, "Pain during exercise", {"Yes"}, height=290), width="stretch")
    ex_lookup = ex_rates.set_index("Pain during exercise")["rate"]
    takeaway([
        f"About <strong>{ex_lookup['Yes']:.0%}</strong> of patients who felt pain while exercising had heart disease, compared with <strong>{ex_lookup['No']:.0%}</strong> of those who did not.",
        "This is an association in the source records and does not establish what caused the symptom.",
    ])

    # Chapter 6: ST depression
    chapter(6, "Larger recorded ST depression aligned with more positive labels",
            "An ECG records the heart's electrical activity. During exercise, a dip in part of the ECG line (doctors call it ST depression) can mean the heart muscle is short of oxygen.")
    frame["ECG dip"] = pd.cut(frame["oldpeak"], [-10, 0.001, 1, 2, 20],
                              labels=["No dip", "Small (up to 1)", "Medium (1 to 2)", "Large (over 2)"])
    st_rates = disease_rates(frame, "ECG dip", ["No dip", "Small (up to 1)", "Medium (1 to 2)", "Large (over 2)"])
    fig = rate_chart(st_rates, "ECG dip", {"Large (over 2)"})
    fig.update_xaxes(title="Size of the dip on the ECG during exercise")
    st.plotly_chart(fig, width="stretch")
    st_lookup = st_rates.set_index("ECG dip")["rate"]
    takeaway([
        f"The observed share rises across these groups. Of patients with <strong>no dip</strong>, {st_lookup['No dip']:.0%} had heart disease. Of those with a <strong>large dip</strong>, {st_lookup['Large (over 2)']:.0%} did.",
        "The chart describes one variable at a time; the model evaluates inputs together.",
    ])

    # Chapter 7: what did not differ
    clean = frame.copy()
    clean[["trestbps", "chol"]] = clean[["trestbps", "chol"]].replace(0, np.nan)
    medians = clean.groupby("has_disease")[["trestbps", "chol"]].median()
    chapter(7, "Blood pressure and cholesterol were less helpful than expected",
            "These are the numbers most people know from regular check-ups. Here is the typical value for each group.")
    left, right = st.columns(2, gap="large")
    left.metric("Typical resting blood pressure", f"{medians.loc[1, 'trestbps']:.0f} vs {medians.loc[0, 'trestbps']:.0f}",
                help="Median value, mm Hg. Patients with heart disease vs without.")
    right.metric("Typical cholesterol", f"{medians.loc[1, 'chol']:.0f} vs {medians.loc[0, 'chol']:.0f}",
                 help="Median value, mg/dL. Patients with heart disease vs without. Missing values excluded.")
    st.caption("First number: patients with heart disease. Second number: patients without.")
    takeaway([
        "Typical blood pressure was <strong>the same</strong> in both groups, and cholesterol was only slightly higher in the heart-disease group.",
        "The group medians overlap; this summary alone does not show whether either measurement adds predictive value in the fitted model.",
    ])

    # Chapter 8: warning signs stack up
    chapter(8, "Put it together: warning signs add up",
            "We counted how many of these four warning signs each patient had: no chest pain (silent type), chest pain during exercise, an ECG dip of 1 or more, and a highest heart rate below 130.")
    sign_frame = frame.dropna(subset=["cp", "exang", "oldpeak", "thalch"]).copy()
    signs = ((sign_frame["cp"] == "asymptomatic").astype(int) + sign_frame["exang"].astype(int)
             + (sign_frame["oldpeak"] >= 1).astype(int) + (sign_frame["thalch"] < 130).astype(int))
    sign_frame["Warning signs"] = signs.map({0: "None", 1: "1 sign", 2: "2 signs", 3: "3 signs", 4: "All 4"})
    sign_rates = disease_rates(sign_frame, "Warning signs", ["None", "1 sign", "2 signs", "3 signs", "All 4"])
    fig = rate_chart(sign_rates, "Warning signs", {"All 4"}, height=340)
    fig.update_xaxes(title="Number of warning signs")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"This comparison uses {len(sign_frame)} records with all four signs observed; incomplete records are excluded.")
    sign_lookup = sign_rates.set_index("Warning signs")["rate"]
    takeaway([
        f"With <strong>no warning signs</strong>, {sign_lookup['None']:.0%} of patients had heart disease. With <strong>all four</strong>, {sign_lookup['All 4']:.0%} did.",
        "The observed disease share rises with this count. These descriptive groups were defined for the chart, not used as a validated risk score.",
    ])

    # Chapter 9: what the model learned
    chapter(9, "Which inputs changed test ROC-AUC most when shuffled",
            "We shuffled each input on the held-out set and measured the mean drop in ROC-AUC across 12 repeats. This is exploratory model reliance, not clinical importance.")
    importance = pd.DataFrame(meta["importance"])
    importance["label"] = importance["feature"].map(FEATURE_LABELS)
    importance = importance.sort_values("mean_auc_drop", ascending=False).head(6)
    top = importance.iloc[0]
    fig = go.Figure(go.Bar(
        x=importance["mean_auc_drop"], y=importance["label"], orientation="h", marker_cornerradius=4,
        marker_color=[COLORS["teal"] if i < 3 else "#a9d3cc" for i in range(len(importance))],
        text=importance["mean_auc_drop"].map(lambda value: f"{value:.3f}"), textposition="outside",
        textfont=dict(color=COLORS["navy"], size=14),
        hovertemplate="<b>%{y}</b><br>Mean ROC-AUC drop: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, importance["mean_auc_drop"].max() * 1.2], title="Mean drop in test ROC-AUC",
                     showgrid=True, gridcolor="#dbe6e4")
    fig.update_yaxes(title="", autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.35)
    st.plotly_chart(figure_style(fig, 330), width="stretch")
    takeaway([
        f"Shuffling <strong>{top['label']}</strong> reduced ROC-AUC by {top['mean_auc_drop']:.3f} on average, the largest observed drop.",
        "Correlated inputs can share predictive information, so these drops do not rank independent clinical effects.",
    ])

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary"><div class="takeaway-label">The story in one paragraph</div><p>Positive labels were more common among older patients and men in these records. Exercise test variables and the asymptomatic chest-pain category showed large differences across groups. In the four-sign descriptive chart, the positive-label share rose from {sign_lookup["None"]:.0%} to {sign_lookup["All 4"]:.0%}. These are associations in a selected historical sample.</p></div>', unsafe_allow_html=True)
    st.caption("These patterns come from patients referred for heart testing, not the general public. They show links in the data, not proven causes. Talk to a doctor about your own health.")


def main():
    style()
    if not MODEL_PATH.exists():
        st.error("The trained model is missing. Run `python train.py` before starting the app.")
        st.stop()
    bundle = load_artifact(MODEL_PATH.stat().st_mtime_ns)
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark"></span>HeartLens</div><div class="sidebar-copy">Heart disease prediction · UCI dataset</div>', unsafe_allow_html=True)
        page = st.radio("Explore", ["Prediction", "Data story", "Model studio"], label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"UCI Heart Disease · {bundle['metadata']['dataset_rows']} distinct records · 4 hospitals")
        st.caption("Educational model. No diagnostic use.")
    if page == "Prediction":
        prediction_page(bundle)
    elif page == "Model studio":
        model_page(bundle["metadata"])
    else:
        data_page(bundle["metadata"])


if __name__ == "__main__":
    main()
