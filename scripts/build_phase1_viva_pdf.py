"""Build the Phase 1 teaching guide from the saved experiment artifacts.

Requires reportlab. Run from any directory with:
    python scripts/build_phase1_viva_pdf.py
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "artifacts/metrics.json").read_text())
EDA = json.loads((ROOT / "artifacts/eda.json").read_text())
OUT = ROOT / "output/pdf/phase1_viva_teaching_guide.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#17253b")
BLUE = colors.HexColor("#1768ac")
TEAL = colors.HexColor("#007f83")
PALE = colors.HexColor("#eaf3f8")
MIST = colors.HexColor("#f4f7fa")
LINE = colors.HexColor("#d7e1e9")
AMBER = colors.HexColor("#fff2d7")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitleX", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverSubX", fontName="Helvetica", fontSize=13, leading=18, textColor=BLUE, spaceAfter=14))
styles.add(ParagraphStyle(name="HeadX", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="SubX", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=4, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", fontName="Helvetica", fontSize=9.1, leading=13.5, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallX", fontName="Helvetica", fontSize=7.8, leading=11.2, textColor=INK, spaceAfter=4))
styles.add(ParagraphStyle(name="TinyX", fontName="Helvetica", fontSize=7.1, leading=10, textColor=INK))
styles.add(ParagraphStyle(name="BulletX", parent=styles["BodyX"], leftIndent=15, firstLineIndent=-9, spaceAfter=3))
styles.add(ParagraphStyle(name="BoxX", parent=styles["BodyX"], spaceAfter=0))
styles.add(ParagraphStyle(name="CenterX", parent=styles["BodyX"], alignment=TA_CENTER))


def p(s: str, style: str = "BodyX") -> Paragraph:
    return Paragraph(s, styles[style])


def h(s: str) -> Paragraph:
    return p(s, "HeadX")


def sub(s: str) -> Paragraph:
    return p(s, "SubX")


def bullet(s: str) -> Paragraph:
    return p("&#8226; " + s, "BulletX")


def box(s: str, bg=PALE):
    t = Table([[p(s, "BoxX")]], colWidths=[507], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def table(headers: list[str], rows: list[list[str]], widths: list[int], size="SmallX"):
    data = [[p(escape(x), "TinyX") for x in headers]]
    data += [[p(str(x), size) for x in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, MIST]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def qa(q: str, a: str):
    return KeepTogether([p("<b>Q. " + q + "</b>", "SmallX"), p("A. " + a, "SmallX"), Spacer(1, 3)])


def percent(x: float) -> str:
    return f"{100*x:.1f}%"


def footer(canvas, doc):
    canvas.saveState()
    w, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(44, 35, w - 44, 35)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEAL)
    canvas.drawString(44, 23, "HeartLens | Phase 1 ML viva teaching guide")
    canvas.setFillColor(INK)
    canvas.drawRightString(w - 44, 23, str(doc.page))
    canvas.restoreState()


story = []
add = story.append
add(Spacer(1, 30))
add(p("HeartLens", "CoverTitleX"))
add(p("Phase 1 Machine Learning: learn the project, explain the code, answer the viva", "CoverSubX"))
add(HRFlowable(width="100%", thickness=2, color=TEAL))
add(Spacer(1, 17))
add(box("<b>Start here.</b> This guide teaches the complete Phase 1 workflow in simple English with viva-ready reminders. Every number comes from the saved EDA and training artifacts. Each student should explain the logic in their own words and run the demo once.", PALE))
add(Spacer(1, 14))
add(table(["At a glance", "Verified project fact"], [
    ["Problem", "Binary classification: is the UCI target num greater than zero?"],
    ["Clean data", f"{EDA['analysis_rows']} records from 4 hospitals, after removing {EDA['duplicate_records_removed']} duplicate records"],
    ["Inputs", f"{len(EDA['features'])} patient variables"],
    ["Model comparison", "Logistic regression, random forest, histogram gradient boosting, XGBoost"],
    ["Selected for app", "Histogram gradient boosting, by highest mean cross-validation F1"],
    ["Final test", f"{METRICS['test_rows']} untouched records; F1 {METRICS['test']['f1']:.3f}; recall {METRICS['test']['recall']:.3f}; ROC-AUC {METRICS['test']['roc_auc']:.3f}"],
], [135, 372]))
add(Spacer(1, 14))
add(p("The 90-second opening answer", "SubX"))
add(box("We built a Phase 1 machine learning project to classify whether heart disease is present in a historical UCI dataset. After removing two duplicate records, we used 918 records from four hospitals and 11 predictors. We audited missing data, prepared numeric and categorical variables in fold-safe pipelines, and compared four ML model families with five-fold cross-validation on 734 training records. Histogram gradient boosting had the highest mean F1, but its lead over random forest was tiny. We chose a 0.47 cutoff using training data only. On 184 untouched test records, F1 was 0.836, recall 0.873, and ROC-AUC 0.901. Hospital-to-hospital results differed, so the Streamlit output is an educational demo, not a diagnosis."))
add(Spacer(1, 12))
add(p("Live demo: <link href='https://mk-sem5-project.streamlit.app/' color='#1768ac'>mk-sem5-project.streamlit.app</link> &nbsp; | &nbsp; Source: <link href='https://github.com/aditya-1509/mk_sem5_project' color='#1768ac'>GitHub repository</link>", "SmallX"))

add(PageBreak())
add(h("1. Problem, target, and source"))
add(p("<b>Research question:</b> From recorded patient variables, can a model classify the UCI Heart Disease label as absence or presence of disease? The source target <b>num</b> originally ranges from 0 to 4. We set y=0 when num=0 and y=1 when num&gt;0. This loses severity detail; we make a presence/absence classifier only."))
add(p("The supplied CSV contains 920 rows. Two records repeat when the row ID is ignored; both are removed before splitting, leaving 918 unique records. Repeated records across train and test can inflate results. The project uses 4 source hospitals: Cleveland, Hungary, Switzerland, and VA Long Beach."))
add(box("<b>Say in viva:</b> The 0.901 ROC-AUC is performance on our held-out subset of this historical dataset. It is not evidence that the app can diagnose a real patient or estimate clinical risk.", AMBER))
add(sub("Why this is a classification task"))
add(p("The output class has two values. The trained model first returns a score/probability for class 1. We compare it with a threshold: score at least 0.47 gives the positive class. A different threshold changes false positives and false negatives."))
add(sub("Literature review: what we took from each source"))
add(table(["Source", "Why it matters here"], [
    ["UCI Heart Disease", "Defines the dataset, four sources, variables, and target; supports the binary problem framing."],
    ["Detrano et al. (1989)", "Historical work on transferring a coronary-disease probability algorithm across sites; motivates our hospital shift check. Our cohort/outcome are not directly comparable."],
    ["Breiman (2001)", "Random forests combine randomized decision trees; motivates a nonlinear bagging baseline."],
    ["Chen &amp; Guestrin (2016)", "XGBoost describes scalable boosted trees; motivates one of our four model families."],
    ["TRIPOD+AI (2024)", "Reporting guidance motivates clear methods, validation, limitations, and missing-data description."],
], [133, 374]))

add(h("2. EDA: what we examined and why"))
add(p("EDA means exploratory data analysis: inspect rows, labels, missingness, ranges, duplicates, and hospital differences before choosing models. The analysis is reproducible in <b>eda.py</b>; its output is <b>artifacts/eda.json</b>."))
add(table(["Check", "Finding", "Why it matters"], [
    ["Class balance", f"508 positive / 410 negative ({percent(508/918)} positive)", "Accuracy alone could hide missed positive records; use recall and F1 too."],
    ["Age", "Median 54 years; range 28-77", "Shows source cohort span; do not extrapolate to children."],
    ["Duplicates", "2 repeated records without ID", "Remove before split to reduce train-test contamination."],
    ["Hospital labels", "Positive share from 36.3% to 93.5%", "Large shift warns that pooled random split can look optimistic for a new hospital."],
], [100, 174, 233]))
add(sub("Hospital profile"))
add(table(["Hospital", "Records", "Positive", "Positive share"], [
    [x["site"], str(x["rows"]), str(x["positive"]), percent(x["positive_rate"])] for x in EDA["site_summary"]
], [178, 100, 100, 129]))
add(p("<b>Interpretation:</b> Switzerland has 115 positive labels among 123 records, leaving only 8 negative records. This is a selected source population, not a statement that 93.5% of people there have disease."))

add(h("3. Data quality and 11 features"))
add(p("Zero resting blood pressure and zero cholesterol are treated as missing because such readings are implausible as recorded measurements. Then we count missingness. These deterministic value replacements happen before splitting; learned imputation values are fitted only on training folds."))
add(table(["Variable", "Plain meaning / type", "Missing of 918"], [
    ["age", "Age in years; numeric", "0"],
    ["trestbps", "Resting blood pressure; numeric", "60"],
    ["chol", "Serum cholesterol; numeric", "201"],
    ["thalch", "Maximum heart rate achieved; numeric", "55"],
    ["oldpeak", "ST depression during exercise; numeric", "62"],
    ["sex", "Sex as coded in source; category", "0"],
    ["cp", "Chest pain type; category", "0"],
    ["fbs", "Fasting blood sugar &gt;120 mg/dL; category", "90"],
    ["restecg", "Resting ECG category", "2"],
    ["exang", "Exercise-induced angina; category", "55"],
    ["slope", "Exercise ST-segment slope; category", "307"],
], [105, 295, 107]))
add(sub("Excluded columns"))
add(p("<b>id</b> is a row identifier. <b>dataset</b> is hospital identity, reserved for transport testing. <b>ca</b> has 609/918 missing values and <b>thal</b> has 484/918 missing values, with site-dependent gaps. Excluding them gives a more consistent feature contract across hospitals; it may also discard predictive information."))
add(box("<b>Viva line:</b> Missingness is part of the data quality story. A feature's high missing count is not proof it has no clinical relevance.", AMBER))

add(h("4. Feature engineering, step by step"))
add(table(["Step", "What the code does", "Why"], [
    ["Target", "num &gt; 0 becomes label 1", "Defines binary classification."],
    ["Cleaning", "Impossible zero pressure/cholesterol becomes missing", "Avoids treating invalid readings as real low values."],
    ["Numeric", "Training-fold median imputation + missingness indicators + standard scaling", "Makes numeric inputs usable and preserves a flag for missing measurements; scaling helps logistic regression."],
    ["Categorical", "Training-fold most-frequent imputation + one-hot encoding", "Avoids arbitrary numeric order; handles unseen categories at inference."],
    ["Pipeline", "Preprocessing and estimator fit together inside CV", "Prevents validation/test information from determining medians, modes, or scaler parameters."],
], [88, 249, 170]))
add(p("A <b>missingness indicator</b> is a 0/1 flag saying whether a numeric input was absent. This can capture informative missingness, but in this multi-hospital dataset it may also learn site-specific measurement habits. That is one reason for the separate hospital check."))
add(p("<b>One-hot example:</b> chest pain type is a category, so the encoder creates separate category columns instead of pretending that codes 1, 2, 3, 4 are equally spaced medical quantities. <b>Scaling:</b> the numeric value x is transformed using training mean and standard deviation. Tree-based models usually do not require scaling, but a shared preprocessing design supports a fairer comparison and the linear baseline."))
add(sub("What data leakage would look like"))
add(p("If the median cholesterol were computed using all 918 records before cross-validation, the validation fold would influence preprocessing. If duplicate records appeared across the train and test sides, the model could see a near-identical answer during training. The project avoids both cases with fold-fitted pipelines and deduplication before splitting."))

add(h("5. Train/validation/test design"))
add(table(["Layer", "Count / method", "Role"], [
    ["Train split", f"{METRICS['train_rows']} records, 406 positive", "Model fitting, 5-fold CV, hyperparameter search, threshold choice."],
    ["Test split", f"{METRICS['test_rows']} records, 102 positive", "One held-out final estimate after selection."],
    ["Cross-validation", "5 stratified folds within train", "Each candidate is trained on 4 folds and checked on 1, rotating five times."],
    ["Random seed", "42", "Makes split and experiment repeatable."],
], [112, 157, 238]))
add(p("<b>Stratification</b> keeps positive/negative proportions similar when splitting. <b>Hyperparameters</b> are settings chosen by a search (such as tree depth); model parameters are the values learned from data. The primary comparison rule was highest mean cross-validation F1, with ROC-AUC only as a tie-breaker."))
add(p("After choosing the model family/configuration, out-of-fold predictions on training records were used to search cutoffs 0.30 to 0.70 in steps of 0.01. The highest training F1 occurred at 0.47. The test labels did not choose the threshold. Because model selection and threshold tuning both use the same training distribution, those training-stage estimates can still be optimistic; the held-out test is the principal internal check."))
add(box("<b>Draw this in viva:</b> 918 clean rows -&gt; stratified 734 train / 184 test -&gt; five-fold model search within 734 -&gt; threshold from train out-of-fold scores -&gt; final test once on 184.", PALE))

add(h("6. The four models: basic intuition"))
add(table(["Model", "How it thinks", "Strength / limitation here"], [
    ["Logistic regression", "Weighted sum of encoded features, passed through a logistic curve.", "Simple, relatively interpretable baseline; a straight-line decision rule may miss complex interactions."],
    ["Random forest", "Many decision trees trained on resampled data vote/average.", "Handles nonlinear patterns; larger model and less directly interpretable than logistic regression."],
    ["Histogram gradient boosting", "Builds trees sequentially to correct earlier errors, with values binned into histograms.", "Can capture interactions efficiently; can overfit small data if not tuned."],
    ["XGBoost", "Regularized sequential tree boosting with its own implementation and controls.", "Strong tabular benchmark; greater tuning complexity and no win under our selected F1 criterion."],
], [117, 181, 209]))
add(p("<b>Why no deep learning?</b> Phase 1 assesses machine learning only. This is also a small structured tabular dataset; a neural network would require a separate, justified experiment rather than being added only for appearance. The user-facing app deploys one selected model, but four model families were actually compared."))

add(h("7. Model comparison and honest selection"))
add(table(["Tuned family", "CV F1 mean (SD)", "CV ROC-AUC mean (SD)"], [
    [c["model"], f"{c['cv_f1']:.4f} ({c['cv_f1_std']:.4f})", f"{c['cv_auc']:.4f} ({c['cv_auc_std']:.4f})"] for c in METRICS["comparison"]
], [191, 150, 166]))
winner = next(c for c in METRICS["comparison"] if c["model"] == METRICS["selected_model"])
forest = next(c for c in METRICS["comparison"] if c["model"] == "Random forest")
add(p(f"The histogram booster has the largest mean F1 ({winner['cv_f1']:.4f}); random forest has {forest['cv_f1']:.4f}. Their difference is only <b>{winner['cv_f1']-forest['cv_f1']:.4f}</b>, far smaller than their fold-to-fold SDs (about 0.029 and 0.032). Thus the chosen model is the result of a predeclared rule, not convincing proof that boosting is generally better. Logistic regression has the highest mean CV ROC-AUC ({METRICS['comparison'][0]['cv_auc']:.3f}); a different priority could choose a different model."))
add(p("The deployed booster uses <b>learning rate 0.03, max leaf nodes 15, and L2 regularization 0</b> from the search. The deployed positive-class cutoff is <b>0.47</b>. We did not claim that untested models such as SVM or k-nearest neighbors performed worse; they were outside this focused four-family comparison."))
add(box("<b>Memorize:</b> Four models tried. Histogram gradient boosting deployed because its mean CV F1 was highest. Its edge over random forest is practically tiny, so we discuss uncertainty honestly.", AMBER))

add(h("8. Held-out results and metric arithmetic"))
tn, fp = METRICS["test"]["confusion_matrix"][0]
fn, tp = METRICS["test"]["confusion_matrix"][1]
add(table(["Actual \\ Predicted", "Predicted 0", "Predicted 1"], [
    ["Actual 0", f"TN = {tn}", f"FP = {fp}"],
    ["Actual 1", f"FN = {fn}", f"TP = {tp}"],
], [169, 169, 169]))
add(p(f"Of {tn+fp} actual negatives, {tn} were correctly negative and {fp} incorrectly flagged. Of {tp+fn} actual positives, {tp} were found and {fn} were missed. <b>False negatives are a real limitation</b>; this app must not be used for medical decisions."))
add(table(["Metric", "Formula", "Value / explanation"], [
    ["Accuracy", "(TP + TN) / all", f"({tp}+{tn})/{METRICS['test_rows']} = {METRICS['test']['accuracy']:.3f}"],
    ["Precision", "TP / (TP + FP)", f"{tp}/({tp}+{fp}) = {METRICS['test']['precision']:.3f}"],
    ["Recall / sensitivity", "TP / (TP + FN)", f"{tp}/({tp}+{fn}) = {METRICS['test']['recall']:.3f}"],
    ["F1", "2PR / (P + R)", f"{METRICS['test']['f1']:.3f}; balances precision and recall"],
    ["ROC-AUC", "Area under TPR vs FPR curve", f"{METRICS['test']['roc_auc']:.3f}; ranking across all cutoffs"],
    ["Average precision", "Summary of precision-recall curve", f"{METRICS['test']['average_precision']:.3f}"],
    ["Brier score", "Mean (predicted probability - label)^2", f"{METRICS['test']['brier']:.3f}; lower is better, not proof of calibration"],
], [116, 172, 219]))
add(p(f"A majority-positive predictor would get F1 about {2*METRICS['test_positive']/(2*METRICS['test_positive']+(METRICS['test_rows']-METRICS['test_positive'])):.3f} on this test composition. The trained model improves on that simple benchmark, while missing {fn} positive records. F1 depends on the 0.47 cutoff; ROC-AUC does not depend on one cutoff."))

add(h("9. Hospital shift and feature importance"))
add(p("A random 80/20 split mixes hospitals into both partitions. To probe transfer, we separately train/tune on three hospitals and hold the fourth out. These are <b>four separate models</b>, evaluated with a fixed 0.50 cutoff; do not confuse their numbers with the deployed random-split model."))
add(table(["Held-out hospital", "N", "F1", "Recall", "ROC-AUC"], [
    [s["site"], str(s["n"]), f"{s['f1']:.3f}", f"{s['recall']:.3f}", f"{s['roc_auc']:.3f}"] for s in METRICS["site_checks"]
], [171, 63, 82, 91, 100]))
add(p("ROC-AUC ranges from 0.702 (VA Long Beach) to 0.864 (Hungary). The variation warns about transport to another hospital. Switzerland's 123 records include only 8 negative labels, so its AUC is especially unstable. These results do not constitute prospective external clinical validation."))
add(sub("Permutation importance: how to explain the chart"))
add(p("For each feature, shuffle its values on the test set and see how much the trained model's ROC-AUC falls, averaged over 12 repeats. A bigger drop suggests greater reliance of this fitted model on the feature in these data."))
add(table(["Feature shuffled", "Mean ROC-AUC drop", "Plain-English interpretation"], [
    [i["feature"], f"{i['mean_auc_drop']:.3f}", "Model ranking worsens when this feature's link to rows is broken."] for i in METRICS["importance"][:5]
], [135, 135, 237]))
add(box("<b>Do not claim causality:</b> Importance can change with correlated features and a different population. Because this exploratory calculation reuses the test set, avoid treating it as an independent clinical finding.", AMBER))

add(h("10. Streamlit demo and code map"))
add(table(["App page / file", "What to show or explain"], [
    ["Prediction / app.py", "Enter a complete sample profile, submit, explain score and 0.47 cutoff. Blank optional inputs go through the fitted imputer. Output is educational."],
    ["Data story / eda.py", "Show label counts, missingness, and hospital mix. Say these are descriptive cohort patterns."],
    ["Model studio / train.py", "Show CV comparison, test confusion matrix, ROC, hospital checks. Point to metrics.json for saved numbers."],
    ["modeling.py", "Show feature schema, cleaning, preprocessing pipeline, and the same input contract used by training and app."],
    ["artifacts/heart_model.joblib", "Saved fitted model artifact; load only from this trusted project source."],
], [158, 349]))
add(p("<b>Five-minute demo order:</b> (1) State the question and dataset. (2) Data story: 918 records and missingness. (3) Model studio: four models and tiny F1 lead. (4) Confusion matrix: 13 false negatives. (5) Prediction page: enter a fictional profile and explain that the output is not medical advice. (6) Open GitHub code and the IEEE report."))
add(sub("Reproduce locally"))
add(box("<font name='Courier'>python3.13 -m venv .venv<br/>source .venv/bin/activate<br/>pip install -r requirements.txt<br/>python eda.py<br/>python train.py<br/>python -m unittest discover -s tests -v<br/>streamlit run app.py</font>"))

add(h("11. How to present the 9 slides"))
slide_rows = [
    ["1. Title", "State the task, Phase 1 scope, and 90-second project summary."],
    ["2. Problem and data", "Explain num&gt;0 target, 920 to 918 rows, four sites and 11 inputs."],
    ["3. Data quality", "Explain impossible zeros, missingness and why ca/thal were excluded."],
    ["4. Leakage control", "Draw 734/184 split and fold-fitted imputation, encoding and scaling."],
    ["5. CV F1", "Name all four families, selection rule, and the 0.0004 near-tie."],
    ["6. Test results", "Point to F1/recall/AUC and explain TP89, FN13, FP22, TN60."],
    ["7. Hospital check", "Explain separate leave-one-site-out models and performance variation."],
    ["8. Interpretation", "Discuss permutation importance, correlation and clinical limitations."],
    ["9. Reproduction", "Show code, app, IEEE report, and which command makes each artifact."],
]
add(table(["Slide", "Speaker notes in your own words"], slide_rows, [126, 381]))
add(p("Each person should practise the entire nine-slide story. A member can own a section of work, but the viva is individual, so everyone must understand all methods and results."))

add(h("12. Viva questions: fundamentals and choices"))
for q, a in [
    ("What does the model predict?", "The binary label num&gt;0 in the historical UCI data, not future heart attacks or a clinician's diagnosis."),
    ("Why convert target values 1-4 into one class?", "The Phase 1 question is disease presence/absence. The cost is losing severity information."),
    ("Why remove duplicates before splitting?", "The same record on both sides could make the test estimate artificially high."),
    ("What is EDA and what did it reveal?", "It audits data before modeling: 918 unique records, 508 positive labels, missingness, and large differences between hospitals."),
    ("Why not use ca and thal?", "609/918 and 484/918 missing, respectively, with site-dependent gaps; excluding them keeps a more consistent feature set. It is a design tradeoff."),
    ("Why not use dataset/hospital as an input?", "Hospital identity is reserved for a shift check; including it could let the model exploit site-specific label prevalence."),
    ("Why use a pipeline?", "It fits imputation, encoding, scaling and model together inside each training fold, reducing leakage."),
    ("What is one-hot encoding?", "It makes separate indicator columns for category levels, without inventing an ordinal distance."),
    ("What is stratified CV?", "Each fold retains a similar class balance so validation is less distorted by label imbalance."),
    ("Why compare four models?", "Linear, bagged-tree, and two boosted-tree families test different assumptions on the same split and primary metric."),
    ("Why deploy histogram gradient boosting?", "It ranked first by the predeclared mean CV F1 rule, 0.8376. Its lead over random forest is only 0.0004."),
    ("Why not claim it is definitely better than random forest?", "The 0.0004 lead is tiny compared with fold SD around 0.03; this dataset does not establish a meaningful difference."),
    ("Why not choose logistic regression?", "Logistic regression had higher CV AUC, but our primary selection rule was F1. If ranking or interpretability were prioritized, it could be preferable."),
    ("Why not SVM/KNN or deep learning?", "They were not in our tested search. We cannot claim they underperformed. Deep learning belongs to Phase 2, and the dataset is small for an unjustified neural model."),
]:
    add(qa(q, a))

add(h("13. Viva questions: evaluation and limitations"))
for q, a in [
    ("What is the difference between precision and recall?", "Precision asks how many predicted positives were truly positive: 89/111=0.802. Recall asks how many actual positives we found: 89/102=0.873."),
    ("Why use F1?", "It balances precision and recall in one score. We still show the confusion matrix because one score hides error counts."),
    ("What is a false negative?", "An actual positive classified negative; there were 13 in the test set."),
    ("Why report ROC-AUC?", "It evaluates positive/negative ranking across thresholds, while F1 uses a single 0.47 cutoff."),
    ("What does AUC 0.901 mean?", "On this held-out sample the model generally ranks positives above negatives well; it does not mean 90.1% diagnostic certainty."),
    ("Why is the cutoff 0.47?", "It maximized out-of-fold training F1 among cutoffs 0.30-0.70 by 0.01. Test labels did not choose it."),
    ("Why not use 0.50?", "0.50 is only a default. The F1-based training search chose 0.47 for this data; clinical use would require a clinical cost analysis."),
    ("What is Brier score?", "Mean squared error between predicted score and binary outcome. Our 0.127 is a probability-error summary, not a calibration certificate."),
    ("What does the hospital holdout test show?", "Separate training on three sites and testing on the fourth gives AUC 0.702-0.864, suggesting site shift."),
    ("Is hospital holdout a true external validation?", "It is a stronger internal transport check within the same historical compiled dataset, not an independent prospective external cohort."),
    ("What is permutation importance?", "Repeatedly shuffle a feature on test rows and see the drop in ROC-AUC. It measures model reliance, not a causal effect."),
    ("Can this app make a medical recommendation?", "No. It lacks prospective validation, clinical calibration, utility testing, and evidence for individual decisions."),
    ("What would you improve next?", "Acquire an independent recent cohort; evaluate calibration, subgroup errors and decision thresholds with clinicians; compare site-aware validation and missing-data strategies."),
    ("What are the largest limitations?", "Small historical dataset, four selected hospitals, missingness and site shift, modest test size, and uncertainty in the near-tied model ranking."),
]:
    add(qa(q, a))

add(h("14. Last-minute revision card"))
add(table(["Remember", "Exact answer"], [
    ["Rows / target", "920 source - 2 duplicates = 918; 410 negative / 508 positive; num&gt;0 is positive."],
    ["Split", "734 train / 184 test, stratified, seed 42; five-fold CV within train."],
    ["Models", "Logistic regression, random forest, histogram gradient boosting, XGBoost."],
    ["Winner caveat", "Histogram booster CV F1 0.8376 vs forest 0.8372; difference 0.0004."],
    ["Threshold", "0.47, selected from training out-of-fold scores."],
    ["Test", "TN60, FP22, FN13, TP89; precision .802, recall .873, F1 .836, AUC .901."],
    ["Site warning", "Hospital holdout AUC .702-.864; deployment is educational only."],
], [118, 389]))
add(Spacer(1, 10))
add(box("<b>Practice rule:</b> Close this PDF and explain each table without reading. Open the app and demonstrate once. If an examiner asks something untested, say clearly that it was not evaluated rather than inventing a result.", PALE))
add(sub("Five-person contribution record to complete truthfully"))
add(table(["Student name", "Actual work", "Evidence (commit/file/experiment)", "Viva topic practised"], [["", "", "", ""] for _ in range(5)], [115, 127, 154, 111]))
add(p("The team must replace the blank cells with its five real members and actual contributions before submitting. The current materials intentionally use a generic team author line because names were not supplied."))
add(sub("Submission checklist"))
for s in [
    "Literature review and theory: IEEE report Section I, source references, this guide.",
    "Dataset quality and EDA: eda.py, artifacts/eda.json, Streamlit Data story.",
    "Feature engineering: modeling.py and IEEE report methods.",
    "Model application/validation: train.py, artifacts/metrics.json, Streamlit Model studio.",
    "Code quality: public GitHub repository, requirements.txt, tests, reproduction commands.",
    "Presentation: nine-slide Phase 1 PPTX. Report: compiled IEEE PDF plus .tex source.",
    "Individual viva: every member practises the demo and the Q&A above.",
]:
    add(bullet(s))

add(h("Sources and project links"))
for label, url in [
    ("UCI Heart Disease", "https://archive.ics.uci.edu/dataset/45/heart+disease"),
    ("Detrano et al., American Journal of Cardiology (1989)", "https://doi.org/10.1016/0002-9149(89)90524-9"),
    ("Breiman, Random Forests (2001)", "https://doi.org/10.1023/A:1010933404324"),
    ("Chen and Guestrin, XGBoost (2016)", "https://doi.org/10.1145/2939672.2939785"),
    ("TRIPOD+AI, BMJ (2024)", "https://doi.org/10.1136/bmj-2023-078378"),
    ("Project GitHub", "https://github.com/aditya-1509/mk_sem5_project"),
    ("Live Streamlit app", "https://mk-sem5-project.streamlit.app/"),
]:
    add(p(f"<link href='{url}' color='#1768ac'>{escape(label)}</link>", "SmallX"))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=48, bottomMargin=50, title="HeartLens Phase 1 Viva Teaching Guide", author="CSAI Project Team", subject="Phase 1 machine learning project teaching and viva preparation")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
