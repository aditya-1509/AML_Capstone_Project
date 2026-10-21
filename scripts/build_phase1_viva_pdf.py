"""Build the Phase 1 teaching guide from the saved experiment artifacts.

Requires reportlab. Run from any directory with:
    python scripts/build_phase1_viva_pdf.py
"""

from __future__ import annotations

import json
import re
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
RUN = ROOT / "artifacts/runs/20261020T193544Z_2a8a1491"
METRICS = json.loads((RUN / "pooled/metrics.json").read_text())
COMPARISON = json.loads((RUN / "pooled/comparison.json").read_text())
if json.loads((RUN / "verification.json").read_text())["status"] != "PASS":
    raise ValueError("Use the verified run only")
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
styles.add(ParagraphStyle(name="HeaderX", fontName="Helvetica-Bold", fontSize=7.1, leading=10, textColor=WHITE))
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
    data = [[p(escape(x), "HeaderX") for x in headers]]
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



def inline(text):
    """Render only the simple Markdown already used in the existing viva guide."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<link href='\2' color='#1768ac'>\1</link>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


story = [
    Spacer(1, 22), p("HeartLens", "CoverTitleX"),
    p("Project 15 - Team T008, Section C", "CoverSubX"),
    p("Heart-Disease Risk - Domain-Structured Bayesian Network vs Neural Network", "HeadX"),
    p("Phase 1 teaching guide: four conventional classifiers and a provisional BN. Neural-network implementation belongs to Phase 2.", "BodyX"),
    HRFlowable(width="100%", thickness=2, color=TEAL), Spacer(1, 14),
    box("<b>Verified evidence.</b> Run 20261020T193544Z_2a8a1491. Random Forest remains selected by nested-CV F1. The 184-row evaluation is retrospective because this holdout was already reported and inspected. The BN graph is provisional and not medically validated.", AMBER),
    Spacer(1, 12),
    p("How to use this guide", "SubX"),
    p("Learn the partition diagram and BN factorization, work through the RF confusion matrix, then practise the questions below. Answers follow VIVA.md and the actual implementation. The confirmed team roster is included below; contribution evidence remains for the team to supply."),
    p("Regenerate with python scripts/build_phase1_viva_pdf.py (ReportLab required). The builder reads VIVA.md and verified saved results without training.", "SmallX"),
    PageBreak(),
]
lines = (ROOT / "VIVA.md").read_text().splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line or line.startswith("# "):
        i += 1
        continue
    if line.startswith("## "):
        if line.startswith("## Team record"):
            story.append(PageBreak())
        story.append(h(line[3:]))
        i += 1
        continue
    if line.startswith("| "):
        block = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
            i += 1
        headers, rows = block[0], block[2:]
        headers = [{"Nested CV F1": "CV F1", "ROC-AUC": "AUC", "Evaluation N": "Eval N"}.get(x, x) for x in headers]
        if len(headers) == 9:
            widths = [137] + [46.25] * 8
            rows = [
                [name, f"{next(c['cv_f1'] for c in COMPARISON if c['model'] == name):.4f}",
                 f"{family['threshold']:.2f}"]
                + [f"{family['tuned'][key]:.4f}" for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")]
                for name, family in METRICS["families"].items()
            ]
        elif len(headers) == 8:
            widths = [90, 30, 30, 55, 110, 45, 65, 82]
            rows = []
            for folder in sorted(RUN.glob("hospital_*")):
                metrics = json.loads((folder / "metrics.json").read_text())
                plan = json.loads((folder / "partitions.json").read_text())
                name = metrics["selected_model"]
                family = metrics["families"][name]
                rows.append([folder.name.removeprefix("hospital_").replace("_", " "),
                             str(len(plan["selection_source_ids"])), str(len(plan["cutoff_source_ids"])),
                             str(metrics["n"]), name, f"{family['threshold']:.2f}",
                             f"{family['tuned']['f1']:.4f}", f"{family['tuned']['roc_auc']:.4f}"])
        else:
            widths = [507 / len(headers)] * len(headers)
        story.append(table(headers, [[inline(cell) for cell in row] for row in rows], widths))
        story.append(Spacer(1, 9))
        continue
    if line.startswith("- "):
        story.append(bullet(inline(line[2:])))
        i += 1
        continue
    if re.match(r"^\d+\. ", line) and "**" not in line:
        story.append(p(inline(line)))
        i += 1
        continue
    paragraph = [line]
    i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(("## ", "| ")):
        paragraph.append(lines[i].strip())
        i += 1
    text = " ".join(paragraph)
    # Technical question lines stay with the answer on the next paragraph.
    style = "SubX" if re.match(r"^\d+\. \*\*", text) else "BodyX"
    story.append(p(inline(text), style))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=48, bottomMargin=50,
                        title="Project 15 - T008 Section C - Phase 1 Viva Teaching Guide", author="Team T008, Section C",
                        subject="Verified retrospective disease-presence experiment and provisional BN")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
