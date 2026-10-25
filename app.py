"""Streamlit entry point for the verified Heartlens study companion.

THESIS: Edit a fictional profile and immediately inspect its frozen model response.
OWN-WORLD: Cream paper, terracotta scores, brown ink, serif headings, native controls.
STORY: Try values, compare models, inspect evidence and limitations, download the study.
FIRST VIEWPORT: Compact editorial header above a two-column input/result workbench.
FORM: Warm research notebook, grounded candidate 4, seed ceb51ca6.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
"""
import streamlit as st
from study import ROOT, evidence

st.set_page_config(page_title="Heartlens · A heart disease ML study", page_icon="◒", layout="wide", initial_sidebar_state="collapsed")
st.html((ROOT / "assets/app.css").read_text())
st.html('<div class="wordmark">heartlens<span> / Heart disease · Phase 1</span></div>')
try:
    evidence()
except (ValueError, OSError, KeyError):
    st.error("The verified study artifacts could not be loaded. Please report this to the project team.")
    st.stop()

page = st.navigation([
    st.Page("views/live.py", title="Live prediction", default=True),
    st.Page("views/models.py", title="Model studio", url_path="models"),
    st.Page("views/data.py", title="Data explorer", url_path="data_explorer"),
    st.Page("views/project.py", title="The project", url_path="project"),
], position="top")
page.run()
st.divider()
st.caption("HEARTLENS / PHASE 01 · Academic demonstration. Use fictional inputs. This app does not provide medical advice.")
