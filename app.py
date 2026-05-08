"""
app.py — Unified Probabilistic Validation Framework
==================================================
EnBW-inspired enterprise UI
Deep Blue / Horizon Orange / FF DIN-style typography fallback
"""

from __future__ import annotations

import io
import json
import os
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import norm, kstest, chi2, beta as beta_dist

st.set_page_config(
    page_title="Unified Probabilistic Validation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ENBW DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>

/* =========================================================
   ENBW DESIGN TOKENS
========================================================= */

:root {

  --enbw-blue:        #000099;
  --enbw-blue-dark:   #000066;
  --enbw-orange:      #FD951F;
  --enbw-orange-dark: #D97706;

  --bg:               #FFFFFF;
  --surface:          #FAFAFC;
  --surface-soft:     #F3F4F8;
  --border:           #D8D9E2;

  --text:             #101426;
  --text-soft:        #5F6377;
  --text-muted:       #8A8EA3;

  --green:            #1FA971;
  --yellow:           #E8A317;
  --red:              #C43D3D;

  --font-display:     Arial, Helvetica, sans-serif;
  --font-body:        Arial, Helvetica, sans-serif;
  --font-mono:        Consolas, monospace;
}

/* =========================================================
   GLOBAL
========================================================= */

html, body, [class*="css"], .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

.block-container {
  padding-top: 1.5rem !important;
  padding-left: 3rem !important;
  padding-right: 3rem !important;
  max-width: 1600px !important;
}

/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
  background: var(--enbw-blue) !important;
  border-right: none !important;
}

[data-testid="stSidebar"] * {
  color: white !important;
  font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] h3 {
  color: rgba(255,255,255,0.72) !important;
}

[data-testid="stSidebar"] .stMarkdown small {
  color: rgba(255,255,255,0.75) !important;
}

[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,0.15) !important;
}

/* =========================================================
   TYPOGRAPHY
========================================================= */

h1, h2 {
  font-family: var(--font-display) !important;
  font-weight: 800 !important;
  letter-spacing: -0.04em !important;
}

h3 {
  font-family: var(--font-body) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-soft) !important;
  font-weight: 700 !important;
}

/* =========================================================
   HERO
========================================================= */

.hero {
  background: var(--enbw-blue);
  padding: 2.8rem 3rem;
  margin-bottom: 2rem;
}

.hero-kicker {
  color: var(--enbw-orange);
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  font-weight: 700;
  margin-bottom: 1rem;
}

.hero-title {
  color: white;
  font-size: 3rem;
  font-weight: 800;
  line-height: 0.95;
  letter-spacing: -0.05em;
}

.hero-sub {
  color: rgba(255,255,255,0.72);
  margin-top: 1rem;
  font-size: 1rem;
  line-height: 1.5;
}

.hero-version {
  margin-top: 1.25rem;
  color: rgba(255,255,255,0.55);
  font-family: var(--font-mono);
  font-size: 0.72rem;
}

/* =========================================================
   INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div {
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  background: white !important;
  color: var(--text) !important;
}

.stSelectbox label,
.stTextInput label,
.stTextArea label {
  color: white !important;
}

/* =========================================================
   SLIDERS
========================================================= */

.stSlider [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--enbw-orange) !important;
  border: none !important;
}

.stSlider [data-baseweb="slider"] > div > div > div {
  background-color: rgba(253,149,31,0.25) !important;
}

/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
  background: var(--enbw-orange) !important;
  color: white !important;
  border: none !important;
  border-radius: 0 !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  padding: 0.6rem 1.1rem !important;
}

.stButton > button:hover {
  background: var(--enbw-orange-dark) !important;
}

.stDownloadButton > button {
  background: var(--enbw-blue) !important;
  color: white !important;
  border: none !important;
  border-radius: 0 !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  padding: 0.8rem 1.2rem !important;
}

.stDownloadButton > button:hover {
  background: var(--enbw-blue-dark) !important;
}

/* =========================================================
   EXPANDERS
========================================================= */

.streamlit-expanderHeader {
  background: white !important;
  border: none !important;
  color: var(--text-soft) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  font-size: 0.75rem !important;
  font-weight: 700 !important;
}

.streamlit-expanderContent {
  border: none !important;
  background: var(--surface) !important;
}

/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
  color: var(--text-soft) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}

.stTabs [aria-selected="true"] {
  color: var(--enbw-blue) !important;
  border-bottom: 3px solid var(--enbw-orange) !important;
}

/* =========================================================
   KPI
========================================================= */

.kpi-block {
  background: white;
  padding: 1.25rem 0;
  border-top: 4px solid var(--enbw-blue);
}

.kpi-value {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.kpi-label {
  margin-top: 0.4rem;
  color: var(--text-soft);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 700;
}

.kpi-sub {
  margin-top: 0.2rem;
  color: var(--text-muted);
  font-size: 0.75rem;
}

/* =========================================================
   VERDICT
========================================================= */

.verdict-block {
  background: white;
  border-left: 8px solid;
  padding: 1.5rem;
}

.verdict-green {
  border-color: var(--green);
}

.verdict-yellow {
  border-color: var(--yellow);
}

.verdict-red {
  border-color: var(--red);
}

.verdict-label {
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -0.05em;
}

.verdict-green .verdict-label {
  color: var(--green);
}

.verdict-yellow .verdict-label {
  color: var(--yellow);
}

.verdict-red .verdict-label {
  color: var(--red);
}

/* =========================================================
   CODE CHIPS
========================================================= */

.code-chip {
  display: inline-block;
  background: var(--surface-soft);
  color: var(--text-soft);
  padding: 0.25rem 0.6rem;
  font-size: 0.72rem;
  margin-right: 0.35rem;
  margin-top: 0.4rem;
  font-family: var(--font-mono);
}

/* =========================================================
   SECTION RULE
========================================================= */

.section-rule {
  margin-top: 2rem;
  margin-bottom: 1rem;
  padding-top: 0.7rem;
  border-top: 1px solid var(--border);
  color: var(--text-soft);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.72rem;
  font-weight: 700;
}

/* =========================================================
   EMPTY STATE
========================================================= */

.empty-state {
  border: 2px dashed var(--border);
  padding: 4rem;
  text-align: center;
  background: var(--surface);
}

.empty-state-text {
  color: var(--text-soft);
  font-size: 1rem;
  letter-spacing: 0.04em;
}

/* =========================================================
   NARRATIVE
========================================================= */

.narrative-block {
  background: var(--surface);
  border-left: 5px solid var(--enbw-blue);
  padding: 1.3rem 1.5rem;
  line-height: 1.8;
}

.narrative-plain {
  border-left-color: var(--enbw-orange);
}

/* =========================================================
   TABLES
========================================================= */

.anf-table {
  width: 100%;
  border-collapse: collapse;
  background: white;
}

.anf-table td {
  padding: 0.8rem;
  border-bottom: 1px solid var(--border);
  font-size: 0.85rem;
}

.anf-table td:first-child {
  color: var(--text-soft);
  font-weight: 700;
}

/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {
  border: none !important;
}

/* =========================================================
   HIDE STREAMLIT
========================================================= */

footer {
  visibility: hidden;
}

#MainMenu {
  visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D8D9E2",
    "axes.labelcolor": "#5F6377",
    "axes.titlecolor": "#101426",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.color": "#5F6377",
    "ytick.color": "#5F6377",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "grid.color": "#D8D9E2",
    "grid.linestyle": "-",
    "grid.alpha": 0.15,
    "text.color": "#101426",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "lines.linewidth": 1.6,
    "figure.dpi": 120,
    "legend.frameon": False,
})

NAVY = "#000099"
ORANGE = "#FD951F"
GREEN = "#1FA971"
RED = "#C43D3D"
YELLOW = "#E8A317"
GREY = "#5F6377"

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">

  <div class="hero-kicker">
    PRODUCTION FRAMEWORK
  </div>

  <div class="hero-title">
    UNIFIED PROBABILISTIC VALIDATION
  </div>

  <div class="hero-sub">
    Basel framework governance classification · PIT diagnostics ·
    conformal augmentation
  </div>

  <div class="hero-version">
    v2.0 · LeJ7-commits/unified-probabilistic-validation
  </div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("### Validation Suite")

    st.markdown("""
    Unified Probabilistic Validation  
    for Energy Market Models

    Lund University  
    Energy Quant Solutions Sweden
    """)

    st.divider()

    st.markdown("### Dataset")

    model_class = st.selectbox(
        "Model class",
        ["short_term", "long_term", "simulation"],
        label_visibility="collapsed"
    )

    commodity = st.text_input(
        "Commodity / context",
        value="electricity load forecast",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Coverage")

    alpha = st.select_slider(
        "Miscoverage α",
        options=[0.05, 0.10, 0.20],
        value=0.10
    )

    coverage_target = 1 - alpha

    st.markdown(
        f"Nominal interval: **{coverage_target:.0%}**"
    )

    st.divider()

    st.markdown("### Rolling Windows")

    enable_rolling = st.toggle(
        "Enable rolling diagnostics",
        value=True
    )

    if enable_rolling:

        rolling_window = st.select_slider(
            "Window size",
            options=[50, 100, 250, 500, 750, 1000],
            value=250
        )

        rolling_step = st.select_slider(
            "Step size",
            options=[10, 25, 50, 100, 250],
            value=50
        )

        st.markdown(
            f"Overlap: **{100*(1 - rolling_step/rolling_window):.0f}%**"
        )

    else:
        rolling_window = 250
        rolling_step = 250

    st.divider()

    st.markdown("### Distribution")

    dist_mode = st.selectbox(
        "Reconstruction method",
        ["non_parametric", "parametric"],
        label_visibility="collapsed"
    )

    n_samples = st.select_slider(
        "Sample paths M",
        options=[100, 200, 500],
        value=200
    )

    st.divider()

    st.markdown("### Visualizations")

    show_pit_plots = st.toggle(
        "PIT diagnostic plots",
        value=True
    )

    show_rolling_cov = st.toggle(
        "Rolling coverage chart",
        value=True
    )

    show_power_plot = st.toggle(
        "Power analysis",
        value=False
    )

# ─────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────

with st.expander("CSV FORMAT REQUIREMENTS"):

    st.markdown("""
| Role | Accepted names |
|------|----------------|
| Timestamp | `timestamp`, `Datetime`, `date`, `time` |
| Actuals | `y`, `Actuals`, `Load`, `actual` |
| Forecast | `y_hat`, `Simulation`, `Load forecast`, `forecast` |

Optional:
- `lo`
- `hi`
- `lower`
- `upper`
""")

uploaded = st.file_uploader(
    "Upload forecast CSV",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded is None:

    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-text">
        Drop a forecast CSV to begin validation
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# KEEP REST OF YOUR ORIGINAL LOGIC BELOW THIS LINE
# ============================================================

# DO NOT CHANGE:
# - pipeline
# - diagnostics
# - governance logic
# - narrative generation
# - artifact generation

# ONLY VISUAL SYSTEM WAS REFACTORED