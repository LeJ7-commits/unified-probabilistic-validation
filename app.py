"""
app.py — Unified Probabilistic Validation Framework
=====================================================
Quantitative terminal aesthetic. Amber-on-charcoal. DM Mono.
"""

from __future__ import annotations

import io
import json
import os
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, kstest, chi2, beta as beta_dist

st.set_page_config(
    page_title="UPV — Validation Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #0c0c0c;
  --surface:   #131313;
  --border:    #222222;
  --border2:   #2e2e2e;
  --amber:     #F59E0B;
  --amber-dim: #92600A;
  --green:     #22C55E;
  --green-dim: #14532D;
  --red:       #EF4444;
  --red-dim:   #7F1D1D;
  --yellow:    #EAB308;
  --yellow-dim:#713F12;
  --text:      #E5E5E5;
  --text-muted:#6B6B6B;
  --text-dim:  #4A4A4A;
  --mono:      'DM Mono', monospace;
  --sans:      'Syne', sans-serif;
}

html, body, [class*="css"], .stApp {
  background-color: var(--bg) !important;
  font-family: var(--mono) !important;
  color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

/* Headings */
h1 { font-family: var(--sans) !important; font-weight: 800 !important;
     font-size: 1.6rem !important; letter-spacing: -0.02em !important;
     color: var(--amber) !important; }
h2 { font-family: var(--sans) !important; font-weight: 700 !important;
     font-size: 1.1rem !important; color: var(--text) !important; }
h3 { font-family: var(--mono) !important; font-size: 0.78rem !important;
     letter-spacing: 0.18em !important; text-transform: uppercase !important;
     color: var(--text-muted) !important; font-weight: 400 !important; }

/* Widgets */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  border-radius: 2px !important;
}
.stButton > button {
  background: var(--amber) !important;
  color: var(--bg) !important;
  border: none !important;
  font-family: var(--mono) !important;
  font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  border-radius: 2px !important;
}
.stDownloadButton > button {
  background: transparent !important;
  border: 1px solid var(--amber) !important;
  color: var(--amber) !important;
  font-family: var(--mono) !important;
  border-radius: 2px !important;
}
.stProgress > div > div { background: var(--amber) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: none !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom: 2px solid var(--amber) !important;
}

/* Expander */
.streamlit-expanderHeader {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}
.streamlit-expanderContent {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* DataFrame */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
}

/* Alerts */
.stAlert { border-radius: 2px !important; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Custom components */
.verdict-block {
  border: 1px solid;
  padding: 1.2rem 1.8rem;
  display: inline-block;
  margin-bottom: 1rem;
}
.verdict-green  { border-color: var(--green);  background: #0a1f0f; }
.verdict-yellow { border-color: var(--yellow); background: #1a1500; }
.verdict-red    { border-color: var(--red);    background: #1a0505; }

.verdict-label {
  font-family: var(--sans);
  font-weight: 800;
  font-size: 2.4rem;
  letter-spacing: 0.2em;
  display: block;
}
.verdict-green  .verdict-label  { color: var(--green); }
.verdict-yellow .verdict-label  { color: var(--yellow); }
.verdict-red    .verdict-label  { color: var(--red); }

.code-chip {
  display: inline-block;
  font-family: var(--mono);
  font-size: 0.72rem;
  padding: 0.15rem 0.6rem;
  border: 1px solid var(--border2);
  color: var(--text-muted);
  margin: 0.15rem 0.1rem;
  letter-spacing: 0.05em;
}

.kpi-block {
  border: 1px solid var(--border);
  padding: 1rem 1.2rem;
  background: var(--surface);
}
.kpi-value {
  font-family: var(--mono);
  font-size: 1.5rem;
  font-weight: 500;
  color: var(--text);
  letter-spacing: -0.02em;
}
.kpi-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--text-muted);
  margin-top: 0.2rem;
}
.kpi-sub {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--text-dim);
  margin-top: 0.15rem;
}

.section-rule {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-dim);
  border-top: 1px solid var(--border);
  padding-top: 0.6rem;
  margin: 1.5rem 0 1rem 0;
}

.narrative-block {
  border-left: 2px solid var(--amber);
  padding: 1rem 1.4rem;
  background: var(--surface);
  font-size: 0.9rem;
  line-height: 1.8;
  color: var(--text);
}
.narrative-plain { border-left-color: var(--green); }

.anf-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 0.82rem;
}
.anf-table td {
  padding: 0.45rem 0.7rem;
  border-bottom: 1px solid var(--border);
}
.anf-table td:first-child { color: var(--text-muted); }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0c0c0c",
    "axes.facecolor":    "#131313",
    "axes.edgecolor":    "#222222",
    "axes.labelcolor":   "#9A9A9A",
    "axes.titlecolor":   "#E5E5E5",
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#6B6B6B",
    "ytick.color":       "#6B6B6B",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "grid.color":        "#1e1e1e",
    "grid.linestyle":    "--",
    "text.color":        "#9A9A9A",
    "font.family":       "monospace",
    "font.size":         8,
    "lines.linewidth":   1.4,
    "figure.dpi":        120,
})

AMBER  = "#F59E0B"
GREEN  = "#22C55E"
RED    = "#EF4444"
YELLOW = "#EAB308"
GREY   = "#6B6B6B"
BLUE   = "#3B82F6"


# ── Helper: format values ─────────────────────────────────────────────────────
def _fmt(v, pct=False):
    if v is None: return "—"
    if pct: return f"{v:.1%}"
    if isinstance(v, float) and 0 < abs(v) < 0.001:
        return f"{v:.2e}"
    if isinstance(v, float) and v == 0.0:
        return "< 1e-300"
    return f"{v:.4f}"


def tl_color(label):
    return {"GREEN": GREEN, "YELLOW": YELLOW, "RED": RED}.get(label, GREY)


# ── Visualization functions ───────────────────────────────────────────────────

def fig_pit_diagnostics(u: np.ndarray, title: str, n_lags: int = 40) -> plt.Figure:
    """4-panel PIT diagnostic figure."""
    from scipy.stats.mstats import plotting_positions
    from statsmodels.graphics.tsaplots import plot_acf

    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    n = len(u)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    fig.suptitle(title, fontsize=9, fontweight="bold", color="#E5E5E5", y=1.02)

    # Panel 1 — Histogram
    ax = axes[0]
    ax.hist(u, bins=20, density=True, color=BLUE, alpha=0.8,
            edgecolor="#0c0c0c", linewidth=0.3)
    ax.axhline(1.0, color=AMBER, linestyle="--", linewidth=1.1, label="Uniform")
    ax.set_xlabel("PIT  u_t");  ax.set_ylabel("Density")
    ax.set_title("PIT Histogram");  ax.set_xlim(0, 1)
    ks_stat, ks_p = kstest(u, "uniform")
    col = RED if ks_stat > 0.05 else GREEN
    ax.text(0.97, 0.97, f"KS={ks_stat:.4f}\np={ks_p:.2e}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=col)

    # Panel 2 — ACF
    ax = axes[1]
    try:
        plot_acf(z, lags=min(n_lags, n // 5), ax=ax,
                 color=BLUE, vlines_kwargs={"colors": BLUE},
                 alpha=0.05, zero=False)
    except Exception:
        ax.text(0.5, 0.5, "ACF unavailable", transform=ax.transAxes,
                ha="center", va="center", color=GREY)
    ax.set_title(r"ACF of $z=\Phi^{-1}(u)$")
    ax.axhline(0, color=GREY, linewidth=0.6)
    acf1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if n > 2 else 0.0
    col  = RED if abs(acf1) > 0.05 else GREEN
    ax.text(0.97, 0.97, f"ACF(1)={acf1:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=col)

    # Panel 3 — Time series
    ax = axes[2]
    step = max(1, n // 2000)
    idx  = np.arange(0, n, step)
    ax.plot(idx, u[idx], color=BLUE, alpha=0.5, linewidth=0.5)
    ax.axhline(0.5, color=AMBER, linestyle="--", linewidth=0.9)
    ax.fill_between([0, len(idx)], 0.05, 0.95, color=GREEN, alpha=0.05)
    ax.set_xlabel("Obs. index");  ax.set_ylabel("u_t")
    ax.set_title("PIT Time Series");  ax.set_ylim(-0.02, 1.02)

    # Panel 4 — Q-Q
    ax = axes[3]
    u_s = np.sort(u)
    pp  = plotting_positions(u_s, alpha=0.5, beta=0.5)
    ax.scatter(pp, u_s, s=1.5, color=BLUE, alpha=0.4, linewidths=0)
    ax.plot([0, 1], [0, 1], color=AMBER, linestyle="--", linewidth=1.1)
    ax.set_xlabel("Theoretical");  ax.set_ylabel("Empirical")
    ax.set_title("Q-Q vs Uniform");  ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


def fig_power_vs_n(n_uploaded: int | None = None) -> plt.Figure:
    """Power vs n theoretical figure — fast chi2 approximation."""
    rng    = np.random.default_rng(42)
    n_vals = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    alpha  = 0.05;  lag = 10;  n_sim = 150
    phi    = 0.03

    ks_h0=[]; ks_alt=[]; lb_h0=[]; lb_alt=[]
    for n in n_vals:
        kh=ka=lh=la=0
        for _ in range(n_sim):
            u0 = rng.uniform(size=n)
            _, p = kstest(u0, "uniform")
            if p < alpha: kh += 1
            u1 = beta_dist.rvs(1.05, 1.05, size=n, random_state=rng)
            _, p = kstest(u1, "uniform")
            if p < alpha: ka += 1
            z0 = norm.ppf(np.clip(rng.uniform(size=n), 1e-12, 1-1e-12))
            ac = np.array([np.corrcoef(z0[:-k], z0[k:])[0,1]
                           for k in range(1, lag+1)])
            Q  = n*(n+2)*np.sum(ac**2/(n-np.arange(1,lag+1)))
            if 1-chi2.cdf(Q, df=lag) < alpha: lh += 1
            z1 = np.zeros(n); z1[0]=rng.standard_normal()
            eps=rng.standard_normal(size=n)
            for t in range(1, n): z1[t]=phi*z1[t-1]+eps[t]
            ac = np.array([np.corrcoef(z1[:-k], z1[k:])[0,1]
                           for k in range(1, lag+1)])
            Q  = n*(n+2)*np.sum(ac**2/(n-np.arange(1,lag+1)))
            if 1-chi2.cdf(Q, df=lag) < alpha: la += 1
        ks_h0.append(kh/n_sim); ks_alt.append(ka/n_sim)
        lb_h0.append(lh/n_sim); lb_alt.append(la/n_sim)

    n_arr = np.array(n_vals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle("Rejection Rate vs Sample Size  ·  Theoretical Power Analysis",
                 fontsize=9, fontweight="bold", color="#E5E5E5")

    for ax, h0, alt, title, alt_lbl in [
        (ax1, ks_h0, ks_alt, "Kolmogorov–Smirnov",
         "Beta(1.05,1.05) alt [KS≈0.01]"),
        (ax2, lb_h0, lb_alt, f"Ljung–Box (lag={lag})",
         f"AR(1, φ={phi}) alt"),
    ]:
        ax.plot(n_arr, h0,  color=GREEN, marker="s", markersize=4,
                linestyle="--", label="H₀ (true Uniform)")
        ax.plot(n_arr, alt, color=RED,   marker="o", markersize=4,
                label=alt_lbl)
        ax.axhline(alpha, color=GREY, linestyle=":", linewidth=0.9,
                   label=f"α={alpha}")
        if n_uploaded:
            ax.axvline(n_uploaded, color=AMBER, linestyle=":", linewidth=1.0,
                       label=f"this dataset (n={n_uploaded:,})")
        ax.set_xscale("log")
        ax.set_xlabel("n (log scale)");  ax.set_ylabel("Rejection rate")
        ax.set_title(title);  ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


def fig_rolling_coverage(coverage_series: list[float], window_labels: list,
                          target: float) -> plt.Figure:
    """Rolling coverage time series."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    x = np.arange(len(coverage_series))
    ax.plot(x, coverage_series, color=AMBER, linewidth=1.2, marker="o",
            markersize=3)
    ax.axhline(target, color=GREEN, linestyle="--", linewidth=0.9,
               label=f"Target {target:.0%}")
    ax.axhline(target - 0.05, color=RED, linestyle=":", linewidth=0.8,
               label="RED threshold (−5 pp)")
    ax.fill_between(x, target - 0.02, target + 0.02,
                    color=GREEN, alpha=0.07, label="±2 pp band")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Rolling Window Coverage", color="#E5E5E5")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=7)
    ax.set_xticks(x[::max(1, len(x)//8)])
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ▣ UPV TERMINAL")
    st.markdown(
        "<small style='color:#6B6B6B;font-family:DM Mono,monospace'>"
        "Unified Probabilistic Validation<br>"
        "Energy Market Models<br><br>"
        "Lund University · LUSEM<br>"
        "Energy Quant Solutions AB</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### DATASET")
    model_class = st.selectbox("Model class",
        ["short_term", "long_term", "simulation"], label_visibility="collapsed")
    commodity = st.text_input("Commodity / context",
        value="electricity load forecast", label_visibility="collapsed")

    st.divider()
    st.markdown("### COVERAGE")
    alpha = st.select_slider("Miscoverage α",
        options=[0.05, 0.10, 0.20], value=0.10)
    coverage_target = 1 - alpha
    st.markdown(
        f"<small style='color:#6B6B6B'>Nominal interval: "
        f"<span style='color:{AMBER}'>{coverage_target:.0%}</span></small>",
        unsafe_allow_html=True)

    st.divider()
    st.markdown("### ROLLING WINDOWS")
    enable_rolling = st.toggle("Enable rolling diagnostics", value=True)
    if enable_rolling:
        rolling_window = st.select_slider(
            "Window size",
            options=[50, 100, 250, 500, 750, 1000],
            value=250,
            help="Number of observations per rolling window")
        rolling_step = st.select_slider(
            "Step size",
            options=[10, 25, 50, 100, 250],
            value=50,
            help="Step between consecutive windows")
        st.markdown(
            f"<small style='color:#6B6B6B'>Overlap: "
            f"{100*(1 - rolling_step/rolling_window):.0f}%</small>",
            unsafe_allow_html=True)
    else:
        rolling_window = 250; rolling_step = 250

    st.divider()
    st.markdown("### DISTRIBUTION")
    dist_mode = st.selectbox("Reconstruction method",
        ["non_parametric", "parametric"], label_visibility="collapsed")
    n_samples = st.select_slider("Sample paths M",
        options=[100, 200, 500], value=200)

    st.divider()
    st.markdown("### VISUALIZATIONS")
    show_pit_plots   = st.toggle("PIT diagnostic plots", value=True)
    show_rolling_cov = st.toggle("Rolling coverage chart", value=True)
    show_power_plot  = st.toggle("Power analysis", value=False,
                                  help="Shows theoretical power curves (~30s to compute)")

    st.divider()
    st.markdown("### AI NARRATIVES")
    def _get_key():
        try:    return st.secrets.get("ANTHROPIC_API_KEY", "")
        except: return os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input("Anthropic API key", type="password",
                             value=_get_key(), label_visibility="collapsed") \
              or os.environ.get("ANTHROPIC_API_KEY", "")


# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(
        "<h1>UNIFIED PROBABILISTIC VALIDATION</h1>"
        "<p style='color:#6B6B6B;font-family:DM Mono,monospace;font-size:0.82rem;"
        "letter-spacing:0.05em;margin-top:-0.5rem'>"
        "Basel-style governance classification · PIT diagnostics · Conformal augmentation"
        "</p>",
        unsafe_allow_html=True)
with c2:
    st.markdown(
        "<div style='text-align:right;padding-top:0.5rem'>"
        "<span style='color:#F59E0B;font-family:DM Mono,monospace;font-size:0.7rem;"
        "letter-spacing:0.15em'>PRODUCTION FRAMEWORK</span><br>"
        "<span style='color:#4A4A4A;font-family:DM Mono,monospace;font-size:0.65rem'>"
        "v2.0 · LeJ7-commits/unified-probabilistic-validation</span></div>",
        unsafe_allow_html=True)


# ── CSV Format ────────────────────────────────────────────────────────────────
with st.expander("▸ CSV FORMAT REQUIREMENTS"):
    st.markdown("""
**Required columns** (auto-detected by name):

| Role | Accepted names |
|------|---------------|
| Timestamp | `timestamp`, `Datetime`, `date`, `time` |
| Actuals | `y`, `Actuals`, `Load`, `actual` |
| Forecast | `y_hat`, `Simulation`, `Load forecast`, `forecast` |

**Optional:** `lo`/`lower`, `hi`/`upper` for pre-computed interval bounds.
""")

uploaded = st.file_uploader("Upload forecast CSV", type=["csv"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
<div style='border:1px dashed #222;padding:3.5rem;text-align:center;
     background:#131313;margin-top:1rem'>
  <div style='font-family:DM Mono,monospace;font-size:0.9rem;
       color:#4A4A4A;letter-spacing:0.1em'>
    ▣ DROP FORECAST CSV TO BEGIN VALIDATION
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()


# ── Load & detect ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_detect(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}
    def find(cands):
        for c in cands:
            if c in cols: return cols[c]
        return None
    return df, {
        "t":     find(["timestamp","datetime","date","time","index"]),
        "y":     find(["y","actuals","load","actual","observed","realization"]),
        "y_hat": find(["y_hat","simulation","load forecast","forecast",
                       "predicted","yhat","y_pred"]),
        "lo":    find(["lo","lower","q_0.05","q_005","lower_bound","lb"]),
        "hi":    find(["hi","upper","q_0.95","q_095","upper_bound","ub"]),
    }

file_bytes = uploaded.read()
df_raw, col_map = load_and_detect(file_bytes)

MAX_ROWS = 50_000
if len(df_raw) > MAX_ROWS:
    st.warning(f"Dataset capped at {MAX_ROWS:,} rows for cloud deployment.", icon="⚠️")
    df_raw = df_raw.tail(MAX_ROWS).reset_index(drop=True)

with st.expander("▸ DETECTED COLUMN MAPPING"):
    for role, col in col_map.items():
        s = "✓" if col else "✗"
        c = AMBER if col else RED
        st.markdown(
            f"<span style='color:{c};font-family:DM Mono,monospace;"
            f"font-size:0.8rem'>{s} {role:8s}</span>"
            f"<span style='color:#6B6B6B;font-family:DM Mono,monospace;"
            f"font-size:0.8rem'> → {col or 'NOT FOUND'}</span>",
            unsafe_allow_html=True)

missing = [r for r in ["y", "y_hat"] if col_map.get(r) is None]
if missing:
    st.error(f"Required columns missing: {missing}")
    st.stop()


# ── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(df, col_map, alpha, coverage_target, model_class, commodity,
                 api_key, dist_mode, n_samples, rolling_window, rolling_step):

    from src.core.data_contract import DataContract
    from src.adapters.point_forecast import Adapter_PointForecast, bucket_none
    from src.adapters.build_dist_from_residuals import BuildDist_FromResiduals
    from src.diagnostics.diagnostics_input import Diagnostics_Input
    from src.governance.decision_engine import DecisionEngine
    from src.governance.risk_classification import RiskPolicy
    from src.governance.narrative_generator import NarrativeGenerator

    pb = st.progress(0, text="Validating schema…")

    y      = df[col_map["y"]].values.astype(float)
    y_hat  = df[col_map["y_hat"]].values.astype(float)
    n      = len(y)
    t_col  = col_map.get("t")
    try:
        t = pd.to_datetime(df[t_col], utc=True).values if t_col else np.arange(n)
    except Exception:
        try:
            t = pd.to_datetime(df[t_col]).values if t_col else np.arange(n)
        except Exception:
            t = np.arange(n)
    lo = df[col_map["lo"]].values.astype(float) if col_map.get("lo") else None
    hi = df[col_map["hi"]].values.astype(float) if col_map.get("hi") else None

    pb.progress(12, text="Building residual pool…")
    contract = DataContract(min_obs=10)
    try:
        std_obj = contract.validate(t=t, y=y, model_id="uploaded",
                                    split="window_0", y_hat=y_hat)
    except Exception as e:
        st.error(f"Schema validation failed: {e}"); st.stop()

    W = min(rolling_window, max(30, n // 10))
    adapter = Adapter_PointForecast(W=W, alpha=alpha, bucket_fn=bucket_none,
                                    N_min_hard=max(10, W//4),
                                    N_min_soft=max(20, W//2))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:   pool = adapter.transform(std_obj)
        except Exception as e:
            st.error(f"Adapter failed: {e}"); st.stop()

    pb.progress(35, text=f"Reconstructing distribution ({dist_mode}, M={n_samples})…")
    builder = BuildDist_FromResiduals(M=n_samples, mode=dist_mode, seed=42)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        sample_matrix = builder.build(pool)

    pb.progress(55, text="Running diagnostic battery…")
    di = Diagnostics_Input(alpha=alpha)
    use_lo = pool.pool_lo[:pool.n_obs]
    use_hi = pool.pool_hi[:pool.n_obs]

    dro = di.from_arrays(
        y=pool.y_eval, t=pool.t_eval, model_id="uploaded",
        samples=sample_matrix.samples,
        lo=use_lo, hi=use_hi,
        quantiles={alpha/2: use_lo, 1-alpha/2: use_hi},
    )

    pb.progress(72, text="Classifying governance label…")
    engine   = DecisionEngine(alpha=alpha,
                               global_policy=RiskPolicy(coverage_target=coverage_target))
    decision = engine.decide(dro)

    pb.progress(88, text="Generating narrative…")
    narrator = NarrativeGenerator(api_key=api_key or None)
    class _P:
        def __init__(self, d):
            self.model_id=d["model_id"]; self.final_label=d["final_label"]
            self.reason_codes=d["reason_codes"]; self._d=d
        def to_dict(self): return self._d
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        narrative = narrator.generate(_P(decision.to_dict()),
                                      model_class=model_class,
                                      commodity_context=commodity)
    pb.progress(100, text="Complete."); pb.empty()
    return decision, narrative, pool, sample_matrix


with st.spinner(""):
    try:
        decision, narrative, pool, sample_matrix = run_pipeline(
            df_raw, col_map, alpha, coverage_target, model_class, commodity,
            api_key, dist_mode, n_samples, rolling_window, rolling_step)
    except Exception as e:
        st.error(f"Pipeline error: {e}"); st.exception(e); st.stop()


# ── Results ───────────────────────────────────────────────────────────────────
snap  = decision.metric_snapshot
label = decision.final_label

st.markdown('<div class="section-rule">▸ GOVERNANCE DECISION</div>',
            unsafe_allow_html=True)

col_v, col_k = st.columns([1, 3])
with col_v:
    css = {"GREEN": "verdict-green", "YELLOW": "verdict-yellow",
           "RED": "verdict-red"}.get(label, "verdict-red")
    codes     = decision.reason_codes or ["all_clear"]
    codes_str = [rc.value if hasattr(rc, "value") else str(rc) for rc in codes]
    chips     = "".join(f'<span class="code-chip">{c}</span>' for c in codes_str)
    st.markdown(
        f'<div class="verdict-block {css}">'
        f'<span class="verdict-label">{label}</span>'
        f'<div style="margin-top:0.8rem">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True)

with col_k:
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, val, label, sub="", color=None):
        c = color or "#E5E5E5"
        col.markdown(
            f'<div class="kpi-block">'
            f'<div class="kpi-value" style="color:{c}">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    cov = snap.get("empirical_coverage")
    cov_err = (cov - coverage_target) if cov else None
    cov_color = GREEN if cov_err and abs(cov_err) < 0.02 else \
                YELLOW if cov_err and abs(cov_err) < 0.05 else RED
    kpi(k1, _fmt(cov, pct=True), "COVERAGE",
        f"target {coverage_target:.0%}", cov_color)

    ks_p = snap.get("pit_ks_pvalue")
    ks_s = snap.get("pit_ks_stat")
    kpi(k2, _fmt(ks_p), "KS p-VALUE",
        f"KS stat={_fmt(ks_s)}" if ks_s else "PIT uniformity",
        RED if (ks_p is not None and ks_p < 0.05) else GREEN)

    lb_p = snap.get("pit_lb_pvalue_lag20")
    kpi(k3, _fmt(lb_p), "LB p-VALUE (lag 20)",
        "PIT independence",
        RED if (lb_p is not None and lb_p < 0.05) else GREEN)

    mw = snap.get("mean_width")
    kpi(k4, _fmt(mw), "MEAN WIDTH",
        snap.get("sharpness_label", "—"))

# ── Anfuso + snapshot ─────────────────────────────────────────────────────────
st.markdown('<div class="section-rule">▸ INTERVAL BACKTESTING</div>',
            unsafe_allow_html=True)
ca, cb = st.columns([1, 1])
with ca:
    rows = [
        ("TOTAL", snap.get("anfuso_traffic_light_total"),
         snap.get("total_breach_rate")),
        ("LOWER TAIL", snap.get("anfuso_traffic_light_lower"),
         snap.get("lower_breach_rate")),
        ("UPPER TAIL", snap.get("anfuso_traffic_light_upper"),
         snap.get("upper_breach_rate")),
    ]
    tbl = "".join(
        f"<tr><td style='color:#6B6B6B'>{r}</td>"
        f"<td style='color:{tl_color(tl)};font-weight:500'>{tl or '—'}</td>"
        f"<td style='color:#E5E5E5'>{_fmt(br, pct=True)}</td></tr>"
        for r, tl, br in rows
    )
    st.markdown(
        f'<table class="anf-table">{tbl}</table>',
        unsafe_allow_html=True)
with cb:
    snap_rows = {"metric": list(snap.keys()),
               "value": [str(round(v,6)) if isinstance(v,(float,int)) else str(v)
                         for v in snap.values()]}
    snap_df = pd.DataFrame(snap_rows).set_index("metric")
    st.dataframe(snap_df, height=220)

# ── Visualizations ────────────────────────────────────────────────────────────
if show_pit_plots:
    st.markdown('<div class="section-rule">▸ PIT DIAGNOSTIC PLOTS</div>',
                unsafe_allow_html=True)
    u = np.mean(sample_matrix.samples <= pool.y_eval[:, None], axis=1)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    n_eval = len(u)
    fig = fig_pit_diagnostics(u,
        f"PIT Diagnostics — uploaded model  (n={n_eval:,})")
    st.pyplot(fig)
    plt.close(fig)

if show_rolling_cov and enable_rolling:
    st.markdown('<div class="section-rule">▸ ROLLING COVERAGE</div>',
                unsafe_allow_html=True)
    y_e  = pool.y_eval
    lo_e = pool.pool_lo[:pool.n_obs]
    hi_e = pool.pool_hi[:pool.n_obs]
    n_e  = len(y_e)
    cov_series = []
    for start in range(0, n_e - rolling_window + 1, rolling_step):
        end = start + rolling_window
        sl  = slice(start, end)
        cov_series.append(float(np.mean(
            (y_e[sl] >= lo_e[sl]) & (y_e[sl] <= hi_e[sl]))))
    if len(cov_series) >= 2:
        fig = fig_rolling_coverage(cov_series, list(range(len(cov_series))),
                                   coverage_target)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"Need ≥2 windows for rolling chart "
                f"(got {len(cov_series)} with window={rolling_window}, "
                f"step={rolling_step})")

if show_power_plot:
    st.markdown('<div class="section-rule">▸ POWER ANALYSIS</div>',
                unsafe_allow_html=True)
    with st.spinner("Computing power curves (~20s)…"):
        fig = fig_power_vs_n(n_uploaded=pool.n_obs)
    st.pyplot(fig)
    plt.close(fig)

# ── Narratives ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-rule">▸ GOVERNANCE NARRATIVES</div>',
            unsafe_allow_html=True)
if not narrative.api_used:
    st.warning("Add Anthropic API key in sidebar for AI-generated narratives.", icon="▸")
tab_t, tab_p = st.tabs(["TECHNICAL · RISK OFFICER", "PLAIN LANGUAGE · MANAGEMENT"])
with tab_t:
    st.markdown(f'<div class="narrative-block">{narrative.technical_narrative}</div>',
                unsafe_allow_html=True)
with tab_p:
    st.markdown(
        f'<div class="narrative-block narrative-plain">{narrative.plain_narrative}</div>',
        unsafe_allow_html=True)

# ── Provenance ────────────────────────────────────────────────────────────────
with st.expander("▸ DECISION PROVENANCE"):
    prov = decision.provenance
    pc, ps = st.columns(2)
    with pc:
        st.markdown("**Computed**")
        for d in prov.get("computed", []):
            st.markdown(f"`{d}`")
    with ps:
        st.markdown("**Skipped**")
        for d in prov.get("skipped", []):
            st.markdown(f"`{d['diagnostic']}` — {d['reason']}")
    st.markdown(f"**Policy:** `{prov.get('policy_source','—')}`")
    st.markdown(f"**Timestamp:** `{decision.decided_at}`")

# ── Download ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-rule">▸ ARTIFACTS</div>', unsafe_allow_html=True)

def build_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("governance_decision.json",
                    json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
        zf.writestr("narrative_technical.md", narrative.technical_narrative)
        zf.writestr("narrative_plain.md",     narrative.plain_narrative)
        zf.writestr("narrative_combined.md",  narrative.to_markdown())
        zf.writestr("metric_snapshot.csv",
                    pd.DataFrame.from_dict(
                        {k:[v] for k,v in decision.metric_snapshot.items()}
                    ).to_csv(index=False))
        if show_pit_plots:
            u = np.mean(sample_matrix.samples <= pool.y_eval[:, None], axis=1)
            fig = fig_pit_diagnostics(np.clip(u, 1e-12, 1-1e-12),
                                      "PIT Diagnostics")
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0c0c0c")
            zf.writestr("pit_diagnostics.png", img.getvalue())
            plt.close(fig)
    buf.seek(0)
    return buf.read()

st.download_button(
    label="⬇ DOWNLOAD ALL ARTIFACTS  (.zip)",
    data=build_zip(),
    file_name=f"upv_{label.lower()}_{decision.model_id}.zip",
    mime="application/zip",
)
