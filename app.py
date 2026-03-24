"""
app.py
=======
Streamlit app for the Unified Probabilistic Validation Framework.

Accepts a CSV upload, runs the core validation pipeline directly
(no shell scripts), and produces a governance decision with AI narrative.

Deploy to Streamlit Cloud:
  1. Push this file + requirements_streamlit.txt to your GitHub repo
  2. Go to share.streamlit.io → New app → select repo → app.py
  3. Add ANTHROPIC_API_KEY in Settings → Secrets

Run locally:
  pip install streamlit
  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="UPV Framework",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  }

  .main { background-color: #0d1117; }

  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
  }

  .label-green {
    background: #0f2d1f;
    border: 2px solid #2ea043;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.15em;
  }
  .label-yellow {
    background: #2d2208;
    border: 2px solid #9e6a03;
    color: #d29922;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.15em;
  }
  .label-red {
    background: #2d0f11;
    border: 2px solid #da3633;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.15em;
  }

  .reason-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    display: inline-block;
    margin: 0.2rem;
  }

  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  .narrative-box {
    background: #161b22;
    border-left: 3px solid #388bfd;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin-bottom: 1rem;
  }

  .narrative-plain {
    border-left-color: #2ea043;
  }

  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e6edf3;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .stAlert { border-radius: 6px; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ UPV Framework")
    st.markdown(
        "Unified Probabilistic Validation  \n"
        "for Energy Market Models"
    )
    st.divider()

    st.markdown("### Configuration")

    model_class = st.selectbox(
        "Model class",
        ["short_term", "long_term", "simulation"],
        help="Short-term: hourly load/price forecasts. "
             "Long-term: renewables generation. "
             "Simulation: Monte Carlo paths.",
    )

    commodity = st.text_input(
        "Commodity / context",
        value="electricity load forecast",
        help="e.g. 'ENTSO-E electricity load', 'PV solar generation', "
             "'natural gas price simulation'",
    )

    alpha = st.select_slider(
        "Miscoverage level α",
        options=[0.05, 0.10, 0.20],
        value=0.10,
        help="1−α is the nominal prediction interval coverage. "
             "Default 0.10 → 90% interval.",
    )

    coverage_target = 1 - alpha

    dist_mode = st.selectbox(
        "Distribution reconstruction",
        ["non_parametric", "parametric"],
        help="non_parametric: bootstrap resamples residuals — no distribution "
             "assumption, preserves skewness and heavy tails (recommended). "
             "parametric: fits a Gaussian to the residual pool — faster but "
             "understates tail risk for heavy-tailed commodities.",
    )

    n_samples = st.select_slider(
        "Sample paths M",
        options=[100, 200, 500],
        value=200,
        help="Number of Monte Carlo paths for PIT and CRPS computation.",
    )

    st.divider()

    st.markdown("### AI Narratives")

    def _get_default_api_key() -> str:
        try:
            return st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            return os.environ.get("ANTHROPIC_API_KEY", "")

    api_key_input = st.text_input(
        "Anthropic API key",
        type="password",
        value=_get_default_api_key(),
        help="Required for AI-generated governance narratives. "
             "Get a key at console.anthropic.com",
    )
    api_key = api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")

    st.divider()
    st.markdown(
        "<small style='color:#8b949e'>Master's Thesis — Lund University  \n"
        "Jia Yang Le & Komila Askarova  \n"
        "Industry Partner: Energy Quant Solutions</small>",
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.8rem; margin-bottom:0.25rem'>"
    "Unified Probabilistic Validation Framework</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e; margin-bottom:2rem'>"
    "Upload a forecast CSV to validate probabilistic calibration "
    "and receive a Basel-style GREEN / YELLOW / RED governance classification.</p>",
    unsafe_allow_html=True,
)


# ── CSV format guide ─────────────────────────────────────────────────────────
with st.expander("📋 CSV format requirements"):
    st.markdown("""
**Minimum required columns** (column names are flexible — the app auto-detects):

| Column | Accepts | Description |
|--------|---------|-------------|
| Timestamp | `timestamp`, `Datetime`, `date`, `time` | Datetime index |
| Actuals | `y`, `Actuals`, `Load`, `actual` | Realized values |
| Point forecast | `y_hat`, `Simulation`, `Load forecast`, `forecast` | Model forecast |

**Optional columns** (enable richer diagnostics):
- `lo` / `lower` / `q_0.05` — lower interval bound
- `hi` / `upper` / `q_0.95` — upper interval bound

**Example:**
```
timestamp,y,y_hat
2020-01-01 00:00,45231,44800
2020-01-01 01:00,43150,43500
...
```
""")


# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload forecast CSV",
    type=["csv"],
    help="Upload a CSV with timestamps, realizations, and point forecasts.",
)

if uploaded is None:
    st.markdown("""
<div style='background:#161b22; border:1px dashed #30363d; border-radius:8px;
     padding:3rem; text-align:center; color:#8b949e; margin-top:1rem'>
  <div style='font-size:2rem; margin-bottom:0.5rem'>📂</div>
  <div style='font-family:IBM Plex Mono,monospace; font-size:0.9rem'>
    Upload a CSV to begin validation
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Column detection ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_detect(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    cols = {c.lower().strip(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    mapping = {
        "t":    find(["timestamp", "datetime", "date", "time", "index"]),
        "y":    find(["y", "actuals", "load", "actual", "observed", "realization"]),
        "y_hat":find(["y_hat", "simulation", "load forecast", "forecast",
                      "predicted", "yhat", "y_pred"]),
        "lo":   find(["lo", "lower", "q_0.05", "q_005", "lower_bound", "lb"]),
        "hi":   find(["hi", "upper", "q_0.95", "q_095", "upper_bound", "ub"]),
    }
    return df, mapping


try:
    file_bytes = uploaded.read()
    df_raw, col_map = load_and_detect(file_bytes)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

MAX_ROWS = 50_000
if len(df_raw) > MAX_ROWS:
    st.warning(
        f"Large dataset: {len(df_raw):,} rows. "
        f"Subsampling to {MAX_ROWS:,} rows for cloud performance. "
        "Run locally for full-dataset validation.",
        icon="⚠️"
    )
    df_raw = df_raw.tail(MAX_ROWS).reset_index(drop=True)

# Show detection results
with st.expander("🔍 Detected column mapping", expanded=False):
    for role, col in col_map.items():
        status = "✅" if col else "⚠️ not found"
        st.markdown(f"- **{role}** → `{col or status}`")

# Validate required columns
missing = [r for r in ["y", "y_hat"] if col_map.get(r) is None]
if missing:
    st.error(
        f"Required columns not found: {missing}. "
        "Please check the CSV format guide above."
    )
    st.stop()


# ── Run validation pipeline ────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, col_map: dict, alpha: float, coverage_target: float,
                 model_class: str, commodity: str, api_key: str,
                 dist_mode: str = "non_parametric", n_samples: int = 200):
    """Run the core validation pipeline directly on the dataframe."""

    from src.core.data_contract import DataContract
    from src.adapters.point_forecast import Adapter_PointForecast, bucket_none
    from src.adapters.build_dist_from_residuals import BuildDist_FromResiduals
    from src.diagnostics.diagnostics_input import Diagnostics_Input
    from src.governance.decision_engine import DecisionEngine
    from src.governance.risk_classification import RiskPolicy
    from src.governance.narrative_generator import NarrativeGenerator

    progress = st.progress(0, text="Validating schema…")

    # ── 1. Parse arrays ───────────────────────────────────────────────────
    y     = df[col_map["y"]].values.astype(float)
    y_hat = df[col_map["y_hat"]].values.astype(float)
    n     = len(y)

    # Timestamps
    if col_map.get("t"):
        try:
            t = pd.to_datetime(df[col_map["t"]]).values
        except Exception:
            t = np.arange(n)
    else:
        t = np.arange(n)

    # Optional bounds
    lo = df[col_map["lo"]].values.astype(float) if col_map.get("lo") else None
    hi = df[col_map["hi"]].values.astype(float) if col_map.get("hi") else None

    progress.progress(15, text="Building residual pool…")

    # ── 2. DataContract ───────────────────────────────────────────────────
    contract = DataContract(min_obs=10)
    try:
        std_obj = contract.validate(
            t=t, y=y, model_id="uploaded_model",
            split="window_0", y_hat=y_hat,
        )
    except Exception as e:
        st.error(f"DataContract validation failed: {e}")
        st.stop()

    progress.progress(25, text="Computing residual intervals…")

    # ── 3. Adapter_PointForecast → residual pool ──────────────────────────
    W = min(720, max(30, n // 10))
    adapter = Adapter_PointForecast(
        W=W,
        alpha=alpha,
        bucket_fn=bucket_none,
        N_min_hard=max(10, W // 4),
        N_min_soft=max(20, W // 2),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            pool = adapter.transform(std_obj)
        except Exception as e:
            st.error(f"Adapter failed: {e}")
            st.stop()

    progress.progress(42, text=f"Reconstructing predictive distribution ({dist_mode})…")

    # ── 4. BuildDist_FromResiduals → sample matrix ────────────────────────
    builder = BuildDist_FromResiduals(M=n_samples, mode=dist_mode, seed=42)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        sample_matrix = builder.build(pool)

    progress.progress(58, text="Running diagnostics…")

    # ── 5. Diagnostics_Input ──────────────────────────────────────────────
    di = Diagnostics_Input(alpha=alpha)

    # Prefer uploaded bounds if available, else use pool bounds
    use_lo = lo[pool.pool_sizes.nonzero()[0][0]:] if lo is not None else pool.pool_lo
    use_hi = hi[pool.pool_sizes.nonzero()[0][0]:] if hi is not None else pool.pool_hi
    use_lo = use_lo[:pool.n_obs] if len(use_lo) > pool.n_obs else use_lo
    use_hi = use_hi[:pool.n_obs] if len(use_hi) > pool.n_obs else use_hi

    dro = di.from_arrays(
        y         = pool.y_eval,
        t         = pool.t_eval,
        model_id  = "uploaded_model",
        samples   = sample_matrix.samples,   # ← PIT + CRPS now available
        lo        = use_lo,
        hi        = use_hi,
        quantiles = {alpha / 2: use_lo, 1 - alpha / 2: use_hi},
    )

    progress.progress(70, text="Classifying governance label…")

    # ── 5. DecisionEngine ─────────────────────────────────────────────────
    engine = DecisionEngine(
        alpha         = alpha,
        global_policy = RiskPolicy(coverage_target=coverage_target),
    )
    decision = engine.decide(dro)

    progress.progress(85, text="Generating AI narrative…")

    # ── 6. NarrativeGenerator ─────────────────────────────────────────────
    narrator = NarrativeGenerator(api_key=api_key or None)

    class _Proxy:
        def __init__(self, d):
            self.model_id     = d["model_id"]
            self.final_label  = d["final_label"]
            self.reason_codes = d["reason_codes"]
            self._d = d
        def to_dict(self): return self._d

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        narrative = narrator.generate(
            _Proxy(decision.to_dict()),
            model_class       = model_class,
            commodity_context = commodity,
        )

    progress.progress(100, text="Complete.")
    progress.empty()

    return decision, narrative, pool


with st.spinner("Running validation pipeline…"):
    try:
        decision, narrative, pool = run_pipeline(
            df_raw, col_map, alpha, coverage_target,
            model_class, commodity, api_key,
            dist_mode=dist_mode, n_samples=n_samples,
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()


# ── Results display ───────────────────────────────────────────────────────────
st.divider()

# ── Top row: label + key metrics ──────────────────────────────────────────────
col_label, col_metrics = st.columns([1, 3])

with col_label:
    st.markdown('<div class="section-header">Governance Decision</div>',
                unsafe_allow_html=True)
    label = decision.final_label
    css_class = {"GREEN": "label-green", "YELLOW": "label-yellow",
                 "RED": "label-red"}.get(label, "label-red")
    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)

    st.markdown("<br>**Reason codes**", unsafe_allow_html=True)
    codes = decision.reason_codes if decision.reason_codes else ["all_clear"]
    codes_str = [rc.value if hasattr(rc, 'value') else str(rc) for rc in codes]
    chips = "".join(f'<span class="reason-chip">{c}</span>' for c in codes_str)
    st.markdown(chips, unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="section-header">Key Diagnostics</div>',
                unsafe_allow_html=True)
    snap = decision.metric_snapshot

    m1, m2, m3, m4 = st.columns(4)

    def _fmt(v, pct=False):
        if v is None: return "—"
        if pct: return f"{v:.1%}"
        if isinstance(v, float) and v < 0.001:
            return f"{v:.2e}" if v > 0 else "< 1e-300"
        return f"{v:.4f}"

    emp_cov = snap.get("empirical_coverage")
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{_fmt(emp_cov, pct=True)}</div>'
            f'<div class="metric-label">Empirical Coverage</div>'
            f'<div style="color:#8b949e;font-size:0.78rem">nominal {coverage_target:.0%}</div>'
            f'</div>', unsafe_allow_html=True
        )

    ks_p = snap.get("pit_ks_pvalue")
    with m2:
        ks_color = "#f85149" if (ks_p is not None and ks_p < 0.05) else "#3fb950"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{ks_color}">'
            f'{_fmt(ks_p)}</div>'
            f'<div class="metric-label">PIT KS p-value</div>'
            f'<div style="color:#8b949e;font-size:0.78rem">H₀: uniform PIT</div>'
            f'</div>', unsafe_allow_html=True
        )

    lb_p = snap.get(f"pit_lb_pvalue_lag20")
    with m3:
        lb_color = "#f85149" if (lb_p is not None and lb_p < 0.05) else "#3fb950"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{lb_color}">'
            f'{_fmt(lb_p)}</div>'
            f'<div class="metric-label">LB p-value (lag 20)</div>'
            f'<div style="color:#8b949e;font-size:0.78rem">H₀: PIT independence</div>'
            f'</div>', unsafe_allow_html=True
        )

    mean_w = snap.get("mean_width")
    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{_fmt(mean_w)}</div>'
            f'<div class="metric-label">Mean Interval Width</div>'
            f'<div style="color:#8b949e;font-size:0.78rem">'
            f'{snap.get("sharpness_label", "—")}</div>'
            f'</div>', unsafe_allow_html=True
        )

st.divider()

# ── Anfuso + full metrics ─────────────────────────────────────────────────────
col_anf, col_full = st.columns([1, 1])

with col_anf:
    st.markdown('<div class="section-header">Anfuso Interval Backtest</div>',
                unsafe_allow_html=True)
    anf_total = snap.get("anfuso_traffic_light_total", "—")
    anf_lo    = snap.get("anfuso_traffic_light_lower", "—")
    anf_hi    = snap.get("anfuso_traffic_light_upper", "—")
    br_total  = snap.get("total_breach_rate")
    br_lo     = snap.get("lower_breach_rate")
    br_hi     = snap.get("upper_breach_rate")

    def tl_color(label):
        return {"GREEN": "#3fb950", "YELLOW": "#d29922",
                "RED": "#f85149"}.get(label, "#8b949e")

    st.markdown(f"""
<table style='width:100%; border-collapse:collapse; font-family:IBM Plex Mono,monospace; font-size:0.85rem'>
  <tr style='border-bottom:1px solid #21262d'>
    <td style='padding:0.4rem; color:#8b949e'>Total breaches</td>
    <td style='padding:0.4rem; color:{tl_color(anf_total)}'>{anf_total}</td>
    <td style='padding:0.4rem; color:#c9d1d9'>{_fmt(br_total, pct=True)}</td>
  </tr>
  <tr style='border-bottom:1px solid #21262d'>
    <td style='padding:0.4rem; color:#8b949e'>Lower tail</td>
    <td style='padding:0.4rem; color:{tl_color(anf_lo)}'>{anf_lo}</td>
    <td style='padding:0.4rem; color:#c9d1d9'>{_fmt(br_lo, pct=True)}</td>
  </tr>
  <tr>
    <td style='padding:0.4rem; color:#8b949e'>Upper tail</td>
    <td style='padding:0.4rem; color:{tl_color(anf_hi)}'>{anf_hi}</td>
    <td style='padding:0.4rem; color:#c9d1d9'>{_fmt(br_hi, pct=True)}</td>
  </tr>
</table>
""", unsafe_allow_html=True)

with col_full:
    st.markdown('<div class="section-header">Full Metric Snapshot</div>',
                unsafe_allow_html=True)
    display_snap = {
        k: (round(float(v), 6) if isinstance(v, (float, int)) else v)
        for k, v in snap.items()
        if not isinstance(v, str) or k in (
            "pit_lb_input", "sharpness_label",
            "anfuso_traffic_light_total",
            "anfuso_traffic_light_lower",
            "anfuso_traffic_light_upper",
        )
    }
    st.dataframe(
        pd.DataFrame.from_dict(display_snap, orient="index", columns=["value"]),
        use_container_width=True,
        height=250,
    )

st.divider()

# ── Narratives ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Governance Narratives</div>',
            unsafe_allow_html=True)

if not narrative.api_used:
    st.warning(
        "AI narratives require an Anthropic API key. "
        "Add your key in the sidebar to enable AI-generated explanations. "
        "Stub narratives are shown below.",
        icon="⚠️",
    )

tab_tech, tab_plain = st.tabs(["📊 Technical (Risk Officer)", "📢 Plain Language (Management)"])

with tab_tech:
    st.markdown(
        f'<div class="narrative-box">{narrative.technical_narrative}</div>',
        unsafe_allow_html=True,
    )

with tab_plain:
    st.markdown(
        f'<div class="narrative-box narrative-plain">{narrative.plain_narrative}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Provenance ────────────────────────────────────────────────────────────────
with st.expander("🔎 Decision provenance"):
    prov = decision.provenance
    col_c, col_s = st.columns(2)
    with col_c:
        st.markdown("**Computed diagnostics**")
        for d in prov.get("computed", []):
            st.markdown(f"- `{d}`")
    with col_s:
        st.markdown("**Skipped diagnostics**")
        for d in prov.get("skipped", []):
            st.markdown(f"- `{d['diagnostic']}` — {d['reason']}")
    st.markdown(f"**Policy source:** `{prov.get('policy_source', '—')}`")
    st.markdown(f"**Decided at:** `{decision.decided_at}`")


# ── Download artifacts ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Download Artifacts</div>',
            unsafe_allow_html=True)

def build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # governance_decision.json
        zf.writestr(
            "governance_decision.json",
            json.dumps(decision.to_dict(), indent=2, ensure_ascii=False),
        )
        # narratives
        zf.writestr("narrative_technical.md", narrative.technical_narrative)
        zf.writestr("narrative_plain.md", narrative.plain_narrative)
        zf.writestr("narrative_combined.md", narrative.to_markdown())
        # metric snapshot as CSV
        snap_df = pd.DataFrame.from_dict(
            {k: [v] for k, v in decision.metric_snapshot.items()}
        )
        zf.writestr("metric_snapshot.csv", snap_df.to_csv(index=False))
    buf.seek(0)
    return buf.read()


st.download_button(
    label="⬇ Download all artifacts (.zip)",
    data=build_zip(),
    file_name=f"upv_validation_{decision.model_id}.zip",
    mime="application/zip",
    use_container_width=False,
)
