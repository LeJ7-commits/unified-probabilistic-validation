"""
experiments/run_007_transition_metrics.py
==========================================
Traffic-Light Stability Meta-Diagnostics

Computes transition probability matrices and entropy-based stability
measures from the rolling window classifications produced by run_001–004b.

For each dataset, the script:

1. Reads rolling_non_overlapping.csv (primary) and rolling_overlapping.csv
2. Classifies each rolling window as GREEN / YELLOW / RED using the same
   policy thresholds as the governance module
3. Builds a 3×3 transition probability matrix T where
     T[i, j] = P(next window = j | current window = i)
4. Computes the stationary distribution π from the dominant eigenvector
5. Computes per-row entropy H_i = -∑_j T[i,j] * log₂(T[i,j])
   (measures unpredictability of transitions from each state)
6. Computes overall classification entropy from the stationary distribution
7. Writes per-dataset JSON and a combined summary CSV

Governance policy thresholds (mirroring src/governance/risk_classification.py):
  Coverage error > ±0.05 (5pp)  → RED on coverage
  Coverage error > ±0.02 (2pp)  → YELLOW on coverage
  PIT uniformity p < 0.05        → RED
  Ljung-Box p < 0.05             → RED
  (If PIT stats not available, classify on coverage only)

Outputs saved to experiments/run_007_transition_metrics/:
  {dataset}_non_overlapping_transitions.json
  {dataset}_overlapping_transitions.json
  transition_metrics_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR   = REPO_ROOT / "experiments"
OUT_DIR   = EXP_DIR / "run_007_transition_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATES       = ["GREEN", "YELLOW", "RED"]
STATE_INDEX  = {s: i for i, s in enumerate(STATES)}

# Governance thresholds (must match src/governance/risk_classification.py)
RED_COV_TOL    = 0.05   # |coverage_error| > 5pp → RED
YELLOW_COV_TOL = 0.02   # |coverage_error| > 2pp → YELLOW
PIT_P_THRESH   = 0.05   # p < threshold → FAIL
LB_P_THRESH    = 0.05


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = [
    ("ENTSO-E",               "run_001_entsoe"),
    ("PV Solar",              "run_002_pv"),
    ("Wind",                  "run_003_wind"),
    ("Sim Price (well-spec)", "run_004_simulation_price"),
    ("Sim Temp (well-spec)",  "run_004_simulation_temp"),
    # Misspecification scenarios
    ("Sim Price — Var Inflation", "run_004b_simulation_price_variance_inflation"),
    ("Sim Price — Mean Bias",     "run_004b_simulation_price_mean_bias"),
    ("Sim Price — Heavy Tails",   "run_004b_simulation_price_heavy_tails"),
    ("Sim Temp — Var Inflation",  "run_004b_simulation_temp_variance_inflation"),
    ("Sim Temp — Mean Bias",      "run_004b_simulation_temp_mean_bias"),
    ("Sim Temp — Heavy Tails",    "run_004b_simulation_temp_heavy_tails"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_window(row: pd.Series) -> str:
    """
    Classify a single rolling window row as GREEN / YELLOW / RED.
    Uses coverage error first, then PIT stats if available.
    """
    cov_error = abs(row.get("empirical_coverage", 0.90) - 0.90)

    # Coverage check
    if cov_error > RED_COV_TOL:
        return "RED"

    # PIT uniformity check (if available)
    pit_p = row.get("pit_ks_pvalue", None)
    if pit_p is not None and not np.isnan(pit_p):
        if pit_p < PIT_P_THRESH:
            return "RED"

    # Ljung-Box check (if available)
    lb_p = row.get("pit_lb_pvalue_lag5", None)
    if lb_p is not None and not np.isnan(lb_p):
        if lb_p < LB_P_THRESH:
            return "RED"

    # Yellow zone (mild coverage deviation)
    if cov_error > YELLOW_COV_TOL:
        return "YELLOW"

    return "GREEN"


def build_transition_matrix(states: list[str]) -> np.ndarray:
    """
    Build a 3×3 transition count matrix from a sequence of state labels.
    Returns normalised probability matrix (rows sum to 1 where possible).
    """
    n = len(STATES)
    counts = np.zeros((n, n), dtype=float)

    for t in range(len(states) - 1):
        i = STATE_INDEX.get(states[t],   -1)
        j = STATE_INDEX.get(states[t+1], -1)
        if i >= 0 and j >= 0:
            counts[i, j] += 1

    # Normalise rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid division by zero
    T = counts / row_sums
    return T, counts


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution π of a transition matrix via
    dominant left eigenvector: π T = π.
    Falls back to uniform if matrix is degenerate.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi  = np.real(eigenvectors[:, idx])
        pi  = np.abs(pi) / np.abs(pi).sum()
        return pi
    except Exception:
        return np.ones(len(STATES)) / len(STATES)


def row_entropy(T: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy (bits) for each row of a transition matrix.
    H_i = -∑_j T[i,j] * log2(T[i,j])
    """
    H = np.zeros(T.shape[0])
    for i in range(T.shape[0]):
        row = T[i]
        nonzero = row[row > 0]
        H[i] = -np.sum(nonzero * np.log2(nonzero))
    return H


def stationary_entropy(pi: np.ndarray) -> float:
    """Shannon entropy of the stationary distribution."""
    nonzero = pi[pi > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def analyse_rolling(csv_path: Path, dataset_label: str, scheme: str) -> dict | None:
    """
    Load a rolling CSV, classify each window, build transition matrix,
    and return a dict of results.
    """
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return None

    # Classify each window
    state_seq = [classify_window(row) for _, row in df.iterrows()]
    n_windows = len(state_seq)

    # State frequency counts
    freq = {s: state_seq.count(s) for s in STATES}
    freq_pct = {s: round(freq[s] / n_windows * 100, 1) for s in STATES}

    # Transition matrix
    T, counts = build_transition_matrix(state_seq)

    # Stationary distribution
    pi = stationary_distribution(T)

    # Entropy
    H_rows      = row_entropy(T)
    H_stationary = stationary_entropy(pi)

    result = {
        "dataset":          dataset_label,
        "scheme":           scheme,
        "n_windows":        n_windows,
        "state_sequence":   state_seq,
        "state_frequency":  freq,
        "state_frequency_pct": freq_pct,
        "transition_matrix": {
            STATES[i]: {STATES[j]: round(float(T[i, j]), 4)
                        for j in range(len(STATES))}
            for i in range(len(STATES))
        },
        "transition_counts": {
            STATES[i]: {STATES[j]: int(counts[i, j])
                        for j in range(len(STATES))}
            for i in range(len(STATES))
        },
        "stationary_distribution": {
            STATES[i]: round(float(pi[i]), 4) for i in range(len(STATES))
        },
        "row_entropy_bits": {
            STATES[i]: round(float(H_rows[i]), 4) for i in range(len(STATES))
        },
        "stationary_entropy_bits": round(H_stationary, 4),
        "max_possible_entropy_bits": round(float(np.log2(len(STATES))), 4),
        "stability_interpretation": (
            "stable"   if H_stationary < 0.5 else
            "moderate" if H_stationary < 1.2 else
            "unstable"
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

summary_rows = []

for label, run_dir_name in DATASETS:
    run_dir = EXP_DIR / run_dir_name
    if not run_dir.exists():
        print(f"[SKIP] {label}: directory not found")
        continue

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")

    for scheme, csv_name in [
        ("non_overlapping", "rolling_non_overlapping.csv"),
        ("overlapping",     "rolling_overlapping.csv"),
    ]:
        csv_path = run_dir / csv_name
        result   = analyse_rolling(csv_path, label, scheme)

        if result is None:
            print(f"  [{scheme}] skipped (file missing or < 2 windows)")
            continue

        # Save per-dataset JSON
        out_file = OUT_DIR / f"{run_dir_name}_{scheme}_transitions.json"
        out_file.write_text(json.dumps(result, indent=2))

        # Print summary
        freq_str = "  ".join(
            f"{s}={result['state_frequency_pct'][s]}%"
            for s in STATES
        )
        print(
            f"  [{scheme}]  n={result['n_windows']}  "
            f"{freq_str}  "
            f"H_stat={result['stationary_entropy_bits']:.3f} bits  "
            f"({result['stability_interpretation']})"
        )
        print(f"  Transition matrix:")
        T_df = pd.DataFrame(result["transition_matrix"]).T[STATES]
        print(T_df.to_string(float_format=lambda x: f"{x:.3f}"))

        # Collect for summary CSV
        summary_rows.append({
            "dataset":            label,
            "run_dir":            run_dir_name,
            "scheme":             scheme,
            "n_windows":          result["n_windows"],
            "pct_green":          result["state_frequency_pct"]["GREEN"],
            "pct_yellow":         result["state_frequency_pct"]["YELLOW"],
            "pct_red":            result["state_frequency_pct"]["RED"],
            "T_GG":               result["transition_matrix"]["GREEN"]["GREEN"],
            "T_RR":               result["transition_matrix"]["RED"]["RED"],
            "stationary_green":   result["stationary_distribution"]["GREEN"],
            "stationary_red":     result["stationary_distribution"]["RED"],
            "H_stationary_bits":  result["stationary_entropy_bits"],
            "stability":          result["stability_interpretation"],
            "H_from_green":       result["row_entropy_bits"]["GREEN"],
            "H_from_red":         result["row_entropy_bits"]["RED"],
        })

# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "transition_metrics_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("TRANSITION STABILITY SUMMARY (non-overlapping)")
    print("="*60)
    sub = df_summary[df_summary["scheme"] == "non_overlapping"]
    if not sub.empty:
        print(sub[[
            "dataset", "n_windows", "pct_green", "pct_red",
            "T_RR", "H_stationary_bits", "stability"
        ]].to_string(index=False))

print(f"\nAll results saved to {OUT_DIR}")
