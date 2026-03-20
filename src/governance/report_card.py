"""
src/governance/report_card.py
==============================
Governance_ReportCard: aggregates TrafficLight_Labeler, transition matrix,
and entropy outputs into a reproducible governance report.

Architecture (per diagram):
  INPUT  : labels + reason codes, key diagnostic stats per window,
           stability outputs (transition matrix, entropy)
  OUTPUT :
    1. Time-series plot of classification labels (coloured bands)
    2. Table of window-level metrics + label
    3. Regime-tag stratified confusion matrix
    4. Narrative "why label changed" sections

Sanity checks (per diagram):
  - Reproducible: same config → same report
  - Configs stored (window size, α levels, regime rules)

Usage
-----
from src.governance.report_card import Governance_ReportCard, ReportCardConfig
from src.governance.stability import Stability_TransitionMatrix, Stability_Entropy

config  = ReportCardConfig(dataset_label="ENTSO-E", alpha=0.1,
                           window_size=250, rolling_step=50)
card    = Governance_ReportCard(config)
outputs = card.generate(rolling_df=df, out_dir=Path("experiments/run_001_entsoe"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional matplotlib — graceful failure if not installed
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend safe for scripts
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from src.governance.reason_codes import ReasonCode
from src.governance.risk_classification import TrafficLight_Labeler, RiskPolicy
from src.governance.stability import (
    Stability_TransitionMatrix,
    Stability_Entropy,
    TransitionResult,
    EntropyResult,
    STATES,
)


# ---------------------------------------------------------------------------
# Colour palette for label bands
# ---------------------------------------------------------------------------

LABEL_COLOURS = {
    "GREEN":  "#2ecc71",
    "YELLOW": "#f1c40f",
    "RED":    "#e74c3c",
}

# ---------------------------------------------------------------------------
# ReportCardConfig
# ---------------------------------------------------------------------------

@dataclass
class ReportCardConfig:
    """
    Reproducibility config for Governance_ReportCard.

    All parameters that affect classification or output are stored here
    so that the same config always produces the same report.

    Parameters
    ----------
    dataset_label : str
        Human-readable name for the dataset (e.g. "ENTSO-E").
    alpha : float
        Nominal miscoverage level (e.g. 0.1 for 90% intervals).
    window_size : int
        Rolling window length in observations.
    rolling_step : int
        Step size between windows.
    coverage_target : float
        Nominal coverage (= 1 - alpha).
    pvalue_red : float
        P-value threshold for RED classification.
    pvalue_yellow : float
        P-value threshold for YELLOW classification.
    coverage_tol_red : float
        Coverage error tolerance for RED (in probability units).
    coverage_tol_yellow : float
        Coverage error tolerance for YELLOW.
    min_windows_for_transition : int
        Minimum windows to estimate transition matrix reliably.
    entropy_threshold_unstable : float
        Stationary entropy above which governance is "unstable" (bits).
    regime_col : str or None
        Column name in rolling_df to use for regime stratification.
        If None, regime confusion matrix is skipped.
    """
    dataset_label:               str   = "dataset"
    alpha:                       float = 0.10
    window_size:                 int   = 250
    rolling_step:                int   = 50
    coverage_target:             float = 0.90
    pvalue_red:                  float = 0.01
    pvalue_yellow:               float = 0.05
    coverage_tol_red:            float = 0.05
    coverage_tol_yellow:         float = 0.02
    min_windows_for_transition:  int   = 5
    entropy_threshold_unstable:  float = 1.2
    regime_col:                  str | None = None

    def to_risk_policy(self) -> RiskPolicy:
        return RiskPolicy(
            pvalue_red=self.pvalue_red,
            pvalue_yellow=self.pvalue_yellow,
            coverage_target=self.coverage_target,
            coverage_tol_red=self.coverage_tol_red,
            coverage_tol_yellow=self.coverage_tol_yellow,
        )


# ---------------------------------------------------------------------------
# ReportCard output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReportCardOutputs:
    """
    All outputs produced by Governance_ReportCard.generate().

    Attributes
    ----------
    config : ReportCardConfig
    window_labels : list[str]
        Classified label per rolling window.
    window_table : pd.DataFrame
        Window-level metrics + label + reason codes.
    transition_result : TransitionResult
    entropy_result : EntropyResult
    regime_confusion : pd.DataFrame or None
        Rows = regime tags, columns = GREEN/YELLOW/RED.
    narrative : str
        Human-readable "why label changed" narrative.
    figure_label_bands : matplotlib.Figure or None
    saved_paths : dict[str, Path]
        Paths of all files saved to disk.
    """
    config:              ReportCardConfig
    window_labels:       list[str]
    window_table:        pd.DataFrame
    transition_result:   TransitionResult
    entropy_result:      EntropyResult
    regime_confusion:    pd.DataFrame | None
    narrative:           str
    figure_label_bands:  Any   # matplotlib.Figure or None
    saved_paths:         dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Governance_ReportCard
# ---------------------------------------------------------------------------

class Governance_ReportCard:
    """
    Aggregates governance diagnostics into a reproducible report.

    Parameters
    ----------
    config : ReportCardConfig
        Reproducibility config — same config always produces same output.

    Example
    -------
    >>> config  = ReportCardConfig(dataset_label="ENTSO-E", alpha=0.1,
    ...                            window_size=250, rolling_step=50)
    >>> card    = Governance_ReportCard(config)
    >>> outputs = card.generate(rolling_df=df,
    ...                         out_dir=Path("experiments/run_001_entsoe"))
    >>> outputs.narrative
    >>> outputs.figure_label_bands.savefig(...)
    """

    def __init__(self, config: ReportCardConfig) -> None:
        self.config  = config
        self._labeler = TrafficLight_Labeler(config.to_risk_policy())
        self._stm     = Stability_TransitionMatrix(
            min_windows=config.min_windows_for_transition
        )
        self._se      = Stability_Entropy(
            entropy_threshold_unstable=config.entropy_threshold_unstable
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        rolling_df: pd.DataFrame,
        out_dir: Path | None = None,
    ) -> ReportCardOutputs:
        """
        Generate the full governance report.

        Parameters
        ----------
        rolling_df : pd.DataFrame
            Rolling diagnostic CSV loaded as DataFrame. Expected columns
            include empirical_coverage, pit_ks_pvalue, pit_cvm_pvalue,
            pit_lb_pvalue_lag{n}, window_start, window_end.
        out_dir : Path, optional
            If provided, all outputs are saved to this directory.

        Returns
        -------
        ReportCardOutputs
        """
        # 1. Classify each window
        window_labels, window_table = self._classify_windows(rolling_df)

        # 2. Transition matrix + entropy
        tr  = self._stm.fit(window_labels)
        ent = self._se.compute_from_transition(tr)

        # 3. Regime confusion (if regime_col provided)
        regime_confusion = self._regime_confusion(rolling_df, window_labels)

        # 4. Narrative
        narrative = self._build_narrative(window_table, tr, ent)

        # 5. Label band plot
        fig = self._plot_label_bands(window_table) if _HAS_MPL else None

        # 6. Save to disk
        saved_paths: dict[str, Path] = {}
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = self._save_all(
                out_dir, window_table, tr, ent,
                regime_confusion, narrative, fig
            )

        return ReportCardOutputs(
            config=self.config,
            window_labels=window_labels,
            window_table=window_table,
            transition_result=tr,
            entropy_result=ent,
            regime_confusion=regime_confusion,
            narrative=narrative,
            figure_label_bands=fig,
            saved_paths=saved_paths,
        )

    # ── Internal methods ────────────────────────────────────────────────────

    def _classify_windows(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[str], pd.DataFrame]:
        """Classify each row of the rolling DataFrame."""
        rows = df.to_dict("records")
        labels   = []
        rc_lists = []
        msg_lists = []

        for row in rows:
            result = self._labeler.label(row)
            labels.append(result.label)
            rc_lists.append([rc.value for rc in result.reason_codes])
            msg_lists.append("; ".join(result.reason_messages))

        out_df = df.copy()
        out_df["governance_label"] = labels
        out_df["reason_codes"]     = rc_lists
        out_df["reason_summary"]   = msg_lists
        return labels, out_df

    def _regime_confusion(
        self,
        df: pd.DataFrame,
        labels: list[str],
    ) -> pd.DataFrame | None:
        """
        Build regime-tag stratified confusion matrix.
        Rows = regime tags, columns = GREEN / YELLOW / RED.
        """
        rc = self.config.regime_col
        if rc is None or rc not in df.columns:
            return None

        regimes = df[rc].values
        records = []
        for regime, label in zip(regimes, labels):
            records.append({"regime": regime, "label": label})

        regime_df = pd.DataFrame(records)
        confusion  = (
            regime_df.groupby(["regime", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=STATES, fill_value=0)
        )
        return confusion

    def _build_narrative(
        self,
        window_table: pd.DataFrame,
        tr: TransitionResult,
        ent: EntropyResult,
    ) -> str:
        """
        Build a narrative description of governance stability,
        including "why label changed" sections at detected transitions.
        """
        cfg = self.config
        lines = []
        lines.append(
            f"# Governance Narrative — {cfg.dataset_label}\n"
            f"Window size: {cfg.window_size}  |  "
            f"Step: {cfg.rolling_step}  |  "
            f"α = {cfg.alpha}  |  "
            f"Coverage target: {cfg.coverage_target:.0%}\n"
        )

        # Overall stability summary
        lines.append("## Overall Stability")
        lines.append(
            f"- Stationary entropy: {ent.stationary_entropy:.3f} bits "
            f"(max = {ent.max_possible_entropy:.3f} bits)\n"
            f"- Stability classification: **{ent.stability_label.upper()}**\n"
            f"- Interpretation: {ent.interpretation}\n"
        )

        # State frequency
        lines.append("## State Distribution")
        for s in STATES:
            pct = tr.state_freq.get(s, 0) * 100
            lines.append(f"- {s}: {pct:.1f}% of windows ({tr.state_counts.get(s, 0)} windows)")
        lines.append("")

        # Absorbing states
        if ReasonCode.ABSORBING_RED in tr.reason_codes:
            lines.append(
                "## Absorbing State Detected\n"
                "RED is an absorbing state (T_RR ≈ 1.0). Once the model enters "
                "the RED zone it never recovers within the evaluation period. "
                "This indicates persistent structural miscalibration requiring "
                "model intervention rather than routine monitoring.\n"
            )

        # Transitions narrative
        if "governance_label" in window_table.columns:
            labels_col = window_table["governance_label"].tolist()
            transitions = []
            for t in range(1, len(labels_col)):
                if labels_col[t] != labels_col[t - 1]:
                    transitions.append({
                        "window_idx": t,
                        "from_label": labels_col[t - 1],
                        "to_label":   labels_col[t],
                        "reason":     window_table.iloc[t].get("reason_summary", ""),
                        "window_start": window_table.iloc[t].get("window_start", ""),
                    })

            if transitions:
                lines.append(f"## Label Transitions ({len(transitions)} detected)")
                for tr_event in transitions[:20]:   # cap at 20 for readability
                    lines.append(
                        f"- Window {tr_event['window_idx']} "
                        f"(start: {tr_event['window_start']}): "
                        f"{tr_event['from_label']} → {tr_event['to_label']}\n"
                        f"  Reason: {tr_event['reason'] or 'N/A'}"
                    )
                if len(transitions) > 20:
                    lines.append(
                        f"  ... and {len(transitions) - 20} further transitions "
                        "(see window_table for full detail)."
                    )
            else:
                lines.append("## Label Transitions\nNo transitions detected — "
                             "classification is constant across all windows.")

        return "\n".join(lines)

    def _plot_label_bands(self, window_table: pd.DataFrame):
        """
        Time-series plot of governance labels as coloured horizontal bands.
        Returns matplotlib Figure.
        """
        if not _HAS_MPL:
            return None

        labels = window_table["governance_label"].values
        n      = len(labels)
        xs     = np.arange(n)

        fig, ax = plt.subplots(figsize=(14, 3))

        # Draw coloured bands
        for i, label in enumerate(labels):
            ax.axvspan(i - 0.5, i + 0.5,
                       color=LABEL_COLOURS.get(label, "#cccccc"),
                       alpha=0.7)

        # Overlay coverage line if available
        if "empirical_coverage" in window_table.columns:
            cov = pd.to_numeric(window_table["empirical_coverage"],
                                errors="coerce").values
            ax2 = ax.twinx()
            ax2.plot(xs, cov, color="black", lw=1.2, label="Coverage")
            ax2.axhline(self.config.coverage_target, color="black",
                        lw=0.8, ls="--", alpha=0.5)
            ax2.set_ylabel("Empirical coverage", fontsize=9)
            ax2.set_ylim(max(0, np.nanmin(cov) - 0.05), 1.0)

        # Legend
        patches = [
            mpatches.Patch(color=LABEL_COLOURS[s], label=s)
            for s in STATES
        ]
        ax.legend(handles=patches, loc="upper left", fontsize=9)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("Window index", fontsize=9)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_title(
            f"Governance Classification — {self.config.dataset_label}  "
            f"(window={self.config.window_size}, step={self.config.rolling_step})",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    def _save_all(
        self,
        out_dir: Path,
        window_table: pd.DataFrame,
        tr: TransitionResult,
        ent: EntropyResult,
        regime_confusion: pd.DataFrame | None,
        narrative: str,
        fig: Any,
    ) -> dict[str, Path]:
        saved: dict[str, Path] = {}

        # Config
        cfg_path = out_dir / "report_card_config.json"
        cfg_path.write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        saved["config"] = cfg_path

        # Window table
        wt_path = out_dir / "report_card_window_table.csv"
        window_table.to_csv(wt_path, index=False)
        saved["window_table"] = wt_path

        # Transition + entropy
        stab_path = out_dir / "report_card_stability.json"
        stab_path.write_text(json.dumps({
            "transition": tr.to_dict(),
            "entropy":    ent.to_dict(),
        }, indent=2), encoding="utf-8")
        saved["stability"] = stab_path

        # Regime confusion
        if regime_confusion is not None:
            rc_path = out_dir / "report_card_regime_confusion.csv"
            regime_confusion.to_csv(rc_path)
            saved["regime_confusion"] = rc_path

        # Narrative
        narr_path = out_dir / "report_card_narrative.md"
        narr_path.write_text(narrative, encoding="utf-8")
        saved["narrative"] = narr_path

        # Label band figure
        if fig is not None:
            fig_path = out_dir / "report_card_label_bands.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved["figure"] = fig_path

        return saved
