"""
scripts/plot_pit_diagnostics.py  (v2)
======================================
Generates four publication-quality figures:

1. pit_diagnostics_entsoe.png   — 4-panel PIT for ENTSO-E Load
2. pit_diagnostics_pv.png       — 4-panel PIT for PV Solar
3. pit_diagnostics_wind.png     — 4-panel PIT for Wind
4. pit_diagnostics_sim.png      — 4-panel PIT for Simulation (synthetic uniform)

5. power_vs_n.png               — Theoretical power curves (methodology motivation)
   KS panel:  Uniform H0 vs Beta(1.05,1.05) alternative  [shape deviation]
   LB panel:  iid H0  vs AR(1, phi=0.03) alternative     [autocorrelation]

6. model_diagnostic_positioning.png  — Where each actual model sits relative to
   effect-size floors (applied to real data, goes in results section)

Usage:
    python scripts/plot_pit_diagnostics.py

Outputs: figures/
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import norm, kstest, chi2, beta as beta_dist
from scipy.stats.mstats import plotting_positions
from statsmodels.graphics.tsaplots import plot_acf

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG  = REPO / "figures"
FIG.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "lines.linewidth":   1.4,
})

GREY  = "#6B7280"
BLUE  = "#2563EB"
RED   = "#DC2626"
AMBER = "#D97706"
GREEN = "#16A34A"
PURPLE = "#7C3AED"

KS_FLOOR  = 0.05   # effect-size floor for KS
ACF_FLOOR = 0.05   # effect-size floor for ACF lag-1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pit(samples_path: Path, y_path: Path) -> np.ndarray:
    y       = np.load(y_path).astype(float)
    samples = np.load(samples_path).astype(float)
    u = np.mean(samples <= y[:, None], axis=1)
    return np.clip(u, 1e-12, 1 - 1e-12)


def inverse_normal(u: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def fast_ljungbox_p(z: np.ndarray, lag: int = 10) -> float:
    n = len(z)
    acf_vals = np.array([
        np.corrcoef(z[:-k], z[k:])[0, 1] for k in range(1, lag + 1)
    ])
    Q = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, lag + 1)))
    return float(1 - chi2.cdf(Q, df=lag))


# ── Figure 1-4: 4-panel PIT diagnostics ──────────────────────────────────────

def plot_pit_diagnostics(
    u: np.ndarray,
    model_name: str,
    out_path: Path,
    n_lags: int = 40,
) -> None:
    z = inverse_normal(u)
    n = len(u)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"PIT Diagnostics — {model_name}  (n={n:,})",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel 1: Histogram
    ax = axes[0]
    ax.hist(u, bins=20, density=True, color=BLUE, alpha=0.75,
            edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color=RED, linestyle="--", linewidth=1.2,
               label="Uniform(0,1)")
    ax.set_xlabel("PIT value $u_t$")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ks_stat, ks_p = kstest(u, "uniform")
    txt_color = RED if ks_stat > KS_FLOOR else AMBER
    ax.text(0.97, 0.97,
            f"KS={ks_stat:.4f}\np={ks_p:.3g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=txt_color)

    # Panel 2: ACF
    ax = axes[1]
    plot_acf(z, lags=min(n_lags, n // 5), ax=ax,
             color=BLUE, vlines_kwargs={"colors": BLUE},
             alpha=0.05, zero=False)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(r"ACF of $z_t = \Phi^{-1}(u_t)$")
    ax.axhline(0, color=GREY, linewidth=0.8)
    acf_lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    txt_color = RED if abs(acf_lag1) > ACF_FLOOR else GREEN
    ax.text(0.97, 0.97,
            f"ACF(1)={acf_lag1:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=txt_color)

    # Panel 3: Time series
    ax = axes[2]
    step = max(1, n // 2000)
    idx  = np.arange(0, n, step)
    ax.plot(idx, u[idx], color=BLUE, alpha=0.5, linewidth=0.6)
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.0,
               label="Uniform median")
    ax.fill_between([0, len(idx)], 0.05, 0.95,
                    color=GREEN, alpha=0.08, label="90% band")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("$u_t$")
    ax.set_title("PIT Time Series")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)

    # Panel 4: Q-Q
    ax = axes[3]
    u_sorted = np.sort(u)
    pp = plotting_positions(u_sorted, alpha=0.5, beta=0.5)
    ax.scatter(pp, u_sorted, s=1.5, color=BLUE, alpha=0.4, linewidths=0)
    ax.plot([0, 1], [0, 1], color=RED, linestyle="--",
            linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Theoretical quantile (Uniform)")
    ax.set_ylabel("Empirical PIT quantile")
    ax.set_title("Q-Q Plot (PIT vs Uniform)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Figure 5: Theoretical power vs n ─────────────────────────────────────────

def plot_power_vs_n(out_path: Path, n_sim: int = 300) -> None:
    """
    KS panel  — H0: Uniform  vs  Alt: Beta(1.05,1.05) [shape deviation, KS≈0.01]
    LB panel  — H0: iid      vs  Alt: AR(1, phi=0.03) [weak autocorrelation]

    Both alternatives are practically negligible but detectable at large n.
    """
    rng   = np.random.default_rng(42)
    n_vals = [100, 250, 500, 1_000, 2_500, 5_000,
              10_000, 25_000, 50_000, 100_000]
    alpha = 0.05
    lag   = 10
    phi   = 0.03   # AR(1) coefficient for LB alternative

    ks_h0  = []; ks_alt = []
    lb_h0  = []; lb_alt = []

    print("  Computing theoretical power curves...")
    for n in n_vals:
        k_h0 = k_alt = l_h0 = l_alt = 0
        for _ in range(n_sim):
            # KS: H0
            u0 = rng.uniform(size=n)
            _, p = kstest(u0, "uniform")
            if p < alpha: k_h0 += 1

            # KS: Alt Beta(1.05,1.05)
            u1 = beta_dist.rvs(1.05, 1.05, size=n, random_state=rng)
            _, p = kstest(u1, "uniform")
            if p < alpha: k_alt += 1

            # LB: H0 (iid normal)
            z0 = rng.standard_normal(size=n)
            if fast_ljungbox_p(z0, lag) < alpha: l_h0 += 1

            # LB: Alt AR(1, phi=0.03)
            z1 = np.zeros(n)
            z1[0] = rng.standard_normal()
            eps = rng.standard_normal(size=n)
            for t in range(1, n):
                z1[t] = phi * z1[t - 1] + eps[t]
            if fast_ljungbox_p(z1, lag) < alpha: l_alt += 1

        ks_h0.append(k_h0 / n_sim);  ks_alt.append(k_alt / n_sim)
        lb_h0.append(l_h0 / n_sim);  lb_alt.append(l_alt / n_sim)
        print(f"    n={n:>7,}  KS(H0)={k_h0/n_sim:.3f} "
              f"KS(alt)={k_alt/n_sim:.3f}  "
              f"LB(H0)={l_h0/n_sim:.3f} LB(alt)={l_alt/n_sim:.3f}")

    n_arr = np.array(n_vals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Theoretical Rejection Rate vs Sample Size\n"
        "Both alternatives are practically negligible — detected only due to large n",
        fontsize=10, fontweight="bold",
    )

    for ax, h0, alt, title, annot, alt_label in [
        (ax1, ks_h0, ks_alt,
         "Kolmogorov–Smirnov Test",
         "Alternative: Beta(1.05, 1.05)\nKS distance ≈ 0.01 from Uniform",
         "Alt: Beta(1.05,1.05) [KS≈0.01]"),
        (ax2, lb_h0, lb_alt,
         f"Ljung–Box Test (lag={lag})",
         f"Alternative: AR(1, φ={phi})\nACF(1) ≈ {phi} — barely detectable",
         f"Alt: AR(1, φ={phi}) [ACF≈{phi}]"),
    ]:
        ax.plot(n_arr, h0, color=GREEN, marker="s", markersize=5,
                linestyle="--", label="H₀ (no misspecification)")
        ax.plot(n_arr, alt, color=RED, marker="o", markersize=5,
                label=alt_label)
        ax.axhline(alpha, color=GREY, linestyle=":", linewidth=1.0,
                   label=f"Nominal α = {alpha}")
        ax.axvline(50_000, color=AMBER, linestyle=":", linewidth=1.2,
                   label="run_009/010 (n≈52k)")
        ax.set_xscale("log")
        ax.set_xlabel("Sample size n (log scale)")
        ax.set_ylabel("Rejection rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.text(0.97, 0.45, annot,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=AMBER,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=AMBER, alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Figure 6: Model diagnostic positioning (applied to real data) ─────────────

def plot_model_positioning(out_path: Path) -> None:
    """
    All model classes positioned by sample size vs KS stat / |ACF lag-1|.

    Models WITH PIT (real data + sim positive control) — plotted on both panels.
    Models WITHOUT PIT (sim class, run_004b) — interval-only diagnostics;
    shown as hollow triangles on a labelled band at the bottom of each panel.

    Legend placed outside/below the axes. RED* relabelled AMBER.
    """

    # ── Models WITH PIT diagnostics ──────────────────────────────────────────
    # fmt: (label, n, ks_stat, acf_lag1, outcome, ks_xy_offset, acf_xy_offset)
    # offsets in display points (dx, dy)
    pit_models = [
        ("ENTSO-E Load\n(run_001)",      209_555, 0.1615, 0.926, "RED",
         (8, 4),   (8, 4)),
        ("PV Solar\n(run_002)",            4_287, 0.1028, 0.660, "RED",
         (-85, 6), (-85, 6)),
        ("Wind\n(run_003)",                9_000, 0.1057, 0.855, "RED",
         (8, -16), (8, -16)),
        # Sim positive control — synthetic uniform PIT (well-specified)
        ("Sim Price\n(run_004)",             365, 0.028,  0.022, "GREEN",
         (8, -16), (8, -16)),
        ("ENTSO-E Wind DE\n(run_009)",    51_933, 0.0083, 0.861, "AMBER",
         (8, -18), (-110, 4)),
        ("ENTSO-E Solar DE\n(run_010)",   51_933, 0.0258, 0.788, "AMBER",
         (8, 8),   (8, 8)),
    ]

    # ── Simulation misspecification (run_004b) — interval-only, no PIT ───────
    # coverage and Anfuso outcome known; KS/ACF not computed for this class
    nopit_models = [
        # (label, n, coverage, outcome)
        ("Sim — Var Inflation\n(price, run_004b)",  365, 0.553, "RED"),
        ("Sim — Var Inflation\n(temp, run_004b)",   365, 0.625, "RED"),
        ("Sim — Mean Bias\n(price, run_004b)",      365, 0.704, "RED"),
        ("Sim — Mean Bias\n(temp, run_004b)",       365, 0.704, "RED"),
        ("Sim — Heavy Tails\n(price, run_004b)",    365, 0.901, "GREEN"),
        ("Sim — Heavy Tails\n(temp, run_004b)",     365, 0.923, "YELLOW"),
    ]

    color_map  = {"RED": RED, "GREEN": GREEN, "AMBER": AMBER,
                  "YELLOW": "#CA8A04"}
    marker_map = {"RED": "X", "GREEN": "o", "AMBER": "D", "YELLOW": "^"}
    size_map   = {"RED": 130, "GREEN": 110, "AMBER": 120,  "YELLOW": 110}

    # ── Panel config ──────────────────────────────────────────────────────────
    panels = [
        ("ks",  KS_FLOOR,  "KS statistic",
         "PIT Uniformity — KS Statistic vs Sample Size",
         (-0.012, 0.225), KS_FLOOR + 0.005),
        ("acf", ACF_FLOOR, "|ACF lag-1|",
         "Serial Independence — |ACF lag-1| vs Sample Size",
         (-0.04,  1.12),  ACF_FLOOR + 0.02),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        "Diagnostic Positioning — All Model Classes\n"
        "Effect-size floors (dashed) separate AMBER (large-n sensitivity) "
        "from RED (genuine miscalibration)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ax, (key, floor, ylabel, title, ylim, floor_label_y) in zip(axes, panels):

        # ── Floor line & shaded zone ──────────────────────────────────────
        ax.axhline(floor, color=AMBER, linestyle="--", linewidth=1.5, zorder=3)
        ax.axhspan(ylim[0], floor, color=AMBER, alpha=0.07, zorder=1)
        ax.text(190, floor_label_y,
                "Below floor → WARN only (large-n sensitivity)",
                fontsize=7.5, color=AMBER, va="bottom")

        # ── No-PIT band: thin grey stripe near y = 0 ─────────────────────
        nopit_y = ylim[0] * 0.6   # sits at very bottom
        ax.axhspan(ylim[0], ylim[0] + abs(ylim[0]) * 0.5 + 0.003,
                   color="#F3F4F6", zorder=0)
        ax.text(190, ylim[0] + 0.001,
                "▼ Simulation class (run_004b): PIT not computed — interval-only",
                fontsize=7, color=GREY, va="bottom", style="italic")

        # Spread no-PIT models across n=365 with x-jitter for legibility
        n_nopit = len(nopit_models)
        x_positions = np.logspace(
            np.log10(200), np.log10(600), n_nopit
        )
        for (lbl, n, cov, out), xpos in zip(nopit_models, x_positions):
            c = color_map[out]
            m = marker_map[out]
            s = size_map[out]
            ypos = ylim[0] + 0.001
            ax.scatter(xpos, ypos, color=c, marker=m, s=s, zorder=5,
                       edgecolors="white", linewidths=0.8, alpha=0.85)
            short = lbl.split("\n")[0].replace("Sim — ", "")
            ax.annotate(
                short,
                (xpos, ypos),
                textcoords="offset points",
                xytext=(0, 10 if n_nopit % 2 == 0 else -14),
                fontsize=6.5, color=c, ha="center",
                rotation=30,
            )

        # ── PIT models ───────────────────────────────────────────────────
        for label, n, ks, acf, outcome, ks_off, acf_off in pit_models:
            val    = ks if key == "ks" else abs(acf)
            offset = ks_off if key == "ks" else acf_off
            c = color_map[outcome]
            m = marker_map[outcome]
            s = size_map[outcome]
            ax.scatter(n, val, color=c, marker=m, s=s, zorder=5,
                       edgecolors="white", linewidths=0.9)
            ax.annotate(
                label, (n, val),
                textcoords="offset points", xytext=offset,
                fontsize=7.5, color=c, va="center",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=4),
            )

        # ── Targeted callout on KS panel ──────────────────────────────────
        if key == "ks":
            ax.annotate(
                "KS=0.0083 < floor\n→ WARN (not FAIL)",
                xy=(51_933, 0.0083),
                xytext=(51_933 * 2.2, 0.035),
                fontsize=7, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8),
            )

        # ── Targeted callout on ACF panel ─────────────────────────────────
        if key == "acf":
            ax.annotate(
                "ACF=0.86 > floor\n→ FAIL (genuine autocorr.)\nrun_009 stays RED",
                xy=(51_933, 0.861),
                xytext=(3_000, 0.65),
                fontsize=7, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8),
            )

        ax.set_xscale("log")
        ax.set_xlabel("Sample size n (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Shared legend — placed below the figure outside the axes ─────────────
    legend_elements = [
        mpatches.Patch(color=RED,      label="RED — genuine miscalibration (above effect-size floor)"),
        mpatches.Patch(color=AMBER,    label="AMBER — large-n sensitivity (below effect-size floor)"),
        mpatches.Patch(color=GREEN,    label="GREEN — well-calibrated (positive control)"),
        mpatches.Patch(color="#CA8A04",label="YELLOW — borderline (heavy-tail misspec)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=GREY,
                   markersize=8, label="▼ Simulation class: interval-only, no PIT"),
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=8.5,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.10),
        frameon=True,
        framealpha=0.95,
        edgecolor=GREY,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  PIT Diagnostic Figures (4-panel)")
    print("=" * 60)

    datasets = [
        {
            "name":    "ENTSO-E Load (run_001)",
            "samples": DATA / "derived_full" / "entsoe_samples.npy",
            "y":       DATA / "derived_full" / "entsoe_y.npy",
            "out":     FIG  / "pit_diagnostics_entsoe.png",
        },
        {
            "name":    "PV Solar (run_002)",
            "samples": DATA / "derived_pv" / "pv_samples.npy",
            "y":       DATA / "derived_pv" / "pv_y.npy",
            "out":     FIG  / "pit_diagnostics_pv.png",
        },
        {
            "name":    "Wind (run_003)",
            "samples": DATA / "derived_wind" / "wind_samples.npy",
            "y":       DATA / "derived_wind" / "wind_y.npy",
            "out":     FIG  / "pit_diagnostics_wind.png",
        },
    ]

    for ds in datasets:
        sp, yp = ds["samples"], ds["y"]
        if not sp.exists():
            print(f"  SKIP {ds['name']}: {sp.name} not found")
            continue
        print(f"\n  Processing: {ds['name']}")
        u = load_pit(sp, yp)
        plot_pit_diagnostics(u, ds["name"], ds["out"])

    # Simulation positive control — synthetic Uniform PIT (well-specified by design)
    print(f"\n  Processing: Synthetic Simulation (run_004, positive control)")
    u_sim = np.random.default_rng(42).uniform(size=365)
    plot_pit_diagnostics(
        u_sim,
        "Synthetic Simulation — Price (run_004, positive control)",
        FIG / "pit_diagnostics_sim.png",
    )

    print("\n" + "=" * 60)
    print("  Figure 5: Theoretical Power vs n")
    print("=" * 60)
    plot_power_vs_n(FIG / "power_vs_n.png")

    print("\n" + "=" * 60)
    print("  Figure 6: Model Diagnostic Positioning (real data)")
    print("=" * 60)
    plot_model_positioning(FIG / "model_diagnostic_positioning.png")

    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
