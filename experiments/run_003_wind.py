from __future__ import annotations

from pathlib import Path
import numpy as np

from src.diagnostics.run_policy import run_diagnostics_policy, write_run_artifacts


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir  = repo_root / "data" / "derived_wind"
    out_dir   = repo_root / "experiments" / "run_003_wind"

    alpha = 0.1  # 90% central interval

    # --- load structural artifacts ---
    y = np.load(data_dir / "wind_y.npy").astype(float)

    # Use asymmetric lo/hi directly — built from rolling empirical quantiles
    # in build_renewables_derived.py. Wind residuals exhibit heavy tails and
    # strong seasonal structure; a symmetric Gaussian reconstruction would
    # understate tail risk.
    lower = np.load(data_dir / "wind_lo_base_90.npy").astype(float)
    upper = np.load(data_dir / "wind_hi_base_90.npy").astype(float)

    if lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: y{y.shape}, lower{lower.shape}, upper{upper.shape}. "
            "All must be identical length. Re-run scripts/build_renewables_derived.py."
        )

    quantiles = {alpha / 2: lower, 1 - alpha / 2: upper}

    # samples optional
    samples_path = data_dir / "wind_samples.npy"
    samples = np.load(samples_path).astype(float) if samples_path.exists() else None

    # --- run policy diagnostics ---
    # model_class="long_term": rolling diagnostics are enabled.
    # rolling_window=720 (30 days of hourly data), rolling_step=168 (1 week).
    # Wind generates around the clock so no nighttime exclusion applies;
    # the full 26,280-row series is evaluated.
    run_out = run_diagnostics_policy(
        model_class="long_term",
        y_true=y,
        samples=samples,
        quantiles=quantiles,
        alpha=alpha,
        rolling_window=720,
        rolling_step=168,
        enable_rolling_for_long_term=True,
        lb_lags=(5, 10, 20),
        coverage_target=0.90,
    )

    # --- write artifacts ---
    paths = write_run_artifacts(
        out_dir=out_dir,
        run_output=run_out,
        alpha=alpha,
        y_true=y,
        quantiles=quantiles,
        coverage_target=0.90,
    )

    print("Run completed. Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


# --- DecisionEngine governance decision ---
    import json
    from src.diagnostics.diagnostics_input import Diagnostics_Input
    from src.governance.decision_engine import DecisionEngine
    from src.governance.risk_classification import RiskPolicy

    di = Diagnostics_Input(alpha=alpha)
    dro = di.from_arrays(
        y=y,
        t=np.arange(len(y)),
        model_id="wind_onshore",
        lo=lower,
        hi=upper,
        quantiles=quantiles,
        samples=samples,
    )
    engine = DecisionEngine(
        alpha=alpha,
        global_policy=RiskPolicy(coverage_target=0.90),
    )
    decision = engine.decide(dro)

    decision_path = out_dir / "governance_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nGovernance decision : {decision.final_label}")
    print(f"Reason codes        : {[str(rc) for rc in decision.reason_codes]}")
    print(f"Decision artifact   : {decision_path}")