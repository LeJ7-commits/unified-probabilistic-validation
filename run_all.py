"""
run_all.py
===========
Full pipeline orchestrator for the Unified Probabilistic Validation Framework.

WHAT THIS DOES:
  Runs all 9 experiment scripts in sequence, then generates AI governance
  narratives for each dataset. A non-Python user only needs to:
    1. Place their data files in the correct data/ directories
    2. Run: python run_all.py
    3. Collect outputs from experiments/

PIPELINE STAGES:
  Stage 1  — Build derived artifacts (data preprocessing)
  Stage 2  — Run 001: ENTSO-E load diagnostics
  Stage 3  — Run 002: PV solar diagnostics
  Stage 4  — Run 003: Wind diagnostics
  Stage 5  — Run 004: Simulation diagnostics (price + temp)
  Stage 6  — Run 004b: Misspecification scenarios
  Stage 7  — Run 005: Multivariate PV+Wind joint diagnostics
  Stage 8  — Run 006: VaR sensitivity analysis
  Stage 9  — Run 007: Transition metrics
  Stage 10 — Run 008: Report cards + AI narratives

OUTPUTS:
  experiments/run_001_entsoe/          ENTSO-E results + governance decision
  experiments/run_002_pv/              PV Solar results + governance decision
  experiments/run_003_wind/            Wind results + governance decision
  experiments/run_004_simulation_*/    Simulation results + governance decision
  experiments/run_004b_simulation_*/   Misspecification results
  experiments/run_005_multivariate/    Multivariate diagnostics
  experiments/run_006_var_sensitivity/ VaR sensitivity
  experiments/run_007_transition_metrics/ Transition stability
  Each run dir also contains:
    governance_decision.json           Structured governance output
    narrative_technical.md             AI technical narrative
    narrative_plain.md                 AI plain language narrative
    narrative_combined.md              Both narratives combined

REQUIREMENTS:
  - Python 3.10+
  - All dependencies in requirements.txt installed
  - Data files in place (see DATA PREREQUISITES below)
  - ANTHROPIC_API_KEY environment variable (optional — enables AI narratives)

DATA PREREQUISITES:
  data/derived_full/          Built by build_entsoe_derived.py
  data/derived_pv/            Built by build_renewables_derived.py
  data/derived_wind/          Built by build_renewables_derived.py
  data/derived_simulation_*/  Built by build_simulation_derived.py

ENVIRONMENT VARIABLES:
  ANTHROPIC_API_KEY   Anthropic API key for AI narrative generation
                      Get one at: https://console.anthropic.com/
                      If not set, stub narratives are written instead.

USAGE:
  # Full pipeline
  python run_all.py

  # Skip build stage (if derived data already exists)
  python run_all.py --skip-build

  # Run specific stages only
  python run_all.py --stages 1,2,3

  # Dry run (show what would run without executing)
  python run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id":          1,
        "name":        "Build derived artifacts",
        "scripts":     [
            "scripts/build_entsoe_derived.py",
            "scripts/build_renewables_derived.py",
            "scripts/build_simulation_derived.py",
        ],
        "optional":    True,   # can skip if derived data already exists
        "description": "Preprocesses raw CSV data into derived .npy artifacts",
    },
    {
        "id":      2,
        "name":    "ENTSO-E load diagnostics",
        "scripts": ["experiments/run_001_entsoe.py"],
        "optional": False,
        "description": "PIT + coverage + Anfuso + DecisionEngine for ENTSO-E load",
    },
    {
        "id":      3,
        "name":    "PV Solar diagnostics",
        "scripts": ["experiments/run_002_pv.py"],
        "optional": False,
        "description": "PIT + coverage + Anfuso + DecisionEngine for PV solar",
    },
    {
        "id":      4,
        "name":    "Wind diagnostics",
        "scripts": ["experiments/run_003_wind.py"],
        "optional": False,
        "description": "PIT + coverage + Anfuso + DecisionEngine for onshore wind",
    },
    {
        "id":      5,
        "name":    "Simulation diagnostics",
        "scripts": ["experiments/run_004_simulation.py"],
        "optional": False,
        "description": "Well-specified positive control (price + temp)",
    },
    {
        "id":      6,
        "name":    "Misspecification scenarios",
        "scripts": ["experiments/run_004b_simulation_misspec.py"],
        "optional": False,
        "description": "Three deliberate misspecification scenarios",
    },
    {
        "id":      7,
        "name":    "Multivariate diagnostics",
        "scripts": ["experiments/run_005_multivariate.py"],
        "optional": False,
        "description": "Joint PV+Wind PIT dependence and bivariate Energy Score",
    },
    {
        "id":      8,
        "name":    "VaR sensitivity analysis",
        "scripts": ["experiments/run_006_var_sensitivity.py"],
        "optional": False,
        "description": "Capital multiplier distortion and reserve sizing error",
    },
    {
        "id":      9,
        "name":    "Transition metrics",
        "scripts": ["experiments/run_007_transition_metrics.py"],
        "optional": False,
        "description": "Rolling governance label stability and transition matrices",
    },
    {
        "id":      10,
        "name":    "Report cards + AI narratives",
        "scripts": ["experiments/run_008_report_cards.py"],
        "optional": False,
        "description": "Governance report cards and AI-generated narratives",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    print("=" * 70)
    print("  Unified Probabilistic Validation Framework — Full Pipeline")
    print("=" * 70)
    print()


def _print_stage(stage: dict, dry_run: bool = False) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Stage {stage['id']}: {stage['name']}")
    print(f"  {stage['description']}")
    for s in stage["scripts"]:
        print(f"  → {s}")


def _run_script(script: str) -> tuple[bool, float]:
    """Run a Python script. Returns (success, elapsed_seconds)."""
    path = REPO_ROOT / script
    if not path.exists():
        print(f"    [WARN] Script not found: {script} — skipping")
        return True, 0.0

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def _check_api_key() -> None:
    import os
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        print("  ANTHROPIC_API_KEY: found — AI narratives will be generated")
    else:
        print("  ANTHROPIC_API_KEY: not set")
        print("  AI narratives will be replaced with stubs.")
        print("  To enable: set ANTHROPIC_API_KEY=<your-key>")
        print("  Get a key at: https://console.anthropic.com/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Unified Probabilistic Validation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Stage 1 (build derived artifacts). "
             "Use if derived data already exists.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stage IDs to run (e.g. '2,3,10'). "
             "Runs all stages if not specified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing anything.",
    )
    args = parser.parse_args()

    _print_header()

    # Determine which stages to run
    if args.stages:
        requested = {int(s.strip()) for s in args.stages.split(",")}
        stages_to_run = [s for s in STAGES if s["id"] in requested]
    else:
        stages_to_run = list(STAGES)

    if args.skip_build:
        stages_to_run = [s for s in stages_to_run if s["id"] != 1]

    print(f"Stages to run: {[s['id'] for s in stages_to_run]}")
    _check_api_key()

    if args.dry_run:
        print("\n--- DRY RUN MODE — no scripts will be executed ---\n")
        for stage in stages_to_run:
            _print_stage(stage, dry_run=True)
        print("\nDry run complete.")
        return

    # Execute pipeline
    results: list[dict] = []
    pipeline_start = time.time()

    for stage in stages_to_run:
        _print_stage(stage)
        stage_start = time.time()
        stage_ok    = True

        for script in stage["scripts"]:
            ok, elapsed = _run_script(script)
            status = "OK" if ok else "FAILED"
            print(f"    [{status}] {script}  ({elapsed:.1f}s)")
            if not ok:
                stage_ok = False

        stage_elapsed = time.time() - stage_start
        results.append({
            "stage":   stage["id"],
            "name":    stage["name"],
            "success": stage_ok,
            "elapsed": stage_elapsed,
        })

        if not stage_ok and not stage.get("optional", False):
            print(f"\n[ABORT] Stage {stage['id']} failed. "
                  "Fix the error above and re-run.")
            break

    # Summary
    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} Stage {r['stage']:2d}: {r['name']:<40} ({r['elapsed']:.1f}s)")

    n_ok     = sum(1 for r in results if r["success"])
    n_total  = len(results)
    print(f"\n  {n_ok}/{n_total} stages completed successfully in {total_elapsed:.1f}s")

    if n_ok == n_total:
        print("\n  All outputs written to experiments/")
        print("  Each run directory contains:")
        print("    governance_decision.json    Structured governance output")
        print("    narrative_technical.md      AI technical narrative")
        print("    narrative_plain.md          AI plain language narrative")
        print("    narrative_combined.md       Both narratives combined")
    else:
        print("\n  Pipeline incomplete. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
