# Unified Probabilistic Validation Framework
## End-to-End Dataset Onboarding Guide

---

## PART 0 — ONE-TIME SETUP (do this once per machine)

### Step 1 — Prerequisites

Before starting, confirm these are installed:

```bash
# Check Python version (need 3.10 or higher)
python --version

# Check Git
git --version
```

If either is missing, install Python from python.org and Git from git-scm.com.

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/LeJ7-commits/unified-probabilistic-validation
cd unified-probabilistic-validation
```

---

### Step 3 — Create a virtual environment and install dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

You should see packages installing. This takes 2–3 minutes on first run.

---

### Step 4 — Verify installation

```bash
# Run the test suite — all 451+ tests should pass
python -m pytest tests/ -q

# Confirm pipeline is wired correctly
python run_all.py --dry-run
```

Expected dry-run output: 18 stages listed with no errors. If you see this, setup is complete.

---

### Step 5 — (Optional) Enable AI narratives

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Mac/Linux
export ANTHROPIC_API_KEY=your-key-here
```

Without this, stub narratives are written automatically and the pipeline still runs fully.

---

## PART 1 — SIMULATION MODEL CLASS

### What your CSV needs to look like

For simulation models the framework needs the raw simulation paths. The build script generates these from the DGP parameters directly. If you are bringing an external simulation engine, you need to export its paths in the following format:

```
as_of_date, path_1, path_2, ..., path_5000, realised_value
2020-01-01, 48.2, 51.3, ..., 49.8, 50.1
2020-01-02, 47.9, 52.1, ..., 48.6, 49.3
```

One row per as-of date. Each path column is one simulated value at horizon h=1.

---

### Step 1 — Prepare your data file

Place your CSV in the `data/` directory:

```bash
# Example: you have a gas price simulation export
copy your_simulation.csv data/gas_price_simulation.csv
```

---

### Step 2 — Create a build script

Copy the nearest existing build script and modify it:

```bash
copy scripts/build_simulation_derived.py scripts/build_my_simulation_derived.py
```

Open `scripts/build_my_simulation_derived.py` and update these lines at the top:

```python
# Change these to match your data
SERIES_NAME = "gas_price"          # your commodity name
N_DAYS      = 365                   # number of as-of dates in your CSV
N_PATHS     = 5000                  # number of simulation paths
ALPHA       = 0.10                  # 0.10 = 90% interval target
```

If you are using the built-in Gaussian DGP, update the parameters to match your commodity:

```python
BASE_VALUE  = 40.0    # typical price level
SIGMA       = 8.0     # standard deviation
```

---

### Step 3 — Run the build script

```bash
python scripts/build_my_simulation_derived.py
```

Expected output:
```
[gas_price] n_days=365, empirical_coverage_90=0.9041, samples_shape=(365, 5000)
```

This creates a directory `data/derived_simulation_gas_price/` with:
- `gas_price_y.npy` — realised values
- `gas_price_yhat.npy` — point forecast (mean of paths)
- `gas_price_lo_base_90.npy` — lower interval bound (5th percentile of paths)
- `gas_price_hi_base_90.npy` — upper interval bound (95th percentile of paths)
- `gas_price_samples.npy` — full sample array (365 × 5000) for PIT diagnostics
- `metadata.json` — build parameters and coverage check

---

### Step 4 — Create an experiment run script

```bash
copy experiments/run_011_sim_elec_price.py experiments/run_020_sim_gas_price.py
```

Open `experiments/run_020_sim_gas_price.py` and change these lines:

```python
SERIES = "gas_price"                          # match your directory name
DATA_DIR = REPO / "data" / "derived_simulation_gas_price"
OUT_DIR  = REPO / "experiments" / "run_020_sim_gas_price"
```

---

### Step 5 — Run the experiment

```bash
python experiments/run_020_sim_gas_price.py
```

Expected output:
```
============================================================
  Simulation — gas_price (run_020)
============================================================
  Governance decision : GREEN
  Reason codes        : ['all_clear']
```

For a well-specified positive control (simulation paths match the DGP), you should see GREEN. If you introduced deliberate misspecification, you will see RED with specific reason codes.

---

### Step 6 — View results

```bash
# Full diagnostic metrics
type experiments\run_020_sim_gas_price\full_sample_metrics.json

# Governance decision with reason codes
type experiments\run_020_sim_gas_price\governance_decision.json
```

**Step 6b — Generate PIT diagnostic figures (simulation)**

Unlike short-term and long-term models where figures are generated automatically by the run script, simulation PIT figures are generated by the plot script:

```bash
python scripts/plot_pit_diagnostics.py
```

This generates the following figures in `figures/`:

```
figures/
├── pit_diagnostics_sim_price.png       ← Sim Price (run_004)
├── pit_diagnostics_sim_temp.png        ← Sim Temp (run_004)
├── pit_diagnostics_sim_elec_price.png  ← Elec Price (run_011)
├── pit_diagnostics_sim_nat_gas.png     ← Natural Gas (run_012)
├── pit_diagnostics_sim_carbon.png      ← Carbon (run_013)
├── pit_diagnostics_sim.png             ← Summary grid (all 5 series)
├── pit_diagnostics_misspec_price_vi.png ← Variance inflation
├── pit_diagnostics_misspec_price_mb.png ← Mean bias
└── pit_diagnostics_misspec_price_ht.png ← Heavy tails
```

For a well-specified simulation positive control, all four panels should look approximately uniform — flat histogram, ACF within confidence bands, no trend in the time series, Q-Q close to the diagonal.

To add your own simulation series to the plot script, open `scripts/plot_pit_diagnostics.py` and add an entry to the `sim_specs` list:

```python
(
    "Sim Gas Price (run_020)",
    DATA / "derived_simulation_gas_price" / "gas_price_samples.npy",
    DATA / "derived_simulation_gas_price" / "gas_price_y.npy",
    FIG  / "pit_diagnostics_sim_gas_price.png",
),
```

Then re-run:

```bash
python scripts/plot_pit_diagnostics.py
```

The new series will be added to both the individual 4-panel figure and the summary grid.

---

## PART 2 — SHORT-TERM FORECAST MODEL CLASS

### What your CSV needs to look like

```
timestamp,          y,        y_hat
2020-01-01 00:00,  45231,    44800
2020-01-01 00:15,  44890,    44500
2020-01-01 00:30,  44512,    44200
...
```

Required columns:
- **timestamp** — datetime of each observation (any standard format)
- **y** — realised value (what actually happened)
- **y_hat** — model forecast (what the model predicted)

Both columns must be in the same units. The framework is unit-agnostic — MW, MWh, €/MWh all work.

---

### Step 1 — Prepare your data file

```bash
copy your_forecast.csv data/my_load_forecast.csv
```

---

### Step 2 — Create a build script

```bash
copy scripts/build_entsoe_derived.py scripts/build_my_forecast_derived.py
```

Open `scripts/build_my_forecast_derived.py` and update:

```python
# Path to your CSV
INPUT_CSV   = DATA_ROOT / "my_load_forecast.csv"

# Column names in your CSV
COL_TIMESTAMP = "timestamp"
COL_Y         = "y"
COL_Y_HAT     = "y_hat"

# Output directory name
OUT_DIR_NAME  = "derived_my_forecast"

# Rolling window parameters
# W = how many historical residuals to use for interval construction
# Larger W = more stable intervals, less adaptive to regime changes
W_GLOBAL      = 672    # 7 days of hourly data (7 × 24 × 4 for quarter-hourly)
ALPHA         = 0.10   # 90% interval

# Bucket conditioning
# 4 buckets = night/morning/afternoon/evening (coarse, stable)
# 24 buckets = hour-of-day (fine, requires more data per bucket)
N_BUCKETS     = 4
```

---

### Step 3 — Run the build script

```bash
python scripts/build_my_forecast_derived.py
```

Expected output:
```
Building derived artifacts for my_load_forecast...
  n_raw=209555, n_eval=208883 (672 warmup rows consumed)
  empirical_coverage_90=0.8823
  Artifacts written to data/derived_my_forecast/
Done.
```

The empirical coverage tells you how well the model's intervals were calibrated before any diagnostic adjustment.

---

### Step 4 — Create an experiment run script

```bash
copy experiments/run_001_entsoe.py experiments/run_021_my_forecast.py
```

Open `experiments/run_021_my_forecast.py` and update:

```python
DATA_DIR  = REPO / "data" / "derived_my_forecast"
OUT_DIR   = REPO / "experiments" / "run_021_my_forecast"
MODEL_ID  = "my_load_forecast"

# Rolling window parameters for diagnostics
ROLLING_WINDOW = 250   # observations per rolling window
ROLLING_STEP   = 50    # step between windows (overlapping)
```

---

### Step 5 — Run the experiment

```bash
python experiments/run_021_my_forecast.py
```

Expected output for a typical operational forecast:
```
============================================================
  ENTSO-E-style diagnostics: my_load_forecast
============================================================
  Governance decision : RED
  Reason codes        : ['PIT_uniformity_fail', 'ACF_dependence_fail']
  Empirical coverage  : 88.2%
```

RED is the common outcome for real operational forecasting models — this does not mean the model is unusable. It means the distributional shape requires attention beyond coverage metrics alone.

---

### Step 6 — Inspect the PIT diagnostic figure

The build and run scripts generate a 4-panel PIT diagnostic figure automatically. Find it at:

```bash
# Open the figure (Windows)
start figures\pit_diagnostics_my_forecast.png
```

The four panels show:
- **PIT histogram** — should be flat/uniform for a calibrated model
- **ACF of transformed PIT** — should show no significant bars for an independent model
- **PIT time series** — should show no visible trends or regime changes
- **Q-Q plot** — should follow the diagonal for a calibrated model

---

### Step 7 — View the governance narrative

```bash
# Technical narrative for risk officers
type experiments\run_021_my_forecast\narrative_technical.md

# Plain language narrative for management
type experiments\run_021_my_forecast\narrative_plain.md
```

---

## PART 3 — LONG-TERM RENEWABLE FORECAST MODEL CLASS

### Special consideration: nighttime exclusion

For PV solar, hours where both simulation and actuals are structurally zero (nighttime) must be excluded from PIT evaluation. The framework handles this automatically via a nighttime threshold filter. For wind, no exclusion is needed.

### What your CSV needs to look like

```
Datetime,              Simulation,  Actuals
2013-01-01 00:00:00,   0.0,         0.0
2013-01-01 01:00:00,   0.0,         0.0
2013-01-06 08:00:00,   245.3,       312.1
2013-01-06 09:00:00,   412.7,       389.4
...
```

Required columns:
- **Datetime** — hourly timestamps
- **Simulation** — model output (expected generation in MW or capacity factor)
- **Actuals** — realised generation

---

### Step 1 — Prepare your data file

```bash
# For PV
copy your_pv_data.csv data/my_pv_generation.csv

# For wind
copy your_wind_data.csv data/my_wind_generation.csv
```

---

### Step 2 — Create a build script

```bash
copy scripts/build_renewables_derived.py scripts/build_my_renewables_derived.py
```

Open `scripts/build_my_renewables_derived.py` and update:

```python
# For PV solar
PV_CSV          = DATA_ROOT / "my_pv_generation.csv"
PV_OUT_DIR      = "derived_my_pv"
NIGHTTIME_THRESHOLD = 1e-9    # rows below this in both columns are excluded

# For wind (no nighttime exclusion)
WIND_CSV        = DATA_ROOT / "my_wind_generation.csv"
WIND_OUT_DIR    = "derived_my_wind"

# Rolling window: 720 = 30 days of hourly observations
# 24 buckets = one bucket per hour-of-day
W               = 720
N_BUCKETS       = 24
ALPHA           = 0.10
```

---

### Step 3 — Run the build script

```bash
python scripts/build_my_renewables_derived.py
```

Expected output:
```
Processing PV solar...
  Raw rows: 26280
  Nighttime rows excluded: 10036
  Warmup rows consumed: 11957
  Evaluable observations: 4287
  Empirical coverage: 0.9137
  Artifacts written to data/derived_my_pv/

Processing wind...
  Raw rows: 26280
  Warmup rows consumed: 17280
  Evaluable observations: 9000
  Empirical coverage: 0.8862
  Artifacts written to data/derived_my_wind/
```

---

### Step 4 — Create experiment run scripts

```bash
# For PV
copy experiments/run_002_pv.py experiments/run_022_my_pv.py

# For wind
copy experiments/run_003_wind.py experiments/run_023_my_wind.py
```

In each script update the DATA_DIR, OUT_DIR, and MODEL_ID to match your directory names.

---

### Step 5 — Run the experiments

```bash
python experiments/run_022_my_pv.py
python experiments/run_023_my_wind.py
```

---

## PART 4 — RUNNING THE FULL PIPELINE

Once all build scripts and run scripts are in place, add your new stages to `run_all.py`. Open `run_all.py` and add entries to the `STAGES` list following the existing pattern:

```python
{
    "id":          20,
    "name":        "My simulation — gas price",
    "scripts":     ["experiments/run_020_sim_gas_price.py"],
    "optional":    False,
    "description": "Simulation positive control — gas price DGP",
},
{
    "id":          21,
    "name":        "My short-term forecast",
    "scripts":     ["experiments/run_021_my_forecast.py"],
    "optional":    False,
    "description": "Point forecast diagnostics — my load forecast",
},
{
    "id":          22,
    "name":        "My PV generation",
    "scripts":     ["experiments/run_022_my_pv.py"],
    "optional":    False,
    "description": "Long-term PV generation diagnostics",
},
```

Also add your build script to Stage 1:

```python
# In Stage 1 scripts list, add:
"scripts/build_my_simulation_derived.py",
"scripts/build_my_forecast_derived.py",
"scripts/build_my_renewables_derived.py",
```

---

### Dry run — preview all stages without executing

```bash
python run_all.py --dry-run
```

This lists every stage that will run. Verify your new stages appear correctly before committing.

---

### Run only the build stage

```bash
python run_all.py --stages 1
```

---

### Run only your new experiments

```bash
python run_all.py --stages 20,21,22
```

---

### Run the full pipeline end to end

```bash
python run_all.py
```

This runs all 18 original stages plus your new ones. Total runtime depends on dataset size — the original 18 stages take approximately 25–40 minutes on a standard laptop.

---

### Run only specific stages and skip build

```bash
python run_all.py --stages 2,3,4,20,21,22 --skip-build
```

Use `--skip-build` when derived artifacts already exist and you only want to re-run the diagnostic experiments without rebuilding from raw CSV.

---

## PART 5 — READING THE OUTPUTS

After the pipeline completes, each experiment directory contains:

```
experiments/run_020_sim_gas_price/
├── full_sample_metrics.json      ← all computed statistics
├── governance_decision.json      ← final label + reason codes + provenance
├── anfuso_full_sample.json       ← interval backtesting results
├── rolling_overlapping.csv       ← per-window rolling diagnostics
├── rolling_non_overlapping.csv   ← independent window snapshots
├── narrative_technical.md        ← AI narrative for risk officers
├── narrative_plain.md            ← AI narrative for management
└── report_card_label_bands.png   ← rolling governance classification figure
```

### Reading the governance decision

```bash
type experiments\run_020_sim_gas_price\governance_decision.json
```

Key fields:
- `"final_label"` — GREEN, YELLOW, or RED
- `"reason_codes"` — what triggered the classification
- `"metric_snapshot"` — all computed numbers
- `"provenance"` — which diagnostics were computed, which were skipped and why

### Reason codes and their meaning

| Reason code | What it means |
|-------------|--------------|
| `all_clear` | Model passed all diagnostic layers |
| `PIT_uniformity_fail` | Distributional shape does not match calibrated model |
| `ACF_dependence_fail` | Forecast errors are serially correlated |
| `undercoverage` | Intervals are too narrow — missing target by more than tolerance |
| `coverage_warn` | Intervals are slightly off-target — monitor but no action required |

---

## QUICK REFERENCE CARD

```
New dataset → full governance classification in 6 steps:

1. Place CSV in data/
2. Copy nearest build script → update paths and parameters
3. python scripts/build_my_X_derived.py
4. Copy nearest run script → update paths
5. python experiments/run_0XX_my_X.py
6. python run_all.py --dry-run  →  python run_all.py

Model class guide:
  Simulation     → copy build_simulation_derived.py + run_011_sim_elec_price.py
  Short-term     → copy build_entsoe_derived.py    + run_001_entsoe.py
  Long-term      → copy build_renewables_derived.py + run_002_pv.py or run_003_wind.py
```
