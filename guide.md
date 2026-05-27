# Unified Probabilistic Validation Framework
## End-to-End Dataset Onboarding Guide

---

## PART 0 — ONE-TIME SETUP

Do this once per machine before anything else.

### Step 1 — Prerequisites

```bash
# Confirm Python 3.10 or higher
python --version

# Confirm Git
git --version
```

If either is missing: Python from python.org, Git from git-scm.com.

---

### Step 2 — Clone and install

```bash
git clone https://github.com/LeJ7-commits/unified-probabilistic-validation
cd unified-probabilistic-validation

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3 — Verify

```bash
# All 451+ tests should pass
python -m pytest tests/ -q

# Pipeline should list 18 stages with no errors
python run_all.py --dry-run
```

---

### Step 4 — (Optional) Enable AI narratives

```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Mac/Linux
export ANTHROPIC_API_KEY=your-key-here
```

Without this, stub narratives are written automatically and the pipeline still runs fully.

---

## PART 1 — ONBOARDING A SINGLE NEW DATASET

Choose the section that matches your model type. Each section is self-contained — complete it end to end before moving to another model class.

---

## 1A — SIMULATION MODEL

### What this model class is

A Monte Carlo simulation engine that produces many possible future paths rather than a single forecast. Examples: commodity price simulation, temperature scenario generation, energy generation scenario engines.

### What your CSV needs to look like

```
as_of_date,  path_1,  path_2,  ...,  path_5000,  realised_value
2020-01-01,  48.2,    51.3,    ...,  49.8,        50.1
2020-01-02,  47.9,    52.1,    ...,  48.6,        49.3
```

One row per as-of date. Each path column is one simulated value at forecast horizon h=1.

---

### Step 1 — Place your data

```bash
copy your_simulation.csv data/my_commodity_simulation.csv
```

---

### Step 2 — Create and configure the build script

```bash
copy scripts/build_simulation_derived.py scripts/build_my_commodity_derived.py
```

Open `scripts/build_my_commodity_derived.py` and update:

```python
SERIES_NAME = "my_commodity"   # used as file prefix throughout
BASE_VALUE  = 55.0             # typical price level of your commodity
SIGMA       = 12.0             # standard deviation
N_DAYS      = 365              # number of as-of dates in your CSV
N_PATHS     = 5000             # number of simulation paths
ALPHA       = 0.10             # 0.10 = 90% interval target
```

---

### Step 3 — Build derived artifacts

```bash
python scripts/build_my_commodity_derived.py
```

Expected output:
```
[my_commodity] n_days=365, empirical_coverage_90=0.9041, samples_shape=(365, 5000)
```

Creates `data/derived_simulation_my_commodity/` containing:
- `my_commodity_y.npy` — realised values
- `my_commodity_yhat.npy` — point forecast (mean of paths)
- `my_commodity_lo_base_90.npy` — lower interval bound
- `my_commodity_hi_base_90.npy` — upper interval bound
- `my_commodity_samples.npy` — full sample array (365 × 5000) for PIT diagnostics
- `metadata.json` — build parameters and coverage check

---

### Step 4 — Create and configure the run script

```bash
copy experiments/run_011_sim_elec_price.py experiments/run_050_my_commodity.py
```

Open `experiments/run_050_my_commodity.py` and update:

```python
SERIES   = "my_commodity"
DATA_DIR = REPO / "data" / "derived_simulation_my_commodity"
OUT_DIR  = REPO / "experiments" / "run_050_my_commodity"
```

---

### Step 5 — Run diagnostics

```bash
python experiments/run_050_my_commodity.py
```

Expected output for a well-specified model:
```
============================================================
  Simulation — my_commodity (run_050)
============================================================
  Governance decision : GREEN
  Reason codes        : ['all_clear']
```

---

### Step 6 — Generate PIT diagnostic figures

```bash
python scripts/plot_pit_diagnostics.py
```

To include your new series, open `scripts/plot_pit_diagnostics.py` and add to the `sim_specs` list:

```python
(
    "Sim My Commodity (run_050)",
    DATA / "derived_simulation_my_commodity" / "my_commodity_samples.npy",
    DATA / "derived_simulation_my_commodity" / "my_commodity_y.npy",
    FIG  / "pit_diagnostics_sim_my_commodity.png",
),
```

Re-run:

```bash
python scripts/plot_pit_diagnostics.py
```

For a well-specified positive control all four panels should look approximately uniform — flat histogram, ACF within confidence bands, no trend in the time series, Q-Q close to the diagonal.

---

### Step 7 — Read results

```bash
# Governance decision
type experiments\run_050_my_commodity\governance_decision.json

# Plain language narrative
type experiments\run_050_my_commodity\narrative_plain.md

# Technical narrative
type experiments\run_050_my_commodity\narrative_technical.md
```

---

## 1B — SHORT-TERM FORECAST MODEL

### What this model class is

An operational forecasting model that produces a single point forecast for each time step at sub-daily resolution. Examples: day-ahead electricity load forecast, day-ahead price forecast, intraday balancing forecast. The framework reconstructs the predictive distribution from historical forecast errors.

### What your CSV needs to look like

```
timestamp,          y,      y_hat
2024-01-01 00:00,  45231,  44800
2024-01-01 01:00,  44890,  44500
2024-01-01 02:00,  44512,  44200
```

- **timestamp** — datetime of each observation
- **y** — realised value
- **y_hat** — model point forecast

Both columns must be in the same units. The framework is unit-agnostic — MW, MWh, €/MWh all work.

---

### Step 1 — Place your data

```bash
copy your_forecast.csv data/my_load_forecast.csv
```

---

### Step 2 — Create and configure the build script

```bash
copy scripts/build_entsoe_derived.py scripts/build_my_load_derived.py
```

Open `scripts/build_my_load_derived.py` and update:

```python
INPUT_CSV     = DATA_ROOT / "my_load_forecast.csv"
OUT_DIR_NAME  = "derived_my_load"

# Column names — match your CSV exactly
COL_TIMESTAMP = "timestamp"
COL_Y         = "y"
COL_Y_HAT     = "y_hat"

# Rolling window: how many historical residuals to use per interval
# 672 = 7 days × 24 hours × 4 quarters (quarter-hourly data)
# 168 = 7 days × 24 hours (hourly data)
W_GLOBAL      = 672

# Bucket conditioning:
# 4  = night/morning/afternoon/evening (coarse, recommended)
# 24 = hour-of-day (fine, requires more data per bucket)
N_BUCKETS     = 4

ALPHA         = 0.10   # 0.10 = 90% interval target
```

---

### Step 3 — Build derived artifacts

```bash
python scripts/build_my_load_derived.py
```

Expected output:
```
Building derived artifacts for my_load_forecast...
  n_raw=209555, n_eval=208883 (672 warmup rows consumed)
  empirical_coverage_90=0.8823
  Artifacts written to data/derived_my_load/
Done.
```

The empirical coverage tells you how well the model's intervals were calibrated before any diagnostic adjustment. It will rarely be exactly 90% — this is a finding, not an error.

---

### Step 4 — Create and configure the run script

```bash
copy experiments/run_001_entsoe.py experiments/run_051_my_load.py
```

Open `experiments/run_051_my_load.py` and update:

```python
DATA_DIR       = REPO / "data" / "derived_my_load"
OUT_DIR        = REPO / "experiments" / "run_051_my_load"
MODEL_ID       = "my_load_forecast"
ROLLING_WINDOW = 250   # observations per rolling window
ROLLING_STEP   = 50    # step between windows
```

---

### Step 5 — Run diagnostics

```bash
python experiments/run_051_my_load.py
```

Expected output for a typical operational forecast:
```
============================================================
  Short-term diagnostics: my_load_forecast
============================================================
  Governance decision : RED
  Reason codes        : ['PIT_uniformity_fail', 'ACF_dependence_fail']
  Empirical coverage  : 88.2%
```

RED is common for real operational forecasting models. It does not mean the model is unusable — it means the distributional shape requires attention beyond coverage metrics alone.

---

### Step 6 — Inspect the PIT diagnostic figure

```bash
# Windows
start figures\pit_diagnostics_my_forecast.png
```

The four panels show:
- **PIT histogram** — flat for a calibrated model
- **ACF of transformed PIT** — no significant bars for an independent model
- **PIT time series** — no visible trends or structural breaks
- **Q-Q plot** — follows the diagonal for a calibrated model

---

### Step 7 — Read results

```bash
# Governance decision
type experiments\run_051_my_load\governance_decision.json

# Plain language narrative
type experiments\run_051_my_load\narrative_plain.md

# Rolling classification figure
start experiments\run_051_my_load\report_card_label_bands.png
```

---

## 1C — LONG-TERM RENEWABLE MODEL

### What this model class is

A simulation-based model producing expected hourly generation output over a long horizon — typically multiple years. Examples: wind speed forecast converted via power curve to expected wind generation; irradiance model converted via efficiency curve to expected solar output.

### Special consideration: nighttime exclusion

For PV solar, hours where both simulation and actuals are structurally zero (nighttime) are excluded from PIT evaluation. The build script handles this automatically. For wind, no exclusion is needed.

### What your CSV needs to look like

```
Datetime,              Simulation,  Actuals
2020-01-01 00:00:00,   0.0,         0.0
2020-01-01 01:00:00,   0.0,         0.0
2020-01-06 08:00:00,   245.3,       312.1
2020-01-06 09:00:00,   412.7,       389.4
```

- **Datetime** — hourly timestamps
- **Simulation** — model output (MW or capacity factor)
- **Actuals** — realised generation

---

### Step 1 — Place your data

```bash
# PV solar
copy your_pv_data.csv data/my_pv_generation.csv

# Wind
copy your_wind_data.csv data/my_wind_generation.csv
```

---

### Step 2 — Create and configure the build script

```bash
copy scripts/build_renewables_derived.py scripts/build_my_renewables_derived.py
```

Open `scripts/build_my_renewables_derived.py` and update:

```python
# PV solar
PV_CSV              = DATA_ROOT / "my_pv_generation.csv"
PV_OUT_DIR          = "derived_my_pv"
NIGHTTIME_THRESHOLD = 1e-9   # rows below this in both columns are excluded

# Wind (no nighttime exclusion needed)
WIND_CSV            = DATA_ROOT / "my_wind_generation.csv"
WIND_OUT_DIR        = "derived_my_wind"

# Rolling window: 720 = 30 days × 24 hours
# 24 buckets = one per hour-of-day
W                   = 720
N_BUCKETS           = 24
ALPHA               = 0.10
```

---

### Step 3 — Build derived artifacts

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

### Step 4 — Create and configure run scripts

```bash
# PV
copy experiments/run_002_pv.py experiments/run_052_my_pv.py

# Wind
copy experiments/run_003_wind.py experiments/run_053_my_wind.py
```

In each script update:

```python
# run_052_my_pv.py
DATA_DIR = REPO / "data" / "derived_my_pv"
OUT_DIR  = REPO / "experiments" / "run_052_my_pv"
MODEL_ID = "my_pv_generation"

# run_053_my_wind.py
DATA_DIR = REPO / "data" / "derived_my_wind"
OUT_DIR  = REPO / "experiments" / "run_053_my_wind"
MODEL_ID = "my_wind_generation"
```

---

### Step 5 — Run diagnostics

```bash
python experiments/run_052_my_pv.py
python experiments/run_053_my_wind.py
```

---

### Step 6 — Read results

```bash
# Governance decisions
type experiments\run_052_my_pv\governance_decision.json
type experiments\run_053_my_wind\governance_decision.json

# Plain narratives
type experiments\run_052_my_pv\narrative_plain.md
type experiments\run_053_my_wind\narrative_plain.md

# Rolling classification figures
start experiments\run_052_my_pv\report_card_label_bands.png
start experiments\run_053_my_wind\report_card_label_bands.png
```

---

## PART 2 — RUNNING ALL THREE MODEL CLASSES TOGETHER

Once you have completed Part 1 for each model class individually, wire all new stages into `run_all.py` to run everything in a single command.

---

### Step 1 — Register your build scripts in Stage 1

Open `run_all.py` and find the Stage 1 entry. Add your build scripts to its `scripts` list:

```python
{
    "id":       1,
    "name":     "Build derived artifacts",
    "scripts":  [
        "scripts/build_entsoe_derived.py",
        "scripts/build_renewables_derived.py",
        "scripts/build_entsoe_renewables_derived.py",
        "scripts/build_simulation_derived.py",
        "scripts/build_simulation_extended_derived.py",
        # ADD YOURS HERE:
        "scripts/build_my_commodity_derived.py",
        "scripts/build_my_load_derived.py",
        "scripts/build_my_renewables_derived.py",
    ],
    ...
},
```

---

### Step 2 — Register your experiment scripts as new stages

Add three new stage entries after the existing stages:

```python
{
    "id":          19,
    "name":        "My simulation — commodity",
    "scripts":     ["experiments/run_050_my_commodity.py"],
    "optional":    False,
    "description": "Simulation positive control — my commodity DGP",
},
{
    "id":          20,
    "name":        "My short-term forecast — load",
    "scripts":     ["experiments/run_051_my_load.py"],
    "optional":    False,
    "description": "Point forecast diagnostics — my load forecast",
},
{
    "id":          21,
    "name":        "My long-term renewable — PV and wind",
    "scripts":     [
        "experiments/run_052_my_pv.py",
        "experiments/run_053_my_wind.py",
    ],
    "optional":    False,
    "description": "Long-term renewable generation diagnostics",
},
```

---

### Step 3 — Dry run to verify

```bash
python run_all.py --dry-run
```

You should now see 21 stages listed. Confirm your new stages appear with the correct names before running.

---

### Step 4 — Run everything

```bash
# First time — full build and run
python run_all.py

# Subsequent runs — skip build if data unchanged
python run_all.py --skip-build

# Run only your new stages
python run_all.py --stages 19,20,21 --skip-build
```

---

### Step 5 — Generate all PIT figures

```bash
python scripts/plot_pit_diagnostics.py
```

---

## PART 3 — READING THE OUTPUTS

After any run completes, each experiment directory contains:

```
experiments/run_050_my_commodity/
├── full_sample_metrics.json        ← all computed statistics
├── governance_decision.json        ← final label + reason codes + provenance
├── anfuso_full_sample.json         ← interval backtesting detail
├── rolling_overlapping.csv         ← per-window rolling diagnostics
├── rolling_non_overlapping.csv     ← independent window snapshots
├── narrative_technical.md          ← AI narrative for risk officers
├── narrative_plain.md              ← AI narrative for management
└── report_card_label_bands.png     ← rolling governance classification figure
```

### Governance decision — key fields

```bash
type experiments\run_050_my_commodity\governance_decision.json
```

| Field | What it contains |
|-------|-----------------|
| `final_label` | GREEN, YELLOW, or RED |
| `reason_codes` | What triggered the classification |
| `metric_snapshot` | All computed diagnostic statistics |
| `provenance` | Which diagnostics ran, which were skipped and why, timestamp |

### Reason codes

| Code | Meaning |
|------|---------|
| `all_clear` | Passed all diagnostic layers |
| `PIT_uniformity_fail` | Distributional shape departs from calibrated model |
| `ACF_dependence_fail` | Forecast errors are serially correlated |
| `undercoverage` | Intervals are too narrow — below tolerance |
| `coverage_warn` | Intervals are slightly off-target — monitor |

---

## QUICK REFERENCE

```
Single new dataset — 7 steps:

1.  Place CSV in data/
2.  copy <nearest build script> scripts/build_my_X_derived.py
3.  Edit: paths, column names, parameters
4.  python scripts/build_my_X_derived.py
5.  copy <nearest run script> experiments/run_0XX_my_X.py
6.  Edit: DATA_DIR, OUT_DIR, MODEL_ID
7.  python experiments/run_0XX_my_X.py

All three classes together:
    Add to Stage 1 scripts list + add stages 19/20/21 → python run_all.py

Template to copy by model class:
  Simulation   → build_simulation_derived.py   + run_011_sim_elec_price.py
  Short-term   → build_entsoe_derived.py       + run_001_entsoe.py
  Long-term    → build_renewables_derived.py   + run_002_pv.py / run_003_wind.py
```
