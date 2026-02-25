# Diagnostics layer (03)

This package orchestrates reliability evaluation over:

- full historical period (primary assessment)
- fixed-length rolling windows (complementary stress test)

Core metric implementations live in:

- `src/calibration/pit.py`
- `src/calibration/diagnostics.py`
- `src/scoring/crps.py`

