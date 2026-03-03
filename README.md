# Running Data Science Portfolio

## Project Goal
Analyze personal running activity data and generate reproducible visual insights using Python.

## What This Repo Shows
- Data ingestion and feature engineering scripts for Garmin activity data
- Automated analysis pipeline with chart generation
- From-scratch K-Means implementation practice
- A separate FastAPI web app (in `../running-deploy`) for model serving

## Automated Analysis Highlights
The script [`Data/automated_analysis.py`](Data/automated_analysis.py) reads `Data/activity_fact.csv` and creates:
- Summary markdown report
- Descriptive statistics CSV
- Distance distribution plot
- Distance vs duration plot
- Monthly mileage trend plot
- Heart-rate zone totals plot

Latest generated outputs:
- [`Data/reports/analysis_summary.md`](Data/reports/analysis_summary.md)
- [`Data/reports/descriptive_stats.csv`](Data/reports/descriptive_stats.csv)
- [`Data/reports/distance_distribution.png`](Data/reports/distance_distribution.png)
- [`Data/reports/distance_vs_duration.png`](Data/reports/distance_vs_duration.png)
- [`Data/reports/monthly_distance.png`](Data/reports/monthly_distance.png)
- [`Data/reports/hr_zone_totals.png`](Data/reports/hr_zone_totals.png)

## Quick Start
### Prerequisites
- Python 3.10+
- `uv` (recommended)

### Setup
```bash
uv sync
```

### Run Automated Analysis
```bash
uv run python Data/automated_analysis.py
```

### Run Tests
```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

## Project Layout
- `Data/`: raw/exported data, analysis scripts, generated reports
- `kmeans_from_scratch/`: K-Means implementation exercise and tests
- `tests/`: automated tests for reproducibility checks
- `pyproject.toml`, `uv.lock`: dependency definitions and lockfile

## Environment Variables (Garmin scripts only)
Some scripts in `Data/` call Garmin APIs directly.

```bash
export GARMIN_UNAME="your_username"
export GARMIN_PWORD="your_password"
```
