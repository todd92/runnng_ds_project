# Data Analysis Automation

## Purpose
`automated_analysis.py` runs a repeatable analysis pipeline on `activity_fact.csv` and generates summary artifacts and charts.

## Input
- `activity_fact.csv`

## Run
From the `runnng_ds_project` root:

```bash
uv run python Data/automated_analysis.py
```

## Output Folder
- `Data/reports/analysis_summary.md`
- `Data/reports/descriptive_stats.csv`
- `Data/reports/distance_distribution.png`
- `Data/reports/distance_vs_duration.png`
- `Data/reports/monthly_distance.png`
- `Data/reports/hr_zone_totals.png`

## What It Computes
- Descriptive statistics for numeric columns
- Distance distribution histogram
- Distance vs duration scatter plot
- Monthly total mileage trend line
- Total seconds in HR zones 1-5 (if those columns are available)
