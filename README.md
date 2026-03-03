# runnng_ds_project

## Overview
This project is a running-focused data science workspace with:
- Garmin data extraction and feature engineering scripts
- A from-scratch K-Means implementation scaffold
- Supporting test/deployment experiments

## Project Layout
- `main.py`: placeholder entrypoint
- `Data/`: Garmin ingestion and analysis scripts
- `kmeans_from_scratch/`: K-Means implementation and tests
- `pyproject.toml`, `uv.lock`: project metadata and lockfile

## Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

## Setup (uv)
From the `runnng_ds_project` directory:

```bash
uv sync
source .venv/bin/activate
```

## Setup (pip alternative)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run
```bash
python main.py
```

## K-Means Scratch Tests
From `runnng_ds_project/kmeans_from_scratch`:

```bash
python -m pip install pytest numpy
python -m pytest -q tests/test_kmeans_scratch.py
```

## Data Scripts
`Data/activity_fact.py` and `Data/garmin.py` use Garmin APIs and local data exploration.

Set credentials before running Garmin API scripts:
```bash
export GARMIN_UNAME=\"your_username\"
export GARMIN_PWORD=\"your_password\"
```

Some data scripts are exploratory and may need extra packages not currently pinned in `pyproject.toml` (for example `matplotlib`, `python-dotenv`, `garminconnect`).

## Notes
- `runnng_ds_project.git/` is intentionally ignored and should not be tracked.
- `*.Zone.Identifier` files are ignored.
