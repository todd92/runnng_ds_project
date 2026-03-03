from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_analysis(
    csv_path: Path = Path("Data/activity_fact.csv"),
    out_dir: Path = Path("Data/reports"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "distance_miles",
        "duration_minutes",
        "elapsedDuration_minutes",
        "maxHR",
        "averageRunningCadenceInStepsPerMinute",
        "hrTimeInZone_1",
        "hrTimeInZone_2",
        "hrTimeInZone_3",
        "hrTimeInZone_4",
        "hrTimeInZone_5",
    ]
    df = _safe_numeric(df, numeric_cols)

    if "startTimeLocal" in df.columns:
        df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
        df["month"] = df["startTimeLocal"].dt.to_period("M").dt.to_timestamp()

    desc = df.select_dtypes(include="number").describe().T
    desc.to_csv(out_dir / "descriptive_stats.csv")

    if "distance_miles" in df.columns:
        plt.figure(figsize=(9, 5))
        df["distance_miles"].dropna().hist(bins=24)
        plt.title("Run Distance Distribution")
        plt.xlabel("Distance (miles)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "distance_distribution.png", dpi=140)
        plt.close()

    if {"distance_miles", "duration_minutes"}.issubset(df.columns):
        plot_df = df[["distance_miles", "duration_minutes"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(9, 5))
            plt.scatter(
                plot_df["distance_miles"],
                plot_df["duration_minutes"],
                alpha=0.6,
            )
            plt.title("Distance vs Duration")
            plt.xlabel("Distance (miles)")
            plt.ylabel("Duration (minutes)")
            plt.tight_layout()
            plt.savefig(out_dir / "distance_vs_duration.png", dpi=140)
            plt.close()

    if {"month", "distance_miles"}.issubset(df.columns):
        month_df = (
            df.dropna(subset=["month", "distance_miles"])
            .groupby("month", as_index=False)["distance_miles"]
            .sum()
            .sort_values("month")
        )
        if not month_df.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(month_df["month"], month_df["distance_miles"], marker="o")
            plt.title("Monthly Distance")
            plt.xlabel("Month")
            plt.ylabel("Total Distance (miles)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / "monthly_distance.png", dpi=140)
            plt.close()

    zone_cols = [f"hrTimeInZone_{i}" for i in range(1, 6)]
    existing_zone_cols = [c for c in zone_cols if c in df.columns]
    if existing_zone_cols:
        zone_totals = df[existing_zone_cols].sum(skipna=True).sort_index()
        if zone_totals.sum() > 0:
            plt.figure(figsize=(8, 5))
            plt.bar(zone_totals.index, zone_totals.values)
            plt.title("Total Time in Heart Rate Zones")
            plt.xlabel("Zone")
            plt.ylabel("Total Seconds")
            plt.tight_layout()
            plt.savefig(out_dir / "hr_zone_totals.png", dpi=140)
            plt.close()

    total_runs = len(df)
    avg_distance = float(df["distance_miles"].mean()) if "distance_miles" in df else 0.0
    avg_duration = (
        float(df["duration_minutes"].mean()) if "duration_minutes" in df else 0.0
    )
    longest = float(df["distance_miles"].max()) if "distance_miles" in df else 0.0

    summary = (
        "# Automated Activity Analysis\n\n"
        f"- Total runs: {total_runs}\n"
        f"- Average distance (miles): {avg_distance:.2f}\n"
        f"- Average duration (minutes): {avg_duration:.2f}\n"
        f"- Longest run (miles): {longest:.2f}\n\n"
        "## Outputs\n"
        "- `descriptive_stats.csv`\n"
        "- `distance_distribution.png`\n"
        "- `distance_vs_duration.png`\n"
        "- `monthly_distance.png`\n"
        "- `hr_zone_totals.png` (if HR-zone fields are present)\n"
    )
    (out_dir / "analysis_summary.md").write_text(summary)


if __name__ == "__main__":
    run_analysis()
