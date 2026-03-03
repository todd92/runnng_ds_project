import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_analysis_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "Data" / "automated_analysis.py"
    spec = importlib.util.spec_from_file_location("automated_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class TestAutomatedAnalysis(unittest.TestCase):
    def test_run_analysis_generates_expected_outputs(self):
        module = load_analysis_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "sample.csv"
            out_dir = tmp_dir / "reports"

            rows = [
                {
                    "activityId": 1,
                    "startTimeLocal": "2025-01-01 07:00:00",
                    "distance_miles": 3.0,
                    "duration_minutes": 28.5,
                    "elapsedDuration_minutes": 30.0,
                    "maxHR": 165,
                    "averageRunningCadenceInStepsPerMinute": 168,
                    "hrTimeInZone_1": 120,
                    "hrTimeInZone_2": 600,
                    "hrTimeInZone_3": 480,
                    "hrTimeInZone_4": 180,
                    "hrTimeInZone_5": 60,
                },
                {
                    "activityId": 2,
                    "startTimeLocal": "2025-02-01 07:00:00",
                    "distance_miles": 5.0,
                    "duration_minutes": 45.0,
                    "elapsedDuration_minutes": 47.0,
                    "maxHR": 172,
                    "averageRunningCadenceInStepsPerMinute": 170,
                    "hrTimeInZone_1": 90,
                    "hrTimeInZone_2": 700,
                    "hrTimeInZone_3": 520,
                    "hrTimeInZone_4": 200,
                    "hrTimeInZone_5": 80,
                },
            ]

            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            module.run_analysis(csv_path=csv_path, out_dir=out_dir)

            expected_files = [
                "analysis_summary.md",
                "descriptive_stats.csv",
                "distance_distribution.png",
                "distance_vs_duration.png",
                "monthly_distance.png",
                "hr_zone_totals.png",
            ]
            for file_name in expected_files:
                self.assertTrue((out_dir / file_name).exists(), f"Missing {file_name}")


if __name__ == "__main__":
    unittest.main()
