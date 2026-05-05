from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.preprocess import DEFAULT_OUTPUT_ROOT, REPORT_DIRNAME
from demixing.data.quality import QualityThresholds, build_quality_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QC quality tiers from preprocessing reports.")
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / REPORT_DIRNAME / "qc_report.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / REPORT_DIRNAME / "quality_manifest.csv",
    )
    args = parser.parse_args()

    counts = build_quality_manifest(args.report_csv, args.output_csv, QualityThresholds())
    print(counts)


if __name__ == "__main__":
    main()
