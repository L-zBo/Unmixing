from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.manifest import build_sample_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample manifest with quality tiers and weak concentration labels.")
    parser.add_argument(
        "--quality-manifest-csv",
        type=Path,
        default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1/_reports/quality_manifest.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs/preprocessing/dataset_preprocessed_v1/_reports/sample_manifest.csv",
    )
    args = parser.parse_args()

    counts = build_sample_manifest(args.quality_manifest_csv, args.output_csv)
    print(counts)


if __name__ == "__main__":
    main()
