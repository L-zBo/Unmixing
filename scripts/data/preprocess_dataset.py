from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
    process_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Raman dataset into a unified output directory.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    args = parser.parse_args()

    summary = process_dataset(args.input_root, args.output_root, protocol_name=args.protocol)
    print(summary)


if __name__ == "__main__":
    main()
