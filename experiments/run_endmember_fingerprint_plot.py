"""Plot endmember pure spectra (PE / PP / starch) with annotated fingerprint peaks.

Single-figure script for the PPT "physical basis" slide. Loads the three
endmembers under the default ALS+L2 protocol and overlays them with literature
peak annotations.

Read-only on dataset/. Outputs to ``outputs/experiments/formal_v15_endmember_fingerprint``
and mirrors to ``outputs/showcase/endmember_fingerprint``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.endmembers import build_default_endmember_library
from preprocessing.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
)
from visualization.preprocessing import plot_endmember_fingerprints


# Literature peak positions (cm-1). PE peaks from Sage 2021; PP from RSC
# Analyst 2024; starch (Raman glucan backbone) from common references.
DEFAULT_FINGERPRINT_PEAKS: dict[str, tuple[int, ...]] = {
    "PE": (1062, 1130, 1295, 1440),
    "PP": (808, 841, 1330),
    "starch": (478, 1124),
}

DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v15_endmember_fingerprint"
SHOWCASE_OUTPUT_ROOT = ROOT / "outputs/showcase/endmember_fingerprint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay PE / PP / starch endmember pure spectra with literature fingerprint peaks.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--starch-source", default="baseline")
    parser.add_argument("--no-showcase", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = build_default_endmember_library(
        input_root=args.input_root,
        include_components=("PE", "PP", "starch"),
        starch_source=args.starch_source,
        feature_mode="normalized",
        protocol_name=args.protocol,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / "endmember_fingerprints.png"
    plot_endmember_fingerprints(
        axis=library.axis,
        endmember_matrix=library.matrix,
        component_names=library.names,
        peaks_by_component=DEFAULT_FINGERPRINT_PEAKS,
        output_path=output_path,
        title=f"Endmember pure spectra with literature fingerprint peaks ({args.protocol})",
        protocol_name=args.protocol,
    )

    if not args.no_showcase:
        SHOWCASE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (SHOWCASE_OUTPUT_ROOT / "endmember_fingerprints.png").write_bytes(output_path.read_bytes())

    print(
        f"[run_endmember_fingerprint_plot] saved {output_path.as_posix()} "
        f"(starch_source={args.starch_source}, protocol={args.protocol})"
    )


if __name__ == "__main__":
    main()
