from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.endmembers import build_default_endmember_library
from demixing.data.preprocess import DEFAULT_INPUT_ROOT
from demixing.data.synthetic_unmixing import (
    SyntheticMapConfig,
    generate_synthetic_map,
    save_synthetic_map,
)


DEFAULT_OUTPUT_ROOT = ROOT / "outputs/synthetic_unmixing/default_als_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic Raman unmixing map with pixel-level abundance ground truth.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--components", nargs="+", default=["PE", "PP", "starch"])
    parser.add_argument("--starch-source", default="baseline")
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--baseline-scale", type=float, default=0.02)
    parser.add_argument("--scale-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = build_default_endmember_library(
        input_root=args.input_root,
        include_components=tuple(args.components),
        starch_source=args.starch_source,
        feature_mode=args.feature_mode,
    )
    config = SyntheticMapConfig(
        width=args.width,
        height=args.height,
        smooth_sigma=args.smooth_sigma,
        noise_std=args.noise_std,
        baseline_scale=args.baseline_scale,
        scale_jitter=args.scale_jitter,
        random_seed=args.seed,
    )
    result = generate_synthetic_map(library, config=config)
    save_synthetic_map(result, args.output_root)
    print(
        {
            "output_root": str(args.output_root),
            "components": list(result.component_names),
            "width": result.width,
            "height": result.height,
            "n_pixels": int(result.width * result.height),
            "n_points": int(result.axis.size),
        }
    )


if __name__ == "__main__":
    main()
