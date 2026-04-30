from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demixing.data.endmembers import build_default_endmember_library
from demixing.data.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    PREPROCESS_PROTOCOLS,
    TARGET_AXIS,
    load_spectrum,
    preprocess_record,
)
from demixing.evaluation.classical_unmixing import (
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    unmix_spectra,
)
from demixing.evaluation.inference import save_predictions
from demixing.visualization.classical_unmixing import (
    plot_abundance_maps,
    plot_reconstruction_examples,
    plot_residual_map,
)
from demixing.visualization.plots import save_experiment_summary


PE_STARCH_DIR = "PE+\u6dc0\u7c89"
PP_STARCH_DIR = "PP+\u6dc0\u7c89"
PP_PE_STARCH_DIR = "PP+PE+\u6dc0\u7c89"

DEFAULT_SAMPLE_DIR = Path(PE_STARCH_DIR) / "1 785mw 2s 2 2 40 40"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/formal_v6_classical_unmixing_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical Raman unmixing on one real Raman mapping directory.")
    parser.add_argument("--input-root", type=Path, default=ROOT / DEFAULT_INPUT_ROOT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method", choices=["nnls", "ols", "fcls", "nmf"], default="nnls")
    parser.add_argument("--feature-mode", choices=["normalized", "corrected"], default="normalized")
    parser.add_argument("--starch-source", default="baseline")
    parser.add_argument("--protocol", choices=sorted(PREPROCESS_PROTOCOLS), default=DEFAULT_PROTOCOL_NAME)
    parser.add_argument("--nmf-components", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def infer_components(sample_dir: Path) -> tuple[str, ...]:
    path_text = sample_dir.as_posix()
    if PP_PE_STARCH_DIR in path_text:
        return ("PE", "PP", "starch")
    if PE_STARCH_DIR in path_text:
        return ("PE", "starch")
    if PP_STARCH_DIR in path_text:
        return ("PP", "starch")
    raise ValueError(f"Unable to infer components from sample_dir={sample_dir.as_posix()!r}")


def parse_xy(name: str) -> tuple[int | None, int | None]:
    match_x = re.search(r"-X(\d+)-", name)
    match_y = re.search(r"-Y(\d+)-", name)
    if match_x is None or match_y is None:
        return None, None
    return int(match_x.group(1)), int(match_y.group(1))


def load_mapping_spectra(
    input_root: Path,
    sample_dir: Path,
    feature_mode: str,
    protocol_name: str,
    limit: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    resolved_dir = sample_dir if sample_dir.is_absolute() else input_root / sample_dir
    csv_files = sorted(resolved_dir.glob("DATA-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DATA-*.csv files found in {resolved_dir}")

    rows: list[dict[str, object]] = []
    spectra: list[np.ndarray] = []
    for index, csv_path in enumerate(csv_files):
        record = load_spectrum(csv_path, input_root)
        _, corrected, normalized, _ = preprocess_record(record, protocol_name=protocol_name)
        x_idx, y_idx = parse_xy(csv_path.name)
        rows.append(
            {
                "relative_path": csv_path.relative_to(input_root).as_posix(),
                "sample_group_id": resolved_dir.relative_to(input_root).as_posix(),
                "x_idx": x_idx,
                "y_idx": y_idx,
            }
        )
        spectra.append(normalized if feature_mode == "normalized" else corrected)
        if limit is not None and (index + 1) >= limit:
            break
    return pd.DataFrame(rows), np.stack(spectra).astype(np.float32)


def save_endmember_table(output_path: Path, axis: np.ndarray, component_names: tuple[str, ...], matrix: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"RamanShift_cm-1": axis}
    for index, name in enumerate(component_names):
        data[f"endmember_{name}"] = matrix[:, index]
    pd.DataFrame(data).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir
    components = infer_components(sample_dir)

    metadata_df, spectra = load_mapping_spectra(
        input_root=args.input_root,
        sample_dir=sample_dir,
        feature_mode=args.feature_mode,
        protocol_name=args.protocol,
        limit=args.limit,
    )
    sample_tag = sample_dir.as_posix().replace("/", "__").replace("\\", "__")
    report_dir = args.output_root / sample_tag / "reports"
    figure_dir = args.output_root / sample_tag / "figures"
    if args.method == "nmf":
        n_components = args.nmf_components or len(components)
        result = blind_nmf_unmix_spectra(spectra, n_components=n_components)
        reference_library = build_default_endmember_library(
            input_root=args.input_root,
            include_components=components,
            starch_source=args.starch_source,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
        )
        result, similarity_df = align_blind_nmf_to_reference(result, reference_library)
        prediction_df = pd.concat([metadata_df.reset_index(drop=True), result.to_frame()], axis=1)
        prediction_df["dominant_component"] = [
            result.component_names[index]
            for index in np.argmax(result.abundances, axis=1)
        ]
        save_predictions(prediction_df, report_dir / "nmf_pixel_unmixing.csv")
        save_endmember_table(report_dir / "nmf_endmembers.csv", TARGET_AXIS, result.component_names, result.endmember_matrix)
        similarity_df.to_csv(report_dir / "nmf_reference_similarity.csv", encoding="utf-8-sig")
        plot_abundance_maps(
            prediction_df,
            component_names=list(result.component_names),
            output_path=figure_dir / "nmf_abundance_maps.png",
            title="NMF abundance maps",
        )
        plot_residual_map(
            prediction_df,
            output_path=figure_dir / "nmf_residual_rmse_map.png",
            title="NMF residual RMSE map",
        )
        plot_reconstruction_examples(
            axis=TARGET_AXIS,
            spectra=spectra,
            reconstructed=result.reconstructed,
            output_path=figure_dir / "nmf_reconstruction_examples.png",
            labels=prediction_df["relative_path"].tolist(),
        )
        mean_abundances = {
            name: float(np.mean(result.abundances[:, index]))
            for index, name in enumerate(result.component_names)
        }
        dominant_counts = prediction_df["dominant_component"].value_counts().to_dict()
        summary = {
            "experiment": "formal_v6_classical_unmixing_real",
            "sample_dir": sample_dir.as_posix(),
            "method": args.method,
            "feature_mode": args.feature_mode,
            "protocol": args.protocol,
            "components": list(components),
            "nmf_components": n_components,
            "n_spectra": int(len(prediction_df)),
            "mean_residual_l2": float(np.mean(result.residual_l2)),
            "mean_residual_rmse": float(np.mean(result.residual_rmse)),
            "mean_residual_r2": float(np.mean(result.residual_r2)),
            "mean_abundances": mean_abundances,
            "dominant_component_counts": dominant_counts,
            "reference_component_names": list(reference_library.names),
        }
    else:
        library = build_default_endmember_library(
            input_root=args.input_root,
            include_components=components,
            starch_source=args.starch_source,
            feature_mode=args.feature_mode,
            protocol_name=args.protocol,
        )
        result = unmix_spectra(spectra, library=library, method=args.method)
        prediction_df = pd.concat([metadata_df.reset_index(drop=True), result.to_frame()], axis=1)
        prediction_df["dominant_component"] = [
            result.component_names[index]
            for index in np.argmax(result.abundances, axis=1)
        ]
        save_predictions(prediction_df, report_dir / f"{args.method}_pixel_unmixing.csv")
        save_endmember_table(report_dir / f"{args.method}_endmembers.csv", library.axis, library.names, library.matrix)
        plot_abundance_maps(
            prediction_df,
            component_names=list(result.component_names),
            output_path=figure_dir / f"{args.method}_abundance_maps.png",
            title=f"{args.method.upper()} abundance maps",
        )
        plot_residual_map(
            prediction_df,
            output_path=figure_dir / f"{args.method}_residual_rmse_map.png",
            title=f"{args.method.upper()} residual RMSE map",
        )
        plot_reconstruction_examples(
            axis=library.axis,
            spectra=spectra,
            reconstructed=result.reconstructed,
            output_path=figure_dir / f"{args.method}_reconstruction_examples.png",
            labels=prediction_df["relative_path"].tolist(),
        )
        mean_abundances = {
            name: float(np.mean(result.abundances[:, index]))
            for index, name in enumerate(result.component_names)
        }
        mean_coefficients = {
            name: float(np.mean(result.coefficients[:, index]))
            for index, name in enumerate(result.component_names)
        }
        dominant_counts = prediction_df["dominant_component"].value_counts().to_dict()
        summary = {
            "experiment": "formal_v6_classical_unmixing_real",
            "sample_dir": sample_dir.as_posix(),
            "method": args.method,
            "feature_mode": args.feature_mode,
            "protocol": args.protocol,
            "components": list(components),
            "starch_source": args.starch_source,
            "n_spectra": int(len(prediction_df)),
            "mean_residual_l2": float(np.mean(result.residual_l2)),
            "mean_residual_rmse": float(np.mean(result.residual_rmse)),
            "mean_residual_r2": float(np.mean(result.residual_r2)),
            "mean_abundances": mean_abundances,
            "mean_coefficients": mean_coefficients,
            "dominant_component_counts": dominant_counts,
            "source_paths": {name: path.as_posix() for name, path in library.source_paths.items()},
        }
    save_experiment_summary(summary, report_dir / f"{args.method}_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
