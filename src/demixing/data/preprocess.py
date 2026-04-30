from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


LASER_WAVELENGTH_NM = 785.0
TARGET_AXIS = np.linspace(103.785, 3696.98, 1024)
DEFAULT_INPUT_ROOT = Path("dataset")
DEFAULT_OUTPUT_ROOT = Path("outputs/preprocessing/dataset_preprocessed_als_l2")
REPORT_DIRNAME = "_reports"
_BASELINE_MATRIX_CACHE: dict[tuple[int, float], sparse.spmatrix] = {}


@dataclass(frozen=True)
class PreprocessProtocol:
    name: str
    baseline_mode: str
    normalize_mode: str
    apply_despike: bool = True
    apply_savgol: bool = True


DEFAULT_PROTOCOL_NAME = "als_l2"
PREPROCESS_PROTOCOLS: dict[str, PreprocessProtocol] = {
    "als_l2": PreprocessProtocol(name="als_l2", baseline_mode="als", normalize_mode="l2"),
    "als_max": PreprocessProtocol(name="als_max", baseline_mode="als", normalize_mode="max"),
    "none_l2": PreprocessProtocol(name="none_l2", baseline_mode="none", normalize_mode="l2"),
}


@dataclass
class SpectrumRecord:
    relative_path: Path
    axis: np.ndarray
    intensity: np.ndarray
    axis_type: str
    source_format: str
    header_axis: str
    header_intensity: str


def read_csv_with_fallbacks(path: Path) -> list[list[str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.reader(handle))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to decode {path}") from last_error


def normalize_header(name: str) -> str:
    return name.strip().replace("\ufeff", "")


def detect_axis_type(header_axis: str, axis: np.ndarray) -> str:
    header_axis = header_axis.strip()
    if "波长" in header_axis or "Wavelength" in header_axis:
        return "wavelength_nm"
    if "拉曼位移" in header_axis or "RamanShift" in header_axis or "波位" in header_axis:
        return "raman_shift_cm-1"
    if float(axis[0]) > 600.0:
        return "wavelength_nm"
    return "raman_shift_cm-1"


def load_spectrum(path: Path, input_root: Path) -> SpectrumRecord:
    rows = read_csv_with_fallbacks(path)
    if not rows:
        raise ValueError(f"Empty file: {path}")

    header = [normalize_header(cell) for cell in rows[0]]
    if len(rows) == 2 and len(header) > 10:
        pairs: list[tuple[float, float]] = []
        for axis_cell, intensity_cell in zip(header[1:], rows[1][1:]):
            axis_cell = normalize_header(axis_cell)
            intensity_cell = normalize_header(intensity_cell)
            if axis_cell == "" or intensity_cell == "":
                continue
            pairs.append((float(axis_cell), float(intensity_cell)))
        axis = np.asarray([pair[0] for pair in pairs], dtype=float)
        intensity = np.asarray([pair[1] for pair in pairs], dtype=float)
        source_format = "wide_2row"
        header_axis = header[0]
        header_intensity = normalize_header(rows[1][0])
    else:
        body = [row for row in rows[1:] if len(row) >= 2 and row[0] != ""]
        axis = np.asarray([float(row[0]) for row in body], dtype=float)
        intensity = np.asarray([float(row[1]) for row in body], dtype=float)
        source_format = "long_2col"
        header_axis = header[0]
        header_intensity = header[1] if len(header) > 1 else "Intensity"

    axis_type = detect_axis_type(header_axis, axis)
    return SpectrumRecord(
        relative_path=path.relative_to(input_root),
        axis=axis,
        intensity=intensity,
        axis_type=axis_type,
        source_format=source_format,
        header_axis=header_axis,
        header_intensity=header_intensity,
    )


def wavelength_to_raman_shift(axis_nm: np.ndarray, laser_nm: float = LASER_WAVELENGTH_NM) -> np.ndarray:
    return (1.0 / laser_nm - 1.0 / axis_nm) * 1e7


def ensure_ascending(axis: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if axis[0] <= axis[-1]:
        return axis, intensity
    return axis[::-1], intensity[::-1]


def hampel_despike(intensity: np.ndarray, window_size: int = 7, n_sigma: float = 4.5) -> np.ndarray:
    if intensity.size < 5:
        return intensity.copy()

    values = intensity.astype(float).copy()
    k = max(1, window_size)
    scale = 1.4826
    for idx in range(values.size):
        start = max(0, idx - k)
        end = min(values.size, idx + k + 1)
        window = values[start:end]
        median = float(np.median(window))
        mad = float(np.median(np.abs(window - median)))
        threshold = n_sigma * scale * (mad + 1e-12)
        if abs(values[idx] - median) > threshold:
            values[idx] = median
    return values


def savitzky_golay(intensity: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    if intensity.size < window or window % 2 == 0:
        return intensity.copy()

    half_window = window // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    vandermonde = np.vander(offsets, order + 1, increasing=True)
    coeffs = np.linalg.pinv(vandermonde)[0]
    padded = np.pad(intensity, (half_window, half_window), mode="edge")
    return np.convolve(padded, coeffs[::-1], mode="valid")


def baseline_als(intensity: np.ndarray, lam: float = 1e5, p: float = 0.01, iterations: int = 10) -> np.ndarray:
    length = intensity.size
    if length < 3:
        return np.zeros_like(intensity)

    cache_key = (length, lam)
    if cache_key not in _BASELINE_MATRIX_CACHE:
        diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc")
        _BASELINE_MATRIX_CACHE[cache_key] = lam * (diff.T @ diff)
    smooth_matrix = _BASELINE_MATRIX_CACHE[cache_key]
    weights = np.ones(length)

    for _ in range(iterations):
        weight_matrix = sparse.diags(weights, 0, shape=(length, length), format="csc")
        baseline = spsolve(weight_matrix + smooth_matrix, weights * intensity)
        weights = p * (intensity > baseline) + (1 - p) * (intensity <= baseline)
    return np.asarray(baseline, dtype=float)


def resample_to_target_axis(axis: np.ndarray, intensity: np.ndarray, target_axis: np.ndarray = TARGET_AXIS) -> np.ndarray:
    return np.interp(target_axis, axis, intensity)


def safe_max_normalize(intensity: np.ndarray) -> np.ndarray:
    max_value = float(np.max(intensity))
    if max_value <= 0:
        return intensity.copy()
    return intensity / max_value


def safe_l2_normalize(intensity: np.ndarray) -> np.ndarray:
    l2_norm = float(np.linalg.norm(intensity))
    if l2_norm <= 0:
        return intensity.copy()
    return intensity / l2_norm


def get_preprocess_protocol(protocol_name: str = DEFAULT_PROTOCOL_NAME) -> PreprocessProtocol:
    if protocol_name not in PREPROCESS_PROTOCOLS:
        choices = ", ".join(sorted(PREPROCESS_PROTOCOLS))
        raise KeyError(f"Unknown preprocess protocol {protocol_name!r}. Available: {choices}")
    return PREPROCESS_PROTOCOLS[protocol_name]


def normalized_column_name(normalize_mode: str) -> str:
    if normalize_mode == "l2":
        return "Intensity_norm_l2"
    if normalize_mode == "max":
        return "Intensity_norm_max"
    return "Intensity_normalized"


def normalize_intensity(intensity: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode == "l2":
        return safe_l2_normalize(intensity)
    if normalize_mode == "max":
        return safe_max_normalize(intensity)
    if normalize_mode == "none":
        return intensity.copy()
    raise ValueError(f"Unsupported normalize_mode={normalize_mode!r}")


def normalized_value_from_row(row: dict[str, str]) -> float:
    for key in ("Intensity_normalized", "Intensity_norm_l2", "Intensity_norm_max"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError("No normalized intensity column found.")


def spectrum_metrics(intensity: np.ndarray) -> dict[str, float]:
    if intensity.size < 3:
        return {
            "min": float(np.min(intensity)),
            "max": float(np.max(intensity)),
            "negative_count": int(np.sum(intensity < 0)),
            "roughness": 0.0,
            "spike_score": 0.0,
        }

    d2 = np.diff(intensity, n=2)
    median = float(np.median(d2))
    mad = float(np.median(np.abs(d2 - median))) + 1e-12
    return {
        "min": float(np.min(intensity)),
        "max": float(np.max(intensity)),
        "negative_count": int(np.sum(intensity < 0)),
        "roughness": float(np.std(np.diff(intensity)) / (np.std(intensity) + 1e-12)),
        "spike_score": float(np.max(np.abs(d2 - median)) / mad),
    }


def write_processed_csv(path: Path, corrected: np.ndarray, normalized: np.ndarray, normalize_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm_key = normalized_column_name(normalize_mode)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["RamanShift_cm-1", "Intensity_corrected", "Intensity_normalized"]
        if norm_key != "Intensity_normalized":
            header.append(norm_key)
        writer.writerow(header)
        for axis_value, corr, norm in zip(TARGET_AXIS, corrected, normalized):
            row = [f"{axis_value:.6f}", f"{corr:.6f}", f"{norm:.6f}"]
            if norm_key != "Intensity_normalized":
                row.append(f"{norm:.6f}")
            writer.writerow(row)


def write_reports(records: Sequence[dict[str, object]], summary: dict[str, object], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    qc_csv = report_dir / "qc_report.csv"
    fieldnames = [
        "relative_path",
        "axis_type",
        "source_format",
        "header_axis",
        "header_intensity",
        "converted_from_wavelength",
        "min_before_clip",
        "negative_count_before_clip",
        "roughness_after",
        "spike_score_after",
        "max_after",
    ]
    with qc_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    issues_csv = report_dir / "remaining_issues.csv"
    with issues_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "issue", "value"])
        for record in records:
            if float(record["spike_score_after"]) > 80:
                writer.writerow([record["relative_path"], "high_spike_score_after", record["spike_score_after"]])
            if float(record["roughness_after"]) > 0.2:
                writer.writerow([record["relative_path"], "high_roughness_after", record["roughness_after"]])
            if float(record["max_after"]) <= 0:
                writer.writerow([record["relative_path"], "non_positive_signal_after", record["max_after"]])

    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def preprocess_record(
    spectrum: SpectrumRecord,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    protocol = get_preprocess_protocol(protocol_name)
    axis = spectrum.axis
    if spectrum.axis_type == "wavelength_nm":
        axis = wavelength_to_raman_shift(axis)

    axis, intensity = ensure_ascending(axis, spectrum.intensity)
    resampled = resample_to_target_axis(axis, intensity, TARGET_AXIS)
    working = resampled
    if protocol.apply_despike:
        working = hampel_despike(working)
    if protocol.apply_savgol:
        working = savitzky_golay(working, window=7, order=2)
    if protocol.baseline_mode == "als":
        baseline = baseline_als(working, lam=1e5, p=0.01, iterations=10)
        corrected = working - baseline
    elif protocol.baseline_mode == "none":
        corrected = working.copy()
    else:
        raise ValueError(f"Unsupported baseline_mode={protocol.baseline_mode!r}")
    before_clip = spectrum_metrics(corrected)
    corrected = np.clip(corrected, 0.0, None)
    normalized = normalize_intensity(corrected, protocol.normalize_mode)
    after_clip = spectrum_metrics(corrected)
    merged = {
        "protocol_name": protocol.name,
        "baseline_mode": protocol.baseline_mode,
        "normalize_mode": protocol.normalize_mode,
        "min_before_clip": before_clip["min"],
        "negative_count_before_clip": before_clip["negative_count"],
        "roughness_after": after_clip["roughness"],
        "spike_score_after": after_clip["spike_score"],
        "max_after": after_clip["max"],
    }
    return TARGET_AXIS, corrected, normalized, merged


def process_dataset(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> dict[str, object]:
    protocol = get_preprocess_protocol(protocol_name)
    csv_files = sorted(input_root.rglob("*.csv"))
    records: list[dict[str, object]] = []
    converted_count = 0
    clipped_count = 0

    for csv_path in csv_files:
        spectrum = load_spectrum(csv_path, input_root)
        _, corrected, normalized, metrics = preprocess_record(spectrum, protocol_name=protocol.name)
        if spectrum.axis_type == "wavelength_nm":
            converted_count += 1
        if metrics["negative_count_before_clip"] > 0:
            clipped_count += 1

        output_path = output_root / spectrum.relative_path
        write_processed_csv(output_path, corrected, normalized, normalize_mode=protocol.normalize_mode)

        records.append(
            {
                "relative_path": str(spectrum.relative_path).replace("\\", "/"),
                "axis_type": spectrum.axis_type,
                "source_format": spectrum.source_format,
                "header_axis": spectrum.header_axis,
                "header_intensity": spectrum.header_intensity,
                "converted_from_wavelength": spectrum.axis_type == "wavelength_nm",
                "min_before_clip": round(float(metrics["min_before_clip"]), 6),
                "negative_count_before_clip": int(metrics["negative_count_before_clip"]),
                "roughness_after": round(float(metrics["roughness_after"]), 6),
                "spike_score_after": round(float(metrics["spike_score_after"]), 6),
                "max_after": round(float(metrics["max_after"]), 6),
            }
        )

    report_dir = output_root / REPORT_DIRNAME
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "protocol_name": protocol.name,
        "baseline_mode": protocol.baseline_mode,
        "normalize_mode": protocol.normalize_mode,
        "normalized_column": normalized_column_name(protocol.normalize_mode),
        "target_axis_points": int(TARGET_AXIS.size),
        "target_axis_start": float(TARGET_AXIS[0]),
        "target_axis_end": float(TARGET_AXIS[-1]),
        "total_processed_files": len(records),
        "converted_from_wavelength_files": converted_count,
        "files_with_negative_values_before_clip": clipped_count,
        "files_with_high_spike_score_after": sum(1 for item in records if float(item["spike_score_after"]) > 80),
        "files_with_high_roughness_after": sum(1 for item in records if float(item["roughness_after"]) > 0.2),
    }
    write_reports(records, summary, report_dir)
    return summary
