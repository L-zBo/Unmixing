from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from preprocessing.preprocess import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_PROTOCOL_NAME,
    TARGET_AXIS,
    load_spectrum,
    preprocess_record,
)


DEFAULT_COMPONENT_PATHS: dict[str, Path] = {
    "PE": Path("PE纯谱/采集图谱.csv"),
    "PP": Path("PP纯谱/采集图谱.csv"),
}

DEFAULT_STARCH_PATHS: dict[str, Path] = {
    "baseline": Path("淀粉纯谱/玉米淀粉/DATA-105635-X0-Y30-8884.csv"),
    "展艺玉米淀粉": Path("泛化/展艺玉米淀粉/淀粉纯谱/采集图谱.csv"),
    "新良小麦淀粉": Path("泛化/新良小麦淀粉/淀粉纯谱/采集图谱.csv"),
    "甘汁园小麦淀粉": Path("泛化/甘汁园小麦淀粉/淀粉纯谱/采集图谱.csv"),
}


@dataclass(frozen=True)
class EndmemberLibrary:
    names: tuple[str, ...]
    axis: np.ndarray
    matrix: np.ndarray
    feature_mode: str
    source_paths: dict[str, Path]

    def spectrum(self, name: str) -> np.ndarray:
        index = self.names.index(name)
        return self.matrix[:, index].copy()

    @property
    def n_points(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_endmembers(self) -> int:
        return int(self.matrix.shape[1])


def list_available_starch_sources() -> tuple[str, ...]:
    return tuple(DEFAULT_STARCH_PATHS.keys())


def resolve_default_component_paths(
    include_components: Sequence[str] = ("PE", "PP", "starch"),
    starch_source: str = "baseline",
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for component in include_components:
        if component == "starch":
            if starch_source not in DEFAULT_STARCH_PATHS:
                choices = ", ".join(sorted(DEFAULT_STARCH_PATHS))
                raise KeyError(f"Unknown starch_source={starch_source!r}. Available: {choices}")
            selected["starch"] = DEFAULT_STARCH_PATHS[starch_source]
            continue
        if component not in DEFAULT_COMPONENT_PATHS:
            choices = ", ".join(sorted((*DEFAULT_COMPONENT_PATHS.keys(), "starch")))
            raise KeyError(f"Unknown component={component!r}. Available: {choices}")
        selected[component] = DEFAULT_COMPONENT_PATHS[component]
    return selected


def _select_feature(corrected: np.ndarray, normalized: np.ndarray, feature_mode: str) -> np.ndarray:
    if feature_mode == "corrected":
        return corrected.astype(np.float32, copy=False)
    if feature_mode == "normalized":
        return normalized.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature_mode={feature_mode!r}. Expected 'corrected' or 'normalized'.")


def load_endmember_spectrum(
    relative_path: Path | str,
    input_root: Path = DEFAULT_INPUT_ROOT,
    feature_mode: str = "normalized",
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> np.ndarray:
    relative_path = Path(relative_path)
    spectrum = load_spectrum(input_root / relative_path, input_root)
    _, corrected, normalized, _ = preprocess_record(spectrum, protocol_name=protocol_name)
    return _select_feature(corrected, normalized, feature_mode)


def build_endmember_library(
    component_paths: Mapping[str, Path | str],
    input_root: Path = DEFAULT_INPUT_ROOT,
    feature_mode: str = "normalized",
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> EndmemberLibrary:
    names = tuple(component_paths.keys())
    if not names:
        raise ValueError("component_paths must not be empty.")

    spectra: list[np.ndarray] = []
    resolved_paths: dict[str, Path] = {}
    for name, relative_path in component_paths.items():
        relative_path = Path(relative_path)
        spectra.append(
            load_endmember_spectrum(
                relative_path,
                input_root=input_root,
                feature_mode=feature_mode,
                protocol_name=protocol_name,
            )
        )
        resolved_paths[name] = relative_path

    matrix = np.column_stack(spectra).astype(np.float32, copy=False)
    return EndmemberLibrary(
        names=names,
        axis=TARGET_AXIS.copy(),
        matrix=matrix,
        feature_mode=feature_mode,
        source_paths=resolved_paths,
    )


def build_default_endmember_library(
    input_root: Path = DEFAULT_INPUT_ROOT,
    include_components: Sequence[str] = ("PE", "PP", "starch"),
    starch_source: str = "baseline",
    feature_mode: str = "normalized",
    protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> EndmemberLibrary:
    component_paths = resolve_default_component_paths(
        include_components=include_components,
        starch_source=starch_source,
    )
    return build_endmember_library(
        component_paths=component_paths,
        input_root=input_root,
        feature_mode=feature_mode,
        protocol_name=protocol_name,
    )
