"""Visualization plots organized by figure type (mirrors LA_test layout).

Top-level re-exports keep the legacy ``from visualization import plot_xxx`` callsite
working while the implementation lives in dedicated subpackages by figure type.
"""
from visualization.abundance import plot_abundance_maps
from visualization.method_comparison import (
    plot_method_abundance_bars,
    plot_method_metric_bars,
)
from visualization.preprocessing import (
    plot_protocol_abundance_grid,
    plot_protocol_spectrum_triptych,
    plot_single_spectrum_preprocessing,
)
from visualization.reconstruction import plot_reconstruction_examples
from visualization.residual import plot_residual_map

__all__ = [
    "plot_abundance_maps",
    "plot_method_abundance_bars",
    "plot_method_metric_bars",
    "plot_protocol_abundance_grid",
    "plot_protocol_spectrum_triptych",
    "plot_reconstruction_examples",
    "plot_residual_map",
    "plot_single_spectrum_preprocessing",
]
