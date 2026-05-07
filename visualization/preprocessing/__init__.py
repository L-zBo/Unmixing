"""Preprocessing-protocol comparison plots."""
from visualization.preprocessing.endmember_fingerprint import (
    plot_endmember_fingerprints,
)
from visualization.preprocessing.protocol_consistency import (
    plot_fingerprint_retention_bars,
    plot_protocol_cv_bars,
    plot_protocol_reconstruction_r2_bars,
)
from visualization.preprocessing.spectrum import (
    plot_protocol_abundance_grid,
    plot_protocol_spectrum_triptych,
    plot_single_spectrum_preprocessing,
)

__all__ = [
    "plot_endmember_fingerprints",
    "plot_fingerprint_retention_bars",
    "plot_protocol_abundance_grid",
    "plot_protocol_cv_bars",
    "plot_protocol_reconstruction_r2_bars",
    "plot_protocol_spectrum_triptych",
    "plot_single_spectrum_preprocessing",
]
