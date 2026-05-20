"""Linear unmixing toolkit: PRISM (main line) + OLS / NNLS / FCLS / NMF / MCR-ALS baselines."""

from unmixing.unmix import (
    BlindNMFResult,
    ClassicalUnmixingResult,
    McrAlsResult,
    PrismUnmixingResult,
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    mcr_als_unmix_spectra,
    prism_unmix_spectra,
    unmix_spectra,
)

__all__ = [
    "PrismUnmixingResult",
    "ClassicalUnmixingResult",
    "BlindNMFResult",
    "McrAlsResult",
    "prism_unmix_spectra",
    "unmix_spectra",
    "blind_nmf_unmix_spectra",
    "mcr_als_unmix_spectra",
    "align_blind_nmf_to_reference",
]
