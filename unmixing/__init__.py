"""Linear unmixing toolkit: PRISM (main line) + OLS / NNLS / FCLS / NMF baselines."""

from unmixing.unmix import (
    BlindNMFResult,
    ClassicalUnmixingResult,
    PrismUnmixingResult,
    align_blind_nmf_to_reference,
    blind_nmf_unmix_spectra,
    prism_unmix_spectra,
    unmix_spectra,
)

__all__ = [
    "PrismUnmixingResult",
    "ClassicalUnmixingResult",
    "BlindNMFResult",
    "prism_unmix_spectra",
    "unmix_spectra",
    "blind_nmf_unmix_spectra",
    "align_blind_nmf_to_reference",
]
