"""Extração de parênquima cerebral em TC só com OpenCV (adaptação de Muschelli et al., 2015)."""

from .nifti import NiftiImage, read_nifti, write_nifti
from .segment import Params, segment_brain

__all__ = ["NiftiImage", "Params", "read_nifti", "segment_brain", "write_nifti"]
__version__ = "0.1.0"
