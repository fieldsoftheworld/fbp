"""Rasterize a polygon set and extract it again, the way FTP scores its output.

All functions take polygon lists in the eval-grid pixel frame and return the
same. ``roundtrip`` turns per-instance polygons into the connected components
FTP's ``inst > 0`` step produces, so ``rt:h2`` is the FTP pipeline as scored.
"""

import numpy as np
import rasterio.features

from . import ftp_compat as ftp


def rasterize(polys: list, shape: tuple[int, int]) -> np.ndarray:
    """Binary foreground mask of a polygon list."""
    if not polys:
        return np.zeros(shape, dtype=np.uint8)
    return rasterio.features.rasterize(
        ((p, 1) for p in polys), out_shape=shape, fill=0, dtype="uint8"
    )


def polygonize(mask: np.ndarray) -> list:
    """Four-connected foreground components, identical to FTP's ``_extract_shapes``."""
    return ftp.extract_shapes(mask)


def roundtrip(polys: list, shape: tuple[int, int]) -> list:
    """Rasterize then polygonize."""
    return polygonize(rasterize(polys, shape))
