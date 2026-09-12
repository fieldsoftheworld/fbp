"""The global Fields of The World prediction store and its grid.

``variables`` is a Zarr v3 array ``(time, band, y, x)`` of float32 softmax,
bands ``[background, field, boundary]``, on a 10 m-at-equator EPSG:4326 grid
(``TX`` degrees per pixel). Time index 0 is 2024 and 1 is 2025. Shards are
8192 x 8192 with 2048 x 2048 inner chunks, so ``TILE`` is the natural unit of
work: one inner chunk is one HTTP range read of about 1.8 s.

The 2048x overview level has exactly one pixel per inner chunk, which is how
``global_tiles`` finds the tiles that hold any field signal.
"""

import math
import time

import numpy as np

URL = "https://data.source.coop/ftw/global-data/predictions/zarr/alpha/global.zarr"
TX = 8.98311982e-05  # degrees per pixel, both axes
X0, Y0 = -180.0, 83.748345  # upper-left corner of pixel (0, 0)
TILE = 2048
YEARS = {0: 2024, 1: 2025}
M_PER_DEG = 111_319.49  # metres per degree of latitude (WGS84 mean)


def open_variables(level: str | None = None):
    """The full-resolution array, or one overview level such as ``"2048x"``."""
    import zarr

    g = zarr.open_group(URL, mode="r")
    return g["variables"] if level is None else g[f"{level}/variables"]


def read_window(v, t: int, y0: int, y1: int, x0: int, x1: int, tries: int = 6):
    """``(3, y1-y0, x1-x0)`` float32 with NaN (no data) mapped to background."""
    for k in range(tries):
        try:
            p = np.asarray(v[t, :, y0:y1, x0:x1], dtype=np.float32)
            break
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    bad = ~np.isfinite(p).all(axis=0)
    if bad.any():
        p[:, bad] = 0.0
        p[0, bad] = 1.0
    return p


def tile_window(row: int, col: int) -> tuple[int, int, int, int]:
    """Pixel window ``(y0, y1, x0, x1)`` of inner chunk ``(row, col)``."""
    return row * TILE, (row + 1) * TILE, col * TILE, (col + 1) * TILE


def tile_center(row: int, col: int) -> tuple[float, float]:
    """``(lon, lat)`` of the tile centre."""
    return X0 + (col + 0.5) * TILE * TX, Y0 - (row + 0.5) * TILE * TX


def pixel_area_m2(lat: float) -> float:
    """Ground area of one store pixel at latitude ``lat`` (degrees)."""
    d = TX * M_PER_DEG
    return d * d * math.cos(math.radians(lat))


def affine(y0: int, x0: int):
    from affine import Affine

    return Affine(TX, 0.0, X0 + x0 * TX, 0.0, -TX, Y0 - y0 * TX)
