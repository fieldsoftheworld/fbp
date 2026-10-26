"""Stochastic watershed (Angulo & Jeulin 2007).

For each of ``draws`` rounds, ``markers_per_draw`` random markers are placed in the
foreground and the distance surface of the FTP pipeline is watershed with lines. A
pixel's line frequency is the fraction of rounds in which it lies on a line. Pixels at
or above ``min_line_frequency`` are cut, and the rest of the foreground is labelled
by connected components.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy.ndimage import label as ndi_label
from skimage.segmentation import watershed

from fbp.core.surfaces import boundary_distance
from fbp.methods.base import Method


@dataclass(frozen=True, kw_only=True)
class StochasticWatershed(Method):
    """Cut where watersheds from random markers often draw a line.

    Attributes:
        markers_per_draw: Random markers placed in the foreground per round.
        draws: Number of rounds.
        min_line_frequency: Cut pixels that are on a watershed line in at least this
            fraction of rounds.
        seed: Random seed for the marker positions.
    """

    markers_per_draw: int
    draws: int
    min_line_frequency: float
    seed: int = 0

    @override
    def split(
        self,
        fg_mask: np.ndarray,
        seg: np.ndarray,
        prob: np.ndarray,
        context: dict | None,
    ) -> np.ndarray:
        """Cut where random-marker watershed lines agree at least ``t``."""
        del prob, context
        if not fg_mask.any():
            return np.zeros(fg_mask.shape, dtype=np.int32)
        dist = boundary_distance(seg) * fg_mask
        rng = np.random.default_rng(self.seed)
        r, c = np.nonzero(fg_mask)
        freq = np.zeros(fg_mask.shape, dtype=np.float32)
        for _ in range(self.draws):
            idx = rng.choice(
                len(r), size=min(self.markers_per_draw, len(r)), replace=False
            )
            markers = np.zeros(fg_mask.shape, dtype=np.int32)
            markers[r[idx], c[idx]] = np.arange(1, len(idx) + 1)
            lab = watershed(-dist, markers=markers, mask=fg_mask, watershed_line=True)
            freq += fg_mask & (lab == 0)
        freq /= self.draws
        out, _ = ndi_label(fg_mask & (freq < self.min_line_frequency))
        return out.astype(np.int32, copy=False)
