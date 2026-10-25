"""The scalar surfaces a split rule seeds its watershed on."""

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

N_ORIENT = 8
LINE_PX = 5  # length of the oriented averaging line, in pixels


def boundary_distance(seg: np.ndarray) -> np.ndarray:
    """EDT of (seg != boundary), the FTP pipeline's watershed surface."""
    return distance_transform_edt(seg != 2).astype(np.float32, copy=False)


def boundary_prob(prob: np.ndarray) -> np.ndarray:
    """1 - p_boundary, high away from a predicted boundary."""
    return (1.0 - prob[2]).astype(np.float32, copy=False)


def field_edge(prob: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """|grad p_field| scaled into [0, 1].

    An edge wherever the field probability drops, whether or not the boundary head
    fired there (centre pivots, plot rims with no visible track).
    """
    pf = prob[1].astype(np.float32)
    gy, gx = np.gradient(pf)
    return np.clip(np.hypot(gx, gy) * gain, 0.0, 1.0).astype(np.float32)


def boundary_or_edge(prob: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """1 - max(p_boundary, field edge): the ``pb`` surface plus field edges."""
    pb = prob[2].astype(np.float32)
    return (1.0 - np.maximum(pb, field_edge(prob, gain))).astype(np.float32, copy=False)


def line_kernel(theta: float, length: int) -> np.ndarray:
    """Normalised line of ``length`` px through the kernel centre at ``theta``."""
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    for t in np.linspace(-c, c, 4 * length + 1):
        r = int(round(c - t * np.sin(theta)))
        q = int(round(c + t * np.cos(theta)))
        k[r, q] = 1.0
    return k / k.sum()


def oriented_signal(pb: np.ndarray, length: int = LINE_PX) -> np.ndarray:
    """``(N_ORIENT, H, W)`` mean boundary probability along a line per orientation."""
    thetas = np.arange(N_ORIENT) * np.pi / N_ORIENT
    return np.stack(
        [convolve(pb, line_kernel(t, length), mode="nearest") for t in thetas]
    )
