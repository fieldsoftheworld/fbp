"""Faster drop-ins for the two scikit-image calls that dominate BoundaryVote.

Profiled on 5,120^2 production windows, ``h_maxima`` (53%) and ``watershed`` (38%)
were 92% of BoundaryVote's runtime. For the ``pb`` surface (1 - p_boundary with the
softmax stored as uint8/255, which is how the global run keeps it) both can run on
integer levels:

``h_maxima_itk``      ITK grayscale reconstruction by dilation. Reconstruction has a
                      unique result, so this is bit-identical to skimage (checked on 5k
                      and 9k windows) and ~10x faster single-threaded.
``watershed_bucket``  O(N) hierarchical-queue flooding with skimage's rules (same
                      neighbour order, cost clamping, watershed-line test, masking). The
                      one difference: equal-cost *marker* pixels leave the queue in scan
                      order, where skimage's binary heap releases them in heap order.
                      That moves lines only at exact ties: 99.99% pixel agreement,
                      ~99.5% of parcels at IoU >= 0.9 end to end on production tiles.

Both refuse surfaces that are not on the 1/255 grid, where the integer mapping is
invalid; ``quantize_prob`` puts a float softmax on that grid, which is exactly
what storing it as uint8 does in production (``+q1`` in a strategy id does both).
"""

import numpy as np

try:
    import SimpleITK as sitk  # noqa: N813  (the package's own conventional alias)
    from numba import njit
except ImportError as e:
    raise ImportError("+q1 needs the fast backend: pip install 'fbp[fast]'") from e
from skimage.morphology._util import _offsets_to_raveled_neighbors

LEVELS = 255.0


def quantize_prob(prob: np.ndarray) -> np.ndarray:
    """Softmax rounded to 1/255 steps per class, as the uint8 production rasters do."""
    return (np.rint(prob.astype(np.float32) * LEVELS) / LEVELS).astype(np.float32)


def _to_levels(x: np.ndarray, name: str) -> np.ndarray:
    u = np.rint(x * LEVELS)
    if np.abs(x * LEVELS - u).max() > 0.25:  # fp16 error is ~0.06 levels
        raise ValueError(f"{name} is not on the 1/255 grid; use +q1 / quantize_prob")
    return u.astype(np.uint8)


def h_maxima_itk(image: np.ndarray, h: float, footprint: None = None) -> np.ndarray:
    """``skimage.morphology.h_maxima`` for a [0, 1] surface on the 1/255 grid."""
    if footprint is not None:
        raise ValueError("only the default (full) footprint is supported")
    img = _to_levels(image, "h_maxima input")
    h_lv = int(np.ceil(h * LEVELS - 1e-9))  # residue >= h  <=>  levels >= ceil(h*255)
    if h_lv > int(img.max()) - int(img.min()):
        return np.zeros(img.shape, np.uint8)
    marker = np.clip(img.astype(np.int16) - h_lv, 0, 255).astype(np.uint8)
    f = sitk.ReconstructionByDilationImageFilter()
    f.SetFullyConnected(True)
    # GetArrayFromImage (a copy): a view of the temporary result can be freed under us.
    rec = sitk.GetArrayFromImage(
        f.Execute(sitk.GetImageFromArray(marker), sitk.GetImageFromArray(img))
    )
    return ((img.astype(np.int16) - rec) >= h_lv).astype(np.uint8)


@njit(cache=True)
def _bucket_kernel(img, locs, structure, mask, output):  # noqa: ANN001, ANN202, PLR0912, PLR0915
    cap = max(1024, 4 * locs.shape[0])
    ix = np.empty(cap, np.int64)
    sr = np.empty(cap, np.int64)
    nx = np.full(cap, -1, np.int64)
    head = np.full(256, -1, np.int64)
    tail = np.full(256, -1, np.int64)
    n = 0
    for k in range(locs.shape[0]):
        v = img[locs[k]]
        ix[n] = locs[k]
        sr[n] = locs[k]
        nx[n] = -1
        if head[v] < 0:
            head[v] = n
        else:
            nx[tail[v]] = n
        tail[v] = n
        n += 1
    cur = 0
    while cur < 256:
        e = head[cur]
        if e < 0:
            cur += 1
            continue
        head[cur] = nx[e]
        if head[cur] < 0:
            tail[cur] = -1
        ei, es = ix[e], sr[e]
        if output[ei] and ei != es:
            continue
        lab = output[es]
        diff = False
        if not mask[ei]:
            diff = True
        else:
            for q in range(structure.shape[0]):
                nb = structure[q] + ei
                if mask[nb]:
                    nl = output[nb]
                    if nl != 0 and nl != lab:  # noqa: PLR1714  (numba: keep it scalar)
                        mask[ei] = False
                        diff = True
                        break
        if not diff:
            output[ei] = lab
        for q in range(structure.shape[0]):
            nb = structure[q] + ei
            if not mask[nb] or output[nb]:
                continue
            v = img[nb]
            if v < cur:  # noqa: PLR1730  skimage: cost >= the popping pixel's cost
                v = cur
            if n == cap:
                cap *= 2
                a = np.empty(cap, np.int64)
                a[:n] = ix[:n]
                ix = a
                a = np.empty(cap, np.int64)
                a[:n] = sr[:n]
                sr = a
                a = np.full(cap, -1, np.int64)
                a[:n] = nx[:n]
                nx = a
            ix[n] = nb
            sr[n] = es
            nx[n] = -1
            if head[v] < 0:
                head[v] = n
            else:
                nx[tail[v]] = n
            tail[v] = n
            n += 1


def watershed_bucket(
    image: np.ndarray,
    markers: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    watershed_line: bool = True,
    connectivity: int = 1,
) -> np.ndarray:
    """Skimage's ``watershed(image, markers, mask=mask, watershed_line=True)``.

    For ``image = -surface`` with ``surface`` on the 1/255 grid.
    """
    if not watershed_line or connectivity != 1:
        raise ValueError("only watershed_line=True, connectivity=1 is implemented")
    if markers is None or mask is None:
        raise ValueError("markers and mask are required")
    lv = _to_levels(image + 1.0, "watershed input")  # -(1 - pb) + 1 = pb
    img = np.pad(lv, 1)
    msk = np.pad(mask.astype(bool), 1).ravel()
    out = np.pad(markers.astype(np.int32), 1)
    st = _offsets_to_raveled_neighbors(
        img.shape, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), center=(1, 1)
    )
    _bucket_kernel(
        img.ravel(),
        np.flatnonzero(out).astype(np.int64),
        st.astype(np.int64),
        msk,
        out.ravel(),
    )
    return out[1:-1, 1:-1].copy()
