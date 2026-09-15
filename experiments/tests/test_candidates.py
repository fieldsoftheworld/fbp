"""Candidate generation must reproduce FTP's watershed exactly."""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from boundaryvote import ftp_compat as ftp
from boundaryvote.candidates import instance_polygons, watershed_with_meta
from boundaryvote.paths import H_SWEEP, cache_dir


def synthetic_seg(rng: np.random.Generator) -> np.ndarray:
    seg = np.zeros((96, 128), dtype=np.uint8)
    seg[10:50, 10:60] = 1
    seg[10:50, 30] = 2  # a boundary line splitting the block
    seg[55:90, 20:110] = 1
    seg[60:85, 70:75] = 2  # partial boundary -> h decides the split
    noise = rng.random(seg.shape) < 0.01
    seg[noise & (seg == 1)] = 0
    return seg


@pytest.mark.parametrize("h", H_SWEEP)
def test_matches_ftp_watershed(h: float) -> None:
    rng = np.random.default_rng(0)
    seg = synthetic_seg(rng)
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    inst, n_markers, fallback = watershed_with_meta(seg, dist, h)
    ref = ftp.watershed_instances(seg, dist, h_min=h, field_class=1)
    np.testing.assert_array_equal(inst, ref)
    assert fallback == (n_markers == 0)


def test_empty_field() -> None:
    seg = np.zeros((16, 16), dtype=np.uint8)
    inst, n_markers, fallback = watershed_with_meta(
        seg, np.ones_like(seg, dtype=np.float32), 2.0
    )
    assert inst.max() == 0 and n_markers == 0 and not fallback


def test_instance_polygons_cover_every_label() -> None:
    rng = np.random.default_rng(1)
    seg = synthetic_seg(rng)
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    inst, _, _ = watershed_with_meta(seg, dist, 1.0)
    polys = instance_polygons(inst)
    assert {i for i, _ in polys} == set(np.unique(inst[inst > 0]).tolist())
    total_px = sum(g.area for _, g in polys)
    assert total_px == pytest.approx(float((inst > 0).sum()))


@pytest.mark.skipif(
    not (cache_dir("smoke_b3/test") / "portugal").exists(), reason="smoke cache missing"
)
def test_on_cached_patch() -> None:
    npz = next((cache_dir("smoke_b3/test") / "portugal").glob("*.npz"))
    with np.load(npz) as d:
        seg = d["prob"].astype(np.float32).argmax(0).astype(np.uint8)
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    for h in (0.5, 2.0, 64.0):
        inst, _, _ = watershed_with_meta(seg, dist, h)
        np.testing.assert_array_equal(inst, ftp.watershed_instances(seg, dist, h_min=h))
