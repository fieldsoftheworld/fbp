"""Candidate generation must reproduce FTP's watershed exactly."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
from scipy.ndimage import distance_transform_edt

from fbp.eval import ftp_compat as ftp
from fbp.eval.candidates import (
    instance_polygons,
    process_country,
    roundtrip,
    watershed_with_meta,
)
from fbp.eval.paths import H_SWEEP, cache_dir


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


def test_roundtrip_merges_touching_polygons() -> None:
    shape = (20, 20)
    left = shapely.geometry.box(0, 0, 5, 10)
    right = shapely.geometry.box(5, 0, 10, 10)
    assert len(roundtrip([left, right], shape)) == 1


def _write_npz(path: Path, seg: np.ndarray, prob: np.ndarray) -> None:
    np.savez(
        path,
        prob=prob.astype(np.float16),
        seg=seg.astype(np.uint8),
        gt=np.zeros_like(seg),
        transform=np.array([1.0, 0, 0, 0, -1.0, 0], dtype=np.float64),
        crs=np.array("EPSG:32633"),
    )


class _FakeMethods:
    """Stand-in for ``fbp.methods``: each id seeds a distinct watershed depth."""

    @staticmethod
    def run(
        name: str,
        seg: np.ndarray,
        _prob: np.ndarray,
        _px_m2: float | None,
        _ctx: dict | None,
    ) -> np.ndarray:
        h = float(name.rsplit("h", maxsplit=1)[-1])
        dist = distance_transform_edt(seg != 2).astype(np.float32)
        inst, _, _ = watershed_with_meta(seg, dist, h)
        return inst


def test_process_country_chunked_tags_match_single_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One tag per chunk must write the same parquets as one big chunk."""
    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    monkeypatch.setattr("fbp.eval.candidates.methods", _FakeMethods)
    from fbp.eval.paths import cache_dir as cdir_fn

    rng = np.random.default_rng(0)
    src = cdir_fn("r/test") / "xx"
    src.mkdir(parents=True)
    for i in range(3):
        seg = synthetic_seg(rng)
        prob = np.zeros((3, *seg.shape), dtype=np.float32)
        prob[1] = seg == 1
        prob[0] = 1 - prob[1]
        _write_npz(src / f"p{i}.npz", seg, prob)

    from fbp.eval.candidates import tag_to_file

    variants = ("h0.5", "h1", "h2")
    n_single = process_country(
        "r", "test", "xx", (), workers=1, variant_names=variants, chunk_tags=100
    )
    single = {
        v: gpd.read_parquet(
            tmp_path / "candidates" / "r" / "test" / "xx" / tag_to_file(f"v:{v}")
        )
        for v in variants
    }

    for f in (tmp_path / "candidates" / "r" / "test" / "xx").glob("*.parquet"):
        f.unlink()

    n_chunked = process_country(
        "r", "test", "xx", (), workers=1, variant_names=variants, chunk_tags=1
    )
    chunked = {
        v: gpd.read_parquet(
            tmp_path / "candidates" / "r" / "test" / "xx" / tag_to_file(f"v:{v}")
        )
        for v in variants
    }

    assert n_single == n_chunked == 3
    for v in variants:
        pd_testing_assert_frame_equal_ignoring_order(single[v], chunked[v])


def pd_testing_assert_frame_equal_ignoring_order(
    a: pd.DataFrame, b: pd.DataFrame
) -> None:
    """Row order across chunks vs. one pass can differ; sort before comparing."""
    key = ["patch_id", "inst_id"]
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    assert a.patch_id.tolist() == b.patch_id.tolist()
    assert a.inst_id.tolist() == b.inst_id.tolist()
    assert [g.equals(h) for g, h in zip(a.geometry, b.geometry, strict=True)]
