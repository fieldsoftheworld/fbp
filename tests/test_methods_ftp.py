"""FTPWatershed and ``ftp_pipeline`` against FTP's own pipeline."""

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from fbp import FTPWatershed, ftp_pipeline
from fbp.core.regions import watershed_regions


def two_fields() -> tuple[np.ndarray, np.ndarray]:
    H, W = 24, 48
    seg = np.zeros((H, W), dtype=np.uint8)
    prob = np.zeros((3, H, W), dtype=np.float32)
    prob[0] = 1.0
    seg[2:22, 2:22] = 1
    seg[2:22, 26:46] = 1
    prob[1][seg == 1] = 0.9
    prob[0][seg == 1] = 0.05
    prob[2][seg == 1] = 0.05
    return seg, prob


def test_ws_rule_matches_the_ftp_pipelines_union_of_instances() -> None:
    seg, prob = two_fields()
    got = FTPWatershed(seed_depth=2)(seg, prob)
    dist = distance_transform_edt(seg != 2).astype(np.float32)
    regions = watershed_regions(dist, seg == 1, 2.0, lines=False)
    want, _ = label(regions.labels > 0)
    assert (got > 0).sum() == (want > 0).sum() and got.max() == want.max()


def test_ws_rule_is_indifferent_to_the_foreground_mode() -> None:
    """FTP always splits the field class, regardless of the ``<fg>`` prefix."""
    seg, prob = two_fields()
    np.testing.assert_array_equal(
        FTPWatershed(seed_depth=2)(seg, prob),
        FTPWatershed(foreground="not_background", seed_depth=2)(seg, prob),
    )


def test_ftp_pipeline_convenience_matches_run() -> None:
    seg, prob = two_fields()
    np.testing.assert_array_equal(
        ftp_pipeline(seg, seed_depth=2.0), FTPWatershed(seed_depth=2)(seg, prob)
    )


def test_empty_field_mask_returns_zeros() -> None:
    seg = np.zeros((8, 8), dtype=np.uint8)
    prob = np.zeros((3, 8, 8), dtype=np.float32)
    prob[0] = 1.0
    assert FTPWatershed(seed_depth=2)(seg, prob).max() == 0
