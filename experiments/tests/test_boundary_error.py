"""Outline extraction and the printed chamfer/count-delta report on synthetic data."""

import pandas as pd
import pytest
import shapely.geometry

from fieldvote.boundary_error import outline_mask, report


def test_outline_mask_is_the_border_not_the_interior() -> None:
    m = outline_mask((10, 10), shapely.geometry.box(2, 2, 8, 8))
    assert m is not None
    assert not m[5, 5]  # interior
    assert m[2, 5]  # top edge
    assert 0 < m.sum() < 36  # a proper subset of the filled 6x6 block


def test_outline_mask_returns_none_for_empty_geometry() -> None:
    off_canvas = shapely.geometry.box(100, 100, 110, 110)
    assert outline_mask((10, 10), off_canvas) is None


def make_rows(
    variant: str,
    pairs: list[tuple[str, str, int, float]],
    n_pred: int,
    n_gt: int,
) -> list[dict[str, str | int | float]]:
    """`pairs` is [(country, patch_id, gt_id, chamfer_m), ...] for one variant."""
    rows = [
        {
            "variant": variant,
            "country": c,
            "patch_id": p,
            "gt_id": g,
            "chamfer_m": chamfer,
        }
        for c, p, g, chamfer in pairs
    ]
    rows.append(
        {
            "variant": variant,
            "country": "a",
            "patch_id": "p1",
            "gt_id": -1,
            "n_pred": n_pred,
            "n_gt": n_gt,
        }
    )
    return rows


def test_report_prints_own_and_common_chamfer_and_count_deltas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # v1 matches both reference parcels; v2 only matches the one v1 also matches.
    v1 = make_rows("v1", [("a", "p1", 0, 1.0), ("a", "p1", 1, 3.0)], n_pred=2, n_gt=2)
    v2 = make_rows("v2", [("a", "p1", 0, 2.0)], n_pred=1, n_gt=2)
    df = pd.DataFrame(v1 + v2)

    report(df, ["v1", "v2"])
    out = capsys.readouterr().out
    lines = {ln.split()[0]: ln.split()[1:] for ln in out.splitlines() if ln.split()}

    own1, common1, dabs1, drel1 = (float(x) for x in lines["v1"])
    assert own1 == pytest.approx(2.0)  # mean(1.0, 3.0)
    assert common1 == pytest.approx(1.0)  # only gt_id 0 is common to both variants
    assert dabs1 == pytest.approx(0.0)
    assert drel1 == pytest.approx(0.0)

    own2, common2, dabs2, drel2 = (float(x) for x in lines["v2"])
    assert own2 == pytest.approx(2.0)
    assert common2 == pytest.approx(2.0)
    assert dabs2 == pytest.approx(1.0)  # |1 - 2|
    assert drel2 == pytest.approx(0.5)  # 1 / 2

    assert "matched by every variant: 1" in out
