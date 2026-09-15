"""compare.py reads a pre-built macro.csv when present, else pools from scratch."""

from pathlib import Path

import pandas as pd
import pytest

from fbp.eval.compare import load_macro
from fbp.eval.matching import THRESHOLDS
from fbp.eval.score import BINS


def _macro_row(strategy: str, pq: float) -> dict:
    return {
        "run": "r",
        "split": "test",
        "protocol": "pixel",
        "reference": "vector",
        "strategy": strategy,
        "n_countries": 1,
        "n_patches": 1,
        "n_scored": 1,
        "pq": pq,
        "sq": pq,
        "rq": 1.0,
    }


def test_load_macro_prefers_results_csv_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fbp.eval.compare.results_root", lambda: tmp_path / "results")
    results = tmp_path / "results" / "r" / "test" / "pixel"
    results.mkdir(parents=True)
    pd.DataFrame([_macro_row("cc", 0.5)]).to_csv(results / "macro.csv", index=False)

    got = load_macro("r", "test", "pixel", ("a",))
    assert got.iloc[0].strategy == "cc"
    assert got.iloc[0].pq == pytest.approx(0.5)


def test_load_macro_pools_from_scores_when_no_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOUNDARYVOTE_OUT", str(tmp_path))
    monkeypatch.setattr(
        "fbp.eval.compare.results_root", lambda: tmp_path / "results-missing"
    )

    from fbp.eval import paths

    root = paths.scores_dir("r/test/pixel")
    root.mkdir(parents=True)
    row = {
        "run": "r",
        "split": "test",
        "protocol": "pixel",
        "reference": "vector",
        "strategy": "cc",
        "country": "a",
        "patch_id": "p",
        "n_gt": 1,
        "n_pred": 1,
        "iou_sum_50": 1.0,
        "gt_split": 0,
        "pred_merge": 0,
        "gt_missed": 0,
        "pred_spurious": 0,
        "px_ha": 0.0009,
        "t_build": 0.0,
    }
    for t in THRESHOLDS:
        k = f"{int(round(t * 100)):02d}"
        row[f"tp_{k}"], row[f"fp_{k}"], row[f"fn_{k}"] = 1, 0, 0
    for b in BINS:
        row[f"tp_50_{b}"] = 1 if b == "large" else 0
        row[f"fp_50_{b}"] = row[f"fn_50_{b}"] = 0
        row[f"iou_sum_50_{b}"] = 1.0 if b == "large" else 0.0
    pd.DataFrame([row]).to_parquet(root / "patches_a.parquet", index=False)

    got = load_macro("r", "test", "pixel", ("a",))
    assert got.iloc[0].strategy == "cc"
    assert got.iloc[0].pq == pytest.approx(1.0)
