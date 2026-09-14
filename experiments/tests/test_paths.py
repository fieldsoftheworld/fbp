"""Filesystem conventions: h_tag formatting and env-driven output roots."""

from pathlib import Path

import pytest

from boundaryvote import paths


def test_h_tag_formats_depths() -> None:
    assert paths.h_tag(0.75) == "h0p75"
    assert paths.h_tag(12.0) == "h12"
    assert paths.h_tag(0.5) == "h0p5"


def test_ftw_planet_repo_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FTW_PLANET_REPO", str(tmp_path))
    assert paths.ftw_planet_repo() == tmp_path.resolve()


def test_out_root_and_scores_dir_honour_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FIELDVOTE_OUT", str(tmp_path))
    assert paths.out_root() == tmp_path.resolve()
    assert paths.scores_dir("planet_b3/test") == tmp_path.resolve() / "scores" / (
        "planet_b3/test"
    )


def test_out_root_default_is_relative_to_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIELDVOTE_OUT", raising=False)
    assert paths.out_root() == Path(paths.__file__).resolve().parents[1] / "out"
