"""Filesystem conventions shared by every stage."""

import os
from pathlib import Path

DENSE10 = (
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "latvia",
    "lithuania",
    "portugal",
    "slovenia",
    "south_africa",
    "sweden",
)

# Watershed depth sweep from main_grsl.tex eq. (depths). h is in pixels of the
# EDT-of-boundary surface (see ftp_compat.watershed_instances).
H_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)


def ftw_planet_repo() -> Path:
    return Path(
        os.environ.get("FTW_PLANET_REPO", Path.home() / "github" / "ftw-planet")
    ).resolve()


def out_root() -> Path:
    default = Path(__file__).resolve().parents[1] / "out"
    return Path(os.environ.get("FIELDVOTE_OUT", default)).resolve()


def h_tag(h: float) -> str:
    """0.75 -> 'h0p75', 12.0 -> 'h12'."""
    s = f"{h:g}".replace(".", "p")
    return f"h{s}"


def cache_dir(run: str) -> Path:
    return out_root() / "cache" / run


def candidates_dir(run: str) -> Path:
    return out_root() / "candidates" / run


def scores_dir(run: str) -> Path:
    return out_root() / "scores" / run
