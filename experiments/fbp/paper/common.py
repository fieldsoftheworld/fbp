"""Shared results readers and row data for the tables and figures.

``tables.py`` and ``supp_tables.py`` both read the same four runs and the
same country names; this is the one place that says what they are.
"""

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[3] / "tables"

# (run, sensor, shown model/patch-size tag), in the order every table and
# figure lists them.
RUNS = [
    ("planet_b3", "PlanetScope", "B3"),
    ("planet_b7", "PlanetScope", "B7"),
    ("s2_b7", "Sentinel-2", "B7"),
    ("s2_b3", "Sentinel-2", "B3"),
]
SENSOR_SHORT = {"PlanetScope": "PS", "Sentinel-2": "S2"}

NAMES = {
    "belgium": "Belgium",
    "cambodia": "Cambodia",
    "croatia": "Croatia",
    "germany": "Germany",
    "latvia": "Latvia",
    "lithuania": "Lithuania",
    "portugal": "Portugal",
    "slovenia": "Slovenia",
    "south_africa": "South Africa",
    "sweden": "Sweden",
}

# The shipped method's row: bold label, light green fill (oursbg in the preamble).
OURS_LABEL = r"\textbf{BoundaryVote (Ours)}"
OURS_ROW = r"\rowcolor{oursbg}"


def macro(run: str, split: str = "test") -> pd.DataFrame:
    """Macro-averaged pixel scores for one run/split, vector reference only."""
    m = pd.read_csv(RES / run / split / "pixel" / "macro.csv")
    return m[m.reference == "vector"].set_index("strategy")


def country(run: str, split: str = "test") -> pd.DataFrame:
    """Per-country pixel scores for one run/split, vector reference only."""
    c = pd.read_csv(RES / run / split / "pixel" / "country.csv")
    return c[c.reference == "vector"]


def f(x: float, nd: int = 1) -> str:
    """Fraction as a percent string, e.g. 0.4231 -> '42.3'."""
    return f"{100 * x:.{nd}f}"
