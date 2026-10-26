"""Methods that turn (background, field, boundary) predictions into field instances."""

from fbp.methods.base import Method
from fbp.methods.boundaryvote import BoundaryVote
from fbp.methods.components import ClosedComponents, Components
from fbp.methods.ftp import FTPWatershed, ftp_pipeline
from fbp.methods.gmm import GaussianMixture
from fbp.methods.iterh import IterativeWatershed
from fbp.methods.owt import OrientedWatershed
from fbp.methods.swst import StochasticWatershed
from fbp.methods.thresholds import ExtentBoundaryThreshold
from fbp.methods.ucm import UltrametricContours

# The setting used for the FTW Global Data 2nd Edition, on 2.5 m probabilities.
FTW_2E = BoundaryVote(
    foreground="not_background",
    surface="boundary_probability",
    seed_depth=0.01,
    min_line_support=0.5,
    min_mean_field_probability=0.25,
    refill_field_probability=0.10,
    refill_min_uncovered=0.02,
    min_area_m2=900,
    fast=True,
)

__all__ = [
    "FTW_2E",
    "BoundaryVote",
    "ClosedComponents",
    "Components",
    "ExtentBoundaryThreshold",
    "FTPWatershed",
    "GaussianMixture",
    "IterativeWatershed",
    "Method",
    "OrientedWatershed",
    "StochasticWatershed",
    "UltrametricContours",
    "ftp_pipeline",
]
