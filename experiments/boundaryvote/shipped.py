"""The one place that says which configuration is BoundaryVote as shipped.

Tables, figures, the bootstrap, the profiler and the transfer study all read
these, so changing the shipped setting is a one-line edit here.
"""

SURFACE = "pb"  # watershed surface: "pb" = boundary probability, "sup" = distance
H = 0.01  # seed depth in probability units; 0 would seed every regional minimum
TAU = 0.3  # a line is kept when at least this share of its pixels votes boundary
MIN_SIZE = "+A900"  # minimum parcel size, square metres, shared by every row

OURS_VARIANT = f"nbg-{SURFACE}-h{H:g}-t{TAU:g}"
OURS_NOFILTER = f"raw:v:{OURS_VARIANT}"
OURS = OURS_NOFILTER + MIN_SIZE
# The earlier form with distance-map proposals, kept as a supplementary ablation.
OURS_DISTANCE_VARIANT = "nbg-sup-h2-t0.7"
OURS_DISTANCE = f"raw:v:{OURS_DISTANCE_VARIANT}{MIN_SIZE}"
# The FTP pipeline as a variant, with and without the minimum-size step.
FTP = "rt:h2"
FTP_MIN = f"raw:v:arg-ws2{MIN_SIZE}"
