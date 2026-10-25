# fbp

Field boundary prediction post-processing in Python.

`fbp` turns semantic field/background/boundary predictions into field instances.
It provides BoundaryVote, watershed and connected-component baselines, and
several published split-and-merge strategies behind one interface.

## Install

```sh
pip install ftw-fbp
```

The package imports as `fbp`.

For development:

```sh
git clone https://github.com/fieldsoftheworld/fbp.git
cd fbp
uv sync --extra dev
```

## Use

Inputs are a `(H, W)` semantic prediction and a `(3, H, W)` probability array
whose channels are background, field, and boundary. The result is a dense
`int32` instance-label array; zero is background.

```python
from fbp import run

instances = run(
    "nbg-pb-h0.01-t0.3+A900",
    segmentation,
    probabilities,
    px_m2=9.0,
)
```

The strategy string above uses non-background pixels as foreground, watersheds
the boundary-probability surface at depth `0.01`, retains proposed splits with
at least `0.3` boundary support, and removes fields smaller than 900 m².

Parse a strategy when you want to inspect or reuse its configuration:

```python
from fbp import parse

method = parse("nbg-pb-h0.01-t0.3+A900")
instances = method(segmentation, probabilities, px_m2=9.0)
```

For rasters too large for one call, `run_windowed` processes `core`-sized
windows with `halo` pixels of context. Each instance is kept by the window whose
core holds its centroid, and copies of a field that crosses a window boundary
are merged into one instance:

```python
from fbp import run_windowed

instances = run_windowed(
    "nbg-pb-h0.01-t0.3+A900",
    segmentation,
    probabilities,
    px_m2=9.0,
    core=8192,
    halo=512,
)
```

## FTW Global Data 2nd Edition

The 2nd Edition field boundaries were made with
`nbg-pb-h0.01-t0.5+R25+F10+G2+A900+q1` on 2.5 m probabilities (`px_m2=6.25`):

- `+R25` drops parcels whose mean field probability is under 0.25.
- `+F10` adds parcels split from the looser `pf10` foreground where the result
  covers less than half of them, and `+G2` skips that step when under 2% of the
  `pf10` mask is uncovered.
- `+q1` rounds the probabilities to the 1/255 grid the rasters store and runs
  h-maxima and the watershed on an integer backend. It needs the `fast` extra
  (SimpleITK and numba): `pip install 'ftw-fbp[fast]'`.

On published probability rasters this reproduces the published raw outlines:
all 112 parcels pixel for pixel on a 10 km window of tile `15TVG_0_0` (Iowa), and
647 of 648 (99.7% pixel for pixel) on tile `35MRT_0_0` (Rwanda).

## Layout

```text
src/fbp/core/      shared geometry and post-processing primitives
src/fbp/methods/   algorithms and the strategy registry
tests/             package tests
```

New algorithms subclass `fbp.Method` and register one or more strategy keywords
with `fbp.methods.registry.register`. Keep paper-specific evaluation, results,
figures, and LaTeX outside this package.

## Development

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
