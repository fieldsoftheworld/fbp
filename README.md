# fbp

Field boundary prediction post-processing in Python.

`fbp` turns background, field, and boundary predictions into field instances. It
includes BoundaryVote, the method behind the FTW Global Data 2nd Edition, along with
watershed, connected-component, and published split-and-merge baselines.

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

A method takes a `(H, W)` class map and a `(3, H, W)` array of background, field,
and boundary probabilities, and returns `int32` instance labels with 0 as background.
Each method is a frozen dataclass, so its settings can be written out and loaded
again as plain data.

```python
from fbp import BoundaryVote

method = BoundaryVote(
    foreground="not_background",
    seed_depth=0.01,
    min_line_support=0.3,
    min_area_m2=900,
)
instances = method(segmentation, probabilities, pixel_area_m2=9.0)
```

This method takes every pixel where field or boundary wins as foreground and floods
1 - p(boundary) from seeds at least 0.01 deep. It keeps a proposed split when at
least 30% of the pixels on the dividing line favor boundary, and drops fields under
900 m². The class docstrings describe every option.

For rasters too large for one call, `run_windowed` processes `core`-sized windows
with `halo` pixels of context on each side. The window whose core holds a field's
centroid keeps that field, and copies of a field that crosses a window boundary are
merged into one instance.

```python
from fbp import run_windowed

instances = run_windowed(
    method,
    segmentation,
    probabilities,
    pixel_area_m2=9.0,
    core=8192,
    halo=512,
)
```

Only one window of the inputs is read at a time. `segmentation` and `probabilities`
can be any objects with a `.shape` whose numpy-style slices are arrays, such as a
reader of raster windows, so the whole raster never has to be in memory. The labels
are int32 and are relabelled in row strips; pass `out=` to write them into a
preallocated int32 array or memmap. A 40,032 x 40,032 raster then needs the 6.4 GB
label array plus one window's working memory.

## FTW Global Data 2nd Edition

`fbp.FTW_2E` is the setting used for the 2nd Edition field boundaries, run on 2.5 m
probabilities:

```python
from fbp import FTW_2E

instances = FTW_2E(segmentation, probabilities, pixel_area_m2=6.25)
```

Besides the BoundaryVote settings above (with `min_line_support=0.5`), it drops
fields whose mean field probability is under 0.25. It then splits the looser
foreground of pixels with field probability above 0.10 and adds each resulting field
that the first pass covers less than half of, which recovers fields whose field
probability rarely wins the argmax. That second pass is skipped when less than 2% of
the looser foreground is uncovered. With `fast=True`, the probabilities are rounded
to the 1/255 grid of the published uint8 rasters, and the watershed runs on integer
levels. This needs the `fast` extra (SimpleITK and numba):

```sh
pip install 'ftw-fbp[fast]'
```

On the published probability rasters, `FTW_2E` reproduces the published raw outlines.
It matches all 112 fields pixel for pixel on a 10 km window of tile `15TVG_0_0`
(Iowa), and 647 of 648 fields (99.7% pixel for pixel) on tile `35MRT_0_0` (Rwanda).

## Layout

```text
src/fbp/core/      geometry and post-processing building blocks
src/fbp/methods/   the methods, one module each
tests/             package tests
```

A new method subclasses `fbp.Method` and implements `split`. Keep paper-specific
evaluation, results, figures, and LaTeX outside this package.

## Development

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
