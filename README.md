# fbp

Field boundary prediction post-processing in Python.

`fbp` turns semantic field/background/boundary predictions into field instances.
It provides BoundaryVote, watershed and connected-component baselines, and
several published split-and-merge strategies behind one interface.

## Install

```sh
pip install fbp
```

For development:

```sh
git clone https://github.com/taylor-geospatial/fbp.git
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
