"""Per-region thresholds from a two-component Gaussian mixture (Amin et al. 2025).

Amin et al. fit one mixture to a study site's extent probabilities and one to
its boundary probabilities, over every predicted patch of the site, and use
the weight-averaged component means as the two thresholds. Regions here are
the ten countries, and a three-class model's extent probability is
``1 - p_bg``. No labels enter the fit, so it is an inference-time step.

Writes ``out/gmm/<run>_<split>.json``: ``{country: {"te": .., "tb": .., "n_px": ..}}``.

    python -m boundaryvote.gmm_thresholds --run planet_b3 --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import DENSE10, cache_dir, out_root

PER_PATCH = 20_000  # pixels sampled from each patch for the fit


def fit_gmm1d(
    x: np.ndarray, iters: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM for a two-component 1-D mixture; returns (weights, means, variances)."""
    x = x.astype(np.float64)
    mu = np.percentile(x, [25.0, 75.0])
    if mu[1] - mu[0] < 1e-6:
        mu = np.array([x.min(), x.max()], dtype=np.float64)
    var = np.full(2, max(x.var() / 4.0, 1e-6))
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        d = x[:, None] - mu
        logp = -0.5 * (d**2 / var + np.log(2.0 * np.pi * var)) + np.log(w)
        logp -= logp.max(axis=1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-12
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / nk + 1e-8
    return w, mu, var


def gmm_threshold(x: np.ndarray) -> float:
    """Weighted mean of the component means, Amin et al.'s separation point."""
    w, mu, _ = fit_gmm1d(x)
    return float((w * mu).sum())


def sample_country(
    run: str, split: str, country: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Extent and boundary probabilities sampled from every cached patch."""
    rng = np.random.default_rng(seed)
    ext, bnd = [], []
    for npz in sorted((cache_dir(f"{run}/{split}") / country).glob("*.npz")):
        with np.load(npz) as d:
            prob = d["prob"].astype(np.float32)
        flat_bg, flat_bnd = prob[0].ravel(), prob[2].ravel()
        idx = rng.choice(flat_bg.size, size=min(PER_PATCH, flat_bg.size), replace=False)
        ext.append(1.0 - flat_bg[idx])
        bnd.append(flat_bnd[idx])
    return np.concatenate(ext), np.concatenate(bnd)


def fit_country(run: str, split: str, country: str) -> dict[str, float]:
    """One extent and one boundary threshold for a country."""
    ext, bnd = sample_country(run, split, country)
    return {"te": gmm_threshold(ext), "tb": gmm_threshold(bnd), "n_px": int(ext.size)}


def thresholds_path(run: str, split: str) -> Path:
    """Where a run/split's fitted thresholds live."""
    return out_root() / "gmm" / f"{run}_{split}.json"


def load(run: str, split: str) -> dict[str, tuple[float, float]]:
    """``{country: (te, tb)}`` from disk, empty when nothing was fitted."""
    f = thresholds_path(run, split)
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    return {c: (v["te"], v["tb"]) for c, v in raw.items()}


def ensure(
    run: str, split: str, countries: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Fit any country missing from disk, save, and return ``{country: (te, tb)}``."""
    f = thresholds_path(run, split)
    raw = json.loads(f.read_text()) if f.exists() else {}
    for c in countries:
        if c not in raw:
            raw[c] = fit_country(run, split, c)
            print(f"{run}/{split}/{c}: te={raw[c]['te']:.3f} tb={raw[c]['tb']:.3f}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(raw, indent=1, sort_keys=True))
    return {c: (v["te"], v["tb"]) for c, v in raw.items()}


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default="planet_b3")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--countries", nargs="*", default=list(DENSE10))
    args = p.parse_args()
    thr = ensure(args.run, args.split, tuple(args.countries))
    for c, (te, tb) in thr.items():
        print(f"{c:14s} te={te:.3f} tb={tb:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
