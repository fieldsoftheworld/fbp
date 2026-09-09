"""Render the real per-patch panels for the FieldVote method-pipeline figure.

Reuses the validated multi_hmin_ensemble pipeline (same H_MIN_VALUES, same
consensus as the headline). For a chosen country it (1) ranks patches by how
much low-h over-fragmentation the consensus removes, then (2) renders six
panels per top candidate: RGB input, field/boundary probability, three
watershed segmentations (low/mid/high h), consensus polygons, an agreement
heatmap, and GT boundaries for reference.

Run from the repo root so data/ and logs/inference/ resolve.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import shapely.geometry

# Locations of the two external inputs, overridable so the renderer runs
# outside the machine it was written on. See scripts/06_figures/README.md.
#   FTW_PLANET_REPO  checkout that provides multi_hmin_ensemble, the
#                    polygon_metrics_eval helpers, the ftw_planet package,
#                    the FTW-Planet dataset under data/, and the per-h_min
#                    GeoJSONs under logs/inference/.
#   FIELDVOTE_STORE  run store holding predict_v2/<country>/<patch>.npz.
REPO = Path(os.environ.get("FTW_PLANET_REPO", "/u/agiri1/github/ftw-planet"))
STORE = Path(os.environ.get("FIELDVOTE_STORE", "/u/agiri1/fieldvote"))
sys.path.insert(0, str(REPO / "scripts/inference"))
import multi_hmin_ensemble as mhe  # noqa: E402

PV = STORE / "predict_v2"
# Set at runtime from the h-values actually exported for the country.
LOW_H, MID_H, HIGH_H = 2.0, 8.0, 12.0
FIELD_GREEN = "#639922"
BOUND_RED = "#D85A30"


def _pick_h(hs):
    hs = sorted(hs)
    return hs[0], hs[len(hs) // 2], hs[-1]


def _poly_iou(a, b):
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def agreement_per_poly(consensus_polys, polys_per_h, patch_id, theta=0.5):
    """Eq. 2: fraction of h-runs whose label map has a basin IoU>=theta."""
    n_h = len(polys_per_h)
    out = []
    for F in consensus_polys:
        hits = 0
        for h, by_patch in polys_per_h.items():
            cand = by_patch.get(patch_id, [])
            if any(_poly_iou(c, F) >= theta for c in cand):
                hits += 1
        out.append(hits / n_h if n_h else 0.0)
    return out


def _norm(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _ax():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def _extent(transform, shape):
    H, W = shape
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (W, H)
    return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]


def _draw_polys(ax, polys, edgecolor=BOUND_RED, lw=0.8, fc_values=None,
                facecolor="none"):
    cmap = plt.cm.RdYlGn
    for i, p in enumerate(polys):
        geoms = [p] if p.geom_type == "Polygon" else list(p.geoms)
        for g in geoms:
            xs, ys = g.exterior.xy
            fc = cmap(fc_values[i]) if fc_values is not None else facecolor
            ax.fill(xs, ys, facecolor=fc, edgecolor=edgecolor, lw=lw, alpha=0.95)


def render_patch(country, pid, sample, polys_per_h, outdir):
    import shapely.ops

    out = outdir / pid
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(sample["label"]) as src:
        gt = src.read(1)
        ltf = src.transform
        shape = gt.shape
    lext = _extent(ltf, shape)

    # Consensus first, so we can crop every panel to the field cluster
    cons = mhe.strategy_c_pixel_vote(polys_per_h, [sample], threshold_frac=0.5)
    cons_polys = cons.get(pid, [])
    all_polys = cons_polys + [p for h in polys_per_h for p in polys_per_h[h].get(pid, [])]
    if all_polys:
        minx, miny, maxx, maxy = shapely.ops.unary_union(all_polys).bounds
    else:
        minx, miny, maxx, maxy = lext[0], lext[2], lext[1], lext[3]
    side = max(maxx - minx, maxy - miny)
    pad = 0.10 * side
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = side / 2 + pad
    cropx = (cx - half, cx + half)
    cropy = (cy - half, cy + half)

    def finish(ax):
        ax.set_xlim(*cropx)
        ax.set_ylim(*cropy)

    # RGB input (Planet BGRN -> RGB via bands 3,2,1), both temporal windows
    for wkey, fname in (("window_a", "1_input.png"), ("window_b", "1b_input.png")):
        wpath = sample.get(wkey)
        if not wpath or not Path(wpath).exists():
            continue
        with rasterio.open(wpath) as src:
            nb = src.count
            bands = [3, 2, 1] if nb >= 3 else [1, 1, 1]
            rgb = np.dstack([src.read(b) for b in bands])
            rext = _extent(src.transform, rgb.shape[:2])
        fig, ax = _ax()
        ax.imshow(_norm(rgb), extent=rext, origin="upper")
        finish(ax)
        _save(fig, out / fname)

    # Probability map: field (green) + boundary (red) from stored softmax
    npz = PV / country / f"{pid}.npz"
    field_backdrop = None
    if npz.exists():
        prob = np.load(npz)["prob"].astype(np.float32)  # (3,H,W)
        field_p, bound_p = prob[1], prob[2]
        field_backdrop = field_p
        rgba = np.zeros((*field_p.shape, 4), dtype=np.float32)
        rgba[..., 1] = 0.6 * field_p           # green for field
        rgba[..., 0] = bound_p                  # red for boundary
        rgba[..., 3] = np.clip(field_p + bound_p, 0, 1)
        fig, ax = _ax()
        ax.imshow(rgba, extent=lext, origin="upper")
        finish(ax)
        _save(fig, out / "2_prob.png")

    def backdrop(ax):
        if field_backdrop is not None:
            ax.imshow(field_backdrop, cmap="Greens", vmin=0, vmax=1, alpha=0.5,
                      extent=lext, origin="upper")

    # Three watershed segmentations
    for tag, h in (("3_ws_low", LOW_H), ("4_ws_mid", MID_H), ("5_ws_high", HIGH_H)):
        polys = polys_per_h.get(h, {}).get(pid, [])
        fig, ax = _ax()
        backdrop(ax)
        _draw_polys(ax, polys, edgecolor=BOUND_RED, lw=0.7)
        finish(ax)
        _save(fig, out / f"{tag}.png")

    # Consensus polygons
    fig, ax = _ax()
    backdrop(ax)
    _draw_polys(ax, cons_polys, edgecolor="#27500A", lw=1.2)
    finish(ax)
    _save(fig, out / "6_consensus.png")

    # Agreement heatmap (consensus polys filled by a(F))
    agr = agreement_per_poly(cons_polys, polys_per_h, pid)
    fig, ax = _ax()
    _draw_polys(ax, cons_polys, edgecolor="#444444", lw=0.6, fc_values=agr)
    finish(ax)
    _save(fig, out / "7_agreement.png")

    # GT field/boundary annotation for reference (semantic 3-class label:
    # 1 = field interior, 2 = boundary), rendered like the probability panel.
    # True separated-instance polygons would need the GT polygon parquet,
    # which is not yet available in the shared store.
    fig, ax = _ax()
    gt_rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
    gt_rgba[..., 1] = 0.6 * (gt == 1)
    gt_rgba[..., 0] = 1.0 * (gt == 2)
    gt_rgba[..., 3] = np.clip((gt == 1) * 0.6 + (gt == 2) * 1.0, 0, 1)
    ax.imshow(gt_rgba, extent=lext, origin="upper")
    finish(ax)
    _save(fig, out / "8_gt.png")

    n_low = len(polys_per_h.get(LOW_H, {}).get(pid, []))
    return dict(pid=pid, n_low=n_low, n_cons=len(cons_polys),
                mean_agr=float(np.mean(agr)) if agr else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="belgium")
    ap.add_argument("--out", type=Path, default=STORE / "methodfig")
    ap.add_argument("--scan-limit", type=int, default=150)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--patch", default=None, help="render one specific patch")
    args = ap.parse_args()

    global LOW_H, MID_H, HIGH_H
    gjs = mhe.find_geojsons(args.country, REPO / "logs/inference")
    LOW_H, MID_H, HIGH_H = _pick_h(gjs.keys())
    print(f"h-values: {sorted(gjs.keys())}  low/mid/high = {LOW_H}/{MID_H}/{HIGH_H}")
    polys_per_h = {h: mhe.load_polygons_by_patch(g) for h, g in gjs.items()}
    ds = mhe.FTWPlanet(root="data", countries=[args.country], split="test",
                       transforms=None, load_boundaries=True)
    samples = {s.get("patch_id") or Path(s["window_a"]).stem: s for s in ds.records}
    print(f"{args.country}: {len(samples)} test patches")

    if args.patch:
        r = render_patch(args.country, args.patch, samples[args.patch], polys_per_h, args.out)
        print("RENDERED", r)
        return 0

    # Rank candidates: big low-h fragmentation removed, legible field count
    cand = []
    for i, (pid, s) in enumerate(samples.items()):
        if i >= args.scan_limit:
            break
        n_low = len(polys_per_h.get(LOW_H, {}).get(pid, []))
        cons = mhe.strategy_c_pixel_vote(polys_per_h, [s], threshold_frac=0.5)
        n_cons = len(cons.get(pid, []))
        with rasterio.open(s["label"]) as src:
            gt = src.read(1)
        n_gt = int(len(np.unique(gt)) - (1 if 0 in np.unique(gt) else 0))
        if 4 <= n_cons <= 18 and n_low > n_cons + 2:
            score = (n_low - n_cons)
            cand.append((score, pid, n_low, n_cons, n_gt))
    cand.sort(reverse=True)
    print(f"candidates (score,pid,n_low,n_cons,n_gt): {cand[:10]}")

    for _, pid, *_ in cand[: args.top]:
        r = render_patch(args.country, pid, samples[pid], polys_per_h, args.out)
        print("RENDERED", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
