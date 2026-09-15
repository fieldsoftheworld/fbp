"""Every variant setting the letter and its supplement report, by group.

Names are ``variants.run_variant`` strategy ids without the ``raw:v:`` prefix.
``python -m boundaryvote.grids sweep priors`` prints the union of the named
groups, one line, for the job scripts; ``--list`` prints the group names.
"""

import argparse

from .shipped import FTP_MIN, OURS, OURS_DISTANCE, OURS_NOFILTER

_MIN = "+A900"


def _pb(h: str, taus: tuple[str, ...]) -> list[str]:
    return [f"nbg-pb-h{h}-t{t}{_MIN}" for t in taus]


_TAUS = ("0", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9")
_TAUS8 = ("0", "0.1", "0.2", "0.3", "0.5", "0.7", "0.8", "0.9")

# The shipped method's own grid: seed depth x support threshold, searched
# with the minimum-size step in place. transfer.py ranks it on validation.
OURS_GRID = (
    *_pb("0", _TAUS),
    *_pb("0.01", _TAUS),
    *_pb("0.02", _TAUS),
    *_pb("0.05", _TAUS8),
    *_pb("0.1", _TAUS8),
    *_pb("0.2", _TAUS8),
    *_pb("0.3", ("0", "0.3", "0.5", "0.7", "0.8", "0.9")),
)

_DIST = OURS_DISTANCE.removeprefix("raw:v:").removesuffix(_MIN)
_OURS = OURS.removeprefix("raw:v:")
_FTP = FTP_MIN.removeprefix("raw:v:").removesuffix(_MIN)

GRIDS: dict[str, tuple[str, ...]] = {
    # Table II and Fig. 3 on every run.
    "main": (
        FTP_MIN.removeprefix("raw:v:"),
        _OURS,
        OURS_NOFILTER.removeprefix("raw:v:"),
    ),
    # Supplementary tau and h sweeps, and the transfer study of the method.
    "sweep": OURS_GRID,
    # Table III of the letter and the supplement's foreground/surface block.
    "ablation": (
        f"arg-pb-h0.01-t0.3{_MIN}",
        f"nbg-none{_MIN}",
        OURS_DISTANCE.removeprefix("raw:v:"),
        _DIST,
        _FTP,
    ),
    # Prior methods (Table I and the supplement's priors table), bare grids
    # plus the validation-chosen setting with the minimum-size step.
    "thr": (
        *(
            f"arg-thr-te{te}-tb{tb}"
            for te, tb in (
                ("0.3", "0.3"),
                ("0.3", "0.5"),
                ("0.4", "0.7"),
                ("0.4", "0.9"),
                ("0.5", "0.3"),
                ("0.5", "0.5"),
                ("0.5", "0.7"),
                ("0.5", "0.8"),
                ("0.5", "0.9"),
                ("0.6", "0.6"),
                ("0.7", "0.5"),
                ("0.7", "0.7"),
            )
        ),
        f"arg-thr-te0.4-tb0.7{_MIN}",
    ),
    "ucm": (
        *(f"nbg-ucm-k0.{k}" for k in range(1, 10)),
        f"nbg-ucm-k0.7{_MIN}",
    ),
    "iterh": (
        *(f"arg-iterh-a{a}" for a in (20000, 50000, 100000, 200000)),
        *(f"nbg-iterh-a{a}" for a in (5000, 20000, 50000, 100000, 200000)),
        f"arg-iterh-a200000{_MIN}",
    ),
    "swst": (
        *(f"nbg-swst-n30-m30-t{t}" for t in ("0.1", "0.2", "0.3")),
        *(f"nbg-swst-n80-m30-t{t}" for t in ("0.1", "0.2", "0.3", "0.4", "0.5")),
        *(f"nbg-swst-n150-m30-t{t}" for t in ("0.3", "0.4", "0.5")),
        f"nbg-swst-n80-m30-t0.3{_MIN}",
    ),
    # Zhu et al.: searched with the minimum-size step (see transfer.FAMILIES).
    "owt": (
        *(f"nbg-owt-l{lv}-h0{_MIN}" for lv in ("0.1", "0.2", "0.3", "0.4")),
        *(
            f"nbg-owt-l{lv}-h{h}{_MIN}"
            for lv in ("0.5", "0.6")
            for h in ("0", "0.01", "0.02", "0.05", "0.2")
        ),
        *(f"nbg-owt-l0.7-h{h}{_MIN}" for h in ("0.01", "0.02", "0.05", "0.2")),
        *(
            f"nbg-owt-l{lv}-h{h}{_MIN}"
            for lv in ("0.8", "0.9", "0.95")
            for h in ("0.05", "0.2")
        ),
        "nbg-owt-l0.5-h0",
    ),
    "gmm": ("nbg-gmm", f"nbg-gmm{_MIN}"),
    # Supplement: alternatives to the argmax foreground.
    "foreground": (
        "otsu-none",
        "pf50-none",
        "arg-close1",
        "arg-close2",
        "arg-pb-h0.1-t0.5",
        "arg-pb-h0.2-t0.5",
        "nbg-close1",
        "nbg-pb-h0.2-t0.5",
    ),
    # Supplement: line statistics on the distance-map form.
    "linestat": (
        *(f"nbg-ssup-h2-t{t}" for t in ("0.5", "0.6", "0.7")),
        "nbg-dsup-h2-t0.5",
        *(f"nbg-asup-h2-t{t}" for t in ("0.3", "0.4", "0.5", "0.6", "0.7")),
    ),
    # Supplement: other split proposals for the vote.
    "proposals": (
        f"nbg-pb-h0.1-t0.7{_MIN}",
        *(f"nbg-osup-h{h}-t0.7{_MIN}" for h in ("0.05", "0.02", "0")),
    ),
    # Supplement: steps added to the distance-map form.
    "steps": tuple(
        f"{_DIST}+{m}"
        for m in (
            "l3",
            "l5",
            "l10",
            "b1",
            "b2",
            "f50",
            "f200",
            "f-1",
            "m20",
            "m50",
            "m200",
        )
    ),
    # Supplement: minimum parcel size.
    "minsize": tuple(
        f"{base}+A{a}"
        for base in (_FTP, _DIST, OURS_NOFILTER.removeprefix("raw:v:"))
        for a in (450, 900, 1800, 3600)
    ),
}

# Which groups were scored on which run (test split; planet_b3 also on val).
RUN_GROUPS: dict[str, tuple[str, ...]] = {
    "planet_b3": tuple(GRIDS),
    "s2_b7": tuple(g for g in GRIDS if g not in ("steps", "minsize")),
    "planet_b7": ("main", "sweep", "ablation", "foreground"),
    "s2_b3": ("main", "sweep", "ablation", "foreground"),
}


def names(groups: tuple[str, ...] | list[str]) -> list[str]:
    """Union of the named groups, first occurrence wins."""
    out: list[str] = []
    for g in groups:
        for n in GRIDS[g]:
            if n not in out:
                out.append(n)
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups", nargs="*", choices=[*GRIDS, "all"])
    p.add_argument("--list", action="store_true", help="print the group names")
    args = p.parse_args()
    if args.list:
        for g, v in GRIDS.items():
            print(f"{g:12s} {len(v)}")
        return 0
    groups = tuple(GRIDS) if "all" in args.groups else tuple(args.groups)
    print(" ".join(names(groups)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
