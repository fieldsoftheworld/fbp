"""Per-country PQ change of BoundaryVote over the FTP pipeline, four model/sensor runs.

Regenerate: ~/github/ftw-planet/.venv/bin/python -m fbp.paper.figures.per_country
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fbp.paper.common import NAMES, RES
from fbp.paper.figures.figstyle import (
    BASELINE,
    COLUMN,
    PALETTE,
    figure,
    save,
    use_style,
)
from fbp.paper.shipped import FTP_MIN as BASE
from fbp.paper.shipped import OURS

ROOT = Path(__file__).resolve().parents[4]
RUNS = [
    ("planet_b3", "PS B3", PALETTE["blue"], "o"),
    ("planet_b7", "PS B7", PALETTE["cyan"], "s"),
    ("s2_b7", "S2 B7", PALETTE["red"], "D"),
    ("s2_b3", "S2 B3", PALETTE["orange"], "^"),
]


def deltas(run: str) -> pd.Series:
    c = pd.read_csv(RES / run / "test" / "pixel" / "country.csv")
    c = c[c.reference == "vector"]
    base = c[c.strategy == BASE].set_index("country").pq
    ours = c[c.strategy == OURS].set_index("country").pq
    return 100 * (ours - base)


def main() -> None:
    use_style(family="serif")
    table = pd.DataFrame({label: deltas(run) for run, label, _, _ in RUNS})
    order = table.mean(axis=1).sort_values().index  # smallest mean gain at the bottom
    table = table.loc[order]

    fig, ax = figure(width=COLUMN, ratio=0.6)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    y = np.arange(len(order))
    for _run, label, color, marker in RUNS:
        ax.scatter(
            table[label].to_numpy(),
            y,
            s=16,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.3,
            label=label,
            zorder=3,
        )
    means = table.mean(axis=1).to_numpy()
    ax.scatter(
        means,
        y,
        marker="|",
        s=90,
        color="black",
        linewidth=1.0,
        zorder=4,
        label="Mean of Four",
    )
    ax.set_yticks(y, [NAMES[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("PQ Gain over the FTP Pipeline (Points)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    legend = ax.legend(
        loc="lower right",
        fontsize=6.5,
        frameon=True,
        framealpha=1.0,
        edgecolor="#AAAAAA",
        borderpad=0.4,
        handletextpad=0.3,
    )
    legend.get_frame().set_linewidth(0.4)
    ax.margins(x=0.08)
    save(fig, str(ROOT / "figures" / "rev_percountry"))
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
