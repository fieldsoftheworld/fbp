"""Per-country PQ change of the boundary veto over the FTP pipeline, four model/sensor runs.

Regenerate: ~/github/ftw-planet/.venv/bin/python experiments/figures/per_country.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from figstyle import BASELINE, COLUMN, PALETTE, figure, save, use_style

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
RUNS = [
    ("planet_b3", "test", "PS B3", PALETTE["blue"], "o"),
    ("planet_b7", "test", "PS B7", PALETTE["cyan"], "s"),
    ("s2_b7", "test/pixel", "S2 B7", PALETTE["red"], "D"),
    ("s2_b3", "test/pixel", "S2 B3", PALETTE["orange"], "^"),
]
OURS = "raw:v:nbg-sup-h2-t0.7+A900"
BASE = "raw:v:arg-ws2+A900"
NAMES = {
    "south_africa": "South Africa",
    "belgium": "Belgium",
    "cambodia": "Cambodia",
    "croatia": "Croatia",
    "germany": "Germany",
    "latvia": "Latvia",
    "lithuania": "Lithuania",
    "portugal": "Portugal",
    "slovenia": "Slovenia",
    "sweden": "Sweden",
}


def deltas(run: str, sub: str) -> pd.Series:
    c = pd.read_csv(RES / run / sub / "country.csv")
    c = c[c.reference == "vector"]
    base = c[c.strategy == BASE].set_index("country").pq
    ours = c[c.strategy == OURS].set_index("country").pq
    return 100 * (ours - base)


def main() -> None:
    use_style(family="serif")
    table = pd.DataFrame({label: deltas(run, sub) for run, sub, label, _, _ in RUNS})
    order = table.mean(axis=1).sort_values().index  # smallest mean gain at the bottom
    table = table.loc[order]

    fig, ax = figure(width=COLUMN, ratio=0.6)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    y = np.arange(len(order))
    for _run, _sub, label, color, marker in RUNS:
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
