"""The headline figure: a training-free statistic against measured difficulty.

Six axes, each one a full paired sweep costing hours of GPU, plotted against a
statistic that takes minutes and trains nothing. The point of the figure is that
the sign is readable before any of that GPU is spent -- axes left of zero came
back null, axes right of it did not.

Drawn from ``runs/screen_prediction.json`` so the figure cannot drift from the
numbers in the text; regenerate it after any sweep that adds seeds.

Two honesty constraints are built into the drawing rather than left to the
caption. Six points is few, so the leave-one-out range is annotated beside the
correlation instead of the coefficient standing alone. And the fit line is drawn
across the observed range only -- extending it would suggest the screen has been
checked where no sweep exists.

    python scripts/figure_screen_scatter.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

#: Short labels and the dataset each axis belongs to, for colour and annotation.
AXES = {
    "oakink_category": ("OakInk-Image", "category x intent"),
    "oakink_attr": ("OakInk-Image", "affordance x intent"),
    "category": ("OakInk-Image", "id-prefix x intent"),
    "oakink_class": ("OakInk-Image", "class x intent"),
    "shape": ("GRAB", "shape x intent"),
    "oakink2_fine": ("OakInk2", "transitions"),
}

#: ``dexycb_fine`` is deliberately absent. Its sweep ran at ``min_chains = 1``,
#: the value that disables the restriction, and ``check_informed_coverage.py``
#: returns TOO THIN on that configuration -- the informed arm sees 2.00 of 4
#: held compositions. Plotting it would show a point the paper withdraws.
#: ``oakink_class`` replaces it and is the axis that broke the correlation: the
#: screen ranked it fifth of six and it returned the largest difficulty measured.

COLOUR = {"OakInk-Image": "#1f4e79", "GRAB": "#a8501e",
          "DexYCB": "#4a7c59", "OakInk2": "#6b4a7c"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="runs/screen_prediction.json")
    ap.add_argument("--out", default="docs/figures/screen_scatter.pdf")
    ap.add_argument("--width", type=float, default=3.5, help="inches; IEEE column")
    ap.add_argument("--height", type=float, default=2.7)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,          # embed TrueType; IEEE PDF checks reject Type 3
    })

    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    corr = report["correlation"]["baseline penalty (difficulty)"]

    # The plotted quantity is the compositional difficulty the sweep measured --
    # the baseline arm's penalty, not the modular advantage. The screen predicts
    # where compositions are hard, not which architecture wins there, and the
    # caption has to say so. ``screen_prediction.json`` records the advantage but
    # not the baseline penalty, so that is recovered from the sweeps themselves
    # rather than duplicated into a second file that could drift.
    from analyse_screen_prediction import MEASURED, paired_by_seed

    xs, ys, labels, colours = [], [], [], []
    for key, (dataset, axis) in AXES.items():
        row = report[key]
        by_seed = paired_by_seed(MEASURED[key])
        xs.append(row["excess"])
        ys.append(float(np.mean([v["perframe"] for v in by_seed.values()])))
        labels.append(f"{axis}\n(n={row['n']})")
        colours.append(COLOUR[dataset])

    x, y = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    ax.axhline(0.0, color="#999999", lw=0.5, zorder=1)
    ax.axvline(0.0, color="#999999", lw=0.5, zorder=1)

    b, a = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 2)
    ax.plot(xr, a + b * xr, color="#333333", lw=0.8, zorder=2)

    for xi, yi, lab, c in zip(x, y, labels, colours):
        ax.scatter([xi], [yi], s=26, color=c, zorder=3, edgecolor="white", lw=0.5)

    # Placed by hand, with an alignment per label rather than an offset alone:
    # the three OakInk-Image axes sit within 0.013 of each other on x and any
    # uniform offset puts two labels on top of a third.
    place = {                       # (dx, dy, horizontal alignment)
        0: (0, 9, "center"),        # category x intent      -- above
        1: (0, -20, "center"),      # affordance x intent    -- below
        2: (-8, 0, "right"),        # id-prefix x intent     -- left
        3: (7, 2, "left"),          # GRAB shape x intent    -- right
        4: (0, 9, "center"),        # DexYCB subject x shape -- above
        5: (0, -20, "center"),      # OakInk2 transitions    -- below
    }
    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        dx, dy, ha = place[i]
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                    fontsize=5.8, color="#333333", linespacing=1.15, ha=ha)

    # Room for the labels that sit outside the data range.
    pad_x = 0.22 * (x.max() - x.min())
    pad_y = 0.28 * (y.max() - y.min())
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    ax.set_xlabel("interaction excess (training-free screen)")
    ax.set_ylabel("measured compositional penalty")
    ax.text(0.03, 0.95,
            f"$r$ = {corr['pearson_r']:.3f} ($p$ = {corr['pearson_p']:.3f})\n"
            f"$\\rho$ = {corr['spearman_rho']:.3f} ($p$ = {corr['spearman_p']:.3f})\n"
            f"leave-one-out $r$: {corr['loo_r_min']:.2f}–{corr['loo_r_max']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
            linespacing=1.35)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")
    print(f"  r = {corr['pearson_r']:.3f}  p = {corr['pearson_p']:.4f}  "
          f"loo {corr['loo_r_min']:.3f}-{corr['loo_r_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
