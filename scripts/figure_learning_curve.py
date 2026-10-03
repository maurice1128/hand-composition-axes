"""Learning curves for the paper's motion-level result (stage I runs, seeds 0-9, 5 M frames).

Lines: training-pool success (the four two-skill training sequences) against frames, mean and
standard error over the 10 RL seeds, trailing average over 25 PPO updates. Markers: held-out
success (the two unseen three-skill sequences) for the same seeds at 1.5 M (stage F evaluation)
and 5 M (stage I evaluation), from `runs/rl/stageI_long/stageI_primary.json`.

    python scripts/figure_learning_curve.py --out docs/submission/figures/learning_curve.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "rl" / "stageI_long"
ARMS = [("keyframe", "keyframe (modular)", "#2446E0"), ("continuous", "continuous latent", "#6E8250"),
        ("raw", "plain PPO (no prior)", "#8A9297")]


def curves(arm, grid, win=25):
    out = []
    for d in sorted(glob.glob(str(BASE / f"{arm}_naive_s*"))):
        if not os.path.isdir(d):
            continue
        a = np.genfromtxt(os.path.join(d, "progress.csv"), delimiter=",", names=True)
        s = a["success"]
        trail = np.array([s[max(0, i - win + 1): i + 1].mean() for i in range(len(s))])
        out.append(np.interp(grid, a["timesteps"], trail))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/submission/figures/learning_curve.pdf")
    a = ap.parse_args()
    prim = json.loads((BASE / "stageI_primary.json").read_text())["strict"]["arms"]
    grid = np.linspace(0.1e6, 5e6, 99)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for arm, name, col in ARMS:
        c = curves(arm, grid)
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax.plot(grid / 1e6, m, color=col, lw=1.6, label=f"{name}, n={len(c)}")
        ax.fill_between(grid / 1e6, m - se, m + se, color=col, alpha=0.18, lw=0)
        h5 = prim[arm]["heldout"]
        ax.plot([5.0], [h5], marker="o", ms=5, color=col, mec="white", mew=0.6, zorder=5, clip_on=False)
        h15 = prim[arm].get("stageF_heldout_1p5M")
        if h15 is not None:
            ax.plot([1.5], [h15], marker="o", ms=5, color=col, mec="white", mew=0.6, zorder=5)
    ax.set_xlim(0, 5.0)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("training frames (millions)")
    ax.set_ylabel("success")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.93, 0.02), fontsize=7, handlelength=1.5)
    fig.tight_layout(pad=0.3)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    for arm, *_ in ARMS:
        print(arm, "held-out 1.5M", prim[arm].get("stageF_heldout_1p5M"), "5M", prim[arm]["heldout"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
