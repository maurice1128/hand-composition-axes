"""Figure 1 (teaser) for docs/tmlr/PAPER_TMLR.md. Two schematic panels (what the training set contains when a label
covers the whole trajectory, and when it covers only the first half) and one bar panel with the measured penalties.
The schematic matches the project-page video; every number is from Table 3 (40 seeds per version; the zero-truth
control at the same budget, 20 seeds), recomputed here from runs/*/results.json.

Output: docs/tmlr/figures/fig0_teaser.png and .pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tmlr" / "figures"
RED, BLUE, GREEN, GOLD = "#d62728", "#1f77b4", "#2ca02c", "#e6a100"
INK, PANEL = "#1a1a1a", "#f3f3f3"
plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "pdf.fonttype": 42})


def rel(d, b=64):
    return np.array([100 * r["penalty"] / r["naive"]["mse_target"]
                     for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]
                     if r["kind"] == "perframe" and r["budget"] == b])


ctrl, al, mis = rel("pc_easy_rerun"), rel("pc_oakink_pair_aligned"), rel("pc_oakink_pair_misaligned")

# Recordings: (label colour, first-half motion, second-half motion).
HELD_A = [(RED, RED, RED), (RED, RED, RED)]
TRAIN_A = [(BLUE, BLUE, BLUE), (GREEN, GREEN, GREEN), (BLUE, BLUE, BLUE), (GREEN, GREEN, GREEN)]
HELD_B = [(RED, RED, BLUE), (RED, RED, GREEN)]
TRAIN_B = [(BLUE, BLUE, RED), (GREEN, GREEN, BLUE), (BLUE, BLUE, GREEN), (GREEN, GREEN, RED)]


def recording(ax, x, y, lab, c1, c2, leak):
    ax.add_patch(Rectangle((x, y + 0.012), 0.07, 0.07, fc=lab, ec=INK, lw=0.6))
    ax.text(x + 0.035, y + 0.047, "L", ha="center", va="center", fontsize=7, color="white", weight="bold")
    ax.add_patch(Rectangle((x + 0.09, y), 0.28, 0.095, fc=c1, ec="white", lw=1.2))
    ax.add_patch(Rectangle((x + 0.37, y), 0.28, 0.095, fc=c2, ec="white", lw=1.2))
    if leak:
        ax.add_patch(Rectangle((x + 0.363, y - 0.012), 0.294, 0.119, fc="none", ec=GOLD, lw=2.2))


def schematic(ax, title, held, train, note, note_col):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, weight="bold", loc="left")
    for (y, h, lab, col) in ((0.69, 0.29, "held\nout", RED), (0.09, 0.57, "training\nset", INK)):
        ax.add_patch(FancyBboxPatch((0.03, y), 0.94, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=PANEL, ec=col, lw=1.2))
        ax.text(0.06, y + h - 0.03, lab, fontsize=8, color=col, weight="bold", va="top")
    for i, r in enumerate(held):
        recording(ax, 0.29, 0.81 - i * 0.12, *r, False)
    for i, r in enumerate(train):
        recording(ax, 0.29, 0.47 - i * 0.12, *r, r[0] != RED and r[2] == RED)
    ax.text(0.5, 0.0, note, ha="center", fontsize=8.5, color=note_col, weight="bold")


fig = plt.figure(figsize=(7.2, 2.9), dpi=200)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.30, left=0.01, right=0.99, top=0.86, bottom=0.12)
schematic(fig.add_subplot(gs[0]), "(a) Label covers all of it", HELD_A, TRAIN_A,
          "no held-out motion in training", GREEN)
schematic(fig.add_subplot(gs[1]), "(b) Label covers half", HELD_B, TRAIN_B,
          "held-out motion enters under other labels", GOLD)
ax = fig.add_subplot(gs[2])
vals = [al.mean(), mis.mean()]
errs = [stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) for x in (al, mis)]
ax.bar([0, 1], vals, 0.6, yerr=errs, color=[RED, "0.45"], capsize=3, error_kw={"lw": 0.8})
ax.axhline(ctrl.mean(), ls="--", color=INK, lw=1)
# The dashed line is the zero-truth control at the same budget; its value is given in the caption.
for i, v in enumerate(vals):
    ax.text(i, v + errs[i] + 0.8, f"+{v:.1f} %", ha="center", fontsize=8.5, weight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["(a)", "(b)"])
ax.set_ylabel("Penalty (% of naive error)", fontsize=8)
ax.set_ylim(0, 25)
ax.set_xlim(-0.6, 1.6)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(c) Penalty", fontsize=9.5, weight="bold", loc="left")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig0_teaser.{ext}", facecolor="white")
print("written", OUT / "fig0_teaser.png", [round(x, 2) for x in vals], round(ctrl.mean(), 2))
