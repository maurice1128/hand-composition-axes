"""Figures for docs/tmlr/PAPER_TMLR.md, drawn from stored per-seed results of finished sweeps only.

Figure 1: per-seed penalty for the zero-truth control, the permuted grids and the real grids.
Figure 2: penalty by class of target window for the two-clip OakInk-Image trajectories.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "tmlr" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def rel(dirs, budget=256, both=False):
    vals = []
    for d in dirs:
        by = {}
        for r in json.loads((ROOT / "runs" / d / "results.json").read_text(encoding="utf-8"))["results"]:
            if r["budget"] == budget:
                by.setdefault(r["seed"], {})[r["kind"]] = r
        vals += [100 * v["perframe"]["penalty"] / v["perframe"]["naive"]["mse_target"]
                 for v in by.values() if "perframe" in v and (not both or "modular" in v)]
    return np.array(vals)


def ci(x):
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


# ------------------------------------------------------------------ Figure 1
c256 = rel(["pc_easy_rerun"])
groups = [
    ("Zero-truth\nsynthetic", c256, "0.55"),
    ("OakInk-Image\npermuted", rel(["sham_oakink_category"]), "0.55"),
    ("OakInk-Image\ncategory\n× intent", rel(["oakink_official_category", "oakink_category_rest"], both=True), "C0"),
    ("TACO\npermuted\ncells", rel(["sham_taco_action_tool"]), "0.55"),
    ("TACO\naction\n× tool", rel(["taco_action_tool"]), "C0"),
    ("GRAB\npermuted", rel(["sham_grab_shape"]), "0.55"),
    ("GRAB\nshape ×\nfine intent", rel(["grab_shape_s4"]), "C0"),
]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.axhspan(c256.mean() - ci(c256), c256.mean() + ci(c256), color="0.9", zorder=0)
ax.axhline(0, color="0.3", lw=0.6)
for i, (name, x, col) in enumerate(groups):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(x)), x, s=6, color=col, alpha=0.55, lw=0)
    ax.errorbar(i + 0.3, x.mean(), yerr=ci(x), fmt="o", color="k", ms=3, capsize=2, lw=0.9)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=7)
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color="0.55", label="zero-truth or permuted grid"),
                   Line2D([], [], ls="", marker="o", ms=4, color="C0", label="real grid"),
                   Line2D([], [], ls="", marker="o", ms=4, color="k", label="mean and 95 % interval")],
          frameon=False, fontsize=7, loc="upper right")
ax.set_ylabel("Penalty (% of naive error)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_per_seed.{ext}", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2
W = {}
for v in ("aligned", "misaligned"):
    J = json.loads((ROOT / "runs" / f"pc_oakink_pair_{v}_wc" / "window_classes.json").read_text(encoding="utf-8"))["runs"]
    seeds = sorted({int(re.match(r"s(\d+)_", k).group(1)) for k in J})
    W[v] = {c: np.array([100 * (J[f"s{s}_perframe_b64_naive"][c]["mse"] - J[f"s{s}_perframe_b64_informed"][c]["mse"])
                         / J[f"s{s}_perframe_b64_naive"][c]["mse"] for s in seeds])
            for c in ("first", "straddle", "second")}
c64 = rel(["pc_easy_rerun"], 64)
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.axhspan(c64.mean() - ci(c64), c64.mean() + ci(c64), color="0.9", zorder=0)
ax.axhline(0, color="0.3", lw=0.6)
labels = ["First clip\n(labelled)", "Across\nthe join", "Second\nclip"]
for j, (v, col, name) in enumerate((("aligned", "C0", "Aligned"),
                                    ("misaligned", "C3", "Misaligned"))):
    m = [W[v][c].mean() for c in ("first", "straddle", "second")]
    e = [ci(W[v][c]) for c in ("first", "straddle", "second")]
    ax.bar(np.arange(3) + (j - 0.5) * 0.36, m, 0.36, yerr=e, color=col, alpha=0.8, capsize=2,
           error_kw={"lw": 0.8}, label=name)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("Penalty (% of naive error)")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_window_classes.{ext}", dpi=300)
plt.close(fig)
print("written", sorted(p.name for p in OUT.iterdir()))
