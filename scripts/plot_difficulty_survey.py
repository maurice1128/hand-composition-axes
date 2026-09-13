"""The survey figure: how much compositional difficulty does each dataset have?

This is the paper's central claim in one picture. Every dataset is measured with
the same instrument, and the instrument's own calibration is on the same axes:
a planted-hard synthetic control at one end, an interpolable synthetic control
in the middle, and DexYCB -- which has no compositional design at all -- at the
other.

Two panels, because the second is what makes the first interpretable:

* **Left, baseline penalty.** How much a bandwidth-matched monolithic prior pays
  for an unseen composition. This is a property of the *dataset and axis*, not
  of any architecture. It spans 200x.
* **Right, modular advantage.** How much less the modular prior pays, with its
  paired-test p-value. A bar here is only meaningful where the left panel shows
  the dataset poses a question at all -- which is why the two are shown together
  and why datasets with near-zero difficulty are greyed out rather than reported
  as negative results.

The row labelled "category->intent" used to read `runs/oakink_n70_v2`, which
is the deprecated id-prefix sweep, so the figure showed that axis under the
official one's name. Reading numbers from artifacts does not help when the
label names a different run than the one being read.

Numbers are read from `runs/*/results.json`, never typed in, so the figure
cannot drift from the artifacts the way the prose repeatedly did.

    python scripts/plot_difficulty_survey.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_paired import t_test_one_sample  # noqa: E402

#: (display name, run dir, baseline kind, treatment kind, is-real-data)
#:
#: Every real-data row points at a ``_v2`` directory. The pre-fix runs were
#: measured on bundles carrying dead degrees of freedom -- DexYCB's thumb
#: abduction, OakInk2's index PIP flexion -- and must not be plotted beside
#: corrected ones. The synthetic rows are unchanged: generated data never goes
#: through MANO retargeting, so neither bug could reach them.
SOURCES = [
    ("OakInk-Image\ncategory x intent",
     ["runs/oakink_official_category", "runs/oakink_category_rest"],
     "perframe", "modular", True),
    ("OakInk-Image\naffordance x intent",
     ["runs/oakink_official_attr", "runs/oakink_attr_more"],
     "perframe", "modular", True),
    ("OakInk-Image\nid-prefix (deprecated)", "runs/oakink_n70_v2",
     "perframe", "modular", True),
    ("GRAB\nshape x intent", "runs/grab_shape_v2", "perframe", "modular", True),
    ("synthetic\nvia-point", "runs/pc_hard", "perframe", "modular", False),
    ("synthetic\ninterpolated", "runs/pc_easy", "perframe", "modular", False),
    ("OakInk2\nannotated primitives", "runs/oakink2_paired_v2", "perframe", "modular", True),
    ("GRAB\nshape x intent class", "runs/grab_shapeclass", "perframe", "modular", True),
    ("OakInk2\nscene x verb", "runs/oakink2_scene_verb", "perframe", "modular", True),
    ("DexYCB\nsubject->shape", "runs/dexycb_paired_v2", "perframe", "modular", True),
]

#: Below this the dataset poses no question, so an architecture comparison on it
#: is uninformative rather than negative. Set from the two datasets that have no
#: compositional design; they land at 0.00006 and 0.00081.
NO_DIFFICULTY = 0.002


def load(run_dir, baseline: str, treatment: str, budget: int = 256) -> dict | None:
    """Load one sweep, or several pooled, keyed by (sweep, seed).

    `run_dir` may be a list when an axis was run in parts. Seeds repeat across
    parts, so the pooling key carries the sweep name; keying on the seed alone
    would silently drop the continuation.
    """
    dirs = run_dir if isinstance(run_dir, (list, tuple)) else [run_dir]
    rows = []
    for d in dirs:
        p = Path(d) / "results.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8"))["results"]:
            if r.get("budget") == budget:
                rows.append((str(d), r))
    if not rows:
        return None
    rows_only = [r for _, r in rows]

    base = np.array([r["penalty"] for r in rows_only if r["kind"] == baseline])
    by_seed: dict[tuple, dict[str, float]] = {}
    for d, r in rows:
        by_seed.setdefault((d, r.get("seed", 0)), {})[r["kind"]] = r["penalty"]
    pairs = [(v[baseline], v[treatment]) for v in by_seed.values()
             if baseline in v and treatment in v]
    if not len(base) or not pairs:
        return None

    diff = np.array([a - b for a, b in pairs])
    _, p_val = t_test_one_sample(diff)
    return {
        "baseline_mean": base.mean(),
        "baseline_sem": base.std(ddof=1) / np.sqrt(len(base)) if len(base) > 1 else 0.0,
        "n_baseline": len(base),
        "advantage_mean": diff.mean(),
        "advantage_sem": diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0,
        "n_pairs": len(diff),
        "p": p_val,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/figures/difficulty_survey.png")
    ap.add_argument("--budget", type=int, default=256)
    args = ap.parse_args()

    rows, labels, real = [], [], []
    for name, d, base, treat, is_real in SOURCES:
        r = load(d, base, treat, args.budget)
        if r is None:
            print(f"skip {name.replace(chr(10), ' ')}: no usable rows in {d}")
            continue
        rows.append(r)
        labels.append(name)
        real.append(is_real)

    if not rows:
        print("nothing to plot")
        return 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    y = np.arange(len(rows))
    has_difficulty = [r["baseline_mean"] >= NO_DIFFICULTY for r in rows]

    colours = ["#1D4ED8" if is_real else "#94A3B8" for is_real in real]
    ax1.barh(y, [r["baseline_mean"] for r in rows],
             xerr=[r["baseline_sem"] for r in rows],
             color=colours, alpha=[0.95 if h else 0.35 for h in has_difficulty][0] if False else None,
             error_kw={"lw": 1.2, "capsize": 3})
    for i, (bar_r, h) in enumerate(zip(rows, has_difficulty)):
        ax1.patches[i].set_alpha(0.95 if h else 0.35)
        ax1.text(bar_r["baseline_mean"] + bar_r["baseline_sem"] + 0.0004, i,
                 f"{bar_r['baseline_mean']:.5f}  (n={bar_r['n_baseline']})",
                 va="center", fontsize=8.5)
    ax1.axvline(NO_DIFFICULTY, color="#B91C1C", ls=":", lw=1.2)
    ax1.text(NO_DIFFICULTY, len(rows) - 0.35, "  no measurable difficulty",
             color="#B91C1C", fontsize=8, va="top")
    ax1.set_yticks(y, labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("baseline compositional penalty\nMSE(never saw composition) - MSE(saw it)")
    ax1.set_title("How hard are the held-out compositions?", fontsize=11)
    ax1.grid(axis="x", alpha=0.25)

    ax2.barh(y, [r["advantage_mean"] for r in rows],
             xerr=[r["advantage_sem"] for r in rows],
             color="#047857", error_kw={"lw": 1.2, "capsize": 3})
    for i, (bar_r, h) in enumerate(zip(rows, has_difficulty)):
        ax2.patches[i].set_alpha(0.95 if h else 0.25)
        star = "***" if bar_r["p"] < 0.001 else "**" if bar_r["p"] < 0.01 else "*" if bar_r["p"] < 0.05 else ""
        note = f"p={bar_r['p']:.3f}{star}" if h else "uninformative"
        ax2.text(max(bar_r["advantage_mean"], 0) + bar_r["advantage_sem"] + 0.0003, i,
                 f"{note}  (n={bar_r['n_pairs']})", va="center", fontsize=8.5,
                 color="#111827" if h else "#9CA3AF")
    ax2.axvline(0, color="0.5", lw=1)
    ax2.set_yticks(y, [""] * len(rows))
    ax2.invert_yaxis()
    ax2.set_xlabel("modular advantage (paired)\nbaseline penalty - modular penalty")
    ax2.set_title("Does modularity help, where there is something to help with?", fontsize=11)
    ax2.grid(axis="x", alpha=0.25)

    fig.text(0.5, 0.005,
             "Grey bars: synthetic controls. Faded: datasets whose held-out compositions cost "
             "the baseline almost nothing, so an architecture comparison there is uninformative, not negative.",
             ha="center", fontsize=8.5, color="#374151")
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}\n")

    print(f"{'dataset':<34}{'baseline':>12}{'n':>4}{'advantage':>12}{'p':>8}  verdict")
    print("-" * 84)
    for lab, r, h in zip(labels, rows, has_difficulty):
        verdict = ("significant" if r["p"] < 0.05 else "not significant") if h else "no difficulty to test"
        print(f"{lab.replace(chr(10), ' '):<34}{r['baseline_mean']:>12.5f}{r['n_baseline']:>4}"
              f"{r['advantage_mean']:>12.5f}{r['p']:>8.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
