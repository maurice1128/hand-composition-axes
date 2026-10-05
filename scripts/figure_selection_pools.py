"""Figure for the Q10 follow-up: per-stage selection vs separate training with matched pools.

For every one of the 184 756 pools of 10 of the 20 flat policies: the test product
(grasp x carry, per-stage test rates) of per-stage selection (B10) and of whole-policy
selection (A10), both chosen on validation; the separately trained skills (C, fixed pools
of 10) as a vertical line. Data: runs/lineC/q10 (validation) and runs/lineC/q10_followup (test).

    python scripts/figure_selection_pools.py --out docs/submission/figures/selection_pools.pdf
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Q10, FU = ROOT / "runs/lineC/q10", ROOT / "runs/lineC/q10_followup"
FLAT, SKILL = range(10, 30), range(10, 20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/submission/figures/selection_pools.pdf")
    a = ap.parse_args()
    rd = lambda p: json.loads(p.read_text())["success"]
    vg = np.array([rd(Q10 / f"val_grasp_flat_s{s}.json") for s in FLAT])
    vc = np.array([rd(Q10 / f"val_carry_flat_s{s}.json") for s in FLAT])
    tg = np.array([rd(FU / f"test_grasp_flat_s{s}.json") for s in FLAT])
    tc = np.array([rd(FU / f"test_carry_flat_s{s}.json") for s in FLAT])
    vgs = np.array([rd(Q10 / f"val_grasp_grasp_s{s}.json") for s in SKILL])
    vcs = np.array([rd(Q10 / f"val_carry_carry_s{s}.json") for s in SKILL])
    tgs = np.array([rd(FU / f"test_grasp_grasp_s{s}.json") for s in SKILL])
    tcs = np.array([rd(FU / f"test_carry_carry_s{s}.json") for s in SKILL])
    c = tgs[np.argmax(vgs)] * tcs[np.argmax(vcs)]
    b10, a10 = [], []
    for sub in itertools.combinations(range(20), 10):
        sub = np.array(sub)
        b10.append(tg[sub[np.argmax(vg[sub])]] * tc[sub[np.argmax(vc[sub])]])
        k = sub[np.argmax(vg[sub] * vc[sub])]
        a10.append(tg[k] * tc[k])
    b10, a10 = np.array(b10), np.array(a10)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    bins = np.linspace(0.0, 0.32, 33)
    ax.hist(a10, bins=bins, weights=np.full(len(a10), 1 / len(a10)), color="#8A9297", alpha=0.75,
            label="A: best flat policy")
    ax.hist(b10, bins=bins, weights=np.full(len(b10), 1 / len(b10)), color="#2446E0", alpha=0.75,
            label="B: best flat per stage")
    ax.axvline(c, color="#B0413E", lw=1.6, ls="--", label="C: separate skills")
    ax.set_xlabel("test success, grasp $\\times$ carry")
    ax.set_ylabel("share of pools")
    ax.set_ylim(0, 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(pad=0.3)
    out = ROOT / a.out
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"C {c:.3f}; B10 mean {b10.mean():.3f}, P(B10>C) {(b10 > c).mean():.3f}; A10 mean {a10.mean():.3f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
