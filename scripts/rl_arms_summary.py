"""Summarise a multi-arm RL sweep: per-arm means and every pairwise paired test.

    python scripts/rl_arms_summary.py runs/rl/stageE_retain --arms keyframe linear continuous modular

Expects ``<out>/<arm>_s<seed>/eval.json`` (retention) or
``<out>/<arm>_<variant>_s<seed>/eval.json`` with ``--variant`` (staged).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(out: Path, arm: str, seeds, variant: str | None):
    vals, jit, tp = [], [], []
    for s in seeds:
        d = out / (f"{arm}_{variant}_s{s}" if variant else f"{arm}_s{s}")
        p = d / "eval.json"
        if not p.exists():
            return None
        e = json.loads(p.read_text())
        vals.append(e["success"]); jit.append(e["jitter"])
        tp.append(e.get("train_pool", {}).get("success", float("nan")))
    return np.array(vals), np.array(jit), np.array(tp)


def paired(a, b):
    d = a - b; n = len(d); wins = int((d > 0).sum()); ties = int((d == 0).sum())
    t_p = float(stats.ttest_1samp(d, 0).pvalue) if n > 1 and d.std() > 0 else None
    sign_p = float(stats.binomtest(wins, n - ties, 0.5).pvalue) if n - ties > 0 else None
    return {"mean": float(d.mean()), "t_p": t_p, "wins": wins, "n": n, "sign_p": sign_p}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    data = {}
    for arm in args.arms:
        r = load(out, arm, args.seeds, args.variant)
        if r is None:
            print(f"{arm}: incomplete"); continue
        data[arm] = r
        v, j, tp = r
        extra = f"  train-pool {np.nanmean(tp):.3f}" if not np.isnan(tp).all() else ""
        print(f"{arm:11s} success {v.mean():.3f}  {np.round(v, 2).tolist()}  jitter {j.mean():7.0f}{extra}")
    summary = {"arms": {a: {"success": data[a][0].tolist(), "jitter": data[a][1].tolist()} for a in data}, "pairs": {}}
    for a, b in itertools.combinations(data, 2):
        pr = paired(data[a][0], data[b][0]); summary["pairs"][f"{a}-{b}"] = pr
        print(f"  {a} - {b}: {pr['mean']:+.3f}  t_p {pr['t_p'] if pr['t_p'] is None else round(pr['t_p'], 3)}  wins {pr['wins']}/{pr['n']}  sign_p {pr['sign_p']}")
    name = f"summary_{args.variant}.json" if args.variant else "summary.json"
    (out / name).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
