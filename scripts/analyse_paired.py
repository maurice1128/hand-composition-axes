"""Paired per-seed analysis of a compositional-penalty sweep.

Why not just compare the two means
-----------------------------------
The summary printed by ``experiment_paired_composition.py`` compares each
model's penalty against zero using the spread across seeds. That spread is
dominated by *which compositions the seed held out*, which varies enormously on
real data (OakInk naive MSE: 0.073 +- 0.017, a 23% relative spread).

But both models see the **same split** within a seed. So the seed-to-seed
variation is common to them, and the difference of their penalties is far less
noisy than either penalty alone. Comparing the two means throws that away and
answers "is each penalty non-zero?" when the question is "does modularity
reduce the penalty?".

This script does the paired comparison: per seed, ``penalty(baseline) -
penalty(modular)``, then a one-sample t-test on those differences.

    python scripts/analyse_paired.py runs/pc_hard runs/oakink_coarse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def t_test_one_sample(x: np.ndarray) -> tuple[float, float]:
    """Return (t, two-sided p). Uses scipy if present, else a normal approx."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("inf") if x.mean() != 0 else 0.0, 0.0
    t = x.mean() / (sd / np.sqrt(n))
    try:
        from scipy import stats

        return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:  # pragma: no cover
        from math import erfc

        return float(t), float(erfc(abs(t) / np.sqrt(2)))


def analyse(run_dir: Path, baseline: str, treatment: str) -> None:
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows = data["results"]
    budgets = sorted({r["budget"] for r in rows})

    print(f"\n{'=' * 78}\n{run_dir.name}   paired: {baseline} - {treatment}\n{'=' * 78}")
    print(f"{'budget':>7} {'n':>3} {'base pen':>11} {'treat pen':>11} {'difference':>18} {'p':>8}")
    print("-" * 78)

    for budget in budgets:
        by_seed: dict[int, dict[str, float]] = {}
        for r in rows:
            if r["budget"] != budget:
                continue
            by_seed.setdefault(r.get("seed", 0), {})[r["kind"]] = r["penalty"]

        pairs = [
            (v[baseline], v[treatment])
            for v in by_seed.values()
            if baseline in v and treatment in v
        ]
        if len(pairs) < 2:
            print(f"{budget:>7} {len(pairs):>3}  (need >=2 paired seeds)")
            continue

        b = np.array([p[0] for p in pairs])
        m = np.array([p[1] for p in pairs])
        d = b - m
        t, p = t_test_one_sample(d)
        # A positive difference means the treatment pays a SMALLER compositional
        # penalty, i.e. modularity helped.
        verdict = (
            "modular helps" if p < 0.05 and d.mean() > 0
            else "modular hurts" if p < 0.05 and d.mean() < 0
            else "not significant"
        )
        print(
            f"{budget:>7} {len(pairs):>3} {b.mean():>11.5f} {m.mean():>11.5f} "
            f"{d.mean():>9.5f}+-{d.std(ddof=1):<7.5f} {p:>8.3f}  {verdict}"
        )
        print(f"{'':>7} per-seed differences: " + ", ".join(f"{x:+.4f}" for x in d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--baseline", default="perframe")
    ap.add_argument("--treatment", default="modular")
    args = ap.parse_args()

    for rd in args.run_dirs:
        path = Path(rd)
        if not (path / "results.json").exists():
            print(f"skip {rd}: no results.json")
            continue
        analyse(path, args.baseline, args.treatment)

    print(
        "\nA positive difference means the modular prior pays a SMALLER compositional\n"
        "penalty than the bandwidth-matched baseline on the same split. The paired\n"
        "test is the right one here because both models share the split within a seed;\n"
        "comparing their independent means answers a different question and has much\n"
        "less power."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
