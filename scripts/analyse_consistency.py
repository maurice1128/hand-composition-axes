"""Summarise a consistency-weight sweep across repeats, with a proper test.

The identity ratio is noisy: three runs of the *identical* unpenalised config
gave 1.060, 0.086 and 0.885 -- a 12x spread. A single run at any weight
therefore establishes nothing, which is how an earlier version of this project
came to report "the contrastive loss raises identity from 0.054 to 1.036" when
the baseline reaches 1.060 unaided.

This groups repeats by weight and reports mean +- sd plus a two-sample test
against the unpenalised runs. It also reports reconstruction, because an
identity ratio bought by wrecking the prior is not a result.

    python scripts/analyse_consistency.py runs/consistency_seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's t against unequal variances; returns (t, df, two-sided p)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = va + vb
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    t = (b.mean() - a.mean()) / np.sqrt(denom)
    df = denom**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df=df))
    except ImportError:  # pragma: no cover
        from math import erfc

        p = float(erfc(abs(t) / np.sqrt(2)))
    return float(t), float(df), p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="+",
                    help="one or more sweep directories; repeats are pooled. Extra "
                         "runs must go to a NEW directory -- re-running a sweep into "
                         "an existing --out overwrites its sweep.json and destroys "
                         "the repeats it was meant to add to.")
    args = ap.parse_args()

    rows = []
    for d in args.run_dir:
        p = Path(d) / "sweep.json"
        if not p.exists():
            print(f"skip {d}: no sweep.json")
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        # Sweeps written before sweep.json recorded its arguments are a bare
        # list. Those are exactly the runs whose bundle is now unknown, so they
        # are read but flagged rather than silently pooled with identified ones.
        if isinstance(blob, list):
            print(f"note: {d} predates argument recording -- bundle unknown")
            rows.extend(blob)
        else:
            print(f"{d}: bundle {blob.get('args', {}).get('bundle', '?')}")
            rows.extend(blob["rows"])
    if not rows:
        print("no results found")
        return 1
    by_w: dict[float, list[dict]] = {}
    for r in rows:
        by_w.setdefault(float(r["weight"]), []).append(r)

    weights = sorted(by_w)
    base = np.array([r["identity_ratio"] for r in by_w.get(0.0, [])])
    base_recon = np.array([r["recon"] for r in by_w.get(0.0, []) if r["recon"]])

    print(f"{' + '.join(args.run_dir)}  ({len(rows)} runs pooled)\n")
    print(f"{'weight':>8} {'n':>3} {'identity':>18} {'recon':>18} {'between':>10} {'p vs w=0':>9}")
    print("-" * 74)

    for w in weights:
        ident = np.array([r["identity_ratio"] for r in by_w[w]])
        recon = np.array([r["recon"] for r in by_w[w] if r["recon"]])
        betw = np.array([r["between"] for r in by_w[w]])
        p = welch(base, ident)[2] if w != 0 and len(base) >= 2 else float("nan")
        print(
            f"{w:>8g} {len(ident):>3} "
            f"{ident.mean():>8.3f}+-{ident.std(ddof=1) if len(ident) > 1 else 0:<8.3f} "
            f"{recon.mean():>8.5f}+-{recon.std(ddof=1) if len(recon) > 1 else 0:<8.5f} "
            f"{betw.mean():>10.4f} "
            f"{p:>9.3f}" if np.isfinite(p) else
            f"{w:>8g} {len(ident):>3} "
            f"{ident.mean():>8.3f}+-{ident.std(ddof=1) if len(ident) > 1 else 0:<8.3f} "
            f"{recon.mean():>8.5f}+-{recon.std(ddof=1) if len(recon) > 1 else 0:<8.5f} "
            f"{betw.mean():>10.4f} {'—':>9}"
        )

    print()
    monotone = all(
        np.mean([r["identity_ratio"] for r in by_w[weights[i]]])
        <= np.mean([r["identity_ratio"] for r in by_w[weights[i + 1]]])
        for i in range(len(weights) - 1)
    )
    best = max((w for w in weights if w > 0), key=lambda w: np.mean([r["identity_ratio"] for r in by_w[w]]), default=None)
    sig = [w for w in weights if w > 0 and np.isfinite(welch(base, np.array([r["identity_ratio"] for r in by_w[w]]))[2]) and welch(base, np.array([r["identity_ratio"] for r in by_w[w]]))[2] < 0.05]

    print(f"identity ratio monotone in weight: {monotone}")
    if base_recon.size:
        for w in weights:
            if w == 0:
                continue
            rec = np.array([r["recon"] for r in by_w[w] if r["recon"]])
            if rec.size:
                print(f"  w={w:g}: recon {100 * (rec.mean() / base_recon.mean() - 1):+.0f}% vs unpenalised")
    print()
    if sig:
        print(f"SIGNIFICANT at weights {sig} (p<0.05). The contrastive term raises identity.")
    else:
        print(
            "NOT SIGNIFICANT at any weight. The direction may be consistent, but with\n"
            "this many repeats it cannot be distinguished from the baseline's own\n"
            "spread -- and that spread is large (three identical unpenalised runs\n"
            "differed by 12x). More repeats, not a better-looking single run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
