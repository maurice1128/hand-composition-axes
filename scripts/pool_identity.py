"""Pool every identity-ratio run on record and test the threshold.

The claim is not "the ratio is 2.8" but "the ratio reliably exceeds 1", because
above 1 is what "primitive identity dominates context" means -- and that clause
is what the word *library* rests on. So the test is one-sample against 1.0, and
every run goes in, including any that fall below it.

Selecting the runs that passed would be conditioning on the outcome. This
project already withdrew an identity ratio for a version of that error: a single
run reported 0.055, and three runs of the identical configuration turned out to
give 1.060, 0.086 and 0.885.

    python scripts/pool_identity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

def without_term() -> list[float]:
    """Every run trained with the contrastive weight at zero.

    This was a hard-coded list of four values copied from the project log, which
    is the same selection-on-outcome the pooling in this script exists to
    prevent: nine such runs are on disk, and the four transcribed happened to
    average 0.659 against the full set's 0.405. The bias ran *against* the
    paper's claim rather than for it, which is exactly why it survived review --
    a number that makes your own result look worse does not invite a second
    look. Reading them off disk removes the choice.
    """
    vals: list[float] = []
    for sweep in sorted(ROOT.glob("runs/*/sweep.json")):
        try:
            rows = json.loads(sweep.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            vals += [r["identity_ratio"] for r in rows
                     if isinstance(r, dict) and r.get("weight") == 0.0
                     and "identity_ratio" in r]
    for comp in sorted(ROOT.glob("runs/*/composition.json")):
        d = json.loads(comp.read_text(encoding="utf-8"))
        # The bricks runs all carry the term; only the plain modular prior does
        # not, and it records no weight field of its own.
        if "consist" not in comp.parent.name and "bricks" not in comp.parent.name:
            # Same nested-or-flat shape ``collect`` handles; reading only the
            # flat key silently dropped one run and moved the control mean.
            ident = d.get("identity")
            ratio = (ident or {}).get("identity_ratio") if isinstance(ident, dict) \
                else d.get("identity_ratio")
            if ratio is not None:
                vals.append(float(ratio))
    return sorted(vals)


def collect() -> dict[str, float]:
    out: dict[str, float] = {}
    for d in sorted(ROOT.glob("runs/bricks_s*")) + [ROOT / "runs/prior_oakink2_consist"]:
        p = d / "composition.json"
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ident = blob.get("identity")
        ratio = (ident or {}).get("identity_ratio") if isinstance(ident, dict) else blob.get("identity_ratio")
        if ratio is not None:
            out[d.name] = float(ratio)
    return out


def main() -> int:
    runs = collect()
    if not runs:
        print("no composition.json with an identity ratio found")
        return 1

    v = np.array(list(runs.values()))
    print(f"{'run':<28}{'identity ratio':>16}")
    print("-" * 44)
    for name, r in sorted(runs.items(), key=lambda kv: kv[1]):
        print(f"{name:<28}{r:>16.3f}{'   FAIL' if r < 1 else ''}")

    print(f"\nn = {len(v)}   mean {v.mean():.3f}   sd {v.std(ddof=1):.3f}   "
          f"median {np.median(v):.3f}   above 1: {(v > 1).sum()}/{len(v)}")

    t, p = stats.ttest_1samp(v, 1.0)
    print(f"\n  vs threshold 1.0        t = {t:.2f}   p = {p:.4f}   "
          f"{'PASS -- identity reliably exceeds 1' if p < 0.05 and v.mean() > 1 else 'not established'}")

    b = np.array(without_term())
    t2, p2 = stats.ttest_ind(v, b, equal_var=False)
    print(f"  vs runs without the term  t = {t2:.2f}   p = {p2:.4f}   "
          f"(without: mean {b.mean():.3f}, n = {len(b)})")

    sign = stats.binomtest(int((v > 1).sum()), len(v), 0.5).pvalue
    print(f"  sign test on 'above 1'    p = {sign:.4f}")

    out = ROOT / "runs" / "identity_pooled.json"
    out.write_text(json.dumps({
        "runs": runs, "n": len(v), "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)), "above_one": int((v > 1).sum()),
        "p_vs_threshold": float(p), "p_vs_without_term": float(p2),
        "p_sign": float(sign), "without_term": without_term(),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
