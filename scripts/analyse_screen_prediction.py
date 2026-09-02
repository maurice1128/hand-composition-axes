"""Does the training-free screen predict the measured compositional advantage?

The headline correlation between ``screen_axes.py``'s interaction excess and
the penalty measured by a full paired sweep was computed by hand and never
committed. That made the paper's most novel number unreproducible, and stale:
the affordance axis has since grown from 70 seeds to more, and nothing
recomputed the correlation against the new value. This script is that
computation, so the number in the paper comes from a command.

It also answers the scale objection a reviewer raises immediately. The
baseline's absolute error is ~34% higher than the modular model's, so a larger
absolute penalty could be a property of the error scale rather than of
composition. The relative penalty -- ``penalty / mse_naive``, the fraction of
the naive model's error attributable to the missing composition -- removes
that, and is reported beside the absolute one rather than instead of it.

Seeds are paired across architectures: for a given seed both kinds see the same
split, so ``paired = penalty_perframe - penalty_modular`` is a within-split
difference and the sign test over seeds is the assumption-light check.

    python scripts/analyse_screen_prediction.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

#: Measured axis -> the sweep(s) that measured it. Several runs may cover one
#: axis when seeds were added later; they are merged on seed, and a seed present
#: in more than one run is taken from the first, so re-running cannot silently
#: double-count.
MEASURED: dict[str, tuple[str, ...]] = {
    # The declared budget was 70 seeds; the first sweep stopped at 53 and
    # ``oakink_category_rest`` ran the remaining 17. Both are pooled here, which
    # is what honouring the budget means.
    "oakink_category": ("oakink_official_category", "oakink_category_rest"),
    "oakink_attr": ("oakink_official_attr", "oakink_attr_more"),
    "category": ("oakink_n70_v2",),
    "shape": ("grab_shape_v2",),
    # dexycb_fine is withdrawn, not merely unswept: its run used
    # --min-chains 1, the value that disables the restriction, because all 200
    # of DexYCB's transitions fall in a single fine group. On that
    # configuration check_informed_coverage.py returns TOO THIN (the informed
    # arm sees 2.00 of 4 held compositions), which is the condition that pins
    # the penalty at zero by construction. No reachable configuration passes,
    # so the dataset cannot support a paired composition sweep at all.
    "oakink_class": ("oakink_class_v1",),
    "oakink2_fine": ("oakink2_paired_v2",),
}

#: Screen rows are keyed by (dataset, axis name), NOT by (dataset, mode):
#: DexYCB contributes two rows both carrying ``mode="fine"`` (subject x shape
#: and subject x object) and GRAB a third, so a mode-keyed lookup silently
#: overwrites one with the other. Doing exactly that put DexYCB's
#: subject x object excess (-0.0899) against a sweep measured on subject x
#: shape (-0.0214) and dragged the correlation down by a third.
SCREEN_KEY: dict[str, tuple[str, str]] = {
    "oakink_category": ("OakInk-Image", "official category x intent"),
    "oakink_attr": ("OakInk-Image", "affordance x intent"),
    "category": ("OakInk-Image", "id-prefix x intent (provenance)"),
    "shape": ("GRAB", "shape x fine intent"),
    "oakink_class": ("OakInk-Image", "official class x intent"),
    "oakink2_fine": ("OakInk2", "annotated transitions"),
}


def paired_by_seed(run_dirs: tuple[str, ...]) -> dict[int, dict[str, float]]:
    """Seed -> the two architectures' penalties, absolute and relative."""
    out: dict[int, dict[str, float]] = {}
    for name in run_dirs:
        f = ROOT / "runs" / name / "results.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8"))["results"]:
            kind, seed = r["kind"], int(r["seed"])
            if kind not in ("perframe", "modular"):
                continue
            slot = out.setdefault(seed, {})
            if kind in slot:            # already taken from an earlier run
                continue
            naive = r["naive"]["mse_target"]
            slot[kind] = r["penalty"]
            slot[kind + "_rel"] = r["penalty"] / naive if naive else float("nan")
    return {s: v for s, v in out.items() if "perframe" in v and "modular" in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", default="runs/axis_screen.json")
    ap.add_argument("--out", default="runs/screen_prediction.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.screen).read_text(encoding="utf-8"))["rows"]
    screen = {(r["dataset"], r["axis"]): r for r in rows}
    if len(screen) != len(rows):
        raise RuntimeError("screen rows collide on (dataset, axis); cannot map safely")

    print(f"{'axis':<17} {'n':>4} {'excess':>8} {'paired':>10} {'t p':>8} "
          f"{'sign p':>8} {'rel paired':>11} {'rel t p':>8}")
    print("-" * 82)

    report, xs, ys, ys_rel, ys_pf, ys_md = {}, [], [], [], [], []
    for axis, dirs in MEASURED.items():
        by_seed = paired_by_seed(dirs)
        if not by_seed:
            print(f"{axis:<17}  no seeds found")
            continue
        d = np.array([v["perframe"] - v["modular"] for v in by_seed.values()])
        d_rel = np.array([v["perframe_rel"] - v["modular_rel"] for v in by_seed.values()])
        exc = screen[SCREEN_KEY[axis]]["excess"]

        t_p = float(stats.ttest_1samp(d, 0.0)[1])
        rel_p = float(stats.ttest_1samp(d_rel, 0.0)[1])
        wins = int((d > 0).sum())
        sign_p = float(stats.binomtest(wins, len(d), 0.5).pvalue)

        print(f"{axis:<17} {len(d):>4} {exc:>+8.4f} {d.mean():>+10.5f} {t_p:>8.4f} "
              f"{sign_p:>8.4f} {d_rel.mean():>+11.4f} {rel_p:>8.4f}")
        report[axis] = {"n": len(d), "excess": exc, "paired": float(d.mean()),
                        "t_p": t_p, "sign_wins": wins, "sign_p": sign_p,
                        "paired_relative": float(d_rel.mean()), "relative_t_p": rel_p}
        xs.append(exc); ys.append(float(d.mean())); ys_rel.append(float(d_rel.mean()))
        ys_pf.append(float(np.mean([v["perframe"] for v in by_seed.values()])))
        ys_md.append(float(np.mean([v["modular"] for v in by_seed.values()])))

    x = np.array(xs)
    print("-" * 82)
    print(f"screen excess against each candidate target, n = {len(x)} axes")
    print()

    # The screen is a claim about COMPOSITIONAL DIFFICULTY -- how much harder
    # an unseen composition is -- which is what the naive-minus-informed
    # penalty measures. It is not a claim that the modular prior wins where
    # difficulty is high; ``screen_axes.py`` says as much in its own docstring,
    # and the rows below keep that distinction visible rather than letting one
    # number stand for both. The advantage rows are reported even though they
    # are the weaker ones, because dropping them would be choosing the target
    # that flatters the screen.
    corr = {}
    for tname, y in (("baseline penalty (difficulty)", np.array(ys_pf)),
                     ("modular penalty", np.array(ys_md)),
                     ("modular advantage (paired)", np.array(ys)),
                     ("modular advantage, relative", np.array(ys_rel))):
        r, rp = stats.pearsonr(x, y)
        rho, rhop = stats.spearmanr(x, y)
        loo = [float(stats.pearsonr(np.delete(x, i), np.delete(y, i))[0])
               for i in range(len(x))]
        print(f"  {tname:<30} r = {r:+.3f} (p = {rp:.4f})   "
              f"rho = {rho:+.3f} (p = {rhop:.4f})")
        print(f"  {'':<30} leave-one-out r: {min(loo):+.3f} to {max(loo):+.3f}")
        corr[tname] = {"pearson_r": float(r), "pearson_p": float(rp),
                       "spearman_rho": float(rho), "spearman_p": float(rhop),
                       "loo_r_min": min(loo), "loo_r_max": max(loo)}

    print()
    print("The headline correlation is the first row. With six points Pearson")
    print("and Spearman disagree on significance, so quote both p values.")
    report["correlation"] = corr
    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
