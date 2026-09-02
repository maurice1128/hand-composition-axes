"""Equivalence tests for the seen-versus-unseen claims.

Every seen-versus-unseen result in this project is a failure to reject, and a
failure to reject is not evidence of similarity: at 19 clips per arm almost
nothing would be rejected. The paper says so, which is honest but leaves the
claim weaker than the data may support. This runs the test that can actually
support it.

Two one-sided tests (TOST). The null is "the arms differ by at least delta";
rejecting it in both directions concludes the difference is smaller than delta.
Unlike a *t*-test, more data and a tighter spread make this *easier* to pass,
which is the property a similarity claim needs.

**The margins are declared here and not tuned.** Choosing delta after seeing the
result is the same error as choosing a sample size after seeing a *p*-value, and
this project has already made that one once:

* Grasp retention, `DELTA_RATE` = 10 percentage points. A prior whose novel
  compositions retain objects within 10 points of its familiar ones is doing
  the thing the word *library* promises; one that loses 20 points is not. The
  figure is chosen to be substantively meaningful, not to be attainable -- at
  19 clips per arm it is a demanding margin and may well fail.
* Abduction correction cost, `DELTA_DEG` = 1.0 degree, against a correction that
  costs 0.9 to 5.5 degrees overall. A difference under a degree does not change
  what the correction costs in practice.
* Penetration after correction, `DELTA_FRAC` = 5 percentage points, against
  post-correction rates of 0.3% to 2.6%.

A failure here is reported as a failure. The honest outcome of an underpowered
equivalence test is "we cannot conclude equivalence either", which leaves the
paper exactly where it was rather than anywhere worse.

    python scripts/analyse_equivalence.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

DELTA_RATE = 0.10       # grasp retention, proportion
DELTA_DEG = 1.0         # abduction cost, degrees
DELTA_FRAC = 0.05       # penetration after correction, proportion


def tost_two_proportions(k1: int, n1: int, k2: int, n2: int,
                         delta: float) -> tuple[float, float]:
    """TOST for two independent proportions. Returns (difference, p)."""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        # Both arms degenerate: the difference is exactly zero and the test is
        # vacuous rather than passing. Report it as undefined.
        return p1 - p2, float("nan")
    d = p1 - p2
    p_lower = stats.norm.cdf((d + delta) / se)          # H0: d <= -delta
    p_upper = stats.norm.sf((d - delta) / se)           # H0: d >= +delta
    return d, max(1 - p_lower, 1 - p_upper) if False else max(
        stats.norm.sf((d + delta) / se), stats.norm.cdf((d - delta) / se))


def tost_paired(a: np.ndarray, b: np.ndarray, delta: float) -> tuple[float, float]:
    """TOST on unpaired samples of a continuous quantity (Welch)."""
    d = float(np.mean(a) - np.mean(b))
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    if se == 0 or not np.isfinite(se):
        return d, float("nan")
    df = len(a) + len(b) - 2
    p_upper = stats.t.cdf((d - delta) / se, df)         # H0: d >= +delta
    p_lower = stats.t.sf((d + delta) / se, df)          # H0: d <= -delta
    return d, max(p_upper, p_lower)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/equivalence.json")
    args = ap.parse_args()

    report: dict = {"margins": {"retention_pp": DELTA_RATE,
                                "abduction_deg": DELTA_DEG,
                                "penetration_pp": DELTA_FRAC}}

    print(f"TOST, margins declared in the docstring: retention "
          f"{DELTA_RATE:.0%}, abduction {DELTA_DEG}°, penetration "
          f"{DELTA_FRAC:.0%}\n")

    # -- grasp retention, pooled per difficulty over the four banks ----------
    print("grasp retention (seen vs unseen)")
    print(f"  {'level':<9}{'seen':>10}{'unseen':>10}{'diff':>9}"
          f"{'TOST p':>10}  verdict")
    by_level: dict[str, list[list[int]]] = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "grasp_s*_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        acc = by_level.setdefault(d["difficulty"], [[0, 0], [0, 0]])
        for i, arm in enumerate(("seen", "unseen")):
            held = d[arm]["held"]
            acc[i][0] += int(np.sum(held))
            acc[i][1] += len(held)

    report["retention"] = {}
    for level in ("easy", "medium", "hard"):
        if level not in by_level:
            continue
        (ks, ns), (ku, nu) = by_level[level]
        diff, p = tost_two_proportions(ks, ns, ku, nu, DELTA_RATE)
        ok = "EQUIVALENT" if p < 0.05 else ("undefined" if np.isnan(p)
                                            else "not established")
        print(f"  {level:<9}{ks / ns:>9.1%}{ku / nu:>10.1%}{diff:>+9.1%}"
              f"{p:>10.4f}  {ok}")
        report["retention"][level] = {"seen": ks / ns, "unseen": ku / nu,
                                      "n_seen": ns, "n_unseen": nu,
                                      "difference": diff, "tost_p": p}

    # -- retargeting: abduction cost and post-correction penetration --------
    print("\nretargeting (seen vs unseen), per prior")
    print(f"  {'run':<26}{'measure':<12}{'diff':>9}{'TOST p':>10}  verdict")
    report["retargeting"] = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "robot_transfer*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        name = Path(f).stem
        entry: dict = {}
        for key, delta, label in (("per_clip_change", DELTA_DEG, "abduction"),
                                  ("per_clip_after", DELTA_FRAC, "penetration")):
            a = np.asarray(d.get("seen", {}).get(key, []), dtype=float)
            b = np.asarray(d.get("unseen", {}).get(key, []), dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            diff, p = tost_paired(a, b, delta)
            ok = "EQUIVALENT" if p < 0.05 else ("undefined" if np.isnan(p)
                                                else "not established")
            unit = "°" if label == "abduction" else ""
            print(f"  {name:<26}{label:<12}{diff:>+8.3f}{unit}{p:>10.4f}  {ok}")
            entry[label] = {"difference": diff, "tost_p": p,
                            "n_seen": len(a), "n_unseen": len(b)}
        if entry:
            report["retargeting"][name] = entry

    # -- the same tests, pooled over all five priors ------------------------
    # Per-prior the tests run at 19-60 clips per arm, which is why most return
    # "not established" rather than a verdict. Pooling gives 165 per arm. The
    # cost is heterogeneity: the five banks differ in difficulty as well as in
    # seed, with pre-correction penetration spanning 0% to 66%, so this is a
    # fixed-effect pooling of unlike runs and is reported alongside the
    # per-prior results rather than instead of them. The margins are the ones
    # declared above; nothing about them changes because n grew.
    print("\npooled over all five priors")
    print(f"  {'measure':<14}{'n/arm':>7}{'diff':>10}{'TOST p':>10}  verdict")
    report["pooled"] = {}
    for key, delta, label in (("per_clip_change", DELTA_DEG, "abduction"),
                              ("per_clip_after", DELTA_FRAC, "penetration")):
        a, b = [], []
        for f in sorted(glob.glob(str(ROOT / "runs" / "robot_transfer*.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            a += list(d.get("seen", {}).get(key, []))
            b += list(d.get("unseen", {}).get(key, []))
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 2:
            continue
        diff, p = tost_paired(a, b, delta)
        ok = "EQUIVALENT" if p < 0.05 else ("undefined" if np.isnan(p)
                                            else "not established")
        unit = "°" if label == "abduction" else ""
        print(f"  {label:<14}{len(a):>7}{diff:>+9.3f}{unit}{p:>10.4f}  {ok}")
        report["pooled"][label] = {"n_per_arm": len(a), "difference": diff,
                                   "tost_p": p}

    (ROOT / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / args.out}")
    print("A margin is only meaningful if it was fixed before the test ran; "
          "\nthese were, and a failure is reported as a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
