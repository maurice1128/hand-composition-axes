"""Is the 200x difficulty range about the composition axis, or about the data?

The survey found that compositional penalty spans 200x across five datasets and
does not track whether the dataset annotates compositions. The explanation
offered was about *axis type*: ``category->intent`` carries kinematic structure,
``annotated primitive transition`` does not.

A competing explanation appeared while adding GRAB. Line up the convention-
inference diagnostics against the measured difficulty and they track each other:

    OakInk-Image   dedicated hand fit       residual  5.9 deg   difficulty +0.0120
    DexYCB         dedicated hand fit       residual  9.5 deg   difficulty +0.00006
    OakInk2        from whole-body SMPL-X   residual 22.0 deg   difficulty +0.00081
    GRAB           from whole-body SMPL-X   residual 12.7 deg   difficulty ?

If a fitting pipeline regularises the hand toward a mean pose, it removes
exactly the between-composition variation the penalty measures -- and the survey
would be reporting annotation quality dressed up as compositional structure.

This script measures the thing the two explanations disagree about, without
training anything: **how much of the pose variance is explained by the
composition label**.

    eta^2 = 1 - SS_within / SS_total

computed on normalised windows pooled over all 27 DOF. High means poses separate
by composition; near zero means the label is not visible in the kinematics at
all, whatever the label is called.

Three details decide whether the number means anything:

* **The null is permutation-based, and permutes whole trajectories.** eta^2 is
  biased upward when there are many small groups, and these datasets differ in
  both. Shuffling window labels independently would give an absurdly low null,
  since 30 fps windows from one trajectory are near-duplicates; the label is a
  property of the trajectory, so the permutation has to be too.
* **Excess over that null is what gets compared**, never raw eta^2.
* **Everything runs off per-trajectory sufficient statistics** -- count, sum and
  sum-of-squares per feature dimension. A permutation only reassigns
  trajectories to labels, so group sums are sums of trajectory sums and the
  window matrix never has to exist twice. The first version held all windows in
  memory and died with a 39.6 MiB allocation failure on a loaded machine.

This is a correlational diagnostic. It can show that difficulty tracks pose
separability rather than annotation status, but it cannot show which way the
causation runs.

    python scripts/analyse_composition_separability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import normalize  # noqa: E402

from experiment_paired_composition import coarsen_labels  # noqa: E402

#: (display name, bundle, granularity).
#:
#: Only the three datasets whose label **is** a single composition appear here.
#: The synthetic controls and OakInk2 label each trajectory with a whole chain
#: (``grip->rearrange->take_outside``), so 2968 of 3000 synthetic labels are
#: unique and eta^2 on them measures trajectory identity, not composition --
#: it came out at +0.0015 excess against a 0.19 null, which is a statement about
#: the label format and nothing else. Their composition unit is a *transition
#: inside* the chain, which needs the segment boundaries this diagnostic does
#: not use. They are excluded rather than reported.
SOURCES = [
    ("OakInk-Image", "data/bundles/oakink.npz", "category"),
    ("DexYCB", "data/bundles/dexycb_shape.npz", "fine"),
    ("GRAB", "data/bundles/grab.npz", "shape"),
]


def traj_stats(
    traj: np.ndarray, length: int, stride: int, max_windows: int, rng: np.random.Generator
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Per-trajectory (count, sum, sum-of-squares) over flattened windows."""
    if len(traj) < length:
        return None
    starts = np.arange(0, len(traj) - length + 1, stride)
    if len(starts) > max_windows:
        starts = np.sort(rng.choice(starts, max_windows, replace=False))
    w = normalize(np.stack([traj[s : s + length] for s in starts]))
    w = w.reshape(len(w), -1).astype(np.float64)
    return len(w), w.sum(axis=0), (w * w).sum(axis=0)


class EtaSquared:
    """Fraction of total window variance explained by a grouping.

    Trace-based multivariate generalisation: sums squares over every feature
    dimension rather than testing one at a time, so a composition that shifts
    several DOF slightly counts the same as one shifting a single DOF a lot.

    A class rather than a function because the permutation null calls this
    hundreds of times and this machine runs the analysis alongside two dozen
    other jobs -- a 1.56 MiB allocation failed on the first attempt. Groups are
    integer codes, not label strings, and the two ``K x d`` accumulators are
    allocated once and refilled.
    """

    def __init__(self, counts: np.ndarray, sums: np.ndarray, sumsqs: np.ndarray, n_groups: int):
        self.counts, self.sums, self.sumsqs = counts, sums, sumsqs
        self.gsum = np.zeros((n_groups, sums.shape[1]))
        self.gsq = np.zeros((n_groups, sums.shape[1]))
        self.n_groups = n_groups
        n = counts.sum()
        self.ss_total = float((sumsqs.sum(0) - sums.sum(0) ** 2 / n).sum())

    def __call__(self, codes: np.ndarray) -> float:
        if self.ss_total <= 0:
            return 0.0
        self.gsum.fill(0.0)
        self.gsq.fill(0.0)
        np.add.at(self.gsum, codes, self.sums)
        np.add.at(self.gsq, codes, self.sumsqs)
        ng = np.bincount(codes, weights=self.counts, minlength=self.n_groups)
        # Empty groups have zero sums, so they contribute nothing either way;
        # the 1.0 only keeps the division finite. einsum avoids materialising
        # a K x d square of gsum.
        safe = np.where(ng > 0, ng, 1.0)
        between = float((np.einsum("kd,kd->k", self.gsum, self.gsum) / safe).sum())
        ss_within = float(self.gsq.sum()) - between
        return 1.0 - ss_within / self.ss_total


def analyse(
    path: Path,
    granularity: str,
    length: int,
    stride: int,
    n_perm: int,
    max_windows_per_traj: int,
    seed: int,
) -> dict | None:
    if not path.exists():
        return None
    b = TrajectoryBundle.load(path)
    labels = coarsen_labels(list(b.labels), granularity)
    rng = np.random.default_rng(seed)

    counts, sums, sumsqs, per_traj = [], [], [], []
    for traj, lab in zip(b.trajectories, labels):
        st = traj_stats(np.asarray(traj), length, stride, max_windows_per_traj, rng)
        if st is None:
            continue
        counts.append(st[0])
        sums.append(st[1])
        sumsqs.append(st[2])
        per_traj.append(lab)
    if not counts:
        return None

    counts = np.asarray(counts, dtype=np.float64)
    sums = np.stack(sums)
    sumsqs = np.stack(sumsqs)

    vocab = sorted(set(per_traj))
    code_of = {lab: i for i, lab in enumerate(vocab)}
    codes = np.array([code_of[l] for l in per_traj], dtype=np.intp)

    eta_of = EtaSquared(counts, sums, sumsqs, len(vocab))
    eta = eta_of(codes)
    null = np.array([eta_of(rng.permutation(codes)) for _ in range(n_perm)])

    sd = float(null.std(ddof=1))
    excess = eta - float(null.mean())
    resid = b.meta.get("projection_residual_deg")
    return {
        "dataset": path.stem,
        "granularity": granularity,
        "n_trajectories": int(len(counts)),
        "n_windows": int(counts.sum()),
        "n_cells": len(vocab),
        "eta2": eta,
        "eta2_null_mean": float(null.mean()),
        "eta2_null_sd": sd,
        "eta2_excess": excess,
        "z": excess / sd if sd > 0 else float("nan"),
        "residual_deg": float(resid["mean"]) if isinstance(resid, dict) else float("nan"),
        "flexion_dominance": float(b.meta.get("flexion_axis_dominance", float("nan"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--max-windows-per-traj", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/separability.json")
    args = ap.parse_args()

    rows = []
    for name, rel, gran in SOURCES:
        r = analyse(ROOT / rel, gran, args.length, args.stride,
                    args.n_perm, args.max_windows_per_traj, args.seed)
        if r is None:
            print(f"skip {name}: {rel} not built", flush=True)
            continue
        r["name"] = name
        rows.append(r)
        print(f"  {name:<24} eta2={r['eta2']:.4f}  null={r['eta2_null_mean']:.4f}"
              f"  excess={r['eta2_excess']:+.4f}  z={r['z']:6.1f}", flush=True)

    if not rows:
        return 1

    print(f"\n{'dataset':<24}{'traj':>6}{'cells':>7}{'eta2':>9}{'null':>8}"
          f"{'excess':>9}{'z':>8}{'resid':>8}{'domin':>8}")
    print("-" * 89)
    for r in rows:
        print(f"{r['name']:<24}{r['n_trajectories']:>6}{r['n_cells']:>7}{r['eta2']:>9.4f}"
              f"{r['eta2_null_mean']:>8.4f}{r['eta2_excess']:>+9.4f}{r['z']:>8.1f}"
              f"{r['residual_deg']:>8.1f}{r['flexion_dominance']:>8.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
